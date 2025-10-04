#!/usr/bin/env python3
"""
EMG Cue-Based Experiment Runner (Myo BLE→UART)
----------------------------------------------
- Reads serial lines: eK,<device_us>,v0,...,v7
- Runs randomized, variable-length wrist-movement trials with 3 s pre-cue
- UI shows: CURRENT cue + countdown, NEXT cue, live 8-ch plots, status
- Controls: Start, Pause/Resume, Mark Noisy, Mark Failed, End & Save
- Outputs:
  1) CSV (continuous samples): device_us,host_ns,stream,ch0..ch7,label,trial_id,phase,marker
  2) JSON (trial table + run metadata)

Author: Mohamad Marwan Sidani
"""

import sys, os, time, json, random, queue, threading, datetime
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial

# ============ CONFIG ============
PORT = "COM5"        # e.g., "COM5" on Windows, "/dev/ttyACM0" on Linux
BAUD = 115200
N_STREAMS = 4
N_CH = 8
FS_VIS = 200.0               # nominal merged rate for time axis (for plotting only)
WINDOW_SEC = 5.0             # visible plot window
BUF_LEN = int(WINDOW_SEC * FS_VIS) * 2
PRINT_RAW_FIRST_N = 5

# Trial design
LABELS = [
    "rest",
    "mild_extension", "full_extension",
    "mild_flexion",   "full_flexion",
    "mild_radial",    "full_radial",
    "mild_ulnar",     "full_ulnar",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: n for n, i in LABEL_TO_ID.items()}

TRIALS_PER_CLASS = 8            # edit for dataset size
PREPARE_SEC = 3.0               # pre-cue preview (“Get ready…”) fixed
WINDOW_RANGE_SEC = (2.0, 5.0)   # randomized action window length
ITI_RANGE_SEC = (2.0, 4.0)      # randomized inter-trial rest between trials
AVOID_IMMEDIATE_REPEATS = True  # don’t cue same class twice in a row
RANDOM_SEED = None              # set int for reproducibility

# Files
OUT_DIR = "emg_runs"
# ===============================

# ---------- Helpers ----------
def now_ns():
    return time.time_ns()

def ts_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_line(line: str):
    """
    Accept tolerant 'eK,<timestamp>,v0,...,v7' with extra spaces/CRLF.
    Returns (stream_index, device_us, [8]) or None.
    """
    sline = line.strip()
    if not sline or sline[0] != 'e' or len(sline) < 3 or not sline[1].isdigit():
        return None
    try:
        s = int(sline[1])
    except ValueError:
        return None
    parts = [p.strip() for p in sline.split(',')]
    if len(parts) < 10:
        return None
    try:
        t_us = int(parts[1])
        vals = [int(p) for p in parts[2:10]]
    except ValueError:
        return None
    if not (0 <= s < N_STREAMS) or len(vals) != 8:
        return None
    return s, t_us, vals

@dataclass
class Trial:
    trial_id: int
    label_id: int
    label_name: str
    prepare_s: float
    window_s: float
    iti_s: float
    t_cue_on_us: int = -1
    t_go_us: int = -1
    t_window_start_us: int = -1
    t_window_end_us: int = -1
    outcome: str = "ok"       # "ok" | "noisy" | "failed" | "skipped"

# ---------- Serial Reader Thread ----------
class SerialReader(threading.Thread):
    def __init__(self, port, baud, out_queue, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.q = out_queue
        self.stop = stop_event
        self.raw_printed = 0

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
        except Exception as e:
            print(f"[Serial] Open failed: {e}")
            return
        with ser:
            while not self.stop.is_set():
                try:
                    line = ser.readline().decode('utf-8', errors='ignore')
                    if PRINT_RAW_FIRST_N and self.raw_printed < PRINT_RAW_FIRST_N:
                        print("[RAW]", line.rstrip())
                        self.raw_printed += 1
                    parsed = parse_line(line)
                    if parsed:
                        # append host time (ns) now to enable cross-device alignment
                        s, t_us, vals = parsed
                        self.q.put_nowait((s, t_us, vals, now_ns()))
                except queue.Full:
                    try: _ = self.q.get_nowait()
                    except Exception: pass
                except Exception:
                    pass

# ---------- Logger ----------
class CSVLogger:
    def __init__(self, basepath_csv):
        self.fp = open(basepath_csv, "w", buffering=1)
        hdr = "device_us,host_ns,stream," + ",".join([f"ch{i}" for i in range(N_CH)]) + ",label,trial_id,phase,marker\n"
        self.fp.write(hdr)

    def write(self, device_us, host_ns, stream, vals, label_id, trial_id, phase, marker):
        row = f"{device_us},{host_ns},{stream}," + ",".join(str(int(v)) for v in vals) + f",{label_id},{trial_id},{phase},{marker}\n"
        self.fp.write(row)

    def close(self):
        try: self.fp.close()
        except: pass

# ---------- Trial Scheduler ----------
def build_trials():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    # Make a pool with TRIALS_PER_CLASS per label (rest included)
    items = []
    tid = 0
    for lid, name in enumerate(LABELS):
        for _ in range(TRIALS_PER_CLASS):
            window = random.uniform(*WINDOW_RANGE_SEC)
            iti = random.uniform(*ITI_RANGE_SEC)
            items.append(Trial(
                trial_id=tid, label_id=lid, label_name=name,
                prepare_s=PREPARE_SEC, window_s=window, iti_s=iti
            ))
            tid += 1
    # Randomize while optionally avoiding immediate repeats
    random.shuffle(items)
    if AVOID_IMMEDIATE_REPEATS:
        for i in range(1, len(items)):
            if items[i].label_id == items[i-1].label_id:
                # swap with a non-adjacent index if possible
                j = (i+1) % len(items)
                items[i], items[j] = items[j], items[i]
    return items

# ---------- GUI ----------
class EMGExperiment(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Experiment — Wrist Cues")
        self.resize(1200, 800)

        # State
        self.q = queue.Queue(maxsize=8192)
        self.stop_event = threading.Event()
        self.reader = SerialReader(PORT, BAUD, self.q, self.stop_event)

        self.running = False
        self.paused = False
        self.marker_pending = None  # "noisy" | "failed"
        self.current_trial = None
        self.trials = build_trials()
        self.trial_idx = -1

        # Buffers for plotting
        self.tline = deque(maxlen=BUF_LEN)
        self.ybufs = [deque(maxlen=BUF_LEN) for _ in range(N_CH)]
        self.expected_s = 0
        self.sample_index = 0

        # Output paths
        ensure_dir(OUT_DIR)
        stamp = ts_iso()
        base = os.path.join(OUT_DIR, f"run_{stamp}")
        self.csv_path = base + ".csv"
        self.json_path = base + ".json"
        self.logger = CSVLogger(self.csv_path)
        self.run_meta = {
            "started_iso": datetime.datetime.now().isoformat(),
            "port": PORT,
            "baud": BAUD,
            "labels": LABELS,
            "config": {
                "prepare_s": PREPARE_SEC,
                "window_range_s": WINDOW_RANGE_SEC,
                "iti_range_s": ITI_RANGE_SEC,
                "trials_per_class": TRIALS_PER_CLASS
            }
        }

        # --- UI ---
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # Top cue panel
        cue_box = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Idle")
        self.lbl_status.setStyleSheet("font-size:16px;")
        self.lbl_current = QtWidgets.QLabel("CURRENT: —")
        self.lbl_current.setStyleSheet("font-size:28px; font-weight:bold;")
        self.lbl_count = QtWidgets.QLabel("00.0 s")
        self.lbl_count.setStyleSheet("font-size:28px;")
        self.lbl_next = QtWidgets.QLabel("NEXT: —")
        self.lbl_next.setStyleSheet("font-size:20px; color: #666;")
        cue_box.addWidget(self.lbl_status, 1)
        cue_box.addWidget(self.lbl_current, 3)
        cue_box.addWidget(self.lbl_count, 1)
        cue_box.addWidget(self.lbl_next, 3)
        layout.addLayout(cue_box)

        # Plot grid 2×4
        gridw = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(gridw); layout.addWidget(gridw, 1)
        self.plots, self.curves = [], []
        for ch in range(N_CH):
            plt = pg.PlotWidget()
            plt.showGrid(x=True, y=True, alpha=0.25)
            plt.setLabel('left', f'ch{ch}')
            plt.setLabel('bottom', 'time (s)')
            curve = plt.plot([], [])
            r = ch // 4; c = ch % 4
            grid.addWidget(plt, r, c)
            self.plots.append(plt); self.curves.append(curve)

        # Controls
        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_noisy = QtWidgets.QPushButton("Mark Noisy")
        self.btn_failed = QtWidgets.QPushButton("Mark Failed")
        self.btn_end = QtWidgets.QPushButton("End & Save")
        btns.addWidget(self.btn_start); btns.addWidget(self.btn_pause); btns.addWidget(self.btn_resume)
        btns.addWidget(self.btn_noisy); btns.addWidget(self.btn_failed); btns.addWidget(self.btn_end)
        layout.addLayout(btns)

        self.btn_pause.setEnabled(False); self.btn_resume.setEnabled(False)
        self.btn_noisy.setEnabled(False); self.btn_failed.setEnabled(False); self.btn_end.setEnabled(False)

        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_resume.clicked.connect(self.on_resume)
        self.btn_noisy.clicked.connect(lambda: self.add_marker("noisy"))
        self.btn_failed.clicked.connect(lambda: self.add_marker("failed"))
        self.btn_end.clicked.connect(self.on_end)

        # Timers
        self.ui_timer = QtCore.QTimer(); self.ui_timer.timeout.connect(self.tick)
        self.ui_timer.start(30)

        self.phase = "idle"  # idle|prepare|window|iti|paused|finished
        self.phase_end_host_ns = None
        self.trial_table = []

    # ---- Experiment Flow ----
    def on_start(self):
        if self.running: return
        self.reader.start()
        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_end.setEnabled(True)
        self.lbl_status.setText("Running")
        self.advance_trial()

    def on_pause(self):
        if not self.running or self.paused: return
        self.paused = True
        self.phase = "paused"
        self.lbl_status.setText("Paused")
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.btn_noisy.setEnabled(True)
        self.btn_failed.setEnabled(True)
        # Marker written via logging during paused phase

    def on_resume(self):
        if not self.running or not self.paused: return
        self.paused = False
        self.lbl_status.setText("Running")
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_noisy.setEnabled(True)
        self.btn_failed.setEnabled(True)
        # resume remaining time of current phase by shifting end time forward
        if self.phase_end_host_ns is not None:
            self.phase_end_host_ns = now_ns() + 1_000_000  # 1 ms grace; next tick will recompute anyway

    def on_end(self):
        self.finish_run("user_end")

    def add_marker(self, kind):
        # Set a pending marker; it will be embedded on the next sample write.
        self.marker_pending = kind
        # If within a trial, also tag the trial outcome (if more severe than 'ok')
        if self.current_trial and kind in ("noisy", "failed"):
            if kind == "failed":
                self.current_trial.outcome = "failed"
            elif self.current_trial.outcome == "ok":
                self.current_trial.outcome = "noisy"

    def advance_trial(self):
        self.trial_idx += 1
        if self.trial_idx >= len(self.trials):
            self.finish_run("completed")
            return
        self.current_trial = self.trials[self.trial_idx]

        # Next label label text
        nxt = self.trials[self.trial_idx+1].label_name if (self.trial_idx+1 < len(self.trials)) else "—"
        self.lbl_next.setText(f"NEXT: {nxt}")

        # Start PREPARE phase
        self.phase = "prepare"
        self.phase_end_host_ns = now_ns() + int(self.current_trial.prepare_s * 1e9)
        self.lbl_current.setText(f"CURRENT: {self.current_trial.label_name} (get ready)")
        self.btn_noisy.setEnabled(True); self.btn_failed.setEnabled(True)

    def finish_run(self, reason):
        self.running = False
        self.stop_event.set()
        self.lbl_status.setText(f"Finished ({reason})")
        self.btn_pause.setEnabled(False); self.btn_resume.setEnabled(False)
        self.btn_noisy.setEnabled(False); self.btn_failed.setEnabled(False); self.btn_end.setEnabled(False)

        # Save JSON sidecar
        meta = self.run_meta.copy()
        meta["ended_iso"] = datetime.datetime.now().isoformat()
        meta["reason"] = reason
        meta["trials"] = [asdict(t) for t in self.trials]
        with open(self.json_path, "w") as f:
            json.dump(meta, f, indent=2)
        self.logger.close()
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved:\n{self.csv_path}\n{self.json_path}")

    # ---- Data/Plot/State Machine ----
    def drain_queue(self, max_items=1000):
        drained = 0
        while drained < max_items and not self.q.empty():
            s, t_us, vals, host_ns = self.q.get_nowait()
            # Merge for plotting: simple fixed-step time base
            t_s = self.sample_index / FS_VIS
            self.tline.append(t_s)
            for ch in range(N_CH):
                self.ybufs[ch].append(vals[ch])
            self.sample_index += 1

            # Determine label, trial, phase & write
            label_id = -1
            trial_id = -1
            phase = self.phase
            marker = self.marker_pending if self.marker_pending else ""
            if self.current_trial:
                label_id = self.current_trial.label_id if phase == "window" else LABEL_TO_ID["rest"]
                trial_id = self.current_trial.trial_id
            else:
                label_id = LABEL_TO_ID["rest"]
            self.logger.write(
                device_us=t_us,
                host_ns=host_ns,
                stream=s,
                vals=vals,
                label_id=label_id,
                trial_id=trial_id,
                phase=phase,
                marker=marker
            )
            self.marker_pending = None
            drained += 1

    def update_plots(self):
        if not self.tline:
            return
        t_last = self.tline[-1]
        t_min = max(0.0, t_last - WINDOW_SEC)
        x = np.fromiter(self.tline, dtype=np.float32)
        for ch in range(N_CH):
            y = np.fromiter(self.ybufs[ch], dtype=np.float32)
            self.curves[ch].setData(x, y)
            self.plots[ch].setXRange(t_min, t_last)
            self.plots[ch].setYRange(-128, 127)

    def state_machine(self):
        if not self.running or self.paused:
            # While paused, keep labeling as 'paused' via drain_queue() phase field
            return
        now = now_ns()
        if self.phase == "prepare":
            # On first sample during prepare, stamp t_cue_on
            if self.current_trial.t_cue_on_us < 0 and not self.q.empty():
                # best-effort: map first arriving device_us as cue-on
                try:
                    s, t_us, _, _ = self.q.queue[0]
                    self.current_trial.t_cue_on_us = t_us
                except Exception:
                    pass
            remaining = max(0.0, (self.phase_end_host_ns - now) / 1e9)
            self.lbl_count.setText(f"{remaining:0.1f} s")
            if now >= self.phase_end_host_ns:
                # Move to window (GO)
                self.phase = "window"
                self.phase_end_host_ns = now + int(self.current_trial.window_s * 1e9)
                self.lbl_current.setText(f"CURRENT: {self.current_trial.label_name} (DO IT)")
                # stamp t_go_us on first sample observed in window
                self.current_trial.t_go_us = -1

        elif self.phase == "window":
            remaining = max(0.0, (self.phase_end_host_ns - now) / 1e9)
            self.lbl_count.setText(f"{remaining:0.1f} s")
            # Stamp window boundaries using first/last device_us seen in this phase
            if self.current_trial.t_window_start_us < 0 and not self.q.empty():
                try:
                    s, t_us, _, _ = self.q.queue[0]
                    if self.current_trial.t_go_us < 0:
                        self.current_trial.t_go_us = t_us
                    self.current_trial.t_window_start_us = t_us
                except Exception:
                    pass
            if now >= self.phase_end_host_ns:
                # finalize window end (best-effort from last sample in queue)
                try:
                    s, t_us, _, _ = self.q.queue[-1]
                    self.current_trial.t_window_end_us = t_us
                except Exception:
                    self.current_trial.t_window_end_us = -1
                # Move to ITI
                self.phase = "iti"
                iti_ns = int(self.current_trial.iti_s * 1e9)
                self.phase_end_host_ns = now + iti_ns
                self.lbl_current.setText("CURRENT: rest (ITI)")

        elif self.phase == "iti":
            remaining = max(0.0, (self.phase_end_host_ns - now) / 1e9)
            self.lbl_count.setText(f"{remaining:0.1f} s")
            if now >= self.phase_end_host_ns:
                self.advance_trial()

    def tick(self):
        # Update status label
        self.lbl_status.setText("Paused" if self.paused else ("Running" if self.running else "Idle"))

        # Drain serial into buffers and log rows
        self.drain_queue(max_items=1500)

        # Phase machine & countdowns
        self.state_machine()

        # Update plots
        self.update_plots()

    # Clean up
    def closeEvent(self, ev):
        try:
            if self.running:
                self.finish_run("window_closed")
        finally:
            self.stop_event.set()
            super().closeEvent(ev)

# ---------- main ----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = EMGExperiment()
    win.show()
    sys.exit(app.exec_())
