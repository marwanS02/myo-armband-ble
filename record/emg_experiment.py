#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMG Experiment Runner — Myo BLE→UART
------------------------------------
Author: Mohamad Marwan Sidani

What this does
==============
- Connects to your existing BLE→UART gateway (e-lines).
- Shows 8 real-time EMG plots (2×4 grid).
- Guides a participant through timed movement cues with colors:
  * RED   = REST
  * ORANGE= GET READY (3 s, also shows "Next: …")
  * GREEN = DO MOVEMENT (randomized duration)
- Keeps classes balanced by always scheduling from the least-observed classes.
- Lets you Pause/Resume at any time. On Resume, it **aborts** any in-flight cue,
  resets to REST, and schedules a fresh cue.
- Auto-pauses if the serial stream stalls (no new samples for NO_DATA_TIMEOUT).
- During pause, you can mark the last trial as "FAILED" or "NOISY".
- Saves two semicolon-separated CSVs:
    1) samples.csv: per-sample timestamp + 8 EMG + active_label (0..8) + state
    2) events.csv : timestamped markers (START/END movement, GET_READY, PAUSE, RESUME, FAIL, NOISY, AUTOPAUSE)
- Automatically creates a session folder:
    emg_sessions/<ParticipantName>/<YYYYmmdd_HHMMSS>/
- Media panel (image/GIF) updates per movement. Put files in ./media/
  Named like: 1_mild_extension.png, 2_full_extension.gif, etc.
"""

import os, sys, time, csv, json, glob, queue, threading, random, math
from dataclasses import dataclass, field
from collections import deque, defaultdict

import numpy as np
import serial

from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# -------- USER CONFIG (edit to taste) --------
PORT = "COM5"     # Linux: /dev/ttyACM0 or /dev/ttyUSB0; Win: "COM5"; macOS: "/dev/tty.usbmodem*"
BAUD = 115200

N_STREAMS, N_CH = 4, 8
WINDOW_SEC = 5.0
FS_AGG = 200.0
BUF_LEN = int(WINDOW_SEC * FS_AGG) * 2
PRINT_RAW_FIRST_N = 5

# Experiment timing
READY_LEAD_SEC = 3.0                 # preview time before DO
REST_RANGE_SEC = (2.0, 4.0)          # randomized rest durations
MOVE_RANGE_SEC = (2.0, 5.0)          # randomized DO durations
TARGET_REPS_PER_CLASS = 10           # stop when each class reaches this count (or click Stop)

# Auto-pause if no data arrives for this long:
NO_DATA_TIMEOUT = 0.75               # seconds

# Where to save sessions
SESSIONS_ROOT = "emg_sessions"

# Label map (0..8)
LABELS = {
    0: "rest",
    1: "mild_extension",
    2: "full_extension",
    3: "mild_flexion",
    4: "full_flexion",
    5: "mild_radial_flexion",
    6: "full_radial_flexion",
    7: "mild_ulnar_flexion",
    8: "full_ulnar_flexion",
}

# Media path pattern per label id (png/gif/jpg supported)
MEDIA_DIR = "media"

# --------- Serial parsing (same tolerant format) ----------
q = queue.Queue(maxsize=4096)
raw_streams = [deque() for _ in range(N_STREAMS)]
last_sample_walltime = 0.0  # for auto-pause

def parse_line(line: str):
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

def serial_reader(stop_event):
    global last_sample_walltime
    raw_printed = 0
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"[Serial] Open failed: {e}")
        return
    with ser:
        while not stop_event.is_set():
            try:
                line = ser.readline().decode('utf-8', errors='ignore')
                if PRINT_RAW_FIRST_N and raw_printed < PRINT_RAW_FIRST_N:
                    print("[RAW]", line.rstrip())
                    raw_printed += 1
                parsed = parse_line(line)
                if not parsed:
                    continue
                last_sample_walltime = time.time()
                q.put_nowait(parsed)
            except queue.Full:
                try: _ = q.get_nowait()
                except Exception: pass
            except Exception:
                pass

# --------- Data writers ----------
@dataclass
class Writers:
    samples_fp: any = None
    samples_csv: csv.writer = None
    events_fp: any = None
    events_csv: csv.writer = None

    def open(self, session_dir):
        os.makedirs(session_dir, exist_ok=True)
        self.samples_fp = open(os.path.join(session_dir, "samples.csv"), "w", newline="")
        self.events_fp  = open(os.path.join(session_dir, "events.csv"),  "w", newline="")
        self.samples_csv = csv.writer(self.samples_fp, delimiter=';')
        self.events_csv  = csv.writer(self.events_fp,  delimiter=';')

        self.samples_csv.writerow([
            "timestamp_us","participant","state","active_label_id","active_label_name",
            "stream_index","ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7"
        ])
        self.events_csv.writerow([
            "timestamp_us","participant","event","detail"
        ])

    def close(self):
        if self.samples_fp: self.samples_fp.close()
        if self.events_fp:  self.events_fp.close()

# --------- Experiment state machine ----------
class State:
    REST = "REST"
    READY = "READY"
    DO = "DO"
    PAUSED = "PAUSED"
    IDLE = "IDLE"

@dataclass
class Trial:
    label_id: int
    t_ready_start: float
    t_do_start: float
    t_do_end: float
    aborted: bool = False

@dataclass
class Scheduler:
    per_class_done: dict = field(default_factory=lambda: defaultdict(int))

    def pick_next_label(self):
        # Never schedule "rest" (0) as a *class* to collect; it's the background
        # Balance over labels 1..8 by choosing from the least-seen group
        counts = {k: self.per_class_done.get(k, 0) for k in range(1, 9)}
        min_seen = min(counts.values())
        candidates = [k for k, v in counts.items() if v == min_seen]
        return random.choice(candidates)

    def mark_completed(self, label_id):
        if label_id in range(1, 9):
            self.per_class_done[label_id] += 1

    def all_targets_reached(self, target):
        return all(self.per_class_done.get(k, 0) >= target for k in range(1, 9))

# --------- GUI ----------
class EMGExperiment(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Experiment Runner")

        # --- Left: controls ---
        left = QtWidgets.QWidget()
        l = QtWidgets.QFormLayout(left)

        self.participant_edit = QtWidgets.QLineEdit()
        self.participant_edit.setPlaceholderText("Participant name (required)")
        l.addRow("Participant:", self.participant_edit)

        self.port_edit = QtWidgets.QLineEdit(PORT); l.addRow("Serial port:", self.port_edit)
        self.baud_spin = QtWidgets.QSpinBox(); self.baud_spin.setRange(9600, 10000000); self.baud_spin.setValue(BAUD)
        l.addRow("Baud:", self.baud_spin)

        # Status banner
        self.banner = QtWidgets.QLabel("IDLE")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setFixedHeight(40)
        self._set_banner(State.IDLE)
        l.addRow(self.banner)

        # Next preview
        self.next_label = QtWidgets.QLabel("Next: —")
        l.addRow(self.next_label)

        # Buttons
        btnbox = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_resume= QtWidgets.QPushButton("Resume")
        self.btn_stop  = QtWidgets.QPushButton("Stop & Save")
        btnbox.addWidget(self.btn_start); btnbox.addWidget(self.btn_pause)
        btnbox.addWidget(self.btn_resume); btnbox.addWidget(self.btn_stop)
        l.addRow(btnbox)

        # While paused
        markbox = QtWidgets.QHBoxLayout()
        self.btn_fail  = QtWidgets.QPushButton("Mark FAILED")
        self.btn_noisy = QtWidgets.QPushButton("Mark NOISY")
        markbox.addWidget(self.btn_fail); markbox.addWidget(self.btn_noisy)
        l.addRow(markbox)

        # Media
        self.media_label = QtWidgets.QLabel()
        self.media_label.setMinimumSize(280, 180)
        self.media_label.setFrameShape(QtWidgets.QFrame.Box)
        self.media_movie = None  # for GIFs
        l.addRow("Movement demo:", self.media_label)

        # --- Right: plots ---
        right = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(right)
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

        # --- Layout ---
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Data buffers
        self.tline = deque(maxlen=BUF_LEN)
        self.ybufs = [deque(maxlen=BUF_LEN) for _ in range(N_CH)]
        self.sample_index = 0
        self.expected_s = 0

        # Experiment state
        self.state = State.IDLE
        self.current_label = 0  # label applied to samples (0=rest)
        self.active_trial = None
        self.scheduler = Scheduler()

        # Session files
        self.session_dir = None
        self.writers = Writers()

        # timers
        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self.tick)
        self.gui_timer.start(30)

        self.state_timer = QtCore.QTimer()
        self.state_timer.setSingleShot(True)
        self.state_timer.timeout.connect(self._advance_state)

        # buttons
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_resume.clicked.connect(self._on_resume)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_fail.clicked.connect(lambda: self._mark_issue("FAILED"))
        self.btn_noisy.clicked.connect(lambda: self._mark_issue("NOISY"))

        # serial thread
        self.stop_event = threading.Event()
        self.reader_thr = None

        # preview/media init
        self._update_media(0)

    # ---------- UI helpers ----------
    def _set_banner(self, state):
        self.banner.setText(state)
        pal = self.banner.palette()
        color = QtGui.QColor("#444444")
        if state == State.REST:   color = QtGui.QColor("#e74c3c")  # red
        if state == State.READY:  color = QtGui.QColor("#e67e22")  # orange
        if state == State.DO:     color = QtGui.QColor("#2ecc71")  # green
        if state == State.PAUSED: color = QtGui.QColor("#95a5a6")  # gray
        if state == State.IDLE:   color = QtGui.QColor("#7f8c8d")  # slate
        pal.setColor(QtGui.QPalette.Window, color)
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        self.banner.setAutoFillBackground(True)
        self.banner.setPalette(pal)

    def _update_media(self, label_id):
        # Try to load image/gif named like "5_mild_radial_flexion.*"
        name = LABELS.get(label_id, "rest")
        pattern = os.path.join(MEDIA_DIR, f"{label_id}_{name}.*")
        files = sorted(glob.glob(pattern))
        if self.media_movie:
            self.media_movie.stop()
            self.media_movie = None
        if files:
            path = files[0]
            ext = os.path.splitext(path)[1].lower()
            if ext in (".gif",):
                self.media_movie = QtGui.QMovie(path)
                self.media_label.setMovie(self.media_movie)
                self.media_movie.start()
            else:
                pix = QtGui.QPixmap(path)
                if not pix.isNull():
                    self.media_label.setPixmap(pix.scaled(
                        self.media_label.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation))
                else:
                    self.media_label.setText("Media failed to load")
        else:
            self.media_label.setText("(Put media files in ./media)")
        # Also update the "Next:" preview text
        if label_id == 0:
            self.next_label.setText("Next: —")
        else:
            self.next_label.setText(f"Next: {LABELS[label_id].replace('_',' ').title()}")

    # ---------- Buttons ----------
    def _on_start(self):
        if self.state != State.IDLE:
            return
        participant = self.participant_edit.text().strip()
        if not participant:
            QtWidgets.QMessageBox.warning(self, "Missing", "Please enter participant name.")
            return

        # Override port/baud
        global PORT, BAUD
        PORT = self.port_edit.text().strip() or PORT
        BAUD = int(self.baud_spin.value())

        # Prepare session dir
        tstr = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(SESSIONS_ROOT, participant, tstr)
        os.makedirs(self.session_dir, exist_ok=True)

        # Save a small config
        with open(os.path.join(self.session_dir, "config.json"), "w") as fp:
            json.dump({
                "participant": participant,
                "port": PORT,
                "baud": BAUD,
                "window_sec": WINDOW_SEC,
                "fs_agg": FS_AGG,
                "ready_lead_sec": READY_LEAD_SEC,
                "rest_range_sec": REST_RANGE_SEC,
                "move_range_sec": MOVE_RANGE_SEC,
                "target_reps_per_class": TARGET_REPS_PER_CLASS
            }, fp, indent=2)

        # Open writers
        self.writers.open(self.session_dir)

        # Start serial thread
        self.stop_event.clear()
        self.reader_thr = threading.Thread(target=serial_reader, args=(self.stop_event,), daemon=True)
        self.reader_thr.start()

        # Reset counters and schedule first cue
        self.state = State.REST
        self.current_label = 0
        self._set_banner(State.REST)
        self._log_event("SESSION_START", "—")
        self._schedule_next_trial()

    def _on_pause(self):
        if self.state == State.PAUSED or self.state == State.IDLE:
            return
        # Abort current trial if any
        self._abort_active_trial(reason="USER_PAUSE")
        self.state = State.PAUSED
        self.current_label = 0
        self._set_banner(State.PAUSED)
        self.state_timer.stop()
        self._log_event("PAUSE", "user")

    def _on_resume(self):
        if self.state != State.PAUSED:
            return
        # Resume into REST → new trial
        self.state = State.REST
        self.current_label = 0
        self._set_banner(State.REST)
        self._log_event("RESUME", "user")
        self._schedule_next_trial()

    def _on_stop(self):
        # Stop everything and close files
        self._abort_active_trial(reason="STOPPED")
        self.state = State.IDLE
        self.current_label = 0
        self._set_banner(State.IDLE)
        self.state_timer.stop()
        self._log_event("SESSION_END", "—")
        self.writers.close()
        try:
            self.stop_event.set()
        except Exception:
            pass
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{self.session_dir}")

    def _mark_issue(self, tag):
        if self.state != State.PAUSED:
            QtWidgets.QMessageBox.information(self, "Note", "Mark issues while paused.")
            return
        self._log_event(tag, "user")

    # ---------- Scheduling ----------
    def _schedule_next_trial(self):
        # If all targets hit, we can stop automatically (or keep cycling if you prefer)
        if self.scheduler.all_targets_reached(TARGET_REPS_PER_CLASS):
            QtWidgets.QMessageBox.information(self, "Done", "Target reps per class reached. Stopping.")
            self._on_stop()
            return

        # Choose next label (1..8)
        next_label = self.scheduler.pick_next_label()
        self._update_media(next_label)

        # Set preview for 3s
        self.state = State.READY
        self._set_banner(State.READY)
        self._log_event("GET_READY", LABELS[next_label])
        # Build trial timings (absolute walltimes in *experiment clock* sense)
        now = time.time()
        do_duration = random.uniform(*MOVE_RANGE_SEC)
        self.active_trial = Trial(
            label_id=next_label,
            t_ready_start=now,
            t_do_start=now + READY_LEAD_SEC,
            t_do_end=now + READY_LEAD_SEC + do_duration
        )
        # Start 3s countdown to DO
        self.state_timer.start(int(READY_LEAD_SEC * 1000))

    def _advance_state(self):
        # Called when READY->DO or DO->REST transitions elapse
        if self.state == State.READY and self.active_trial:
            # Begin DO
            self.state = State.DO
            self.current_label = self.active_trial.label_id
            self._set_banner(State.DO)
            self._log_event("MOVE_START", LABELS[self.current_label])
            # schedule DO end
            remain = max(0.0, self.active_trial.t_do_end - time.time())
            self.state_timer.start(int(remain * 1000))
            return

        if self.state == State.DO and self.active_trial:
            # End DO → REST with random rest duration
            self._log_event("MOVE_END", LABELS[self.active_trial.label_id])
            self.scheduler.mark_completed(self.active_trial.label_id)
            self.active_trial = None
            self.state = State.REST
            self.current_label = 0
            self._set_banner(State.REST)
            rest_dur = random.uniform(*REST_RANGE_SEC)
            self.state_timer.start(int(rest_dur * 1000))
            # after REST, schedule next trial
            QtCore.QTimer.singleShot(int(rest_dur * 1000), self._schedule_next_trial)

    def _abort_active_trial(self, reason="ABORT"):
        if self.active_trial:
            self._log_event("MOVE_ABORT", f"{LABELS[self.active_trial.label_id]}|{reason}")
            self.active_trial = None
        # Also clear preview
        self._update_media(0)
        self.next_label.setText("Next: —")

    # ---------- Logging ----------
    def _log_event(self, event, detail):
        t_us = int(time.time() * 1e6)
        part = self.participant_edit.text().strip() or "-"
        try:
            self.writers.events_csv.writerow([t_us, part, event, detail])
        except Exception:
            pass

    def _log_sample(self, t_us, stream_idx, vals):
        # Per-sample row with current label & state
        part = self.participant_edit.text().strip() or "-"
        try:
            self.writers.samples_csv.writerow([
                t_us, part, self.state, self.current_label, LABELS.get(self.current_label, "rest"),
                stream_idx, *vals
            ])
        except Exception:
            pass

    # ---------- Main tick ----------
    def tick(self):
        # 1) Drain serial queue → per-stream buffers
        drained = 0
        while drained < 1000 and not q.empty():
            s, t_us, vals = q.get_nowait()
            raw_streams[s].append((t_us, vals))
            self._log_sample(t_us, s, vals)
            drained += 1

        # 2) Merge by round-robin
        merges = 0
        while merges < 1000:
            if any(len(raw_streams[s]) == 0 for s in range(N_STREAMS)):
                break
            s = self.expected_s
            t_us, vals = raw_streams[s].popleft()

            t_s = self.sample_index / FS_AGG
            self.tline.append(t_s)
            for ch in range(N_CH):
                self.ybufs[ch].append(vals[ch])
            self.sample_index += 1
            self.expected_s = (self.expected_s + 1) % N_STREAMS
            merges += 1

        # 3) Update plots
        if self.tline:
            t_last = self.tline[-1]
            t_min = max(0.0, t_last - WINDOW_SEC)
            x = np.fromiter(self.tline, dtype=np.float32)
            for ch in range(N_CH):
                y = np.fromiter(self.ybufs[ch], dtype=np.float32)
                self.curves[ch].setData(x, y)
                self.plots[ch].setXRange(t_min, t_last)
                self.plots[ch].setYRange(-128, 127)

        # 4) Auto-pause if no new data
        if self.state not in (State.IDLE, State.PAUSED):
            if time.time() - last_sample_walltime > NO_DATA_TIMEOUT:
                # Only trigger once per outage
                self._log_event("AUTOPAUSE_NO_DATA", f">{NO_DATA_TIMEOUT}s")
                self._on_pause()

        # 5) Update "Next:" countdown during READY
        if self.state == State.READY and self.active_trial:
            remain = max(0.0, self.active_trial.t_do_start - time.time())
            name = LABELS[self.active_trial.label_id].replace("_", " ").title()
            self.next_label.setText(f"Next: {name} ({remain:0.1f}s)")

# ---------- Main ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = EMGExperiment()
    win.resize(1400, 800)
    win.show()

    def on_quit():
        try:
            win.stop_event.set()
        except Exception:
            pass
        try:
            win.writers.close()
        except Exception:
            pass

    app.aboutToQuit.connect(on_quit)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
