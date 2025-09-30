"""
Myo EMG — Real-Time Plotter & Stream Merger
-------------------------------------------
Author: Mohamad Marwan Sidani

Overview
- Reads EMG frames from a UART serial port that bridges BLE→UART.
- Accepts tolerant lines 'eK,<timestamp_us>,v0,...,v7' (K=0..3).
- Merges up to 4 streams in round-robin, builds a shared timeline,
  and displays 8 channels using PyQtGraph (2×4 grid).

Key parameters (edit below)
- PORT, BAUD: serial port settings
- WINDOW_SEC: visible time window (s)
- FS_AGG: aggregate sample rate per channel after merging (Hz)
- N_STREAMS, N_CH: number of BLE sources and channels (8)
- BUF_LEN: plotting buffer length
- PRINT_RAW_FIRST_N: print first raw lines for debugging

Threads & Queues
- A serial reader thread pushes parsed frames into a Queue.
- GUI timer drains the queue, merges frames, and updates plots.

Notes
- Uses relaxed parsing and skips when queues become imbalanced.
- For Windows use 'COMx' for PORT; Linux '/dev/ttyACM0' or '/dev/ttyUSB0'; macOS '/dev/tty.usbmodem*'.
"""

import sys, time, serial, threading, queue
import numpy as np
from collections import deque
from PyQt5 import QtWidgets
import pyqtgraph as pg

# ---------- CONFIG ----------
PORT = "/dev/ttyUSB0"     # Linux: /dev/ttyACM0 or /dev/ttyUSB0; Windows: "COM5"; macOS: "/dev/tty.usbmodem*"
BAUD = 115200
WINDOW_SEC = 5.0          # visible window
FS_AGG = 200.0            # aggregate sample rate per channel after merging (Hz)
N_STREAMS, N_CH = 4, 8
BUF_LEN = int(WINDOW_SEC * FS_AGG) * 2
PRINT_RAW_FIRST_N = 10    # set 0 to disable raw prints

# ---------- THREAD PIPE ----------
q = queue.Queue(maxsize=4096)

# Per-stream raw buffers before merging (store tuples (t_us, [8]))
raw_streams = [deque() for _ in range(N_STREAMS)]

# Merged timeline buffers (shared time + per-channel data)
tline = deque(maxlen=BUF_LEN)                 # time in seconds
ybufs = [deque(maxlen=BUF_LEN) for _ in range(N_CH)]

# ---------- PARSER ----------
def parse_line(line: str):
    """
    Accept tolerant 'eK,<timestamp>,v0,...,v7' with extra spaces/CRLF.
    Returns (s, t_us, vals[8]) or None.
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

# ---------- SERIAL READER THREAD ----------
def serial_reader(stop_event):
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
                q.put_nowait(parsed)
            except queue.Full:
                # Drop oldest to keep up
                try: _ = q.get_nowait()
                except Exception: pass
            except Exception:
                pass

# ---------- GUI ----------
class EMGWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Myo EMG — merged 8 channels")
        cw = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(cw); self.setCentralWidget(cw)

        # 2 rows × 4 columns layout for 8 channels
        self.plots, self.curves = [], []
        for ch in range(N_CH):
            plt = pg.PlotWidget()
            plt.showGrid(x=True, y=True, alpha=0.25)
            plt.setLabel('left', f'ch{ch}')
            plt.setLabel('bottom', 'time (s)')
            curve = plt.plot([], [])
            r = ch // 4
            c = ch % 4
            grid.addWidget(plt, r, c)
            self.plots.append(plt); self.curves.append(curve)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

        # merging state
        self.expected_s = 0
        self.sample_index = 0        # merged sample counter to form time axis

        self.last_status_t = time.time()
        self.parsed_count = 0
        self.merged_count = 0
        self.skips = 0

    def drain_queue(self, max_items=500):
        drained = 0
        while drained < max_items and not q.empty():
            s, t_us, vals = q.get_nowait()
            raw_streams[s].append((t_us, vals))
            self.parsed_count += 1
            drained += 1

    def merge_roundrobin(self, max_merge=200):
        """
        Merge by strict round-robin: expect s0,s1,s2,s3 repeating.
        If a stream lags, wait a little; if imbalance gets large, skip to next available to avoid freezing.
        """
        merges = 0
        while merges < max_merge:
            # If any stream is empty, stop (can't complete a full cycle)
            if any(len(raw_streams[s]) == 0 for s in range(N_STREAMS)):
                break

            # Pop from expected stream
            s = self.expected_s
            if len(raw_streams[s]) == 0:
                # extreme case, shouldn't happen due to check above
                break

            t_us, vals = raw_streams[s].popleft()
            # build timeline time from sample_index
            t_s = self.sample_index / FS_AGG
            tline.append(t_s)
            for ch in range(N_CH):
                ybufs[ch].append(vals[ch])
            self.sample_index += 1
            self.merged_count += 1

            # next stream expected
            self.expected_s = (self.expected_s + 1) % N_STREAMS
            merges += 1

        # If one stream starves and others pile up, allow a gentle resync to avoid UI stall
        lengths = [len(raw_streams[s]) for s in range(N_STREAMS)]
        if max(lengths) - min(lengths) > 50:
            # skip to the stream with the largest backlog to catch up
            self.expected_s = int(np.argmax(lengths))
            self.skips += 1

    def update_plots(self):
        if not tline:
            return
        t_last = tline[-1]
        t_min = max(0.0, t_last - WINDOW_SEC)
        for ch in range(N_CH):
            # Convert deques to numpy views
            y = np.fromiter(ybufs[ch], dtype=np.float32)
            # All channels share the same time axis
            # To avoid copying tline eight times, rebuild x only once:
            # But here for clarity we recompute per channel (cheap at these sizes)
            x = np.fromiter(tline, dtype=np.float32)
            # Show only the last WINDOW_SEC
            # (set XRange; data still full buffer for simplicity)
            self.curves[ch].setData(x, y)
            self.plots[ch].setXRange(t_min, t_last)
            self.plots[ch].setYRange(-128, 127)

    def tick(self):
        self.drain_queue(max_items=1000)
        self.merge_roundrobin(max_merge=1000)
        self.update_plots()

        now = time.time()
        if now - self.last_status_t > 1.0:
            self.setWindowTitle(
                f"Myo EMG — merged 8 channels | parsed={self.parsed_count} merged={self.merged_count} skips={self.skips}"
            )
            self.last_status_t = now

# ---------- MAIN ----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = EMGWindow(); win.show()

    stop_event = threading.Event()
    t = threading.Thread(target=serial_reader, args=(stop_event,), daemon=True)
    t.start()

    def on_close():
        stop_event.set()
        time.sleep(0.2)

    app.aboutToQuit.connect(on_close)
    sys.exit(app.exec_())
