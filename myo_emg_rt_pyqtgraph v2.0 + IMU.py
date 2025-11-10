"""
Myo EMG+IMU — Real-Time Plotter & Stream Merger
Author: Mohamad Marwan Sidani

Adds IMU streaming support:
- Accepts 'iK,<t_us>,qw,qx,qy,qz,ax,ay,az,gx,gy,gz'
- Scales to float and plots accel/gyro; shows RPY from quaternion.

Myo nominal scales (can be tuned if your unit differs):
  QUAT_SCALE   = 16384.0  # unit quaternion components = int16 / 16384
  ACC_LSB_PER_G= 2048.0   # g = ax/2048; m/s^2 = (ax/2048)*9.80665
  GYR_LSB_PER_DPS = 16.0  # deg/s = gx/16
"""
import sys, time, serial, threading, queue, math
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ---------- CONFIG ----------
PORT = "COM10"
BAUD = 115200
WINDOW_SEC = 5.0
FS_AGG = 200.0
N_STREAMS, N_CH = 4, 8
BUF_LEN = int(WINDOW_SEC * FS_AGG) * 2
PRINT_RAW_FIRST_N = 10

# IMU scales (tweak here if needed)
QUAT_SCALE = 16384.0
ACC_LSB_PER_G = 2048.0
GYR_LSB_PER_DPS = 16.0

# ---------- THREAD PIPE ----------
q = queue.Queue(maxsize=4096)

# Per-stream raw buffers (EMG)
raw_streams = [deque() for _ in range(N_STREAMS)]
# IMU buffer (single merged timeline; one Myo assumed, but supports source K)
imu_t = deque(maxlen=BUF_LEN)
acc_buf = [deque(maxlen=BUF_LEN) for _ in range(3)]  # ax,ay,az
gyr_buf = [deque(maxlen=BUF_LEN) for _ in range(3)]  # gx,gy,gz
quat_latest = [0.0, 0.0, 0.0, 1.0]  # qx,qy,qz,qw for display

# Merged EMG timeline buffers
tline = deque(maxlen=BUF_LEN)
ybufs = [deque(maxlen=BUF_LEN) for _ in range(N_CH)]

def parse_line(line: str):
    sline = line.strip()
    if not sline or len(sline) < 3:
        return None
    tag = sline[0]
    if tag not in ('e','i'):
        return None
    if not sline[1].isdigit():
        return None
    try:
        s = int(sline[1])
    except ValueError:
        return None

    parts = [p.strip() for p in sline.split(',')]
    if tag == 'e':
        if len(parts) < 10: return None
        try:
            t_us = int(parts[1]); vals = [int(p) for p in parts[2:10]]
        except ValueError:
            return None
        if not (0 <= s < N_STREAMS) or len(vals) != 8:
            return None
        return ('e', s, t_us, vals)

    elif tag == 'i':
        if len(parts) < 12: return None
        try:
            t_us = int(parts[1])
            qw,qx,qy,qz = [int(parts[i]) for i in range(2,6)]
            ax,ay,az    = [int(parts[i]) for i in range(6,9)]
            gx,gy,gz    = [int(parts[i]) for i in range(9,12)]
        except ValueError:
            return None
        return ('i', s, t_us, (qw,qx,qy,qz, ax,ay,az, gx,gy,gz))

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
                    print("[RAW]", line.rstrip()); raw_printed += 1
                parsed = parse_line(line)
                if not parsed: continue
                q.put_nowait(parsed)
            except queue.Full:
                try: _ = q.get_nowait()
                except Exception: pass
            except Exception:
                pass

# ---- math helpers ----
def quat_to_rpy(qw,qx,qy,qz):
    # normalize
    norm = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)
    if norm == 0: return (0.0,0.0,0.0)
    qw,qx,qy,qz = qw/norm, qx/norm, qy/norm, qz/norm
    # ZYX yaw-pitch-roll
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    sinp = 2*(qw*qy - qz*qx)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))
    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    return roll, pitch, yaw

class EMGIMUWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Myo EMG+IMU — real-time viewer")

        tabs = QtWidgets.QTabWidget(); self.setCentralWidget(tabs)

        # ---- EMG tab (2x4 grid) ----
        emg_w = QtWidgets.QWidget(); g = QtWidgets.QGridLayout(emg_w)
        self.emg_plots, self.emg_curves = [], []
        for ch in range(N_CH):
            plt = pg.PlotWidget()
            plt.showGrid(x=True, y=True, alpha=0.25)
            plt.setLabel('left', f'ch{ch}')
            plt.setLabel('bottom', 'time (s)')
            curve = plt.plot([], [])
            g.addWidget(plt, ch//4, ch%4)
            self.emg_plots.append(plt); self.emg_curves.append(curve)
        tabs.addTab(emg_w, "EMG (8ch)")

        # ---- IMU accel (3 traces) ----
        self.acc_plt = pg.PlotWidget(); self.acc_plt.showGrid(x=True,y=True,alpha=0.25)
        self.acc_plt.setLabel('left', 'acc (m/s^2)'); self.acc_plt.setLabel('bottom','time (s)')
        self.acc_curves = [self.acc_plt.plot([], []) for _ in range(3)]
        tabs.addTab(self.acc_plt, "IMU — Accel")

        # ---- IMU gyro (3 traces) ----
        self.gyr_plt = pg.PlotWidget(); self.gyr_plt.showGrid(x=True,y=True,alpha=0.25)
        self.gyr_plt.setLabel('left', 'gyro (deg/s)'); self.gyr_plt.setLabel('bottom','time (s)')
        self.gyr_curves = [self.gyr_plt.plot([], []) for _ in range(3)]
        tabs.addTab(self.gyr_plt, "IMU — Gyro")

        # ---- Orientation quick readout ----
        ori_w = QtWidgets.QWidget(); vbox = QtWidgets.QVBoxLayout(ori_w)
        self.lbl_q = QtWidgets.QLabel("q (w,x,y,z):")
        self.lbl_rpy = QtWidgets.QLabel("RPY (deg):")
        f = ori_w.font(); f.setPointSize(12)
        self.lbl_q.setFont(f); self.lbl_rpy.setFont(f)
        vbox.addWidget(self.lbl_q); vbox.addWidget(self.lbl_rpy); vbox.addStretch(1)
        tabs.addTab(ori_w, "IMU — Orientation")

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.tick); self.timer.start(30)

        # EMG merge state
        self.expected_s = 0
        self.sample_index = 0
        self.last_status_t = time.time()
        self.parsed_count = 0; self.merged_count = 0; self.skips = 0

    def drain_queue(self, max_items=1000):
        drained = 0
        while drained < max_items and not q.empty():
            item = q.get_nowait(); drained += 1
            tag = item[0]
            if tag == 'e':
                _, s, t_us, vals = item
                raw_streams[s].append((t_us, vals))
            else:
                _, s, t_us, payload = item
                qw,qx,qy,qz, ax,ay,az, gx,gy,gz = payload
                t_s = t_us * 1e-6
                imu_t.append(t_s)
                # Scale:
                qw_f = qw/QUAT_SCALE; qx_f = qx/QUAT_SCALE; qy_f = qy/QUAT_SCALE; qz_f = qz/QUAT_SCALE
                ax_ms2 = (ax/ACC_LSB_PER_G)*9.80665
                ay_ms2 = (ay/ACC_LSB_PER_G)*9.80665
                az_ms2 = (az/ACC_LSB_PER_G)*9.80665
                gx_dps = gx/GYR_LSB_PER_DPS; gy_dps = gy/GYR_LSB_PER_DPS; gz_dps = gz/GYR_LSB_PER_DPS
                for i,v in enumerate((ax_ms2,ay_ms2,az_ms2)): acc_buf[i].append(v)
                for i,v in enumerate((gx_dps,gy_dps,gz_dps)): gyr_buf[i].append(v)
                # Store latest in (qx,qy,qz,qw) order for display helper
                quat_latest[0], quat_latest[1], quat_latest[2], quat_latest[3] = qx_f,qy_f,qz_f,qw_f
            self.parsed_count += 1

    def merge_roundrobin(self, max_merge=200):
        merges = 0
        while merges < max_merge:
            if any(len(raw_streams[s]) == 0 for s in range(N_STREAMS)):
                break
            s = self.expected_s
            t_us, vals = raw_streams[s].popleft()
            t_s = self.sample_index / FS_AGG
            tline.append(t_s)
            for ch in range(N_CH): ybufs[ch].append(vals[ch])
            self.sample_index += 1; self.merged_count += 1
            self.expected_s = (self.expected_s + 1) % N_STREAMS
            merges += 1

        lengths = [len(raw_streams[s]) for s in range(N_STREAMS)]
        if max(lengths) - min(lengths) > 50:
            self.expected_s = int(np.argmax(lengths)); self.skips += 1

    def update_emg_plots(self):
        if not tline: return
        t_last = tline[-1]; t_min = max(0.0, t_last - WINDOW_SEC)
        x = np.fromiter(tline, dtype=np.float32)
        for ch in range(N_CH):
            y = np.fromiter(ybufs[ch], dtype=np.float32)
            self.emg_curves[ch].setData(x, y)
            self.emg_plots[ch].setXRange(t_min, t_last)
            self.emg_plots[ch].setYRange(-128, 127)

    def update_imu_plots(self):
        if not imu_t: return
        t_last = imu_t[-1]; t_min = max(0.0, t_last - WINDOW_SEC)
        x = np.fromiter(imu_t, dtype=np.float32)
        # accel
        for i in range(3):
            y = np.fromiter(acc_buf[i], dtype=np.float32)
            self.acc_curves[i].setData(x, y)
        self.acc_plt.setXRange(t_min, t_last)
        # gyro
        for i in range(3):
            y = np.fromiter(gyr_buf[i], dtype=np.float32)
            self.gyr_curves[i].setData(x, y)
        self.gyr_plt.setXRange(t_min, t_last)
        # orientation labels
        qx,qy,qz,qw = quat_latest
        roll,pitch,yaw = quat_to_rpy(qw,qx,qy,qz)
        self.lbl_q.setText(f"q (w,x,y,z): {qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}")
        self.lbl_rpy.setText(f"RPY (deg):  R={roll:+6.1f}  P={pitch:+6.1f}  Y={yaw:+6.1f}")

    def tick(self):
        self.drain_queue(max_items=2000)
        self.merge_roundrobin(max_merge=2000)
        self.update_emg_plots()
        self.update_imu_plots()
        now = time.time()
        if now - self.last_status_t > 1.0:
            self.setWindowTitle(
                f"Myo EMG+IMU | parsed={self.parsed_count} merged={self.merged_count} skips={self.skips}"
            )
            self.last_status_t = now

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = EMGIMUWindow(); win.show()
    stop_event = threading.Event()
    t = threading.Thread(target=serial_reader, args=(stop_event,), daemon=True); t.start()
    def on_close(): stop_event.set(); time.sleep(0.2)
    app.aboutToQuit.connect(on_close)
    sys.exit(app.exec_())
