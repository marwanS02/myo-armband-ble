"""
EMG Real-Time → Model → Smoothed Probabilities → Live Histogram (5 classes)
Author: Mohamad Marwan Sidani

- Reads tolerant UART lines: 'eK,<timestamp_us>,v0,...,v7' (K=0..3).
- Merges up to 4 streams (round-robin) into 8-ch signal @ FS_AGG.
- Preprocess: DC remove → optional stateful bandpass+notch → rect → moving RMS → baseline norm.
- Model inference (your checkpoint), temperature-scale (T*), 9→5 merges.
- Smooth class probabilities (EMA + optional confidence-adaptive).
- τ* gating on the **smoothed** probabilities (stable predicted class).
- Live histogram displays smoothed probs + big predicted label.

Dependencies: pyqtgraph, PyQt5, numpy, torch, (optional) scipy, pyserial
"""

import os, sys, time, json, threading, queue, math, warnings
from collections import deque
import numpy as np

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import socket
import math  # already imported

# --- IMU parse & scaling ---
QUAT_SCALE = 16384.0  # qw,qx,qy,qz scale
IMU_FRESH_S = 0.5     # ignore IMU if older than this (s)

def quat_to_rpy(qw,qx,qy,qz):
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz) or 1.0
    qw,qx,qy,qz = qw/n, qx/n, qy/n, qz/n
    # ZYX
    siny_cosp = 2*(qw*qz + qx*qy);  cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    sinp = 2*(qw*qy - qz*qx);       pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))
    sinr_cosp = 2*(qw*qx + qy*qz);  cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    return roll, pitch, yaw

CLASS_SINK = ("127.0.0.1", 6001)


# ---------- CONFIG: I/O ----------
PORT = "COM10"       # Linux: '/dev/ttyACM0' or '/dev/ttyUSB0'; Windows: 'COMx'; macOS: '/dev/tty.usbmodem*'
BAUD = 115200
N_STREAMS = 4
N_CH = 8

# ---------- CONFIG: sampling / windows ----------
FS_AGG = 200.0              # aggregate sample rate after merging (Hz)
WINDOW_MS = 200             # window length (ms)
HOP_MS = 30                 # hop length (ms)
ENV_MS = 50                 # envelope RMS window (ms)
BASELINE_WARMUP_SEC = 3.0   # seconds of windows to estimate baseline
DISPLAY_SEC = 5.0

# ---------- CONFIG: model + calibration ----------
RUN_DIR = r"C:\Users\Marwa\Documents\GitHub\myo-armband-ble\train\models\EMGModel\2025-10-16_08-58-00_with_loss_0.5539"
USE_GPU = True

T_STAR = 0.929
CLASS_NAMES = ["rest","extension","flexion","radial_flexion","ulnar_flexion"]
IDX_REST = CLASS_NAMES.index("rest")
TAU_STAR = np.array([0.688, 0.121, 0.715, 0.307, 0.344], dtype=np.float32)

# ---------- CONFIG: filters (optional; script works without SciPy) ----------
USE_FILTERS = True          # if scipy unavailable, script auto-falls back to no filtering
BP_LO, BP_HI = 20.0, 90.0   # bandpass
BP_ORDER = 4
NOTCH_Q = 30.0              # 50 Hz notch; set None to disable

# ---------- CONFIG: probability smoothing ----------
# EMA time-constant in seconds (try 0.15–0.30 for smooth + responsive)
SMOOTH_TAU_S = 0.20
SMOOTH_ADAPT = True     # adapt EMA based on confidence (top prob)
SMOOTH_TOP_LO = 0.35    # below this → stronger smoothing
SMOOTH_TOP_HI = 0.75    # above this → more responsive
SMOOTH_MEDIAN_K = 1     # optional post-EMA median window (odd). 1 disables.

# ---------- UART tolerant parser ----------
def parse_line(line: str):
    """
    Accept tolerant:
      EMG: 'eK,<timestamp>,v0..v7'
      IMU: 'iK,<timestamp>,qw,qx,qy,qz,ax,ay,az,gx,gy,gz' (we only need quat for pitch)
    Returns a tuple tagged with 'e' or 'i', or None.
    """
    s = line.strip()
    if len(s) < 3 or not s[1].isdigit():
        return None
    tag = s[0]
    parts = [p.strip() for p in s.split(',')]
    try:
        K = int(s[1])
    except ValueError:
        return None

    if tag == 'e':
        if len(parts) < 10: return None
        try:
            t_us = int(parts[1])
            vals = [int(p) for p in parts[2:10]]
        except ValueError:
            return None
        return ('e', K, t_us, vals)

    if tag == 'i':
        if len(parts) < 12: return None  # iK, t_us, 10 ints
        try:
            t_us = int(parts[1])
            qw,qx,qy,qz = [int(parts[i]) for i in range(2,6)]
            # ax,ay,az,gx,gy,gz = parts[6:12]  # not needed for gating now
        except ValueError:
            return None
        return ('i', K, t_us, (qw,qx,qy,qz))

    return None


# ---------- Serial thread ----------
def serial_reader(stop_event: threading.Event, q: queue.Queue):
    try:
        import serial
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"[Serial] Open failed: {e}")
        return
    with ser:
        while not stop_event.is_set():
            try:
                line = ser.readline().decode('utf-8', errors='ignore')
                item = parse_line(line)
                if item:
                    q.put_nowait(item)  # ('e',...) or ('i',...)
            except queue.Full:
                try: _ = q.get_nowait()
                except Exception: pass
            except Exception:
                pass

# ---------- Round-robin merger ----------
class StreamMerger:
    def __init__(self, fs_agg=200.0):
        self.buffers = [deque() for _ in range(N_STREAMS)]
        self.expected = 0
        self.sample_index = 0
        self.fs = fs_agg
        self.t_samples = deque(maxlen=int(DISPLAY_SEC*fs_agg)*2)
        self.y_ch = [deque(maxlen=int(DISPLAY_SEC*fs_agg)*2) for _ in range(N_CH)]
        self.skips = 0

    def push(self, s, t_us, vals):
        self.buffers[s].append((t_us, vals))

    def merge_some(self, max_merge=2000):
        merges = 0
        while merges < max_merge:
            if any(len(self.buffers[s]) == 0 for s in range(N_STREAMS)):
                break
            s = self.expected
            t_us, vals = self.buffers[s].popleft()
            t = self.sample_index / self.fs
            self.t_samples.append(t)
            for ch in range(N_CH):
                self.y_ch[ch].append(vals[ch])
            self.sample_index += 1
            self.expected = (self.expected + 1) % N_STREAMS
            merges += 1

        lens = [len(self.buffers[s]) for s in range(N_STREAMS)]
        if max(lens) - min(lens) > 50:
            self.expected = int(np.argmax(lens))
            self.skips += 1

    def pull_latest_block(self, needed):
        """Return latest `needed` samples as (8, needed) float32, or None if not enough."""
        if len(self.y_ch[0]) < needed:
            return None
        X = np.stack([np.fromiter(self.y_ch[ch], dtype=np.float32)[-needed:] for ch in range(N_CH)], axis=0)
        return X

# ---------- Preprocess (stateful) ----------
def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    # x: (C,T) non-negative
    if win <= 1:
        return np.sqrt(x)
    pad = win - 1
    x2 = x**2
    csum = np.cumsum(np.pad(x2, ((0,0),(pad,0))), axis=1)
    wsum = csum[:, win:] - csum[:, :-win]
    left = np.repeat(wsum[:, :1], pad, axis=1)
    rms = np.sqrt(np.concatenate([left, wsum], axis=1) / float(win))
    return rms

class StatefulFilters:
    def __init__(self, fs, use_filters=True):
        self.fs = fs
        self.use = use_filters
        self.have_scipy = False
        self.bp = None
        self.notch = None
        self.zi_bp = None
        self.zi_notch = None
        if self.use:
            try:
                from scipy.signal import butter, iirnotch, lfilter_zi
                self.have_scipy = True
                b_bp, a_bp = butter(BP_ORDER, [BP_LO/(fs/2.0), BP_HI/(fs/2.0)], btype='band')
                self.bp = (b_bp, a_bp)
                if NOTCH_Q is not None:
                    # NB: iirnotch expects normalized w0 in rad/sec domain; this version uses normalized freq
                    b_n, a_n = iirnotch(50.0/(fs/2.0), NOTCH_Q)
                    self.notch = (b_n, a_n)
                # init zi per channel
                self.zi_bp = [lfilter_zi(b_bp, a_bp) * 0.0 for _ in range(N_CH)] if self.bp else [None]*N_CH
                self.zi_notch = [lfilter_zi(*self.notch) * 0.0 for _ in range(N_CH)] if self.notch else [None]*N_CH
            except Exception as e:
                warnings.warn(f"[Filters] SciPy unavailable or failed ({e}); proceeding without filters.")
                self.have_scipy = False
                self.use = False

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        x: (C,T) float32
        Returns filtered x with state carried across calls.
        """
        if not self.use or not self.have_scipy:
            return x
        from scipy.signal import lfilter
        y = x
        if self.bp is not None:
            b,a = self.bp
            y_f = np.empty_like(y)
            for ch in range(N_CH):
                y_f[ch], self.zi_bp[ch] = lfilter(b, a, y[ch], zi=self.zi_bp[ch])
            y = y_f
        if self.notch is not None:
            b,a = self.notch
            y_f = np.empty_like(y)
            for ch in range(N_CH):
                y_f[ch], self.zi_notch[ch] = lfilter(b, a, y[ch], zi=self.zi_notch[ch])
            y = y_f
        return y

class Preprocessor:
    def __init__(self, fs, win_ms, env_ms):
        self.fs = fs
        self.win_samp = int(round(fs * win_ms / 1000.0))
        self.env_win = int(max(1, round(fs * env_ms / 1000.0)))
        self.filters = StatefulFilters(fs, USE_FILTERS)
        self.baseline = np.ones(N_CH, dtype=np.float32)
        self.have_baseline = False
        self._warm_envelopes = []

    def warmup_baseline(self, raw_win: np.ndarray):
        env = self._compute_envelope(raw_win)
        self._warm_envelopes.append(env)
        need_warm = int(BASELINE_WARMUP_SEC * self.fs / max(1, self.win_samp//2))
        if len(self._warm_envelopes) >= max(6, need_warm):
            big = np.concatenate(self._warm_envelopes, axis=1)
            base = np.median(big, axis=1)
            base[base == 0] = 1.0
            self.baseline = base.astype(np.float32)
            self.have_baseline = True
            self._warm_envelopes.clear()

    def _compute_envelope(self, raw_win: np.ndarray) -> np.ndarray:
        x = raw_win - raw_win.mean(axis=1, keepdims=True)  # DC
        x = self.filters.apply(x)
        x = np.abs(x)
        env = moving_rms(x, self.env_win)
        return env

    def process(self, raw_win: np.ndarray) -> np.ndarray:
        env = self._compute_envelope(raw_win)
        env_norm = env / (self.baseline[:, None] + 1e-6)
        return env_norm.astype(np.float32)

# ---------- Model + postprocess ----------
def undot(d):
    out = {}
    for k, v in d.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out

def load_model(run_dir: str):
    import torch
    sys.path.insert(0, os.path.dirname(run_dir))
    from networks import EMGModel  # your class
    device = torch.device("cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    with open(os.path.join(run_dir, "hyperparameters.json"), "r", encoding="utf-8") as f:
        hp = undot(json.load(f))
    model = EMGModel(hp).to(device)
    ckpt = torch.load(os.path.join(run_dir, "best_model_state.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    native_classes = hp.get("model", {}).get("num_classes", None)
    print(f"[Model] Loaded. Native classes: {native_classes}, val_loss={ckpt.get('val_loss')}")
    return model, device

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def merge_probs_9_to_5(p9: np.ndarray) -> np.ndarray:
    out = np.zeros((p9.shape[0], 5), dtype=np.float32)
    out[:, 0] = p9[:, 0] + p9[:, 1] + p9[:, 5] + p9[:, 7]   # rest
    out[:, 1] = p9[:, 2]                                    # extension (full)
    out[:, 2] = p9[:, 3] + p9[:, 4]                         # flexion (mild+full)
    out[:, 3] = p9[:, 6]                                    # radial (full)
    out[:, 4] = p9[:, 8]                                    # ulnar  (full)
    return out

def predict_with_tau(P5: np.ndarray, tau=TAU_STAR, fallback_idx=IDX_REST) -> np.ndarray:
    meets = P5 >= tau.reshape(1, -1)
    any_pass = meets.any(axis=1)
    scores = P5.copy()
    scores[~meets] = -1.0
    yhat = scores.argmax(axis=1)
    yhat[~any_pass] = fallback_idx
    return yhat

# ---------- Probability smoothing ----------
class ProbSmoother:
    """
    Smooths class probabilities in time for nicer histogram motion.
    EMA with time-constant tau_s (seconds), optional confidence-adaptive alpha,
    and optional small median filter after EMA.
    """
    def __init__(self, K, hop_s=0.03, tau_s=0.20,
                 adapt=True, top_lo=0.35, top_hi=0.75,
                 k_median=1):
        import collections
        self.K = K
        self.hop_s = float(hop_s)
        self.tau_s = max(1e-3, float(tau_s))
        self.adapt = bool(adapt)
        self.top_lo = float(top_lo)
        self.top_hi = float(top_hi)
        self.alpha_base = 1.0 - np.exp(-self.hop_s / self.tau_s)
        self.P = None
        self.k_median = max(1, int(k_median))
        self.hist = [collections.deque(maxlen=self.k_median) for _ in range(K)]

    def _alpha_adaptive(self, top_prob):
        if not self.adapt:
            return self.alpha_base
        # Map top prob to multiplier: low conf -> more smoothing, high conf -> more responsive
        if top_prob <= self.top_lo:
            mult = 0.65
        elif top_prob >= self.top_hi:
            mult = 1.25
        else:
            r = (top_prob - self.top_lo) / max(1e-6, (self.top_hi - self.top_lo))
            mult = 0.65 + r * (1.25 - 0.65)
        alpha = self.alpha_base * mult
        return float(np.clip(alpha, 0.02, 0.50))

    def step(self, P_new: np.ndarray) -> np.ndarray:
        P_new = np.asarray(P_new, dtype=np.float32)
        P_new /= max(1e-9, P_new.sum())  # safety
        if self.P is None:
            self.P = P_new.copy()
            for k in range(self.K):
                self.hist[k].clear()
                self.hist[k].append(float(P_new[k]))
            return self.P.copy()

        alpha = self._alpha_adaptive(float(P_new.max()))
        self.P = (1.0 - alpha) * self.P + alpha * P_new

        if self.k_median > 1:
            for k in range(self.K):
                self.hist[k].append(float(self.P[k]))
                self.P[k] = np.median(self.hist[k])

        s = float(self.P.sum())
        if s > 0:
            self.P /= s
        return self.P.copy()

# ---------- Realtime Histogram UI ----------
class HistogramUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG → Model → Smoothed Probabilities (5 classes)")
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Probability')
        self.plot.setLabel('bottom', 'Class')
        self.plot.setYRange(0, 1.05)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        v.addWidget(self.plot)

        self.x_idx = np.arange(len(CLASS_NAMES))
        self.width = 0.7
        self.bg = pg.BarGraphItem(x=self.x_idx, height=np.zeros_like(self.x_idx, dtype=float),
                                  width=self.width)
        self.plot.addItem(self.bg)

        self.text_items = []
        for i, name in enumerate(CLASS_NAMES):
            ti = pg.TextItem(name, anchor=(0.5, 1.5), angle=0)
            self.plot.addItem(ti)
            ti.setPos(self.x_idx[i], 0.0)
            self.text_items.append(ti)

        self.pred_label = QtWidgets.QLabel("Prediction: —")
        self.pred_label.setAlignment(QtCore.Qt.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 22px; font-weight: 600; padding: 6px;")
        v.addWidget(self.pred_label)

    def update_hist(self, probs5: np.ndarray, pred_idx: int):
        h = probs5.astype(float)
        self.bg.setOpts(height=h)
        for i, ti in enumerate(self.text_items):
            ti.setText(f"{CLASS_NAMES[i]}\n{h[i]:.2f}")
            ti.setPos(self.x_idx[i], max(0.02, h[i]) + 0.03)
        self.pred_label.setText(f"Prediction: {CLASS_NAMES[pred_idx]}")

# ---------- Main app controller ----------
class AppController:
    def __init__(self):
        # threads & queues
        self.q = queue.Queue(maxsize=8192)
        self.stop_event = threading.Event()
        self.reader_thread = threading.Thread(target=serial_reader, args=(self.stop_event, self.q), daemon=True)

        # merger / windowing
        self.merger = StreamMerger(fs_agg=FS_AGG)
        self.win_samp = int(round(FS_AGG * WINDOW_MS / 1000.0))
        self.hop_samp = int(round(FS_AGG * HOP_MS / 1000.0))
        self._slide_buf = np.zeros((N_CH, 0), dtype=np.float32)

        # preprocessing
        self.pp = Preprocessor(FS_AGG, WINDOW_MS, ENV_MS)

        # model
        self.model, self.device = load_model(RUN_DIR)

        # smoother
        self.smoother = ProbSmoother(
            K=5,
            hop_s=HOP_MS/1000.0,
            tau_s=SMOOTH_TAU_S,
            adapt=SMOOTH_ADAPT,
            top_lo=SMOOTH_TOP_LO,
            top_hi=SMOOTH_TOP_HI,
            k_median=SMOOTH_MEDIAN_K,
        )

        # UI
        self.ui = HistogramUI()

        # timers
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)  # ~50 FPS UI timer

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_pitch_deg = 0.0
        self.last_imu_wall = 0.0



    def start(self):
        self.reader_thread.start()

    def stop(self):
        self.stop_event.set()
        try:
            self.reader_thread.join(timeout=0.5)
        except Exception:
            pass
        try: 
            self.sock.close()
        except Exception: 
            pass


    def _drain_queue(self, max_items=4000):
        drained = 0
        now = time.time()
        while drained < max_items and not self.q.empty():
            item = self.q.get_nowait()
            drained += 1
            tag = item[0]
            if tag == 'e':
                _, s, t_us, vals = item
                self.merger.push(s, t_us, vals)
            else:
                _, s, t_us, (qw,qx,qy,qz) = item
                # scale quaternion
                qw,qx,qy,qz = (qw/QUAT_SCALE, qx/QUAT_SCALE, qy/QUAT_SCALE, qz/QUAT_SCALE)
                roll, pitch, yaw = quat_to_rpy(qw,qx,qy,qz)
                self.last_pitch_deg = float(pitch)
                self.last_imu_wall = now


    def _forward(self, env_win_norm: np.ndarray) -> np.ndarray:
        """
        env_win_norm: (8,T) float32 → returns logits for native classes (9 or 5)
        """
        import torch
        x = torch.from_numpy(env_win_norm[None, ...]).to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(x).detach().cpu().numpy().astype(np.float64)  # (1,C)
        return logits[0]

    def tick(self):
        # 1) bring fresh serial data and merge
        self._drain_queue()
        self.merger.merge_some()

        # 2) pull latest and build sliding windows with hop
        latest = self.merger.pull_latest_block(self.win_samp)
        if latest is None:
            return

        if self._slide_buf.shape[1] == 0:
            self._slide_buf = latest.copy()
        else:
            append = latest[:, -max(0, self.hop_samp):]
            self._slide_buf = np.concatenate([self._slide_buf[:, self.hop_samp:], append], axis=1)

        if self._slide_buf.shape[1] < self.win_samp:
            return

        raw_win = self._slide_buf[:, -self.win_samp:]

        # 3) baseline warmup, then preprocess
        if not self.pp.have_baseline:
            self.pp.warmup_baseline(raw_win)
            # show zeros until baseline ready
            self.ui.update_hist(np.zeros(5, dtype=float), pred_idx=IDX_REST)
            return

        env_norm = self.pp.process(raw_win)  # (8,T)

        # 4) model forward
        logits_native = self._forward(env_norm)  # (C,)

        # 5) postprocess: T*, softmax, maybe 9→5
        P_native_cal = softmax_np(logits_native / T_STAR)
        Cnative = P_native_cal.shape[0]
        if Cnative == 9:
            P5_raw = merge_probs_9_to_5(P_native_cal[None, :])[0]
        elif Cnative == 5:
            P5_raw = P_native_cal.astype(np.float32)
        else:
            return

        # 6) smooth probabilities (for histogram + prediction stability)
        P5_smooth = self.smoother.step(P5_raw)

        # 7) τ* gating on the smoothed probabilities (reduces class hopping)
        y_tau = predict_with_tau(P5_smooth[None, :], TAU_STAR, IDX_REST)[0]

        # --- PITCH GATE: force REST if |pitch| > 25° and IMU is fresh ---
        gate_active = (time.time() - self.last_imu_wall) < IMU_FRESH_S
        if gate_active and abs(self.last_pitch_deg) > 25.0:
            y_tau = IDX_REST
            # (Optional) also force the histogram to show REST = 1.0 so UI matches the output
            P5_smooth = np.array([1,0,0,0,0], dtype=np.float32)

        # send out
        try:
            self.sock.sendto(f"{int(y_tau)}\n".encode("utf-8"), CLASS_SINK)
        except Exception:
            pass

        # UI update
        self.ui.update_hist(P5_smooth, int(y_tau))


# ---------- main ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = AppController()
    controller.ui.show()
    controller.start()

    def on_close():
        controller.stop()
        time.sleep(0.2)

    app.aboutToQuit.connect(on_close)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
