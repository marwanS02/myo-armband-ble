# realtime_autocal.py
# EMG realtime with one-shot per-user calibration at startup.
# Baseline → guided 5-class capture → quick head fine-tune → fast T*/τ tune → run.

import os, sys, time, json, threading, queue, warnings
from collections import deque
import numpy as np
from dataclasses import dataclass

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import socket
import math

# ---------------- IMU ----------------
QUAT_SCALE = 16384.0
IMU_FRESH_S = 0.5

def quat_to_rpy(qw,qx,qy,qz):
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz) or 1.0
    qw,qx,qy,qz = qw/n, qx/n, qy/n, qz/n
    siny_cosp = 2*(qw*qz + qx*qy);  cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    sinp = 2*(qw*qy - qz*qx);       pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))
    sinr_cosp = 2*(qw*qx + qy*qz);  cosr_cosp = 1 - 2*(qx*qx + qz*qz)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    return roll, pitch, yaw

CLASS_SINK = ("127.0.0.1", 6001)

# ---------------- I/O ----------------
PORT = "COM10"
BAUD = 115200
N_STREAMS = 4
N_CH = 8

# ---------------- Sampling/windows ----------------
FS_AGG = 200.0
WINDOW_MS = 200
HOP_MS = 30
ENV_MS = 50
BASELINE_WARMUP_SEC = 3.0
DISPLAY_SEC = 5.0

# ---------------- Model ----------------
RUN_DIR = r"C:\Users\Marwa\Documents\GitHub\myo-armband-ble\train\models\EMGModel\2025-10-16_08-58-00_with_loss_0.5539"
USE_GPU = True

CLASS_NAMES = ["rest","extension","flexion","radial_flexion","ulnar_flexion"]
IDX_REST = 0
# Defaults used until calibration finishes (then overwritten)
T_STAR = 0.929
TAU_STAR = np.array([0.688, 0.121, 0.715, 0.307, 0.344], dtype=np.float32)

# ---------------- Filters ----------------
USE_FILTERS = True
BP_LO, BP_HI = 20.0, 90.0
BP_ORDER = 4
NOTCH_Q = 30.0

# ---------------- Smoothing ----------------
SMOOTH_TAU_S = 0.20
SMOOTH_ADAPT = True
SMOOTH_TOP_LO = 0.35
SMOOTH_TOP_HI = 0.75
SMOOTH_MEDIAN_K = 1

# ---------------- Calibration recipe ----------------
# Total capture ~ short: baseline + 5 prompts
CALIB_BASELINE_SEC = 4.0                 # baseline stabilization before capture
CALIB_PER_CLASS_SEC = 3.0                # time to hold each gesture
CALIB_REST_BETWEEN_SEC = 1.0             # tiny neutral pause between classes
CALIB_ORDER = ["rest","extension","flexion","radial_flexion","ulnar_flexion"]

HEAD_EPOCHS = 4
HEAD_LR = 5e-4
BATCH_SIZE = 64

# Temperature tune (fast, differentiable)
TEMP_TUNE_STEPS = 150
TEMP_TUNE_LR = 0.05

# τ tuning (fast quantiles on val probs)
TAU_MIN = 0.05
TAU_MAX = 0.95
TAU_QUANTILES = np.linspace(0.20, 0.95, 10)  # per-class candidate thresholds

# ---------------- Serial parsing ----------------
def parse_line(line: str):
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
        if len(parts) < 12: return None
        try:
            t_us = int(parts[1])
            qw,qx,qy,qz = [int(parts[i]) for i in range(2,6)]
        except ValueError:
            return None
        return ('i', K, t_us, (qw,qx,qy,qz))
    return None

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
                    q.put_nowait(item)
            except queue.Full:
                try: _ = q.get_nowait()
                except Exception: pass
            except Exception:
                pass

# ---------------- Merging ----------------
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
        if len(self.y_ch[0]) < needed:
            return None
        X = np.stack([np.fromiter(self.y_ch[ch], dtype=np.float32)[-needed:] for ch in range(N_CH)], axis=0)
        return X

# ---------------- Preprocess ----------------
def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
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
                    b_n, a_n = iirnotch(50.0/(fs/2.0), NOTCH_Q)
                    self.notch = (b_n, a_n)
                self.zi_bp = [lfilter_zi(b_bp, a_bp) * 0.0 for _ in range(N_CH)] if self.bp else [None]*N_CH
                self.zi_notch = [lfilter_zi(*self.notch) * 0.0 for _ in range(N_CH)] if self.notch else [None]*N_CH
            except Exception as e:
                warnings.warn(f"[Filters] SciPy unavailable or failed ({e}); proceeding without filters.")
                self.have_scipy = False
                self.use = False
    def apply(self, x: np.ndarray) -> np.ndarray:
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
        x = raw_win - raw_win.mean(axis=1, keepdims=True)
        x = self.filters.apply(x)
        x = np.abs(x)
        env = moving_rms(x, self.env_win)
        return env
    def process(self, raw_win: np.ndarray) -> np.ndarray:
        env = self._compute_envelope(raw_win)
        env_norm = env / (self.baseline[:, None] + 1e-6)
        return env_norm.astype(np.float32)

# ---------------- Model & postproc ----------------
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
    from networks import EMGModel
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
    out[:, 0] = p9[:, 0] + p9[:, 1] + p9[:, 5] + p9[:, 7]
    out[:, 1] = p9[:, 2]
    out[:, 2] = p9[:, 3] + p9[:, 4]
    out[:, 3] = p9[:, 6]
    out[:, 4] = p9[:, 8]
    return out

def predict_with_tau(P5: np.ndarray, tau: np.ndarray, fallback_idx: int) -> np.ndarray:
    meets = P5 >= tau.reshape(1, -1)
    any_pass = meets.any(axis=1)
    scores = P5.copy()
    scores[~meets] = -1.0
    yhat = scores.argmax(axis=1)
    yhat[~any_pass] = fallback_idx
    return yhat

# ---------------- Smoother ----------------
class ProbSmoother:
    def __init__(self, K, hop_s=0.03, tau_s=0.20,
                 adapt=True, top_lo=0.35, top_hi=0.75, k_median=1):
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
        if not self.adapt: return self.alpha_base
        if top_prob <= self.top_lo: mult = 0.65
        elif top_prob >= self.top_hi: mult = 1.25
        else:
            r = (top_prob - self.top_lo) / max(1e-6, (self.top_hi - self.top_lo))
            mult = 0.65 + r * (1.25 - 0.65)
        alpha = self.alpha_base * mult
        return float(np.clip(alpha, 0.02, 0.50))
    def step(self, P_new: np.ndarray) -> np.ndarray:
        P_new = np.asarray(P_new, dtype=np.float32)
        P_new /= max(1e-9, P_new.sum())
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
        if s > 0: self.P /= s
        return self.P.copy()

# ---------------- UI ----------------
class HistogramUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG → Model → Smoothed Probabilities (5 classes)")
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)

        self.prompt = QtWidgets.QLabel("Calibrating…")
        self.prompt.setAlignment(QtCore.Qt.AlignCenter)
        self.prompt.setStyleSheet("font-size: 20px; padding: 6px;")
        v.addWidget(self.prompt)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Probability'); self.plot.setLabel('bottom', 'Class')
        self.plot.setYRange(0, 1.05); self.plot.showGrid(x=True, y=True, alpha=0.25)
        v.addWidget(self.plot)

        self.x_idx = np.arange(len(CLASS_NAMES))
        self.width = 0.7
        self.bg = pg.BarGraphItem(x=self.x_idx, height=np.zeros_like(self.x_idx, dtype=float), width=self.width)
        self.plot.addItem(self.bg)

        self.text_items = []
        for i, name in enumerate(CLASS_NAMES):
            ti = pg.TextItem(name, anchor=(0.5, 1.5), angle=0)
            self.plot.addItem(ti); ti.setPos(self.x_idx[i], 0.0); self.text_items.append(ti)

        self.pred_label = QtWidgets.QLabel("Prediction: —")
        self.pred_label.setAlignment(QtCore.Qt.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 22px; font-weight: 600; padding: 6px;")
        v.addWidget(self.pred_label)

    def set_prompt(self, text):
        self.prompt.setText(text)

    def update_hist(self, probs5: np.ndarray, pred_idx: int):
        h = probs5.astype(float)
        self.bg.setOpts(height=h)
        for i, ti in enumerate(self.text_items):
            ti.setText(f"{CLASS_NAMES[i]}\n{h[i]:.2f}")
            ti.setPos(self.x_idx[i], max(0.02, h[i]) + 0.03)
        self.pred_label.setText(f"Prediction: {CLASS_NAMES[pred_idx]}")

# ---------------- Calibration containers ----------------
@dataclass
class CalibBatch:
    X: list   # list of np.ndarray (8,T) envelope-normalized windows
    y: list   # list of int in 0..4 (merged class ids)

# ---------------- Controller ----------------
class AppController:
    def __init__(self):
        # threads & queues
        self.q = queue.Queue(maxsize=8192)
        self.stop_event = threading.Event()
        self.reader_thread = threading.Thread(target=serial_reader, args=(self.stop_event, self.q), daemon=True)

        # merger/windowing
        self.merger = StreamMerger(fs_agg=FS_AGG)
        self.win_samp = int(round(FS_AGG * WINDOW_MS / 1000.0))
        self.hop_samp = int(round(FS_AGG * HOP_MS / 1000.0))
        self._slide_buf = np.zeros((N_CH, 0), dtype=np.float32)

        # preprocessing
        self.pp = Preprocessor(FS_AGG, WINDOW_MS, ENV_MS)

        # model
        self.model, self.device = load_model(RUN_DIR)

        # runtime params (will be updated by calibration)
        self.T_star = float(T_STAR)
        self.tau_star = TAU_STAR.copy()

        self.calibrated = False
        self.collecting = False

        # smoother
        self.smoother = ProbSmoother(
            K=5, hop_s=HOP_MS/1000.0, tau_s=SMOOTH_TAU_S,
            adapt=SMOOTH_ADAPT, top_lo=SMOOTH_TOP_LO, top_hi=SMOOTH_TOP_HI, k_median=SMOOTH_MEDIAN_K,
        )

        # UI
        self.ui = HistogramUI()

        # timers
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_pitch_deg = 0.0
        self.last_imu_wall = 0.0

        # launch calibration thread
        self.calib_thread = threading.Thread(target=self.run_calibration, daemon=True)

    # -------- Threads control --------
    def start(self):
        self.reader_thread.start()
        self.calib_thread.start()

    def stop(self):
        self.stop_event.set()
        try: self.reader_thread.join(timeout=0.5)
        except Exception: pass
        try: self.sock.close()
        except Exception: pass

    # -------- Queue drain --------
    def _drain_queue(self, max_items=4000):
        drained = 0; now = time.time()
        while drained < max_items and not self.q.empty():
            item = self.q.get_nowait(); drained += 1
            tag = item[0]
            if tag == 'e':
                _, s, t_us, vals = item
                self.merger.push(s, t_us, vals)
            else:
                _, s, t_us, (qw,qx,qy,qz) = item
                qw,qx,qy,qz = (qw/QUAT_SCALE, qx/QUAT_SCALE, qy/QUAT_SCALE, qz/QUAT_SCALE)
                _, pitch, _ = quat_to_rpy(qw,qx,qy,qz)
                self.last_pitch_deg = float(pitch); self.last_imu_wall = now

    # -------- Model forward --------
    def _forward_logits(self, env_win_norm: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(env_win_norm[None, ...]).to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(x).detach().cpu().numpy().astype(np.float64)  # (1,C)
        return logits[0]

    # -------- Tick (UI + streaming) --------
    def tick(self):
        self._drain_queue(); self.merger.merge_some()
        latest = self.merger.pull_latest_block(self.win_samp)
        if latest is None: return

        if self._slide_buf.shape[1] == 0:
            self._slide_buf = latest.copy()
        else:
            append = latest[:, -max(0, self.hop_samp):]
            self._slide_buf = np.concatenate([self._slide_buf[:, self.hop_samp:], append], axis=1)
        if self._slide_buf.shape[1] < self.win_samp: return

        raw_win = self._slide_buf[:, -self.win_samp:]

        # Baseline warmup
        if not self.pp.have_baseline:
            self.pp.warmup_baseline(raw_win)
            self.ui.set_prompt("Calibrating… baseline")
            self.ui.update_hist(np.zeros(5, dtype=float), pred_idx=IDX_REST)
            return

        env_norm = self.pp.process(raw_win)  # (8,T)

        # During calibration, only update prompt/hist for feedback; don't send outputs
        if not self.calibrated:
            self.ui.update_hist(np.zeros(5, dtype=float), pred_idx=IDX_REST)
            return

        # Normal runtime
        logits_native = self._forward_logits(env_norm)
        P_native_cal = softmax_np(logits_native / self.T_star)
        Cnative = P_native_cal.shape[0]
        if Cnative == 9:
            P5_raw = merge_probs_9_to_5(P_native_cal[None, :])[0]
        elif Cnative == 5:
            P5_raw = P_native_cal.astype(np.float32)
        else:
            return

        P5_smooth = self.smoother.step(P5_raw)
        y_tau = predict_with_tau(P5_smooth[None, :], self.tau_star, IDX_REST)[0]

        # IMU pitch gate to REST
        gate_active = (time.time() - self.last_imu_wall) < IMU_FRESH_S
        if gate_active and abs(self.last_pitch_deg) > 25.0:
            y_tau = IDX_REST
            P5_smooth = np.array([1,0,0,0,0], dtype=np.float32)

        # send
        try:
            self.sock.sendto(f"{int(y_tau)}\n".encode("utf-8"), CLASS_SINK)
        except Exception:
            pass

        # UI
        self.ui.set_prompt("Ready")
        self.ui.update_hist(P5_smooth, int(y_tau))

    # -------- Calibration pipeline --------
    def run_calibration(self):
        """
        1) Wait for baseline to be ready
        2) Guided capture per class (CALIB_ORDER); store env windows & labels
        3) Head-only fine-tune (few epochs)
        4) Learn T* on a held-out split (Adam)
        5) Choose τ per class from validation P5 quantiles
        6) Set calibrated flag
        """
        # 1) wait baseline
        t0 = time.time()
        self.ui.set_prompt("Calibrating… baseline")
        while not self.pp.have_baseline and not self.stop_event.is_set():
            time.sleep(0.05)
        # small extra stabilization
        time.sleep(max(0.0, CALIB_BASELINE_SEC - (time.time()-t0)))

        # 2) guided capture
        calib = self._capture_guided()
        if len(calib.X) < 20:
            print("[Calibration] Not enough samples; using defaults.")
            self.calibrated = True
            self.ui.set_prompt("Ready (defaults)")
            return

        # build train/val split (per class 80/20)
        X = np.stack(calib.X, axis=0)  # (N,8,T)
        y = np.array(calib.y, dtype=np.int64)  # 0..4
        tr_idx, va_idx = self._split_per_class(y, train_frac=0.8)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # 3) head-only fine-tune
        self.ui.set_prompt("Calibrating… tuning head")
        self._finetune_head(Xtr, ytr)

        # 4) learn temperature on val logits
        self.ui.set_prompt("Calibrating… tuning T*")
        logits_val = self._batch_logits(Xva)  # (N, Cnative)
        T_opt = self._learn_temperature(logits_val, yva)
        self.T_star = float(T_opt)

        # 5) τ from quantiles on validation probs (with T*)
        self.ui.set_prompt("Calibrating… tuning τ")
        P5_val = self._to_P5_probs(logits_val, self.T_star)
        tau = self._choose_tau_by_quantiles(P5_val, yva)
        self.tau_star = tau

        # done
        self.calibrated = True
        self.ui.set_prompt("Ready")
        print(f"[Calibration] Done. T*={self.T_star:.3f}  τ={np.round(self.tau_star,3)}")

    def _capture_guided(self) -> CalibBatch:
        """
        Displays prompts and records (8,T) windows while user holds each pose.
        Uses current preprocessor and sliding window.
        """
        batch = CalibBatch(X=[], y=[])
        label_to_id = {n:i for i,n in enumerate(CLASS_NAMES)}
        # allow stream to progress; we sample at tick cadence
        per_class_deadline = CALIB_PER_CLASS_SEC
        pause_between = CALIB_REST_BETWEEN_SEC

        for name in CALIB_ORDER:
            cls = label_to_id[name]
            self.ui.set_prompt(f"Hold: {name} ({CALIB_PER_CLASS_SEC:.0f}s)")
            t_start = time.time()
            while (time.time() - t_start) < per_class_deadline and not self.stop_event.is_set():
                # attempt to harvest a window if baseline ready and we have enough samples
                latest = self.merger.pull_latest_block(self.win_samp)
                if latest is not None and self.pp.have_baseline:
                    raw_win = latest[:, -self.win_samp:]
                    env_norm = self.pp.process(raw_win)
                    batch.X.append(env_norm)
                    batch.y.append(cls)
                time.sleep(self.hop_samp / FS_AGG)  # approx hop cadence
            # small neutral pause
            self.ui.set_prompt("Relax…")
            time.sleep(pause_between)
        return batch

    def _split_per_class(self, y, train_frac=0.8):
        tr_idx, va_idx = [], []
        for c in range(5):
            idxs = np.where(y == c)[0]
            if len(idxs) == 0: continue
            np.random.shuffle(idxs)
            k = int(round(train_frac * len(idxs)))
            tr_idx.extend(idxs[:k]); va_idx.extend(idxs[k:])
        return np.array(tr_idx, dtype=np.int64), np.array(va_idx, dtype=np.int64)

    def _finetune_head(self, Xtr, ytr):
        import torch
        from torch import nn, optim
        self.model.eval()
        # freeze all, unfreeze classifier
        for p in self.model.parameters(): p.requires_grad = False
        head = getattr(self.model, "classifier", None)
        assert head is not None, "Model must expose `classifier`"
        for p in head.parameters(): p.requires_grad = True

        ds_n = Xtr.shape[0]
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(head.parameters(), lr=HEAD_LR)

        steps_per_epoch = max(1, int(np.ceil(ds_n / BATCH_SIZE)))
        for ep in range(HEAD_EPOCHS):
            running = 0.0; n = 0
            perm = np.random.permutation(ds_n)
            for b in range(steps_per_epoch):
                sl = perm[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                xb = torch.from_numpy(Xtr[sl]).float().to(self.device)   # (B,8,T)
                yb = torch.from_numpy(ytr[sl]).long().to(self.device)    # (B,)
                opt.zero_grad()
                logits = self.model(xb)         # (B, Cnative=9 or 5); model expects (B,C,T)
                if logits.shape[1] == 9:
                    # train on merged 5 labels: NLL(log(P5))
                    P9 = torch.softmax(logits, dim=1)
                    M = self._merge_matrix_9x5_torch(self.device)
                    P5 = P9 @ M
                    loss = torch.nn.NLLLoss()(torch.log(P5.clamp_min(1e-9)), yb)
                else:
                    loss = crit(logits, yb)
                loss.backward(); opt.step()
                running += float(loss.item())*len(sl); n += len(sl)
            print(f"[Head] Epoch {ep+1}/{HEAD_EPOCHS}  Loss={running/max(1,n):.4f}")

        # keep eval mode
        self.model.eval()

    def _batch_logits(self, X):
        import torch
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], BATCH_SIZE):
                xb = torch.from_numpy(X[i:i+BATCH_SIZE]).float().to(self.device)
                lo = self.model(xb).detach().cpu().numpy()
                outs.append(lo)
        return np.concatenate(outs, axis=0)  # (N, Cnative)

    def _to_P5_probs(self, logits_native: np.ndarray, T: float) -> np.ndarray:
        Pnat = softmax_np(logits_native / float(T))
        if Pnat.shape[1] == 9:
            return merge_probs_9_to_5(Pnat)
        return Pnat.astype(np.float32)

    def _learn_temperature(self, logits_val: np.ndarray, y_true5: np.ndarray) -> float:
        """
        Optimize a single T on val set (fast Adam on NLL of merged P5).
        """
        import torch
        from torch import nn, optim
        device = self.device
        logits = torch.from_numpy(logits_val).float().to(device)
        targets = torch.from_numpy(y_true5.astype(np.int64)).to(device)
        logT = torch.tensor(np.log(max(1e-3, 0.9)), dtype=torch.float32, device=device, requires_grad=True)
        M = self._merge_matrix_9x5_torch(device)
        nll = nn.NLLLoss()
        opt = optim.Adam([logT], lr=TEMP_TUNE_LR)
        for _ in range(TEMP_TUNE_STEPS):
            opt.zero_grad()
            T = torch.exp(logT)
            scaled = logits / T
            if logits.shape[1] == 9:
                P9 = torch.softmax(scaled, dim=1)
                P5 = (P9 @ M).clamp_min(1e-12)
            else:
                P5 = torch.softmax(scaled, dim=1).clamp_min(1e-12)
            loss = nll(torch.log(P5), targets)
            loss.backward(); opt.step()
        T_opt = float(torch.exp(logT).clamp(1e-3, 10.0).item())
        return T_opt

    def _choose_tau_by_quantiles(self, P5_val: np.ndarray, y_true5: np.ndarray) -> np.ndarray:
        """
        For each class k, sweep quantile candidates of P5[:,k] as τ_k.
        Keep others at 0 (no min gate), but apply fallback-to-REST logic overall.
        Joint small grid for speed: coordinate ascent 1 pass.
        """
        tau = np.clip(np.median(P5_val, axis=0), TAU_MIN, TAU_MAX)  # init
        # one pass coordinate update
        for k in range(5):
            cand = np.clip(np.quantile(P5_val[:, k], TAU_QUANTILES), TAU_MIN, TAU_MAX)
            best_acc = -1.0; best = tau[k]
            for tk in cand:
                tau_try = tau.copy(); tau_try[k] = tk
                yhat = predict_with_tau(P5_val, tau_try, IDX_REST)
                acc = (yhat == y_true5).mean()
                if acc > best_acc:
                    best_acc = acc; best = tk
            tau[k] = best
        return tau

    @staticmethod
    def _merge_matrix_9x5_torch(device):
        import torch
        M = torch.zeros(9,5, device=device)
        M[0,0]=1; M[1,0]=1; M[5,0]=1; M[7,0]=1
        M[2,1]=1
        M[3,2]=1; M[4,2]=1
        M[6,3]=1
        M[8,4]=1
        return M

# ---------------- main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = AppController()
    controller.ui.show()
    controller.start()

    def on_close():
        controller.stop(); time.sleep(0.2)
    app.aboutToQuit.connect(on_close)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()