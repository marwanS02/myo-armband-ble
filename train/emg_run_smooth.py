"""
EMG Real-Time → Model → Postprocess → SMOOTHED Live Histogram (5 classes)
Author: Mohamad Marwan Sidani

Pipeline:
- UART tolerant reader (eK,<timestamp_us>,v0..v7), up to 4 streams.
- Round-robin merge → 8ch signal @ FS_AGG.
- Sliding windows → DC removal → (optional) stateful bandpass + 50Hz notch → rectification → moving RMS → baseline norm.
- Model inference → temperature scale (T*) → softmax → possibly 9→5 merge → τ* gate.
- Temporal smoothing: EMA(prob) → τ* → hysteresis (k_on/k_off/conf_on) + instability lockout.
- UI: live histogram of smoothed probs; label shows smoothed class.

Requires: pyserial, PyQt5, pyqtgraph, numpy, torch; (optional) scipy for filters
"""

import os, sys, time, json, threading, queue, warnings
from collections import deque

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# =========================
# CONFIG — I/O & MODEL
# =========================
PORT = "COM5"   # Linux: '/dev/ttyACM0' or '/dev/ttyUSB0'; Windows: 'COMx'; macOS: '/dev/tty.usbmodem*'
BAUD = 115200
N_STREAMS = 4
N_CH = 8

RUN_DIR = r"C:\Users\Marwa\Documents\GitHub\myo-armband-ble\train\models\EMGModel\2025-10-16_08-58-00_with_loss_0.5539"
USE_GPU = True

# =========================
# CONFIG — signal processing
# =========================
FS_AGG = 200.0      # Hz after merging
WINDOW_MS = 200     # ms
HOP_MS = 30         # ms
ENV_MS = 50         # ms
BASELINE_WARMUP_SEC = 3.0  # time used to estimate per-channel baseline

USE_FILTERS = True          # requires scipy; auto-fallback to no filters if missing
BP_LO, BP_HI = 20.0, 90.0   # Hz bandpass
BP_ORDER = 4
NOTCH_Q = 30.0              # 50 Hz notch Q (None to disable)

# =========================
# CONFIG — calibration & classes
# =========================
T_STAR = 0.929
CLASS_NAMES = ["rest","extension","flexion","radial_flexion","ulnar_flexion"]
IDX_REST = CLASS_NAMES.index("rest")
TAU_STAR = np.array([0.688, 0.121, 0.715, 0.307, 0.344], dtype=np.float32)

# =========================
# CONFIG — temporal smoothing
# =========================
EMA_ALPHA = 0.6       # 0..1 (higher = smoother)
EMA_SHARPEN = 1.25    # >1 gently sharpens peaks post-EMA
K_ON = 3              # frames needed to switch INTO a non-rest class
K_OFF = 4             # frames (rest) to switch BACK to rest
CONF_ON = 0.65        # avg prob during K_ON streak
SWITCH_WINDOW = 9     # window to count switches
MAX_SWITCHES = 2      # threshold of switches to trigger lockout
LOCKOUT_LEN = 6       # lockout duration in frames

# =========================
# UART tolerant parser
# =========================
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
                parsed = parse_line(line)
                if parsed:
                    q.put_nowait(parsed)
            except queue.Full:
                try: _ = q.get_nowait()
                except Exception: pass
            except Exception:
                pass

# =========================
# Merger (round-robin)
# =========================
class StreamMerger:
    def __init__(self, fs_agg=200.0):
        self.buffers = [deque() for _ in range(N_STREAMS)]
        self.expected = 0
        self.sample_index = 0
        self.fs = fs_agg
        self.skips = 0

    def push(self, s, t_us, vals): self.buffers[s].append((t_us, vals))

    def merge_some(self, max_merge=2000):
        merges = 0
        while merges < max_merge:
            if any(len(self.buffers[s]) == 0 for s in range(N_STREAMS)):
                break
            s = self.expected
            _t_us, vals = self.buffers[s].popleft()
            self.sample_index += 1
            self.expected = (self.expected + 1) % N_STREAMS
            merges += 1

        lens = [len(self.buffers[s]) for s in range(N_STREAMS)]
        if max(lens) - min(lens) > 50:
            self.expected = int(np.argmax(lens))
            self.skips += 1

    def pull_latest_block(self, store, needed):
        """
        Append merged samples into 'store' (8xT deque buffers).
        Return latest (8,needed) float32 if available, else None.
        """
        # As we don't store a separate merged ring here, we keep 'store' in controller.
        if store[0].__len__() < needed:
            return None
        X = np.stack([np.fromiter(store[ch], dtype=np.float32)[-needed:] for ch in range(N_CH)], axis=0)
        return X

# =========================
# Preprocess (stateful filters)
# =========================
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
        self.bp = None; self.notch = None
        self.zi_bp = None; self.zi_notch = None
        if self.use:
            try:
                from scipy.signal import butter, iirnotch, lfilter_zi
                self.have_scipy = True
                b_bp, a_bp = butter(BP_ORDER, [BP_LO/(fs/2.0), BP_HI/(fs/2.0)], btype='band')
                self.bp = (b_bp, a_bp)
                if NOTCH_Q is not None:
                    # normalized 50 Hz
                    w0 = 50.0/(fs/2.0)
                    b_n, a_n = iirnotch(w0, NOTCH_Q)
                    self.notch = (b_n, a_n)
                if self.bp is not None:
                    self.zi_bp = [lfilter_zi(*self.bp) * 0.0 for _ in range(N_CH)]
                if self.notch is not None:
                    self.zi_notch = [lfilter_zi(*self.notch) * 0.0 for _ in range(N_CH)]
            except Exception as e:
                warnings.warn(f"[Filters] SciPy unavailable/failed ({e}); using no filters.")
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
        self._warm_envs = []

    def _envelope(self, raw_win: np.ndarray) -> np.ndarray:
        x = raw_win - raw_win.mean(axis=1, keepdims=True)  # DC
        x = self.filters.apply(x)
        x = np.abs(x)
        env = moving_rms(x, self.env_win)
        return env

    def warmup(self, raw_win: np.ndarray):
        env = self._envelope(raw_win)
        self._warm_envs.append(env)
        # approx how many windows in warmup period:
        need = int(BASELINE_WARMUP_SEC * self.fs / max(1, self.win_samp//2))
        if len(self._warm_envs) >= max(6, need):
            big = np.concatenate(self._warm_envs, axis=1)
            base = np.median(big, axis=1)
            base[base == 0] = 1.0
            self.baseline = base.astype(np.float32)
            self.have_baseline = True
            self._warm_envs.clear()

    def process(self, raw_win: np.ndarray) -> np.ndarray:
        env = self._envelope(raw_win)
        return (env / (self.baseline[:, None] + 1e-6)).astype(np.float32)

# =========================
# Model & postprocess
# =========================
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

def predict_with_tau(P5: np.ndarray, tau=TAU_STAR, fallback_idx=IDX_REST) -> np.ndarray:
    meets = P5 >= tau.reshape(1, -1)
    any_pass = meets.any(axis=1)
    scores = P5.copy()
    scores[~meets] = -1.0
    yhat = scores.argmax(axis=1)
    yhat[~any_pass] = fallback_idx
    return yhat

# =========================
# Smoothing: EMA + Hysteresis + Lockout
# =========================
class ProbEMASmoother:
    def __init__(self, K=5, alpha=EMA_ALPHA, sharpen=EMA_SHARPEN):
        self.K = K
        self.alpha = float(alpha)
        self.sharpen = float(sharpen)
        self.state = np.ones(K, dtype=np.float32) / K

    def step(self, p):
        self.state = self.alpha * p + (1.0 - self.alpha) * self.state
        s = self.state.copy()
        if self.sharpen != 1.0:
            s = np.power(s, self.sharpen)
            s = s / (s.sum() + 1e-9)
        return s

class HysteresisGate:
    def __init__(self, idx_rest=0, k_on=K_ON, k_off=K_OFF, conf_on=CONF_ON,
                 switch_window=SWITCH_WINDOW, max_switches=MAX_SWITCHES, lockout_len=LOCKOUT_LEN):
        from collections import deque
        self.idx_rest = int(idx_rest)
        self.k_on = int(k_on); self.k_off = int(k_off)
        self.conf_on = float(conf_on)
        self.switch_window = int(switch_window)
        self.max_switches = int(max_switches)
        self.lockout_len = int(lockout_len)
        self.cur = self.idx_rest
        self.lockout = 0
        self.on_streak_cls = None
        self.on_streak_len = 0
        self.on_streak_conf_sum = 0.0
        self.hist_preds = deque(maxlen=max(self.k_off, self.switch_window))
        self.switch_hist = deque(maxlen=self.switch_window)

    def step(self, cls_proposed, conf_proposed):
        if self.lockout > 0:
            self.lockout -= 1
            self.cur = self.idx_rest
            self.hist_preds.append(self.cur)
            self.switch_hist.append(0)
            self.on_streak_cls = None; self.on_streak_len = 0; self.on_streak_conf_sum = 0.0
            return self.cur

        switched = int(cls_proposed != self.cur)
        self.switch_hist.append(switched)
        if sum(self.switch_hist) >= self.max_switches:
            self.lockout = self.lockout_len
            self.cur = self.idx_rest
            self.hist_preds.append(self.cur)
            self.on_streak_cls = None; self.on_streak_len = 0; self.on_streak_conf_sum = 0.0
            return self.cur

        if cls_proposed == self.idx_rest:
            self.hist_preds.append(self.idx_rest)
            if list(self.hist_preds).count(self.idx_rest) >= self.k_off:
                self.cur = self.idx_rest
            self.on_streak_cls = None; self.on_streak_len = 0; self.on_streak_conf_sum = 0.0
            return self.cur

        # proposing non-rest
        self.hist_preds.append(cls_proposed)
        if self.on_streak_cls != cls_proposed:
            self.on_streak_cls = cls_proposed
            self.on_streak_len = 0
            self.on_streak_conf_sum = 0.0

        self.on_streak_len += 1
        self.on_streak_conf_sum += float(conf_proposed)

        if (self.on_streak_len >= self.k_on and
            (self.on_streak_conf_sum / self.on_streak_len) >= self.conf_on):
            self.cur = cls_proposed
            self.on_streak_len = 0
            self.on_streak_conf_sum = 0.0

        return self.cur

class HybridSmoother:
    def __init__(self, tau_star, idx_rest=IDX_REST,
                 ema_alpha=EMA_ALPHA, ema_sharpen=EMA_SHARPEN,
                 k_on=K_ON, k_off=K_OFF, conf_on=CONF_ON,
                 switch_window=SWITCH_WINDOW, max_switches=MAX_SWITCHES, lockout_len=LOCKOUT_LEN):
        self.tau = np.asarray(tau_star, dtype=np.float32)
        self.idx_rest = int(idx_rest)
        self.ema = ProbEMASmoother(K=len(self.tau), alpha=ema_alpha, sharpen=ema_sharpen)
        self.gate = HysteresisGate(idx_rest=idx_rest, k_on=k_on, k_off=k_off, conf_on=conf_on,
                                   switch_window=switch_window, max_switches=max_switches, lockout_len=lockout_len)

    @staticmethod
    def _tau_gate(p5, tau, idx_rest):
        meets = p5 >= tau
        if not np.any(meets):
            return idx_rest
        cand = p5.copy()
        cand[~meets] = -1.0
        return int(np.argmax(cand))

    def step(self, P5_frame):
        P5s = self.ema.step(P5_frame)                           # smooth probs
        y_tau = self._tau_gate(P5s, self.tau, self.idx_rest)    # τ* on smoothed probs
        pred = self.gate.step(y_tau, float(P5s.max()))          # hysteresis/lockout
        return pred, P5s, y_tau

# =========================
# UI — live histogram
# =========================
class HistogramUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG → Model → SMOOTHED Real-Time Histogram (5 classes)")
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Probability')
        self.plot.setLabel('bottom', 'Class')
        self.plot.setYRange(0, 1.05)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        v.addWidget(self.plot)

        self.x_idx = np.arange(len(CLASS_NAMES))
        self.bg = pg.BarGraphItem(x=self.x_idx, height=np.zeros_like(self.x_idx, dtype=float), width=0.7)
        self.plot.addItem(self.bg)
        self.text_items = []
        for i, name in enumerate(CLASS_NAMES):
            ti = pg.TextItem(name, anchor=(0.5, 1.5))
            self.plot.addItem(ti)
            ti.setPos(self.x_idx[i], 0.02)
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

# =========================
# Controller
# =========================
def load_torch_model(run_dir):
    return load_model(run_dir)

class AppController:
    def __init__(self):
        # threads & queues
        self.q = queue.Queue(maxsize=8192)
        self.stop_event = threading.Event()
        self.reader = threading.Thread(target=serial_reader, args=(self.stop_event, self.q), daemon=True)

        # merger & sample store
        self.merger = StreamMerger(fs_agg=FS_AGG)
        self.store = [deque(maxlen=int(5*FS_AGG)*2) for _ in range(N_CH)]  # ring buffers per channel

        # windowing
        self.win_samp = int(round(FS_AGG * WINDOW_MS / 1000.0))
        self.hop_samp = int(round(FS_AGG * HOP_MS / 1000.0))
        self._slide_buf = np.zeros((N_CH, 0), dtype=np.float32)

        # preprocessing
        self.pp = Preprocessor(FS_AGG, WINDOW_MS, ENV_MS)

        # model
        self.model, self.device = load_torch_model(RUN_DIR)

        # smoother
        self.smoother = HybridSmoother(
            tau_star=TAU_STAR, idx_rest=IDX_REST,
            ema_alpha=EMA_ALPHA, ema_sharpen=EMA_SHARPEN,
            k_on=K_ON, k_off=K_OFF, conf_on=CONF_ON,
            switch_window=SWITCH_WINDOW, max_switches=MAX_SWITCHES, lockout_len=LOCKOUT_LEN
        )

        # UI
        self.ui = HistogramUI()

        # timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)  # ~50 fps

    def start(self): self.reader.start()
    def stop(self):
        self.stop_event.set()
        try: self.reader.join(timeout=0.5)
        except Exception: pass

    def _drain(self, max_items=4000):
        drained = 0
        while drained < max_items and not self.q.empty():
            try:
                s, t_us, vals = self.q.get_nowait()
                self.merger.push(s, t_us, vals)
                # also mirror into per-channel stores as merged sequence order
                # NOTE: we simulate "merged arrival" here by pushing raw values directly;
                # the merger primarily maintains RR fairness & backlog tracking.
                for ch in range(N_CH):
                    self.store[ch].append(vals[ch])
                drained += 1
            except Exception:
                break

    def _forward(self, env_win_norm: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(env_win_norm[None, ...])  # (1,8,T)
        x = x.to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(x).detach().cpu().numpy().astype(np.float64)  # (1,C)
        return logits[0]

    def tick(self):
        # 1) bring new data
        self._drain()
        self.merger.merge_some()

        # 2) build sliding window
        if len(self.store[0]) < self.win_samp:
            return
        latest = np.stack([np.fromiter(self.store[ch], dtype=np.float32)[-self.win_samp:] for ch in range(N_CH)], axis=0)

        if self._slide_buf.shape[1] == 0:
            self._slide_buf = latest.copy()
        else:
            # append hop-sized new segment
            append = np.stack([np.fromiter(self.store[ch], dtype=np.float32)[-self.hop_samp:] for ch in range(N_CH)], axis=0)
            self._slide_buf = np.concatenate([self._slide_buf[:, self.hop_samp:], append], axis=1)

        if self._slide_buf.shape[1] < self.win_samp:
            return

        raw_win = self._slide_buf[:, -self.win_samp:]

        # 3) baseline warmup or process
        if not self.pp.have_baseline:
            self.pp.warmup(raw_win)
            self.ui.update_hist(np.zeros(5, dtype=float), pred_idx=IDX_REST)
            return

        env_norm = self.pp.process(raw_win)  # (8,T)

        # 4) model → logits
        logits_native = self._forward(env_norm)  # (C,)
        P_native_cal = softmax_np(logits_native / T_STAR)
        Cnative = P_native_cal.shape[0]
        if Cnative == 9:
            P5 = merge_probs_9_to_5(P_native_cal[None, :])[0]
        elif Cnative == 5:
            P5 = P_native_cal.astype(np.float32)
        else:
            return

        # 5) smoothing
        pred_smooth, P5_smooth, _y_tau_raw = self.smoother.step(P5)

        # 6) UI
        self.ui.update_hist(P5_smooth, int(pred_smooth))

# =========================
# main
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    ctrl = AppController()
    ctrl.ui.show()
    ctrl.start()

    def on_close():
        ctrl.stop()
        time.sleep(0.2)

    app.aboutToQuit.connect(on_close)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
