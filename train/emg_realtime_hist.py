"""
EMG Real-Time → Model → Postprocess → Live Histogram (5 classes)
Author: Mohamad Marwan Sidani

- Reads tolerant UART lines: 'eK,<timestamp_us>,v0,...,v7' (K=0..3).
- Merges up to 4 streams in round-robin into a single 8-ch signal @ FS_AGG.
- Builds sliding windows, preprocesses to model-ready features.
- Inference with your checkpoint; temperature-scale (T*), merge 9→5, threshold with τ*.
- Shows a live histogram of class probabilities + the current predicted label.

Dependencies: pyqtgraph, PyQt5, numpy, torch, (optional) scipy
"""

import os, sys, time, json, threading, queue, math, warnings
from collections import deque, Counter

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ---------- CONFIG: I/O ----------
PORT = "COM5"       # Linux: '/dev/ttyACM0' or '/dev/ttyUSB0'; Windows: 'COMx'; macOS: '/dev/tty.usbmodem*'
BAUD = 115200
N_STREAMS = 4
N_CH = 8

# ---------- CONFIG: sampling / windows ----------
FS_AGG = 200.0              # aggregate sample rate after merging (Hz)
WINDOW_MS = 200             # window length (ms)
HOP_MS = 30                 # hop length (ms)
ENV_MS = 50                 # envelope RMS window (ms)
BASELINE_WARMUP_SEC = 3.0   # collect this much time to estimate baseline
DISPLAY_SEC = 5.0           # how much history to keep (not plotted here, but kept if you extend UI)

# ---------- CONFIG: model + calibration ----------
RUN_DIR = r"C:\Users\Marwa\Documents\GitHub\myo-armband-ble\train\models\EMGModel\2025-10-16_08-58-00_with_loss_0.5539"
USE_GPU = True

T_STAR = 0.929
CLASS_NAMES = ["rest","extension","flexion","radial_flexion","ulnar_flexion"]
IDX_REST = CLASS_NAMES.index("rest")
TAU_STAR = np.array([0.688, 0.121, 0.715, 0.307, 0.344], dtype=np.float32)

# ---------- CONFIG: filters (optional; script works without SciPy) ----------
USE_FILTERS = True          # if scipy is unavailable, this will auto-fallback to no filtering
BP_LO, BP_HI = 20.0, 90.0   # bandpass
BP_ORDER = 4
NOTCH_Q = 30.0              # 50 Hz notch; None to disable

# ---------- UART tolerant parser ----------
def parse_line(line: str):
    """
    Accept tolerant 'eK,<timestamp>,v0,...,v7'.
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

# ---------- Serial thread (pyserial) ----------
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

        # soft resync if imbalance grows
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
                    b_n, a_n = iirnotch(50.0/(fs/2.0), NOTCH_Q)
                    self.notch = (b_n, a_n)
                # init zi per channel
                self.zi_bp = [None]*N_CH
                self.zi_notch = [None]*N_CH
                # prepare zi using 1-sample prime per channel at zero
                if self.bp is not None:
                    self.zi_bp = [lfilter_zi(b_bp, a_bp) * 0.0 for _ in range(N_CH)]
                if self.notch is not None:
                    self.zi_notch = [lfilter_zi(b_n, a_n) * 0.0 for _ in range(N_CH)]
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
        x = raw_win - raw_win.mean(axis=1, keepdims=True)  # DC remove
        x = self.filters.apply(x)
        x = np.abs(x)
        env = moving_rms(x, self.env_win)
        return env

    def process(self, raw_win: np.ndarray) -> np.ndarray:
        """
        raw_win: (8, T) → returns (8, T) envelope normalized by baseline.
        """
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
    sys.path.insert(0, os.path.dirname(run_dir))  # ensure local imports if needed
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
    out[:, 0] = p9[:, 0] + p9[:, 1] + p9[:, 5] + p9[:, 7]   # rest bucket
    out[:, 1] = p9[:, 2]                                    # full extension
    out[:, 2] = p9[:, 3] + p9[:, 4]                         # flexion
    out[:, 3] = p9[:, 6]                                    # full radial
    out[:, 4] = p9[:, 8]                                    # full ulnar
    return out

def predict_with_tau(P5: np.ndarray, tau=TAU_STAR, fallback_idx=IDX_REST) -> np.ndarray:
    meets = P5 >= tau.reshape(1, -1)
    any_pass = meets.any(axis=1)
    scores = P5.copy()
    scores[~meets] = -1.0
    yhat = scores.argmax(axis=1)
    yhat[~any_pass] = fallback_idx
    return yhat

# ---------- Realtime Histogram UI ----------
class HistogramUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG → Model → Real-Time Histogram (5 classes)")
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Probability')
        self.plot.setLabel('bottom', 'Class')
        self.plot.setYRange(0, 1.05)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        v.addWidget(self.plot)

        # bar graph
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
        """
        probs5: (5,) array
        """
        h = probs5.astype(float)
        self.bg.setOpts(height=h)
        # move labels above bars with value
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
        self._since_last = 0

        # preprocessing
        self.pp = Preprocessor(FS_AGG, WINDOW_MS, ENV_MS)

        # model
        self.model, self.device = load_model(RUN_DIR)

        # UI
        self.ui = HistogramUI()

        # timers
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)  # ~50 FPS UI timer

    def start(self):
        self.reader_thread.start()

    def stop(self):
        self.stop_event.set()
        try:
            self.reader_thread.join(timeout=0.5)
        except Exception:
            pass

    def _drain_queue(self, max_items=4000):
        drained = 0
        while drained < max_items and not self.q.empty():
            try:
                s, t_us, vals = self.q.get_nowait()
                self.merger.push(s, t_us, vals)
                drained += 1
            except Exception:
                break

    def _forward(self, env_win_norm: np.ndarray) -> np.ndarray:
        """
        env_win_norm: (8,T) float32 → returns probs for native classes (9 or 5)
        """
        import torch
        x = torch.from_numpy(env_win_norm[None, ...])  # (1,8,T)
        x = x.to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(x).detach().cpu().numpy().astype(np.float64)  # (1,C)
        return logits[0]

    def tick(self):
        # 1) bring fresh serial data and merge
        self._drain_queue()
        self.merger.merge_some()

        # 2) pull latest and build sliding windows
        need = self.win_samp
        latest = self.merger.pull_latest_block(need)
        if latest is None:
            return

        # sliding hop logic
        if self._slide_buf.shape[1] == 0:
            self._slide_buf = latest.copy()
            self._since_last = self.hop_samp
        else:
            # append only new samples
            append = latest[:, -max(0, self.hop_samp):]
            self._slide_buf = np.concatenate([self._slide_buf[:, self.hop_samp:], append], axis=1)

        if self._slide_buf.shape[1] < self.win_samp:
            return

        raw_win = self._slide_buf[:, -self.win_samp:]

        # 3) baseline warmup, then preprocess
        if not self.pp.have_baseline:
            self.pp.warmup_baseline(raw_win)
            # Show zeros until baseline ready
            self.ui.update_hist(np.zeros(5, dtype=float), pred_idx=IDX_REST)
            return

        env_norm = self.pp.process(raw_win)  # (8,T)

        # 4) model forward
        logits_native = self._forward(env_norm)  # (C,)

        # 5) postprocess: T*, softmax, maybe 9→5, τ*
        P_native_cal = softmax_np(logits_native / T_STAR)
        Cnative = P_native_cal.shape[0]
        if Cnative == 9:
            P5 = merge_probs_9_to_5(P_native_cal[None, :])[0]
        elif Cnative == 5:
            P5 = P_native_cal.astype(np.float32)
        else:
            # unknown model output; do nothing
            return

        y_tau = predict_with_tau(P5[None, :], TAU_STAR, IDX_REST)[0]
        self.ui.update_hist(P5, int(y_tau))

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
