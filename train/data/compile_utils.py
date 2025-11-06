import os
import shutil
import time
from io import StringIO
# Requires: numpy, pandas, scipy (for filters), tqdm (optional)
import os, glob, json, math, warnings
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

# REST noise filter thresholds

MARGIN_MS     = 250
REST_RMS_MULT = 1.5         # keep as is (envelope-based gate)
REST_AMP_MULT = 2.0        # NEW: per-channel max rectified vs per-channel median
# Hardcoded per-channel medians of MAX rectified amplitude for REST (label 0)
REST_MEDIAN_MAX = np.array(
    [3.3847, 8.1614, 14.3932, 10.2116, 4.0837, 3.2342, 3.1550, 2.9368],
    dtype=np.float32
)

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

"""
The shutil library in Python is a standard utility module that provides a range 
of high-level operations on files and collections of files.
"""

def compile_dataset(path, root, filename, processing_params, force_compile=False):
    """
    Ensures that the dataset is available in a clean directory, handles existing
    data appropriately, and provides flexibility to force re-compiles when 
    necessary.
    path: path to the participant data
    root: directory where to compile the data to
    """
    
    #check if the root directory exists or if the folder is empty.
    if not os.path.exists(root) or not os.listdir(root) or force_compile:
        #To compile from the path check if the directory exists and delete it to 
        # recreate it with compile_participant if directory doesn't exist start compile_participant
        target_path = os.path.join(root, filename)
        if os.path.isdir(target_path) and force_compile:
            shutil.rmtree(target_path)
        elif os.path.isfile(target_path) and force_compile:
            os.remove(target_path)

        os.makedirs(root, exist_ok=True)
        final_file_path = os.path.join(root, filename)
        if not os.path.exists(final_file_path):
            if processing_params is not None:
                fs=processing_params['fs']
                window_len=processing_params['window_len']
                hop_len=processing_params['hop_len']
                envelope_len=processing_params['envelope_len']
                f_low=processing_params['f_low']
                f_high=processing_params['f_high']
                filter_order=processing_params['filter_order']
                notch=processing_params['notch']
                _ = build_dataset(sessions_root=path, out_dir=root, filename=filename, 
                                fs=fs, window_len=window_len, hop_len=hop_len, envelope_len=envelope_len, 
                                f_low=f_low, f_high=f_high, filter_order=filter_order, notch=notch)
            else:
                _ = build_dataset(sessions_root=path, out_dir=root, filename=filename)


def build_dataset(sessions_root, out_dir, filename, fs=200, window_len=200, hop_len=30, envelope_len=50, 
                  f_low=20, f_high=90, filter_order=4, notch=30):
    sessions = sorted(glob.glob(os.path.join(sessions_root, "*")))
    if not sessions:
        raise RuntimeError(f"No sessions found under {sessions_root}")

    win_samp = int(round(fs * window_len / 1000.0)) # Compute the number of samples per window
    hop_samp = int(round(fs * hop_len / 1000.0)) # Compute how far to move the window forward between consecutive windows — in samples.
    env_win  = int(max(1, round(fs * envelope_len / 1000.0))) # Compute the smoothing window length for the RMS envelope (the moving average of squared EMG)

    # Filters
    b_bp = a_bp = b_notch = a_notch = None # creates “empty” filter variables, so even if the filter design fails later, the code won’t crash.
    try:
        b_bp, a_bp = butter_bandpass(f_low, f_high, fs, order=filter_order)
        if notch is not None:
            b_notch, a_notch = iir_notch_50(fs, notch)
    except Exception as e:
        warnings.warn(f"Filter design failed ({e}). Proceeding without filters.")
        b_bp = a_bp = b_notch = a_notch = None

    X_list, y_list, meta_list = [], [], []
    class_counts = {k:0 for k in LABELS.keys()}
    total_kept = 0

    print(f"[INFO] Searching sessions under: {sessions_root}")
    for sess in sessions:
        samples_p = os.path.join(sess, "samples.csv")
        events_p  = os.path.join(sess, "events.csv")
        if not os.path.exists(samples_p):
            print(f"[WARN] Missing samples.csv in {sess}, skipping.")
            continue

        # robust readers you added earlier
        samples = read_csv_smart(samples_p, sep=';')
        # events may be absent/messy; not needed anymore but keep for future
        #events  = read_csv_smart(events_p,  sep=';') if os.path.exists(events_p) else None
        samples["state"] = samples["state"].astype(str).str.strip().str.upper()

        # sanity check
        needed = {"timestamp_us","state","active_label_id","active_label_name","stream_index",
                  "ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7"}
        if not needed.issubset(set(samples.columns)):
            print(f"[WARN] Missing columns in {samples_p}, skipping.")
            continue

        # Raw arrays
        #t_us = samples["timestamp_us"].astype(np.int64).values  # not used for indexing anymore
        data = samples[["ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7"]].astype(np.float32).values

        # REST mask from samples
        rest_mask = (samples["state"] == "REST")

        # 4) Baseline from REST segments (allow shorter segments if needed)
        rest_segments = []
        for r0, r1 in contiguous_blocks(rest_mask.values):
            seg = data[r0:r1+1].T
            if seg.shape[1] >= int(fs * 1.0):  # was fs*2.0; 1s is more forgiving
                rest_segments.append(seg)
        baseline = compute_baseline_rms(rest_segments, env_win) if rest_segments else np.ones(8, dtype=np.float32)


        # 5) DO intervals (also use normalized "state" inside parse function)
        do_intervals = parse_do_intervals_from_samples(samples, fs, MARGIN_MS, window_len)
        print(f"[{Path(sess).name}] DO intervals: {len(do_intervals)}")


        # 6) Movement windows (unchanged)
        for lid, s0, s1 in do_intervals:
            for i0, i1, raw_win in window_iter_idx(data, s0, s1, win_samp, hop_samp):
                env_win_norm = preprocess_window(raw_win, fs, b_bp, a_bp, b_notch, a_notch, env_win, baseline)
                X_list.append(env_win_norm); y_list.append(int(lid))
                meta_list.append((Path(sess).name, int(i0), int(i1)))
                class_counts[lid] += 1; total_kept += 1

        #   Median over entire recording per channel (robust and simple)
        # 7) PRECOMPUTE per-channel median amplitude (HARDCODED from pre-analysis)
        median_ch = np.array([3.622, 7.514, 13.690, 9.933, 4.381, 3.770, 3.719, 3.233], dtype=np.float32)


        # 8) REST windows with amplitude gates (no spectral gate)
        for r0, r1 in contiguous_blocks(rest_mask.values):
            for i0, i1, raw_win in window_iter_idx(data, r0, r1, win_samp, hop_samp):
                # 1) (optional) keep your envelope/RMS gate if you still want it:
                env_win_norm = preprocess_window(raw_win, fs, b_bp, a_bp, b_notch, a_notch, env_win, baseline)
                w_rms  = np.sqrt(np.mean(env_win_norm**2, axis=1))
                thr_rms = REST_RMS_MULT * baseline
                if np.any(w_rms > thr_rms):
                    continue  # too “active” to be rest

                # 2) Channel-2 median gate (strict: mult=1.75)
                ok, mx, thr = rest_window_pass(raw_win, check_ch=2, mult=1.75, center_dc=True)
                if not ok:
                    continue  # drop this REST window

                # 3) Accept REST
                X_list.append(env_win_norm); y_list.append(0)
                meta_list.append((Path(sess).name, int(i0), int(i1)))
                class_counts[0] += 1; total_kept += 1



    if not X_list:
        raise RuntimeError("No windows collected. Check session names, states, or margins.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 8, T)
    y = np.array(y_list, dtype=np.int64)

    # Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_npz = os.path.join(out_dir, filename)
    meta = {
        "created": datetime.now().isoformat(),
        "sessions_root": sessions_root,
        "fs": fs, "win_ms": window_len, "hop_ms": hop_len, "env_ms": envelope_len,
        "bp": [f_low, f_high], "notch_q": notch,
        "labels": LABELS, "shape": {"N": int(X.shape[0]), "C": int(X.shape[1]), "T": int(X.shape[2])},
        "rest_filters": {"rms_mult": REST_RMS_MULT,
        "amp_mult": REST_AMP_MULT,},
        "margin_ms": MARGIN_MS,
        "index_space": "sample_indices",  # important: intervals are in indices, not wall-time
    }
    np.savez_compressed(out_npz, X=X, y=y, meta=json.dumps(meta))

    print(f"\nSaved: {out_npz}")
    print(f"X: {X.shape}  y: {y.shape}  (N, C, T)")
    print("Class counts:")
    for k in sorted(class_counts):
        print(f"  {k:>2} {LABELS[k]:<22}: {class_counts[k]}")
    return out_npz, X.shape, class_counts



# -------- Helpers --------

def butter_bandpass(bplo, bphi, fs, order=4):
    from scipy.signal import butter
    '''
    Digital filters require cutoff frequencies to be normalized to the Nyquist frequency, which is half the sampling rate
    Most digital filter design functions expect cutoff frequencies between 0 and 1, where 1 corresponds to the Nyquist frequency. 
    This is because digital filters operate in the discrete frequency domain.
    TODO: Continue here
    '''
    return butter(order, [bplo/(fs/2.0), bphi/(fs/2.0)], btype='band') 

def iir_notch_50(fs, q=30):
    from scipy.signal import iirnotch
    return iirnotch(50.0, q, fs) #TODO: understand this

def lfilter_zi(b, a, x):
    from scipy.signal import lfilter, lfilter_zi
    zi = lfilter_zi(b, a) * x[:, :1]
    y, zf = lfilter(b, a, x, axis=1, zi=zi) #TODO: understand this
    return y

def moving_rms(x, win_samp):
    # x: (C, T) non-negative after rectification; returns (C, T)
    if win_samp <= 1:
        return np.sqrt(x)
    # efficient cumulative moving average of squares
    pad = win_samp - 1
    x2 = x**2
    csum = np.cumsum(np.pad(x2, ((0,0),(pad,0))), axis=1)
    # window sums
    wsum = csum[:, win_samp:] - csum[:, :-win_samp]
    # pad left to original length
    left = np.repeat(wsum[:, :1], pad, axis=1)
    rms = np.sqrt(np.concatenate([left, wsum], axis=1) / float(win_samp)) 
    return rms #TODO: understand this

def compute_baseline_rms(rest_segments, env_win):
    """
    rest_segments: list of (C,T) arrays (preprocessed up to rectification, before RMS),
    returns per-channel baseline RMS after envelope → scalar per channel
    """
    if not rest_segments:
        return np.ones(8, dtype=np.float32)
    env_all = []
    for seg in rest_segments:
        env = moving_rms(np.abs(seg), env_win)
        env_all.append(env)
    big = np.concatenate(env_all, axis=1)  # (C, total_T)
    # robust baseline as median across time
    base = np.median(big, axis=1)
    base[base == 0] = 1.0
    return base.astype(np.float32)

def preprocess_window(raw_win, fs, b_bp=None, a_bp=None, b_notch=None, a_notch=None, env_win=10, baseline=None):
    """
    raw_win: (C,T) float array
    returns: (C,T) normalized envelope window
    """
    x = raw_win - raw_win.mean(axis=1, keepdims=True)  # DC
    if b_bp is not None:
        x = lfilter_zi(b_bp, a_bp, x)
    if b_notch is not None:
        x = lfilter_zi(b_notch, a_notch, x)
    x = np.abs(x)
    env = moving_rms(x, env_win)
    if baseline is not None:
        env = env / (baseline[:, None] + 1e-6)
    return env.astype(np.float32)


def spectral_ratio(window, fs):
    # window: (C,T) after any preprocessing step you wish to measure; here we use rectified raw
    X = np.abs(np.fft.rfft(window, axis=1))
    low = np.sum(X[:, (np.fft.rfftfreq(window.shape[1], 1/fs) < 15)], axis=1)
    mid = np.sum(X[:, (np.fft.rfftfreq(window.shape[1], 1/fs) >= 20) & (np.fft.rfftfreq(window.shape[1], 1/fs) < 90)], axis=1)
    ratio = np.mean((mid + 1e-9) / (low + 1e-9))
    return ratio

# -------- Load & build intervals from events --------
from io import StringIO

def read_csv_smart(path, sep=';', **kwargs):
    """
    Robust CSV reader:
    - tries utf-8 / utf-8-sig / cp1252 / latin1
    - falls back to utf-8 with 'replace' on decode
    """
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, engine='python',
                               on_bad_lines='skip', **kwargs)
        except UnicodeDecodeError:
            continue
    # Last resort: manual decode with replacement to avoid crash
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8', errors='replace')
    return pd.read_csv(StringIO(text), sep=sep, engine='python',
                       on_bad_lines='skip', **kwargs)


# --- PATCH: use sample-based intervals instead of events-based (fixes clock mismatch) ---

def contiguous_blocks(mask):
    """Yield (start_idx, end_idx) inclusive for contiguous True blocks in a boolean array."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            yield start, prev
            start = i
        prev = i
    yield start, prev

def parse_do_intervals_from_samples(samples_df, fs, margin_ms, window_len):
    """
    Build DO intervals from samples.csv only (no events.csv).
    We split by contiguous DO regions and by label_id within those regions.
    Returns a list of (label_id, start_idx, end_idx) in sample indices AFTER trimming margins.
    """
    if samples_df is None or samples_df.empty:
        return []
    s = samples_df.copy()
    s["state"] = s["state"].astype(str).str.strip().str.upper()
    is_do = (s["state"] == "DO") & (s["active_label_id"].between(1, 8))
    
    if not is_do.any():
        return []

    intervals = []
    # margin in samples
    m = int(round((margin_ms / 1000.0) * fs))

    # Walk contiguous DO blocks first
    for b0, b1 in contiguous_blocks(is_do.values):
        # Within a DO block, the label could (rarely) change if operator interrupted; split by label
        labels = s["active_label_id"].values[b0:b1+1]
        # Split whenever label changes
        seg_start = b0
        for i in range(b0+1, b1+1):
            if labels[i-b0] != labels[i-1-b0]:
                seg_end = i-1
                lid = int(s["active_label_id"].values[seg_start])
                # trim margins in samples
                s0 = seg_start + m
                s1 = seg_end   - m
                if s1 - s0 + 1 >= max(1, int(round(fs * window_len / 1000.0))):
                    intervals.append((lid, s0, s1))
                seg_start = i
        # last subsegment
        seg_end = b1
        lid = int(s["active_label_id"].values[seg_start])
        s0 = seg_start + m
        s1 = seg_end   - m
        if s1 - s0 + 1 >= max(1, int(round(fs * window_len / 1000.0))):
            intervals.append((lid, s0, s1))

    return intervals

def window_iter_idx(data, start_idx, end_idx, win_samp, hop_samp):
    """
    Iterate windows fully within [start_idx, end_idx] (inclusive), using sample indices.
    data: (N, 8) array
    yields (i0, i1, (8, T))
    """
    # Ensure inclusive range
    last_start = end_idx - win_samp + 1
    for i0 in range(start_idx, last_start + 1, hop_samp):
        i1 = i0 + win_samp
        yield i0, i1, data[i0:i1].T



def rest_window_pass(raw_win, check_ch=2, mult=1.0, center_dc=True):
    """
    Return (ok, max_rect, threshold) for a REST window based on channel `check_ch`.

    raw_win : (8, T) float array (one 200 ms window)
    check_ch: which channel to compare (0..7); default 2
    mult    : optional slack multiplier on the median threshold (1.0 = strict median)
    center_dc: if True, subtract per-channel mean before rectifying (recommended)

    ok is True if this window should be kept; False means drop it.
    """
    x = raw_win
    if center_dc:
        x = x - x.mean(axis=1, keepdims=True)     # remove DC before rectifying
    max_rect = float(np.max(np.abs(x[check_ch]))) # max rectified amplitude in channel
    thr = float(REST_MEDIAN_MAX[check_ch] * mult) # median-based threshold
    return (max_rect <= thr), max_rect, thr





