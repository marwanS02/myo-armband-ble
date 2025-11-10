# -*- coding: utf-8 -*-
"""
Real-time EMG inference visualizer (time-axis, big state/label text, live probability bars).

Usage (PowerShell):
(venv) PS ...\train> python simulate_rt_live.py `
  --samples_csv "C:\...\record\emg_sessions\val\20251004_142357\samples.csv" `
  --run_dir "C:\...\train\models\EMGModel\2025-10-16_08-58-00_with_loss_0.5539" `
  --backend Qt5Agg --fast 2.0
"""

import os, json, time, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch

# ---------------------------
# CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_csv", required=True, type=str)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--backend", default="Qt5Agg", type=str,
                   help="Matplotlib backend (Qt5Agg, TkAgg).")
    p.add_argument("--fs", type=int, default=200)
    p.add_argument("--window_ms", type=int, default=200)
    p.add_argument("--hop_ms", type=int, default=30)
    p.add_argument("--env_ms", type=int, default=50)
    p.add_argument("--bp_lo", type=float, default=20.0)
    p.add_argument("--bp_hi", type=float, default=90.0)
    p.add_argument("--bp_order", type=int, default=4)
    p.add_argument("--notch", type=float, default=30.0,
                   help="Notch Q at 50Hz (set 0 to disable)")
    p.add_argument("--fast", type=float, default=1.0,
                   help="Playback speed multiplier (1.0 = real-time)")
    p.add_argument("--skip_paused", action="store_true", default=True)
    return p.parse_args()

# ---------------------------
# Model loader
# ---------------------------
def undot(d):
    out = {}
    for k, v in d.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out

def load_model(run_dir, device):
    from networks import EMGModel  # your model class in repo
    with open(os.path.join(run_dir, "hyperparameters.json"), "r", encoding="utf-8") as f:
        hp = undot(json.load(f))
    model = EMGModel(hp).to(device)
    ckpt = torch.load(os.path.join(run_dir, "best_model_state.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model: {os.path.join(run_dir,'best_model_state.pth')} | val_loss: {ckpt.get('val_loss')}")
    return model, hp

# ---------------------------
# Preprocessing
# ---------------------------
def butter_bandpass(bplo, bphi, fs, order=4):
    return butter(order, [bplo/(fs/2.0), bphi/(fs/2.0)], btype='band')

def apply_iir(b, a, x):
    zi = lfilter_zi(b, a) * x[:, :1]
    y, _ = lfilter(b, a, x, axis=1, zi=zi)
    return y

def moving_rms(x, win_samp):
    if win_samp <= 1:
        return np.sqrt(x)
    pad = win_samp - 1
    x2 = x**2
    csum = np.cumsum(np.pad(x2, ((0,0),(pad,0))), axis=1)
    wsum = csum[:, win_samp:] - csum[:, :-win_samp]
    left = np.repeat(wsum[:, :1], pad, axis=1)
    return np.sqrt(np.concatenate([left, wsum], axis=1) / float(win_samp))

def preprocess_window(raw_win, fs, b_bp=None, a_bp=None, b_notch=None, a_notch=None, env_win=10):
    x = raw_win - raw_win.mean(axis=1, keepdims=True)  # DC
    if b_bp is not None:
        x = apply_iir(b_bp, a_bp, x)
    if (b_notch is not None) and (a_notch is not None):
        x = apply_iir(b_notch, a_notch, x)
    x = np.abs(x)
    env = moving_rms(x, env_win)
    return env.astype(np.float32)

def compute_baseline_rms_from_rest(df, data, fs, env_ms=50):
    s = df.copy()
    s["state"] = s["state"].astype(str).str.strip().str.upper()
    rest_mask = (s["state"] == "REST").values
    if not rest_mask.any():
        return np.ones(8, dtype=np.float32)

    segments = []
    idx = np.where(rest_mask)[0]
    start = None
    prev = None
    for i in idx:
        if start is None:
            start = i; prev = i
        elif i == prev + 1:
            prev = i
        else:
            if (prev - start + 1) >= fs*1.0:
                segments.append(data[start:prev+1].T)
            start = i; prev = i
    if start is not None and (prev - start + 1) >= fs*1.0:
        segments.append(data[start:prev+1].T)

    if not segments:
        return np.ones(8, dtype=np.float32)

    env_win = int(max(1, round(fs * (env_ms/1000.0))))
    env_all = [moving_rms(np.abs(seg), env_win) for seg in segments]
    big = np.concatenate(env_all, axis=1)
    base = np.median(big, axis=1)
    base[base == 0] = 1.0
    return base.astype(np.float32)

# ---------------------------
# Data helpers
# ---------------------------
def read_samples_csv(path):
    df = pd.read_csv(path, sep=';', engine='python', on_bad_lines='skip')
    needed = {"timestamp_us","state","active_label_id","active_label_name",
              "ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")
    return df

def class_names_9():
    return [
        "rest",
        "mild_extension", "full_extension",
        "mild_flexion", "full_flexion",
        "mild_radial_flexion", "full_radial_flexion",
        "mild_ulnar_flexion",  "full_ulnar_flexion",
    ]

# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()
    matplotlib.use(args.backend)
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, hp = load_model(args.run_dir, device)

    # read session
    df = read_samples_csv(args.samples_csv)
    df["state"] = df["state"].astype(str).str.strip().str.upper()

    ts_us = df["timestamp_us"].astype(np.int64).values
    t0_us = int(ts_us[0])
    data = df[["ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7"]].astype(np.float32).values
    y_true_ids = df["active_label_id"].astype(int).values
    y_true_names = df["active_label_name"].astype(str).values
    states = df["state"].values

    fs = int(args.fs)
    win_samp = int(round(fs * (args.window_ms/1000.0)))
    hop_samp = int(round(fs * (args.hop_ms/1000.0)))
    env_win  = int(max(1, round(fs * (args.env_ms/1000.0))))

    b_bp = a_bp = b_notch = a_notch = None
    try:
        b_bp, a_bp = butter_bandpass(args.bp_lo, args.bp_hi, fs, order=args.bp_order)
        if args.notch and args.notch > 0:
            b_notch, a_notch = iirnotch(50.0, args.notch, fs)
    except Exception as e:
        print("Filter design failed:", e)

    baseline = compute_baseline_rms_from_rest(df, data, fs, env_ms=args.env_ms)

    class_names = class_names_9()
    n_classes = len(class_names)

    # PLOTTING (2 rows): top scatter + class-streak shading, bottom bar (current probs)
    plt.ion()
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.3])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    # big text at top (state + true label)
    big_txt = fig.text(0.5, 0.965, "", ha="center", va="center", fontsize=18, fontweight="bold")

    # scatter setup
    ax0.set_title("Real-time predictions vs time (s)")
    ax0.set_xlabel("time (s)")
    ax0.set_ylabel("class id")
    ax0.set_yticks(range(n_classes))
    ax0.set_yticklabels(class_names, fontsize=9)
    ax0.grid(True, alpha=0.25)

    cmap = plt.get_cmap("tab10")
    cls_color = {i: cmap(i % 10) for i in range(n_classes)}

    scat_x, scat_y, scat_c, scat_txt = [], [], [], []
    scatter_plot = ax0.scatter([], [], s=14)

    # true-class streak shading
    active_patch = None
    streak_class = None
    streak_t0_s = None

    # current probability bar
    cur_bar = ax1.bar(range(n_classes), np.zeros(n_classes))
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels([s.replace("_"," ") for s in class_names], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("probability")
    ax1.set_title("Current probabilities")

    N = data.shape[0]
    last_t_shown = 0.0

    def update_plots(t_sec, pred_id, true_id, probs):
        nonlocal scatter_plot, active_patch, streak_class, streak_t0_s, last_t_shown

        correct = int(pred_id) == int(true_id)
        color = "g" if correct else "r"

        scat_x.append(t_sec)
        scat_y.append(pred_id)
        scat_c.append(color)
        scatter_plot.remove()
        scatter_plot = ax0.scatter(scat_x, scat_y, c=scat_c, s=14)

        # annotate misclass with predicted class name
        if not correct:
            txt = ax0.text(t_sec, pred_id + 0.15, class_names[pred_id].replace("_"," "),
                           fontsize=7, rotation=45, ha="left", va="bottom", color="r", alpha=0.85)
            scat_txt.append(txt)

        # TRUE-label streak shading
        if streak_class is None:
            streak_class = int(true_id)
            streak_t0_s = t_sec
            if active_patch is not None:
                active_patch.remove()
            active_patch = Rectangle((streak_t0_s, streak_class - 0.5),
                                     width=0.0001, height=1.0,
                                     linewidth=0, facecolor=cls_color[streak_class], alpha=0.12)
            ax0.add_patch(active_patch)
        else:
            if int(true_id) != streak_class:
                if active_patch is not None:
                    active_patch.set_width(t_sec - streak_t0_s)
                streak_class = int(true_id)
                streak_t0_s = t_sec
                active_patch = Rectangle((streak_t0_s, streak_class - 0.5),
                                         width=0.0001, height=1.0,
                                         linewidth=0, facecolor=cls_color[streak_class], alpha=0.12)
                ax0.add_patch(active_patch)
            else:
                if active_patch is not None:
                    active_patch.set_width(max(0.0001, t_sec - streak_t0_s))

        # update current probability bars
        for i, b in enumerate(cur_bar):
            b.set_height(float(probs[i]))
            b.set_color(cls_color[i])

        # rolling ~15s window
        if t_sec - last_t_shown > 1.0:
            ax0.set_xlim(max(0, t_sec - 15), t_sec + 2)
            last_t_shown = t_sec

    step = 0
    last_draw = time.time()
    t_draw_min_interval = 1.0/30.0  # 30 fps cap
    i = 0

    while i + win_samp <= N:
        j = i + win_samp
        # skip windows that include any PAUSED samples
        if args.skip_paused:
            if (states[i:j] == "PAUSED").any():
                i += hop_samp
                continue

        # x-axis time (seconds) at window end
        t_sec = (ts_us[j-1] - t0_us) / 1e6

        # raw window (C,T)
        raw_win = data[i:j].T

        # preprocess to envelope (C,T)
        env = preprocess_window(raw_win, fs, b_bp, a_bp, b_notch, a_notch, env_win=env_win)
        env = env / (baseline[:, None] + 1e-6)

        x = torch.from_numpy(env[None, ...]).to(device)  # (1,8,T)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        true_id = int(y_true_ids[j-1])

        # big text (state + true label)
        big_txt.set_text(f"STATE: {states[j-1]}   |   TRUE: {y_true_names[j-1]}")

        update_plots(t_sec, pred_id, true_id, probs)

        now = time.time()
        if now - last_draw >= t_draw_min_interval:
            plt.pause(0.001)
            last_draw = now

        # pacing like acquisition
        if args.fast > 0:
            dt = (args.hop_ms / 1000.0) / args.fast
            time.sleep(max(0.0, dt))

        i += hop_samp
        step += 1

    if streak_t0_s is not None and active_patch is not None and len(scat_x) > 0:
        active_patch.set_width(scat_x[-1] - streak_t0_s)

    plt.ioff()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
