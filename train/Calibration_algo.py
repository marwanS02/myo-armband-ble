# Calibration_algo_v5_fast.py
# Few-shot calibration with differentiable 9→5 training + fast T*/τ tuning.

import os, json, random
from collections import defaultdict
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ==== YOUR IMPORTS (adjust if needed) =================================
from data.EMG_dataset import EMGDataset
from networks import EMGModel

# ==== CONFIG ==========================================================
PRETRAIN_DIR   = r"C:\Users\Marwa\Documents\GitHub\myo-armband-ble\train\models\EMGModel\2025-10-16_08-58-00_with_loss_0.5539"
PARTICIPANT    = r"C:\Users\Marwa\Documents\GitHub\myo-armband-ble\record\emg_sessions\Sofiia"
COMPILED_ROOT  = r"compiled_data"
FILENAME       = "test_data.npz"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Few-shot per merged (5) class
N_CAL_PER_5CLASS = 150
BATCH_SIZE = 64

# Fine-tune phases
EPOCHS_HEAD = 5
LR_HEAD     = 5e-4

DO_PARTIAL_UNFREEZE = False
EPOCHS_PARTIAL      = 2
LR_PARTIAL          = 1e-4

# Inference post-processing defaults
CLASS_NAMES = ["rest","extension","flexion","radial_flexion","ulnar_flexion"]
IDX_REST = 0
TAU_STAR = np.array([0.688, 0.121, 0.715, 0.307, 0.344], dtype=np.float32)
T_STAR   = 0.929

# Fast tuner settings
DO_FAST_TUNE = True
FAST_TAU_QUANTILES = 51          # ~50 candidates per class
FAST_TAU_PASSES    = 3           # coordinate ascent passes

# ==== UTILITIES =======================================================
def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def undot(d):
    out = {}
    for k, v in d.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out

def load_pretrained_model(run_dir, device):
    with open(os.path.join(run_dir, "hyperparameters.json"), "r", encoding="utf-8") as f:
        hp = undot(json.load(f))
    model = EMGModel(hp).to(device)
    ckpt = torch.load(os.path.join(run_dir, "best_model_state.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded. val_loss={ckpt.get('val_loss')}")
    return model, hp

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def merge_probs_9_to_5_np(p9: np.ndarray) -> np.ndarray:
    out = np.zeros((p9.shape[0], 5), dtype=np.float32)
    out[:, 0] = p9[:, 0] + p9[:, 1] + p9[:, 5] + p9[:, 7]   # rest
    out[:, 1] = p9[:, 2]                                    # extension
    out[:, 2] = p9[:, 3] + p9[:, 4]                         # flexion
    out[:, 3] = p9[:, 6]                                    # radial
    out[:, 4] = p9[:, 8]                                    # ulnar
    return out

# 9→5 mapping (GT)
_MAP_9_TO_5 = {0:0, 1:0, 5:0, 7:0, 2:1, 3:2, 4:2, 6:3, 8:4}
def y9_to_y5(y_true_9):
    return np.array([_MAP_9_TO_5.get(int(y), 0) for y in y_true_9], dtype=np.int64)

# 9→5 merge matrix (probability domain), shape (9,5)
def merge_matrix_9x5_torch(device):
    M = torch.zeros(9, 5, device=device)
    M[0,0] = 1.0; M[1,0] = 1.0; M[5,0] = 1.0; M[7,0] = 1.0   # rest column
    M[2,1] = 1.0                                            # extension
    M[3,2] = 1.0; M[4,2] = 1.0                              # flexion
    M[6,3] = 1.0                                            # radial
    M[8,4] = 1.0                                            # ulnar
    return M  # P5 = P9 @ M

def predict_with_tau(P5: np.ndarray, tau: np.ndarray, fallback_idx: int) -> np.ndarray:
    meets = P5 >= tau.reshape(1, -1)
    any_pass = meets.any(axis=1)
    scores = P5.copy()
    scores[~meets] = -1.0
    yhat = scores.argmax(axis=1)
    yhat[~any_pass] = fallback_idx
    return yhat

def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    assert x.dim() == 3, f"Expected (B,C,T), got {tuple(x.shape)}"
    return model(x)

@torch.no_grad()
def evaluate(model, dataloader, device, temp=T_STAR, tau=TAU_STAR):
    y_true_5, y_pred_5 = [], []
    model.eval()
    for batch in dataloader:
        x = batch['signal'].to(device).float()
        if x.dim() == 4 and x.shape[1] == 1: x = x.squeeze(1)

        logits = forward_logits(model, x)  # (B,Cnative)
        P_native = softmax_np(logits.cpu().numpy() / float(temp))
        P5 = merge_probs_9_to_5_np(P_native) if P_native.shape[1]==9 else P_native.astype(np.float32)

        yhat_5 = predict_with_tau(P5, tau, IDX_REST)
        y_true_9 = batch['label'].cpu().numpy()
        y_true_5.extend(y9_to_y5(y_true_9).tolist())
        y_pred_5.extend(yhat_5.tolist())

    cm = confusion_matrix(y_true_5, y_pred_5, labels=list(range(5)))
    acc = accuracy_score(y_true_5, y_pred_5)
    return cm, acc

def plot_cm(cm, title):
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=120)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format='d')
    ax.set_title(title); plt.tight_layout(); plt.show()

def freeze_all(mod: nn.Module, freeze=True):
    for p in mod.parameters():
        p.requires_grad = not (freeze)

def adapt_bn_stats(model, loader, device, num_batches=50):
    was_train = model.training
    model.train()
    it = iter(loader)
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                xb = next(it)['signal'].to(device).float()
            except StopIteration:
                break
            if xb.dim()==4 and xb.shape[1]==1: xb = xb.squeeze(1)
            _ = forward_logits(model, xb)
    if not was_train:
        model.eval()

def build_few_shot_indices_5class(labels9, n_per_5class):
    labels5 = y9_to_y5(labels9)
    by5 = defaultdict(list)
    for i, y5 in enumerate(labels5):
        by5[int(y5)].append(i)
    calib, val = [], []
    for c in range(5):
        idxs = by5[c]
        random.shuffle(idxs)
        take = min(n_per_5class, len(idxs)//2)
        calib.extend(idxs[:take]); val.extend(idxs[take:])
    return calib, val

def class_weights_from_indices_5class(ds, indices, device):
    ys = np.array([ds.labels[i] for i in indices], dtype=np.int64)
    y5 = y9_to_y5(ys)
    counts = np.bincount(y5, minlength=5).astype(np.float32)
    weights = (counts.sum() / np.maximum(1.0, counts))
    return torch.tensor(weights, device=device)

def sanity_checks(ds, model):
    s = ds[0]['signal']
    assert isinstance(s, torch.Tensor)
    assert s.ndim == 2 and s.shape[0] == 8, f"Expected (C,T), got {tuple(s.shape)}"
    y = np.array(ds.labels, dtype=np.int64)
    assert y.min() >= 0 and y.max() <= 8
    xb = s.unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        _ = forward_logits(model, xb)

# ---------- FAST TUNER ------------------------------------------------
@torch.no_grad()
def cache_val_logits(model, val_loader):
    """Return (logits_native [N,C], y_true5 [N], Cnative)."""
    model.eval()
    logits_list, y5_list = [], []
    for batch in val_loader:
        x = batch['signal'].to(DEVICE).float()
        if x.dim()==4 and x.shape[1]==1: x = x.squeeze(1)
        y9 = batch['label'].cpu().numpy()
        logits = forward_logits(model, x).cpu()
        logits_list.append(logits)
        y5_list.append(y9_to_y5(y9))
    logits_all = torch.cat(logits_list, dim=0)            # (N, Cnative)
    y5_all     = np.concatenate(y5_list, axis=0)          # (N,)
    return logits_all, y5_all, logits_all.shape[1]

# --- replace the old version ---
def probs5_from_logits(logits_native: torch.Tensor,
                       T: torch.Tensor,
                       M: torch.Tensor | None = None) -> torch.Tensor:
    """
    Return P5 (N,5). T MUST be a tensor (keeps grad path).
    logits_native: (N, 9) or (N, 5)
    """
    if M is None:
        M = merge_matrix_9x5_torch(logits_native.device)  # shape (9,5)
    # keep T as tensor; no float(T)!
    # broadcast division works: (N,C) / scalar-tensor -> (N,C)
    scaled = logits_native / T
    p_nat = torch.softmax(scaled, dim=1)
    if p_nat.shape[1] == 9:
        return p_nat @ M
    return p_nat

# --- replace learn_temperature_lbfgs entirely ---
def learn_temperature_lbfgs(logits_native: torch.Tensor,
                            y_true5: np.ndarray,
                            T_init: float = T_STAR) -> float:
    """
    Optimize a single temperature T by minimizing NLL over the cached val logits.
    Keeps the computation graph through T (no float casts).
    """
    device = logits_native.device
    targets = torch.from_numpy(y_true5).long().to(device)
    logT = torch.tensor(np.log(max(1e-3, T_init)),
                        dtype=torch.float32, device=device, requires_grad=True)
    M = merge_matrix_9x5_torch(device)  # build once
    nll = nn.NLLLoss()

    optimizer = optim.LBFGS([logT], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        # keep T as a tensor; optionally keep it positive/stable
        T = torch.exp(logT)  # scalar tensor, grad-enabled
        P5 = probs5_from_logits(logits_native, T, M=M).clamp_min(1e-12)
        loss = nll(torch.log(P5), targets)
        loss.backward()  # grads flow to logT via T
        return loss

    optimizer.step(closure)
    T_opt = float(torch.exp(logT).clamp(1e-3, 10.0).item())
    return T_opt

def learn_temperature_adam(logits_native: torch.Tensor,
                           y_true5: np.ndarray,
                           T_init: float = T_STAR,
                           steps: int = 200, lr: float = 0.05) -> float:
    device = logits_native.device
    targets = torch.from_numpy(y_true5).long().to(device)
    logT = torch.tensor(np.log(max(1e-3, T_init)),
                        dtype=torch.float32, device=device, requires_grad=True)
    M = merge_matrix_9x5_torch(device)
    nll = nn.NLLLoss()
    opt = optim.Adam([logT], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        T = torch.exp(logT)
        P5 = probs5_from_logits(logits_native, T, M=M).clamp_min(1e-12)
        loss = nll(torch.log(P5), targets)
        loss.backward()
        opt.step()
    return float(torch.exp(logT).clamp(1e-3, 10.0).item())


def acc_with_tau(P5_np: np.ndarray, y_true5: np.ndarray, tau: np.ndarray) -> float:
    yhat = predict_with_tau(P5_np, tau, IDX_REST)
    return (yhat == y_true5).mean()

def tune_tau_coordinate_ascent(P5_np: np.ndarray, y_true5: np.ndarray,
                               tau_init: np.ndarray, n_passes=3, n_quant=51) -> np.ndarray:
    """Coordinate ascent over τ_k using quantile candidates from P5[:,k]."""
    tau = tau_init.astype(np.float32).copy()
    N, K = P5_np.shape
    for _ in range(n_passes):
        for k in range(K):
            candidates = np.quantile(P5_np[:, k], np.linspace(0.0, 0.999, n_quant))
            best_tau_k, best_acc = tau[k], -1.0
            for t in candidates:
                tau_try = tau.copy()
                tau_try[k] = float(np.clip(t, 0.01, 0.99))
                acc = acc_with_tau(P5_np, y_true5, tau_try)
                if acc > best_acc:
                    best_acc, best_tau_k = acc, tau_try[k]
            tau[k] = best_tau_k
    return tau

# ==== MAIN =============================================================
def main():
    set_seed(0)

    # 1) Build/compile dataset
    ds = EMGDataset(
        transform=None,
        root=COMPILED_ROOT,
        filename=FILENAME,
        participant_path=PARTICIPANT,
        force_compile=True,
    )
    print(f"[Data] Loaded {len(ds)} windows from {FILENAME}")
    labels_np = np.array(ds.labels, dtype=np.int64)

    # 2) Few-shot split in 5-class space
    calib_idx, val_idx = build_few_shot_indices_5class(labels_np, N_CAL_PER_5CLASS)
    calib_set, val_set = Subset(ds, calib_idx), Subset(ds, val_idx)
    calib_loader = DataLoader(calib_set, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 3) Load model + sanity
    model, hp = load_pretrained_model(PRETRAIN_DIR, DEVICE)
    sanity_checks(ds, model)
    print("[Sanity] Shapes/labels OK.")

    # 4) BEFORE calibration
    cm_before, acc_before = evaluate(model, val_loader, DEVICE, temp=T_STAR, tau=TAU_STAR)
    print(f"[Before] Acc = {acc_before*100:.2f}%")
    plot_cm(cm_before, "Confusion Matrix — Before Calibration")

    # 5) BN adaptation
    adapt_bn_stats(model, calib_loader, DEVICE, num_batches=50)

    # 6) Head-only fine-tune (differentiable 9→5)
    head = getattr(model, "classifier", None)
    assert head is not None, "Model must expose `classifier`."
    freeze_all(model, freeze=True); freeze_all(head, freeze=False)

    weights5 = class_weights_from_indices_5class(ds, calib_idx, DEVICE)
    criterion_nll = nn.NLLLoss(weight=weights5)
    optimizer = optim.Adam(head.parameters(), lr=LR_HEAD)
    M = merge_matrix_9x5_torch(DEVICE)

    model.train()
    for ep in range(1, EPOCHS_HEAD+1):
        running = 0.0; n = 0
        for batch in calib_loader:
            x = batch['signal'].to(DEVICE).float()
            y9 = batch['label'].to(DEVICE)
            if x.dim()==4 and x.shape[1]==1: x = x.squeeze(1)

            logits = forward_logits(model, x)
            if logits.shape[1] == 9:
                P9 = torch.softmax(logits, dim=1)
                P5 = P9 @ M
                logP5 = torch.log(P5.clamp_min(1e-9))
                y5 = torch.from_numpy(y9_to_y5(y9.detach().cpu().numpy())).to(DEVICE)
                loss = criterion_nll(logP5, y5)
            else:
                y5 = torch.from_numpy(y9_to_y5(y9.detach().cpu().numpy())).to(DEVICE)
                loss = nn.CrossEntropyLoss(weight=weights5)(logits, y5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()*x.size(0); n += x.size(0)
        print(f"[Head] Epoch {ep}/{EPOCHS_HEAD}  Loss={running/max(1,n):.4f}")

    # 7) Optional partial unfreeze
    if DO_PARTIAL_UNFREEZE:
        unfrozen = []
        for name in ["separableConv", "extra_stack", "time_adapt", "fc_norm", "classifier"]:
            if hasattr(model, name):
                for p in getattr(model, name).parameters():
                    p.requires_grad = True
                unfrozen.append(name)
        print(f"[Partial] Unfrozen: {unfrozen}")
        opt_partial = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_PARTIAL)
        model.train()
        for ep in range(1, EPOCHS_PARTIAL+1):
            running = 0.0; n = 0
            for batch in calib_loader:
                x = batch['signal'].to(DEVICE).float()
                y9 = batch['label'].to(DEVICE)
                if x.dim()==4 and x.shape[1]==1: x = x.squeeze(1)
                logits = forward_logits(model, x)
                if logits.shape[1] == 9:
                    P9 = torch.softmax(logits, dim=1)
                    P5 = P9 @ M
                    logP5 = torch.log(P5.clamp_min(1e-9))
                    y5 = torch.from_numpy(y9_to_y5(y9.detach().cpu().numpy())).to(DEVICE)
                    loss = criterion_nll(logP5, y5)
                else:
                    y5 = torch.from_numpy(y9_to_y5(y9.detach().cpu().numpy())).to(DEVICE)
                    loss = nn.CrossEntropyLoss(weight=weights5)(logits, y5)
                opt_partial.zero_grad(); loss.backward(); opt_partial.step()
                running += loss.item()*x.size(0); n += x.size(0)
            print(f"[Partial] Epoch {ep}/{EPOCHS_PARTIAL}  Loss={running/max(1,n):.4f}")

    # 8) AFTER calibration (no tuning)
    model.eval()
    cm_after, acc_after = evaluate(model, val_loader, DEVICE, temp=T_STAR, tau=TAU_STAR)
    print(f"[After (τ*,T* unchanged)] Acc = {acc_after*100:.2f}%")
    plot_cm(cm_after, "Confusion Matrix — After Calibration (τ*, T* unchanged)")

    # 9) FAST T*/τ tuning on cached logits
    if DO_FAST_TUNE:
        logits_val, y_true5, Cnative = cache_val_logits(model, val_loader)

        # learn T* via NLL (LBFGS)
        # T_opt = learn_temperature_lbfgs(logits_val, y_true5, T_init=T_STAR)
        T_opt = learn_temperature_lbfgs(logits_val, y_true5, T_init=T_STAR)
        # or: T_opt = learn_temperature_adam(logits_val, y_true5, T_init=T_STAR)


        # compute P5 with T_opt, then coordinate-ascent for τ
        with torch.no_grad():
            P5 = probs5_from_logits(logits_val, T_opt).cpu().numpy().astype(np.float32)

        tau0 = TAU_STAR.copy()
        tau_opt = tune_tau_coordinate_ascent(
            P5, y_true5, tau_init=tau0,
            n_passes=FAST_TAU_PASSES,
            n_quant=FAST_TAU_QUANTILES
        )

        # final accuracy with tuned T/τ
        yhat = predict_with_tau(P5, tau_opt, IDX_REST)
        acc_fast = (yhat == y_true5).mean()
        cm_fast = confusion_matrix(y_true5, yhat, labels=list(range(5)))
        print(f"[Fast Tune] T*={T_opt:.3f}  τ={np.round(tau_opt,3)}  Acc={acc_fast*100:.2f}%")
        plot_cm(cm_fast, "Confusion Matrix — After Calibration (fast T*/τ tuned)")

    # 10) Summary
    print(f"Accuracy Before : {acc_before*100:.2f}%")
    print(f"Accuracy After  : {acc_after*100:.2f}% (τ*,T* unchanged)")

if __name__ == "__main__":
    main()
