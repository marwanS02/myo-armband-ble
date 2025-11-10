import random
import torch
import copy
from copy import deepcopy
from tqdm import tqdm 
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np
import contextlib
from datetime import datetime
import time
import os
import json
import pickle
import shutil  # To copy files
from training_utils.file_handling import create_folders, save_losses, save_hyperparameters, load_hyperparameters
from training_utils.plot_handling import create_plot, update_plot
from pprint import pprint
from torch.utils.data import DataLoader

from contextlib import redirect_stdout
import math
import torch.nn as nn
import torch.optim as optim
import io
import contextlib
import psutil
import glob
"""
How to Load Everything that was Saved during the Training Session:

1. **Model State** (`model_state.pth`):
    - To load the model's state (weights and architecture) after training, use the following code:
    
    ```python
    model = YourModelClass(hparams)  # Replace YourModelClass with the class name of your model
    model.load_state_dict(torch.load("path_to/model_state.pth")['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    ```

2. **Optimizer State** (part of `model_state.pth`):
    - To restore the optimizer state (for resuming training from the saved point):
    
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])  # Ensure you use the same optimizer as used in training
    optimizer.load_state_dict(torch.load("path_to/model_state.pth")['optimizer_state_dict'])
    ```

3. **Full Model Object** (`model.pth`):
    - If you saved the entire model object (not just the state dict), you can load it like this:
    
    ```python
    model = torch.load("path_to/model.pth")
    model.eval()  # Set the model to evaluation mode
    ```

4. **Losses** (`training_losses.json`):
    - To load the saved training and validation losses for plotting or analysis, use the following code:
    
    ```python
    import json
    with open("path_to/training_losses.json", "r") as f:
        losses = json.load(f)
    
    train_losses = losses['train_losses']
    val_losses = losses['val_losses']
    # You can now plot or analyze these loss values.
    ```

5. **Training Log** (`training_log.txt`):
    - The full training log, including printed outputs, loss updates, and model architecture, is saved in this file.
    - You can open and read this file manually or programmatically:

    ```python
    with open("path_to/training_log.txt", "r") as log_file:
        print(log_file.read())
    ```

6. **Loss Plot** (`loss_plot.png`):
    - A PNG file containing the plot of the training and validation loss curves. You can open it directly as an image for visualization.

Directory Structure:
---------------------
After training, the files are organized into a directory structured like this:

`models/{model_class_name}/{timestamp_with_loss}`:
    - `model.pth`: Full saved model object.
    - `model_state.pth`: Saved model state dict and optimizer state.
    - `training_log.txt`: Log file with training progress and architecture.
    - `loss_plot.png`: Final loss plot after training.
    - `training_losses.json`: Training and validation loss values for plotting.
"""


def train_model(model, train_loader, val_loader, device, hparams, plot_losses=True):
    # --- unchanged setup ---
    model_class_dir, train_session_dir, logs_dir, plots_dir = create_folders(model)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_class_name = model.__class__.__name__
    log_filename = f"{logs_dir}/{model_class_name}_training_{now}.txt"
    start_time = time.time()

    # (A) OPTIONAL: Determinism seed (helps bit-exact resumes with RNG save)
    def _seed_all(s):
        import random as _r, numpy as _np, torch as _t
        _r.seed(s); _np.random.seed(s); _t.manual_seed(s); _t.cuda.manual_seed_all(s)
    if 'seed' in hparams:
        _seed_all(hparams['seed'])
        if hparams.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # tiny state for graceful stop (unchanged) ...
    _stop_requested = {"flag": False}
    def _mark_stop(): _stop_requested["flag"] = True
    def _should_stop(): return _stop_requested["flag"]

    import signal
    _old_handler = signal.getsignal(signal.SIGINT)
    def _handler(sig, frame):
        print("\n[GRACEFUL STOP] Ctrl-C detected — finishing current batch, validating, and saving…")
        _mark_stop()
    signal.signal(signal.SIGINT, _handler)

    try:
        with open(log_filename, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file):
            print("Hyperparameters:\n"); pprint(hparams)
            print("Model Architecture:\n", model)
            print(f"Training Start Time: {now}\n")

            # optimizer / loss (unchanged)
            if isinstance(hparams['optimization']['optimizer'], str):
                optimizer = eval(hparams['optimization']['optimizer'])(model.parameters(), hparams['optimization']['lr'])
            else:
                optimizer = hparams['optimization']['optimizer'](model.parameters(), hparams['optimization']['lr'])

            if isinstance(hparams['optimization']['criterion'], str):
                criterion = eval(hparams['optimization']['criterion'])()
            else:
                criterion = hparams['optimization']['criterion']()

            num_epochs = hparams['optimization']['epochs']

            # --- SCHEDULER (already in your version; keep it) ---
            sched_cfg = hparams.get('scheduler', None)
            scheduler, scheduler_step_mode, scheduler_monitor = None, 'epoch', 'val_loss'
            if sched_cfg is not None:
                sched_ctor = sched_cfg.get('class', sched_cfg.get('name', None))
                scheduler_step_mode = sched_cfg.get('step_mode', 'epoch')
                scheduler_monitor   = sched_cfg.get('monitor', 'val_loss')
                sched_kwargs = {k: v for k, v in sched_cfg.items() if k not in ('class','name','step_mode','monitor')}
                if isinstance(sched_ctor, str):
                    sched_ctor = eval(sched_ctor)
                # OneCycle convenience
                try:
                    name = getattr(sched_ctor, '__name__', str(sched_ctor))
                    if 'OneCycleLR' in name and 'total_steps' not in sched_kwargs and 'steps_per_epoch' not in sched_kwargs:
                        sched_kwargs.setdefault('steps_per_epoch', len(train_loader))
                        sched_kwargs.setdefault('epochs', num_epochs)
                except Exception: pass
                try:
                    scheduler = sched_ctor(optimizer, **sched_kwargs)
                    print(f"[SCHEDULER] Enabled: {scheduler.__class__.__name__}, step_mode={scheduler_step_mode}, monitor={scheduler_monitor}")
                    if len(sched_kwargs): print(f"[SCHEDULER] kwargs: {sched_kwargs}")
                except Exception as e:
                    print(f"[SCHEDULER] Failed to create scheduler: {e}")
                    scheduler = None

            # plotting (unchanged)
            if plot_losses:
                loss_fig, loss_ax, loss_line_train, loss_line_val = create_plot("Loss")
            train_losses, val_losses = [], []

            # --- EARLY STOP (add burn_in/cooldown minimal) ---
            es_cfg = hparams.get('early_stopping', None)
            if es_cfg is not None:
                monitor = es_cfg.get('monitor', 'val_loss')
                mode = es_cfg.get('mode', ('min' if 'loss' in monitor.lower() else 'max'))
                patience = int(es_cfg.get('patience', 10))
                min_delta = float(es_cfg.get('min_delta', 0.0))
                restore_best = bool(es_cfg.get('restore_best', True))
                burn_in  = int(es_cfg.get('burn_in', 0))
                cooldown = int(es_cfg.get('cooldown', 0))
                cool_left = 0

                if mode not in ('min', 'max'):
                    print(f"[EARLY STOP] Invalid mode '{mode}', defaulting to 'min'."); mode = 'min'
                if mode == 'min':
                    best_score = float('inf');  _improved = lambda curr, best: (curr < best - min_delta)
                else:
                    best_score = -float('inf'); _improved = lambda curr, best: (curr > best + min_delta)

                es_bad_epochs = 0
                es_best_snapshot = None
                es_best_epoch = -1
                print(f"[EARLY STOP] Enabled: monitor={monitor}, mode={mode}, patience={patience}, "
                      f"min_delta={min_delta}, burn_in={burn_in}, cooldown={cooldown}, restore_best={restore_best}")

            interrupted = False
            for epoch in range(num_epochs):
                try:
                    from math import isnan
                    with tqdm(train_loader, unit='batch') as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

                        # (B) GRADIENT CLIPPING: pass clip value into train()
                        grad_clip = hparams.get('optimization', {}).get('grad_clip', None)
                        avg_train_loss, train_acc, was_interrupted = train(
                            model, tepoch, device, criterion, optimizer, stop_flag=_should_stop, grad_clip=grad_clip
                        )

                        # batch-mode scheduler step
                        if scheduler is not None and scheduler_step_mode == 'batch':
                            try: scheduler.step()
                            except Exception as e: print(f"[SCHEDULER] batch step failed: {e}")

                    # validation (unchanged)
                    avg_val_loss, val_acc = val(model, val_loader, device, criterion)

                    # (C) LR LOGGING: per-epoch (lightweight)
                    try:
                        curr_lr = next(iter(optimizer.param_groups))['lr']
                        print(f"[LR] epoch={epoch+1} lr={curr_lr:.6g}")
                    except Exception: pass

                    print(f"Epoch [{epoch+1}/{num_epochs}] "
                          f"Train: loss {avg_train_loss:.4f}, acc {train_acc:.2f}% | "
                          f"Val: loss {avg_val_loss:.4f}, acc {val_acc:.2f}%")

                    train_losses.append(avg_train_loss)
                    val_losses.append(avg_val_loss)

                    if plot_losses:
                        clear_output(wait=True)
                        update_plot(epoch, train_losses, val_losses, loss_fig, loss_ax, loss_line_train, loss_line_val)

                    # epoch/plateau scheduler step (uses latest val metrics)
                    if scheduler is not None and scheduler_step_mode in ('epoch', 'plateau'):
                        try:
                            if scheduler_step_mode == 'plateau':
                                metric = avg_val_loss if scheduler_monitor == 'val_loss' else val_acc
                                scheduler.step(metric)
                            else:
                                scheduler.step()
                        except Exception as e:
                            print(f"[SCHEDULER] epoch step failed: {e}")

                    # --- EARLY STOP check (with burn-in/cooldown) ---
                    if es_cfg is not None:
                        curr_score = avg_val_loss if monitor == 'val_loss' else val_acc
                        if _improved(curr_score, best_score):
                            print(f"[EARLY STOP] Improvement at epoch {epoch+1}: {monitor} {best_score:.6f} -> {curr_score:.6f}")
                            best_score = curr_score
                            es_bad_epochs = 0
                            es_best_epoch = epoch
                            cool_left = cooldown  # reset cooldown
                            es_best_snapshot = {
                                'model_state_dict': deepcopy(model.state_dict()),
                                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                                'scheduler_state_dict': deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                                'val_loss': float(avg_val_loss),
                                'val_acc': float(val_acc),
                                'epoch': int(epoch),
                                'monitor_score': float(curr_score),
                            }
                            best_path = f"{train_session_dir}/best_model_state.pth"
                            torch.save({**es_best_snapshot, 'hparams': hparams}, best_path)
                            print(f"[EARLY STOP] Best snapshot saved at {best_path}")
                        else:
                            if epoch + 1 > burn_in:
                                if cool_left > 0:
                                    cool_left -= 1
                                    print(f"[EARLY STOP] No improvement (cooldown {cool_left} left).")
                                else:
                                    es_bad_epochs += 1
                                    print(f"[EARLY STOP] No improvement (bad_epochs={es_bad_epochs}/{patience}).")
                                    if es_bad_epochs > patience:
                                        print("[EARLY STOP] Patience exceeded — stopping after validation and saving…")
                                        interrupted = True
                                        break
                            else:
                                print(f"[EARLY STOP] Burn-in (epoch {epoch+1}/{burn_in}), not counting bad epoch.")

                    if was_interrupted or _should_stop():
                        interrupted = True
                        print("[GRACEFUL STOP] Breaking after validation and saving checkpoint…")
                        break

                except KeyboardInterrupt:
                    print("\n[GRACEFUL STOP] KeyboardInterrupt caught — validating and saving…")
                    avg_val_loss, val_acc = val(model, val_loader, device, criterion)
                    interrupted = True
                    if len(train_losses) == 0 or len(val_losses) == 0 or len(val_losses) < len(train_losses):
                        train_losses.append(avg_train_loss if 'avg_train_loss' in locals() else float('nan'))
                        val_losses.append(avg_val_loss)
                    break

            # Save plots & losses (unchanged)
            if plot_losses:
                png1 = f"{plots_dir}/{model_class_name}_training_{now}_loss_plot.png"
                loss_fig.savefig(png1); print(f"Final loss plot saved at {png1}")
                png2 = f"{train_session_dir}/loss_plot.png"
                loss_fig.savefig(png2); print(f"Final loss plot saved at {png2}")

            losses_filename = f"{train_session_dir}/training_losses.json"
            save_losses(train_losses, val_losses, losses_filename)

            # (D) Restore best (also scheduler) before final save
            if es_cfg is not None and es_best_snapshot is not None and restore_best:
                try:
                    model.load_state_dict(es_best_snapshot['model_state_dict'])
                    optimizer.load_state_dict(es_best_snapshot['optimizer_state_dict'])
                    if scheduler is not None and es_best_snapshot.get('scheduler_state_dict') is not None:
                        scheduler.load_state_dict(es_best_snapshot['scheduler_state_dict'])
                    print(f"[EARLY STOP] Restored best epoch {es_best_snapshot['epoch']+1}.")
                except Exception as e:
                    print(f"[EARLY STOP] Restore best failed (continuing with last weights): {e}")

            # Final save (prefer best val loss if ES used)
            if es_cfg is not None and es_best_snapshot is not None:
                final_val_loss = float(es_best_snapshot['val_loss'])
            else:
                final_val_loss = val_losses[-1] if len(val_losses) else float('nan')

            _save_checkpoint(model, optimizer, final_val_loss, hparams,
                             train_session_dir, model_class_name, plots_dir,
                             loss_fig if plot_losses else None,
                             scheduler=scheduler)  # <-- pass scheduler

            # Copy log, save hparams (unchanged aside from ES summary)
            log_copy_filename = f"{train_session_dir}/training_log.txt"
            shutil.copy(log_filename, log_copy_filename)
            print(f"Copied training log to {log_copy_filename}")

            hparams['val_loss'] = float(final_val_loss)
            if 'early_stopping' in hparams:
                hparams['early_stopping'] = {
                    **hparams['early_stopping'],
                    'best_epoch': int(es_best_epoch) if 'es_best_epoch' in locals() else -1,
                    'best_score': float(best_score) if 'best_score' in locals() else float('nan'),
                }
            hparams_filename = f"{train_session_dir}/hyperparameters.json"
            save_hyperparameters(hparams, hparams_filename)
            print(f"Hyperparameters saved to {hparams_filename}")

            print(f"Training {'interrupted' if interrupted else 'complete'}. "
                  f"Total time: {time.time() - start_time:.2f}s")

        new_train_session_dir = f"{model_class_dir}/{now}_with_loss_{float(final_val_loss):.4f}"
        os.rename(train_session_dir, new_train_session_dir)
        print(f"Renamed training session directory to {new_train_session_dir}")

    finally:
        try: signal.signal(signal.SIGINT, _old_handler)
        except Exception: pass



def train(model, train_loader, device, criterion, optimizer, stop_flag=None, grad_clip=None):
    """
    Minimal change: optional grad_clip (float). If provided, clip global norm after backward.
    """
    model.train()
    running_train_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        samples = batch['signal'].to(device)
        labels  = batch['label'].to(device).long()

        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, labels)
        loss.backward()

        # --- NEW: gradient clipping (if requested) ---
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        optimizer.step()

        running_train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        try:
            acc = 100.0 * correct / max(1, total)
            train_loader.set_postfix_str(f'loss: {loss.item():.4f} | acc: {acc:.2f}%')
        except AttributeError:
            pass

        if stop_flag is not None and stop_flag():
            return (running_train_loss / max(1, len(train_loader)),
                    100.0 * correct / max(1, total),
                    True)

    avg_loss = running_train_loss / max(1, len(train_loader))
    avg_acc  = 100.0 * correct / max(1, total)
    return avg_loss, avg_acc, False


@torch.no_grad()
def val(model, val_loader, device, criterion):
    running_val_loss = 0.0
    correct = 0
    total = 0
    model.eval()

    for batch in val_loader:
        samples = batch['signal'].to(device)        # [B, C, T]
        labels  = batch['label'].to(device).long()  # [B]

        outputs = model(samples)                    # [B, num_classes]
        loss = criterion(outputs, labels)
        running_val_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_val_loss / max(1, len(val_loader))
    avg_acc  = 100.0 * correct / max(1, total)
    return avg_loss, avg_acc

# --- NEW tiny helper: save a checkpoint quickly ---
def _save_checkpoint(model, optimizer, avg_val_loss, hparams, train_session_dir,
                     model_class_name, plots_dir, loss_fig=None, scheduler=None):
    # save state dict
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(avg_val_loss),
        'hparams': hparams
    }

    # (E) Save scheduler state if present
    if scheduler is not None:
        try:
            state['scheduler_state_dict'] = scheduler.state_dict()
        except Exception as e:
            print(f"[CKPT] Skipped scheduler state: {e}")

    # (F) Save RNG state for bit-exact resumes
    try:
        state['rng'] = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    except Exception as e:
        print(f"[CKPT] Skipped RNG state: {e}")

    model_state_filename = f"{train_session_dir}/model_state.pth"
    torch.save(state, model_state_filename)
    print(f"[GRACEFUL STOP] Model state dict saved at {model_state_filename}")

    # Full model (unchanged)
    try:
        model_filename = f"{train_session_dir}/model.pth"
        torch.save(model, model_filename)
        print(f"[GRACEFUL STOP] Full model saved at {model_filename}")
    except Exception as e:
        print(f"[GRACEFUL STOP] Skipped saving full model: {e}")

    # Loss plot if available (unchanged)
    if loss_fig is not None:
        plot_path = f"{train_session_dir}/loss_plot.png"
        try:
            loss_fig.savefig(plot_path)
            print(f"[GRACEFUL STOP] Loss plot saved at {plot_path}")
        except Exception as e:
            print(f"[GRACEFUL STOP] Could not save loss plot: {e}")