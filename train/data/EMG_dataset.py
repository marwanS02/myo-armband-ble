from .base_dataset import BaseDataset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor, RandomHorizontalFlip
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
import json, random



class EMGDataset(BaseDataset):
    def __init__(self, *args, transform = None, **kwargs): 
        super().__init__(*args,**kwargs)
        self.transform = transform
        self.data_file = os.path.join(self.root_path,self.dataset_name)
        self.prepare_data(self.data_file) # function to extract info from the file
                
        
    def prepare_data(self,file):
        #extract samples and list of names of each material 
        #(used as dict to map labels to the material they correspod to)
        data = np.load(file, allow_pickle=True)
        self.samples = data["X"]
        self.labels = data["y"]
        self.meta = json.loads(data["meta"].item())

        
    def __len__(self):
        return len(self.labels) 
        # how many samples we have based on how many labels
    
        
    def __getitem__(self,idx):
        #idx will be used for first dimension only since each sample
        #should include all info from the other dimensions
        sample = self.samples[idx,:,:]
        sample = torch.from_numpy(sample).float()
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return {'signal':sample.squeeze(),'label': label} #return a dict 
    
    def visualize(self,idx):
        sample = self.samples[idx,:,:]
        label_id = int(self.labels[idx])
        LABELS = self.meta["labels"]
        label_name = LABELS[str(label_id)] if isinstance(LABELS, dict) else LABELS[label_id]
        fs = self.meta["fs"]
        WIN_MS = self.meta["win_ms"]

        # --- Time axis ---
        T = sample.shape[1]
        t = np.arange(T) / fs

        # --- EMG strip styling ---
        # Use a fixed per-channel vertical scale so traces look like clinical strips.
        # 'offset' defines channel spacing; 'amp_step' sets the vertical scale unit (grid step).
        trace_peak = max(1e-6, float(np.max(np.abs(sample))))
        offset    = 1.6 * trace_peak         # channel spacing
        amp_step  = 0.2 * trace_peak         # vertical major grid step (~like 100 µV in a.u.)
        time_major = 0.10                     # 100 ms major grid
        time_minor = 0.02                     # 20 ms minor grid

        fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")

        # Black EMG traces with per-channel isoelectric baselines
        for ch in range(8):
            base = ch * offset
            y_shifted = sample[ch] + base
            ax.plot(t, y_shifted, color="black", lw=1.2)
            # baseline (isoelectric)
            ax.hlines(base, t[0], t[-1], colors="#bbbbbb", linestyles="--", linewidth=0.8)
            # inline channel label at t=0
            ax.text(t[0] - 0.005, base, f"ch{ch}", va='center', ha='right',
                    fontsize=9, color="black")

            # Major: 100 ms (x) × amp_step (y); Minor: 20 ms × amp_step/5

        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MultipleLocator(time_major))
        ax.xaxis.set_minor_locator(MultipleLocator(time_minor))
        ax.yaxis.set_major_locator(MultipleLocator(amp_step))
        ax.yaxis.set_minor_locator(MultipleLocator(amp_step/5))

        ax.grid(True, which='major', color='#cfcfcf', linewidth=0.9)
        ax.grid(True, which='minor', color='#e9e9e9', linestyle='--', linewidth=0.6)

        # Keep the same horizontal start point, hide y tick labels
        ax.set_yticklabels([])
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Electromyogram-style strip — Label {label_id}: {label_name.replace('_',' ').title()}")

        # --- Calibration bars (bottom-right corner) ---
        # 100 ms × 1 amp_step, placed where traces won't overlap.
        x_end = float(t[-1])
        x_cal_w = 0.10   # 100 ms bar
        y_bottom = -0.8 * offset       # below ch0 baseline to avoid overlapping signals
        x_start = x_end - 0.14         # inset from right edge

        # vertical (amplitude) bar
        ax.plot([x_start, x_start], [y_bottom, y_bottom + amp_step], color='black', lw=2)
        ax.text(x_start - 0.003, y_bottom + amp_step/2, f"{amp_step:.2f} a.u.",
                va='center', ha='right', fontsize=9)

        # horizontal (time) bar
        ax.plot([x_start, x_start + x_cal_w], [y_bottom, y_bottom], color='black', lw=2)
        ax.text(x_start + x_cal_w/2, y_bottom - 0.06*offset, "100 ms",
                va='top', ha='center', fontsize=9)

        # Tight layout and show
        plt.tight_layout()
        plt.show()

        print(f"Index: {idx}")
        print(f"Label ID: {label_id}")
        print(f"Label Name: {label_name}")
        print(f"Shape: {sample.shape} (C,T) | Duration: {WIN_MS/1000:.2f} s")

