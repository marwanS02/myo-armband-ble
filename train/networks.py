import torch
import torch.nn as nn

class EMGModel(nn.Module):
    def __init__(self, hparams):
        """
        EMG binary classifier (Sigmoid output)

        Expected hparams:
            - 'channels' : number of EMG channels (8)
            - 'F1'        : number of temporal filters
            - 'D'         : depth multiplier for spatial conv
            - 'F2'        : separable filters
            - 'kernel_length': temporal kernel length
            - 'dropout'   : dropout rate
            - 'pool1_time': temporal pool for depthwise layer
            - 'pool2_time': temporal pool for separable layer
        """
        super().__init__()
        self.h = hparams

        # --- First temporal conv ---
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, hparams['F1'], kernel_size=(1, hparams['kernel_length']),
                      stride=1, padding=(0, hparams['kernel_length']//2), bias=False),
            nn.BatchNorm2d(hparams['F1'])
        )

        # --- Depthwise spatial filtering ---
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(hparams['F1'], hparams['F1'] * hparams['D'],
                      kernel_size=(hparams['channels'], 1),
                      groups=hparams['F1'], bias=False),
            nn.BatchNorm2d(hparams['F1'] * hparams['D']),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, hparams['pool1_time'])),
            nn.Dropout(hparams['dropout'])
        )

        # --- Separable conv ---
        self.separableConv = nn.Sequential(
            nn.Conv2d(hparams['F1'] * hparams['D'], hparams['F2'],
                      kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(hparams['F2']),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, hparams['pool2_time'])),
            nn.Dropout(hparams['dropout'])
        )

        # --- AFTER self.separableConv ---
        n_extra = self.h.get('extra_blocks', 2)          # how many extra layers
        k_extra = self.h.get('extra_kernel', 7)          # temporal kernel
        dil     = self.h.get('extra_dilation', 1)        # temporal dilation
        p_drop2 = self.h.get('extra_dropout', hparams['dropout'])

        blocks = [SepTemporalBlock(C=hparams['F2'], k=k_extra, dilation=dil, p_drop=p_drop2)
                for _ in range(int(n_extra))] if n_extra else []
        self.extra_stack = nn.Sequential(*blocks) if blocks else nn.Identity()


        # --- Adaptive pooling to fixed length ---
        self.time_adapt = nn.AdaptiveAvgPool2d((1, 8))  # (B, F2, 1, 8)
        self.fc_norm = nn.BatchNorm1d(hparams['F2'] * 8)
        num_classes = hparams['num_classes']   # e.g. 8
        self.classifier = nn.Linear(hparams['F2'] * 8, num_classes)


        self._initialize_weights()

    def forward(self, x):
        # Input shape: (B, C=8, T)
        x = x.unsqueeze(1)              # (B, 1, C, T)
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.extra_stack(x)
        x = self.time_adapt(x)          # (B, F2, 1, 8)
        x = x.flatten(1)                # (B, F2*8)
        x = self.fc_norm(x)
        logits = self.classifier(x)
        return logits    # output probability in [0,1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class SepTemporalBlock(nn.Module):
    """Residual depthwise–separable 1×k temporal block (keeps shape)."""
    def __init__(self, C=128, k=7, dilation=1, p_drop=0.0):
        super().__init__()
        pad = (0, (k // 2) * dilation)
        self.dw = nn.Conv2d(C, C, kernel_size=(1, k), padding=pad,
                            dilation=(1, dilation), groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(C)
        self.act = nn.ELU()
        self.do = nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.do(y)
        return x + y



import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularEMGModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.h = hparams

        # --- First temporal conv (temporal only, no channel padding needed) ---
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, hparams['F1'], kernel_size=(1, hparams['kernel_length']),
                      stride=1, padding=(0, hparams['kernel_length']//2), bias=False),
            nn.BatchNorm2d(hparams['F1'])
        )

        # ---- NEW: choose a local channel kernel & circular padding on channels ----
        # Use a small odd kernel across channels so "same" output height works cleanly.
        k_ch   = int(hparams.get('ch_kernel', 3))                  # e.g., 3 or 5 (must be odd)
        dil_ch = int(hparams.get('ch_dilation', 1))
        assert k_ch % 2 == 1, "Use an odd ch_kernel to keep output height equal to input."
        pad_ch = (dil_ch * (k_ch - 1)) // 2                        # 'same' padding for odd kernels

        # --- Depthwise spatial filtering with circular padding along channels only ---
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(hparams['F1'], hparams['F1'] * hparams['D'],
                      kernel_size=(k_ch, 1),
                      padding=(pad_ch, 0),               # pad only along channels (height)
                      dilation=(dil_ch, 1),
                      groups=hparams['F1'],
                      padding_mode='circular',           # <--- key line: circular along channels
                      bias=False),
            nn.BatchNorm2d(hparams['F1'] * hparams['D']),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, hparams['pool1_time'])),
            nn.Dropout(hparams['dropout'])
        )

        # --- Separable conv (temporal only) ---
        self.separableConv = nn.Sequential(
            nn.Conv2d(hparams['F1'] * hparams['D'], hparams['F2'],
                      kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(hparams['F2']),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, hparams['pool2_time'])),
            nn.Dropout(hparams['dropout'])
        )

        # --- Optional extra temporal blocks (no channel padding needed) ---
        n_extra = self.h.get('extra_blocks', 2)
        k_extra = self.h.get('extra_kernel', 7)
        dil     = self.h.get('extra_dilation', 1)
        p_drop2 = self.h.get('extra_dropout', hparams['dropout'])
        blocks = [CircularSepTemporalBlock(C=hparams['F2'], k=k_extra, dilation=dil, p_drop=p_drop2)
                  for _ in range(int(n_extra))] if n_extra else []
        self.extra_stack = nn.Sequential(*blocks) if blocks else nn.Identity()

        # --- Adaptive pooling to fixed length ---
        self.time_adapt = nn.AdaptiveAvgPool2d((1, 8))  # (B, F2, 1, 8)
        self.fc_norm = nn.BatchNorm1d(hparams['F2'] * 8)
        num_classes = hparams['num_classes']
        self.classifier = nn.Linear(hparams['F2'] * 8, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)              # (B, 1, C, T) -> channels on height
        x = self.firstConv(x)
        x = self.depthwiseConv(x)       # circular padding along channels happens here
        x = self.separableConv(x)
        x = self.extra_stack(x)
        x = self.time_adapt(x)          # (B, F2, 1, 8)
        x = x.flatten(1)                # (B, F2*8)
        x = self.fc_norm(x)
        logits = self.classifier(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); 
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

class CircularSepTemporalBlock(nn.Module):
    """Residual depthwise–separable 1×k temporal block (keeps shape)."""
    def __init__(self, C=128, k=7, dilation=1, p_drop=0.0):
        super().__init__()
        pad = (0, (k // 2) * dilation)
        self.dw = nn.Conv2d(C, C, kernel_size=(1, k), padding=pad,
                            dilation=(1, dilation), groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(C)
        self.act = nn.ELU()
        self.do = nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.dw(x); y = self.pw(y); y = self.bn(y); y = self.act(y); y = self.do(y)
        return x + y







import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- small helpers ---------
def _pick_gn_groups(num_channels: int) -> int:
    """
    Pick a sensible number of GN groups that divides num_channels.
    Tries 32,16,8,4,2,1 in that order.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


class ResidualSepTemporalBlock(nn.Module):
    """
    Residual depthwise–separable temporal block (1 x k) that preserves shape.
    Uses GroupNorm for small-batch stability.
    """
    def __init__(self, C: int, k: int = 7, dilation: int = 1, p_drop: float = 0.0):
        super().__init__()
        pad_t = (k // 2) * dilation  # 'same' along time
        self.dw = nn.Conv2d(C, C, kernel_size=(1, k),
                            padding=(0, pad_t), dilation=(1, dilation),
                            groups=C, bias=False)
        gn_groups = _pick_gn_groups(C)
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(gn_groups, C)
        self.act = nn.ELU()
        self.do  = nn.Dropout(p_drop) if (p_drop and p_drop > 0) else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.gn(y)
        y = self.act(y)
        y = self.do(y)
        return x + y


# --------- main model ---------
class CircularEMGModel2(nn.Module):
    """
    EMG multi-class classifier with circular padding across channels
    in the spatial (channel) conv, plus a stabilizing channel-mix bottleneck.

    Expected hparams:
        channels, F1, D, F2, kernel_length, dropout, pool1_time, pool2_time, num_classes
        # Circular spatial extras (defaults are safe):
        ch_kernel (odd int, default=3), ch_dilation (int, default=1)
        # Optional extra temporal residual blocks:
        extra_blocks (int, default=0), extra_kernel (int, default=7),
        extra_dilation (int, default=1), extra_dropout (float, default=dropout)
    """
    def __init__(self, hparams: dict):
        super().__init__()
        self.h = hparams
        C_in  = int(hparams['channels'])
        F1    = int(hparams['F1'])
        D     = int(hparams['D'])
        F2    = int(hparams['F2'])
        k_t   = int(hparams['kernel_length'])
        p1    = int(hparams['pool1_time'])
        p2    = int(hparams['pool2_time'])
        p_drop= float(hparams['dropout'])
        n_cls = int(hparams['num_classes'])

        # ---- First temporal conv (1 x k_t) ----
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, k_t),
                      stride=1, padding=(0, k_t // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ---- Circular depthwise spatial conv across channels ----
        k_ch   = int(hparams.get('ch_kernel', 3))       # use 3 first; try 5 later if needed
        dil_ch = int(hparams.get('ch_dilation', 1))
        assert k_ch % 2 == 1, "Use an odd ch_kernel (e.g., 3 or 5) to keep height (channels) unchanged."
        pad_ch = (dil_ch * (k_ch - 1)) // 2

        # Depthwise over F1 groups; output channels = F1 * D
        self.spatial_circ = nn.Conv2d(
            F1, F1 * D, kernel_size=(k_ch, 1),
            padding=(pad_ch, 0),
            dilation=(dil_ch, 1),
            groups=F1,
            padding_mode='circular',
            bias=False
        )
        gn_groups_spatial = _pick_gn_groups(F1 * D)
        self.spatial_block = nn.Sequential(
            self.spatial_circ,
            nn.GroupNorm(gn_groups_spatial, F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, p1)),
            nn.Dropout(p_drop),
        )

        # ---- Channel-mix bottleneck + collapse channel axis to 1 ----
        # This re-stabilizes the representation for the downstream temporal stack.
        self.channel_mix = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=1, bias=False),
            nn.GroupNorm(_pick_gn_groups(F1 * D), F1 * D),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, None))   # (B, F1*D, 1, T')
        )

        # ---- Separable temporal conv (1 x 16) ----
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, p2)),
            nn.Dropout(p_drop),
        )

        # ---- Optional extra temporal residual blocks ----
        n_extra = int(hparams.get('extra_blocks', 0))
        k_extra = int(hparams.get('extra_kernel', 7))
        d_extra = int(hparams.get('extra_dilation', 1))
        p_drop2 = float(hparams.get('extra_dropout', p_drop))

        if n_extra > 0:
            self.extra_stack = nn.Sequential(
                *[ResidualSepTemporalBlock(C=F2, k=k_extra, dilation=d_extra, p_drop=p_drop2)
                  for _ in range(n_extra)]
            )
        else:
            self.extra_stack = nn.Identity()

        # ---- Temporal squeeze + head ----
        self.time_adapt = nn.AdaptiveAvgPool2d((1, 8))  # -> (B, F2, 1, 8)
        self.fc_norm    = nn.BatchNorm1d(F2 * 8)
        self.classifier = nn.Linear(F2 * 8, n_cls)

        self._init_weights_center_tap(self.spatial_circ)  # identity-friendly init

    def forward(self, x):
        # x: (B, C, T)  -> treat channels as image height
        x = x.unsqueeze(1)              # (B, 1, C, T)
        x = self.firstConv(x)           # (B, F1, C, T)
        x = self.spatial_block(x)       # (B, F1*D, C, T')
        x = self.channel_mix(x)         # (B, F1*D, 1, T')
        x = self.separableConv(x)       # (B, F2, 1, T'')
        x = self.extra_stack(x)         # (B, F2, 1, T'')
        x = self.time_adapt(x)          # (B, F2, 1, 8)
        x = x.flatten(1)                # (B, F2*8)
        x = self.fc_norm(x)
        logits = self.classifier(x)
        return logits

    # --------- inits ---------
    def _init_weights_center_tap(self, circ_dw: nn.Conv2d):
        """
        Kaiming everywhere, but put a stronger center tap (=1.0) in the circular
        depthwise conv so it starts near an identity across channels.
        """
        # Generic init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Identity-ish tweak for the circular spatial depthwise conv
        with torch.no_grad():
            w = circ_dw.weight  # shape: [F1*D, 1, k_ch, 1] because groups=F1
            w.zero_()
            k_ch = w.shape[2]
            center = k_ch // 2
            # Set center tap to 1.0 for all output channels
            w[:, 0, center, 0] = 1.0
