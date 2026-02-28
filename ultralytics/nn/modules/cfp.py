"""
Centralized Feature Pyramid (CFP) modules for CRT-YOLO.
Based on: "A High-Performance Thermal Infrared Object Detection Framework
with Centralized Regulation" (2505.10825v1)

EVC (Explicit Visual Center) = Stem + LightweightMLP + LVC
GCR (Global Centralized Regulation) is expressed via YAML (Upsample + Concat + Conv).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EVC", "LVC"]


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
        return x.div(keep) * mask


class LightweightMLP(nn.Module):
    """Lightweight MLP block from EVC (Fig. 5).

    Two residual sub-blocks:
      1) GN → DWConv(1x1) → ChannelScale → DropPath → +residual
      2) GN → ChannelMLP(c → 4c → c) → ChannelScale → DropPath → +residual
    """

    def __init__(self, channels: int, mlp_ratio: int = 4, drop_path: float = 0.1):
        super().__init__()
        # Sub-block 1: depthwise conv
        self.norm1 = nn.GroupNorm(1, channels)
        self.dwconv = nn.Conv2d(channels, channels, 1, groups=channels)
        self.scale1 = nn.Parameter(torch.ones(channels, 1, 1) * 1e-4)
        self.drop1 = DropPath(drop_path)

        # Sub-block 2: channel MLP
        self.norm2 = nn.GroupNorm(1, channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.scale2 = nn.Parameter(torch.ones(channels, 1, 1) * 1e-4)
        self.drop2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sub-block 1
        x = x + self.drop1(self.scale1 * self.dwconv(self.norm1(x)))
        # Sub-block 2
        x = x + self.drop2(self.scale2 * self.mlp(self.norm2(x)))
        return x


class LVC(nn.Module):
    """Learnable Visual Center (Fig. 6).

    Soft codebook assignment (VLAD-like) with learnable codewords and smoothing factors.
    Eq. 8-10 from the paper.
    """

    def __init__(self, channels: int, num_codes: int = 64):
        super().__init__()
        self.num_codes = num_codes
        # Project input features
        self.conv = nn.Conv2d(channels, channels, 1)
        # Learnable codebook: K codewords of dimension C
        self.codebook = nn.Parameter(torch.empty(num_codes, channels).normal_(0, 0.01))
        # Learnable smoothing factors (log-space for positivity)
        self.log_smooth = nn.Parameter(torch.zeros(num_codes))
        # Impact factor prediction
        self.fc = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Project: (B, C, H, W) → (B, N, C) where N = H*W
        x_proj = self.conv(x).flatten(2).permute(0, 2, 1)  # (B, N, C)

        # Smoothing factors
        smooth = self.log_smooth.exp()  # (K,)

        # Compute distances: ||x'_i - b_k||^2 for all i, k
        # x_proj: (B, N, C), codebook: (K, C)
        diff = x_proj.unsqueeze(2) - self.codebook.unsqueeze(0).unsqueeze(0)  # (B, N, K, C)
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, N, K)

        # Soft assignment weights: exp(-s_k * ||x'_i - b_k||^2) / sum_j(...)
        logits = -smooth.unsqueeze(0).unsqueeze(0) * dist_sq  # (B, N, K)
        assign = F.softmax(logits, dim=2)  # (B, N, K)

        # Aggregate: e_k = sum_i assign_ik * (x'_i - b_k)
        # assign: (B, N, K), diff: (B, N, K, C) → e: (B, K, C)
        e = (assign.unsqueeze(-1) * diff).sum(dim=1)  # (B, K, C)

        # Impact factor: FC(e) → sigmoid → (B, C)
        # Average over codewords then FC
        e_avg = e.mean(dim=1)  # (B, C)
        impact = torch.sigmoid(self.fc(e_avg))  # (B, C)

        # Channel-wise reweight + residual (eq. 9, 10)
        out = x * impact.view(b, c, 1, 1) + x  # Z = X * sigma(FC(e)) + X
        return out


class EVC(nn.Module):
    """Explicit Visual Center module (Section 4.3.1).

    Processes top-level backbone feature through:
      - Stem: Conv + BN + SiLU
      - Parallel branches: LightweightMLP and LVC
      - Output: Cat(MLP_out, LVC_out) → 2 * c_evc channels

    Args:
        c1: Input channels (from backbone top-level feature).
        c_evc: Internal EVC dimension (default 256, as in paper).
        num_codes: Number of LVC codebook entries (default 64).
        drop_path: DropPath rate for LightweightMLP (default 0.1).
        stem_k: Stem conv kernel size (default 7, use 3 for lightweight).
        mlp_ratio: MLP expansion ratio (default 4, use 2 for lightweight).

    YAML examples:
        # Original (paper): [256, 64, 0.1]        → stem_k=7, mlp_ratio=4
        # Light:            [128, 32, 0.1, 3, 2]   → stem_k=3, mlp_ratio=2
        # Tiny:             [64, 16, 0.1, 3, 2]    → stem_k=3, mlp_ratio=2
    """

    def __init__(self, c1: int, c_evc: int = 256, num_codes: int = 64,
                 drop_path: float = 0.1, stem_k: int = 7, mlp_ratio: int = 4):
        super().__init__()
        stem_k = int(stem_k)
        mlp_ratio = int(mlp_ratio)
        # Stem block (eq. 5): Conv → BN → SiLU
        self.stem = nn.Sequential(
            nn.Conv2d(c1, c_evc, stem_k, padding=stem_k // 2, bias=False),
            nn.BatchNorm2d(c_evc),
            nn.SiLU(inplace=True),
        )
        # Branch A: Lightweight MLP
        self.mlp = LightweightMLP(c_evc, mlp_ratio=mlp_ratio, drop_path=drop_path)
        # Branch B: Learnable Visual Center
        self.lvc = LVC(c_evc, num_codes=num_codes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xin = self.stem(x)
        mlp_out = self.mlp(xin)
        lvc_out = self.lvc(xin)
        return torch.cat([mlp_out, lvc_out], dim=1)  # 2 * c_evc channels
