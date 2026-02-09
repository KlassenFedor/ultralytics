"""Deformable Convolution v2 modules for YOLO11 backbone.

Replaces fixed-grid convolutions in C3k2 bottleneck blocks with DCNv2,
enabling adaptive spatial sampling for thermal image feature extraction.
"""

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3, C3k2


class DCNv2Block(nn.Module):
    """Deformable Convolution v2 with offset + modulation prediction."""

    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Offset conv: predicts 2*k*k offsets + k*k modulation masks
        offset_channels = 3 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            c1, offset_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        # Zero-init so DCN starts as standard convolution
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # Deformable convolution
        self.dcn = DeformConv2d(
            c1, c2, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False,
        )
        # GroupNorm: works with any spatial size (including 1x1) and small batches
        num_groups = min(32, c2)
        while c2 % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        k2 = self.kernel_size ** 2
        out = self.offset_conv(x)
        offset = out[:, :2 * k2, :, :]
        # Clamp offsets to prevent segfault from NaN/inf (e.g. torch.empty in thop profiling)
        offset = offset.clamp(-64.0, 64.0).nan_to_num(0.0)
        mask = torch.sigmoid(out[:, 2 * k2:, :, :])
        return self.act(self.gn(self.dcn(x, offset, mask)))


class Bottleneck_DCN(nn.Module):
    """Standard YOLO Bottleneck with DCNv2 replacing the 3x3 conv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)       # Standard 1x1
        self.cv2 = DCNv2Block(c_, c2, kernel_size=k[1])  # DCNv2 3x3
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_DCN(C3):
    """C3k with DCNv2 Bottleneck blocks (CSP split-merge pattern)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_DCN(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_DCN(C3k2):
    """C3k2 with Deformable Convolution Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_DCN(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )
