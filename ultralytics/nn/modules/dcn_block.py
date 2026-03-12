"""Deformable Convolution v2 modules for YOLO11 backbone.

Replaces fixed-grid convolutions in C3k2 bottleneck blocks with DCNv2,
enabling adaptive spatial sampling for thermal image feature extraction.

Variants:
- DCNv2Block: base DCN (offset_groups=1, standard 3x3 offset prediction)
- DCNv2Block_MG: multi-group offsets (offset_groups=4) — different channel groups see different deformations
- DCNv2Block_DL: dilated offset prediction (dilation=2) — larger context for offset prediction
- DCNv2Block_SE: DCN + Squeeze-and-Excitation channel attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3, C3k2


# =============================================================================
# DCNv2 Block variants
# =============================================================================

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


class DCNv2Block_MG(nn.Module):
    """DCNv2 with multi-group offsets.

    Different channel groups learn independent spatial deformations.
    offset_groups=4: 4 groups of channels each attend to different spatial locations.
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1, offset_groups=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.offset_groups = offset_groups
        padding = kernel_size // 2
        k2 = kernel_size * kernel_size

        # Multi-group: offset_groups sets of (2*k*k offsets + k*k masks)
        self.offset_conv = nn.Conv2d(
            c1, 3 * k2 * offset_groups, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.dcn = DeformConv2d(
            c1, c2, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False,
        )
        num_groups = min(32, c2)
        while c2 % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        k2 = self.kernel_size ** 2
        og = self.offset_groups
        out = self.offset_conv(x)
        offset = out[:, :2 * k2 * og, :, :]
        offset = offset.clamp(-64.0, 64.0).nan_to_num(0.0)
        mask = torch.sigmoid(out[:, 2 * k2 * og:, :, :])
        return self.act(self.gn(self.dcn(x, offset, mask)))


class DCNv2Block_DL(nn.Module):
    """DCNv2 with dilated offset prediction.

    Offset prediction uses dilation=2 for 5x5 effective receptive field,
    giving offsets more spatial context at no extra parameter cost.
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1, offset_dilation=2):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Dilated offset prediction: RF = 5x5 with dilation=2
        offset_channels = 3 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            c1, offset_channels, kernel_size,
            stride=stride, padding=offset_dilation, dilation=offset_dilation, bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.dcn = DeformConv2d(
            c1, c2, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False,
        )
        num_groups = min(32, c2)
        while c2 % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        k2 = self.kernel_size ** 2
        out = self.offset_conv(x)
        offset = out[:, :2 * k2, :, :]
        offset = offset.clamp(-64.0, 64.0).nan_to_num(0.0)
        mask = torch.sigmoid(out[:, 2 * k2:, :, :])
        return self.act(self.gn(self.dcn(x, offset, mask)))


class DCNv2Block_SE(nn.Module):
    """DCNv2 with Squeeze-and-Excitation channel attention.

    Combines spatial adaptivity (DCN) with channel adaptivity (SE).
    SE reweights channels after deformable convolution.
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1, reduction=4):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        offset_channels = 3 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            c1, offset_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.dcn = DeformConv2d(
            c1, c2, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False,
        )
        num_groups = min(32, c2)
        while c2 % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU(inplace=True)

        # SE block
        mid = max(c2 // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        k2 = self.kernel_size ** 2
        out = self.offset_conv(x)
        offset = out[:, :2 * k2, :, :]
        offset = offset.clamp(-64.0, 64.0).nan_to_num(0.0)
        mask = torch.sigmoid(out[:, 2 * k2:, :, :])
        out = self.act(self.gn(self.dcn(x, offset, mask)))
        return out * self.se(out)


# =============================================================================
# Bottleneck variants (cv1=Conv 1x1, cv2=DCN variant)
# =============================================================================

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


class Bottleneck_DCN_MG(nn.Module):
    """Bottleneck with multi-group DCNv2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv2Block_MG(c_, c2, kernel_size=k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DCN_DL(nn.Module):
    """Bottleneck with dilated-offset DCNv2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv2Block_DL(c_, c2, kernel_size=k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DCN_SE(nn.Module):
    """Bottleneck with DCNv2 + SE attention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv2Block_SE(c_, c2, kernel_size=k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# =============================================================================
# C3k variants (CSP with DCN bottlenecks)
# =============================================================================

class C3k_DCN(C3):
    """C3k with DCNv2 Bottleneck blocks (CSP split-merge pattern)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_DCN(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k_DCN_MG(C3):
    """C3k with multi-group DCNv2 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_DCN_MG(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k_DCN_DL(C3):
    """C3k with dilated-offset DCNv2 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_DCN_DL(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k_DCN_SE(C3):
    """C3k with DCNv2+SE Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_DCN_SE(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


# =============================================================================
# C3k2 variants (top-level modules for YAML configs)
# =============================================================================

class C3k2_DCN(C3k2):
    """C3k2 with Deformable Convolution Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_DCN(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


class C3k2_DCN_MG(C3k2):
    """C3k2 with multi-group offset DCNv2 (offset_groups=4)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_DCN_MG(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DCN_MG(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


class C3k2_DCN_DL(C3k2):
    """C3k2 with dilated offset prediction DCNv2 (dilation=2)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_DCN_DL(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DCN_DL(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


class C3k2_DCN_SE(C3k2):
    """C3k2 with DCNv2 + Squeeze-and-Excitation attention."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_DCN_SE(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DCN_SE(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


# =============================================================================
# Depthwise Separable DCN — reduces irregular memory reads on CPU
# =============================================================================

class DCNv2Block_DW(nn.Module):
    """Depthwise Separable DCNv2: depthwise deformable conv + pointwise conv.

    Standard DCNv2 performs C_in irregular reads per spatial-kernel position.
    Depthwise variant performs 1 read per position (groups=C_in), then mixes
    channels via standard pointwise Conv2d (BLAS/SIMD-optimized).
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Offset + mask prediction (shared across all channels)
        offset_channels = 3 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            c1, offset_channels, kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # Depthwise deformable conv
        self.dcn_dw = DeformConv2d(
            c1, c1, kernel_size,
            stride=stride, padding=padding, groups=c1, bias=False,
        )

        # Pointwise conv for channel mixing (standard, BLAS-friendly)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)

        num_groups = min(32, c2)
        while c2 % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        k2 = self.kernel_size ** 2
        out = self.offset_conv(x)
        offset = out[:, :2 * k2, :, :].clamp(-64.0, 64.0).nan_to_num(0.0)
        mask = torch.sigmoid(out[:, 2 * k2:, :, :])
        return self.act(self.gn(self.pw(self.dcn_dw(x, offset, mask))))


class Bottleneck_DCN_DW(nn.Module):
    """Bottleneck with Depthwise Separable DCNv2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv2Block_DW(c_, c2, kernel_size=k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_DCN_DW(C3):
    """C3k with Depthwise Separable DCNv2 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_DCN_DW(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_DCN_DW(C3k2):
    """C3k2 with Depthwise Separable DCNv2 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_DCN_DW(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DCN_DW(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


# =============================================================================
# Large Kernel — CPU-friendly DCN replacement (no irregular memory access)
# =============================================================================

class LargeKernelBlock(nn.Module):
    """Large depthwise kernel + pointwise as CPU-friendly DCN replacement.

    Approximates DCN's expanded receptive field using regular memory access.
    DWConv 7x7 covers spatial extent similar to typical DCN offset range
    while being fully SIMD/Winograd/BLAS-compatible on CPU.
    """

    def __init__(self, c1, c2, kernel_size=7, stride=1):
        super().__init__()
        padding = kernel_size // 2

        # Large depthwise conv for expanded receptive field
        self.dw = nn.Conv2d(
            c1, c1, kernel_size, stride=stride,
            padding=padding, groups=c1, bias=False,
        )

        # Pointwise conv for channel mixing
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)

        num_groups = min(32, c2)
        while c2 % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.pw(self.dw(x))))


class Bottleneck_LK(nn.Module):
    """Bottleneck with Large Kernel block replacing 3x3 conv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = LargeKernelBlock(c_, c2, kernel_size=7)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_LK(C3):
    """C3k with Large Kernel Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_LK(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_LK(C3k2):
    """C3k2 with Large Kernel Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_LK(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_LK(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


# =============================================================================
# Large Kernel 13×13 — increased receptive field
# =============================================================================

class Bottleneck_LK13(nn.Module):
    """Bottleneck with Large Kernel 13×13 block."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = LargeKernelBlock(c_, c2, kernel_size=13)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_LK13(C3):
    """C3k with Large Kernel 13×13 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_LK13(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_LK13(C3k2):
    """C3k2 with Large Kernel 13×13 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_LK13(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_LK13(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


# =============================================================================
# Large Kernel with Structural Re-parameterization
# Train: parallel DW branches (main + 5×5 + 3×3). Inference: fused single DW.
# =============================================================================

class LargeKernelBlock_RepParam(nn.Module):
    """Large depthwise kernel with structural re-parameterization.

    Training: main DW kernel + parallel 5×5 and 3×3 DW branches for richer gradients.
    Inference: branches fused into single DW kernel (zero extra cost).
    Reference: RepLKNet (Ding et al., CVPR 2022).
    """

    def __init__(self, c1, c2, kernel_size=7, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Main large kernel branch
        self.dw_main = nn.Conv2d(c1, c1, kernel_size, stride=stride,
                                  padding=padding, groups=c1, bias=False)
        # Parallel small kernel branches (enrich gradient flow during training)
        self.dw_small1 = nn.Conv2d(c1, c1, 5, stride=stride,
                                    padding=2, groups=c1, bias=False)
        self.dw_small2 = nn.Conv2d(c1, c1, 3, stride=stride,
                                    padding=1, groups=c1, bias=False)

        # Pointwise + norm + act
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        num_groups = min(32, c2)
        while c2 % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU(inplace=True)

        self._fused = False

    def forward(self, x):
        if self._fused:
            dw_out = self.dw_main(x)
        else:
            dw_out = self.dw_main(x) + self.dw_small1(x) + self.dw_small2(x)
        return self.act(self.gn(self.pw(dw_out)))

    def fuse_reparam(self):
        """Merge parallel branches into main kernel for inference."""
        if self._fused:
            return
        main_k = self.kernel_size
        fused_weight = self.dw_main.weight.data.clone()
        for branch in [self.dw_small1, self.dw_small2]:
            bk = branch.kernel_size[0]
            pad = (main_k - bk) // 2
            fused_weight += F.pad(branch.weight.data, [pad, pad, pad, pad])
        self.dw_main.weight.data = fused_weight
        del self.dw_small1
        del self.dw_small2
        self._fused = True


class Bottleneck_LK_RepParam(nn.Module):
    """Bottleneck with re-parameterizable Large Kernel 7×7."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = LargeKernelBlock_RepParam(c_, c2, kernel_size=7)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_LK_RepParam(C3):
    """C3k with re-parameterizable LK7 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_LK_RepParam(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_LK_RepParam(C3k2):
    """C3k2 with re-parameterizable LK7 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_LK_RepParam(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_LK_RepParam(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


# =============================================================================
# Large Kernel 13×13 + Re-parameterization (RepLKNet-style)
# =============================================================================

class Bottleneck_LK13_RepParam(nn.Module):
    """Bottleneck with re-parameterizable Large Kernel 13×13."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = LargeKernelBlock_RepParam(c_, c2, kernel_size=13)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_LK13_RepParam(C3):
    """C3k with re-parameterizable LK13 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_LK13_RepParam(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_LK13_RepParam(C3k2):
    """C3k2 with re-parameterizable LK13 Bottleneck blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3k_LK13_RepParam(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_LK13_RepParam(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )
