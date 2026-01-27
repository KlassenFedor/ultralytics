"""
Custom attention modules for YOLO11
EMA: Efficient Multi-Scale Attention (ICASSP 2023)
LCA: Lightweight Coordinate Attention (ALSS-YOLO 2024)
CoordAtt: Coordinate Attention (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['EMA', 'LCA', 'CoordAtt', 'EMA_Temp', 'EMA_MultiScale', 'EMA_TIR']


class EMA(nn.Module):
    """Efficient Multi-Scale Attention Module"""
    def __init__(self, c1, c2=None, factor=32):
        super().__init__()
        c2 = c2 or c1
        self.groups = factor
        assert c1 // self.groups > 0
        
        group_channels = c1 // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, 1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group = c // self.groups
        x = x.view(b * self.groups, group, h, w)
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        x1 = self.gn(x_h * x_w.permute(0, 1, 3, 2))
        x1 = self.conv1x1(x1)
        x1 = x * self.softmax(self.agp(x1).view(b * self.groups, group, 1, 1))
        
        x2 = self.conv3x3(x)
        x2_h = self.pool_h(x2)
        x2_w = self.pool_w(x2).permute(0, 1, 3, 2)
        x11 = self.softmax(self.gn(x2_h * x2_w.permute(0, 1, 3, 2)))
        x12 = x2 * x11
        
        out = (x1 + x12).view(b, c, h, w)
        return out


class CoordAtt(nn.Module):
    """Coordinate Attention Module (CVPR 2021)"""
    def __init__(self, c1, c2=None, reduction=32):
        super().__init__()
        c2 = c2 or c1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, c2, 1)
        self.conv_w = nn.Conv2d(mip, c2, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h


class LCA(nn.Module):
    """Lightweight Coordinate Attention (ALSS-YOLO 2024)"""
    def __init__(self, c1, c2=None, reduction=32):
        super().__init__()
        c2 = c2 or c1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // reduction)
        
        # Depthwise separable convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, groups=c1),  # Depthwise
            nn.Conv2d(c1, mip, 1),             # Pointwise
        )
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, c2, 1)
        self.conv_w = nn.Conv2d(mip, c2, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h
    

class EMA_TIR(nn.Module):
    """
    EMA-TIR: Thermal Infrared оптимизированный Efficient Multi-Scale Attention.
    
    Унифицированный модуль со всеми модификациями, управляемыми флагами True/False.
    
    Args:
        c1 (int): Входные каналы
        c2 (int): Не используется (для совместимости с Ultralytics)
        factor (int): Количество групп (default=32)
        
        # Temperature Scaling
        temperature (float): Температура softmax (default=1.0)
            - τ < 1.0: sharper attention (рекомендуется 0.5-0.7 для thermal)
            - τ = 1.0: стандартный softmax  
            - τ > 1.0: smoother attention
        use_temperature (bool): Включить temperature scaling (default=False)
        learnable_temp (bool): Сделать τ обучаемой (default=False)
        
        # Multi-Scale Convolutions
        use_multiscale (bool): Dilated convolutions d=1,2 (default=False)
        
        # Contrast Normalization
        use_contrast (bool): Local Contrast Normalization (default=False)
        
        # Residual Connection
        use_residual (bool): Skip connection (default=False)
        
        # SE Attention
        use_se (bool): Squeeze-and-Excitation (default=False)
    
    YAML format:
        # [c1, factor, temp, use_temp, learnable, use_ms, use_contr, use_res, use_se]
        - [-1, 1, EMA_TIR, [1024, 32, 0.7, True, False, True, False, True, False]]
    """
    
    def __init__(
        self,
        c1: int,
        c2: int = None,
        factor: int = 32,
        # Temperature
        temperature: float = 1.0,
        use_temperature: bool = False,
        learnable_temp: bool = False,
        # Multi-scale
        use_multiscale: bool = False,
        # Contrast
        use_contrast: bool = False,
        contrast_kernel: int = 5,
        # Residual
        use_residual: bool = False,
        # SE
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()
        
        # Сохраняем флаги
        self.use_temperature = use_temperature
        self.use_multiscale = use_multiscale
        self.use_contrast = use_contrast
        self.use_residual = use_residual
        self.use_se = use_se
        
        self.groups = factor
        assert c1 // self.groups > 0, f"c1={c1} должно делиться на factor={factor}"
        group_c = c1 // self.groups
        
        # ========== Temperature Scaling ==========
        if use_temperature:
            if learnable_temp:
                self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temperature))))
            else:
                self.register_buffer('log_temp', torch.log(torch.tensor(float(temperature))))
            self.learnable_temp = learnable_temp
        else:
            self.register_buffer('log_temp', torch.zeros(1))  # temp=1.0
            self.learnable_temp = False
        
        # ========== Contrast Normalization ==========
        if use_contrast:
            self.contrast_kernel = contrast_kernel
            self.contrast_scale = nn.Parameter(torch.ones(1))
            self.contrast_bias = nn.Parameter(torch.zeros(1))
        
        # ========== Residual Connection ==========
        if use_residual:
            self.residual_scale = nn.Parameter(torch.zeros(1))  # Init 0
        
        # ========== Squeeze-and-Excitation ==========
        if use_se:
            se_c = max(group_c // se_reduction, 8)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(group_c, se_c),
                nn.ReLU(inplace=True),
                nn.Linear(se_c, group_c),
                nn.Sigmoid()
            )
        
        # ========== Core EMA ==========
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_c, group_c)
        self.conv1x1 = nn.Conv2d(group_c, group_c, 1)
        
        # ========== Multi-scale или 3x3 ==========
        if use_multiscale:
            self.conv_d1 = nn.Conv2d(group_c, group_c, 3, padding=1, dilation=1)
            self.conv_d2 = nn.Conv2d(group_c, group_c, 3, padding=2, dilation=2)
            self.conv_fuse = nn.Conv2d(group_c * 2, group_c, 1)
        else:
            self.conv3x3 = nn.Conv2d(group_c, group_c, 3, padding=1)
    
    @property
    def temperature(self) -> torch.Tensor:
        """Текущая температура."""
        return torch.exp(self.log_temp).clamp(0.1, 5.0)
    
    def scaled_softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Softmax с опциональным temperature scaling."""
        if self.use_temperature:
            return F.softmax(x / self.temperature, dim=dim)
        return self.softmax(x)
    
    def contrast_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Local Contrast Normalization."""
        k = self.contrast_kernel
        local_mean = F.avg_pool2d(x, k, stride=1, padding=k//2)
        local_sq = F.avg_pool2d(x**2, k, stride=1, padding=k//2)
        local_std = torch.sqrt((local_sq - local_mean**2).clamp(min=1e-6))
        return self.contrast_scale * (x - local_mean) / local_std + self.contrast_bias
    
    def spatial_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial convolution (single или multi-scale)."""
        if self.use_multiscale:
            return self.conv_fuse(torch.cat([self.conv_d1(x), self.conv_d2(x)], dim=1))
        return self.conv3x3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        g = c // self.groups
        
        # Group processing
        xg = x.view(b * self.groups, g, h, w)
        identity = xg
        
        # Contrast normalization (для вычисления attention)
        xp = self.contrast_norm(xg) if self.use_contrast else xg
        
        # Branch 1: 1x1 Cross-Spatial
        xh = self.pool_h(xp)
        xw = self.pool_w(xp).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([xh, xw], dim=2))
        xh, xw = torch.split(hw, [h, w], dim=2)
        
        x1 = self.gn(xh * xw.permute(0, 1, 3, 2))
        x1 = self.conv1x1(x1)
        x1 = xg * self.scaled_softmax(self.agp(x1).view(b*self.groups, g, 1, 1))
        
        # Branch 2: Spatial Conv + Cross-Spatial
        x2 = self.spatial_conv(xp)
        x2h = self.pool_h(x2)
        x2w = self.pool_w(x2).permute(0, 1, 3, 2)
        x2_attn = self.scaled_softmax(self.gn(x2h * x2w.permute(0, 1, 3, 2)))
        x2 = self.spatial_conv(xg) * x2_attn
        
        # Combine
        out = x1 + x2
        
        # SE attention
        if self.use_se:
            se_w = self.se(xg).view(b*self.groups, g, 1, 1)
            out = out + xg * se_w
        
        # Residual
        if self.use_residual:
            out = out + self.residual_scale * identity
        
        return out.view(b, c, h, w)
    
    def extra_repr(self) -> str:
        mods = []
        if self.use_temperature:
            s = f"τ={self.temperature.item():.2f}"
            if self.learnable_temp:
                s += "(learnable)"
            mods.append(s)
        if self.use_multiscale:
            mods.append("multiscale")
        if self.use_contrast:
            mods.append("contrast")
        if self.use_residual:
            mods.append("residual")
        if self.use_se:
            mods.append("SE")
        return ", ".join(mods) if mods else "baseline"