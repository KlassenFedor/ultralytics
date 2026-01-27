"""
Custom attention modules for YOLO11
EMA: Efficient Multi-Scale Attention (ICASSP 2023)
LCA: Lightweight Coordinate Attention (ALSS-YOLO 2024)
CoordAtt: Coordinate Attention (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['EMA', 'LCA', 'CoordAtt']


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
    

class EMA_Temp(nn.Module):
    """EMA с Temperature Scaling для thermal изображений."""
    def __init__(self, c1, c2=None, factor=32, temperature=1.0, learnable=False):
        super().__init__()
        self.groups = factor
        assert c1 // self.groups > 0
        
        group_c = c1 // self.groups
        
        # Temperature parameter
        if learnable:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        else:
            self.register_buffer('log_temp', torch.log(torch.tensor(float(temperature))))
        self.learnable = learnable
        
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_c, group_c)
        
        self.conv1x1 = nn.Conv2d(group_c, group_c, 1)
        self.conv3x3 = nn.Conv2d(group_c, group_c, 3, padding=1)

    @property
    def temperature(self):
        return torch.exp(self.log_temp).clamp(min=0.1, max=5.0)
    
    def scaled_softmax(self, x, dim=-1):
        return F.softmax(x / self.temperature, dim=dim)

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
        x1 = x * self.scaled_softmax(self.agp(x1).view(b * self.groups, group, 1, 1))
        
        x2 = self.conv3x3(x)
        x2_h = self.pool_h(x2)
        x2_w = self.pool_w(x2).permute(0, 1, 3, 2)
        x11 = self.scaled_softmax(self.gn(x2_h * x2_w.permute(0, 1, 3, 2)))
        x12 = x2 * x11
        
        out = x1 + x12
        return out.view(b, c, h, w)


class EMA_MultiScale(nn.Module):
    """EMA с Multi-Scale Dilated Convolutions."""
    def __init__(self, c1, c2=None, factor=32, dilations=(1, 2)):
        super().__init__()
        self.groups = factor
        assert c1 // self.groups > 0
        
        group_c = c1 // self.groups
        
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_c, group_c)
        
        self.conv1x1 = nn.Conv2d(group_c, group_c, 1)
        self.conv_d1 = nn.Conv2d(group_c, group_c, 3, padding=dilations[0], dilation=dilations[0])
        self.conv_d2 = nn.Conv2d(group_c, group_c, 3, padding=dilations[1], dilation=dilations[1])
        self.conv_fuse = nn.Conv2d(group_c * 2, group_c, 1)

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
        
        # Multi-scale branch
        feat_d1 = self.conv_d1(x)
        feat_d2 = self.conv_d2(x)
        x2 = self.conv_fuse(torch.cat([feat_d1, feat_d2], dim=1))
        
        x2_h = self.pool_h(x2)
        x2_w = self.pool_w(x2).permute(0, 1, 3, 2)
        x11 = self.softmax(self.gn(x2_h * x2_w.permute(0, 1, 3, 2)))
        x12 = x2 * x11
        
        out = x1 + x12
        return out.view(b, c, h, w)
