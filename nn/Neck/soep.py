# Ultralytics YOLOMM - SOEP (Small Object Enhance Pyramid) Module
# SOEP: 小目标增强金字塔模块

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

__all__ = ['SPDConv', 'FGM', 'MKC', 'HiSoCF']


class SPDConv(nn.Module):
    def __init__(self, inc, ouc, dimension=1):
        """Initialize SPDConv with input/output channels."""
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        """Apply space-to-depth transformation followed by convolution."""
        # 将2x2区域的4个位置分别提取并拼接到通道维度
        x = torch.cat([
            x[..., ::2, ::2],   # 左上
            x[..., 1::2, ::2],  # 右上
            x[..., ::2, 1::2],  # 左下
            x[..., 1::2, 1::2]  # 右下
        ], 1)
        x = self.conv(x)
        return x


class FGM(nn.Module):
    def __init__(self, dim):
        """Initialize FGM with learnable alpha and beta parameters."""
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        """Apply frequency gating mechanism."""
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        # 频域变换
        x2_fft = torch.fft.fft2(x2, norm='backward')

        # 频域门控
        out = x1 * x2_fft

        # 逆变换回空间域
        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)

        # 加权融合
        return out * self.alpha + x * self.beta


class MKC(nn.Module):

    def __init__(self, dim):
        super().__init__()

        ker = 31
        pad = ker // 2

        # 输入输出卷积
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

        # 多尺度深度可分离卷积
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)  # 水平
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)  # 垂直
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)           # 全局
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)               # 局部

        self.act = nn.ReLU()

        # 空间通道注意力 (SCA)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 频域通道注意力 (FCA)
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        """Apply multi-scale kernels with frequency and spatial attention."""
        out = self.in_conv(x)

        # === 频域通道注意力 (FCA) ===
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)

        # === 空间通道注意力 (SCA) ===
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)

        # === 多尺度核融合 ===
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class HiSoCF(nn.Module):

        >>> csp_ok = CSPOmniKernel(256, e=0.25)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = csp_ok(x)  # shape: (1, 256, 32, 32)
    """

    def __init__(self, dim, e=0.25):
        """Initialize CSP-OmniKernel with channel split ratio."""
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        """Apply CSP structure with OmniKernel."""
        # 分支分割
        ok_branch, identity = torch.split(
            self.cv1(x),
            [int(self.cv1.conv.out_channels * self.e), int(self.cv1.conv.out_channels * (1 - self.e))],
            dim=1
        )
        # 融合输出
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))
