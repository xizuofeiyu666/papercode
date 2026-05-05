
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1,
                 bn: bool = True, act: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HiTPE-Lite(nn.Module):
    def __init__(
        self,
        inplanes: int = 64,
        embed_dims: Tuple[int, int, int] = (256, 512, 1024),
        in_chans: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        bn = use_bn

        self.stem_ir = nn.Sequential(
            ConvBNAct(in_chans, inplanes, k=3, s=2, p=1, bn=bn, act=True),
            ConvBNAct(inplanes, inplanes, k=3, s=1, p=1, bn=bn, act=True),
            ConvBNAct(inplanes, inplanes, k=3, s=1, p=1, bn=bn, act=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> 4x
        )

        self.conv2 = ConvBNAct(inplanes, 2 * inplanes, k=3, s=2, p=1, bn=bn, act=True)
        self.conv3 = ConvBNAct(2 * inplanes, 4 * inplanes, k=3, s=2, p=1, bn=bn, act=True)
        self.conv4 = ConvBNAct(4 * inplanes, 4 * inplanes, k=3, s=2, p=1, bn=bn, act=True)

        c8, c16, c32 = embed_dims
        self.out_8 = nn.Conv2d(2 * inplanes, c8, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_16 = nn.Conv2d(4 * inplanes, c16, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_32 = nn.Conv2d(4 * inplanes, c32, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x4 = self.stem_ir(ir)
        x8 = self.conv2(x4)
        x16 = self.conv3(x8)
        x32 = self.conv4(x16)

        t8 = self.out_8(x8)
        t16 = self.out_16(x16)
        t32 = self.out_32(x32)
        return t8, t16, t32


class ConvMixFusion(nn.Module):
    def __init__(self, channels: int, kernels: Tuple[int, ...] = (3, 3, 5, 7), groups: int = 4):
        super().__init__()
        assert channels % groups == 0, "channels must be divisible by groups"
        assert len(kernels) == groups, "len(kernels) must equal groups"

        self.groups = groups
        self.channels = channels
        self.channel_per_group = channels // groups

        convs_rgb, convs_ir = [], []
        for ks in kernels:
            pad = (ks - 1) // 2
            convs_rgb.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=pad, bias=True))
            convs_ir.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=pad, bias=True))

        self.convs_rgb = nn.ModuleList(convs_rgb)
        self.convs_ir = nn.ModuleList(convs_ir)
        # shared gate (per-group): 1x1 conv on partial channels -> 1 map
        self.gate = nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x) -> torch.Tensor:
        # 仅支持列表/元组传参的双输入模式
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ConvMixFusion 需要以 [rgb, ir] 列表/元组形式传入两路特征")
        rgb, ir = x
        if not (isinstance(rgb, torch.Tensor) and isinstance(ir, torch.Tensor)):
            raise TypeError("ConvMixFusion 的两路输入必须为张量")
        if rgb.shape != ir.shape:
            raise ValueError(f"两路输入形状需一致，got {rgb.shape} vs {ir.shape}")
        B, C, H, W = rgb.shape
        outs = []
        for i in range(self.groups):
            sl = slice(i * self.channel_per_group, (i + 1) * self.channel_per_group)
            rgb_i = self.convs_rgb[i](rgb[:, sl, :, :])
            ir_i = self.convs_ir[i](ir[:, sl, :, :])
            mix = rgb_i + ir_i
            alpha = torch.sigmoid(self.gate(mix))  # gate in [0,1]
            outs.append(rgb_i * alpha + ir_i * (1 - alpha))
        return torch.cat(outs, dim=1)


class ScalarGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(2 * channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ScalarGate 需要以 [rgb, ir] 列表/元组形式传入两路特征")
        rgb, ir = x
        if not (isinstance(rgb, torch.Tensor) and isinstance(ir, torch.Tensor)):
            raise TypeError("ScalarGate 的两路输入必须为张量")
        if rgb.shape != ir.shape:
            raise ValueError(f"两路输入形状需一致，got {rgb.shape} vs {ir.shape}")
        gap_rgb = F.adaptive_avg_pool2d(rgb, output_size=1)
        gap_ir = F.adaptive_avg_pool2d(ir, output_size=1)
        g = torch.cat([gap_rgb, gap_ir], dim=1)  # (B, 2C, 1, 1)
        z = torch.sigmoid(self.fc(g))  # (B, 1, 1, 1)
        return rgb * (1 - z) + ir * z


class SMRF(nn.Module):

    def __init__(self, channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(1, int(channels * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SMRF 需要以 [rgb, ir] 列表/元组形式传入两路特征")
        rgb, ir = x
        if not (isinstance(rgb, torch.Tensor) and isinstance(ir, torch.Tensor)):
            raise TypeError("SMRF 的两路输入必须为张量")
        if rgb.shape != ir.shape:
            raise ValueError(f"两路输入形状需一致，got {rgb.shape} vs {ir.shape}")
        gap_rgb = F.adaptive_avg_pool2d(rgb, output_size=1)
        gap_ir = F.adaptive_avg_pool2d(ir, output_size=1)
        g = torch.cat([gap_rgb, gap_ir], dim=1)  # (B, 2C, 1, 1)
        z = torch.sigmoid(self.mlp(g))           # (B, C, 1, 1)
        return rgb * (1 - z) + ir * z


def ncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "Inputs must have the same shape"
    B = a.shape[0]
    a_flat = a.view(B, -1)
    b_flat = b.view(B, -1)
    a_mean = a_flat.mean(dim=1, keepdim=True)
    b_mean = b_flat.mean(dim=1, keepdim=True)
    num = ((a_flat - a_mean) * (b_flat - b_mean)).sum(dim=1)
    den = torch.sqrt(((a_flat - a_mean) ** 2).sum(dim=1) * ((b_flat - b_mean) ** 2).sum(dim=1) + 1e-12)
    return num / den


__all__ = [
    " HiTPE-Lite",
    "ConvMixFusion",
    "ScalarGate",
    "SMRF",
    "ncc",
]
