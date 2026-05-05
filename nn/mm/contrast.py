"""
Contrastive learning utilities for multimodal (RGB+X) detection.

Components
- RoiExtractor: Map GT boxes to feature maps and extract region vectors by average pooling.
- ProjectionHead: 2-layer MLP with optional lazy init on first call; outputs L2-normalized embeddings.
- InfoNCELoss: Symmetric NT-Xent over in-batch pairs with temperature tau.
- ContrastController: Orchestrates hooks → ROI → projection → loss; designed to be used from model.loss().

Notes
- 不做任何自动降级：若本步没有有效正对，直接返回 None，由调用方决定是否加入对比损失。
- 分阶段名称来自 Hook 自动命名，如 'P3','P4','P5'。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils import LOGGER


def _parse_stage_from_name(name: str) -> Optional[str]:
    # Expect names like: CL.RGB.P4.L6.output[#k]
    m = re.match(r"^CL\.(RGB|X)\.(P\d+)\.", name)
    return m.group(2) if m else None


def _parse_mod_from_name(name: str) -> Optional[str]:
    m = re.match(r"^CL\.(RGB|X)\.", name)
    return m.group(1) if m else None


class RoiExtractor:
    """Simple ROI extractor using average pooling within projected bbox on feature map."""

    def __init__(self, min_size: int = 1):
        self.min_size = min_size

    @staticmethod
    def _xywhn_to_xyxyf(xywhn: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # xywh normalized (0..1) → xyxy in feature coordinates
        x, y, w, h = xywhn.unbind(-1)
        cx = x * W
        cy = y * H
        ww = w * W
        hh = h * H
        x1 = (cx - ww / 2).clamp(0, W - 1)
        y1 = (cy - hh / 2).clamp(0, H - 1)
        x2 = (cx + ww / 2).clamp(0, W - 1)
        y2 = (cy + hh / 2).clamp(0, H - 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _roi_pool(self, feat: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        # feat: (1,C,H, W) slice for a single image; boxes: (N,4) in feature coords
        _, C, H, W = feat.shape
        if boxes_xyxy.numel() == 0:
            return feat.new_zeros((0, C))
        # Round to ints
        x1 = boxes_xyxy[:, 0].floor().long().clamp(0, W - 1)
        y1 = boxes_xyxy[:, 1].floor().long().clamp(0, H - 1)
        x2 = boxes_xyxy[:, 2].ceil().long().clamp(0, W - 1)
        y2 = boxes_xyxy[:, 3].ceil().long().clamp(0, H - 1)
        # Ensure at least 1x1
        x2 = torch.maximum(x2, x1)
        y2 = torch.maximum(y2, y1)

        out = []
        f0 = feat[0]  # (C,H,W)
        for i in range(x1.numel()):
            r = f0[:, y1[i] : y2[i] + 1, x1[i] : x2[i] + 1]
            out.append(r.mean(dim=(1, 2)))
        return torch.stack(out, dim=0)

    def extract(
        self,
        feat: torch.Tensor,
        xywhn: torch.Tensor,
        batch_idx: torch.Tensor,
        image_index: int,
        max_rois: int = 64,
    ) -> torch.Tensor:
        # Select GT boxes from batch belonging to image_index; xywh normalized
        # feat: (B,C,H,W)
        sel = (batch_idx == image_index).nonzero(as_tuple=False).flatten()
        if sel.numel() == 0:
            return feat.new_zeros((0, feat.shape[1]))
        sel = sel[:max_rois]
        boxes = xywhn[sel]
        H, W = feat.shape[2], feat.shape[3]
        boxes_xyxy = self._xywhn_to_xyxyf(boxes, H, W)
        return self._roi_pool(feat[image_index : image_index + 1], boxes_xyxy)


class ProjectionHead(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        # LazyLinear to ensure parameters are created on first forward with correct device/dtype
        self.fc1 = nn.LazyLinear(max(128, out_dim))
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.LazyLinear(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run projection in FP32 for numerical stability under AMP
        with autocast(enabled=False):
            x = x.float()
            z = self.fc2(self.act(self.fc1(x)))
            z = F.normalize(z, dim=-1, eps=1e-6)
        return z


class InfoNCELoss(nn.Module):
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2: (N,d) L2-normalized
        if z1.numel() == 0 or z2.numel() == 0:
            return z1.new_zeros(())
        assert z1.shape == z2.shape, "z1 and z2 must have same shape"
        with autocast(enabled=False):
            z1 = z1.float()
            z2 = z2.float()
            N = z1.shape[0]
            logits = (z1 @ z2.t()) / float(self.tau)  # (N,N)
            labels = torch.arange(N, device=z1.device)
            loss_12 = F.cross_entropy(logits, labels)
            loss_21 = F.cross_entropy(logits.t(), labels)
            return 0.5 * (loss_12 + loss_21)


@dataclass
class ContrastConfig:
    tau: float = 0.07
    proj_dim: int = 128
    lambda_weight: float = 0.1
    max_rois_per_image: int = 64
    share_head: bool = False
    preferred_stages: Tuple[str, ...] = ("P4", "P5", "P3")


class ContrastController(nn.Module):
    """Manage extracting ROI pairs from hooked features and computing InfoNCE loss."""

    def __init__(self, cfg: Optional[ContrastConfig] = None):
        super().__init__()
        self.cfg = cfg or ContrastConfig()
        self.head_rgb = ProjectionHead(out_dim=self.cfg.proj_dim)
        self.head_x = self.head_rgb if self.cfg.share_head else ProjectionHead(out_dim=self.cfg.proj_dim)
        self.infonce = InfoNCELoss(tau=self.cfg.tau)
        self.roi = RoiExtractor()

    def _pair_stage_features(self, buffers: Dict[str, torch.Tensor]) -> Optional[Tuple[str, torch.Tensor, torch.Tensor]]:
        # Group by stage: {'P4': {'RGB':[...], 'X':[...]} }
        stage_map: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        for name, t in buffers.items():
            mod = _parse_mod_from_name(name)
            stg = _parse_stage_from_name(name)
            if mod is None or stg is None:
                continue
            stage_map.setdefault(stg, {}).setdefault(mod, []).append(t)

        chosen_stage = None
        feat_rgb = feat_x = None
        for stg in self.cfg.preferred_stages:
            if stg in stage_map and 'RGB' in stage_map[stg] and 'X' in stage_map[stg]:
                chosen_stage = stg
                feat_rgb = stage_map[stg]['RGB'][-1]
                feat_x = stage_map[stg]['X'][-1]
                break
        if chosen_stage is None:
            for stg, mods in stage_map.items():
                if 'RGB' in mods and 'X' in mods:
                    chosen_stage = stg
                    feat_rgb = mods['RGB'][-1]
                    feat_x = mods['X'][-1]
                    break
        if chosen_stage is None or feat_rgb is None or feat_x is None:
            return None
        return chosen_stage, feat_rgb, feat_x

    def forward(self, hook_buffers: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        pair = self._pair_stage_features(hook_buffers)
        if pair is None:
            return None, {}
        stage, f_rgb, f_x = pair
        # Debug: check finiteness of tapped features
        try:
            if not torch.isfinite(f_rgb).all() or not torch.isfinite(f_x).all():
                def _nf(t):
                    return {
                        'nan': int(torch.isnan(t).sum().item()),
                        'inf': int(torch.isinf(t).sum().item()),
                        'min': float(torch.nanmin(t.float()).item()),
                        'max': float(torch.nanmax(t.float()).item()),
                        'dtype': str(t.dtype),
                        'device': str(t.device),
                    }
                LOGGER.warning(f"[CL][tap] non-finite feature at stage={stage}: RGB={_nf(f_rgb)}, X={_nf(f_x)}")
        except Exception:
            pass
        if f_rgb.dim() != 4 or f_x.dim() != 4:
            return None, {}
        if f_rgb.shape[0] != f_x.shape[0]:
            return None, {}

        img = batch.get('img')
        device = img.device if isinstance(img, torch.Tensor) else f_rgb.device
        batch_idx = batch.get('batch_idx')
        bboxes = batch.get('bboxes')  # normalized xywh
        if batch_idx is None or bboxes is None:
            return None, {}
        batch_idx = batch_idx.to(device)
        bboxes = bboxes.to(device)

        # Collect ROI vectors per image
        zs_rgb, zs_x = [], []
        B = f_rgb.shape[0]
        for bi in range(B):
            roi_rgb = self.roi.extract(f_rgb, bboxes, batch_idx, image_index=bi, max_rois=self.cfg.max_rois_per_image)
            roi_x = self.roi.extract(f_x, bboxes, batch_idx, image_index=bi, max_rois=self.cfg.max_rois_per_image)
            n = min(roi_rgb.shape[0], roi_x.shape[0])
            if n > 0:
                zs_rgb.append(roi_rgb[:n])
                zs_x.append(roi_x[:n])

        if not zs_rgb:
            return None, {}

        z_rgb = torch.cat(zs_rgb, dim=0)
        z_x = torch.cat(zs_x, dim=0)
        # Debug: check ROI vectors
        try:
            if not torch.isfinite(z_rgb).all() or not torch.isfinite(z_x).all():
                def _nf(t):
                    return {
                        'nan': int(torch.isnan(t).sum().item()),
                        'inf': int(torch.isinf(t).sum().item()),
                        'min': float(torch.nanmin(t.float()).item()),
                        'max': float(torch.nanmax(t.float()).item()),
                        'dtype': str(t.dtype),
                        'device': str(t.device),
                    }
                LOGGER.warning(f"[CL][roi] non-finite ROI: RGB={_nf(z_rgb)}, X={_nf(z_x)}")
        except Exception:
            pass

        # Projection (FP32 inside the head)
        z_rgb = self.head_rgb(z_rgb)
        z_x = self.head_x(z_x)
        # Debug: check projected embeddings
        try:
            if not torch.isfinite(z_rgb).all() or not torch.isfinite(z_x).all():
                def _nf(t):
                    return {
                        'nan': int(torch.isnan(t).sum().item()),
                        'inf': int(torch.isinf(t).sum().item()),
                        'min': float(torch.nanmin(t.float()).item()),
                        'max': float(torch.nanmax(t.float()).item()),
                        'dtype': str(t.dtype),
                        'device': str(t.device),
                    }
                LOGGER.warning(f"[CL][proj] non-finite embedding: RGB={_nf(z_rgb)}, X={_nf(z_x)}")
        except Exception:
            pass

        # Contrastive loss (FP32 inside InfoNCELoss)
        loss_c = self.infonce(z_rgb, z_x)
        # Debug: check contrastive loss
        try:
            if not torch.isfinite(loss_c):
                LOGGER.warning(f"[CL][loss] non-finite InfoNCE: val={float(loss_c.detach().cpu())}")
        except Exception:
            pass

        sim_mean = float((z_rgb * z_x).sum(-1).mean().detach().cpu())
        stats = {
            'stage': {'P': int(stage[1:]) if stage and len(stage) > 1 else -1},
            'num_pairs': int(z_rgb.shape[0]),
            'sim_mean': sim_mean,
        }
        return loss_c, stats
