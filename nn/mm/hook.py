"""
Ultralytics Multimodal Hook Utilities

This module provides lightweight hook management for feature tapping in multimodal models.
It hides the need for explicit `buffer` names by auto-generating stable, human-readable
identifiers based on (modality, stage, layer_idx, tap), e.g.:

    CL.RGB.P4.L6.output

If multiple hooks are attached to the same layer with identical key fields, a numeric
suffix is appended (e.g., `#1`, `#2`).

Notes
- This module is intentionally self-contained and does not integrate with training loops.
- It can be safely imported by the package without side effects.
- No automatic graceful-degradation is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils import LOGGER


@dataclass
class BatchContext:
    """
    Batch-scoped metadata container for downstream consumers.

    Attributes:
        device: Preferred torch device
        img_size: (H, W) original image size
        stride_map: Mapping from stage name (e.g., 'P3') to stride (int)
        gt: Optional ground-truth structure for ROI mapping
        extras: Optional dict for arbitrary context
    """

    device: Optional[torch.device] = None
    img_size: Optional[tuple] = None
    stride_map: Optional[Dict[str, int]] = None
    gt: Optional[Any] = None
    extras: Optional[Dict[str, Any]] = None


class FeatureTap:
    """
    A thin wrapper around PyTorch forward hooks for consistent feature tapping.

    Parameters:
        module: The nn.Module to attach to
        name: Logical name used in HookManager buffers
        tap: 'output' (default) or 'input'
        detach: Whether to detach the captured tensor from graph
        normalize: Whether to L2-normalize along channel dimension
        on_capture: Callback to receive (name, tensor) when captured
    """

    def __init__(
        self,
        module: nn.Module,
        name: str,
        tap: str = "output",
        detach: bool = False,
        normalize: bool = False,
        on_capture: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> None:
        self.module = module
        self.name = name
        self.tap = tap
        self.detach = detach
        self.normalize = normalize
        self.on_capture = on_capture
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _choose_tensor(self, module: nn.Module, inputs: tuple, output: Any) -> Optional[torch.Tensor]:
        if self.tap == "output":
            t = output
        else:
            # tap == 'input': pick the first Tensor-like input
            t = None
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    t = x
                    break
        if t is None:
            return None
        if not isinstance(t, torch.Tensor):
            # For tuple/list outputs, use the first tensor if present
            if isinstance(t, (tuple, list)):
                for x in t:
                    if isinstance(x, torch.Tensor):
                        t = x
                        break
            if not isinstance(t, torch.Tensor):
                return None
        return t

    def _hook_fn(self, module: nn.Module, inputs: tuple, output: Any) -> None:
        t = self._choose_tensor(module, inputs, output)
        if t is None:
            return
        if self.detach:
            t = t.detach()
        # CL路径数值护栏：将捕获特征有限化（仅作用于对比分支缓存，不回写主干）
        with autocast(enabled=False):
            if isinstance(t, torch.Tensor):
                t = torch.nan_to_num(t.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if self.normalize:
            # FP32 normalization for stability under AMP
            with autocast(enabled=False):
                if t.dim() >= 2:
                    dim = 1 if t.dim() >= 2 else 0
                    t = F.normalize(t.float(), dim=dim, eps=1e-6)
        if self.on_capture:
            try:
                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                    LOGGER.warning(f"[CL][hook] non-finite captured tensor name={self.name}, dtype={t.dtype}, device={t.device}, nan={int(torch.isnan(t).sum())}, inf={int(torch.isinf(t).sum())}")
            except Exception:
                pass
            self.on_capture(self.name, t)

    def _pre_hook_fn(self, module: nn.Module, inputs: tuple) -> None:
        # Forward-pre-hook provides inputs only
        t = None
        for x in inputs:
            if isinstance(x, torch.Tensor):
                t = x
                break
        if t is None:
            return
        if self.detach:
            t = t.detach()
        # CL路径数值护栏：将捕获特征有限化（仅作用于对比分支缓存，不回写主干）
        with autocast(enabled=False):
            if isinstance(t, torch.Tensor):
                t = torch.nan_to_num(t.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if self.normalize:
            with autocast(enabled=False):
                if t.dim() >= 2:
                    dim = 1 if t.dim() >= 2 else 0
                    t = F.normalize(t.float(), dim=dim, eps=1e-6)
        if self.on_capture:
            try:
                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                    LOGGER.warning(f"[CL][hook-pre] non-finite captured tensor name={self.name}, dtype={t.dtype}, device={t.device}, nan={int(torch.isnan(t).sum())}, inf={int(torch.isinf(t).sum())}")
            except Exception:
                pass
            self.on_capture(self.name, t)

    def register(self) -> None:
        if self._handle is not None:
            return
        if self.tap == "input":
            self._handle = self.module.register_forward_pre_hook(self._pre_hook_fn, with_kwargs=False)
        else:
            self._handle = self.module.register_forward_hook(self._hook_fn, with_kwargs=False)

    def remove(self) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            finally:
                self._handle = None


class HookManager:
    """
    Manage feature-tapping hooks and provide auto-named buffers.

    Usage example:
        hm = HookManager()
        hm.register(module=m, spec={
            'modality': 'RGB', 'stage': 'P4', 'tap': 'output',
            'detach': True, 'normalize': True, 'layer_idx': 6,  # buffer optional
        })

        # During forward passes, captured tensors are stored in hm.buffers
        feats = hm.collect(pop=False)  # {'CL.RGB.P4.L6.output': tensor, ...}
    """

    def __init__(self) -> None:
        self.buffers: Dict[str, torch.Tensor] = {}
        self._taps: List[FeatureTap] = []
        self._per_layer_counts: Dict[str, int] = {}
        self._used_names: set[str] = set()

    @staticmethod
    def _sanitize(s: Any) -> str:
        return str(s).replace(" ", "_")

    def _base_name(self, modality: str, stage: str, layer_idx: Optional[int], tap: str) -> str:
        mod = self._sanitize(modality)
        stg = self._sanitize(stage)
        idx = f"L{layer_idx}" if layer_idx is not None else "L?"
        tapv = tap if tap in {"input", "output"} else "output"
        return f"CL.{mod}.{stg}.{idx}.{tapv}"

    def _unique_name(self, base: str) -> str:
        # If base unused, return it; else append #k
        if base not in self._used_names:
            self._used_names.add(base)
            return base
        k = self._per_layer_counts.get(base, 0) + 1
        name = f"{base}#{k}"
        while name in self._used_names:
            k += 1
            name = f"{base}#{k}"
        self._per_layer_counts[base] = k
        self._used_names.add(name)
        return name

    def _on_capture(self, name: str, t: torch.Tensor) -> None:
        # Store latest tensor for the given name; do not move devices implicitly
        self.buffers[name] = t

    def register(self, module: nn.Module, spec: Dict[str, Any]) -> str:
        """
        Register a hook on `module` using spec fields.

        Supported keys in spec:
            modality (str): Required
            stage (str): Required
            tap (str): 'output' (default) or 'input'
            detach (bool): default False
            normalize (bool): default False
            layer_idx (int): Optional, improves reproducible naming
            buffer (str): Optional explicit name; if omitted, auto-generated

        Returns:
            str: The resolved buffer name used for this hook
        """
        modality = spec.get("modality")
        stage = spec.get("stage")
        tap = spec.get("tap", "output")
        detach = bool(spec.get("detach", False))
        normalize = bool(spec.get("normalize", False))
        layer_idx = spec.get("layer_idx", None)

        if modality is None or stage is None:
            raise ValueError("HookManager.register requires 'modality' and 'stage' in spec")

        explicit = spec.get("buffer", None)
        if explicit:
            name = self._sanitize(explicit)
            if name in self._used_names:
                raise ValueError(f"Duplicate hook buffer name: {name}")
            self._used_names.add(name)
        else:
            base = self._base_name(modality, stage, layer_idx, tap)
            name = self._unique_name(base)

        tapper = FeatureTap(
            module=module,
            name=name,
            tap=tap,
            detach=detach,
            normalize=normalize,
            on_capture=self._on_capture,
        )
        tapper.register()
        self._taps.append(tapper)
        return name

    def collect(self, pop: bool = False) -> Dict[str, torch.Tensor]:
        """Return captured buffers; optionally clear them."""
        out = dict(self.buffers)
        if pop:
            self.buffers.clear()
        return out

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.buffers.get(name, None)

    def has_hooks(self) -> bool:
        """Return True if any hooks have been registered."""
        return len(self._taps) > 0

    def list_registered(self) -> list[str]:
        """Return a sorted list of registered hook buffer names."""
        return sorted(self._used_names)

    def summary_text(self, max_items: int = 20) -> str:
        """Return a one-shot textual summary of registered hooks (no emojis)."""
        names = self.list_registered()
        lines = [
            f"HOOK enabled: {len(names)} registered",
        ]
        if names:
            k = min(max_items, len(names))
            lines.append("HOOK buffers (sample):")
            for i in range(k):
                lines.append(f"- {names[i]}")
            if len(names) > k:
                lines.append(f"... and {len(names) - k} more")
        return "\n".join(lines)

    def clear(self) -> None:
        for t in self._taps:
            t.remove()
        self._taps.clear()
        self.buffers.clear()
        self._per_layer_counts.clear()
        self._used_names.clear()
