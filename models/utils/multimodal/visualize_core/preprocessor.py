"""Preprocessor for visualize_core.

Provides unified letterbox and normalization for single or dual modality inputs.
No pseudocolor is applied to X modality per project requirement; X is kept in its
original channel count and numeric range before normalization.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import cv2


class Preprocessor:
    @staticmethod
    def passthrough(x: Any) -> Any:
        return x

    @staticmethod
    def model_input_channels(model: Any, default: int = 3) -> int:
        """
        Infer expected input channel count for visualization preprocessing.

        Notes:
        - Prefer multimodal router's Dual channel config (3+Xch) when available, because some
          multimodal architectures (e.g., mm-mid) keep the first Conv in_channels=3 but still
          accept a Dual input tensor that is routed internally by the router.
        - Fallback to the first Conv2d.in_channels for non-multimodal models.
        """

        # 1) Prefer multimodal router config (Dual = 3 + Xch)
        router = None
        for chain in (
            ("mm_router",),
            ("multimodal_router",),
            ("model", "mm_router"),
            ("model", "multimodal_router"),
        ):
            m = model
            ok = True
            for a in chain:
                if hasattr(m, a):
                    m = getattr(m, a)
                else:
                    ok = False
                    break
            if ok and m is not None:
                router = m
                break

        if router is not None:
            try:
                sources = getattr(router, "INPUT_SOURCES", None)
                if isinstance(sources, dict) and "Dual" in sources:
                    return int(sources["Dual"])
            except Exception:
                pass

        # 2) Fallback: first Conv2d.in_channels (typical for standard YOLO)
        try:
            import torch.nn as nn  # local import to keep module lightweight

            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    return int(m.in_channels)

            # Generic fallback: any module exposing in_channels/out_channels
            for m in model.modules():
                if hasattr(m, "in_channels") and hasattr(m, "out_channels"):
                    ic = int(getattr(m, "in_channels"))
                    if ic > 0:
                        return ic
        except Exception:
            pass

        return int(default)

    @staticmethod
    def model_input_size(model: Any, default: int = 640) -> int:
        try:
            if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
                return int(model.args.imgsz)
        except Exception:
            pass
        return int(default)

    @staticmethod
    def _letterbox(im: np.ndarray, new_shape: Union[int, Tuple[int, int]] = 640, color=(114, 114, 114), stride: int = 32) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int, int, int]]:
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (top, bottom, left, right)

    @staticmethod
    def _to_float01(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        if img.max() > 1.0:
            return img.astype(np.float32) / float(img.max())
        return img.astype(np.float32)

    @staticmethod
    def letterbox_single(image: np.ndarray, size: int) -> np.ndarray:
        lb, _, _ = Preprocessor._letterbox(image, new_shape=size)
        return Preprocessor._to_float01(lb)

    @staticmethod
    def letterbox_dual(rgb: np.ndarray, x: np.ndarray, size: int) -> np.ndarray:
        # First image defines padding for consistency
        rgb_lb, _, padding = Preprocessor._letterbox(rgb, new_shape=size)
        x_lb, _, _ = Preprocessor._letterbox(x, new_shape=size)
        # Normalize
        rgb_f = Preprocessor._to_float01(rgb_lb)
        x_f = Preprocessor._to_float01(x_lb)
        # Ensure channel dims
        if rgb_f.ndim == 2:
            rgb_f = rgb_f[:, :, None]
        if x_f.ndim == 2:
            x_f = x_f[:, :, None]
        # Concatenate channels [RGB(3), X(Xch)] without pseudocolor
        return np.concatenate([rgb_f, x_f], axis=2)

    @staticmethod
    def letterbox_dual_aligned(
        rgb: np.ndarray,
        x: np.ndarray,
        size: int,
        align_base: str = "rgb",
        stride: int = 32,
    ) -> np.ndarray:
        """
        Letterbox two modalities with the SAME ratio/padding derived from the base image.

        This enforces pixel-wise alignment on the final canvas by reusing the scaling ratio and
        padding computed from the base modality (RGB or X) for the other modality.

        Args:
            rgb: RGB image (HWC)
            x:   X-modality image (HWC or HW or HWC with 1+ channels)
            size: square target size (int)
            align_base: 'rgb' or 'x' indicating which modality defines ratio/padding
            stride: model stride for padding alignment (default: 32)

        Returns:
            Concatenated HWC float32 array with channels [RGB(3), X(Xch)] in [0,1].
        """
        base = rgb if str(align_base).lower() == "rgb" else x

        # Compute base letterbox once to obtain target padding
        base_lb, ratio, pad = Preprocessor._letterbox(base, new_shape=size, stride=stride)
        r = ratio[0]
        top, bottom, left, right = pad

        def _apply_with_ratio(im: np.ndarray) -> np.ndarray:
            shape = im.shape[:2]
            # Resize with base ratio r
            new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
            if shape[::-1] != new_unpad:
                im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            # Apply the same padding tuple as base
            return cv2.copyMakeBorder(
                im,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )

        if base is rgb:
            rgb_lb = base_lb
            x_lb = _apply_with_ratio(x)
        else:
            x_lb = base_lb
            rgb_lb = _apply_with_ratio(rgb)

        # Normalize to [0,1]
        rgb_f = Preprocessor._to_float01(rgb_lb)
        x_f = Preprocessor._to_float01(x_lb)

        # Ensure explicit channel dims
        if rgb_f.ndim == 2:
            rgb_f = rgb_f[:, :, None]
        if x_f.ndim == 2:
            x_f = x_f[:, :, None]

        return np.concatenate([rgb_f, x_f], axis=2)

    @staticmethod
    def prepare_inputs(inputs: Dict[str, np.ndarray], model: Any) -> np.ndarray:
        size = Preprocessor.model_input_size(model)
        if 'rgb' in inputs and 'x' in inputs:
            return Preprocessor.letterbox_dual(inputs['rgb'], inputs['x'], size)
        if 'rgb' in inputs:
            return Preprocessor.letterbox_single(inputs['rgb'], size)
        return Preprocessor.letterbox_single(inputs['x'], size)
