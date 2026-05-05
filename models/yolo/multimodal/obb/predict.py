# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.data.augment import LetterBox

from ultralytics.models.yolo.multimodal.predict import MultiModalDetectionPredictor


class MultiModalOBBPredictor(MultiModalDetectionPredictor):
    """
    多模态 OBB 预测器：复用 YOLOMM 输入解析 + 旋转框结果封装与可视化。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"
        self._last_ratio_pad = None  # 保存最近一次多模态 letterbox 的缩放/填充，用于还原坐标

    @staticmethod
    def _duplicate_obb_for_side_by_side(batch_ids, cls_ids, bboxes_xrwhr_norm, confs=None):
        """复制归一化 xywhr 旋转框到左右半幅，用于并排可视化（宽度按 0.5 缩放，角度保持不变）。"""
        if bboxes_xrwhr_norm is None:
            return batch_ids, cls_ids, bboxes_xrwhr_norm, confs
        if hasattr(bboxes_xrwhr_norm, "numel") and bboxes_xrwhr_norm.numel() == 0:
            return batch_ids, cls_ids, bboxes_xrwhr_norm, confs

        is_torch = isinstance(bboxes_xrwhr_norm, torch.Tensor)
        if is_torch:
            device = bboxes_xrwhr_norm.device
            if batch_ids is not None and not isinstance(batch_ids, torch.Tensor):
                batch_ids = torch.as_tensor(batch_ids, device=device, dtype=torch.long)
            if cls_ids is not None and not isinstance(cls_ids, torch.Tensor):
                cls_ids = torch.as_tensor(cls_ids, device=device, dtype=torch.long)
            if confs is not None and not isinstance(confs, torch.Tensor):
                confs = torch.as_tensor(confs, device=device, dtype=torch.float32)

            left = bboxes_xrwhr_norm.clone()
            right = bboxes_xrwhr_norm.clone()

            left[:, 0] *= 0.5
            left[:, 2] *= 0.5
            right[:, 0] = right[:, 0] * 0.5 + 0.5
            right[:, 2] *= 0.5

            bboxes_dup = torch.cat([left, right], dim=0)
            batch_dup = torch.cat([batch_ids, batch_ids], dim=0) if batch_ids is not None else None
            cls_dup = torch.cat([cls_ids, cls_ids], dim=0) if cls_ids is not None else None
            confs_dup = torch.cat([confs, confs], dim=0) if confs is not None else None
        else:
            import numpy as np

            left = bboxes_xrwhr_norm.copy()
            right = bboxes_xrwhr_norm.copy()

            left[:, 0] *= 0.5
            left[:, 2] *= 0.5
            right[:, 0] = right[:, 0] * 0.5 + 0.5
            right[:, 2] *= 0.5

            bboxes_dup = np.concatenate([left, right], axis=0)
            batch_dup = np.concatenate([batch_ids, batch_ids], axis=0) if batch_ids is not None else None
            cls_dup = np.concatenate([cls_ids, cls_ids], axis=0) if cls_ids is not None else None
            confs_dup = np.concatenate([confs, confs], axis=0) if confs is not None else None

        return batch_dup, cls_dup, bboxes_dup, confs_dup

    def _process_dual_modality(self, im):
        """
        重新实现双模态预处理：先在原始空间对齐尺寸，再一次性 letterbox 到目标分辨率，
        记录 ratio/pad 供后续坐标还原，避免两路独立 letterbox 带来的缩放偏差。
        """
        import numpy as np
        import cv2

        # 已经是 6 通道 tensor 的快速通道
        if isinstance(im, torch.Tensor) and im.dim() == 4 and im.shape[1] == 6:
            self._last_ratio_pad = None
            return self._finalize_tensor(im)

        rgb_images, x_images = self._parse_dual_modal_input(im)
        if not (isinstance(rgb_images, (list, tuple)) and isinstance(x_images, (list, tuple))):
            raise ValueError("双模态输入解析失败，期望列表形式的 RGB 与 X 源。")
        if len(rgb_images) != len(x_images):
            raise ValueError(f"双模态批大小不一致: RGB={len(rgb_images)}, X={len(x_images)}")

        processed = []
        ratio_pads = []
        for idx, (rgb_img, x_img) in enumerate(zip(rgb_images, x_images)):
            rgb_np = np.asarray(rgb_img)
            x_np = np.asarray(x_img)

            # 将单通道扩展为 3 通道，保持与训练一致
            if rgb_np.ndim == 2:
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2BGR)
            if x_np.ndim == 2:
                x_np = cv2.cvtColor(x_np, cv2.COLOR_GRAY2BGR)

            # 若两路原始尺寸不同，先将 X 缩放到 RGB 尺寸以保持空间对齐
            if rgb_np.shape[:2] != x_np.shape[:2]:
                LOGGER.warning(
                    f"MM OBB: 第{idx}对输入尺寸不一致 RGB{rgb_np.shape[:2]} vs X{x_np.shape[:2]}，将 X 重采样到 RGB 尺寸。"
                )
                x_np = cv2.resize(x_np, (rgb_np.shape[1], rgb_np.shape[0]), interpolation=cv2.INTER_LINEAR)

            merged = np.concatenate([rgb_np, x_np], axis=2)  # [H,W,6]

            # 与训练一致的 letterbox：center padding，使用模型 stride
            target_shape = self.imgsz if isinstance(self.imgsz, (list, tuple)) else (self.imgsz, self.imgsz)

            # AutoBackend.stride 为 int；纯 torch 模型可能为 Tensor 或 list/tuple
            stride = getattr(self.model, "stride", 32)
            if isinstance(stride, torch.Tensor):
                stride = int(stride.max())
            elif isinstance(stride, (list, tuple)):
                stride = int(max(stride))
            else:
                stride = int(stride)

            lb = LetterBox(new_shape=target_shape, auto=False, scale_fill=False, scaleup=True, stride=stride)

            # 记录与 LetterBox 完全一致的 ratio_pad，供 scale_boxes 反算
            h0, w0 = merged.shape[:2]
            r = min(target_shape[0] / h0, target_shape[1] / w0)
            new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))
            dw, dh = target_shape[1] - new_unpad_w, target_shape[0] - new_unpad_h
            dw, dh = dw / 2, dh / 2
            padw = int(round(dw - 0.1))
            padh = int(round(dh - 0.1))

            merged_lb = lb(image=merged)
            ratio_pads.append(((r, r), (padw, padh)))

            processed.append(merged_lb)

        # 组装为 tensor
        arr = np.stack([m.transpose(2, 0, 1) for m in processed])  # BCHW
        tensor = torch.from_numpy(arr).to(self.device)
        tensor = tensor.half() if self.model.fp16 else tensor.float()
        tensor /= 255

        # 缓存 ratio/pad 供坐标反算与调试
        self._last_ratio_pad = ratio_pads
        LOGGER.debug(
            f"MM OBB: 预处理完成 batch={len(processed)}, target_shape={processed[0].shape[1:]}, "
            f"ratio_pad={ratio_pads}"
        )
        return tensor

    def construct_result(self, pred, img, orig_img, img_path):
        """OBB 结果封装，使用与预处理一致的 ratio/pad 还原坐标并输出调试日志。"""
        # 取出对应样本的 ratio_pad
        ratio_pad = None
        if isinstance(self._last_ratio_pad, list) and len(self._last_ratio_pad):
            ratio_pad = self._last_ratio_pad.pop(0)
        elif isinstance(self._last_ratio_pad, tuple):
            ratio_pad = self._last_ratio_pad

        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, ratio_pad=ratio_pad, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)

        LOGGER.debug(
            f"MM OBB: 构建结果 n={len(obb)}, ratio_pad={ratio_pad}, "
            f"img_shape={img.shape[2:]}, orig_shape={orig_img.shape[:2]}"
        )
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb)

    def write_results(self, i: int, p: Path, im: torch.Tensor, s) -> str:
        """
        复用 YOLOMM 的统一可视化，但使用 OBB 旋转框。
        - 先调用父类以保留 txt/crop 保存逻辑
        - RGB、X 原图各输出一张；并排图使用旋转框复制到双倍宽度坐标系
        """

        orig_save = getattr(self.args, "save", False)
        try:
            self.args.save = False
            string = super().write_results(i, p, im, s)
        finally:
            self.args.save = orig_save

        if not orig_save:
            return string

        import cv2
        import numpy as np

        from ultralytics.models.utils.multimodal.vis import concat_side_by_side, ensure_batch_idx_long
        from ultralytics.models.utils.multimodal.vis import clip_boxes_norm_xywh as _clip_norm_xywh
        from ultralytics.utils.plotting import plot_images

        def _np_to_tensor3ch(img_np: np.ndarray) -> torch.Tensor:
            if img_np is None:
                raise RuntimeError("缺少原始图像用于可视化背景")
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError(f"不支持的原始图像形状: {img_np.shape}")
            t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0)

        def _get_orig_modal_tensors():
            if not hasattr(self, "_orig_imgs_cache") or self._orig_imgs_cache is None:
                raise RuntimeError("未找到原始图像缓存，无法生成以原图为背景的可视化")
            oi = self._orig_imgs_cache
            rgb_t, x_t = None, None
            if isinstance(oi, (list, tuple)):
                if len(oi) == 2:
                    rgb_t = _np_to_tensor3ch(oi[0])
                    x_t = _np_to_tensor3ch(oi[1])
                elif len(oi) == 1:
                    if self.modality and str(self.modality).lower() == "rgb":
                        rgb_t = _np_to_tensor3ch(oi[0])
                    else:
                        x_t = _np_to_tensor3ch(oi[0])
                else:
                    raise RuntimeError(f"原始图像数量异常: {len(oi)}")
            else:
                if self.modality and str(self.modality).lower() == "rgb":
                    rgb_t = _np_to_tensor3ch(oi)
                else:
                    x_t = _np_to_tensor3ch(oi)
            return rgb_t, x_t

        def _reproject_xywhr_to_target_norm(boxes_xywhr_px: torch.Tensor, orig_hw: tuple[int, int], target_h: int, target_w: int) -> torch.Tensor:
            if boxes_xywhr_px is None or boxes_xywhr_px.numel() == 0:
                return torch.zeros((0, 5), dtype=torch.float32)
            oh, ow = float(orig_hw[0]), float(orig_hw[1])
            sx, sy = float(target_w) / ow, float(target_h) / oh
            b = boxes_xywhr_px.clone().float()
            b[:, 0] *= sx
            b[:, 2] *= sx
            b[:, 1] *= sy
            b[:, 3] *= sy
            b[:, 0] /= float(target_w)
            b[:, 2] /= float(target_w)
            b[:, 1] /= float(target_h)
            b[:, 3] /= float(target_h)
            # 仅裁剪 xywh，角度保持
            clipped = _clip_norm_xywh(b[:, :4], 0.0, 1.0, 0.0, 1.0)
            b[:, :4] = clipped
            return b

        def _resolve_x_modality_strict():
            if self.is_single_modal and self.modality and self.modality.lower() == "rgb":
                return "rgb"
            return "x"

        result = self.results[i]
        obb_res = getattr(result, "obb", None)
        if obb_res is None:
            return string

        rgb_tensor, x_tensor = _get_orig_modal_tensors()
        base = p.stem

        n_boxes = 0 if obb_res is None else len(obb_res)
        if n_boxes:
            cls_ids = obb_res.cls
            boxes_px = torch.as_tensor(obb_res.xywhr, dtype=torch.float32)
            orig_h, orig_w = obb_res.orig_shape
            confs = torch.as_tensor(obb_res.conf, dtype=torch.float32)
            if not isinstance(cls_ids, torch.Tensor):
                cls_ids = torch.as_tensor(cls_ids, dtype=torch.long)
            batch_idx = ensure_batch_idx_long(torch.zeros(cls_ids.shape[0], device=boxes_px.device))
        else:
            cls_ids = torch.zeros((0,), dtype=torch.long)
            boxes_px = torch.zeros((0, 5), dtype=torch.float32)
            orig_h, orig_w = 1, 1
            confs = torch.zeros((0,), dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros((0,), dtype=torch.long))

        names = getattr(self.model, "names", {})
        x_modality = _resolve_x_modality_strict()

        if self.is_single_modal:
            if self.modality.lower() == "rgb":
                if rgb_tensor is None:
                    raise RuntimeError("期望RGB原图用于可视化，但缓存缺失")
                Ht, Wt = int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])
                boxes_norm_rgb = _reproject_xywhr_to_target_norm(boxes_px, (orig_h, orig_w), Ht, Wt)
                fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
                plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                            paths=[str(p)], fname=fname_rgb, names=names)
            else:
                if x_tensor is None:
                    raise RuntimeError("期望X原图用于可视化，但缓存缺失")
                Ht, Wt = int(x_tensor.shape[-2]), int(x_tensor.shape[-1])
                boxes_norm_x = _reproject_xywhr_to_target_norm(boxes_px, (orig_h, orig_w), Ht, Wt)
                fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
                plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                            paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                            fname=fname_x, names=names)
            return string

        if rgb_tensor is None or x_tensor is None:
            raise RuntimeError("双模态可视化需要RGB与X原图，但缓存缺失")

        Hr, Wr = int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])
        boxes_norm_rgb = _reproject_xywhr_to_target_norm(boxes_px, (orig_h, orig_w), Hr, Wr)
        fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
        plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                    paths=[str(p)], fname=fname_rgb, names=names)

        Hx, Wx = int(x_tensor.shape[-2]), int(x_tensor.shape[-1])
        boxes_norm_x = _reproject_xywhr_to_target_norm(boxes_px, (orig_h, orig_w), Hx, Wx)
        fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
        plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                    paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                    fname=fname_x, names=names)

        if (Hr, Wr) != (Hx, Wx):
            x_tensor_resized = torch.nn.functional.interpolate(
                x_tensor, size=(Hr, Wr), mode="bilinear", align_corners=False
            )
        else:
            x_tensor_resized = x_tensor

        side = concat_side_by_side(rgb_tensor, x_tensor_resized)
        batch_dup, cls_dup, boxes_dup, confs_dup = self._duplicate_obb_for_side_by_side(
            batch_idx, cls_ids, boxes_norm_rgb, confs
        )
        fname_mm = self.save_dir / f"pred_{base}_labels_multimodal.jpg"
        plot_images(side, batch_dup, cls_dup, boxes_dup, confs=confs_dup,
                    paths=[str(p.with_name(f"{base}_multimodal{p.suffix}"))],
                    fname=fname_mm, names=names)

        return string
