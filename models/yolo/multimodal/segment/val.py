from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import smart_inference_mode, compute_model_gflops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import callbacks
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device


class MultiModalSegmentationValidator(SegmentationValidator):
    """
    Multimodal segmentation validator for RGB+X inputs.

    Provides channel-aware warmup and accepts a runtime modality parameter,
    mirroring the detection-side multimodal validator behavior.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

        if args:
            if isinstance(args, dict):
                self.modality = args.get("modality", None)
            else:
                self.modality = getattr(args, "modality", None)
        else:
            self.modality = None

        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):  # type: ignore[override]
        self.training = trainer is not None
        if self.training:
            # Align with training setup
            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            model.eval()
            # Inject runtime modality to router during train-time val
            try:
                if hasattr(model, "mm_router") and model.mm_router and self.modality:
                    model.mm_router.set_runtime_params(self.modality)
            except Exception:
                pass
        else:
            # 方案A：不在子类中手动创建 AutoBackend，避免二次包裹与错误 fuse 调用。
            # 委托父类创建并管理 AutoBackend（包括设备、数据、warmup 等）。
            pass

        # Defer to parent implementation for the main loop
        return super().__call__(trainer=trainer, model=model)

    # -----------------
    # Dataset building
    # -----------------
    def build_dataset(self, img_path, mode: str = "val", batch=None):
        """
        构建多模态分割验证数据集。

        通过传递 multi_modal_image=True 启用 YOLOMultiModalImageDataset，
        并从 data.yaml/配置中解析 x_modality 与目录映射。严格遵循分割任务，
        使用 task='segment' 使 use_segments=True。
        """
        # 解析模态组合
        data = getattr(self, "data", {}) or {}
        modalities = None
        mod_map = None
        if "modality_used" in data and isinstance(data["modality_used"], list):
            modalities = data["modality_used"]
        elif "models" in data and isinstance(data["models"], list):
            modalities = data["models"]
        mod_map = data.get("modality") or data.get("modalities")

        # 提取 X 模态
        x_modality = None
        if isinstance(modalities, list):
            non_rgb = [m for m in modalities if m != "rgb"]
            if non_rgb:
                x_modality = non_rgb[0]
        if x_modality is None:
            x_modality = data.get("x_modality", None)
        # X 目录
        x_modality_dir = None
        if isinstance(mod_map, dict) and x_modality in mod_map:
            x_modality_dir = mod_map[x_modality]
        elif x_modality:
            x_modality_dir = f"images_{x_modality}"

        # stride
        stride = getattr(self, "stride", 32) or 32

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=True,
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=False,  # 验证默认关闭自体生成
        )

    # -----------------
    # Visualization (GT samples)
    # -----------------
    def plot_val_samples(self, batch, ni):
        """多模态分割验证：绘制 GT（RGB/X/并排 + mask）。"""
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
            resolve_x_modality,
        )

        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        masks = batch.get("masks", None)

        batch_idx = ensure_batch_idx_long(batch.get("batch_idx")) if "batch_idx" in batch else None
        if batch_idx is None:
            import torch
            batch_idx = ensure_batch_idx_long(torch.zeros(cls.shape[0], dtype=torch.long))

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = resolve_x_modality(getattr(self, 'modality', None), getattr(self, 'data', None))

        # RGB
        plot_images(
            rgb_images,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=paths,
            fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )
        # X
        x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
        plot_images(
            x_visual,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )
        # 并排（仅 bbox）
        side = concat_side_by_side(rgb_images, x_visual)
        bidx2, cls2, bb2, _ = duplicate_bboxes_for_side_by_side(batch_idx, cls, bboxes, None)
        plot_images(
            side,
            bidx2,
            cls2,
            bb2,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )

    # -----------------
    # Visualization (Preds)
    # -----------------
    def plot_predictions(self, batch, preds, ni):
        """多模态分割验证：绘制预测（RGB/X/并排 + mask）。

        批量绘制策略：与YOLOMM检测验证器保持一致的输出风格。
        - 按batch绘制，而非逐图绘制
        - 输出格式：val_batch{ni}_pred_{modality}.jpg
        - 每图最多100个实例，避免性能问题
        """

        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            resolve_x_modality,
        )
        import torch
        from ultralytics.utils import ops

        CAP = 100  # 每图绘制上限

        images = batch["img"]
        paths = batch["im_file"]
        B, C, H, W = images.shape

        # 1. 合并batch内所有预测（参考检测验证器策略）
        for i, p in enumerate(preds):
            p["batch_idx"] = torch.ones_like(p["cls"]) * i

        keys = preds[0].keys()
        batched_preds = {}

        # 合并所有预测字段
        for k in keys:
            if k == "masks":
                # masks特殊处理：限制每图最多CAP个
                masks_list = []
                for p in preds:
                    m = p["masks"][:CAP] if p["masks"].numel() else p["masks"]
                    masks_list.append(m.to(torch.uint8).cpu())
                batched_preds[k] = torch.cat(masks_list, dim=0) if masks_list and masks_list[0].numel() else torch.zeros((0, H, W), dtype=torch.uint8)
            else:
                batched_preds[k] = torch.cat([p[k][:CAP] for p in preds], dim=0)

        # 2. 转换坐标格式：xyxy -> xywhn
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])
        batched_preds["bboxes"][:, 0] /= W
        batched_preds["bboxes"][:, 1] /= H
        batched_preds["bboxes"][:, 2] /= W
        batched_preds["bboxes"][:, 3] /= H

        # 3. 拆分模态
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = resolve_x_modality(getattr(self, 'modality', None), getattr(self, 'data', None))

        # 4. 批量绘制三视图
        # RGB视图
        plot_images(
            rgb_images,
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            confs=batched_preds.get("conf"),
            masks=batched_preds.get("masks"),
            paths=paths,
            fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # X视图
        x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
        plot_images(
            x_visual,
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            confs=batched_preds.get("conf"),
            masks=batched_preds.get("masks"),
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 并排视图
        side = concat_side_by_side(rgb_images, x_visual)
        bidx2, cls2, bb2, conf2 = duplicate_bboxes_for_side_by_side(
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            batched_preds.get("conf")
        )
        masks_side = torch.cat([batched_preds["masks"], batched_preds["masks"]], dim=2) if batched_preds["masks"].numel() else batched_preds["masks"]
        plot_images(
            side,
            bidx2,
            cls2,
            bb2,
            confs=conf2,
            masks=masks_side,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
