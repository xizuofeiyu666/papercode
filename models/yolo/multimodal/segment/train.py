from copy import copy, deepcopy
from typing import Optional

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel, compute_model_gflops
from ultralytics.nn.tasks import SegmentationModel


class MultiModalSegmentationTrainer(SegmentationTrainer):
    """
    Multimodal segmentation trainer for RGB+X inputs.

    This mirrors the YOLOMM detection trainer while switching the task to
    segmentation and using SegmentationModel for the head.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

        self.modality = getattr(self.args, "modality", None)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

    # -----------------
    # Dataset building
    # -----------------
    def build_dataset(self, img_path, mode: str = "train", batch: Optional[int] = None):
        # Resolve x-modality info from data config
        x_modality, x_dir = self._resolve_x_modality_and_dir()

        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
        )

    # -----------
    # Model init
    # -----------
    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        # Determine channel count
        if self.is_dual_modal:
            x_channels = self.data.get("Xch", 3)
            channels = 3 + x_channels
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态分割模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch")
        else:
            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态分割模型初始化: {(self.modality or 'RGB')}(3ch)")

        # Inject dataset_config into YAML for router parsing
        cfg_dict = None
        if isinstance(cfg, str):
            try:
                from ultralytics.nn.tasks import yaml_model_load

                cfg_dict = yaml_model_load(cfg)
            except Exception:
                cfg_dict = None
        elif isinstance(cfg, dict):
            cfg_dict = deepcopy(cfg)

        if cfg_dict is not None:
            cfg_dict["dataset_config"] = dict(self.data)
            model = SegmentationModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        else:
            model = SegmentationModel(cfg, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)

        if hasattr(model, "multimodal_router") and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"已更新MultiModalRouter的数据集配置，Xch={self.data.get('Xch', 3)}")

        if hasattr(model, "mm_router") and model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )

        if weights:
            model.load(weights)

        # Optional FLOPs logs
        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            arch_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=False)
            if self.modality:
                route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=self.modality, route_aware=True)
                LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[{self.modality}]): {route_gflops:.2f}")
            else:
                route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=True)
                LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[dual]): {route_gflops:.2f}")
        except Exception:
            pass

        return model

    def get_validator(self):
        from ultralytics.models.yolo.multimodal.segment.val import MultiModalSegmentationValidator

        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "seg_loss"
        return MultiModalSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    # -----------------
    # Helper utilities
    # -----------------
    def _resolve_x_modality_and_dir(self) -> tuple[str, Optional[str]]:
        x_mod = None
        x_dir = None
        data = getattr(self, "data", {}) or {}

        if "modality_used" in data and isinstance(data["modality_used"], list):
            non_rgb = [m for m in data["modality_used"] if m != "rgb"]
            if non_rgb:
                x_mod = non_rgb[0]

        if x_mod is None and "models" in data and isinstance(data["models"], list):
            non_rgb = [m for m in data["models"] if m != "rgb"]
            if non_rgb:
                x_mod = non_rgb[0]

        if x_mod is None:
            x_mod = data.get("x_modality", None)

        mod_map = data.get("modality") or data.get("modalities")
        if isinstance(mod_map, dict) and x_mod in mod_map:
            x_dir = mod_map[x_mod]
        elif x_mod:
            x_dir = f"images_{x_mod}"

        if x_mod is None:
            LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
            x_mod = "depth"
            x_dir = data.get("modality", {}).get("depth", "images_depth") if isinstance(mod_map, dict) else "images_depth"

        return x_mod, x_dir

    # -----------------
    # Visualization
    # -----------------
    def plot_training_samples(self, batch, ni):
        """
        绘制多模态训练样本，支持分割任务的掩码叠加。

        - 多模态（未传入单模态 ablation）: 分别输出 RGB、X（灰度可视化）与并排对比图；
        - 单模态（传入 modality）: 仅输出该模态；
        - 分割叠加：在 RGB 与 X 单独图像上叠加 mask；并排图由于坐标系变化，暂不叠加 mask。
        """
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
            resolve_x_modality,
        )

        images = batch["img"]  # [B, 3+Xch, H, W]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]

        batch_idx = ensure_batch_idx_long(batch["batch_idx"]) if "batch_idx" in batch else None
        if batch_idx is None:
            # fallback to zeros if missing
            import torch

            batch_idx = ensure_batch_idx_long(torch.zeros(cls.shape[0], dtype=torch.long))
            batch["batch_idx"] = batch_idx

        # masks（可选）
        masks = batch.get("masks", None)

        # X 通道数
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))

        # 单模态
        if self.modality:
            if self.modality == 'rgb':
                plot_images(
                    rgb_images,
                    batch_idx,
                    cls,
                    bboxes,
                    masks=masks,
                    paths=paths,
                    fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
                    on_plot=self.on_plot,
                )
            else:
                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch_idx,
                    cls,
                    bboxes,
                    masks=masks,  # 同尺寸，可直接叠加
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
                    fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
                    on_plot=self.on_plot,
                )
            return

        # 双模态：RGB
        plot_images(
            rgb_images,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        # 双模态：X（灰度）
        x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
        plot_images(
            x_visual,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )

        # 并排对比（仅 bbox 以避免 mask 坐标系偏移）
        side_by_side_images = concat_side_by_side(rgb_images, x_visual)
        batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
            batch_idx, cls, bboxes, None
        )
        plot_images(
            side_by_side_images,
            batch_ids_dup,
            cls_ids_dup,
            bboxes_dup,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )

    # -----------------
    # Checkpoint I/O
    # -----------------
    def save_model(self):
        """保存模型并写入多模态元信息（与检测版一致）。"""
        from ultralytics.utils.patches import torch_load
        import torch

        super().save_model()

        if hasattr(self, 'multimodal_config'):
            ckpt = torch_load(self.last, map_location='cpu')
            ckpt['multimodal_config'] = getattr(self, 'multimodal_config', None)
            ckpt['modality'] = self.modality
            torch.save(ckpt, self.last)

            if self.best.exists():
                ckpt_best = torch_load(self.best, map_location='cpu')
                ckpt_best['multimodal_config'] = getattr(self, 'multimodal_config', None)
                ckpt_best['modality'] = self.modality
                torch.save(ckpt_best, self.best)

    def final_eval(self):
        """最终评估后记录多模态信息。"""
        super().final_eval()

        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
            if self.modality:
                LOGGER.info(f"最终评估完成 - 单模态训练: {self.modality}-only")
            else:
                LOGGER.info(f"最终评估完成 - 双模态训练: RGB+{x_modality}")
        else:
            LOGGER.info("最终评估完成 - 多模态分割")
