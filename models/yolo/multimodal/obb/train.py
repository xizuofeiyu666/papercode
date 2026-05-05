# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy
from typing import Optional

from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel, compute_model_gflops
from ultralytics.nn.tasks import OBBModel
from ultralytics.data.dataset import YOLOMultiModalImageDataset


class MultiModalOBBTrainer(OBBTrainer):
    """
    多模态旋转框训练器（RGB+X），复用 YOLOMM 路由与 6+ 通道输入，输出旋转框预测。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

        # 与 YOLOMM 检测/分割保持一致的模态控制
        self.modality = getattr(self.args, "modality", None)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        if self.modality:
            LOGGER.info(f"初始化 MultiModalOBBTrainer - 单模态训练: {self.modality}-only")
        else:
            LOGGER.info("初始化 MultiModalOBBTrainer - 双模态训练")

    # -----------------
    # Dataset building
    # -----------------
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建多模态 OBB 数据集，开启 RGB+X 图像管线。
        """
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
            x_modality=self._determine_x_modality_from_data(),
            x_modality_dir=self._get_x_modality_path(self._determine_x_modality_from_data()),
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
        )

    # -----------
    # Model init
    # -----------
    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        """
        使用 OBBModel，并按数据集的 Xch 动态确定输入通道数。
        """
        # 输入通道：双模态 3+Xch，单模态 3
        if self.is_dual_modal:
            x_channels = self.data.get("Xch", 3)
            channels = 3 + x_channels
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"多模态 OBB 模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch")
        else:
            channels = 3
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"单模态 OBB 模型初始化: {(self.modality or 'RGB')}(3ch)")

        model = OBBModel(cfg, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        if hasattr(model, "mm_router") and model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )

        if weights:
            model.load(weights)

        # Optional FLOPs log
        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            arch_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=False)
            if self.modality:
                route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=self.modality, route_aware=True)
                LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[{self.modality}]): {route_gflops:.2f}")
            else:
                route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=True)
                LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[dual]): {route_gflops:.2f}")
        except Exception as e:
            LOGGER.warning(f"GFLOPs 统计失败（可忽略）: {e}")

        return model

    def get_validator(self):
        """
        返回多模态 OBB 验证器，保持与训练时损失项一致。
        """
        from ultralytics.models.yolo.multimodal.obb.val import MultiModalOBBValidator

        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MultiModalOBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    # -----------------
    # Helper utilities
    # -----------------
    def _determine_x_modality_from_data(self):
        """
        参考 detection trainer 的推断逻辑，解析 data.yaml 中的 X 模态名称。
        """
        data = getattr(self, "data", {}) or {}
        # modality_used 或 models 字段
        for key in ("modality_used", "models"):
            if key in data and isinstance(data[key], list):
                non_rgb = [m for m in data[key] if m != "rgb"]
                if non_rgb:
                    return non_rgb[0]
        # 兼容字段
        if "x_modality" in data:
            return data["x_modality"]
        return "depth"

    def _get_x_modality_path(self, x_modality: str):
        """
        根据 data.yaml modalities 映射获取 X 模态目录。
        """
        data = getattr(self, "data", {}) or {}
        mod_map = data.get("modalities") or data.get("modality")
        if isinstance(mod_map, dict) and x_modality in mod_map:
            return mod_map[x_modality]
        return f"images_{x_modality}"
