# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.mm.filling import generate_modality_filling
from ultralytics.models.yolo.multimodal.visualize.utils import load_image
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, YAML
from ultralytics.utils.torch_utils import compute_model_gflops


class YOLO(Model):
    """
    YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO models, automatically switching to specialized model types
    (YOLOWorld or YOLOE) based on the model filename. It supports various computer vision tasks including object
    detection, segmentation, classification, pose estimation, and oriented bounding box detection.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, segment, classify, pose, obb).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model with automatic type detection.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained YOLOv11n detection model
        >>> model = YOLO("yolo11n.pt")

        Load a pretrained YOLO11n segmentation model
        >>> model = YOLO("yolo11n-seg.pt")

        Initialize from a YAML configuration
        >>> model = YOLO("yolo11n.yaml")
    """

    def __init__(self, model: Union[str, Path] = "yolo11n.pt", task: Optional[str] = None, verbose: bool = False):
        """
        Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types
        (YOLOWorld or YOLOE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo11n.pt', 'yolo11n.yaml'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'.
                Defaults to auto-detection based on model.
            verbose (bool): Display model info on load.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")  # load a pretrained YOLOv11n detection model
            >>> model = YOLO("yolo11n-seg.pt")  # load a pretrained YOLO11n segmentation model
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "-mm" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOMM multi-modal model
            new_instance = YOLOMM(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
                from ultralytics import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """
    YOLO-World object detection model.

    YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions
    without requiring training on specific classes. It extends the YOLO architecture to support real-time
    open-vocabulary detection.

    Attributes:
        model: The loaded YOLO-World model instance.
        task: Always set to 'detect' for object detection.
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOv8-World model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        set_classes: Set the model's class names for detection.

    Examples:
        Load a YOLOv8-World model
        >>> model = YOLOWorld("yolov8s-world.pt")

        Set custom classes for detection
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: Union[str, Path] = "yolov8s-world.pt", verbose: bool = False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: List[str]) -> None:
        """
        Set the model's class names for detection.

        Args:
            classes (List[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """
    YOLOE object detection and segmentation model.

    YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with
    improved performance and additional features like visual and text positional embeddings.

    Attributes:
        model: The loaded YOLOE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text positional embeddings for the given texts.
        get_visual_pe: Get visual positional embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the YOLOE model.
        get_vocab: Get vocabulary for the given class names.
        set_classes: Set the model's class names and embeddings for detection.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE detection model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Set vocabulary and class names
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        Predict with visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(
        self, model: Union[str, Path] = "yoloe-11s-seg.pt", task: Optional[str] = None, verbose: bool = False
    ) -> None:
        """
        Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """
        Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires
        that the model is an instance of YOLOEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = model.model.backbone(img)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: List[str], names: List[str]) -> None:
        """
        Set vocabulary and class names for the YOLOE model.

        This method configures the vocabulary and class names used by the model for text processing and
        classification tasks. The model must be an instance of YOLOEModel.

        Args:
            vocab (List[str]): Vocabulary list containing tokens or words used by the model for text processing.
            names (List[str]): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: List[str], embeddings) -> None:
        """
        Set the model's class names and embeddings for detection.

        Args:
            classes (List[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor): Embeddings corresponding to the classes.
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_classes(classes, embeddings)
        # Verify no background class is present
        assert " " not in classes
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: Optional[str] = None,
        **kwargs,
    ):
        """
        Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: Dict[str, List] = {},
        refer_image=None,
        predictor=None,
        **kwargs,
    ):
        """
        Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths,
                directory paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a
                generator as they are computed.
            visual_prompts (Dict[str, List]): Dictionary containing visual prompts for the model. Must include
                'bboxes' and 'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable, optional): Custom predictor function. If None, a predictor is automatically
                loaded based on the task.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (List | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            if not isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
                self.predictor = (predictor or yolo.yoloe.YOLOEVPDetectPredictor)(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # NOTE: set the first frame as refer image for videos/streams inference
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # reset predictor
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None  # reset predictor if no visual prompts

        return super().predict(source, stream, **kwargs)


class YOLOMM(Model):
    """
    YOLO MultiModal (YOLOMM) object detection model.

    YOLOMM extends the YOLO architecture to support multi-modal input (RGB + X modality) for enhanced
    object detection performance. It supports flexible channel configurations and automatic modality
    routing for RGB, X, and Dual modality inputs.

    Attributes:
        model: The loaded YOLOMM model instance.
        task: The task type (detect, segment, pose, obb).
        overrides: Configuration overrides for the model.
        input_channels: Number of input channels (3 for RGB-only, 6 for RGB+X).
        modality_config: Configuration for supported modalities.

    Methods:
        __init__: Initialize YOLOMM model with multi-modal configuration.
        task_map: Map tasks to their corresponding multi-modal model, trainer, validator, and predictor classes.
        validate_input_channels: Validate input channels against model configuration.
        get_modality_info: Get information about supported modalities.

    Examples:
        Load a YOLOMM detection model
        >>> model = YOLOMM("yolo11n-mm.yaml")

        Load with specific channel configuration
        >>> model = YOLOMM("yolo11n-mm.yaml", ch=6)  # RGB+X modality

        RGB-only mode
        >>> model = YOLOMM("yolo11n-mm.yaml", ch=3)  # RGB-only
    """

    def __init__(self, model: Union[str, Path] = "yolo11n-mm.yaml", task: Optional[str] = None,
                 ch: Optional[int] = None, verbose: bool = False) -> None:
        """
        Initialize YOLOMM multi-modal model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo11n-mm.yaml', 'yolo11n-mm.pt'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment', 'pose', 'obb'.
                Defaults to auto-detection based on model.
            ch (int, optional): Number of input channels. If None, auto-detected from model config.
                Supported values: 3 (RGB-only), 6 (RGB+X).
            verbose (bool): Display model info on load.

        Examples:
            >>> model = YOLOMM("yolo11n-mm.yaml")  # Auto-detect channels
            >>> model = YOLOMM("yolo11n-mm.yaml", ch=6)  # RGB+X modality
            >>> model = YOLOMM("yolo11n-mm.yaml", ch=3)  # RGB-only mode
        """
        # Store multi-modal specific attributes
        self.input_channels = ch
        self.modality_config = {}

        # Initialize base model
        super().__init__(model=model, task=task, verbose=verbose)

        # Validate and configure multi-modal settings
        self._configure_multimodal_settings(verbose)

    def _configure_multimodal_settings(self, verbose: bool = False) -> None:
        """
        Configure multi-modal settings based on model configuration.

        Args:
            verbose (bool): Display configuration info.
        """
        try:
            # Get model configuration
            if hasattr(self.model, 'yaml') and self.model.yaml:
                model_yaml = self.model.yaml

                # Check for multimodal layers in configuration
                has_multimodal_layers = self._detect_multimodal_layers(model_yaml)

                # Determine input channels from model configuration or multimodal detection
                model_channels = model_yaml.get('ch', model_yaml.get('channels', 3))

                # If multimodal layers detected, determine channel count
                if has_multimodal_layers:
                    # Check for Dual modality layers (6 channels)
                    has_dual_layers = self._has_dual_modality_layers(model_yaml)
                    if has_dual_layers:
                        model_channels = 6
                    else:
                        model_channels = 3  # RGB or X only

                # Validate input channels
                if self.input_channels is None:
                    self.input_channels = model_channels
                    if verbose:
                        print(f"Auto-detected input channels: {self.input_channels}")
                elif self.input_channels != model_channels:
                    if verbose:
                        print(f"Warning: Specified channels ({self.input_channels}) differ from model config ({model_channels})")

                # Validate channel configuration
                self.validate_input_channels()

                # Configure modality information based on detected multimodal layers
                if has_multimodal_layers:
                    if self.input_channels == 6:
                        self.modality_config.update({
                            'rgb_channels': [0, 1, 2],
                            'x_channels': [3, 4, 5],
                            'supported_modalities': ['RGB', 'X', 'Dual'],
                            'default_modality': 'Dual'
                        })
                    else:
                        self.modality_config.update({
                            'rgb_channels': [0, 1, 2],
                            'x_channels': [3, 4, 5],
                            'supported_modalities': ['RGB', 'X'],
                            'default_modality': 'RGB'
                        })
                else:
                    self.modality_config.update({
                        'rgb_channels': [0, 1, 2],
                        'x_channels': [],
                        'supported_modalities': ['RGB'],
                        'default_modality': 'RGB'
                    })

                if verbose and self.modality_config:
                    print(f"YOLOMM configured: {self.input_channels} channels, "
                          f"modalities: {self.modality_config.get('supported_modalities', [])}")

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to configure multi-modal settings: {e}")
            # Set default configuration
            self.input_channels = self.input_channels or 3
            self.modality_config = {
                'supported_modalities': ['RGB'],
                'default_modality': 'RGB'
            }

    def _detect_multimodal_layers(self, model_yaml: dict) -> bool:
        """
        Detect if the model configuration contains multimodal layers.

        Args:
            model_yaml (dict): Model YAML configuration

        Returns:
            bool: True if multimodal layers detected
        """
        all_layers = model_yaml.get('backbone', []) + model_yaml.get('head', [])

        for layer_config in all_layers:
            if len(layer_config) >= 5:
                input_source = layer_config[4]
                if input_source in ['RGB', 'X', 'Dual']:
                    return True
        return False

    def _has_dual_modality_layers(self, model_yaml: dict) -> bool:
        """
        Check if the model configuration has Dual modality layers.

        Args:
            model_yaml (dict): Model YAML configuration

        Returns:
            bool: True if Dual modality layers found
        """
        all_layers = model_yaml.get('backbone', []) + model_yaml.get('head', [])

        for layer_config in all_layers:
            if len(layer_config) >= 5:
                input_source = layer_config[4]
                if input_source == 'Dual':
                    return True
        return False

    def validate_input_channels(self) -> None:
        """
        Validate input channels against supported configurations.

        Raises:
            ValueError: If input channels are not supported.
        """
        supported_channels = [3, 6]
        if self.input_channels not in supported_channels:
            raise ValueError(
                f"Unsupported input channels: {self.input_channels}. "
                f"Supported channels: {supported_channels} "
                f"(3=RGB-only, 6=RGB+X)"
            )

    def get_modality_info(self) -> Dict[str, Any]:
        """
        Get information about supported modalities and configuration.

        Returns:
            dict: Modality configuration information.
        """
        return {
            'input_channels': self.input_channels,
            'modality_config': self.modality_config.copy(),
            'model_type': 'YOLOMM',
            'task': getattr(self, 'task', 'detect')
        }

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initialize a new model and inference mode for YOLOMM with flexible channel configuration.

        按配置自动识别任务类型（detect/segment/pose/obb 等），确保分割 YAML 走分割链路，
        同时根据多模态路由规则检测所需输入通道数。

        Args:
            cfg (str): Model configuration file path
            task (str): Task type (auto-detected if None)
            model: Existing model (optional)
            verbose (bool): Verbose output
        """
        from ultralytics.utils import LOGGER
        from ultralytics.nn.tasks import yaml_model_load, guess_model_task

        cfg_dict = yaml_model_load(cfg)

        # 自动判定任务类型（不再强制 detect），确保 *-seg.yaml 走分割链路
        task = task or guess_model_task(cfg_dict)

        # 设置配置和任务
        self.cfg = cfg
        self.task = task

        # MultiModalRouter智能处理输入通道配置：
        # - 早期融合: 配置'Dual'时自动适配6通道输入
        # - 中期融合: 配置'RGB'/'X'时使用3通道，路由器处理模态分发
        # - 通道数由配置文件的Input字段路由系统自动决定

        # Detect required input channels from configuration
        required_channels = self._detect_required_channels(cfg_dict)
        self.model = model or self.task_map[self.task]["model"](cfg_dict, ch=required_channels, verbose=verbose)

        # 设置基本属性
        self.ckpt = None
        self.ckpt_path = None

        # 重要：设置overrides字典，包含train方法需要的"model"和"task"键
        self.overrides = {
            "model": self.cfg,  # 这是train方法需要访问的键
            "task": self.task,
        }

        self.metrics = None
        self.session = None

        # 设置模型属性（与父类保持一致）
        from ultralytics.cfg import DEFAULT_CFG_DICT
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg

        if verbose:
            LOGGER.info(f"YOLOMM model initialized: {cfg} (task={self.task})")

    def _detect_required_channels(self, cfg_dict: dict) -> int:
        """
        Detect required input channels from configuration.

        Args:
            cfg_dict (dict): Model configuration dictionary

        Returns:
            int: Required input channels (3 or 6)
        """
        # Check first layer for Dual modality (6 channels)
        backbone_layers = cfg_dict.get('backbone', [])
        if backbone_layers:
            first_layer = backbone_layers[0]
            if len(first_layer) >= 5 and first_layer[4] == 'Dual':
                return 6

        # Check for any Dual modality layers
        all_layers = backbone_layers + cfg_dict.get('head', [])
        for layer_config in all_layers:
            if len(layer_config) >= 5 and layer_config[4] == 'Dual':
                return 6

        # Default to 3 channels for RGB-only or RGB/X separate paths
        return 3
    
    def cocoval(
        self,
        validator=None,
        **kwargs,
    ):
        """
        使用COCO评估指标对多模态模型进行验证。
        
        这个方法提供了专门的COCO格式验证功能，支持完整的12项COCO指标计算，
        包括不同IoU阈值、不同目标尺寸和不同检测数量限制下的平均精度和召回率。
        
        与标准val()方法的区别：
        - 使用COCO标准评估协议和指标
        - 提供更详细的性能分析（12项指标 vs 4项标准指标）
        - 支持按目标尺寸分析（small/medium/large）
        - 提供标准COCO格式的输出报告
        - 支持JSON格式结果保存
        
        COCO指标说明：
        - AP (IoU=0.50:0.95): 主指标，IoU阈值0.5-0.95平均
        - AP50: IoU阈值0.5时的AP  
        - AP75: IoU阈值0.75时的AP
        - APsmall/APmedium/APlarge: 不同尺寸对象的AP
        - AR1/AR10/AR100: 不同检测限制下的平均召回率
        - ARsmall/ARmedium/ARlarge: 不同尺寸对象的AR
        
        Args:
            validator (MultiModalCOCOValidator, optional): 自定义COCO验证器实例。
                如果为None，将使用默认的MultiModalCOCOValidator。
            **kwargs (Any): 验证配置参数，支持所有标准验证参数：
                data (str): 验证数据集配置文件路径
                imgsz (int): 输入图像尺寸，默认640
                batch_size (int): 批次大小
                conf (float): 置信度阈值
                iou (float): NMS IoU阈值 
                modality (str): 单模态验证时指定模态类型，如'rgb'、'thermal'等
                save_json (bool): 是否保存JSON格式结果，默认True
                save_conf (bool): 是否保存混淆矩阵，默认False
                plots (bool): 是否生成可视化图表，默认True
                verbose (bool): 是否显示详细输出，默认False
        
        Returns:
            (dict): COCO验证指标字典，包含以下键：
                - 'metrics/coco/AP': 主指标mAP@0.5:0.95
                - 'metrics/coco/AP50': mAP@0.5
                - 'metrics/coco/AP75': mAP@0.75  
                - 'metrics/coco/APsmall': 小目标AP
                - 'metrics/coco/APmedium': 中等目标AP
                - 'metrics/coco/APlarge': 大目标AP
                - 'metrics/coco/AR1': AR@1
                - 'metrics/coco/AR10': AR@10
                - 'metrics/coco/AR100': AR@100
                - 'metrics/coco/ARsmall': 小目标AR
                - 'metrics/coco/ARmedium': 中等目标AR
                - 'metrics/coco/ARlarge': 大目标AR
                - 'fitness': 主指标，用于模型选择
                - 'val/speed_*': 速度统计信息
        
        Raises:
            AssertionError: 如果模型不是PyTorch模型
            ImportError: 如果MultiModalCOCOValidator不可用
            ValueError: 如果验证数据集配置无效
        
        Examples:
            基本COCO验证:
            >>> model = YOLOMM('yolo11n-mm.yaml')
            >>> results = model.cocoval(data='coco8.yaml')
            >>> print(f"mAP@0.5:0.95: {results['metrics/coco/AP']:.3f}")
            
            单模态COCO验证:
            >>> results = model.cocoval(data='thermal_dataset.yaml', modality='thermal')
            >>> print(f"Thermal-only mAP: {results['metrics/coco/AP']:.3f}")
            
            详细配置验证:
            >>> results = model.cocoval(
            ...     data='dataset.yaml',
            ...     imgsz=640,
            ...     batch_size=16,
            ...     conf=0.001,
            ...     iou=0.6,
            ...     save_json=True,
            ...     plots=True,
            ...     verbose=True
            ... )
            
            获取特定指标:
            >>> ap50 = results['metrics/coco/AP50']
            >>> ap75 = results['metrics/coco/AP75'] 
            >>> small_ap = results['metrics/coco/APsmall']
            >>> print(f"AP@0.5: {ap50:.3f}, AP@0.75: {ap75:.3f}, Small AP: {small_ap:.3f}")
        
        Notes:
            - COCO验证比标准验证计算更耗时，因为需要计算更多指标
            - 建议在最终模型评估时使用，日常训练验证可使用val()方法
            - 支持所有多模态配置：早期融合、中期融合、单模态等
            - 验证结果会自动保存到runs/val目录下
            - 可通过modality参数进行消融研究，比较不同模态的贡献
        """
        # 检查模型是否为PyTorch模型
        self._check_is_pytorch_model()
        
        # 设置COCO验证的默认参数
        custom = {
            "rect": True,  # 矩形推理以提高效率
            "save_json": kwargs.get("save_json", True),  # 默认保存JSON结果
            "save_conf": kwargs.get("save_conf", False),  # 默认不保存混淆矩阵
            "plots": kwargs.get("plots", True),  # 默认生成可视化图表
            "conf": 0.05,  # 默认置信度阈值为0.05
        }
        
        # 构建验证参数，优先级：用户参数 > 自定义默认值 > 模型覆盖参数
        args = {**self.overrides, **custom, **kwargs, "mode": "cocoval"}  # 设置mode为cocoval
        
        # 创建或使用提供的 COCO 验证器（按 task 分发：detect/segment）
        if validator is None:
            try:
                if self.task == "segment":
                    from ultralytics.models.yolo.multimodal.segment.cocoval import MultiModalSegmentationCOCOValidator

                    validator = MultiModalSegmentationCOCOValidator(
                        dataloader=None, save_dir=None, args=args, _callbacks=self.callbacks
                    )
                elif self.task == "detect":
                    from ultralytics.models.yolo.multimodal.cocoval import MultiModalCOCOValidator

                    # 使用与 val 方法一致的参数传递模式，包括 pbar 参数
                    validator = MultiModalCOCOValidator(
                        dataloader=None, save_dir=None, pbar=None, args=args, _callbacks=self.callbacks
                    )
                else:
                    raise NotImplementedError(
                        f"YOLOMM.cocoval 当前仅支持 detect/segment 任务，当前 task={self.task!r}。"
                    )
            except ImportError as e:
                raise ImportError(
                    "COCO validator is not available. Please ensure the COCO validator module is properly installed.\n"
                    f"Error details: {e}"
                )
        else:
            # 如果提供了验证器，更新其参数
            validator.args = args
            validator.callbacks = self.callbacks
        
        # 执行COCO验证
        # 在验证前统一输出GFLOPs（架构级 + 路由感知），便于对齐多模态实际
        try:
            imgsz = int(args.get("imgsz", 640))
            modality = args.get("modality", None)
            arch_gflops = compute_model_gflops(self.model, imgsz=imgsz, modality=None, route_aware=False)
            route_gflops = compute_model_gflops(self.model, imgsz=imgsz, modality=modality, route_aware=True)
            mod_tag = (modality or 'dual')
            from ultralytics.utils import LOGGER as _LOGGER
            _LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[{mod_tag}]): {route_gflops:.2f}")
        except Exception:
            pass

        # 执行 COCO 验证（返回 dict 结果）
        results = validator(model=self.model)

        # 保存验证指标到模型实例（dict）
        self.metrics = results

        return results

    def vis(self,
            rgb_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
            x_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
            method: str = 'heat',
            layers: Optional[List[int]] = None,
            modality: Optional[str] = None,
            save: bool = True,
            out_dir: Optional[Union[str, Path]] = None,
            device: Optional[str] = None,
            **kwargs) -> Union['VisualizationResult', List['VisualizationResult']]:
        """
        可视化入口（重构版）：委托到家族 Runner（YOLOMMVisualizationRunner）。

        框架阶段仅完成入口与 Fail‑Fast 约束，具体方法插件（heat/feature）将在后续步骤实现。
        旧实现已通过注释形式保留在文件底部常量 LEGACY_YOLOMM_VIS_IMPL 中。
        """
        from ultralytics.models.yolo.multimodal.visualize.runner import YOLOMMVisualizationRunner

        return YOLOMMVisualizationRunner.run(
            model=self.model,
            rgb_source=rgb_source,
            x_source=x_source,
            method=method,
            layers=layers,
            modality=modality,
            save=save,
            out_dir=str(out_dir) if out_dir is not None else None,
            device=device,
            **kwargs,
        )

    # --- 便捷封装：仅设置 method 并转发 ---
    def vis_heat(
        self,
        rgb_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        x_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        layers: Optional[List[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        out_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        return self.vis(
            rgb_source=rgb_source,
            x_source=x_source,
            method='heat',
            layers=layers,
            modality=modality,
            save=save,
            out_dir=out_dir,
            device=device,
            **kwargs,
        )

    def vis_feature(
        self,
        rgb_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        x_source: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        layers: Optional[List[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        out_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        return self.vis(
            rgb_source=rgb_source,
            x_source=x_source,
            method='feature',
            layers=layers,
            modality=modality,
            save=save,
            out_dir=out_dir,
            device=device,
            **kwargs,
        )

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to multi-modal model, trainer, validator, and predictor classes."""
        try:
            # Import multi-modal components (only if available)
            from ultralytics.models.yolo.multimodal import (
                MultiModalDetectionTrainer,
                MultiModalDetectionValidator,
                YOLOMMPredictor,  # 新推理引擎适配器
            )
            # Multi-modal OBB components
            from ultralytics.models.yolo.multimodal.obb import (
                MultiModalOBBTrainer,
                MultiModalOBBValidator,
                MultiModalOBBPredictor,
            )
            # Multi-modal segmentation components
            from ultralytics.models.yolo.multimodal.segment import (
                MultiModalSegmentationTrainer,
                MultiModalSegmentationValidator,
                MultiModalSegmentationPredictor,
            )
            # Import COCO validator for multi-modal models
            from ultralytics.models.yolo.multimodal.cocoval import MultiModalCOCOValidator

            # Multi-modal task mapping
            multimodal_task_map = {
                "detect": {
                    "model": DetectionModel,  # Use standard DetectionModel with multi-modal routing
                    "trainer": MultiModalDetectionTrainer,
                    "validator": MultiModalDetectionValidator,
                    "predictor": YOLOMMPredictor,  # 新推理引擎适配器
                },
                "obb": {
                    "model": OBBModel,  # OBB 头+多模态路由
                    "trainer": MultiModalOBBTrainer,
                    "validator": MultiModalOBBValidator,
                    "predictor": MultiModalOBBPredictor,
                },
                "segment": {
                    "model": SegmentationModel,  # Seg task with multi-modal routing
                    "trainer": MultiModalSegmentationTrainer,
                    "validator": MultiModalSegmentationValidator,
                    "predictor": MultiModalSegmentationPredictor,
                },
                "cocoval": {
                    "model": DetectionModel,  # Use standard DetectionModel with multi-modal routing
                    "trainer": MultiModalDetectionTrainer,  # 保持一致性，虽然COCO验证不需要训练器
                    "validator": MultiModalCOCOValidator,  # 使用COCO验证器
                    "predictor": YOLOMMPredictor,  # 新推理引擎适配器
                },
                # Note: Other tasks (segment, pose, obb) can be added when multi-modal versions are available
            }

            # For now, only support detection task in multi-modal mode
            # Other tasks fall back to standard YOLO components
            standard_task_map = {
                "classify": {
                    "model": ClassificationModel,
                    "trainer": yolo.classify.ClassificationTrainer,
                    "validator": yolo.classify.ClassificationValidator,
                    "predictor": yolo.classify.ClassificationPredictor,
                },
                "segment": {
                    "model": SegmentationModel,
                    "trainer": yolo.segment.SegmentationTrainer,
                    "validator": yolo.segment.SegmentationValidator,
                    "predictor": yolo.segment.SegmentationPredictor,
                },
                "pose": {
                    "model": PoseModel,
                    "trainer": yolo.pose.PoseTrainer,
                    "validator": yolo.pose.PoseValidator,
                    "predictor": yolo.pose.PosePredictor,
                },
                "obb": {
                    "model": OBBModel,
                    "trainer": yolo.obb.OBBTrainer,
                    "validator": yolo.obb.OBBValidator,
                    "predictor": yolo.obb.OBBPredictor,
                },
            }

            # Merge multi-modal and standard task maps
            task_map = {**standard_task_map, **multimodal_task_map}
            return task_map

        except ImportError as e:
            # If multi-modal components are not available, fall back to standard YOLO
            print(f"Warning: Multi-modal components not available ({e}), using standard YOLO components")
            return {
                "classify": {
                    "model": ClassificationModel,
                    "trainer": yolo.classify.ClassificationTrainer,
                    "validator": yolo.classify.ClassificationValidator,
                    "predictor": yolo.classify.ClassificationPredictor,
                },
                "detect": {
                    "model": DetectionModel,
                    "trainer": yolo.detect.DetectionTrainer,
                    "validator": yolo.detect.DetectionValidator,
                    "predictor": yolo.detect.DetectionPredictor,
                },
                "segment": {
                    "model": SegmentationModel,
                    "trainer": yolo.segment.SegmentationTrainer,
                    "validator": yolo.segment.SegmentationValidator,
                    "predictor": yolo.segment.SegmentationPredictor,
                },
                "pose": {
                    "model": PoseModel,
                    "trainer": yolo.pose.PoseTrainer,
                    "validator": yolo.pose.PoseValidator,
                    "predictor": yolo.pose.PosePredictor,
                },
                "obb": {
                    "model": OBBModel,
                    "trainer": yolo.obb.OBBTrainer,
                    "validator": yolo.obb.OBBValidator,
                    "predictor": yolo.obb.OBBPredictor,
                },
            }

# ------------------------------
# Legacy vis implementation (commented for refactor history)
# ------------------------------
# The previous implementation of YOLOMM.vis performed:
# - method alias mapping {'heat'|'heatmap','feature'|'feature_map'}
# - strict layers validation with custom exceptions (LayerNotSpecifiedError, etc.)
# - modality auto-inference/conflict checks (dual/rgb/x)
# - device consistency check without auto-switch
# - delegation to VisualizationPipeline(self.model) with project/name dispatch to runs/visualize/yolo
# - alg forwarding for heatmap
# Kept here as high-level reference to avoid hard deletion per refactor guideline.
