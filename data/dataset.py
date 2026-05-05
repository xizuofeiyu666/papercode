# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Return dictionary of labels for YOLO training.
        build_transforms: Build and append transforms to the list.
        close_mosaic: Set mosaic, copy_paste and mixup options to 0.0 and build transformations.
        update_labels_info: Update label format for different tasks.
        collate_fn: Collate data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data["channels"], **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict:
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Add text information for multi-modal model training.
        build_transforms: Enhance data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Add text information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Enhance data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (Set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOMultiModalImageDataset(YOLODataset):
    """
    多模态YOLO数据集类 - 支持RGB+X模态图像的目标检测数据集

    此类扩展了标准的YOLODataset，添加了对多模态图像（RGB+X）的支持。
    支持的X模态包括：深度图、热红外图、LiDAR点云图等。

    核心特性：
    - RGB+X双模态图像加载和处理
    - 6通道图像构建（RGB 3通道 + X模态 3通道）
    - 智能多模态缓存系统（内存/磁盘）
    - 自体模态生成支持（当X模态缺失时）
    - 与多模态数据增强的无缝集成
    - 高效的索引管理和验证

    数据组织结构：
    ```
    dataset/
    ├── images/
    │   ├── rgb/          # RGB图像目录
    │   └── depth/        # X模态图像目录（如深度图）
    └── labels/           # 标注文件目录
    ```

    Attributes:
        x_modality (str): X模态类型，如'depth', 'thermal', 'lidar'等
        x_modality_dir (str): X模态图像目录名
        x_modality_suffix (str): X模态图像文件后缀
        enable_self_modal_generation (bool): 是否启用自体模态生成
        x_ims (list): X模态图像缓存列表
        x_im_hw0 (list): X模态图像原始尺寸缓存
        x_im_hw (list): X模态图像调整后尺寸缓存

    Methods:
        __getitem__: 获取多模态数据样本（6通道图像+标签）
        get_valid_indices: 获取有完整多模态数据的有效索引
        load_multimodal_image: 加载RGB+X模态图像并构建6通道图像
        cache_images: 缓存RGB和X模态图像到内存或磁盘

    Examples:
        >>> # 基本使用
        >>> dataset = YOLOMultiModalImageDataset(
        ...     img_path="dataset/images/rgb",
        ...     x_modality="depth",
        ...     x_modality_dir="depth"
        ... )
        >>> sample = dataset[0]  # 返回6通道图像和标签
        >>> print(sample["img"].shape)  # (6, H, W) - RGB+Depth

        >>> # 启用自体模态生成
        >>> dataset = YOLOMultiModalImageDataset(
        ...     img_path="dataset/images/rgb",
        ...     x_modality="thermal",
        ...     enable_self_modal_generation=True
        ... )
    """

    def __init__(
        self,
        *args,
        x_modality="depth",
        x_modality_dir=None,
        x_modality_suffix=None,
        enable_self_modal_generation=False,
        **kwargs
    ):
        """
        初始化多模态YOLO数据集

        Args:
            x_modality (str): X模态类型，如'depth', 'thermal', 'lidar'等
            x_modality_dir (str, optional): X模态图像目录名，默认与x_modality相同
            x_modality_suffix (str, optional): X模态图像文件后缀，默认与RGB相同
            enable_self_modal_generation (bool): 是否启用自体模态生成
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(
            ...     img_path="images/rgb",
            ...     x_modality="depth",
            ...     x_modality_dir="depth"
            ... )
        """
        # 设置X模态相关属性
        self.x_modality = x_modality
        self.x_modality_dir = x_modality_dir or x_modality
        self.x_modality_suffix = x_modality_suffix
        self.enable_self_modal_generation = enable_self_modal_generation

        # 初始化多模态缓存
        self.x_ims = None
        self.x_im_hw0 = None
        self.x_im_hw = None

        # 调用父类初始化
        super().__init__(*args, **kwargs)

        # 验证多模态数据完整性
        self._validate_multimodal_data()

        # 记录多模态配置信息（移除非对/错/警告类表情符号）
        LOGGER.info(f"MultiModal: RGB+{self.x_modality}双模态数据集已初始化")
        LOGGER.info(f"MultiModal: 总样本数={self.ni}, X模态目录='{self.x_modality_dir}'")
        if self.enable_self_modal_generation:
            LOGGER.info("MultiModal: 自体模态生成已启用")

    def _validate_multimodal_data(self):
        """验证多模态数据的完整性和一致性"""
        if not hasattr(self, 'im_files') or not self.im_files:
            LOGGER.warning("RGB图像文件列表为空，无法验证多模态数据")
            return

        # 检查X模态目录是否存在（使用与路径构建一致的逻辑）
        rgb_path = Path(self.im_files[0])
        dataset_root = rgb_path.parent.parent.parent  # /dataset
        split_dir = rgb_path.parent.name              # train/val/test
        x_dir = dataset_root / self.x_modality_dir / split_dir

        if not x_dir.exists():
            if self.enable_self_modal_generation:
                LOGGER.info(f"X模态目录 {x_dir} 不存在，将使用自体模态生成")
            else:
                LOGGER.debug(f"X模态目录 {x_dir} 不存在，将尝试从其他位置加载或使用自体生成")

        # 统计有效的多模态样本数量
        valid_count = 0
        sample_size = min(len(self.im_files), 100)  # 采样检查前100个

        for i in range(sample_size):
            try:
                rgb_path = self.im_files[i]
                x_path = self._find_corresponding_x_image(rgb_path)
                if Path(x_path).exists() or self.enable_self_modal_generation:
                    valid_count += 1
            except Exception:
                continue

        valid_ratio = valid_count / sample_size if sample_size > 0 else 0
        LOGGER.info(f"MultiModal: 多模态数据完整性检查 - {valid_ratio:.1%} ({valid_count}/{sample_size})")

        if valid_ratio < 0.5 and not self.enable_self_modal_generation:
            LOGGER.warning("多模态数据完整性较低，建议启用自体模态生成或检查数据路径")

    def __getitem__(self, index):
        """
        获取多模态数据样本

        Args:
            index (int): 样本索引

        Returns:
            dict: 包含6通道多模态图像和标签的字典
                - 'img': 6通道图像张量 [6, H, W] (RGB+X)
                - 其他标签信息（bbox, cls等）

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> sample = dataset[0]
            >>> img = sample['img']  # 6通道图像
            >>> print(img.shape)  # torch.Size([6, H, W])
        """
        # 获取基础标签信息（使用父类方法）
        label = self.get_image_and_label(index)

        # 加载多模态图像（替换单模态RGB图像）
        multimodal_img = self.load_multimodal_image(index)
        label["img"] = multimodal_img

        # 应用数据变换
        return self.transforms(label)

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        构建并返回多模态专用增强流水线（独立于通用 v8_transforms）。

        说明：
        - 多模态数据集一律走独立 mm_transforms() 链路，可复用 v8 组件；
        - 单/双模态的路由与消融由路由层负责，数据层不做区分。
        """
        from ultralytics.data.multimodal_augment import mm_transforms, mm_seg_transforms
        from ultralytics.data.augment import Compose, LetterBox, Format

        if self.augment:
            # 强制走多模态增强链；分割任务走 mm_seg_transforms
            if getattr(self, "use_segments", False):
                transforms = mm_seg_transforms(self, self.imgsz, hyp)
            else:
                transforms = mm_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        # 统一追加 Format；多模态训练不使用 bgr 概率翻转
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,  # 禁用 BGR 随机翻转，避免触动 X 通道
            )
        )
        return transforms

    def get_valid_indices(self):
        """
        获取有完整多模态数据的有效索引列表

        此方法用于与MultiModalMosaic和MultiModalMixUp数据增强兼容，
        确保增强过程中选择的图像都有完整的RGB+X模态数据。

        Returns:
            list: 有效索引列表

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> valid_indices = dataset.get_valid_indices()
            >>> print(f"有效样本数: {len(valid_indices)}")
        """
        if not hasattr(self, '_valid_indices_cache'):
            self._valid_indices_cache = []

            # 仅在主进程/主 worker 打印日志，避免多 worker 噪声
            log_enabled = True
            try:
                from torch.utils.data import get_worker_info
                wi = get_worker_info()
                log_enabled = (wi is None) or (wi.id == 0)
            except Exception:
                log_enabled = True

            if log_enabled:
                LOGGER.info("MultiModal: 正在扫描有效的多模态索引...")

            for i in range(self.ni):
                try:
                    # 检查RGB图像是否存在
                    rgb_path = self.im_files[i]
                    if not Path(rgb_path).exists():
                        continue

                    # 检查X模态图像是否存在或可生成
                    x_path = self._find_corresponding_x_image(rgb_path)
                    if Path(x_path).exists() or self.enable_self_modal_generation:
                        self._valid_indices_cache.append(i)

                except Exception as e:
                    LOGGER.debug(f"索引 {i} 验证失败: {e}")
                    continue

            valid_count = len(self._valid_indices_cache)
            valid_ratio = valid_count / self.ni if self.ni > 0 else 0

            if log_enabled:
                LOGGER.info(f"✅ MultiModal: 发现 {valid_count}/{self.ni} ({valid_ratio:.1%}) 个有效多模态样本")

                if valid_count == 0:
                    LOGGER.error("未发现任何有效的多模态样本，请检查数据路径和配置")
                elif valid_ratio < 0.8:
                    LOGGER.warning(f"有效多模态样本比例较低 ({valid_ratio:.1%})，建议检查数据完整性")

        return self._valid_indices_cache

    def load_multimodal_image(self, i):
        """
        加载RGB+X模态图像并构建6通道图像

        Args:
            i (int): 图像索引

        Returns:
            np.ndarray: 6通道图像数组 [H, W, 6] (RGB+X)

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> img_6ch = dataset.load_multimodal_image(0)
            >>> print(img_6ch.shape)  # (H, W, 6)
        """
        # 加载RGB图像（使用父类方法）
        rgb_img, (h0, w0), (h, w) = self.load_image(i)

        # 预先记录X模态候选路径（用于日志）
        x_path_hint = None
        try:
            x_path_hint = self._find_corresponding_x_image(self.im_files[i])
        except Exception:
            pass

        # 加载X模态图像
        try:
            x_img, x_hw0, x_hw = self._load_x_image_cached(i, target_shape=(h, w))
        except Exception as e:
            if self.enable_self_modal_generation:
                LOGGER.debug(f"X模态图像加载失败，使用自体生成: {e}")
                x_img = self._generate_self_modal(rgb_img)
            else:
                LOGGER.error(f"无法加载X模态图像 {i}: {e}")
                raise

        # 确保X模态图像尺寸与RGB匹配
        if x_img.shape[:2] != rgb_img.shape[:2]:
            x_img = cv2.resize(x_img, (rgb_img.shape[1], rgb_img.shape[0]))

        # 获取期望的X模态通道数
        expected_xch = getattr(self, 'data', {}).get('Xch', 3)
        
        # 处理X模态图像通道数（在启用多模态增强时，严格要求与 data['Xch'] 一致，杜绝静默扩展）
        # 实际通道（未整形前）
        actual_xch = 1 if len(x_img.shape) == 2 else (x_img.shape[2] if len(x_img.shape) == 3 else 1)
        use_mm_aug = bool(getattr(self, 'hyp', None) and getattr(self.hyp, 'use_multimodal_aug', False))
        if use_mm_aug and actual_xch != expected_xch:
            raise ValueError(
                f"MultiModal: X 通道不一致，data.yaml Xch={expected_xch}，但实际为 {actual_xch}。"
                f" 文件: {x_path_hint or 'unknown'}；请修正数据或配置（禁用静默通道扩展）。"
            )

        # 常规整形（非严格模式下允许），严格模式已在上方阻断
        if len(x_img.shape) == 2:
            # 灰度图像
            if expected_xch == 1:
                x_img = x_img[:, :, np.newaxis]  # 保持1通道
            else:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)  # 转换为3通道
        elif x_img.shape[2] == 1:
            # 单通道图像
            if expected_xch == 1:
                pass  # 保持1通道
            else:
                x_img = np.repeat(x_img, 3, axis=2)  # 扩展为3通道
        elif x_img.shape[2] == 4:
            # RGBA图像
            x_img = x_img[:, :, :3]  # 移除alpha通道
            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        elif x_img.shape[2] == 3:
            # RGB图像
            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            else:
                pass  # 保持3通道
        
        # 验证X模态通道数
        if x_img.shape[2] != expected_xch:
            LOGGER.warning(
                f"X模态图像通道数({x_img.shape[2]})与配置Xch={expected_xch}不匹配，已自动调整; 文件: {x_path_hint or 'unknown'}"
            )

        # 构建多通道图像：RGB(前3通道) + X(后Xch通道)
        multimodal_img = np.concatenate([rgb_img, x_img], axis=2)

        return multimodal_img

    def _find_corresponding_x_image(self, rgb_path):
        """
        根据RGB图像路径找到对应的X模态图像路径

        Args:
            rgb_path (str): RGB图像路径

        Returns:
            str: X模态图像路径

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(x_modality_dir="images_ir")
            >>> rgb_path = "/dataset/images/train/img001.jpg"
            >>> x_path = dataset._find_corresponding_x_image(rgb_path)
            >>> print(x_path)  # "/dataset/images_ir/train/img001.jpg"
        """
        rgb_path = Path(rgb_path)

        # 构建X模态图像路径
        # 保持相同的子目录结构 (train/val/test)，只替换基础模态目录
        # RGB路径: /dataset/images/train/img.jpg
        # X路径:   /dataset/images_ir/train/img.jpg
        dataset_root = rgb_path.parent.parent.parent  # /dataset (跳过images和train)
        split_dir = rgb_path.parent.name              # train/val/test
        x_dir = dataset_root / self.x_modality_dir / split_dir

        # 处理文件后缀
        if self.x_modality_suffix:
            x_filename = rgb_path.stem + self.x_modality_suffix + rgb_path.suffix
        else:
            x_filename = rgb_path.name

        x_path = x_dir / x_filename

        # 如果指定路径不存在，尝试常见的文件扩展名（包含npy/npz特征文件）
        if not x_path.exists():
            common_extensions = ['.npy', '.npz', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            for ext in common_extensions:
                test_path = x_dir / (rgb_path.stem + ext)
                if test_path.exists():
                    return str(test_path)

        return str(x_path)

    def _load_x_modality(self, x_path):
        """
        加载X模态图像

        Args:
            x_path (str): X模态图像路径

        Returns:
            np.ndarray: X模态图像数组

        Raises:
            FileNotFoundError: 当X模态图像文件不存在时
        """
        x_path = Path(x_path)

        if not x_path.exists():
            raise FileNotFoundError(f"X模态图像不存在: {x_path}")

        # 根据文件扩展名选择加载方式
        if x_path.suffix.lower() in ['.npy']:
            # NumPy数组文件
            x_img = np.load(x_path)
        elif x_path.suffix.lower() in ['.npz']:
            # 多数组压缩文件，按约定顺序择优取出
            with np.load(x_path) as npz_file:
                preferred_keys = ('image', 'arr_0', 'array', 'data')
                selected_key = next((k for k in preferred_keys if k in npz_file.files), None)

                if selected_key is None:
                    if len(npz_file.files) == 1:
                        selected_key = npz_file.files[0]
                    else:
                        raise ValueError(
                            f"npz文件 {x_path} 含多个数组 {npz_file.files}，无法确定默认键，请显式使用标准键(image/arr_0)。"
                        )

                x_img = npz_file[selected_key]
        elif x_path.suffix.lower() in ['.tiff', '.tif']:
            # TIFF文件（常用于深度图）
            x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)
        else:
            # 标准图像文件
            x_img = cv2.imread(str(x_path))

        if x_img is None:
            raise ValueError(f"无法读取X模态图像: {x_path}")

        return x_img

    def _generate_self_modal(self, rgb_img):
        """
        生成自体模态图像（当X模态图像缺失时）

        Args:
            rgb_img (np.ndarray): RGB图像

        Returns:
            np.ndarray: 生成的X模态图像
        """
        # 简单的自体模态生成策略：
        # 1. 转换为灰度图
        # 2. 应用边缘检测
        # 3. 转换回3通道

        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊和边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 转换为3通道
        x_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return x_img

    def cache_images(self, cache=None):
        """
        缓存RGB和X模态图像到内存或磁盘

        Args:
            cache (str, bool, or None): 缓存模式
                - True or 'ram': 缓存到内存
                - 'disk': 缓存到磁盘
                - False: 不缓存
                - None: 使用self.cache属性

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> dataset.cache_images('ram')  # 缓存到内存
            >>> dataset.cache_images('disk')  # 缓存到磁盘
            >>> dataset.cache_images()  # 使用self.cache属性
        """
        # Use parameter if provided, otherwise fall back to instance attribute
        cache_mode = cache if cache is not None else self.cache

        # 如果不缓存，初始化空缓存
        if not cache_mode:
            self.x_ims = [None] * self.ni
            self.x_im_hw0 = [None] * self.ni
            self.x_im_hw = [None] * self.ni
            return

        # 传统缓存模式
        # 初始化多模态缓存存储
        self.x_ims = [None] * self.ni
        self.x_im_hw0 = [None] * self.ni
        self.x_im_hw = [None] * self.ni

        # 调用父类方法缓存RGB图像
        super().cache_images(cache_mode)

        # 缓存X模态图像
        if cache_mode:
            self._cache_x_modality_images(cache_mode)

    def _cache_x_modality_images(self, cache):
        """缓存X模态图像"""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self._cache_x_images_to_disk if cache == "disk" else self._load_x_image_for_cache

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    # 磁盘缓存时，x是文件路径，计算文件大小
                    x_npy_file = self._get_x_npy_file(i)
                    if x_npy_file.exists():
                        b += x_npy_file.stat().st_size
                else:  # 'ram'
                    if x is not None:
                        self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i] = x
                        b += self.x_ims[i].nbytes if self.x_ims[i] is not None else 0
                pbar.desc = f"{self.prefix}Caching {self.x_modality} images ({b / gb:.1f}GB {cache})"
            pbar.close()

    def _load_x_image_for_cache(self, i):
        """为缓存加载X模态图像"""
        try:
            # 获取RGB图像路径并找到对应的X模态图像
            rgb_path = self.im_files[i]
            x_path = self._find_corresponding_x_image(rgb_path)

            # 加载X模态图像
            x_img = self._load_x_modality(x_path)

            # 获取RGB图像信息用于尺寸匹配
            if self.ims[i] is not None:
                # 如果RGB已缓存，使用其尺寸
                rgb_img = self.ims[i]
                h0, w0 = self.im_hw0[i]
                target_shape = rgb_img.shape[:2]
            else:
                # 如果RGB未缓存，先加载RGB获取目标尺寸
                rgb_img, (h0, w0), target_shape = self.load_image(i)

            # 调整X模态图像尺寸以匹配RGB
            if x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))

            return x_img, (h0, w0), x_img.shape[:2]

        except FileNotFoundError as e:
            LOGGER.warning(f"无法加载X模态图像 {i}: {e}")
            return None, None, None

    def _cache_x_images_to_disk(self, i):
        """将X模态图像保存为.npy文件到磁盘"""
        x_npy_file = self._get_x_npy_file(i)
        if not x_npy_file.exists():
            try:
                # 加载X模态图像数据
                result = self._load_x_image_for_cache(i)
                if result[0] is not None:
                    x_img, _, _ = result
                    np.save(x_npy_file.as_posix(), x_img, allow_pickle=False)
            except Exception as e:
                LOGGER.warning(f"无法缓存X模态图像到磁盘 {i}: {e}")

    def _get_x_npy_file(self, i):
        """获取X模态图像的.npy缓存文件路径"""
        rgb_npy_file = self.npy_files[i]
        # 将RGB的.npy文件名修改为X模态的
        x_npy_file = rgb_npy_file.parent / f"{rgb_npy_file.stem}_{self.x_modality}.npy"
        return x_npy_file

    def load_image(self, i, rect_mode=True):
        """重写load_image方法，支持多模态缓存加载"""
        # 加载RGB图像（使用父类方法）
        rgb_img, ori_shape, resized_shape = super().load_image(i, rect_mode)
        return rgb_img, ori_shape, resized_shape

    def _load_x_image_cached(self, i, target_shape=None):
        """从缓存或磁盘加载X模态图像"""
        # 检查内存缓存
        if self.x_ims is not None and self.x_ims[i] is not None:
            return self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i]

        # 检查磁盘缓存
        x_npy_file = self._get_x_npy_file(i)
        if x_npy_file.exists():
            try:
                x_img = np.load(x_npy_file)
                h0, w0 = x_img.shape[:2]  # 假设缓存时已调整尺寸
                return x_img, (h0, w0), x_img.shape[:2]
            except Exception as e:
                LOGGER.warning(f"加载X模态缓存文件失败 {x_npy_file}: {e}")

        # 从原始文件加载
        try:
            rgb_path = self.im_files[i]
            x_path = self._find_corresponding_x_image(rgb_path)
            x_img = self._load_x_modality(x_path)

            # 如果提供了目标尺寸，调整X模态图像尺寸
            if target_shape and x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))

            h0, w0 = x_img.shape[:2]
            return x_img, (h0, w0), x_img.shape[:2]

        except FileNotFoundError as e:
            LOGGER.error(f"无法加载X模态图像 {i}: {e}")
            raise

    def check_cache_ram(self, safety_margin=0.5):
        """检查多模态图像缓存的内存需求"""
        # 检查RGB图像内存需求（使用父类方法）
        rgb_cache_ok = super().check_cache_ram(safety_margin)

        # 检查X模态图像内存需求
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images

        for _ in range(n):
            try:
                # 随机选择一个样本估算X模态图像大小
                rgb_path = random.choice(self.im_files)
                x_path = self._find_corresponding_x_image(rgb_path)
                x_img = self._load_x_modality(x_path)

                ratio = self.imgsz / max(x_img.shape[0], x_img.shape[1])
                b += x_img.nbytes * ratio**2
            except Exception:
                # 如果加载失败，使用RGB图像大小作为估算
                rgb_img = cv2.imread(random.choice(self.im_files))
                if rgb_img is not None:
                    ratio = self.imgsz / max(rgb_img.shape[0], rgb_img.shape[1])
                    b += rgb_img.nbytes * ratio**2

        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache X modality
        mem = psutil.virtual_memory()
        x_cache_ok = mem_required < mem.available

        if not x_cache_ok:
            LOGGER.info(
                f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache {self.x_modality} images '
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if x_cache_ok else f'not caching {self.x_modality} images ⚠️'}"
            )

        # 只有当RGB和X模态都能缓存时才返回True
        return rgb_cache_ok and x_cache_ok


class GroundingDataset(YOLODataset):
    """
    Dataset class for object detection tasks using annotations from a JSON file in grounding format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Return empty list as image files are read in get_labels.
        get_labels: Load annotations from a JSON file and prepare them for training.
        build_transforms: Configure augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            max_samples (int): Maximum number of samples to load for text augmentation.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> List:
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels: List[Dict[str, Any]]) -> None:
        """
        Verify the number of instances in the dataset matches expected counts.

        This method checks if the total number of bounding box instances in the provided
        labels matches the expected count for known datasets. It performs validation
        against a predefined set of datasets with known instance counts.

        Args:
            labels (List[Dict[str, Any]]): List of label dictionaries, where each dictionary
                contains dataset annotations. Each label dict must have a 'bboxes' key with
                a numpy array or tensor containing bounding box coordinates.

        Raises:
            AssertionError: If the actual instance count doesn't match the expected count
                for a recognized dataset.

        Note:
            For unrecognized datasets (those not in the predefined expected_counts),
            a warning is logged and verification is skipped.
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (Dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Configure augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Set mosaic, copy_paste and mixup options to 0.0 and build transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Dataset class for image classification tasks extending torchvision ImageFolder functionality.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Return subset of data and targets corresponding to given indices.
        __len__: Return the total number of samples in the dataset.
        verify_images: Verify all images in dataset.
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """
        Initialize YOLO classification dataset with root directory, arguments, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> Dict:
        """
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self) -> List[Tuple]:
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
