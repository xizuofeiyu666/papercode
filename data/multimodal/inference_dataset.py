# Ultralytics Multimodal Inference - Inference Dataset
# Dedicated dataset for multimodal inference (images only)
# Version: v1.0
# Date: 2026-01-13

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Union
from ultralytics.utils import LOGGER
from ultralytics.data.augment import LetterBox
from .utils import align_and_validate_x, letterbox_with_ratio_pad, to_tensor_rgb, to_tensor_x


class MultiModalInferenceDataset:
    """
    多模态推理专用数据集

    职责：
    - 从PairingResolver提供的MultiModalSampleSpec列表构建可迭代样本流
    - 加载RGB/X为numpy，并构造MultiModalSample
    - 严格校验：存在性、可读性、X通道数一致性
    - 统一预处理：LetterBox + 转tensor + 归一化

    设计原则：
    - 推理模式：不走label cache，不要求labels/目录存在
    - 严格输入契约：不做自动填充/降级
    - Xch任意：动态适配Router配置的通道数
    """

    def __init__(
        self,
        samples: List[Dict[str, Union[str, Path]]],
        imgsz: Union[int, tuple],
        dataset_config: Dict,
        stride: int = 32,
        pad: float = 0.5,
        verbose: bool = True
    ):
        """
        初始化推理数据集

        Args:
            samples: MultiModalSampleSpec列表（由PairingResolver提供）
            imgsz: 推理输入尺寸（int或(H,W)）
            dataset_config: 数据集配置，必须包含 'Xch' 和 'x_modality'
            stride: 模型stride（用于LetterBox对齐）
            pad: LetterBox padding比例
            verbose: 是否输出详细日志
        """
        self.samples = samples
        self.ni = len(samples)
        self.verbose = verbose

        # 推理必要配置
        self.Xch = int(dataset_config.get('Xch', 3))
        self.x_modality = str(dataset_config.get('x_modality', 'unknown'))

        # 图像尺寸配置
        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        else:
            self.imgsz = tuple(imgsz)

        self.stride = stride
        self.pad = pad

        # LetterBox配置（推理模式：不缩放放大、居中对齐）
        self.letterbox = LetterBox(
            new_shape=self.imgsz,
            auto=False,
            scale_fill=False,
            scaleup=False,  # 推理不放大
            center=True,
            stride=self.stride
        )

        # 提取文件路径列表（用于日志和索引）
        self.rgb_files = [str(s['rgb_path']) for s in samples]
        self.x_files = [str(s['x_path']) for s in samples]

        if self.verbose:
            LOGGER.info(f"MultiModalInferenceDataset 初始化完成:")
            LOGGER.info(f"  样本数: {self.ni}")
            LOGGER.info(f"  推理尺寸: {self.imgsz}")
            LOGGER.info(f"  X模态类型: {self.x_modality}")
            LOGGER.info(f"  X模态通道数: {self.Xch}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.ni

    def __getitem__(self, index: int) -> Dict:
        """
        获取单个多模态样本（支持单模态零填充）

        Args:
            index: 样本索引

        Returns:
            MultiModalSample字典，包含：
                - id: 样本ID
                - paths: {"rgb": Path or None, "x": Path or None}
                - orig_imgs: {"rgb": np.ndarray or None, "x": np.ndarray or None}
                - meta: {x_modality, xch, ori_shape, imgsz, ratio_pad}
                - im: tensor [1, 3+Xch, H, W]
        """
        sample_spec = self.samples[index]

        # 处理RGB模态
        rgb_path = sample_spec['rgb_path']
        if rgb_path is not None:
            # 加载RGB原图（BGR格式）
            rgb0 = cv2.imread(str(rgb_path))
            if rgb0 is None:
                raise ValueError(f"无法读取RGB图像: {rgb_path}")

            # LetterBox处理
            rgb_lb, ratio_pad = letterbox_with_ratio_pad(self.letterbox, rgb0)
            # Tensor化: BGR->RGB + 归一化
            rgb_t = to_tensor_rgb(rgb_lb)  # [3, H, W]
            ori_shape = rgb0.shape[:2]  # (H, W)
        else:
            # RGB缺失：使用零填充
            H, W = self.imgsz
            rgb_t = torch.zeros([3, H, W], dtype=torch.float32)
            rgb0 = None
            ori_shape = (H, W)
            ratio_pad = (1.0, (0.0, 0.0))  # 无缩放无padding

        # 处理X模态
        x_path = sample_spec['x_path']
        if x_path is not None:
            # 加载X模态原图
            x0 = self._load_x_modality(x_path)

            # 空间对齐 + 通道数校验（如果RGB存在则对齐，否则直接使用）
            if rgb0 is not None:
                x0 = align_and_validate_x(x0, rgb0, expected_xch=self.Xch)
                x_lb, _ = letterbox_with_ratio_pad(self.letterbox, x0)
            else:
                # RGB缺失时，X模态独立处理
                if x0.ndim == 2:
                    x0 = np.expand_dims(x0, axis=2)
                if x0.shape[2] != self.Xch:
                    raise ValueError(f"X模态通道数不匹配: 期望{self.Xch}, 实际{x0.shape[2]}")
                x_lb, ratio_pad = letterbox_with_ratio_pad(self.letterbox, x0)
                ori_shape = x0.shape[:2]

            # Tensor化: 不翻转通道 + 归一化
            x_t = to_tensor_x(x_lb)  # [Xch, H, W]
        else:
            # X模态缺失：使用零填充
            H, W = self.imgsz
            x_t = torch.zeros([self.Xch, H, W], dtype=torch.float32)
            x0 = None

        # 拼接为多模态输入: [RGB, X]
        im = torch.cat([rgb_t, x_t], dim=0).unsqueeze(0)  # [1, 3+Xch, H, W]

        # 构造MultiModalSample
        imgsz_hw = tuple(im.shape[2:4])  # (H, W)

        return {
            "id": sample_spec['id'],
            "paths": {
                "rgb": rgb_path,
                "x": x_path
            },
            "orig_imgs": {
                "rgb": rgb0,  # BGR格式原图或None
                "x": x0  # 对齐后的X模态原图或None
            },
            "meta": {
                "x_modality": self.x_modality,
                "xch": self.Xch,
                "ori_shape": ori_shape,  # (H, W)
                "imgsz": imgsz_hw,  # (H, W)
                "ratio_pad": ratio_pad  # (gain, (padw, padh))
            },
            "im": im  # [1, 3+Xch, H, W]
        }

    def _load_x_modality(self, x_path: Path) -> np.ndarray:
        """
        加载X模态图像（支持多种格式）

        Args:
            x_path: X模态图像路径

        Returns:
            X模态图像数组（H×W 或 H×W×C）

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 无法读取文件
        """
        if not x_path.exists():
            raise FileNotFoundError(f"X模态文件不存在: {x_path}")

        # 根据扩展名选择加载方式
        suffix = x_path.suffix.lower()

        try:
            if suffix == '.npy':
                # NumPy数组文件
                x_img = np.load(x_path)

            elif suffix == '.npz':
                # 压缩NumPy数组文件
                with np.load(x_path) as npz_file:
                    # 按优先级查找键
                    preferred_keys = ('image', 'arr_0', 'array', 'data')
                    selected_key = next(
                        (k for k in preferred_keys if k in npz_file.files),
                        None
                    )

                    if selected_key is None:
                        if len(npz_file.files) == 1:
                            selected_key = npz_file.files[0]
                        else:
                            raise ValueError(
                                f"npz文件含多个数组 {npz_file.files}，"
                                f"无法确定默认键，请使用标准键(image/arr_0)"
                            )

                    x_img = npz_file[selected_key]

            elif suffix in {'.tiff', '.tif'}:
                # TIFF文件（支持多通道，如16位深度图）
                x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)

            else:
                # 标准图像文件（png/jpg/bmp等）
                x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)

            if x_img is None:
                raise ValueError(f"无法读取X模态图像: {x_path}")

            return x_img

        except Exception as e:
            raise ValueError(f"加载X模态图像失败: {x_path}\n错误: {e}")

    def __iter__(self):
        """迭代器接口"""
        for i in range(len(self)):
            yield self[i]

