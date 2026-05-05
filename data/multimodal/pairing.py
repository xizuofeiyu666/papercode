# Ultralytics Multimodal Inference - Explicit Pairing Resolver
# Resolves explicit RGB+X inputs into paired sample specifications
# Version: v2.0 (Breaking Change - 废弃隐式source参数)
# Date: 2026-01-13

from pathlib import Path
from typing import List, Dict, Union
from ultralytics.utils import LOGGER


class PairingResolver:
    """
    多模态推理显式配对解析器（新API）

    职责：
    - 接收显式的 rgb_source 和 x_source 参数
    - 验证输入合法性和文件存在性
    - 生成统一的配对样本规格

    新API设计：
    - 强制要求同时提供 rgb_source 和 x_source
    - 支持单对、批量推理
    - 不再支持隐式列表 [rgb, x] 格式（breaking change）
    """

    def __init__(self, x_modality: str = "unknown", verbose: bool = True):
        """
        初始化配对解析器

        Args:
            x_modality: X模态类型名称（如 'thermal', 'depth', 'ir'等）
            verbose: 是否输出详细日志
        """
        self.x_modality = x_modality
        self.verbose = verbose

    def resolve(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]], None] = None,
        x_source: Union[str, Path, List[Union[str, Path]], None] = None
    ) -> List[Dict[str, Union[str, Path, None]]]:
        """
        解析显式RGB和X模态输入为配对样本列表（支持单模态推理）

        Args:
            rgb_source: RGB图像源（可为None表示缺失）
                - 单图: '/path/to/rgb.jpg' 或 Path('/path/to/rgb.jpg')
                - 批量: ['/path/rgb1.jpg', '/path/rgb2.jpg']
                - 缺失: None (将使用零填充)
            x_source: X模态图像源（可为None表示缺失）
                - 单图: '/path/to/thermal.jpg'
                - 批量: ['/path/thermal1.jpg', '/path/thermal2.jpg']
                - 缺失: None (将使用零填充)

        Returns:
            配对样本规格列表 [
                {
                    'id': 'sample_001',
                    'rgb_path': Path('/path/rgb.jpg') or None,
                    'x_path': Path('/path/thermal.jpg') or None,
                    'x_modality': 'thermal'
                },
                ...
            ]

        Raises:
            ValueError: 输入格式不合法或数量不匹配
            FileNotFoundError: 文件不存在
        """
        # 单模态推理处理
        if rgb_source is None and x_source is not None:
            # 单X模态推理
            x_list = self._normalize_to_list(x_source, "x_source")
            samples = []
            for idx, x_path in enumerate(x_list):
                samples.append(self._create_sample_spec(
                    rgb_path=None,
                    x_path=Path(x_path),
                    sample_idx=idx
                ))
            if self.verbose:
                LOGGER.info(f"单{self.x_modality}模态推理: {len(samples)} 个样本（RGB将使用零填充）")
            return samples

        elif rgb_source is not None and x_source is None:
            # 单RGB模态推理
            rgb_list = self._normalize_to_list(rgb_source, "rgb_source")
            samples = []
            for idx, rgb_path in enumerate(rgb_list):
                samples.append(self._create_sample_spec(
                    rgb_path=Path(rgb_path),
                    x_path=None,
                    sample_idx=idx
                ))
            if self.verbose:
                LOGGER.info(f"单RGB模态推理: {len(samples)} 个样本（{self.x_modality}将使用零填充）")
            return samples

        # 双模态推理（现有逻辑）
        # 统一转为列表格式
        rgb_list = self._normalize_to_list(rgb_source, "rgb_source")
        x_list = self._normalize_to_list(x_source, "x_source")

        # 验证数量匹配
        if len(rgb_list) != len(x_list):
            raise ValueError(
                f"RGB和X模态数量不匹配：\n"
                f"  rgb_source: {len(rgb_list)} 张\n"
                f"  x_source: {len(x_list)} 张\n"
                f"请确保两者数量相同。"
            )

        # 配对并验证
        samples = []
        for idx, (rgb_path, x_path) in enumerate(zip(rgb_list, x_list)):
            samples.append(self._create_sample_spec(
                rgb_path=Path(rgb_path),
                x_path=Path(x_path),
                sample_idx=idx
            ))

        if self.verbose:
            LOGGER.info(f"双模态配对完成: {len(samples)} 对有效样本")

        return samples

    def _normalize_to_list(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        param_name: str
    ) -> List[Path]:
        """
        统一输入为列表格式

        Args:
            source: 输入源（单个或列表）
            param_name: 参数名称（用于错误提示）

        Returns:
            Path对象列表
        """
        if isinstance(source, (str, Path)):
            return [Path(source)]
        elif isinstance(source, list):
            if not source:
                raise ValueError(f"{param_name} 不能为空列表")
            return [Path(item) for item in source]
        else:
            raise ValueError(
                f"{param_name} 类型不支持: {type(source)}\n"
                f"支持类型: str, Path, List[str], List[Path]"
            )

    def _create_sample_spec(
        self,
        rgb_path: Union[Path, None],
        x_path: Union[Path, None],
        sample_idx: int
    ) -> Dict[str, Union[str, Path, None]]:
        """
        创建配对样本规格并验证文件存在性（支持None占位）

        Args:
            rgb_path: RGB图像路径（可为None表示缺失）
            x_path: X模态图像路径（可为None表示缺失）
            sample_idx: 样本索引（用于生成ID）

        Returns:
            样本规格字典

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 路径不是文件
        """
        # 验证RGB文件（如果提供）
        if rgb_path is not None:
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")
            if not rgb_path.is_file():
                raise ValueError(f"RGB路径不是文件: {rgb_path}")

        # 验证X模态文件（如果提供）
        if x_path is not None:
            if not x_path.exists():
                raise FileNotFoundError(f"X模态文件不存在: {x_path}")
            if not x_path.is_file():
                raise ValueError(f"X模态路径不是文件: {x_path}")

        # 生成样本ID（优先使用RGB文件名，否则使用X模态文件名）
        if rgb_path is not None:
            sample_id = rgb_path.stem
        elif x_path is not None:
            sample_id = x_path.stem
        else:
            sample_id = f"sample_{sample_idx:03d}"

        return {
            "id": sample_id,
            "rgb_path": rgb_path,
            "x_path": x_path,
            "x_modality": self.x_modality
        }
