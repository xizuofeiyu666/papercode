# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多模态 OBB 任务入口

提供 YOLOMM 旋转框训练 / 验证 / 推理组件：
- MultiModalOBBTrainer
- MultiModalOBBValidator
- MultiModalOBBPredictor
"""

from .train import MultiModalOBBTrainer
from .val import MultiModalOBBValidator
from .predict import MultiModalOBBPredictor

__all__ = ["MultiModalOBBTrainer", "MultiModalOBBValidator", "MultiModalOBBPredictor"]
