# Ultralytics Multimodal Inference - Saver Component
# Handles saving of multimodal inference results
# Version: v1.0
# Date: 2026-01-13

import cv2
from pathlib import Path
from typing import Optional
from .results import MultiModalResults


class MultiModalSaver:
    """
    多模态推理结果保存器（显式且可预测）

    输出文件命名规则（以 RGB 为主路径基准）：
    - {id}_rgb.jpg: 必出
    - {id}_{x_modality}.jpg: 仅当 xch ∈ {1,3} 出
    - {id}_multimodal.jpg: 仅当 RGB 与 X 都可视化时出（并排）
    - labels/{id}.txt: 可选（坐标以 RGB 尺寸归一化）
    - {id}.json: 可选（paths + boxes + meta）
    """

    def __init__(
        self,
        save_dir: Path,
        save_txt: bool = False,
        save_json: bool = False,
        save_conf: bool = False
    ):
        """
        初始化保存器

        Args:
            save_dir: 保存根目录
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_conf: txt中是否包含置信度
        """
        self.save_dir = Path(save_dir)
        self.save_txt = save_txt
        self.save_json = save_json
        self.save_conf = save_conf

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.save_txt:
            (self.save_dir / 'labels').mkdir(parents=True, exist_ok=True)

        if self.save_json:
            (self.save_dir / 'json').mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: MultiModalResults,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True
    ):
        """
        保存单个推理结果

        Args:
            result: MultiModalResults 实例
            conf: 可视化时是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
        """
        sample_id = result.meta['id']
        x_modality = result.meta['x_modality']

        # 1. 保存可视化图像
        annotated = result.plot(
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels
        )

        # 1.1 RGB 可视化（必出）
        rgb_path = self.save_dir / f"{sample_id}_rgb.jpg"
        cv2.imwrite(str(rgb_path), annotated['rgb'])

        # 1.2 X 模态可视化（仅当 xch ∈ {1,3}）
        if 'x' in annotated:
            x_path = self.save_dir / f"{sample_id}_{x_modality}.jpg"
            cv2.imwrite(str(x_path), annotated['x'])

        # 1.3 双模态并排合并图（仅当 RGB 与 X 都可视化时）
        merged = result.plot_merged(
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels
        )
        if merged is not None:
            merged_path = self.save_dir / f"{sample_id}_multimodal.jpg"
            cv2.imwrite(str(merged_path), merged)

        # 2. 保存txt标签（可选）
        if self.save_txt:
            txt_path = self.save_dir / 'labels' / f"{sample_id}.txt"
            result.save_txt(txt_path, save_conf=self.save_conf)

        # 3. 保存json结果（可选）
        if self.save_json:
            json_path = self.save_dir / 'json' / f"{sample_id}.json"
            result.save_json(json_path)
