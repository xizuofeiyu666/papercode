# Ultralytics Multimodal Inference - Results Container
# Multimodal-aware result container with visualization support
# Version: v1.0
# Date: 2026-01-13

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, Optional
from ultralytics.utils.plotting import Annotator, colors


class MultiModalResults:
    """
    多模态推理结果容器（语义完整）

    核心字段：
    - boxes: 检测框 [N, 6] (x1, y1, x2, y2, conf, cls)
    - paths: {'rgb': Path, 'x': Path}
    - orig_imgs: {'rgb': np.ndarray, 'x': np.ndarray}
    - meta: {id, x_modality, xch, ori_shape, imgsz}

    可视化规则：
    - 永远输出 RGB 可视化
    - 只有当 xch ∈ {1,3} 时才允许输出 X 可视化
    - 当 xch > 3: plot() 仅返回 RGB 结果
    """

    def __init__(
        self,
        boxes: np.ndarray,
        paths: Dict[str, Path],
        orig_imgs: Dict[str, np.ndarray],
        meta: Dict,
        names: Optional[Dict[int, str]] = None
    ):
        """
        初始化多模态结果容器

        Args:
            boxes: 检测框 [N, 6] (x1, y1, x2, y2, conf, cls)
            paths: 图像路径字典
            orig_imgs: 原始图像字典
            meta: 元数据字典
            names: 类别名称字典 {class_id: class_name}
        """
        self.boxes = boxes
        self.paths = paths
        self.orig_imgs = orig_imgs
        self.meta = meta
        self.names = names or {}

        # 检测数量
        self.num_dets = len(boxes)

        # 可视化条件
        self.xch = meta.get('xch', 3)
        self.can_visualize_x = self.xch in {1, 3}

    def plot(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        绘制检测结果（不带标题和装饰性文字，仅图）

        Args:
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签

        Returns:
            {'rgb': annotated_rgb, 'x': annotated_x} 或 {'rgb': annotated_rgb}

        Note:
            - plot() 始终返回 'rgb' key，即使单X模态推理时也会返回黑底占位图
            - 单模态推理时，缺失的模态不会出现在返回字典中
        """
        results = {}

        # 预取原图（单模态推理时可能为 None）
        rgb0 = self.orig_imgs.get('rgb', None)
        x0 = self.orig_imgs.get('x', None)

        # 1. RGB 可视化（必出：为兼容保存流程，RGB缺失时使用黑底占位图）
        if rgb0 is not None:
            rgb_img = rgb0.copy()
        else:
            # X-only 推理：使用黑底占位图（不显示X内容）
            if 'ori_shape' not in self.meta:
                raise ValueError("meta 缺少 ori_shape，无法为缺失的RGB构造占位画布")
            h, w = self.meta['ori_shape']
            rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

        rgb_annotated = self._annotate_image(
            rgb_img,
            boxes=self.boxes,
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels
        )
        results['rgb'] = rgb_annotated

        # 2. X 模态可视化（仅当 xch ∈ {1,3} 且 X 真实存在）
        if self.can_visualize_x and x0 is not None:
            x_img = x0.copy()

            # 处理 X 模态图像（确保是3通道BGR）
            if len(x_img.shape) == 2:  # 灰度图
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
            elif x_img.shape[2] == 1:  # 单通道
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)

            x_annotated = self._annotate_image(
                x_img,
                boxes=self.boxes,
                conf=conf,
                line_width=line_width,
                font_size=font_size,
                labels=labels
            )
            results['x'] = x_annotated

        return results

    def _annotate_image(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True
    ) -> np.ndarray:
        """
        在图像上标注检测框

        Args:
            img: 原始图像（BGR格式）
            boxes: 检测框 [N, 6]
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签

        Returns:
            标注后的图像
        """
        annotator = Annotator(
            img,
            line_width=line_width,
            font_size=font_size,
            pil=False  # 使用cv2模式
        )

        for box in boxes:
            x1, y1, x2, y2, confidence, cls = box
            cls = int(cls)

            # 构造标签
            if labels:
                class_name = self.names.get(cls, str(cls))
                if conf:
                    label = f"{class_name} {confidence:.2f}"
                else:
                    label = class_name
            else:
                label = ""

            # 绘制框和标签
            color = colors(cls, True)  # 根据类别获取颜色
            annotator.box_label(
                box=[x1, y1, x2, y2],
                label=label,
                color=color
            )

        return annotator.result()

    def plot_merged(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True
    ) -> Optional[np.ndarray]:
        """
        绘制双模态并排合并图（不带任何标题和装饰性文字）

        Args:
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签

        Returns:
            并排合并图，如果不满足合并条件（RGB和X都存在且X可视化）则返回 None
        """
        # 仅当 RGB 与 X 都真实存在，且 X 可视化成立时才允许合并
        has_rgb = self.orig_imgs.get('rgb', None) is not None
        has_x = self.orig_imgs.get('x', None) is not None
        if not (has_rgb and has_x and self.can_visualize_x):
            return None

        # 获取标注后的图像
        annotated = self.plot(
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels
        )

        rgb_img = annotated['rgb']
        x_img = annotated['x']

        # 确保两图高度一致
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_x, w_x = x_img.shape[:2]

        if h_rgb != h_x:
            # 以RGB高度为准，resize X
            x_img = cv2.resize(x_img, (w_x, h_rgb))

        # 并排拼接（RGB在左，X在右）
        merged = np.hstack([rgb_img, x_img])

        return merged

    def save_txt(
        self,
        save_path: Path,
        save_conf: bool = False
    ):
        """
        保存YOLO格式的txt标签文件

        Args:
            save_path: 保存路径
            save_conf: 是否保存置信度
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.num_dets == 0:
            # 创建空文件
            save_path.write_text("")
            return

        # 坐标归一化到 RGB 尺寸
        h, w = self.meta['ori_shape']

        with open(save_path, 'w') as f:
            for box in self.boxes:
                x1, y1, x2, y2, confidence, cls = box

                # YOLO 格式：class x_center y_center width height [conf]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                if save_conf:
                    f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
                else:
                    f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def save_json(
        self,
        save_path: Path
    ):
        """
        保存JSON格式的推理结果

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 构造JSON数据
        rgb_path = self.paths.get('rgb', None)
        x_path = self.paths.get('x', None)
        has_rgb = self.orig_imgs.get('rgb', None) is not None
        has_x = self.orig_imgs.get('x', None) is not None

        # 推理模态标识
        if has_rgb and has_x:
            modality = 'rgb+x'
        elif has_rgb:
            modality = 'rgb'
        elif has_x:
            modality = 'x'
        else:
            modality = 'none'

        # RGB 可视化来源（与 plot() 逻辑保持一致）
        if has_rgb:
            rgb_rendered_from = 'rgb'
        else:
            rgb_rendered_from = 'blank'  # X-only: 黑底占位图

        data = {
            'id': self.meta['id'],
            'paths': {
                'rgb': str(rgb_path) if rgb_path is not None else None,
                'x': str(x_path) if x_path is not None else None
            },
            'meta': {
                'x_modality': self.meta['x_modality'],
                'xch': self.meta['xch'],
                'ori_shape': self.meta['ori_shape'],
                'imgsz': self.meta['imgsz'],
                'modality': modality,
                'modalities': {'rgb': has_rgb, 'x': has_x},
                'visualization': {
                    'rgb_rendered_from': rgb_rendered_from,
                    'x_visualizable': bool(has_x and self.xch in {1, 3})
                }
            },
            'detections': []
        }

        # 添加检测结果
        h, w = self.meta['ori_shape']
        for box in self.boxes:
            x1, y1, x2, y2, confidence, cls = box
            cls = int(cls)

            detection = {
                'class': cls,
                'class_name': self.names.get(cls, str(cls)),
                'confidence': float(confidence),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'bbox_normalized': {
                    'x_center': float((x1 + x2) / 2 / w),
                    'y_center': float((y1 + y2) / 2 / h),
                    'width': float((x2 - x1) / w),
                    'height': float((y2 - y1) / h)
                }
            }
            data['detections'].append(detection)

        # 写入JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
