# Ultralytics 多模态可视化复用组件（统一绘图工具）
"""
作用与目标
---------
本组件为 YOLOMM 多模态（RGB + X）任务提供统一的绘图与几何处理工具，面向 train/val/cocoval/predict
四个阶段复用，确保输出风格一致、坐标处理正确，避免黑图、错位、跨图等问题。

统一约定
-------
- 多模态输入的实际通道顺序为严格的 [RGB(3), X(Xch)]，与数据加载保持一致。
- 所有绘图使用归一化坐标（xywhn）时，值域应在 [0, 1] 内。
- 并排（side-by-side）视图的坐标域为 2W 的组合图：左半（RGB）x ∈ [0, 0.5]，右半（X）x ∈ [0.5, 1.0]。

提供的函数
---------
- split_modalities(images, xch):
  将 (B, 3+Xch, H, W) 的多模态张量拆分为 RGB(B,3,H,W) 与 X(B,Xch,H,W)。

- visualize_x_to_3ch(x, colorize=False, x_modality='depth'):
  将 X 模态可视化为 3 通道（灰度复制或伪彩色：depth/thermal/ir 等），用于直接落图或对比图。

- concat_side_by_side(rgb, x3):
  将 RGB(3ch) 与 X(3ch 可视化) 在宽度维度拼接，形成并排对比图 (B,3,H,2W)。

- to_norm_xywh_for_plot(batch_ids, cls_ids, boxes_xywh_px, confs, img_hw):
  将像素系 xywh 转换为归一化 xywh（0~1），便于 plot_images 使用。

- duplicate_bboxes_for_side_by_side(batch_ids, cls_ids, bboxes_xywh_norm, confs=None):
  将单图域的归一化 xywh bbox 复制为左右两份，自动缩放/平移到并排图的左/右半区域，并裁剪到半幅合法域。

- clip_boxes_norm_xywh(bboxes_xywh_norm, x_min, x_max, y_min, y_max):
  将归一化 xywh bbox 裁剪到指定归一化边界内，自动处理异常几何关系（x1>x2 / y1>y2）。

- adjust_bboxes_for_side_by_side(bboxes_xywh_norm):
  将单图域归一化 bbox 的 x,w 缩到半宽，用于并排左半（RGB）绘制的快速适配。

- ensure_batch_idx_long(batch_idx):
  将 batch 索引统一为 torch.long 类型，避免浮点比较错配问题。

- resolve_x_modality(modality_param, data):
  解析 X 模态类型（优先 data.yaml 的 modality_used/models/modality），用于命名与伪彩映射。

- get_x_modality_path(modality_name, data):
  获取 X 模态在数据配置中的目录路径（优先 data.yaml 的 modality/modalities）。

说明
----
以上函数构成统一的“复用组件链”，典型使用路径为：
拆分模态 →（X 伪彩）→ 并排拼接 → 坐标归一化/裁剪 → bbox 复制 → plot_images 落图。
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional, Union, Any
from ultralytics.utils import LOGGER
from typing import Optional


def split_modalities(images: torch.Tensor, xch: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    拆分多模态图像为RGB和X模态。
    
    严格遵循 [RGB, X] 顺序。
    
    Args:
        images: 多模态图像张量 (B, 3+Xch, H, W)
        xch: X模态通道数
        
    Returns:
        tuple: (rgb, x) - RGB图像和X模态图像
    """
    total_channels = 3 + xch
    if images.shape[1] != total_channels:
        LOGGER.warning(f"期望{total_channels}通道，但收到{images.shape[1]}通道")
        
    # 通道顺序：[RGB(0:3), X(3:3+xch)]
    rgb_images = images[:, :3, :, :]           # 前3通道：RGB模态
    x_images = images[:, 3:3+xch, :, :]        # 后Xch通道：X模态
    
    return rgb_images, x_images


def visualize_x_to_3ch(x: torch.Tensor, colorize: bool = False, x_modality: str = 'depth') -> torch.Tensor:
    """
    将X模态转换为3通道用于可视化。
    
    Args:
        x: X模态张量 (B, Xch, H, W)
        colorize: 是否启用伪彩色映射（YOLOMM可启用，RTDETRMM默认灰度）
        x_modality: X模态类型
        
    Returns:
        torch.Tensor: 3通道可视化张量 (B, 3, H, W)
    """
    original_device = x.device
    
    if x.shape[1] == 1:
        # 单通道X模态
        single_channel = x[:, 0:1, :, :]
        
        if colorize:
            # 应用伪彩色映射
            colorized_images = []
            for i in range(single_channel.shape[0]):
                img_np = single_channel[i, 0].cpu().numpy()
                
                # 归一化到0-255
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                # 根据模态类型选择颜色映射
                if x_modality in ['depth']:
                    colormap = cv2.COLORMAP_VIRIDIS  # 深度用绿蓝色系
                elif x_modality in ['thermal', 'infrared', 'ir']:
                    colormap = cv2.COLORMAP_INFERNO  # 热红外用红黄色系
                else:
                    colormap = cv2.COLORMAP_JET  # 其他用彩虹色
                    
                colored_img = cv2.applyColorMap(img_np, colormap)
                colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
                
                # 转换回tensor格式
                colored_tensor = torch.from_numpy(colored_img.transpose(2, 0, 1)).float().to(original_device)
                if colored_tensor.max() > 1.0:
                    colored_tensor /= 255.0
                    
                colorized_images.append(colored_tensor)
            
            return torch.stack(colorized_images)
        else:
            # 灰度复制成3通道（RTDETRMM默认，稳定对齐）
            return single_channel.repeat(1, 3, 1, 1)
    
    elif x.shape[1] == 3:
        # 已经是3通道，直接返回
        return x.to(original_device)
    
    else:
        # 多通道情况，使用前3通道或重复第一通道
        if x.shape[1] >= 3:
            return x[:, :3, :, :].to(original_device)
        else:
            # 不足3通道，重复第一通道
            first_channel = x[:, 0:1, :, :]
            return first_channel.repeat(1, 3, 1, 1).to(original_device)


def concat_side_by_side(rgb: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """
    创建RGB和X模态的并排可视化。
    
    Args:
        rgb: RGB图像 (B, 3, H, W)
        x3: X模态3通道图像 (B, 3, H, W)
        
    Returns:
        torch.Tensor: 并排拼接图像 (B, 3, H, 2W)
    """
    # 确保两个张量在同一设备上
    if rgb.device != x3.device:
        x3 = x3.to(rgb.device)
    
    # 水平拼接：RGB在左，X在右
    side_by_side = torch.cat([rgb, x3], dim=3)  # 在宽度维度拼接
    
    return side_by_side


def to_norm_xywh_for_plot(batch_ids, cls_ids, boxes_xywh_px, confs, img_hw: Tuple[int, int]) -> tuple:
    """
    将像素系xywh转换为归一化xywh用于plot_images。
    
    Args:
        batch_ids: 批次索引
        cls_ids: 类别ID
        boxes_xywh_px: 像素坐标系的xywh边界框
        confs: 置信度
        img_hw: 图像尺寸 (H, W)
        
    Returns:
        tuple: (batch_ids, cls_ids, boxes_xywh_norm, confs) 用于plot_images
    """
    H, W = img_hw
    
    if isinstance(boxes_xywh_px, torch.Tensor):
        boxes_norm = boxes_xywh_px.clone().float()
    else:
        boxes_norm = torch.tensor(boxes_xywh_px, dtype=torch.float32)
    
    # 将像素坐标转换为归一化坐标
    if boxes_norm.numel() > 0:
        # x, y 除以 W, H
        boxes_norm[:, 0] /= W  # x
        boxes_norm[:, 1] /= H  # y
        boxes_norm[:, 2] /= W  # w
        boxes_norm[:, 3] /= H  # h
    
    return batch_ids, cls_ids, boxes_norm, confs


def duplicate_bboxes_for_side_by_side(batch_ids, cls_ids, bboxes_xywh_norm, confs=None):
    """
    为并排图像复制边界框到两侧：左半（RGB）和右半（X模态）。
    
    输入必须是归一化到单张图 (W,H) 的 xywh（0~1 范围，相对原单图）。
    输出将bbox复制成两份，拼接返回（用于side-by-side的2W坐标系）。
    
    Args:
        batch_ids: 批次索引
        cls_ids: 类别ID
        bboxes_xywh_norm: 归一化xywh边界框 (相对单图W,H)
        confs: 置信度 (可选)
        
    Returns:
        tuple: (batch_ids_dup, cls_ids_dup, bboxes_dup, confs_dup) 
               复制后的参数，适用于side-by-side图像的2W坐标系
    """
    # 处理空输入
    if bboxes_xywh_norm is None or (hasattr(bboxes_xywh_norm, 'numel') and bboxes_xywh_norm.numel() == 0):
        return batch_ids, cls_ids, bboxes_xywh_norm, confs
    if isinstance(bboxes_xywh_norm, np.ndarray) and bboxes_xywh_norm.size == 0:
        return batch_ids, cls_ids, bboxes_xywh_norm, confs
    
    is_torch = isinstance(bboxes_xywh_norm, torch.Tensor)
    
    if is_torch:
        # 统一索引/置信度为与bboxes一致的torch类型，避免类型混用
        device = bboxes_xywh_norm.device
        if batch_ids is not None and not isinstance(batch_ids, torch.Tensor):
            batch_ids = torch.as_tensor(batch_ids, device=device, dtype=torch.long)
        if cls_ids is not None and not isinstance(cls_ids, torch.Tensor):
            cls_ids = torch.as_tensor(cls_ids, device=device, dtype=torch.long)
        if confs is not None and not isinstance(confs, torch.Tensor):
            confs = torch.as_tensor(confs, device=device, dtype=torch.float32)
        # 验证是否为归一化坐标
        if bboxes_xywh_norm.numel() > 0:
            is_norm = bboxes_xywh_norm[:, :4].max() <= 1.1
            if not is_norm:
                LOGGER.warning("duplicate_bboxes_for_side_by_side: 输入应为归一化坐标(0~1)")
        
        # 统一类型到torch，解决numpy/torch混用问题
        device = bboxes_xywh_norm.device
        if batch_ids is not None and not isinstance(batch_ids, torch.Tensor):
            batch_ids = torch.as_tensor(batch_ids, device=device, dtype=torch.long)
        if cls_ids is not None and not isinstance(cls_ids, torch.Tensor):
            cls_ids = torch.as_tensor(cls_ids, device=device, dtype=torch.long)
        if confs is not None and not isinstance(confs, torch.Tensor):
            confs = torch.as_tensor(confs, device=device, dtype=torch.float32)
        
        # 复制bbox：左半 + 右半
        left_bboxes = bboxes_xywh_norm.clone()
        right_bboxes = bboxes_xywh_norm.clone()
        
        # 左半：x *= 0.5, w *= 0.5 (对应RGB)
        left_bboxes[:, 0] *= 0.5  # x坐标减半
        left_bboxes[:, 2] *= 0.5  # 宽度减半
        
        # 右半：x = x*0.5 + 0.5, w *= 0.5 (对应X模态)
        right_bboxes[:, 0] = right_bboxes[:, 0] * 0.5 + 0.5  # x坐标缩放后偏移到右半
        right_bboxes[:, 2] *= 0.5  # 宽度减半
        
        # 半幅域裁剪，防止跨子图（左半: x∈[0,0.5]；右半: x∈[0.5,1.0]）
        left_bboxes = clip_boxes_norm_xywh(left_bboxes, x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0)
        right_bboxes = clip_boxes_norm_xywh(right_bboxes, x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0)

        # 拼接bbox
        bboxes_dup = torch.cat([left_bboxes, right_bboxes], dim=0)
        
        # 复制其他参数
        batch_ids_dup = torch.cat([batch_ids, batch_ids], dim=0) if batch_ids is not None else None
        cls_ids_dup = torch.cat([cls_ids, cls_ids], dim=0) if cls_ids is not None else None
        confs_dup = torch.cat([confs, confs], dim=0) if confs is not None else None
        
    else:  # numpy
        # 验证是否为归一化坐标
        if bboxes_xywh_norm.size > 0:
            is_norm = bboxes_xywh_norm[:, :4].max() <= 1.1
            if not is_norm:
                LOGGER.warning("duplicate_bboxes_for_side_by_side: 输入应为归一化坐标(0~1)")
        
        # 复制bbox：左半 + 右半
        left_bboxes = bboxes_xywh_norm.copy()
        right_bboxes = bboxes_xywh_norm.copy()
        
        # 左半：x *= 0.5, w *= 0.5 (对应RGB)
        left_bboxes[:, 0] *= 0.5  # x坐标减半
        left_bboxes[:, 2] *= 0.5  # 宽度减半
        
        # 右半：x = x*0.5 + 0.5, w *= 0.5 (对应X模态)
        right_bboxes[:, 0] = right_bboxes[:, 0] * 0.5 + 0.5  # x坐标缩放后偏移到右半
        right_bboxes[:, 2] *= 0.5  # 宽度减半
        
        # 半幅域裁剪，防止跨子图（左半/右半）
        left_bboxes = clip_boxes_norm_xywh(left_bboxes, x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0)
        right_bboxes = clip_boxes_norm_xywh(right_bboxes, x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0)

        # 拼接bbox
        bboxes_dup = np.concatenate([left_bboxes, right_bboxes], axis=0)
        
        # 复制其他参数
        batch_ids_dup = np.concatenate([batch_ids, batch_ids], axis=0) if batch_ids is not None else None
        cls_ids_dup = np.concatenate([cls_ids, cls_ids], axis=0) if cls_ids is not None else None
        confs_dup = np.concatenate([confs, confs], axis=0) if confs is not None else None
    
    return batch_ids_dup, cls_ids_dup, bboxes_dup, confs_dup


def _xywh_to_xyxy(data, is_torch: bool):
    """Convert normalized xywh to normalized xyxy for torch or numpy arrays."""
    if is_torch:
        x, y, w, h = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return torch.stack([x1, y1, x2, y2], dim=1)
    else:
        x1 = data[:, 0] - data[:, 2] / 2.0
        y1 = data[:, 1] - data[:, 3] / 2.0
        x2 = data[:, 0] + data[:, 2] / 2.0
        y2 = data[:, 1] + data[:, 3] / 2.0
        return np.stack([x1, y1, x2, y2], axis=1)


def _xyxy_to_xywh(data, is_torch: bool):
    """Convert normalized xyxy to normalized xywh for torch or numpy arrays."""
    if is_torch:
        x1, y1, x2, y2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        w = torch.clamp(x2 - x1, min=0.0)
        h = torch.clamp(y2 - y1, min=0.0)
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        return torch.stack([x, y, w, h], dim=1)
    else:
        x1, y1, x2, y2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        w = np.clip(x2 - x1, 0.0, None)
        h = np.clip(y2 - y1, 0.0, None)
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        return np.stack([x, y, w, h], axis=1)


def clip_boxes_norm_xywh(
    bboxes_xywh_norm: Optional[Union[torch.Tensor, np.ndarray]],
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
):
    """
    裁剪归一化xywh边界框到给定的归一化域边界内，支持torch与numpy。

    处理逻辑：xywh→xyxy→clamp→xywh，保证几何边界被裁到域内，避免跨子图越界。

    Args:
        bboxes_xywh_norm: 归一化xywh边界框，形状[N,4]
        x_min, x_max, y_min, y_max: 归一化域边界

    Returns:
        与输入同类型、同形状的裁剪后xywh
    """
    if bboxes_xywh_norm is None:
        return bboxes_xywh_norm

    if isinstance(bboxes_xywh_norm, torch.Tensor):
        if bboxes_xywh_norm.numel() == 0:
            return bboxes_xywh_norm
        xyxy = _xywh_to_xyxy(bboxes_xywh_norm, is_torch=True)
        xyxy[:, 0] = torch.clamp(xyxy[:, 0], min=x_min, max=x_max)
        xyxy[:, 2] = torch.clamp(xyxy[:, 2], min=x_min, max=x_max)
        xyxy[:, 1] = torch.clamp(xyxy[:, 1], min=y_min, max=y_max)
        xyxy[:, 3] = torch.clamp(xyxy[:, 3], min=y_min, max=y_max)
        # 修正可能出现的x1>x2或y1>y2
        xyxy[:, 2] = torch.maximum(xyxy[:, 2], xyxy[:, 0])
        xyxy[:, 3] = torch.maximum(xyxy[:, 3], xyxy[:, 1])
        return _xyxy_to_xywh(xyxy, is_torch=True)
    elif isinstance(bboxes_xywh_norm, np.ndarray):
        if bboxes_xywh_norm.size == 0:
            return bboxes_xywh_norm
        xyxy = _xywh_to_xyxy(bboxes_xywh_norm, is_torch=False)
        xyxy[:, 0] = np.clip(xyxy[:, 0], x_min, x_max)
        xyxy[:, 2] = np.clip(xyxy[:, 2], x_min, x_max)
        xyxy[:, 1] = np.clip(xyxy[:, 1], y_min, y_max)
        xyxy[:, 3] = np.clip(xyxy[:, 3], y_min, y_max)
        # 修正可能出现的x1>x2或y1>y2
        xyxy[:, 2] = np.maximum(xyxy[:, 2], xyxy[:, 0])
        xyxy[:, 3] = np.maximum(xyxy[:, 3], xyxy[:, 1])
        return _xyxy_to_xywh(xyxy, is_torch=False)
    else:
        LOGGER.warning(f"clip_boxes_norm_xywh: 未知的bboxes类型 {type(bboxes_xywh_norm)}")
        return bboxes_xywh_norm


def adjust_bboxes_for_side_by_side(bboxes_xywh_norm) -> Any:
    """
    调整归一化边界框坐标以适应并排图像。
    
    仅对归一化坐标执行x,w *= 0.5的半宽缩放，像素系输入不做处理。
    
    Args:
        bboxes_xywh_norm: 归一化xywh边界框
        
    Returns:
        调整后的边界框坐标，保持输入类型
    """
    if bboxes_xywh_norm is None:
        return bboxes_xywh_norm
    
    if isinstance(bboxes_xywh_norm, torch.Tensor):
        if bboxes_xywh_norm.numel() == 0:
            return bboxes_xywh_norm
            
        # 判断是否为归一化坐标
        is_norm = bboxes_xywh_norm[:, :4].max() <= 1.1
        
        adjusted_bboxes = bboxes_xywh_norm.clone()
        
        if is_norm:
            # 将x坐标和宽度都乘以0.5，适配到左半（RGB）区域
            adjusted_bboxes[:, 0] *= 0.5  # x坐标减半
            adjusted_bboxes[:, 2] *= 0.5  # 宽度减半
            
        return adjusted_bboxes
        
    elif isinstance(bboxes_xywh_norm, np.ndarray):
        if bboxes_xywh_norm.size == 0:
            return bboxes_xywh_norm
            
        # 判断是否为归一化坐标
        is_norm = bboxes_xywh_norm[:, :4].max() <= 1.1
        
        adjusted_bboxes = bboxes_xywh_norm.copy()
        
        if is_norm:
            # 将x坐标和宽度都乘以0.5
            adjusted_bboxes[:, 0] *= 0.5  # x坐标减半
            adjusted_bboxes[:, 2] *= 0.5  # 宽度减半
            
        return adjusted_bboxes
        
    else:
        LOGGER.warning(f"adjust_bboxes_for_side_by_side: 未知的bboxes类型 {type(bboxes_xywh_norm)}")
        return bboxes_xywh_norm


def ensure_batch_idx_long(batch_idx) -> torch.Tensor:
    """
    确保batch_idx为torch.long类型，避免浮点比较错配。
    
    Args:
        batch_idx: 批次索引张量
        
    Returns:
        torch.Tensor: long类型的batch_idx
    """
    if isinstance(batch_idx, torch.Tensor):
        return batch_idx.long()
    else:
        return torch.tensor(batch_idx, dtype=torch.long)


def resolve_x_modality(modality_param: Optional[str], data: Optional[dict]) -> str:
    """
    解析X模态类型，与YOLOMM/RTDETRMM验证器保持一致。
    
    Args:
        modality_param: 用户指定的模态参数
        data: 数据配置字典
        
    Returns:
        str: X模态类型标识符
    """
    # 优先级1：从data.yaml的modality_used字段读取
    if data and 'modality_used' in data:
        modality_used = data['modality_used']
        if isinstance(modality_used, list) and len(modality_used) >= 2:
            x_modalities = [m for m in modality_used if m != 'rgb']
            if x_modalities:
                return x_modalities[0]

    # 优先级2：从data.yaml的models字段读取
    if data and 'models' in data:
        models = data['models']
        if isinstance(models, list) and len(models) >= 2:
            x_modalities = [m for m in models if m != 'rgb']
            if x_modalities:
                return x_modalities[0]

    # 优先级3：从modality字段推断
    if data and 'modality' in data:
        modality = data['modality']
        if isinstance(modality, dict):
            x_modalities = [k for k in modality.keys() if k != 'rgb']
            if x_modalities:
                return x_modalities[0]

    # 优先级4：用户指定的模态参数
    if modality_param and modality_param != 'rgb':
        if modality_param.upper() == 'X':
            # 'X'是特殊标记，需要进一步解析
            return 'depth'  # 默认depth
        return modality_param

    # 最后使用默认值
    LOGGER.warning("无法解析X模态类型，使用默认值: depth")
    return 'depth'


def get_x_modality_path(modality_name: str, data: Optional[dict]) -> str:
    """
    获取X模态的目录路径。
    
    Args:
        modality_name: 模态名称
        data: 数据配置字典
        
    Returns:
        str: 模态目录路径
    """
    # 优先从data.yaml的modality字段读取
    if data and 'modality' in data:
        modality_paths = data['modality']
        if isinstance(modality_paths, dict) and modality_name in modality_paths:
            return modality_paths[modality_name]
    
    # 向后兼容：检查modalities字段
    if data and 'modalities' in data:
        modalities = data['modalities']
        if isinstance(modalities, dict) and modality_name in modalities:
            return modalities[modality_name]
    
    # 默认格式
    return f'images_{modality_name}'
