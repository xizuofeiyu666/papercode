# Ultralytics Multimodal Inference - Predictor Engine
# Independent multimodal inference engine (no BasePredictor dependency)
# Version: v1.0
# Date: 2026-01-13

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Generator
from ultralytics.utils import LOGGER, ops
from ultralytics.data.multimodal import PairingResolver, MultiModalInferenceDataset
from .results import MultiModalResults
from .saver import MultiModalSaver


class MultiModalPredictor:
    """
    多模态推理引擎核心（完全独立，不依赖 BasePredictor）

    职责：
    - 从模型读取 Router 配置（Xch）
    - 构建 dataset 并迭代样本
    - 对每个样本执行：forward -> NMS -> scale_boxes -> 组装结果
    - 支持真正的 stream（边迭代边 yield）

    设计原则：
    - 1 sample -> 1 result（不做中途变形）
    - 不解析 YAML 推断融合策略（依赖 Router 的 forward 行为）
    - stream=True 为真正流式：边迭代边 yield
    """

    def __init__(
        self,
        model,
        imgsz: Union[int, tuple] = 640,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 300,
        device: str = '',
        verbose: bool = True,
        debug: bool = False
    ):
        """
        初始化多模态推理引擎

        Args:
            model: YOLOMM/RTDETRMM 模型实例
            imgsz: 推理输入尺寸
            conf: 置信度阈值
            iou: NMS IOU 阈值
            max_det: 最大检测框数量
            device: 设备 ('cuda', 'cpu', 或 'cuda:0')
            verbose: 是否输出详细日志（常规日志）
            debug: 是否输出DEBUG调试日志
        """
        self.model = model
        self.verbose = verbose
        self.debug = debug  # 新增debug参数

        # 从模型读取 Router 配置
        if hasattr(model, 'model') and hasattr(model.model, 'mm_router'):
            self.router = model.model.mm_router
            self.xch = self.router.INPUT_SOURCES.get('X', 3)
            self.x_modality_type = getattr(self.router, 'x_modality_type', 'unknown')
        else:
            raise ValueError(
                "模型缺少 mm_router。确保使用 YOLOMM 或 RTDETRMM 模型。"
            )

        # 检测模型类型
        self.is_rtdetr = self._detect_rtdetr_model(model)
        if self.verbose:
            LOGGER.info(f"  模型类型: {'RTDETR' if self.is_rtdetr else 'YOLO'}")

        # 推理参数
        self.imgsz = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

        # 设备配置
        if device == '':
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            LOGGER.info(f"MultiModalPredictor 初始化完成:")
            LOGGER.info(f"  设备: {self.device}")
            LOGGER.info(f"  推理尺寸: {self.imgsz}")
            LOGGER.info(f"  X模态类型: {self.x_modality_type}")
            LOGGER.info(f"  X模态通道数: {self.xch}")
            LOGGER.info(f"  置信度阈值: {self.conf}")
            LOGGER.info(f"  NMS IOU阈值: {self.iou}")

    def __call__(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]]],
        x_source: Union[str, Path, List[Union[str, Path]]],
        stream: bool = False,
        save: bool = False,
        save_txt: bool = False,
        save_dir: Optional[Path] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        **kwargs
    ):
        """
        执行多模态推理（新API - 显式RGB和X模态输入）

        Args:
            rgb_source: RGB图像源
            x_source: X模态图像源
            stream: 是否流式返回结果（True=生成器，False=列表）
            save: 是否保存可视化结果
            save_txt: 是否保存txt标签
            save_dir: 保存目录（可选）
            conf: 置信度阈值（可选，覆盖初始化时的默认值）
            iou: NMS IOU阈值（可选，覆盖初始化时的默认值）
            max_det: 最大检测框数量（可选，覆盖初始化时的默认值）
            **kwargs: 其他参数

        Returns:
            stream=True: Generator[MultiModalResult]
            stream=False: List[MultiModalResult]
        """
        # 解析输入源为配对样本
        resolver = PairingResolver(x_modality=self.x_modality_type, verbose=self.verbose)
        sample_specs = resolver.resolve(rgb_source=rgb_source, x_source=x_source)

        # 构建推理数据集
        dataset = MultiModalInferenceDataset(
            samples=sample_specs,
            imgsz=self.imgsz,
            dataset_config={'Xch': self.xch, 'x_modality': self.x_modality_type},
            stride=getattr(self.model, 'stride', 32),
            verbose=self.verbose
        )

        # 流式推理
        if stream:
            return self._stream_inference(
                dataset, save=save, save_txt=save_txt, save_dir=save_dir,
                conf=conf, iou=iou, max_det=max_det
            )
        else:
            return list(self._stream_inference(
                dataset, save=save, save_txt=save_txt, save_dir=save_dir,
                conf=conf, iou=iou, max_det=max_det
            ))

    def _stream_inference(
        self,
        dataset: MultiModalInferenceDataset,
        save: bool = False,
        save_txt: bool = False,
        save_dir: Optional[Path] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None
    ) -> Generator:
        """
        流式推理实现（真正的 stream：边迭代边 yield）

        Args:
            dataset: 多模态推理数据集
            save: 是否保存可视化
            save_txt: 是否保存txt标签
            save_dir: 保存目录
            conf: 置信度阈值（可选，覆盖self.conf）
            iou: NMS IOU阈值（可选，覆盖self.iou）
            max_det: 最大检测框数量（可选，覆盖self.max_det）

        Yields:
            MultiModalResult: 单个样本的推理结果
        """
        # 使用动态参数或默认值
        conf_threshold = conf if conf is not None else self.conf
        iou_threshold = iou if iou is not None else self.iou
        max_detections = max_det if max_det is not None else self.max_det

        if self.verbose:
            LOGGER.info(f"推理参数 - conf: {conf_threshold}, iou: {iou_threshold}, max_det: {max_detections}")
        for sample in dataset:
            # 1. Forward（路由在模型内部完成）
            im_tensor = sample['im'].to(self.device)  # [1, 3+Xch, H, W]

            with torch.no_grad():
                preds = self.model(im_tensor)  # 模型 forward

            # 调试日志：查看模型输出（debug控制）
            if self.debug:
                LOGGER.info(f"[DEBUG] 模型输出类型: {type(preds)}")
                if isinstance(preds, torch.Tensor):
                    LOGGER.info(f"[DEBUG] 模型输出形状: {preds.shape}")
                elif isinstance(preds, (list, tuple)):
                    LOGGER.info(f"[DEBUG] 模型输出是list/tuple，长度: {len(preds)}")
                    LOGGER.info(f"[DEBUG] preds[0]类型: {type(preds[0])}, 形状: {preds[0].shape if isinstance(preds[0], torch.Tensor) else 'N/A'}")

            # 2. 根据模型类型选择后处理路径（不再依赖输出格式判断）
            if self.debug:
                LOGGER.info(f"[DEBUG] 模型类型: {'RTDETR' if self.is_rtdetr else 'YOLO'}")

            if self.is_rtdetr:
                # RTDETR 专用后处理：归一化cxcywh -> 像素xyxy
                if self.debug:
                    LOGGER.info("[DEBUG] 使用RTDETR后处理分支")

                # 从AutoBackend输出提取真正的Tensor
                # AutoBackend将RTDETRDecoder的(y, x)转为[y, x]
                preds_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds

                if self.debug:
                    LOGGER.info(f"[DEBUG] 提取后的Tensor形状: {preds_tensor.shape}")

                pred = self._postprocess_rtdetr(preds_tensor, sample, conf_threshold, iou_threshold, max_detections)
            else:
                # YOLO 标准后处理：像素xywh -> NMS -> 像素xyxy
                if self.debug:
                    LOGGER.info("[DEBUG] 使用YOLO后处理分支")
                preds = ops.non_max_suppression(
                    preds,
                    conf_thres=conf_threshold,
                    iou_thres=iou_threshold,
                    max_det=max_detections
                )
                pred = preds[0]  # 取第一个（batch_size=1）

                if self.debug:
                    LOGGER.info(f"[DEBUG] NMS后检测数量: {len(pred)}")
                    if len(pred) > 0:
                        LOGGER.info(f"[DEBUG] NMS后前3个框: {pred[:3, :4]}")

                if len(pred):
                    # 使用 sample.meta['ratio_pad'] 和 sample.meta['ori_shape'] 还原坐标
                    # ratio_pad 格式: (gain, (padw, padh))
                    # scale_boxes 需要: ratio_pad=((gain, gain), (padw, padh))
                    gain = sample['meta']['ratio_pad'][0]
                    pad = sample['meta']['ratio_pad'][1]

                    if self.debug:
                        LOGGER.info(f"[DEBUG] gain={gain}, pad={pad}")
                        LOGGER.info(f"[DEBUG] imgsz={sample['meta']['imgsz']}, ori_shape={sample['meta']['ori_shape']}")

                    pred[:, :4] = ops.scale_boxes(
                        img1_shape=sample['meta']['imgsz'],  # 推理输入尺寸
                        boxes=pred[:, :4],
                        img0_shape=sample['meta']['ori_shape'],  # RGB 原图尺寸
                        ratio_pad=((gain, gain), pad),  # 转换为 scale_boxes 期望的格式
                        padding=True
                    )

                    if self.debug:
                        LOGGER.info(f"[DEBUG] scale_boxes后前3个框: {pred[:3, :4]}")

            # 3. 组装 MultiModalResult
            result = self._create_result(sample, pred)

            # 4. 保存（如果需要）
            if save or save_txt:
                self._save_result(result, save_txt=save_txt, save_dir=save_dir)

            # 边迭代边 yield
            yield result

    def _is_rtdetr_output(self, preds: torch.Tensor) -> bool:
        """
        检测模型输出是否为RTDETR格式

        Args:
            preds: 模型输出张量

        Returns:
            bool: True表示RTDETR格式 (bs, num_queries, 4+nc)
                  False表示YOLO格式 (bs, num_boxes, 4+nc) 或 (bs, 4+nc, num_boxes)
        """
        # RTDETR: (bs, 300, 4+nc) 或固定查询数
        # YOLO: (bs, num_boxes, 5+nc) 或 (bs, 4+nc, num_boxes)

        if isinstance(preds, (list, tuple)):
            # YOLO NMS前: list of tensors
            return False

        if not isinstance(preds, torch.Tensor):
            return False

        # RTDETR的典型特征：维度2固定为300（或其他查询数）且较小
        # 输出形状: (bs, 300, 4+nc)
        if len(preds.shape) == 3:
            # 检查是否有固定的查询数（RTDETR典型为300）
            num_queries = preds.shape[1]
            if 100 <= num_queries <= 900:  # RTDETR查询数通常在此范围
                # 进一步验证：最后一维应该是 4 + nc (类别数)
                last_dim = preds.shape[2]
                if last_dim >= 5:  # 至少4个bbox坐标 + 1个类别
                    return True

        return False

    def _detect_rtdetr_model(self, model) -> bool:
        """
        检测模型是否为RTDETR类型

        Args:
            model: 模型实例

        Returns:
            bool: True表示RTDETR，False表示YOLO
        """
        # 检查模型head类型
        try:
            # 获取实际的PyTorch模型
            pt_model = model
            if hasattr(model, 'model'):
                pt_model = model.model

            # 方法1：检查模型类名
            model_class_name = pt_model.__class__.__name__
            if self.verbose:
                LOGGER.info(f"  PyTorch模型类名: {model_class_name}")
            if 'RTDETR' in model_class_name:
                return True

            # 方法2：检查最后一层 (Sequential 或 list)
            if hasattr(pt_model, 'model'):
                model_layers = pt_model.model
                # 处理Sequential或list
                if hasattr(model_layers, '__getitem__'):
                    try:
                        last_layer = model_layers[-1]
                        layer_name = last_layer.__class__.__name__
                        if self.verbose:
                            LOGGER.info(f"  模型head类型: {layer_name}")
                        return 'RTDETR' in layer_name
                    except:
                        pass

        except Exception as e:
            if self.verbose:
                LOGGER.warning(f"无法检测模型类型: {e}")

        return False

    def _postprocess_rtdetr(
        self,
        preds: torch.Tensor,
        sample: Dict,
        conf_threshold: float = None,
        iou_threshold: float = None,
        max_detections: int = None
    ) -> torch.Tensor:
        """
        RTDETR专用后处理：归一化cxcywh -> 像素xyxy -> 原图坐标

        Args:
            preds: RTDETR输出 (bs, num_queries, 4+nc)，bbox为归一化cxcywh [0,1]
            sample: 数据集样本，包含meta信息
            conf_threshold: 置信度阈值（可选，覆盖self.conf）
            iou_threshold: NMS IOU阈值（可选，覆盖self.iou）
            max_detections: 最大检测框数量（可选，覆盖self.max_det）

        Returns:
            torch.Tensor: [N, 6] (x1, y1, x2, y2, conf, cls) 原图坐标
        """
        # 使用动态参数或默认值
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou
        max_det = max_detections if max_detections is not None else self.max_det

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 输入preds形状: {preds.shape}")

        # 1. 拆分bbox和分类分数
        # RTDETR输出: (bs, 300, 4+nc)
        # bbox: (bs, 300, 4) - 归一化cxcywh
        # scores: (bs, 300, nc) - 类别概率
        bboxes = preds[0, :, :4]  # (300, 4) 归一化cxcywh
        scores = preds[0, :, 4:]  # (300, nc) 类别分数

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] bbox形状: {bboxes.shape}, scores形状: {scores.shape}")
            LOGGER.info(f"[DEBUG RTDETR] 前3个bbox (归一化cxcywh): {bboxes[:3]}")

        # 2. 置信度过滤
        # 对每个查询，取最大类别分数作为置信度
        conf, cls = scores.max(dim=1)  # (300,), (300,)

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 置信度阈值: {conf_thresh}")
            LOGGER.info(f"[DEBUG RTDETR] 最大置信度: {conf.max()}, 最小置信度: {conf.min()}")

        # 过滤低置信度
        mask = conf >= conf_thresh
        bboxes = bboxes[mask]  # (N, 4)
        conf = conf[mask]      # (N,)
        cls = cls[mask]        # (N,)

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 过滤后检测数量: {len(bboxes)}")

        # 如果没有检测结果，返回空张量
        if len(bboxes) == 0:
            if self.debug:
                LOGGER.info("[DEBUG RTDETR] 无检测结果，返回空张量")
            return torch.zeros((0, 6), device=preds.device)

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 过滤后前3个bbox (归一化cxcywh): {bboxes[:3]}")

        # 3. 坐标转换：归一化cxcywh -> 归一化xyxy
        bboxes = ops.xywh2xyxy(bboxes)  # (N, 4) 归一化xyxy

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 转换为归一化xyxy后前3个bbox: {bboxes[:3]}")

        # 4. 缩放到letterbox尺寸（像素坐标）
        h_letterbox, w_letterbox = sample['meta']['imgsz']
        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] letterbox尺寸: {h_letterbox}x{w_letterbox}")

        bboxes[:, [0, 2]] *= w_letterbox  # x坐标
        bboxes[:, [1, 3]] *= h_letterbox  # y坐标

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 缩放到letterbox像素坐标后前3个bbox: {bboxes[:3]}")

        # 5. NMS去重（RTDETR虽然用了匈牙利匹配，但可能仍有重叠框）
        # 组装为[N, 6]格式用于NMS
        det = torch.cat([bboxes, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1)  # (N, 6)

        # 应用NMS（使用torchvision的nms）
        from torchvision.ops import nms as torch_nms
        keep_indices = torch_nms(
            boxes=det[:, :4],
            scores=det[:, 4],
            iou_threshold=iou_thresh
        )
        det = det[keep_indices]

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] NMS后检测数量: {len(det)}")

        # 6. 限制最大检测数
        if len(det) > max_det:
            det = det[:max_det]
            if self.debug:
                LOGGER.info(f"[DEBUG RTDETR] 限制到max_det={max_det}")

        # 7. 缩放到原图尺寸
        if len(det):
            gain = sample['meta']['ratio_pad'][0]
            pad = sample['meta']['ratio_pad'][1]

            if self.debug:
                LOGGER.info(f"[DEBUG RTDETR] gain={gain}, pad={pad}")
                LOGGER.info(f"[DEBUG RTDETR] ori_shape={sample['meta']['ori_shape']}")
                LOGGER.info(f"[DEBUG RTDETR] scale_boxes前前3个bbox: {det[:3, :4]}")

            det[:, :4] = ops.scale_boxes(
                img1_shape=sample['meta']['imgsz'],
                boxes=det[:, :4],
                img0_shape=sample['meta']['ori_shape'],
                ratio_pad=((gain, gain), pad),
                padding=True
            )

            if self.debug:
                LOGGER.info(f"[DEBUG RTDETR] scale_boxes后前3个bbox: {det[:3, :4]}")

        return det

    def _create_result(self, sample: Dict, pred: torch.Tensor) -> MultiModalResults:
        """
        组装 MultiModalResults

        Args:
            sample: 数据集样本（包含 paths, orig_imgs, meta, im）
            pred: NMS 后的预测结果 [N, 6] (x1,y1,x2,y2,conf,cls)

        Returns:
            MultiModalResults 实例
        """
        boxes = pred.cpu().numpy() if len(pred) else np.zeros((0, 6))

        # 获取类别名称（如果模型有提供）
        names = getattr(self.model, 'names', None)

        # 将 id 添加到 meta 中（从 sample 顶层提取）
        meta = sample['meta'].copy()
        meta['id'] = sample['id']

        result = MultiModalResults(
            boxes=boxes,
            paths=sample['paths'],
            orig_imgs=sample['orig_imgs'],
            meta=meta,
            names=names
        )

        return result

    def _save_result(
        self,
        result: MultiModalResults,
        save_txt: bool = False,
        save_dir: Optional[Path] = None
    ):
        """
        保存推理结果（使用 MultiModalSaver）

        Args:
            result: MultiModalResults 实例
            save_txt: 是否保存txt标签
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = Path('runs/predict')

        save_dir = Path(save_dir)

        # 使用 MultiModalSaver 保存
        saver = MultiModalSaver(
            save_dir=save_dir,
            save_txt=save_txt,
            save_json=True,  # 默认保存json
            save_conf=False
        )

        saver.save(result)
