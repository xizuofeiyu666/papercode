# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import de_parallel
import torch
from ultralytics.utils.checks import check_imgsz
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import TQDM, callbacks, emojis
from ultralytics.utils.ops import Profile
import numpy as np
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
import json


class MultiModalDetectionValidator(DetectionValidator):
    """
    多模态检测验证器，处理RGB+X输入的验证和评估。
    
    这个类继承DetectionValidator，重写关键方法以支持多模态数据集和动态通道输入。
    与MultiModalDetectionTrainer保持一致的多模态数据处理能力。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        初始化多模态检测验证器。

        Args:
            dataloader: 数据加载器
            save_dir: 保存目录
            pbar: 进度条（当前项目不支持，忽略）
            args: 参数配置
            _callbacks: 回调函数
        """
        # 适配当前项目的DetectionValidator.__init__签名（不包含pbar参数）
        super().__init__(dataloader, save_dir, args, _callbacks)
        
        # Get modality parameter from standard cfg system (与训练器保持一致)
        # Modality validation is handled by cfg system, no local validation needed
        # Handle both dict and object-like args
        if args:
            if isinstance(args, dict):
                self.modality = args.get('modality', None)
            else:
                self.modality = getattr(args, 'modality', None)
        else:
            self.modality = None
        
        # Initialize modality-specific attributes
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        
        # 日志输出
        if self.modality:
            LOGGER.info(f"初始化MultiModalDetectionValidator - 单模态验证模式: {self.modality}-only")
        else:
            LOGGER.info("初始化MultiModalDetectionValidator - 双模态验证模式")
        
        # 初始化多模态配置（稍后在有data属性时解析）
        self.multimodal_config = None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        执行验证过程，支持6通道多模态输入。
        
        重写基类方法以支持6通道warmup和多模态数据处理。
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            # 关键修复：保护多模态验证器的data配置不被覆盖
            # 只有当验证器没有data配置时才从trainer获取
            if self.data is None:
                self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
            # 将runtime模态参数注入router（训练态）
            if hasattr(model, 'mm_router') and model.mm_router and self.modality:
                model.mm_router.set_runtime_params(
                    self.modality,
                    strategy=getattr(self.args, 'ablation_strategy', None),
                    seed=getattr(self.args, 'seed', None),
                )
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 6, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            # 将runtime模态参数注入router（仅在PyTorch后端可用）
            try:
                if hasattr(model, 'pt') and model.pt and hasattr(model, 'model') and hasattr(model.model, 'mm_router') and model.model.mm_router and self.modality:
                    model.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, 'ablation_strategy', None),
                        seed=getattr(self.args, 'seed', None),
                    )
            except Exception:
                pass
            # 动态通道数warmup，支持可配置的Xch
            if hasattr(self, 'data') and self.data and 'Xch' in self.data:
                x_channels = self.data.get('Xch', 3)
                total_channels = 3 + x_channels
                LOGGER.info(f"执行{total_channels}通道多模态YOLO模型warmup (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:
                # 向后兼容：6通道默认配置
                LOGGER.info("执行6通道多模态YOLO模型warmup (默认)")
                model.warmup(imgsz=(1 if pt else self.args.batch, 6, imgsz, imgsz))

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    # 在训练态的验证过程中，需与训练时的loss项数量保持一致（含对比项）。
                    # 由于上面进行了 model.eval() 推理，这里仅在计算 loss 时临时切换到 train()，
                    # 以启用对比分支（不会进行前向传播，且处于 inference_mode，无梯度、不会更新BN统计）。
                    orig_mode = model.training
                    try:
                        model.train()
                        self.loss += model.loss(batch, preds)[1]
                    finally:
                        if not orig_mode:
                            model.eval()

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def _parse_multimodal_config(self):
        """
        解析和验证数据配置文件中的多模态设置。
        
        与MultiModalDetectionTrainer使用相同的配置解析逻辑，
        确保训练和验证阶段使用一致的多模态配置。
        
        优先支持用户指定的单模态验证参数。
        
        Returns:
            dict: 解析后的多模态配置
        """
        # 优先检查用户指定的modality参数（单模态验证）
        if self.modality:
            # 构建单模态配置 - 智能确定X模态类型
            if self.modality == 'rgb':
                # RGB单模态：使用RGB + 动态确定的X模态进行零填充
                x_modality = self._determine_x_modality_from_data()
                config = {
                    'models': ['rgb', x_modality],
                    'modalities': {
                        'rgb': 'images',
                        x_modality: f'images_{x_modality}'
                    }
                }
                LOGGER.info(f"RGB单模态验证，动态确定X模态: {x_modality}")
            else:
                # 处理 'X' 特殊标记（大小写不敏感）
                if self.modality.upper() == 'X':
                    # 'X' 是特殊标记，需要解析为实际的X模态
                    actual_x_modality = self._determine_x_modality_from_data()
                    x_modality_path = self._get_x_modality_path(actual_x_modality)
                    
                    config = {
                        'models': ['rgb', actual_x_modality],
                        'modalities': {
                            'rgb': 'images',
                            actual_x_modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态验证: {actual_x_modality}-only (从'X'解析)")
                else:
                    # 用户指定了具体的模态名称（如 'depth', 'thermal', 'ir' 等）
                    x_modality_path = self._get_x_modality_path(self.modality)
                    
                    config = {
                        'models': ['rgb', self.modality],
                        'modalities': {
                            'rgb': 'images',
                            self.modality: x_modality_path
                        }
                    }
                    LOGGER.info(f"X模态单模态验证: {self.modality}-only")
            
            return config
        
        # 双模态验证：使用原有配置解析逻辑（优先从数据配置读取）
        config = self._get_default_multimodal_config()
        
        if not self.data:
            LOGGER.warning("验证器未提供数据配置，使用默认多模态配置: rgb+depth")
            return config
        
        # 解析modality_used字段（使用的模态组合）
        if 'modality_used' in self.data:
            modality_used = self.data['modality_used']

            # 验证modality_used格式
            if not isinstance(modality_used, list):
                raise ValueError(f"验证配置中'modality_used'必须是列表格式，当前为: {type(modality_used)}")

            if len(modality_used) != 2:
                raise ValueError(f"多模态验证要求恰好2个模态，当前提供: {len(modality_used)} - {modality_used}")

            if 'rgb' not in modality_used:
                raise ValueError(f"多模态验证必须包含'rgb'模态，当前: {modality_used}")

            config['models'] = modality_used
            LOGGER.info(f"验证使用配置中的模态组合: {modality_used}")
        else:
            LOGGER.info(f"验证未找到'modality_used'配置，使用默认组合: {config['models']}")
        
        # 解析modality字段（模态路径映射）
        if 'modality' in self.data:
            modality_paths = self.data['modality']

            # 验证modality格式
            if not isinstance(modality_paths, dict):
                raise ValueError(f"验证配置中'modality'必须是字典格式，当前为: {type(modality_paths)}")

            # 初始化modalities配置
            modalities = {'rgb': 'images'}  # RGB默认路径

            # 验证所有必需模态都有路径配置
            for modality in config['models']:
                if modality == 'rgb':
                    continue  # RGB已设置默认路径
                elif modality in modality_paths:
                    modalities[modality] = modality_paths[modality]
                else:
                    modalities[modality] = f'images_{modality}'  # X模态默认路径
                    LOGGER.warning(f"验证未找到'{modality}'模态路径配置，使用默认: images_{modality}")

            config['modalities'] = modalities
            LOGGER.info(f"验证使用配置中的模态路径映射: {modalities}")
        else:
            # 为当前模态组合生成默认路径映射
            x_modality = [m for m in config['models'] if m != 'rgb'][0]
            config['modalities']['rgb'] = 'images'
            config['modalities'][x_modality] = f'images_{x_modality}'
            LOGGER.info(f"验证未找到'modality'配置，生成默认路径映射: {config['modalities']}")
        
        return config
    
    def _get_x_modality_path(self, modality_name):
        """
        获取指定模态的实际路径。
        
        优先从data.yaml的modality字段读取，
        如果不存在则使用默认格式 'images_{modality_name}'。
        
        Args:
            modality_name (str): 模态名称（如 'ir', 'depth', 'thermal'）
            
        Returns:
            str: 模态对应的目录路径
        """
        # 优先从data.yaml的modality字段读取
        if self.data and 'modality' in self.data:
            modality_paths = self.data['modality']
            if isinstance(modality_paths, dict) and modality_name in modality_paths:
                return modality_paths[modality_name]
        
        # 向后兼容：检查modalities字段
        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict) and modality_name in modalities:
                return modalities[modality_name]
        
        # 如果没有配置，使用默认格式
        return f'images_{modality_name}'
    
    def _determine_x_modality_from_data(self):
        """
        智能确定X模态类型，避免硬编码depth。（与训练器完全一致）

        优先级:
        1. 从data.yaml的modality_used字段读取（最高优先级）
        2. 从data.yaml的models字段读取
        3. 从modality字段推断
        4. 从数据目录结构推断
        5. 最后使用depth作为默认值

        Returns:
            str: X模态类型标识符
        """
        # 方法1: 从data.yaml的modality_used字段读取（最高优先级）
        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                x_modalities = [m for m in modality_used if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从data.yaml的modality_used读取X模态: {x_modality}")
                    return x_modality

        # 方法2: 从data.yaml的models字段读取（向后兼容）
        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                x_modalities = [m for m in models if m != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从数据配置读取X模态: {x_modality}")
                    return x_modality
        
        # 方法3: 从modality字段推断（检查配置的模态类型）
        if self.data and 'modality' in self.data:
            modality = self.data['modality']
            if isinstance(modality, dict):
                x_modalities = [k for k in modality.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从data.yaml的modality配置推断X模态: {x_modality}")
                    return x_modality

        # 方法4: 检查modalities配置（向后兼容）
        if self.data and 'modalities' in self.data:
            modalities = self.data['modalities']
            if isinstance(modalities, dict):
                x_modalities = [k for k in modalities.keys() if k != 'rgb']
                if x_modalities:
                    x_modality = x_modalities[0]
                    LOGGER.info(f"验证-从modalities配置推断X模态: {x_modality}")
                    return x_modality

        # 方法5: 从数据目录结构推断（最低优先级）
        if self.data and 'path' in self.data:
            try:
                import os
                data_path = self.data['path']
                if os.path.exists(data_path):
                    # 查找images_xxx目录
                    for item in os.listdir(data_path):
                        if item.startswith('images_') and item != 'images':
                            x_modality = item.replace('images_', '')
                            LOGGER.info(f"验证-从目录结构推断X模态: {x_modality}")
                            return x_modality
            except Exception as e:
                LOGGER.debug(f"验证-目录结构推断失败: {e}")
        
        # 使用depth作为默认值
        LOGGER.warning("验证-无法自动确定X模态类型，使用默认值: depth")
        return 'depth'
    
    def _get_default_multimodal_config(self):
        """
        获取默认的多模态验证配置，优先从数据配置文件读取。（与训练器保持一致）
        
        Returns:
            dict: 默认多模态配置
        """
        # 优先从数据配置读取（优先检查modality_used字段）
        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                LOGGER.info(f"验证-从modality_used配置读取模态组合: {modality_used}")
                config = {
                    'models': modality_used,
                    'modalities': {
                        'rgb': 'images'  # RGB路径固定
                    }
                }
                # 为非RGB模态生成路径（从modality字段读取或使用默认）
                for modality in modality_used:
                    if modality != 'rgb':
                        if self.data and 'modality' in self.data and modality in self.data['modality']:
                            config['modalities'][modality] = self.data['modality'][modality]
                        else:
                            config['modalities'][modality] = f'images_{modality}'
                return config

        # 备选：从models字段读取
        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                LOGGER.info(f"验证-从models配置读取模态组合: {models}")
                config = {
                    'models': models,
                    'modalities': {
                        'rgb': 'images'  # RGB路径固定
                    }
                }
                # 为非RGB模态生成默认路径
                for modality in models:
                    if modality != 'rgb':
                        config['modalities'][modality] = f'images_{modality}'
                return config
        
        # 智能推断默认配置
        x_modality = self._determine_x_modality_from_data()
        config = {
            'models': ['rgb', x_modality],  # 动态确定的模态组合
            'modalities': {  # 动态生成的模态路径映射
                'rgb': 'images',
                x_modality: f'images_{x_modality}'
            }
        }
        LOGGER.info(f"验证-生成默认多模态配置: rgb+{x_modality}")
        return config

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        构建多模态验证数据集。
        
        重写父类方法，通过传递multi_modal_image=True参数启用YOLOMultiModalImageDataset，
        确保验证阶段也能正确处理多模态数据，与训练器保持一致。
        
        Args:
            img_path (str): 图像路径
            mode (str): 模式（val/test）
            batch (int, optional): 批次大小
            
        Returns:
            YOLOMultiModalImageDataset: 多模态验证数据集对象
        """
        # 延迟解析多模态配置（确保data属性已设置）
        if self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态验证配置解析完成 - 模态: {self.multimodal_config['models']}")
        
        # 使用解析后的模态配置
        modalities = self.multimodal_config['models']
        modalities_dict = self.multimodal_config['modalities']

        # 获取X模态信息
        x_modalities = [m for m in modalities if m != 'rgb']
        x_modality = x_modalities[0] if x_modalities else None
        x_modality_dir = modalities_dict.get(x_modality) if x_modality else None

        # 获取stride参数（确保已设置）
        stride = self.stride if hasattr(self, 'stride') and self.stride else 32

        # 优化日志输出，区分单模态和双模态验证，与训练器保持一致的格式
        if self.modality:
            # 单模态验证日志 - 与训练器格式保持一致
            LOGGER.info(f"构建多模态验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")
            LOGGER.info(f"启用单模态验证: {self.modality}-only，将应用智能模态填充")
        else:
            # 双模态验证日志 - 与训练器格式保持一致
            LOGGER.info(f"构建多模态验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")

        # 调用build_yolo_dataset，传递multi_modal_image=True启用多模态数据集
        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=True,  # 验证模式默认使用矩形推理
            stride=stride,
            multi_modal_image=True,  # 关键参数：启用YOLOMultiModalImageDataset
            x_modality=x_modality,  # 传递X模态类型
            x_modality_dir=x_modality_dir,  # 传递X模态目录路径
            modalities=modalities,  # 传递模态配置（向后兼容）
            # 移除train_modality参数传递，改为在验证器中实现模态消融逻辑
        )

    def init_metrics(self, model):
        """
        初始化评估指标。
        
        多模态验证使用标准YOLO评估指标：
        - mAP@0.5
        - mAP@0.5:0.95
        - Precision
        - Recall
        
        保持与DetectionValidator完全一致的评估体系。
        """
        super().init_metrics(model)
        
        # 确保stride属性被正确设置
        if model and not hasattr(self, 'stride'):
            self.stride = max(int(de_parallel(model).stride.max() if hasattr(model, 'stride') else 0), 32)
        
        # LOGGER.info("初始化多模态评估指标 - 使用标准YOLO指标")
        
    def preprocess(self, batch):
        """
        预处理批次数据，包括模态消融逻辑。
        
        确保6通道数据正确处理，保持与训练阶段一致的预处理流程。
        当启用单模态验证时，应用模态消融逻辑。
        
        Args:
            batch (dict): 包含图像和标签的批次数据
            
        Returns:
            dict: 预处理后的批次数据
        """
        # 调用父类预处理方法
        batch = super().preprocess(batch)
        
        # 支持动态通道数验证
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        expected_channels = 3 + xch
        if batch["img"].shape[1] != expected_channels:
            LOGGER.warning(f"期望{expected_channels}通道输入（RGB:3 + X:{xch}），但收到 {batch['img'].shape[1]} 通道")
            return batch
        
        # 模态消融逻辑改由路由层统一完成
        # if self.modality:
        #     self._apply_modality_ablation(batch)
        #     LOGGER.debug(f"已应用{self.modality}模态消融")
        
        return batch
        
    def _apply_modality_ablation(self, batch):
        """
        应用模态消融逻辑，通过将非选定模态的通道置零来实现单模态验证。
        
        通道映射：[RGB(0:3), X(3:3+Xch)]
        
        Args:
            batch (dict): 包含图像数据的批次
        """
        if not self.modality:
            return
        
        images = batch["img"]  # Shape: [B, 3+Xch, H, W]
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        
        if self.modality == 'rgb':
            # RGB单模态验证：将X模态通道(3:3+Xch)置零
            images[:, 3:3+xch, :, :] = 0
            LOGGER.debug(f"单模态RGB验证: X模态通道(3:{3+xch})已置零")
        elif self.modality.upper() == 'X':
            # X模态验证：将RGB通道(0:3)置零
            images[:, 0:3, :, :] = 0
            LOGGER.debug("单模态X验证: RGB通道(0:3)已置零")
        else:
            # 具体X模态验证（如depth、thermal等）：将RGB通道置零
            images[:, 0:3, :, :] = 0
            LOGGER.debug(f"单模态{self.modality}验证: RGB通道(0:3)已置零")
        
        batch["img"] = images
        
    def plot_val_samples(self, batch, ni):
        """
        绘制验证样本，支持多模态可视化。
        
        使用统一的复用组件实现，遵循[RGB, X]通道顺序，
        实现RGB、X模态、多模态并排三种可视化输出。
        可选默认灰度可视化（不启用伪彩）。
        
        Args:
            batch (dict): 批次数据
            ni (int): 批次索引
        """
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities, visualize_x_to_3ch, concat_side_by_side,
            duplicate_bboxes_for_side_by_side, ensure_batch_idx_long, resolve_x_modality
        )
        
        # 获取多模态图像数据
        multimodal_images = batch["img"]  # Shape: (batch, 3+Xch, H, W)
        
        # 动态获取X通道数
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        
        # 使用复用组件拆分模态：遵循[RGB, X]通道顺序
        rgb_images, x_images = split_modalities(multimodal_images, xch)
        
        # 获取X模态类型
        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))
        
        # 确保batch_idx类型正确
        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx
        
        # 根据验证模式决定输出
        if self.modality:
            # 单模态验证：仅输出指定模态
            if self.modality == 'rgb':
                # RGB单模态
                plot_images(
                    rgb_images,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            else:
                # X模态单模态（默认灰度，可关闭伪彩）
                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
        else:
            # 双模态验证：输出三种图像
            try:
                # 1. RGB模态验证样本
                plot_images(
                    rgb_images,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 2. X模态验证样本（默认灰度，可关闭伪彩）
                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 3. 多模态并排对比图 - 使用duplicate函数为两侧绘制bbox
                side_by_side_images = concat_side_by_side(rgb_images, x_visual)
                # 复制bbox到两侧：左半(RGB) + 右半(X)
                batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
                    batch["batch_idx"], batch["cls"].squeeze(-1), batch["bboxes"], None
                )
                plot_images(
                    side_by_side_images,
                    batch_ids_dup,
                    cls_ids_dup,
                    bboxes_dup,
                    paths=[p.replace('.jpg', '_multimodal.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
            except Exception as e:
                LOGGER.warning(f"绘制{x_modality}模态验证样本失败: {e}")
        
    def plot_predictions(self, batch, preds, ni):
        """
        绘制预测结果，支持多模态可视化。
        
        统一坐标制处理：output_to_target→像素xywh→归一化xywh→绘图，
        确保并排图bbox正确缩放到左半区域。
        默认灰度可视化（不启用伪彩）。
        
        Args:
            batch (dict): 批次数据
            preds (list): 预测结果
            ni (int): 批次索引
        """
        from ultralytics.utils.plotting import plot_images, output_to_target
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities, visualize_x_to_3ch, concat_side_by_side,
            to_norm_xywh_for_plot, duplicate_bboxes_for_side_by_side, resolve_x_modality,
            ensure_batch_idx_long, clip_boxes_norm_xywh
        )
        
        # 获取多模态图像数据
        multimodal_images = batch["img"]
        
        # 动态获取X通道数
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        
        # 使用复用组件拆分模态：遵循[RGB, X]通道顺序
        rgb_images, x_images = split_modalities(multimodal_images, xch)
        
        # 获取X模态类型
        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))
        
        # 统一坐标制：output_to_target → 像素xywh → 归一化xywh
        batch_ids, cls_ids, boxes_xywh_px, confs = output_to_target(preds, max_det=self.args.max_det)
        
        # 获取图像尺寸用于坐标归一化
        _, _, H, W = rgb_images.shape
        img_hw = (H, W)
        
        # 转换为归一化坐标用于plot_images
        batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm = to_norm_xywh_for_plot(
            batch_ids, cls_ids, boxes_xywh_px, confs, img_hw
        )
        
        # 确保batch_idx类型正确
        batch_ids_norm = ensure_batch_idx_long(batch_ids_norm)
        # 先做几何裁剪到单图域[0,1]，防止xywh分量clamp不生效导致的越界
        if (isinstance(boxes_norm, torch.Tensor) and boxes_norm.numel() > 0) or (
            isinstance(boxes_norm, np.ndarray) and boxes_norm.size > 0
        ):
            boxes_norm = clip_boxes_norm_xywh(boxes_norm, 0.0, 1.0, 0.0, 1.0)
        
        # 根据验证模式决定输出
        if self.modality:
            # 单模态验证：仅输出指定模态
            if self.modality == 'rgb':
                # RGB单模态预测
                plot_images(
                    rgb_images,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            else:
                # X模态单模态预测（默认灰度，可关闭伪彩）
                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
        else:
            # 双模态验证：输出三种预测图
            try:
                # 1. RGB预测结果
                plot_images(
                    rgb_images,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 2. X模态预测结果（默认灰度，可关闭伪彩）
                x_visual = visualize_x_to_3ch(x_images, colorize=False, x_modality=x_modality)
                plot_images(
                    x_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 3. 多模态并排预测图 - 使用duplicate函数为两侧绘制bbox
                side_by_side_images = concat_side_by_side(rgb_images, x_visual)
                # 复制bbox到两侧：左半(RGB) + 右半(X) 
                batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm
                )
                plot_images(
                    side_by_side_images,
                    batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup,
                    paths=[p.replace('.jpg', '_multimodal.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
            except Exception as e:
                LOGGER.warning(f"绘制{x_modality}模态预测结果失败: {e}")

 
