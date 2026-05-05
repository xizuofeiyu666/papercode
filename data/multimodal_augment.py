# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多模态数据增强模块 - 基于Input字段路由系统的增强策略

此模块包含专门为YOLOMM多模态模型设计的数据增强类，支持：

融合策略适配：
- 早期融合('Dual'): 6通道RGB+X统一增强处理
- 中期融合('RGB'/'X'): 独立模态增强后路由合并  
- 晚期融合: 高层特征级增强

核心特色：
- 配置驱动的增强策略选择
- 模态感知的增强算法（如RGB的HSV变换，X模态保持原始特性）
- 零拷贝tensor操作优化
- 支持任意X模态类型的增强适配
"""

import numpy as np
import cv2
from ultralytics.utils import LOGGER
from ultralytics.data.augment import (
    Mosaic,
    MixUp,
    Compose,
    RandomPerspective,
    LetterBox,
    BaseMixTransform,
)
from ultralytics.data.augment import RandomFlip as V8RandomFlip
import random
try:
    import albumentations as A
    _ALBU_OK = True
except Exception:
    _ALBU_OK = False


class MultiModalRandomHSV:
    """
    多模态随机HSV增强类 - 模态感知的颜色空间增强
    
    智能处理多模态数据的HSV增强：
    - 早期融合(6通道): 对RGB部分应用HSV变换，X模态保持不变
    - 中期融合: 通过路由系统独立处理RGB和X模态
    
    专门为RGB+X多模态图像设计，避免对深度图、热红外等X模态应用不合适的颜色变换。
    只对前3个通道(RGB)应用HSV变换，后3个通道(X模态)保持不变。
    
    这样可以避免对深度图、红外图等X模态应用不适合的颜色变换。
    
    Attributes:
        hgain (float): 色调变化的最大范围 [0, 1]
        sgain (float): 饱和度变化的最大范围 [0, 1]  
        vgain (float): 亮度变化的最大范围 [0, 1]
        
    Methods:
        __call__: 应用多模态HSV增强到输入标签
        
    Examples:
        >>> augmenter = MultiModalRandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> labels = {"img": multimodal_img}  # 6通道图像 [H,W,6]
        >>> augmented_labels = augmenter(labels)
    """
    
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        初始化多模态随机HSV增强器
        
        Args:
            hgain (float): 色调变化的最大范围，应在[0, 1]范围内
            sgain (float): 饱和度变化的最大范围，应在[0, 1]范围内  
            vgain (float): 亮度变化的最大范围，应在[0, 1]范围内
            
        Examples:
            >>> hsv_aug = MultiModalRandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        
    def __call__(self, labels):
        """
        对多模态图像应用随机HSV增强
        
        此方法只对多模态图像的前3个通道(RGB)应用HSV变换，
        其余通道(X模态)保持不变，避免破坏X模态数据的特性。
        
        Args:
            labels (Dict): 包含图像数据的标签字典，必须包含'img'键
                         'img': 6通道多模态图像 numpy数组 [H,W,6]
                         
        Returns:
            (Dict): 返回修改后的标签字典，'img'为增强后的6通道图像
            
        Examples:
            >>> augmenter = MultiModalRandomHSV(hgain=0.5, sgain=0.5, vgain=0.5) 
            >>> labels = {"img": np.random.rand(640, 640, 6).astype(np.uint8)}
            >>> result = augmenter(labels)
            >>> enhanced_img = result["img"]
        """
        img = labels["img"]
        
        # 验证输入图像格式
        if len(img.shape) != 3 or img.shape[2] < 4:
            LOGGER.warning(f"MultiModalRandomHSV expects 3+Xch image, got shape {img.shape}")
            return labels
            
        # 如果没有设置任何增强参数，直接返回
        if not (self.hgain or self.sgain or self.vgain):
            return labels
            
        # 分离RGB和X模态
        rgb_img = img[:, :, :3].copy()  # 前3通道：RGB
        x_img = img[:, :, 3:].copy()    # 后Xch通道：X模态

        # OpenCV LUT 仅支持 8-bit，若因与float的X模态拼接导致类型提升，这里将RGB显式转换
        if rgb_img.dtype != np.uint8:
            rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
        
        # 生成随机增强参数
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        
        # 转换RGB到HSV并应用增强
        hue, sat, val = cv2.split(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV))
        dtype = rgb_img.dtype  # 保持原始数据类型
        
        # 创建查找表
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        # 应用查找表变换
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        
        # 转换回BGR
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=rgb_img)
        
        # 重新组合多通道图像：增强后的RGB + 原始X模态
        enhanced_img = np.concatenate([rgb_img, x_img], axis=2)
        
        # 更新标签字典
        labels["img"] = enhanced_img
        
        return labels


# 可扩展的多模态增强基类
class BaseMultiModalTransform:
    """
    多模态变换基类
    
    为多模态数据增强提供通用接口和工具方法。
    子类应该实现__call__方法来定义具体的增强逻辑。
    
    Methods:
        split_modalities: 将6通道图像分离为RGB和X模态
        merge_modalities: 将RGB和X模态合并为6通道图像
        validate_input: 验证输入图像格式
    """
    
    @staticmethod
    def split_modalities(img):
        """
        将多通道图像分离为RGB和X模态（RGB固定前3通道，X为其余通道）
        
        Args:
            img (np.ndarray): 6通道输入图像 [H,W,6]
            
        Returns:
            tuple: (rgb_img, x_img) RGB图像和X模态图像
        """
        if len(img.shape) != 3 or img.shape[2] < 4:
            raise ValueError(f"Expected 3+Xch image, got shape {img.shape}")
        
        rgb_img = img[:, :, :3]  # 前3通道：RGB
        x_img = img[:, :, 3:]    # 后3通道：X模态
        return rgb_img, x_img
    
    @staticmethod  
    def merge_modalities(rgb_img, x_img):
        """
        将RGB和X模态合并为3+Xch通道图像
        
        Args:
            rgb_img (np.ndarray): RGB图像 [H,W,3]
            x_img (np.ndarray): X模态图像 [H,W,3]
            
        Returns:
            np.ndarray: 6通道合并图像 [H,W,6]
        """
        return np.concatenate([rgb_img, x_img], axis=2)
    
    @staticmethod
    def validate_input(img):
        """
        验证输入图像格式（至少4通道，RGB+X）
        
        Args:
            img (np.ndarray): 输入图像
            
        Returns:
            bool: 如果是有效的6通道图像返回True
        """
        return len(img.shape) == 3 and img.shape[2] >= 4


# 示例：可扩展的多模态几何变换
class MultiModalRandomFlip(BaseMultiModalTransform):
    """
    多模态随机翻转增强
    
    对6通道图像进行同步的水平或垂直翻转，
    确保RGB和X模态保持对应关系。
    
    Attributes:
        p (float): 翻转概率 [0, 1]
        direction (str): 翻转方向 'horizontal' 或 'vertical'
    """
    
    def __init__(self, p=0.5, direction="horizontal"):
        """
        初始化多模态随机翻转
        
        Args:
            p (float): 翻转概率
            direction (str): 翻转方向
        """
        assert direction in {"horizontal", "vertical"}, f"direction must be 'horizontal' or 'vertical', got {direction}"
        assert 0 <= p <= 1.0, f"probability must be in [0, 1], got {p}"
        
        self.p = p
        self.direction = direction
        
    def __call__(self, labels):
        """
        应用多模态随机翻转
        
        Args:
            labels (Dict): 包含图像的标签字典
            
        Returns:
            Dict: 处理后的标签字典
        """
        img = labels["img"]
        
        if not self.validate_input(img):
            LOGGER.warning(f"MultiModalRandomFlip expects 3+Xch image, got shape {img.shape}")
            return labels
            
        # 根据概率决定是否翻转
        if np.random.random() > self.p:
            return labels
            
        # 同步翻转整个6通道图像
        if self.direction == "horizontal":
            img = np.fliplr(img)
        else:  # vertical
            img = np.flipud(img)
            
        labels["img"] = np.ascontiguousarray(img)
        return labels


# 工具函数
def create_multimodal_transforms(rgb_transforms, preserve_x_modality=True):
    """
    从标准RGB变换创建多模态变换的工厂函数
    
    Args:
        rgb_transforms (list): RGB变换列表
        preserve_x_modality (bool): 是否保护X模态不受变换影响
        
    Returns:
        list: 适配的多模态变换列表
        
    Notes:
        这是一个扩展接口，用于将来可能的自动适配功能
    """
    # 这里可以实现自动适配逻辑
    # 目前返回空列表，作为接口预留
    LOGGER.info("create_multimodal_transforms is under development")
    return []


# 多模态Mosaic和MixUp类
class MultiModalMosaic(Mosaic):
    """
    多模态Mosaic增强类

    继承自标准Mosaic类，确保随机选择的图像索引都有完整的多模态数据。
    通过调用数据集的get_valid_indices()方法获取有效索引列表。
    """

    def get_indexes(self, buffer=True):
        """
        获取多模态Mosaic拼接的随机索引

        重写父类方法，确保选择的索引都有完整的多模态数据

        Args:
            buffer (bool): 是否从buffer选择图像（与父类保持兼容）

        Returns:
            list: n-1个随机有效索引（n为mosaic网格数）
        """
        # 获取有效的多模态索引
        if hasattr(self.dataset, 'get_valid_indices'):
            valid_indices = self.dataset.get_valid_indices()
        else:
            # 向后兼容：如果数据集没有get_valid_indices方法，使用全部索引
            valid_indices = list(range(len(self.dataset)))
            LOGGER.warning("Dataset does not have get_valid_indices method, using all indices for Mosaic")

        # 需要的索引数量：n-1（n是mosaic网格数，默认4，所以需要3个额外图像）
        num_needed = self.n - 1

        # 从有效索引中随机选择
        if len(valid_indices) < num_needed:
            LOGGER.warning(f"Not enough valid multimodal images ({len(valid_indices)}) for Mosaic augmentation")
            # 如果有效索引不够，用重复选择策略
            if len(valid_indices) == 0:
                return [0] * num_needed  # 极端情况，返回默认索引
            # 为了确保有重复，我们从有效索引中随机选择，允许重复
            return [valid_indices[np.random.randint(0, len(valid_indices))] for _ in range(num_needed)]

        # 随机选择num_needed个不同的有效索引
        selected_indices = np.random.choice(len(valid_indices), num_needed, replace=False)
        return [valid_indices[i] for i in selected_indices]


class MultiModalMixUp(MixUp):
    """
    多模态MixUp增强类

    继承自标准MixUp类，确保随机选择的图像索引都有完整的多模态数据。
    通过调用数据集的get_valid_indices()方法获取有效索引列表。
    """

    def get_indexes(self):
        """
        获取多模态MixUp混合的随机索引

        重写父类方法，确保选择的索引都有完整的多模态数据

        Returns:
            list: 1个随机有效索引
        """
        # 获取有效的多模态索引
        if hasattr(self.dataset, 'get_valid_indices'):
            valid_indices = self.dataset.get_valid_indices()
        else:
            # 向后兼容：如果数据集没有get_valid_indices方法，使用全部索引
            valid_indices = list(range(len(self.dataset)))
            LOGGER.warning("Dataset does not have get_valid_indices method, using all indices for MixUp")

        # 从有效索引中随机选择1个（MixUp需要2个图像，包括当前1个+随机1个）
        if len(valid_indices) < 2:
            LOGGER.warning(f"Not enough valid multimodal images ({len(valid_indices)}) for MixUp augmentation")
            return [valid_indices[0] if valid_indices else 0]

        # 随机选择1个有效索引
        return [valid_indices[np.random.choice(len(valid_indices))]]


__all__ = [
    "MultiModalRandomHSV",
    "BaseMultiModalTransform",
    "MultiModalRandomFlip",
    "create_multimodal_transforms",
    "MultiModalMosaic",
    "MultiModalMixUp",
]


class MultiModalAlbumentations(BaseMultiModalTransform):
    """
    RGB专属（非几何）增强：仅对前3通道 RGB 应用 Albumentations 的非空间算子。
    - 禁止任何空间类变换（如 Rotate/ShiftScaleRotate/RandomResizedCrop 等）。
    - X 通道保持不变。
    """

    def __init__(self, p: float = 1.0, cfg=None, target: str = "rgb"):
        """
        初始化 RGB/X 专属非几何增强。
        Args:
            p: 应用概率
            cfg: albumentations 配置列表 [{'type': 'RandomGamma', 'p': 0.2}, ...]
            target: 'rgb' | 'x'
        """
        assert target in {"rgb", "x"}
        self.p = p
        self.cfg = cfg or []
        self.target = target
        self._transform_cache = {}  # 按通道数缓存已构建的 Compose
        if not _ALBU_OK:
            raise ImportError(
                "MultiModalAlbumentations requires 'albumentations>=1.0.3'. "
                "Install via: pip install -U albumentations, or disable by setting mm_rgb_albu_enable=false/mm_x_non_geo_enable=false."
            )

    def _build_transform(self, channels: int):
        if not _ALBU_OK:
            return None
        # 允许的非空间算子白名单（按通道数）
        allowed_3 = {"RandomBrightnessContrast", "CLAHE", "RandomGamma", "ToGray", "ImageCompression", "GaussianNoise"}
        allowed_1 = {"CLAHE", "RandomGamma", "GaussianNoise"}
        allowed = allowed_3 if channels == 3 else allowed_1
        T = []
        if self.cfg:
            for item in self.cfg:
                tname = item.get("type")
                if tname not in allowed:
                    raise ValueError(
                        f"MultiModalAlbumentations[{self.target} ch={channels}]: transform '{tname}' not allowed (spatial or incompatible)."
                    )
                params = {k: v for k, v in item.items() if k != "type"}
                try:
                    T.append(getattr(A, tname)(**params))
                except Exception as e:
                    raise ValueError(f"Invalid params for {tname}: {e}")
        else:
            # 默认温和配置
            if channels == 3:
                T = [
                    A.RandomBrightnessContrast(p=0.2),
                    A.CLAHE(p=0.1),
                    A.RandomGamma(p=0.1),
                    A.ImageCompression(quality_range=(80, 100), p=0.05),
                ]
            else:
                T = [
                    A.RandomGamma(p=0.1),
                    A.CLAHE(p=0.1),
                    A.GaussianNoise(p=0.1),
                ]
        return A.Compose(T)

    def __call__(self, labels):
        # 概率判定（不再依赖遗留 self.transform 字段）
        if random.random() > self.p:
            return labels
        img = labels.get("img")
        if img is None or not self.validate_input(img):
            return labels
        rgb = img[:, :, :3]
        x = img[:, :, 3:]

        if self.target == "rgb":
            ch = rgb.shape[2]
            key = (self.target, ch)
            if key not in self._transform_cache:
                self._transform_cache[key] = self._build_transform(ch)
            t = self._transform_cache[key]
            if t is not None:
                rgb = t(image=rgb)["image"]
        else:  # target == 'x'
            ch = x.shape[2] if x.ndim == 3 else 1
            if x.ndim == 2:
                x = x[:, :, np.newaxis]
                ch = 1
            key = (self.target, ch)
            if key not in self._transform_cache:
                self._transform_cache[key] = self._build_transform(ch)
            t = self._transform_cache[key]
            if t is not None:
                x = t(image=x)["image"]

        labels["img"] = np.concatenate([rgb, x], axis=2)
        return labels


class MultiModalAsyncPerturb(BaseMultiModalTransform):
    """
    微小异步几何扰动：对指定模态（'x'/'rgb'/'random'）施加极小平移/旋转，模拟传感器微错位。
    - 仅改变一个模态，另一侧保持不动；
    - 不修改 labels['instances']。
    """

    def __init__(self, p: float = 0.0, translate_px=(1, 2), rotate_deg=(-0.5, 0.5), target: str = "x", border: int = 114):
        assert 0.0 <= p <= 1.0
        assert target in {"x", "rgb", "random"}
        self.p = p
        self.translate_px = translate_px
        self.rotate_deg = rotate_deg
        self.target = target
        self.border = border

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        img = labels.get("img")
        if img is None or not self.validate_input(img):
            return labels
        H, W, C = img.shape
        rgb = img[:, :, :3].copy()
        x = img[:, :, 3:].copy()
        # 选择目标模态
        target = self.target
        if target == "random":
            target = random.choice(["rgb", "x"]) if C >= 4 else "rgb"

        # 生成仿射参数
        tx = random.randint(self.translate_px[0], self.translate_px[1]) * random.choice([-1, 1])
        ty = random.randint(self.translate_px[0], self.translate_px[1]) * random.choice([-1, 1])
        angle = random.uniform(self.rotate_deg[0], self.rotate_deg[1])
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        M[:, 2] += (tx, ty)

        if target == "rgb":
            rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(self.border, self.border, self.border))
        else:
            # X 通道 warp，保持通道数
            if x.ndim == 3 and x.shape[2] > 0:
                x = cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(self.border, self.border, self.border))

        labels["img"] = np.concatenate([rgb, x], axis=2)
        return labels


class MultiModalModalDropout(BaseMultiModalTransform):
    """
    模态丢失：在指定模态的随机矩形区域进行抹除（zero/one/const/noise），另一模态保持不变。
    - 用于提升在单模态缺失/遮挡场景下的鲁棒性。
    """

    def __init__(self, p: float = 0.0, area: tuple = (0.02, 0.15), ratio: tuple = (0.5, 2.0),
                 target: str = "x", fill: str = "zero", fill_const: int = 114):
        assert 0.0 <= p <= 1.0
        assert target in {"x", "rgb", "random"}
        assert fill in {"zero", "one", "const", "noise"}
        self.p = p
        self.area = area
        self.ratio = ratio
        self.target = target
        self.fill = fill
        self.fill_const = fill_const

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        img = labels.get("img")
        if img is None or not self.validate_input(img):
            return labels
        H, W, C = img.shape
        rgb = img[:, :, :3].copy()
        x = img[:, :, 3:].copy()

        # 采样矩形
        area = random.uniform(*self.area) * H * W
        aspect = random.uniform(*self.ratio)
        h = int(max(1, round(np.sqrt(area / aspect))))
        w = int(max(1, round(np.sqrt(area * aspect))))
        if h > H: h = H
        if w > W: w = W
        y1 = random.randint(0, H - h)
        x1 = random.randint(0, W - w)

        def _fill_patch(tensor):
            if self.fill == "zero":
                tensor[y1:y1 + h, x1:x1 + w, ...] = 0
            elif self.fill == "one":
                tensor[y1:y1 + h, x1:x1 + w, ...] = 255
            elif self.fill == "const":
                tensor[y1:y1 + h, x1:x1 + w, ...] = self.fill_const
            else:  # noise
                patch_shape = tensor[y1:y1 + h, x1:x1 + w, ...].shape
                tensor[y1:y1 + h, x1:x1 + w, ...] = np.random.randint(0, 256, size=patch_shape, dtype=tensor.dtype)

        target = self.target
        if target == "random":
            target = random.choice(["rgb", "x"]) if C >= 4 else "rgb"
        if target == "rgb":
            _fill_patch(rgb)
        else:
            _fill_patch(x)

        labels["img"] = np.concatenate([rgb, x], axis=2)
        return labels


class MultiModalIRNonGeoAug(BaseMultiModalTransform):
    """
    IR 专属非空间增强（仅作用于 X 模态），提升热信号对比与可读性。
    - 严格非几何：不改变几何与标注，仅逐像素映射；
    - 通道与 dtype 保持：支持 Xch=1/3，保持输入 dtype 与通道数；
    - 白名单算子：windowing/clahe/gamma/invert_hot/noise/drift。
    """

    _IR_SYNONYMS = {"ir", "infrared", "thermal", "the"}

    def __init__(self, ops_cfg=None):
        self.ops = self._normalize_ops_cfg(ops_cfg)

    @staticmethod
    def _normalize_ops_cfg(ops_cfg):
        if ops_cfg is None:
            return {
                "windowing": {"p": 0.5, "lo_hi_percentile": [[3, 97], [5, 95]]},
                "clahe": {"p": 0.3, "clip_limit": [1.5, 3.0], "tile_grid": [8, 8]},
                "gamma": {"p": 0.3, "range": [0.7, 1.4]},
                "invert_hot": {"p": 0.1},
                "noise": {"p": 0.2, "sigma": [1, 3]},
                "drift": {"p": 0.2, "offset": [-5, 5], "scale": [0.95, 1.05]},
            }
        # 兼容 list-of-singleton-dicts 与 dict 两种形式
        if isinstance(ops_cfg, list):
            norm = {}
            for item in ops_cfg:
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError("mm_ir_ops_cfg list items must be single-key dicts like [{'gamma': {...}}]")
                k = list(item.keys())[0]
                norm[k] = item[k]
            return norm
        if not isinstance(ops_cfg, dict):
            raise ValueError("mm_ir_ops_cfg must be dict or list of dicts")
        return ops_cfg

    @staticmethod
    def _ensure_3d(x):
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        return x

    @staticmethod
    def _maxv_from_dtype(dtype):
        if dtype == np.uint8:
            return 255
        if dtype == np.uint16:
            return 65535
        # 其他类型视为 0..1 浮点范围
        return 1.0

    @staticmethod
    def _rand_from_range(r):
        if isinstance(r, (list, tuple)) and len(r) == 2:
            return random.uniform(r[0], r[1])
        return float(r)

    @staticmethod
    def _choice_from_list(L):
        if isinstance(L, (list, tuple)) and len(L) > 0:
            return random.choice(L)
        return L

    @staticmethod
    def _clip_cast(x, maxv, dtype):
        x = np.clip(x, 0, maxv)
        if dtype == np.uint8:
            return x.astype(np.uint8)
        if dtype == np.uint16:
            return x.astype(np.uint16)
        return x.astype(np.float32)

    def _op_windowing(self, x, cfg, maxv, dtype):
        percs = cfg.get("lo_hi_percentile", [[5, 95]])
        lohi = self._choice_from_list(percs)
        lo_p, hi_p = float(lohi[0]), float(lohi[1])
        x_f = x.astype(np.float32)
        H, W, C = x.shape
        for c in range(C):
            ch = x_f[:, :, c]
            lo = np.percentile(ch, lo_p)
            hi = np.percentile(ch, hi_p)
            if hi <= lo:
                continue
            ch = (ch - lo) / (hi - lo) * maxv
            x_f[:, :, c] = ch
        return self._clip_cast(x_f, maxv, dtype)

    def _op_clahe(self, x, cfg, maxv, dtype):
        clip_limit = self._rand_from_range(cfg.get("clip_limit", [1.5, 3.0]))
        tile_grid = cfg.get("tile_grid", [8, 8])
        H, W, C = x.shape
        out = np.empty_like(x)
        for c in range(C):
            ch = x[:, :, c]
            if dtype == np.uint8:
                src = ch
                clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
                out[:, :, c] = clahe.apply(src)
            elif dtype == np.uint16:
                # 16-bit 转 8-bit 处理后再还原
                scale = 255.0 / maxv
                ch8 = (ch.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
                y8 = clahe.apply(ch8).astype(np.float32)
                out[:, :, c] = self._clip_cast(y8 / 255.0 * maxv, maxv, dtype)
            else:
                # 浮点：按 0..1 归一至 0..255 处理
                ch8 = (ch * 255.0).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
                y8 = clahe.apply(ch8).astype(np.float32)
                out[:, :, c] = np.clip(y8 / 255.0, 0, 1.0)
        return out

    def _op_gamma(self, x, cfg, maxv, dtype):
        g = self._rand_from_range(cfg.get("range", [0.7, 1.4]))
        if dtype in (np.uint8, np.uint16):
            L = int(maxv) + 1
            idx = np.arange(L, dtype=np.float32) / maxv
            lut = (np.power(idx, g) * maxv).round().clip(0, maxv).astype(dtype)
            if x.ndim == 3 and x.shape[2] > 1:
                return cv2.LUT(x, lut)
            else:
                return self._clip_cast(lut[x], maxv, dtype)
        else:
            x_f = np.power(x.astype(np.float32) / maxv, g) * maxv
            return self._clip_cast(x_f, maxv, dtype)

    def _op_invert(self, x, maxv, dtype):
        return self._clip_cast(maxv - x, maxv, dtype)

    def _op_noise(self, x, cfg, maxv, dtype):
        sigma = self._rand_from_range(cfg.get("sigma", [1, 3]))
        scale = (maxv / 255.0) if dtype in (np.uint16,) else 1.0
        noise = np.random.normal(0.0, sigma * scale, size=x.shape).astype(np.float32)
        x_f = x.astype(np.float32) + noise
        return self._clip_cast(x_f, maxv, dtype)

    def _op_drift(self, x, cfg, maxv, dtype):
        offset = self._rand_from_range(cfg.get("offset", [-5, 5]))
        scale = self._rand_from_range(cfg.get("scale", [0.95, 1.05]))
        off_scale = (maxv / 255.0) if dtype in (np.uint16,) else 1.0
        x_f = x.astype(np.float32) * float(scale) + float(offset) * off_scale
        return self._clip_cast(x_f, maxv, dtype)

    # 压缩增强（compression）已从项目移除

    def _apply_ir_op(self, op_name, x, cfg, maxv, dtype):
        if op_name == "windowing":
            return self._op_windowing(x, cfg, maxv, dtype)
        if op_name == "clahe":
            return self._op_clahe(x, cfg, maxv, dtype)
        if op_name == "gamma":
            return self._op_gamma(x, cfg, maxv, dtype)
        if op_name == "invert_hot":
            return self._op_invert(x, maxv, dtype)
        if op_name == "noise":
            return self._op_noise(x, cfg, maxv, dtype)
        if op_name == "drift":
            return self._op_drift(x, cfg, maxv, dtype)
        # 不再支持 'compression' 算子
        raise ValueError(f"Unsupported IR op '{op_name}'")

    def __call__(self, labels):
        img = labels.get("img")
        if img is None or not self.validate_input(img):
            return labels
        rgb = img[:, :, :3]
        x = img[:, :, 3:]
        x = self._ensure_3d(x)
        dtype = x.dtype
        maxv = self._maxv_from_dtype(dtype)

        for name, cfg in self.ops.items():
            p = float(cfg.get("p", 1.0))
            if random.random() > p:
                continue
            x = self._apply_ir_op(name, x, cfg, maxv, dtype)

        labels["img"] = np.concatenate([rgb, x], axis=2)
        return labels


class MultiModalDepthNonGeoAug(BaseMultiModalTransform):
    """
    深度/视差专属非空间增强（仅作用于 X 模态）。
    - 目标：模拟真实深度传感器的量化、噪声、标定漂移与缺测特性；
    - 约束：严格非几何，不改变标注与几何。
    """

    _DEPTH_SYNONYMS = {"depth", "dep", "disparity", "tof", "lidar", "z", "dmap"}

    def __init__(self, ops_cfg=None):
        self.ops = self._normalize_ops_cfg(ops_cfg)

    @staticmethod
    def _normalize_ops_cfg(ops_cfg):
        if ops_cfg is None:
            return {
                "windowing": {"p": 0.5, "lo_hi_percentile": [[2, 98], [5, 95]]},
                "calib_drift": {"p": 0.3, "scale": [0.98, 1.02], "offset": [-5, 5]},
                "quantize": {"p": 0.2, "bits": [10, 12, 16]},
                "noise": {"p": 0.3, "type": "gauss", "k0": [0, 2], "k1": [0.0, 0.01]},
                "gamma": {"p": 0.2, "range": [0.9, 1.1]},
                "clahe": {"p": 0.2, "clip_limit": [1.5, 3.0], "tile_grid": [8, 8]},
                "holes": {"p": 0.2, "mode": "sparse", "density": [0.005, 0.02], "preserve_invalid": True},
            }
        if isinstance(ops_cfg, list):
            norm = {}
            for item in ops_cfg:
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError("mm_depth_ops_cfg list items must be single-key dicts like [{'gamma': {...}}]")
                k = list(item.keys())[0]
                norm[k] = item[k]
            return norm
        if not isinstance(ops_cfg, dict):
            raise ValueError("mm_depth_ops_cfg must be dict or list of dicts")
        return ops_cfg

    @staticmethod
    def _ensure_3d(x):
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        return x

    @staticmethod
    def _maxv_from_dtype(dtype):
        if dtype == np.uint8:
            return 255
        if dtype == np.uint16:
            return 65535
        return 1.0

    @staticmethod
    def _clip_cast(x, maxv, dtype):
        x = np.clip(x, 0, maxv)
        if dtype == np.uint8:
            return x.astype(np.uint8)
        if dtype == np.uint16:
            return x.astype(np.uint16)
        return x.astype(np.float32)

    @staticmethod
    def _rand_from_range(r):
        if isinstance(r, (list, tuple)) and len(r) == 2:
            return random.uniform(r[0], r[1])
        return float(r)

    @staticmethod
    def _choice_from_list(L):
        if isinstance(L, (list, tuple)) and len(L) > 0:
            return random.choice(L)
        return L

    @staticmethod
    def _make_invalid_mask(x, dtype, invalid_code=None):
        H, W, C = x.shape
        mask = np.zeros((H, W, 1), dtype=bool)
        if invalid_code is None:
            if dtype in (np.uint8, np.uint16):
                invalid_candidates = [0]
            else:
                invalid_candidates = [np.nan]
        else:
            invalid_candidates = invalid_code if isinstance(invalid_code, (list, tuple)) else [invalid_code]
        for code in invalid_candidates:
            if isinstance(code, float) and np.isnan(code):
                mask |= np.any(np.isnan(x), axis=2, keepdims=True)
            else:
                mask |= np.any(x == code, axis=2, keepdims=True)
        return mask

    def _apply_windowing(self, x, cfg, maxv, dtype, invalid_mask):
        percs = cfg.get("lo_hi_percentile", [[5, 95]])
        lohi = self._choice_from_list(percs)
        lo_p, hi_p = float(lohi[0]), float(lohi[1])
        out = x.astype(np.float32).copy()
        H, W, C = x.shape
        for c in range(C):
            ch = x[:, :, c].astype(np.float32)
            valid = ~invalid_mask[:, :, 0]
            if np.count_nonzero(valid) < 10:
                continue
            vals = ch[valid]
            lo = np.percentile(vals, lo_p)
            hi = np.percentile(vals, hi_p)
            if hi <= lo:
                continue
            ch2 = (ch - lo) / (hi - lo) * maxv
            out[:, :, c] = ch2
        out = self._clip_cast(out, maxv, dtype)
        out[invalid_mask.repeat(out.shape[2], axis=2)] = x[invalid_mask.repeat(out.shape[2], axis=2)]
        return out

    def _apply_calib_drift(self, x, cfg, maxv, dtype, invalid_mask):
        a = self._rand_from_range(cfg.get("scale", [0.98, 1.02]))
        b = self._rand_from_range(cfg.get("offset", [-5, 5]))
        out = x.astype(np.float32)
        out = out * float(a) + float(b) * (maxv / 255.0 if dtype == np.uint16 else 1.0)
        out = self._clip_cast(out, maxv, dtype)
        out[invalid_mask.repeat(out.shape[2], axis=2)] = x[invalid_mask.repeat(out.shape[2], axis=2)]
        return out

    def _apply_quantize(self, x, cfg, maxv, dtype, invalid_mask):
        bits = int(round(self._choice_from_list(cfg.get("bits", [12]))))
        levels = max(2, 1 << bits)
        step = maxv / (levels - 1)
        out = x.astype(np.float32)
        out = np.round(out / step) * step
        out = self._clip_cast(out, maxv, dtype)
        out[invalid_mask.repeat(out.shape[2], axis=2)] = x[invalid_mask.repeat(out.shape[2], axis=2)]
        return out

    def _apply_noise(self, x, cfg, maxv, dtype, invalid_mask):
        t = str(cfg.get("type", "gauss")).lower()
        k0 = self._rand_from_range(cfg.get("k0", [0, 2]))
        k1 = self._rand_from_range(cfg.get("k1", [0.0, 0.01]))
        x_f = x.astype(np.float32)
        z_norm = x_f / float(maxv)
        sigma = k0 + k1 * z_norm  # H,W,C
        if dtype == np.uint16:
            scale = maxv / 255.0
        else:
            scale = 1.0
        if t == "speckle":
            n = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
            y = x_f + x_f * n * scale
        else:
            n = np.random.normal(0.0, sigma * scale, size=x.shape).astype(np.float32)
            y = x_f + n
        y = self._clip_cast(y, maxv, dtype)
        y[invalid_mask.repeat(y.shape[2], axis=2)] = x[invalid_mask.repeat(y.shape[2], axis=2)]
        return y

    def _apply_gamma(self, x, cfg, maxv, dtype, invalid_mask):
        g = self._rand_from_range(cfg.get("range", [0.95, 1.05]))
        if dtype in (np.uint8, np.uint16):
            L = int(maxv) + 1
            idx = np.arange(L, dtype=np.float32) / maxv
            lut = (np.power(idx, g) * maxv).round().clip(0, maxv).astype(dtype)
            if x.ndim == 3 and x.shape[2] > 1:
                y = cv2.LUT(x, lut)
            else:
                y = self._clip_cast(lut[x], maxv, dtype)
        else:
            y = np.power(x.astype(np.float32) / maxv, g) * maxv
            y = self._clip_cast(y, maxv, dtype)
        y[invalid_mask.repeat(y.shape[2], axis=2)] = x[invalid_mask.repeat(y.shape[2], axis=2)]
        return y

    def _apply_clahe(self, x, cfg, maxv, dtype, invalid_mask):
        clip_limit = self._rand_from_range(cfg.get("clip_limit", [1.5, 3.0]))
        tile_grid = cfg.get("tile_grid", [8, 8])
        H, W, C = x.shape
        out = np.empty_like(x)
        for c in range(C):
            ch = x[:, :, c]
            if dtype == np.uint8:
                src = ch
                clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
                out[:, :, c] = clahe.apply(src)
            elif dtype == np.uint16:
                scale = 255.0 / maxv
                ch8 = (ch.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
                y8 = clahe.apply(ch8).astype(np.float32)
                out[:, :, c] = self._clip_cast(y8 / 255.0 * maxv, maxv, dtype)
            else:
                ch8 = (ch * 255.0).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
                y8 = clahe.apply(ch8).astype(np.float32)
                out[:, :, c] = np.clip(y8 / 255.0, 0, 1.0)
        out[invalid_mask.repeat(out.shape[2], axis=2)] = x[invalid_mask.repeat(out.shape[2], axis=2)]
        return out

    def _apply_holes(self, x, cfg, maxv, dtype, invalid_mask):
        mode = str(cfg.get("mode", "sparse")).lower()
        dens = self._rand_from_range(cfg.get("density", [0.005, 0.02]))
        preserve_invalid = bool(cfg.get("preserve_invalid", True))
        H, W, C = x.shape
        num = int(round(H * W * dens))
        if num <= 0:
            return x
        ys = np.random.randint(0, H, size=num)
        xs = np.random.randint(0, W, size=num)
        out = x.copy()
        # 使用无效编码或常量/噪声填充：此处采用无效编码 0（整数）或 NaN（浮点）作为默认
        if dtype in (np.uint8, np.uint16):
            fill = 0
        else:
            fill = np.nan
        out[ys, xs, :] = fill
        if preserve_invalid:
            # 保留原有无效像素
            out[invalid_mask.repeat(out.shape[2], axis=2)] = x[invalid_mask.repeat(out.shape[2], axis=2)]
        return out

    def _apply_op(self, name, x, cfg, maxv, dtype, invalid_mask):
        if name == "windowing":
            return self._apply_windowing(x, cfg, maxv, dtype, invalid_mask)
        if name == "calib_drift":
            return self._apply_calib_drift(x, cfg, maxv, dtype, invalid_mask)
        if name == "quantize":
            return self._apply_quantize(x, cfg, maxv, dtype, invalid_mask)
        if name == "noise":
            return self._apply_noise(x, cfg, maxv, dtype, invalid_mask)
        if name == "gamma":
            return self._apply_gamma(x, cfg, maxv, dtype, invalid_mask)
        if name == "clahe":
            return self._apply_clahe(x, cfg, maxv, dtype, invalid_mask)
        if name == "holes":
            return self._apply_holes(x, cfg, maxv, dtype, invalid_mask)
        raise ValueError(f"Unsupported depth op '{name}'")

    def __call__(self, labels):
        img = labels.get("img")
        if img is None or not self.validate_input(img):
            return labels
        rgb = img[:, :, :3]
        x = img[:, :, 3:]
        x = self._ensure_3d(x)
        dtype = x.dtype
        maxv = self._maxv_from_dtype(dtype)
        invalid_mask = self._make_invalid_mask(x, dtype)

        for name, cfg in self.ops.items():
            p = float(cfg.get("p", 1.0))
            if random.random() > p:
                continue
            x = self._apply_op(name, x, cfg, maxv, dtype, invalid_mask)

        labels["img"] = np.concatenate([rgb, x], axis=2)
        return labels


def mm_transforms(dataset, imgsz: int, hyp) -> Compose:
    """
    多模态专用增强链（独立于 v8_transforms，复用其可组合组件）。
    返回 Compose（不包含 Format），由数据集 build_transforms 追加 Format。
    """
    use_obb = getattr(dataset, "use_obb", False)
    # 多模态 Mosaic + Affine
    mm_mosaic = MultiModalMosaic(dataset, imgsz=imgsz, p=getattr(hyp, 'mosaic', 0.0))
    # 为与 v8_transforms 行为一致：在 RandomPerspective 前固定 LetterBox 到 (imgsz, imgsz)
    # 这样即使 mosaic 关闭，批内图像也会被统一到相同尺寸，避免 collate_fn 堆叠失败
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mm_mosaic, affine])

    tfl = [pre_transform]

    # 异步几何扰动（可选，仅一侧模态）—— OBB 下关闭以避免标签不一致
    if not use_obb and getattr(hyp, 'mm_async_geo_p', 0.0) > 0.0:
        tfl.append(
            MultiModalAsyncPerturb(
                p=getattr(hyp, 'mm_async_geo_p', 0.0),
                translate_px=tuple(getattr(hyp, 'mm_async_geo_translate_px', [1, 2])),
                rotate_deg=tuple(getattr(hyp, 'mm_async_geo_rotate_deg', [-0.5, 0.5])),
                target=getattr(hyp, 'mm_async_geo_target', 'x'),
                border=getattr(hyp, 'mm_async_geo_border', 114),
            )
        )

    # MixUp（使用多模态版本 + pre_transform）
    tfl.append(MultiModalMixUp(dataset, pre_transform=pre_transform, p=getattr(hyp, 'mixup', 0.0)))

    # 模态丢失（可选，仅一侧模态）—— OBB 下关闭以避免几何错位
    if not use_obb and getattr(hyp, 'mm_modal_dropout_p', 0.0) > 0.0:
        tfl.append(
            MultiModalModalDropout(
                p=getattr(hyp, 'mm_modal_dropout_p', 0.0),
                area=tuple(getattr(hyp, 'mm_modal_dropout_area', [0.02, 0.15])),
                ratio=tuple(getattr(hyp, 'mm_modal_dropout_ratio', [0.5, 2.0])),
                target=getattr(hyp, 'mm_modal_dropout_target', 'x'),
                fill=getattr(hyp, 'mm_modal_dropout_fill', 'zero'),
                fill_const=getattr(hyp, 'mm_modal_dropout_fill_const', 114),
            )
        )

    # IR/X 专属非空间增强（已按项目要求脱钩，不再注入）
    # if getattr(hyp, 'mm_ir_aug_enable', False):
    #     # 触发判定：优先 dataset.x_modality，其次 data['x_modality']
    #     x_mod = None
    #     try:
    #         x_mod = getattr(dataset, 'x_modality', None)
    #     except Exception:
    #         x_mod = None
    #     if not x_mod and isinstance(getattr(dataset, 'data', None), dict):
    #         x_mod = dataset.data.get('x_modality')
    #     if x_mod is None:
    #         LOGGER.warning("mm_ir_aug_enable=true 但未检测到数据集 x_modality 配置，已跳过 IR 专属增强注入。")
    #     else:
    #         if str(x_mod).lower() in MultiModalIRNonGeoAug._IR_SYNONYMS:
    #             tfl.append(MultiModalIRNonGeoAug(ops_cfg=getattr(hyp, 'mm_ir_ops_cfg', None)))

    # Depth/X 专属非空间增强（已按项目要求脱钩，不再注入）
    # if getattr(hyp, 'mm_depth_aug_enable', False):
    #     x_mod = None
    #     try:
    #         x_mod = getattr(dataset, 'x_modality', None)
    #     except Exception:
    #         x_mod = None
    #     if not x_mod and isinstance(getattr(dataset, 'data', None), dict):
    #         x_mod = dataset.data.get('x_modality')
    #     if x_mod is None:
    #         LOGGER.warning("mm_depth_aug_enable=true 但未检测到数据集 x_modality 配置，已跳过 Depth 专属增强注入。")
    #     else:
    #         if str(x_mod).lower() in MultiModalDepthNonGeoAug._DEPTH_SYNONYMS:
    #             tfl.append(MultiModalDepthNonGeoAug(ops_cfg=getattr(hyp, 'mm_depth_ops_cfg', None)))

    # RGB专属（非几何）增强
    if getattr(hyp, 'mm_rgb_albu_enable', False):
        tfl.append(MultiModalAlbumentations(p=1.0, cfg=getattr(hyp, 'mm_rgb_albu_cfg', []), target='rgb'))

    # 可选：X 专属非几何增强（默认关闭）
    if getattr(hyp, 'mm_x_non_geo_enable', False):
        tfl.append(MultiModalAlbumentations(p=1.0, cfg=getattr(hyp, 'mm_x_non_geo_cfg', []), target='x'))

    # RGB HSV
    tfl.append(MultiModalRandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v))

    # 翻转（复用 v8 组件，几何同步）
    tfl.append(V8RandomFlip(direction="vertical", p=hyp.flipud))
    tfl.append(V8RandomFlip(direction="horizontal", p=hyp.fliplr))

    return Compose(tfl)


# 导出新增符号
__all__ += [
    "MultiModalAlbumentations",
    "MultiModalAsyncPerturb",
    "MultiModalModalDropout",
    "MultiModalIRNonGeoAug",
    "MultiModalDepthNonGeoAug",
    "mm_transforms",
]

# -----------------------------
# Segmentation-specific chain
# -----------------------------

def mm_seg_transforms(dataset, imgsz: int, hyp) -> Compose:
    """
    多模态分割专用增强链。

    与 mm_transforms 一致使用多模态 Mosaic/RandomPerspective/MixUp 等，
    并确保几何类算子对 'segments' 同步生效。返回 Compose（不包含 Format），
    由数据集 build_transforms 统一追加 Format(return_mask=True)。
    """
    # Segmentation-friendly Mosaic + Affine (pre-transform)
    mm_mosaic = MultiModalMosaic(dataset, imgsz=imgsz, p=getattr(hyp, 'mosaic', 0.0))
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mm_mosaic, affine])

    tfl = [pre_transform]

    # Instance-level Copy-Paste (segmentation-aware), safe for multi-channel
    cp_p = float(getattr(hyp, 'copy_paste', 0.0) or 0.0)
    cp_mode = str(getattr(hyp, 'copy_paste_mode', 'flip'))
    if cp_p > 0.0:
        if cp_mode == 'flip':
            # Insert between Mosaic and Affine for flip mode
            tfl.insert(1, MultiModalCopyPaste(p=cp_p, mode=cp_mode))
            tfl.insert(2, MultiModalSegCleanup())  # post-cleanup
        else:
            # Compose a donor-pre-transform like v8 for mixup mode
            donor_pre = Compose([MultiModalMosaic(dataset, imgsz=imgsz, p=getattr(hyp, 'mosaic', 0.0)), affine])
            tfl.append(MultiModalCopyPaste(dataset=dataset, pre_transform=donor_pre, p=cp_p, mode=cp_mode))
            tfl.append(MultiModalSegCleanup())

    # Async perturb (optional)
    if getattr(hyp, 'mm_async_geo_p', 0.0) > 0.0:
        tfl.append(
            MultiModalAsyncPerturb(
                p=getattr(hyp, 'mm_async_geo_p', 0.0),
                translate_px=tuple(getattr(hyp, 'mm_async_geo_translate_px', [1, 2])),
                rotate_deg=tuple(getattr(hyp, 'mm_async_geo_rotate_deg', [-0.5, 0.5])),
                target=getattr(hyp, 'mm_async_geo_target', 'x'),
                border=getattr(hyp, 'mm_async_geo_border', 114),
            )
        )

    # MixUp after pre_transform
    tfl.append(MultiModalMixUp(dataset, pre_transform=pre_transform, p=getattr(hyp, 'mixup', 0.0)))

    # IR/Depth X-modality non-geo aug (optional, non-spatial; preserve labels)
    x_mod = None
    try:
        x_mod = getattr(dataset, 'x_modality', None)
    except Exception:
        x_mod = None
    if x_mod is None and isinstance(getattr(dataset, 'data', None), dict):
        x_mod = dataset.data.get('x_modality')

    if getattr(hyp, 'mm_ir_aug_enable', False) and x_mod is not None:
        if str(x_mod).lower() in MultiModalIRNonGeoAug._IR_SYNONYMS:
            tfl.append(MultiModalIRNonGeoAug(ops_cfg=getattr(hyp, 'mm_ir_ops_cfg', None)))

    if getattr(hyp, 'mm_depth_aug_enable', False) and x_mod is not None:
        # Depth synonyms defined inside class
        try:
            syn = getattr(MultiModalDepthNonGeoAug, '_DEPTH_SYNONYMS', {"depth", "d"})
        except Exception:
            syn = {"depth"}
        if str(x_mod).lower() in syn:
            tfl.append(MultiModalDepthNonGeoAug(ops_cfg=getattr(hyp, 'mm_depth_ops_cfg', None)))

    # Optional modal dropout
    if getattr(hyp, 'mm_modal_dropout_p', 0.0) > 0.0:
        tfl.append(
            MultiModalModalDropout(
                p=getattr(hyp, 'mm_modal_dropout_p', 0.0),
                area=tuple(getattr(hyp, 'mm_modal_dropout_area', [0.02, 0.15])),
                ratio=tuple(getattr(hyp, 'mm_modal_dropout_ratio', [0.5, 2.0])),
                target=getattr(hyp, 'mm_modal_dropout_target', 'x'),
                fill=getattr(hyp, 'mm_modal_dropout_fill', 'zero'),
                fill_const=getattr(hyp, 'mm_modal_dropout_fill_const', 114),
            )
        )

    # RGB-only non-geo
    if getattr(hyp, 'mm_rgb_albu_enable', False):
        tfl.append(MultiModalAlbumentations(p=1.0, cfg=getattr(hyp, 'mm_rgb_albu_cfg', []), target='rgb'))

    # X-only non-geo (optional)
    if getattr(hyp, 'mm_x_non_geo_enable', False):
        tfl.append(MultiModalAlbumentations(p=1.0, cfg=getattr(hyp, 'mm_x_non_geo_cfg', []), target='x'))

    # HSV on RGB
    tfl.append(MultiModalRandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v))

    # Flips
    tfl.append(V8RandomFlip(direction="vertical", p=hyp.flipud))
    tfl.append(V8RandomFlip(direction="horizontal", p=hyp.fliplr))

    return Compose(tfl)

__all__ += ["mm_seg_transforms", "MultiModalCopyPaste", "MultiModalSegCleanup"]


# -----------------------------
# MultiModal Copy-Paste & Cleanup
# -----------------------------

from copy import deepcopy
from ultralytics.utils.metrics import bbox_ioa


class MultiModalCopyPaste(BaseMixTransform):
    """
    Segmentation-aware Copy-Paste that is safe for multi-channel (RGB+X) arrays.

    It composites donor instances onto the target using a single-channel binary
    mask derived from polygons, then applies the same replacement across all
    channels. Follows v8 behavior for index sampling and overlap filtering.
    """

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip") -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {"flip", "mixup"}, f"Expected `mode` to be `flip` or `mixup`, but got {mode}."
        self.mode = mode

    def _mix_transform(self, labels: dict) -> dict:
        labels2 = labels["mix_labels"][0]
        return self._transform(labels, labels2)

    def __call__(self, labels: dict) -> dict:
        inst = labels.get("instances")
        if inst is None or len(inst.segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            return self._transform(labels)

        # donor indices
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            mix_labels = [self.pre_transform(m) for m in mix_labels]
        labels["mix_labels"] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _transform(self, labels1: dict, labels2: dict = {}) -> dict:
        import cv2
        import numpy as np

        im = labels1["img"]
        H, W = im.shape[:2]
        cls = labels1["cls"]
        inst1 = labels1.pop("instances")
        inst1.convert_bbox(format="xyxy")
        inst1.denormalize(W, H)

        inst2 = labels2.pop("instances", None)
        if inst2 is None:
            inst2 = deepcopy(inst1)
            inst2.fliplr(W)

        # filter donor instances by low IoA overlap
        ioa = bbox_ioa(inst2.bboxes, inst1.bboxes)
        idx = np.nonzero((ioa < 0.30).all(1))[0]
        if idx.size == 0:
            labels1["instances"], labels1["cls"] = inst1, cls
            return labels1

        # sort by max overlap ascending and select p ratio
        sorted_idx = np.argsort(ioa.max(1)[idx])
        idx = idx[sorted_idx]
        n = len(idx)
        take = idx[: max(1, round(self.p * n))]

        # build binary mask from donor polygons
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(inst2.segments):
            for j in take:
                poly = inst2.segments[[j]].astype(np.int32)
                cv2.drawContours(mask, poly, -1, 1, cv2.FILLED)

        # donor image (if provided) else flip target
        donor_img = labels2.get("img", cv2.flip(im, 1))
        if donor_img.ndim == 2:
            donor_img = donor_img[..., None]

        # composite across all channels
        m = mask.astype(bool)
        im[m, :] = donor_img[m, :]

        # concat instances & classes
        from ultralytics.utils.instance import Instances

        cls = np.concatenate((cls, labels2.get("cls", cls)[take]), axis=0)
        new_inst = Instances.concatenate((inst1, inst2[take]), axis=0)
        new_inst.clip(W, H)
        new_inst.remove_zero_area_boxes()

        labels1["img"] = im
        labels1["cls"] = cls
        labels1["instances"] = new_inst
        return labels1


class MultiModalSegCleanup:
    """Cleanup degenerates after copy-paste: clip to image bounds and drop zero-area boxes."""

    def __call__(self, labels: dict) -> dict:
        inst = labels.get("instances", None)
        img = labels.get("img", None)
        if inst is None or img is None:
            return labels
        H, W = img.shape[:2]
        inst.clip(W, H)
        inst.remove_zero_area_boxes()
        labels["instances"] = inst
        return labels
