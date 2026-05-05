# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.backbone import (
    TimmBackbone,
    convnextv2_atto,
    convnextv2_base,
    convnextv2_femto,
    convnextv2_huge,
    convnextv2_large,
    convnextv2_nano,
    convnextv2_pico,
    convnextv2_tiny,
    repvit_m0_9,
    repvit_m1_0,
    repvit_m1_1,
    repvit_m1_5,
    repvit_m2_3,
    efficientformerv2_s0,
    efficientformerv2_s1,
    efficientformerv2_s2,
    efficientformerv2_l,
    EfficientViT_M0,
    EfficientViT_M1,
    EfficientViT_M2,
    EfficientViT_M3,
    EfficientViT_M4,
    EfficientViT_M5,
    SwinTransformer_Tiny,
)
from ultralytics.nn.modules import (
    AIFI,
    AIFI_DyT,
    AIFI_EDFFN,
    AIFI_LPE,
    AIFI_Mona,
    AIFI_RepBN,
    AIFI_SEFFN,
    AIFI_SEFN,
    TransformerEncoderLayer_AdditiveTokenMixer,
    TransformerEncoderLayer_ASSA,
    TransformerEncoderLayer_ASSA_SEFN,
    TransformerEncoderLayer_ASSA_SEFN_Mona,
    TransformerEncoderLayer_ASSA_SEFN_Mona_DyT,
    TransformerEncoderLayer_DAttention,
    TransformerEncoderLayer_DHSA,
    TransformerEncoderLayer_DPB,
    TransformerEncoderLayer_EfficientAdditiveAttnetion,
    TransformerEncoderLayer_HiLo,
    TransformerEncoderLayer_LocalWindowAttention,
    TransformerEncoderLayer_MSLA,
    TransformerEncoderLayer_MSMHSA,
    TransformerEncoderLayer_Pola,
    TransformerEncoderLayer_Pola_EDFFN_Mona_DyT,
    TransformerEncoderLayer_Pola_SEFFN_Mona_DyT,
    TransformerEncoderLayer_Pola_SEFN,
    TransformerEncoderLayer_Pola_SEFN_Mona,
    TransformerEncoderLayer_Pola_SEFN_Mona_DyT,
    TransformerEncoderLayer_TSSA,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    BasicBlock,
    Blocks,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvNormLayer,
    ConvTranspose,
    FourierConv,
    Detect,
    v8Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    MCFGatedFusion,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
    v26Detect,
)
from ultralytics.nn.modules.fusion import *  # noqa: F401,F403
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    E2ELoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)

# 多模态组件导入 (仅导入已迁移的组件)
try:
    from ultralytics.nn.mm import MultiModalRouter, MultiModalConfigParser, HookManager
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
try:
    # Contrast components are optional and only used if hooks are present
    from ultralytics.nn.mm.contrast import ContrastController, ContrastConfig
    CONTRAST_AVAILABLE = True
except Exception:
    CONTRAST_AVAILABLE = False

_LSCD_IMPORT_ERROR = None
# LSCD 检测头导入 (轻量化共享卷积检测头)
try:
    from ultralytics.nn.Head import (
        Detect_LSCD,
        Segment_LSCD,
        Pose_LSCD,
        OBB_LSCD,
        Conv_GN,
        Scale,
    )
    LSCD_AVAILABLE = True
except ImportError as _err:
    _LSCD_IMPORT_ERROR = _err
    LSCD_AVAILABLE = False
    LOGGER.warning(f"LSCD head import failed: {_err}")
    def _missing_lscd(*args, **kwargs):
        raise RuntimeError(f"LSCD head modules unavailable: {_err}")

    for _name in ['Detect_LSCD', 'Segment_LSCD', 'Pose_LSCD', 'OBB_LSCD', 'Conv_GN', 'Scale']:
        globals()[_name] = _missing_lscd


from ultralytics.nn.Head.yolo11_head_variants import (
    Detect_AFPN_P2345,
    Detect_AFPN_P2345_Custom,
    Detect_AFPN_P345,
    Detect_AFPN_P345_Custom,
    DetectAux,
    Detect_Efficient,
    Detect_LADH,
    Detect_LQE,
    Detect_LSCD_LQE,
    Detect_LSCSBD,
    Detect_LSDECD,
    Detect_MultiSEAM,
    Detect_RSCD,
    Detect_SEAM,
    OBB_LADH,
    OBB_LQE,
    OBB_LSCD_LQE,
    OBB_LSCSBD,
    OBB_LSDECD,
    OBB_RSCD,
    Pose_LADH,
    Pose_LQE,
    Pose_LSCD_LQE,
    Pose_LSCSBD,
    Pose_LSDECD,
    Pose_RSCD,
    Segment_Efficient,
    Segment_LADH,
    Segment_LQE,
    Segment_LSCD_LQE,
    Segment_LSCSBD,
    Segment_LSDECD,
    Segment_RSCD,
)

_SOEP_IMPORT_ERROR = None
# SOEP 颈部模块导入 (小目标增强金字塔)
try:
    from ultralytics.nn.Neck import (
        SPDConv,
        FGM,
        OmniKernel,
        CSPOmniKernel,
        SNI,
        GSConvE,
        MFM,
    )
    SOEP_AVAILABLE = True
except ImportError as _err:
    _SOEP_IMPORT_ERROR = _err
    SOEP_AVAILABLE = False
    LOGGER.warning(f"SOEP neck import failed: {_err}")
    def _missing_soep(*args, **kwargs):
        raise RuntimeError(f"SOEP neck modules unavailable: {_err}")

    for _name in ['SPDConv', 'FGM', 'MKC', 'HiSoCF', 'SNI', 'GSConvE', 'MFM']:
        globals()[_name] = _missing_soep

_C3K2_IMPORT_ERROR = None

# C3k2 Extraction 模块导入 (C3k2变体模块) — 精确子模块导入，避免聚合导入连带失败
try:
    from ultralytics.nn.extraction.c3k2_variants import (
        # Batch 1
        C3k2_Faster,
        C3k2_PConv,
        C3k2_ODConv,
        C3k2_Faster_EMA,
        C3k2_DBB,
        C3k2_WDBB,
        C3k2_DeepDBB,
        # Batch 2
        C3k2_CloAtt,
        C3k2_SCConv,
        C3k2_ScConv,
        C3k2_EMSC,
        C3k2_EMSCP,
        # Batch 3
        C3k2_ContextGuided,
        C3k2_MSBlock,
        C3k2_EMBC,
        C3k2_EMA,
        # Batch 4
        C3k2_DLKA,
        C3k2_DAttention,
        C3k2_Parc,
        C3k2_DWR,
        C3k2_RFAConv,
        # Batch 5
        C3k2_RFCBAMConv,
        C3k2_RFCAConv,
        C3k2_FocusedLinearAttention,
        C3k2_MLCA,
        C3k2_AKConv,
        # Batch 6
        C3k2_UniRepLKNetBlock,
        C3k2_DRB,
        C3k2_DWR_DRB,
        C3k2_AggregatedAtt,
        C3k2_SWC,
        # Batch 7
        C3k2_iRMB,
        C3k2_iRMB_Cascaded,
        C3k2_iRMB_DRB,
        C3k2_iRMB_SWC,
        C3k2_DynamicConv,
        # Batch 8
        C3k2_GhostDynamicConv,
        C3k2_RVB,
        C3k2_RVB_SE,
        C3k2_RVB_EMA,
        # Batch 9
        C3k2_PKIModule,
        C3k2_PPA,
        C3k2_Faster_CGLU,
        C3k2_Star,
        # Batch 10
        C3k2_Star_CAA,
        C3k2_EIEM,
        C3k2_DEConv,
        # Batch 11
        C3k2_gConv,
        C3k2_AdditiveBlock,
        C3k2_AdditiveBlock_CGLU,
        # Batch 12 - 新迁移
        C3k2_RetBlock,
        C3k2_Heat,
        C3k2_WTConv,
        C3k2_FMB,
        C3k2_MSMHSA_CGLU,
        C3k2_MogaBlock,
        C3k2_SHSA,
        C3k2_SHSA_CGLU,
        C3k2_MutilScaleEdgeInformationEnhance,
        C3k2_MutilScaleEdgeInformationSelect,
        C3k2_FFCM,
        C3k2_SMAFB,
        C3k2_SMAFB_CGLU,
        C3k2_MSM,
        C3k2_HDRAB,
        C3k2_RAB,
        C3k2_LFE,
        C3k2_IDWC,
        C3k2_IDWB,
        C3k2_CAMixer,
    )
    C3K2_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _C3K2_IMPORT_ERROR = _err
    # 为后续逻辑提供占位，避免 NameError
    C3k2_DAttention = C3k2_Parc = C3k2_FocusedLinearAttention = None
    C3K2_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"C3k2 extraction modules import failed: {_err}")

_C2PSA_IMPORT_ERROR = None
# C2PSA Extraction 模块导入 (C2PSA变体模块) — 精确子模块导入
try:
    from ultralytics.nn.extraction.c2psa_variants import (
        # Batch 1 - 基础类和第一批变体
        C2PSA,
        C2fPSA,
        C2BRA,
        BRABlock,
        # Batch 2 - 注意力机制变体
        C2CGA,
        CGABlock,
        C2DA,
        DABlock,
        C2DPB,
        DPBlock,
        C2Pola,
        Polalock,
        C2TSSA,
        TSSAlock,
        # Batch 3 - 注意力机制变体 + 归一化增强变体
        C2ASSA,
        ASSAlock,
        C2MSLA,
        MSLAlock,
        C2PSA_DYT,
        PSABlock_DYT,
        C2TSSA_DYT,
        TSSAlock_DYT,
        C2Pola_DYT,
        Polalock_DYT,
        # Batch 4 - FFN增强变体
        C2PSA_FMFFN,
        PSABlock_FMFFN,
        C2PSA_CGLU,
        PSABlock_CGLU,
        C2PSA_SEFN,
        PSABlock_SEFN,
        C2PSA_SEFFN,
        PSABlock_SEFFN,
        C2PSA_EDFFN,
        PSABlock_EDFFN,
        # Batch 5 - Mona模块化注意力归一化 + 复合增强变体
        C2PSA_Mona,
        PSABlock_Mona,
        C2TSSA_DYT_Mona,
        TSSAlock_DYT_Mona,
        C2TSSA_DYT_Mona_SEFN,
        TSSAlock_DYT_Mona_SEFN,
        C2TSSA_DYT_Mona_SEFFN,
        TSSAlock_DYT_Mona_SEFFN,
        C2TSSA_DYT_Mona_EDFFN,
        TSSAlock_DYT_Mona_EDFFN,
    )
    C2PSA_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _C2PSA_IMPORT_ERROR = _err
    C2PSA_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"C2PSA extraction modules import failed: {_err}")
    # 占位符，确保后续引用时抛出明确错误而非 NameError
    def _missing_c2psa(*args, **kwargs):
        raise RuntimeError(f"C2PSA extraction modules unavailable: {_err}")

    for _name in [
        'C2PSA', 'C2fPSA', 'C2BRA', 'BRABlock', 'C2CGA', 'CGABlock', 'C2DA', 'DABlock', 'C2DPB', 'DPBlock',
        'C2Pola', 'Polalock', 'C2TSSA', 'TSSAlock', 'C2ASSA', 'ASSAlock', 'C2MSLA', 'MSLAlock', 'C2PSA_DYT',
        'PSABlock_DYT', 'C2TSSA_DYT', 'TSSAlock_DYT', 'C2Pola_DYT', 'Polalock_DYT', 'C2PSA_FMFFN', 'PSABlock_FMFFN',
        'C2PSA_CGLU', 'PSABlock_CGLU', 'C2PSA_SEFN', 'PSABlock_SEFN', 'C2PSA_SEFFN', 'PSABlock_SEFFN', 'C2PSA_EDFFN',
        'PSABlock_EDFFN', 'C2PSA_Mona', 'PSABlock_Mona', 'C2TSSA_DYT_Mona', 'TSSAlock_DYT_Mona', 'C2TSSA_DYT_Mona_SEFN',
        'TSSAlock_DYT_Mona_SEFN', 'C2TSSA_DYT_Mona_SEFFN', 'TSSAlock_DYT_Mona_SEFFN', 'C2TSSA_DYT_Mona_EDFFN', 'TSSAlock_DYT_Mona_EDFFN',
    ]:
        globals()[_name] = _missing_c2psa

_SPPF_IMPORT_ERROR = None
# SPPF Extraction 模块导入（SPPF 及空间池化/融合变体） — 精确子模块导入
try:
    from ultralytics.nn.extraction.sppf_base import (
        # Batch 1 - 标准 SPPF 变体
        SPPF_LSKA,
        # Batch 2 - GOLD-YOLO 聚合/融合
        PyramidPoolAgg,
        PyramidPoolAgg_PCE,
        SimFusion_3in,
        SimFusion_4in,
        AdvPoolFusion,
        # Batch 3 - 注入/小波模块
        IFM,
        InjectionMultiSum_Auto_pool,
        WaveletPool,
        WaveletUnPool,
    )
    SPPF_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _SPPF_IMPORT_ERROR = _err
    SPPF_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"SPPF extraction modules import failed: {_err}")
    def _missing_sppf(*args, **kwargs):
        raise RuntimeError(f"SPPF extraction modules unavailable: {_err}")

    for _name in ['SPPF_LSKA', 'PyramidPoolAgg', 'PyramidPoolAgg_PCE', 'SimFusion_3in', 'SimFusion_4in', 'AdvPoolFusion', 'IFM', 'InjectionMultiSum_Auto_pool', 'WaveletPool', 'WaveletUnPool']:
        globals()[_name] = _missing_sppf

# C2f Extraction 模块导入（Batch 01）— 不做自动降级：导入失败应直接显式报错
from ultralytics.nn.extraction.c2f_variants import (
    C2f_CAMixer,
    C2f_Heat,
    C2f_FMB,
    C2f_MSMHSA_CGLU,
    C2f_MogaBlock,
    C2f_SHSA,
    C2f_SHSA_CGLU,
    C2f_HDRAB,
    C2f_RAB,
    C2f_FFCM,
    C2f_SMAFB,
    C2f_SMAFB_CGLU,
    C2f_AP,
    C2f_CSI,
    C2f_gConv,
    C2f_FCA,
    C2f_FDConv,
    C2f_FDT,
    C2f_FourierConv,
    C2f_GlobalFilter,
    C2f_LSBlock,
    C2f_Strip,
    C2f_StripCGLU,
    C2f_wConv,
    # Batch 03
    C2f_FasterFDConv,
    C2f_FasterSFSConv,
    C2f_Faster_KAN,
    C2f_FAT,
    C2f_SMPCGLU,
    C2f_DBlock,
    # Batch 04 (partial)
    C2f_AdditiveBlock,
    C2f_AdditiveBlock_CGLU,
    C2f_IEL,
    C2f_DTAB,
    C2f_PFDConv,
    C2f_SFSConv,
    C2f_PSFSConv,
    C2f_EBlock,
    # Batch 04 (no-CUDA extras)
    C2f_HFERB,
    C2f_JDPM,
    C2f_ETB,
    C2f_SFHF,
    C2f_MSM,
    C2f_ELGCA,
    C2f_ELGCA_CGLU,
    C2f_LEGM,
    C2f_LFEM,
    C2f_ESC,
    C2f_KAT,
)

_NECK_IMPORT_ERROR = None
# Neck 模块导入（AFPN/HSFPN/CFPT/BiPAN 等） — 精确子模块导入
try:
    from ultralytics.nn.Neck import (
        AFPN_P345,
        AFPN_P345_Custom,
        AFPN_P2345,
        AFPN_P2345_Custom,
        HFP,
        SDP,
        SDP_Improved,
        ChannelAttention_HSFPN,
        ELA_HSFPN,
        CA_HSFPN,
        CAA_HSFPN,
        CrossLayerSpatialAttention,
        CrossLayerChannelAttention,
        FreqFusion,
        LocalSimGuidedSampler,
        Fusion,
        SDI,
        CSPStage,
        BiFusion,
        OREPANCSPELAN4,
        SBA,
        EUCB,
        MSDC,
        MSCB,
        CSP_MSCB,
    )
    NECK_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _NECK_IMPORT_ERROR = _err
    NECK_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"Neck modules import failed: {_err}")
    def _missing_neck(*args, **kwargs):
        raise RuntimeError(f"Neck modules unavailable: {_err}")

    for _name in [
        'AFPN_P345', 'AFPN_P345_Custom', 'AFPN_P2345', 'AFPN_P2345_Custom', 'HFP', 'SDP', 'SDP_Improved',
        'ChannelAttention_HSFPN', 'ELA_HSFPN', 'CA_HSFPN', 'CAA_HSFPN', 'CrossLayerSpatialAttention',
        'CrossLayerChannelAttention', 'FreqFusion', 'LocalSimGuidedSampler', 'Fusion', 'SDI', 'CSPStage', 'BiFusion',
        'OREPANCSPELAN4', 'SBA', 'EUCB', 'MSCB', 'CSP_MSCB',
    ]:
        globals()[_name] = _missing_neck

# ===== Head Class Sets (align with upstream behavior) =====
# 统一以集合方式识别检测/分割/姿态/旋转头，便于在 parse_model/_apply/stride 推断等位置一致处理。
DETECT_CLASS: tuple = (
    Detect,
    v8Detect,
    WorldDetect,
    YOLOEDetect,
    v10Detect,
    v26Detect,
    ImagePoolingAttn,
)
SEGMENT_CLASS: tuple = (
    Segment,
    YOLOESegment,
)
POSE_CLASS: tuple = (Pose,)
OBB_CLASS: tuple = (OBB,)

# 动态扩展 LSCD 系列头
if LSCD_AVAILABLE:
    DETECT_CLASS = DETECT_CLASS + (Detect_LSCD,)
    SEGMENT_CLASS = SEGMENT_CLASS + (Segment_LSCD,)
    POSE_CLASS = POSE_CLASS + (Pose_LSCD,)
    OBB_CLASS = OBB_CLASS + (OBB_LSCD,)

# YOLO11 头部变体（无需编译）
DETECT_CLASS = DETECT_CLASS + (
    Detect_AFPN_P345,
    Detect_AFPN_P345_Custom,
    Detect_AFPN_P2345,
    Detect_AFPN_P2345_Custom,
    Detect_Efficient,
    DetectAux,
    Detect_SEAM,
    Detect_MultiSEAM,
    Detect_LADH,
    Detect_LSCSBD,
    Detect_LSDECD,
    Detect_RSCD,
    Detect_LQE,
    Detect_LSCD_LQE,
)
SEGMENT_CLASS = SEGMENT_CLASS + (
    Segment_Efficient,
    Segment_LADH,
    Segment_LSCSBD,
    Segment_LSDECD,
    Segment_RSCD,
    Segment_LQE,
    Segment_LSCD_LQE,
)
POSE_CLASS = POSE_CLASS + (
    Pose_LADH,
    Pose_LSCSBD,
    Pose_LSDECD,
    Pose_RSCD,
    Pose_LQE,
    Pose_LSCD_LQE,
)
OBB_CLASS = OBB_CLASS + (
    OBB_LADH,
    OBB_LSCSBD,
    OBB_LSDECD,
    OBB_RSCD,
    OBB_LQE,
    OBB_LSCD_LQE,
)

# ===== Neck Module Class Sets =====
NECK_CLASS: tuple = ()
if NECK_EXTRACTION_AVAILABLE:
    NECK_CLASS = (
        AFPN_P345,
        AFPN_P345_Custom,
        AFPN_P2345,
        AFPN_P2345_Custom,
        HFP,
        SDP,
        SDP_Improved,
        ChannelAttention_HSFPN,
        ELA_HSFPN,
        CA_HSFPN,
        CAA_HSFPN,
        CrossLayerSpatialAttention,
        CrossLayerChannelAttention,
        FreqFusion,
        LocalSimGuidedSampler,
        Fusion,
        SDI,
        CSPStage,
        BiFusion,
        OREPANCSPELAN4,
        SBA,
        EUCB,
        MSCB,
        CSP_MSCB,
    )

C3K2_CLASS: tuple = (C3k2,)

# 动态扩展 C3k2 Extraction 系列模块
if C3K2_EXTRACTION_AVAILABLE:
    C3K2_CLASS = C3K2_CLASS + (
        # Batch 1
        C3k2_Faster, C3k2_PConv, C3k2_ODConv, C3k2_Faster_EMA,
        C3k2_DBB, C3k2_WDBB, C3k2_DeepDBB,
        # Batch 2
        C3k2_CloAtt, C3k2_SCConv, C3k2_ScConv,
        C3k2_EMSC, C3k2_EMSCP,
        # Batch 3
        C3k2_ContextGuided, C3k2_MSBlock, C3k2_EMBC, C3k2_EMA,
        # Batch 4
        C3k2_DLKA, C3k2_DAttention, C3k2_Parc, C3k2_DWR, C3k2_RFAConv,
        # Batch 5
        C3k2_RFCBAMConv, C3k2_RFCAConv,
        C3k2_FocusedLinearAttention, C3k2_MLCA, C3k2_AKConv,
        # Batch 6
        C3k2_UniRepLKNetBlock, C3k2_DRB, C3k2_DWR_DRB,
        C3k2_AggregatedAtt, C3k2_SWC,
        # Batch 7
        C3k2_iRMB, C3k2_iRMB_Cascaded, C3k2_iRMB_DRB,
        C3k2_iRMB_SWC, C3k2_DynamicConv,
        # Batch 8
        C3k2_GhostDynamicConv, C3k2_RVB, C3k2_RVB_SE, C3k2_RVB_EMA,
        # Batch 9
        C3k2_PKIModule, C3k2_PPA, C3k2_Faster_CGLU, C3k2_Star,
        # Batch 10
        C3k2_Star_CAA, C3k2_EIEM, C3k2_DEConv,
        # Batch 11
        C3k2_gConv, C3k2_AdditiveBlock, C3k2_AdditiveBlock_CGLU,
        # Batch 12
        C3k2_RetBlock, C3k2_Heat, C3k2_WTConv, C3k2_FMB,
        C3k2_MSMHSA_CGLU, C3k2_MogaBlock, C3k2_SHSA, C3k2_SHSA_CGLU,
        C3k2_MutilScaleEdgeInformationEnhance, C3k2_MutilScaleEdgeInformationSelect, C3k2_FFCM,
        C3k2_SMAFB, C3k2_SMAFB_CGLU,
        C3k2_MSM, C3k2_HDRAB, C3k2_RAB, C3k2_LFE,
    )

# ===== SPPF Variant Class Set =====
SPPF_CLASS: tuple = (SPPF,)
if SPPF_EXTRACTION_AVAILABLE:
    SPPF_CLASS = SPPF_CLASS + (SPPF_LSKA,)

# ===== C2PSA Module Class Sets =====
# 统一以集合方式识别 C2PSA 及其变体，便于在 parse_model/repeat_modules 等位置一致处理。
C2PSA_CLASS: tuple = ()
if C2PSA_EXTRACTION_AVAILABLE:
    C2PSA_CLASS = (C2PSA, C2fPSA) + (
        # Batch 1/2/3 - 注意力主干变体
        C2BRA, C2CGA, C2DA, C2DPB, C2Pola, C2TSSA, C2ASSA, C2MSLA,
        C2PSA_DYT, C2TSSA_DYT, C2Pola_DYT,
        # Batch 4 - FFN 增强
        C2PSA_FMFFN, C2PSA_CGLU, C2PSA_SEFN, C2PSA_SEFFN, C2PSA_EDFFN,
        # Batch 5 - Mona 复合增强
        C2PSA_Mona, C2TSSA_DYT_Mona, C2TSSA_DYT_Mona_SEFN, C2TSSA_DYT_Mona_SEFFN, C2TSSA_DYT_Mona_EDFFN,
    )


class BaseModel(torch.nn.Module):
    """
    Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion,
    information display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # ===== MULTIMODAL EXTENSION START - 多模态路由初始化 =====
        mm_router = None
        mm_routing_enabled = False
        mm_input_sources = None

        # Check if this model has a persistent router (RTDETRDetectionModel)
        if hasattr(self, 'mm_router') and self.mm_router is not None:
            # Use persistent router from model initialization
            mm_router = self.mm_router
            mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
            if profile:
                LOGGER.info("MultiModal: 使用持久化router")
        elif MULTIMODAL_AVAILABLE:
            try:
                from ultralytics.nn.mm import MultiModalRouter
                # Create temporary router for other model types
                config_dict = getattr(self, 'yaml', None)
                mm_router = MultiModalRouter(config_dict, verbose=profile)
                mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
                if profile:
                    LOGGER.info("MultiModal: 使用临时router")
            except Exception as e:
                if profile:
                    LOGGER.warning(f"MultiModal routing initialization failed: {e}")
        # ===== MULTIMODAL EXTENSION END =====

        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            # ===== MULTIMODAL EXTENSION START - 多模态层级路由处理 =====
            # Apply multimodal routing if enabled and module has MM attributes
            if mm_routing_enabled and mm_input_sources and mm_router:
                routed_x = mm_router.route_layer_input(x, m, mm_input_sources, profile)
                if routed_x is not None:
                    x = routed_x

            # Check for spatial reset requirement
            if mm_router and hasattr(m, '_mm_spatial_reset') and m._mm_spatial_reset:
                x = mm_router.reset_spatial_input(x, m, mm_input_sources, profile)
            # ===== MULTIMODAL EXTENSION END =====

            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
                if isinstance(m, YOLOEDetect) and hasattr(self, "pe"):
                    m.fuse(self.pe.to(next(self.model.parameters()).device))
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()/Segment()/Pose()/OBB()
        heads = DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
        if isinstance(m, heads):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        det_loss_vec, det_items = self.criterion(preds, batch)

        # Optional contrastive branch (enabled only if hooks exist and compute succeeds)
        # Requirements: MULTIMODAL + CONTRAST + registered hooks via 6th field
        mm_hm = getattr(self, 'mm_hook_manager', None)
        has_hooks = False
        if mm_hm is not None:
            # Only enable contrast branch when YAML registered hooks (6th-field) exist
            try:
                has_hooks = bool(mm_hm.has_hooks())
            except Exception:
                has_hooks = False
        use_contrast = (
            self.training
            and MULTIMODAL_AVAILABLE
            and CONTRAST_AVAILABLE
            and has_hooks
        )
        if not use_contrast:
            return det_loss_vec, det_items

        # Read configs from args with safe defaults
        args = getattr(self, 'args', None)
        cfg = ContrastConfig(
            tau=getattr(args, 'contrast_tau', 0.07) if args is not None else 0.07,
            proj_dim=getattr(args, 'contrast_dim', 128) if args is not None else 128,
            lambda_weight=getattr(args, 'contrast_lambda', 0.1) if args is not None else 0.1,
            max_rois_per_image=getattr(args, 'contrast_max_rois', 64) if args is not None else 64,
            share_head=getattr(args, 'contrast_share_head', False) if args is not None else False,
            preferred_stages=tuple(getattr(args, 'contrast_stages', ("P4", "P5", "P3"))) if args is not None else ("P4", "P5", "P3"),
        )

        # Lazy create controller (only when hooks exist)
        if getattr(self, 'mm_contrast_controller', None) is None and use_contrast:
            # Create on correct device to avoid CPU/CUDA mismatch when forward() is first called
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                # Fallback to batch image device if model has no parameters (unlikely)
                img = batch.get('img')
                dev = img.device if isinstance(img, torch.Tensor) else torch.device('cpu')
            self.mm_contrast_controller = ContrastController(cfg).to(dev)

        # Collect hooked features (do not pop to allow external visualization; keep bounded by latest write)
        hook_buffers = mm_hm.collect(pop=False)
        loss_c, stats = self.mm_contrast_controller(hook_buffers, batch)
        # Debug: detect non-finite contrastive loss (when enabled and computed)
        try:
            if loss_c is not None and not torch.isfinite(loss_c):
                from ultralytics.utils import LOGGER as _LOGGER
                _LOGGER.warning(f"[CL][loss] non-finite loss_c detected: {float(loss_c.detach().cpu())}")
        except Exception:
            pass
        if loss_c is None:
            return det_loss_vec, det_items  # no valid pairs this step

        # Compose outputs: scale contrast by lambda for backprop, but report raw value in items
        lambda_w = cfg.lambda_weight
        if det_loss_vec.dim() == 0:
            # safety: ensure vector form
            det_loss_vec = det_loss_vec.unsqueeze(0)
        total_vec = torch.cat([det_loss_vec, (loss_c * lambda_w).unsqueeze(0)], dim=0)
        total_items = torch.cat([det_items, loss_c.detach().unsqueeze(0)], dim=0)
        # Optionally, expose stats via side-effect for loggers (trainer can read from model)
        self._contrast_last_stats = stats
        return total_vec, total_items

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """
    YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass,
    augmented inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 传递多模态路由器（如果存在）
        if hasattr(self.model, 'multimodal_router'):
            self.multimodal_router = self.model.multimodal_router
        else:
            self.multimodal_router = None
        # Persist router for runtime ablation/filling so BaseModel forward can reuse it
        self.mm_router = self.multimodal_router if self.multimodal_router is not None else None

        # 对比学习控制器延后创建：避免在info()阶段触发Lazy参数未初始化错误
        # 实际创建在训练器的get_model阶段或首次使用时（见BaseModel.loss）。

        # 传递HookManager（如果存在）
        self.mm_hook_manager = getattr(self.model, 'mm_hook_manager', None)

        # Build strides
        m = self.model[-1]  # Detect()/Segment()/Pose()/OBB()/...
        heads = DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
        if isinstance(m, heads):  # includes all Detect/Segment/Pose/OBB subclasses (e.g., LSCD variants)
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                seg_pose_obb = SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
                return self.forward(x)[0] if isinstance(m, seg_pose_obb) else self.forward(x)

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            _stride_out = _forward(torch.zeros(1, ch, s, s))
            # DetectAux：训练输出包含辅助分支（2*nl 个特征图），stride 只应基于主分支 nl 个特征图计算
            if hasattr(m, "dfl_aux") and isinstance(_stride_out, (list, tuple)) and hasattr(m, "nl"):
                nl = int(getattr(m, "nl"))
                if len(_stride_out) >= nl:
                    _stride_out = _stride_out[:nl]
            m.stride = torch.tensor([s / x.shape[-2] for x in _stride_out])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            # 初始化检测头偏置（若实现提供）
            if hasattr(m, "bias_init") and callable(getattr(m, "bias_init")):
                m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        Clip YOLO augmented inference tails.

        Args:
            y (List[torch.Tensor]): List of detection tensors.

        Returns:
            (List[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel.

        When loss26=true is set in training args, uses YOLO26-style loss functions:
        - E2ELoss with ProgLoss (progressive weight decay) when loss_prog=true
        - E2EDetectLoss with STAL when loss_prog=false
        - TaskAlignedAssigner with stride enhancement when loss_stride_enhance=true
        - STAL (Selective Top-K Aligned Learning) when loss_stal=true
        """
        args = getattr(self, 'args', None)
        is_end2end = getattr(self, 'end2end', False)

        # Check if loss26 enhancements are enabled
        loss26 = getattr(args, 'loss26', False) if args is not None else False

        if not loss26:
            # Default behavior: original loss functions
            return E2EDetectLoss(self) if is_end2end else v8DetectionLoss(self)

        # YOLO26 loss enhancements enabled
        loss_prog = getattr(args, 'loss_prog', True) if args is not None else True
        loss_stal = getattr(args, 'loss_stal', True) if args is not None else True
        loss_stride = getattr(args, 'loss_stride_enhance', True) if args is not None else True

        # Build STAL and stride enhancement parameters
        tal_topk2 = 1 if loss_stal else None  # STAL uses topk2=1 for secondary filtering
        tal_stride = [8, 16] if loss_stride else None  # Small object enhancement

        if is_end2end:
            if loss_prog:
                # E2ELoss with ProgLoss (progressive weight decay o2m: 0.8->0.1, o2o: 0.2->0.9)
                return E2ELoss(self, loss_fn=v8DetectionLoss, tal_stride=tal_stride)
            else:
                # E2EDetectLoss with STAL/stride parameters
                return E2EDetectLoss(self, tal_topk2=tal_topk2, tal_stride=tal_stride)
        else:
            # Non-end2end model with optional stride enhancement
            return v8DetectionLoss(self, tal_topk=10, tal_topk2=tal_topk2, tal_stride=tal_stride)


class OBBModel(DetectionModel):
    """
    YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized
    loss computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo11n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """
    YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized
    loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """
    YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized
    loss computation for keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """
        Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """
    YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks,
    providing model initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo11n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """
        Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """
        Update a TorchVision classification model to class count 'n' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        
        # ===== MULTIMODAL EXTENSION START - 持久化MultiModalRouter =====
        self.mm_router = None
        self.mm_routing_enabled = False
        self.mm_input_sources = None
        
        if MULTIMODAL_AVAILABLE:
            try:
                from ultralytics.nn.mm import MultiModalRouter
                # Create persistent router with model configuration
                config_dict = getattr(self, 'yaml', None)
                self.mm_router = MultiModalRouter(config_dict, verbose=verbose)
                if verbose:
                    LOGGER.info("RTDETRDetectionModel: 持久化MultiModalRouter已创建")
            except Exception as e:
                if verbose:
                    LOGGER.warning(f"RTDETRDetectionModel: MultiModalRouter初始化失败: {e}")
        # ===== MULTIMODAL EXTENSION END =====

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.yaml["nc"], use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions.

        Returns:
            loss_sum (torch.Tensor): Total loss value.
            loss_items (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        # ===== MULTIMODAL EXTENSION START - 多模态路由初始化 =====
        mm_router = self.mm_router
        mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
        # ===== MULTIMODAL EXTENSION END =====
        
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
            # ===== MULTIMODAL EXTENSION START - 多模态层级路由处理 =====
            # Apply multimodal routing if enabled and module has MM attributes
            if mm_routing_enabled and mm_input_sources and mm_router:
                routed_x = mm_router.route_layer_input(x, m, mm_input_sources, profile)
                if routed_x is not None:
                    x = routed_x

            # Check for spatial reset requirement
            if mm_router and hasattr(m, '_mm_spatial_reset') and m._mm_spatial_reset:
                x = mm_router.reset_spatial_input(x, m, mm_input_sources, profile)
            # ===== MULTIMODAL EXTENSION END =====
            
            if profile:
                self._profile_one_layer(m, x, dt)
            # ===== 多输出主干（单模块多输出）支持：与 RTDETR-main 对齐 =====
            if hasattr(m, "backbone"):
                base_idx = len(y)
                feats = m(x)
                if not isinstance(feats, (list, tuple)):
                    raise TypeError(
                        f"backbone 模块 '{m.__class__.__name__}' 输出必须为 list/tuple，当前={type(feats).__name__}"
                    )
                feats = list(feats)
                if len(feats) > 5:
                    raise ValueError(
                        f"backbone 模块 '{m.__class__.__name__}' 输出层数过多：len={len(feats)}，期望<=5"
                    )
                for _ in range(5 - len(feats)):
                    feats.insert(0, None)
                for local_i, feat in enumerate(feats):
                    global_i = base_idx + local_i
                    y.append(feat if global_i in self.save else None)
                x = feats[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """
    YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based
    class specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x) or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """
    YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts,
    supporting both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings cooperated with reprta module.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """
        Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (List[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """
        Get fused vocabulary layer from the model.

        Args:
            names (list): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (List[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """
        Get class positional embeddings.

        Args:
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """
    YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts,
    providing specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """
    Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through
    model averaging or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """
        Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """
    Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.

    Args:
        weights (str | List[str]): Model weights path(s).
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        (torch.nn.Module): Loaded model.
    """
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = getattr(model, "task", guess_model_task(model))
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """
    Load a single model weights.

    Args:
        weight (str): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def _validate_and_fill_dea_args(args, c_left):
    """Validate DEA args strictly and auto-fill only the channel when为None.

    期望签名：DEA(channel, kernel_size, p_kernel=None, m_kernel=None, reduction=16)
    允许的最简形式：DEA([None, kernel_size]) 或 DEA([channel, kernel_size])。
    不做旧参数顺序的自动更正，遇到不合规直接报错。
    """
    a = list(args) if isinstance(args, (list, tuple)) else [args]

    # 最简形式 [channel/None, kernel_size]
    if len(a) == 2:
        ch, ks = a
        ch = c_left if ch is None else ch
        if not isinstance(ch, int) or not isinstance(ks, int) or ks <= 0:
            raise ValueError(
                f"DEA expects [channel(or None), kernel_size] as minimal form, got {args}"
            )
        return [ch, ks, None, None, 16]

    # 完整形式：补齐长度到5
    while len(a) < 5:
        a.append(None)
    ch, ks, pk, mk, rd = a[:5]
    ch = c_left if ch is None else ch
    # 严格校验类型与取值
    if not isinstance(ch, int) or ch <= 0:
        raise ValueError(f"DEA arg[0]=channel must be positive int, got {ch}")
    if not isinstance(ks, int) or ks <= 0:
        raise ValueError(f"DEA arg[1]=kernel_size must be positive int, got {ks}")
    if pk is not None and (not isinstance(pk, (list, tuple)) or len(pk) != 2):
        raise ValueError(f"DEA arg[2]=p_kernel must be 2-list/tuple or None, got {pk}")
    if mk is not None and (not isinstance(mk, (list, tuple)) or len(mk) != 2):
        raise ValueError(f"DEA arg[3]=m_kernel must be 2-list/tuple or None, got {mk}")
    if rd is None:
        rd = 16
    if not isinstance(rd, int) or rd <= 0:
        raise ValueError(f"DEA arg[4]=reduction must be positive int, got {rd}")

    return [ch, ks, pk, mk, rd]


def parse_model(d, ch, verbose=True, dataset_config=None):
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.
        dataset_config (dict, optional): Dataset configuration containing Xch and other multimodal info.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))

    # 多模态配置解析（仅处理已迁移的组件）
    multimodal_router = None
    hook_manager = HookManager() if MULTIMODAL_AVAILABLE else None
    hook_parser = MultiModalConfigParser() if MULTIMODAL_AVAILABLE else None
    if MULTIMODAL_AVAILABLE and 'multimodal' in d:
        try:
            # 解析多模态配置
            config_parser = MultiModalConfigParser()
            model_config = config_parser.parse_config(d)

            # 创建多模态路由器
            if model_config.get('has_multimodal_layers', False):
                multimodal_router = MultiModalRouter(model_config)
                if verbose:
                    LOGGER.info(f"{colorstr('multimodal:')} Router initialized with {len(model_config.get('input_layers', []))} input layers")
        except Exception as e:
            if verbose:
                LOGGER.warning(f"Failed to initialize multimodal router: {e}")
            multimodal_router = None
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    is_backbone = False  # 单模块多输出主干出现后，后续层索引需要整体偏移 4（与 RTDETR-main 对齐）
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            FourierConv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            # 支持 SPPF 及其变体
            *SPPF_CLASS,
            # C2PSA 系列在 C2PSA_CLASS 中统一管理
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            # C2f 变体（RTDETRMM mm-mid-c2f-*）：需作为 base_modules 才能正确注入 c1/c2（并支持 repeats -> n）
            C2f_CAMixer,
            C2f_Heat,
            C2f_FMB,
            C2f_MSMHSA_CGLU,
            C2f_MogaBlock,
            C2f_SHSA,
            C2f_SHSA_CGLU,
            C2f_HDRAB,
            C2f_RAB,
            C2f_FFCM,
            C2f_SMAFB,
            C2f_SMAFB_CGLU,
            C2f_AP,
            C2f_CSI,
            C2f_gConv,
            C2f_FCA,
            C2f_FDConv,
            C2f_FDT,
            C2f_FourierConv,
            C2f_GlobalFilter,
            C2f_LSBlock,
            C2f_Strip,
            C2f_StripCGLU,
            C2f_wConv,
            C2f_FasterFDConv,
            C2f_FasterSFSConv,
            C2f_Faster_KAN,
            C2f_FAT,
            C2f_SMPCGLU,
            C2f_DBlock,
            C2f_AdditiveBlock,
            C2f_AdditiveBlock_CGLU,
            C2f_IEL,
            C2f_DTAB,
            C2f_PFDConv,
            C2f_SFSConv,
            C2f_PSFSConv,
            C2f_EBlock,
            C2f_HFERB,
            C2f_JDPM,
            C2f_ETB,
            C2f_SFHF,
            C2f_MSM,
            C2f_ELGCA,
            C2f_ELGCA_CGLU,
            C2f_LEGM,
            C2f_LFEM,
            C2f_ESC,
            C2f_KAT,
            C2f_BiFocus,
            RepNCSPELAN4,
            RepNCSPELAND,  # ELAN followed by dictionary injection (YOLO-RD)
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            ConvNormLayer,  # RTDETR module
        } | set(C3K2_CLASS) | set(C2PSA_CLASS) | set(NECK_CLASS)  # 动态添加已迁移模块
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C2f_CAMixer,
            C2f_Heat,
            C2f_FMB,
            C2f_MSMHSA_CGLU,
            C2f_MogaBlock,
            C2f_SHSA,
            C2f_SHSA_CGLU,
            C2f_HDRAB,
            C2f_RAB,
            C2f_FFCM,
            C2f_SMAFB,
            C2f_SMAFB_CGLU,
            C2f_AP,
            C2f_CSI,
            C2f_gConv,
            C2f_FCA,
            C2f_FDConv,
            C2f_FDT,
            C2f_FourierConv,
            C2f_GlobalFilter,
            C2f_LSBlock,
            C2f_Strip,
            C2f_StripCGLU,
            C2f_wConv,
            C2f_FasterFDConv,
            C2f_FasterSFSConv,
            C2f_Faster_KAN,
            C2f_FAT,
            C2f_SMPCGLU,
            C2f_DBlock,
            C2f_AdditiveBlock,
            C2f_AdditiveBlock_CGLU,
            C2f_IEL,
            C2f_DTAB,
            C2f_PFDConv,
            C2f_SFSConv,
            C2f_PSFSConv,
            C2f_EBlock,
            C2f_HFERB,
            C2f_JDPM,
            C2f_ETB,
            C2f_SFHF,
            C2f_MSM,
            C2f_ELGCA,
            C2f_ELGCA_CGLU,
            C2f_LEGM,
            C2f_LFEM,
            C2f_ESC,
            C2f_KAT,
            C2f_BiFocus,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fCIB,
            A2C2f,
        } | set(C3K2_CLASS) | set(C2PSA_CLASS)  # 动态添加所有 C3k2 与 C2PSA 变体
    )
    # ===== MULTIMODAL EXTENSION START - 多模态路由器初始化 =====
    mm_router = None
    if MULTIMODAL_AVAILABLE:
        try:
            from ultralytics.nn.mm import MultiModalRouter
            # 构建配置字典，包含dataset_config
            config_dict = d.copy()
            if dataset_config:
                config_dict['dataset_config'] = dataset_config
            mm_router = MultiModalRouter(config_dict, verbose)
        except Exception as e:
            if verbose:
                LOGGER.warning(f"MultiModal router initialization failed: {e}")
    # ===== MULTIMODAL EXTENSION END =====

    # 通用解析子程序：两输入的跨模态注意力/融合模块（CTF/MHCA）
    def _parse_two_input_equal_attn(module, f, args, ch, i):
        if isinstance(f, int) or len(f) != 2:
            raise ValueError(f"{module.__name__} expects 2 inputs, got {f} at layer {i}")
        c_left, c_right = ch[f[0]], ch[f[1]]
        # Auto-fill first dim arg if missing/None
        if len(args) == 0:
            args.insert(0, c_left)
        else:
            if args[0] is None:
                args[0] = c_left
        # Module-specific defaults and output channel calc
        if module is MultiHeadCrossAttention:
            # Ensure num_heads exists
            if len(args) < 2:
                args.append(2)
            c2_val = c_left  # tuple outputs with per-branch channels = C
        elif module is CrossTransformerFusion:
            c2_val = c_left * 2  # concat two branches
        else:
            c2_val = c_left
        return c2_val, args

    # 针对 C3k2 变体的参数归一化，补齐缺失的关键形参以匹配上游实现
    def _normalize_c3k2_args(module, raw_args):
        # 保留原列表，避免外部引用被直接修改
        args = list(raw_args)
        c2_val = args[0] if args else None

        # 如果 C3k2 变体未成功导入，则抛出明确错误，避免静默降级
        if not C3K2_EXTRACTION_AVAILABLE:
            raise RuntimeError(
                "C3k2 extraction modules导入失败，无法解析C3k2层参数。原始异常: "
                f"{_C3K2_IMPORT_ERROR}"
            )

        # fmapsize 相关
        if module in (C3k2_DAttention, C3k2_Parc, C3k2_FocusedLinearAttention):
            if len(args) < 2 or isinstance(args[1], bool):
                args = [c2_val, (20, 20)] + args[2:]
            # 确保第三参为 shortcut
            if len(args) < 3 or not isinstance(args[2], bool):
                args = args[:2] + [True] + args[3:]

        # 聚合注意力：input_resolution & sr_ratio
        elif module is C3k2_AggregatedAtt:
            need_fill = len(args) < 3 or isinstance(args[1], bool) or isinstance(args[2], bool)
            if need_fill:
                # 按通道规模推导默认输入分辨率与步幅比
                input_res = 40 if c2_val is not None and c2_val <= 512 else 20
                sr_ratio = 2 if c2_val is not None and c2_val <= 512 else 1
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [c2_val, input_res, sr_ratio, shortcut]

        # SWC 位移卷积：kernel_size
        elif module is C3k2_SWC:
            if len(args) < 2 or isinstance(args[1], bool):
                ks = 11 if c2_val is not None and c2_val <= 256 else (9 if c2_val is not None and c2_val <= 512 else 7)
                shortcut = True
                # 若第二/第三参数本就是布尔，视作 shortcut
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                elif len(args) > 2 and isinstance(args[2], bool):
                    shortcut = args[2]
                args = [c2_val, ks, shortcut]

        # UniRep 大核：k
        elif module is C3k2_UniRepLKNetBlock:
            if len(args) < 2 or isinstance(args[1], bool):
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [c2_val, 7, shortcut]

        # iRMB 派生：深度卷积核
        elif module in (C3k2_iRMB_DRB, C3k2_iRMB_SWC):
            if len(args) < 2 or isinstance(args[1], bool):
                ks = 13 if c2_val is not None and c2_val <= 256 else (11 if c2_val is not None and c2_val <= 512 else 9)
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                elif len(args) > 2 and isinstance(args[2], bool):
                    shortcut = args[2]
                args = [c2_val, ks, shortcut]

        # PKIModule 复合参数
        elif module is C3k2_PKIModule:
            need_fill = len(args) < 6 or isinstance(args[1], bool)
            if need_fill or not hasattr(args[1], "__iter__"):
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [
                    c2_val,
                    (3, 5, 7, 9, 11),  # kernel_sizes
                    1.0,               # expansion
                    True,              # with_caa
                    11,                # caa_kernel_size
                    True,              # add_identity
                    shortcut,
                ]

        return args

    for i, layer_config in enumerate(d["backbone"] + d["head"]):  # from, number, module, args, [input_type, hook]
        # ===== MULTIMODAL EXTENSION START - 层配置解析 =====
        # Parse layer configuration with optional 5th field for multimodal routing
        if mm_router:
            c1, mm_input_source, mm_attributes = mm_router.parse_layer_config(layer_config, i, ch, verbose)
            f, n, m, args = layer_config[:4]  # Extract standard 4 fields
        else:
            # Standard parsing for non-multimodal layers
            if len(layer_config) == 4:
                f, n, m, args = layer_config
            elif len(layer_config) == 5:
                f, n, m, args, _ = layer_config  # Ignore 5th field if no MM router
            else:
                raise ValueError(f"Invalid layer definition at index {i}: expected 4 or 5 elements, got {len(layer_config)}")
            c1 = None
            mm_input_source = None
            mm_attributes = {}
        # ===== MULTIMODAL EXTENSION END =====
        
        try:
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else globals()[m]
            )  # get module
        except KeyError as e:
            # 如果YAML中使用的模块找不到，抛出错误并终止程序
            raise ImportError(
                f"模块 '{m}' 在YAML配置的第 {i} 层中被使用，但未找到该模块。"
                f"请确保模块已正确导入。"
                f"如果这是来自extra_modules的模块，请检查它是否在__init__.py中正确导出。"
            ) from e
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        m_created = False
        m_instance = None
        if m in base_modules:
            # 针对特定 C3k2 变体补齐缺省参数（需在装配 c1/c2 之前执行）
            args = _normalize_c3k2_args(m, args)
            # 确保 args 可变（支持后续 args[0] 写回）
            args = list(args)

            # ===== MULTIMODAL EXTENSION START - 多模态通道计算 =====
            # Use multimodal router computed c1 if available, otherwise use standard logic
            if mm_input_source and c1 is not None:
                # For multimodal layers, c1 is computed by router, c2 from args
                c2 = args[0]
                # c1 is already set by mm_router.parse_layer_config()
            else:
                c1, c2 = ch[f], args[0]
            # ===== MULTIMODAL EXTENSION END =====

            # 统一缩放 c2 并写回 args[0]（确保后续 [c1, *args] 插入时使用缩放后的值）
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args[0] = c2  # 写回缩放后的 c2，防止后续使用未缩放值

            # C2fAttn 特殊处理：在插入 c1 之前调整 embed channels 和 num heads
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            # SPPF 变体参数长度校验（硬错误，防止参数错位）
            if m is SPPF:
                if not (1 <= len(args) <= 4):
                    raise ValueError(
                        f"Layer {i} (SPPF): 参数长度非法，期望 args=[c2] 或 [c2,k] 或 [c2,k,n] 或 [c2,k,n,shortcut]，"
                        f"但收到 args={args}"
                    )
            elif SPPF_EXTRACTION_AVAILABLE and m is SPPF_LSKA:
                if not (1 <= len(args) <= 2):
                    raise ValueError(
                        f"Layer {i} (SPPF_LSKA): 参数长度非法，期望 args=[c2] 或 [c2,k]，"
                        f"但收到 args={args}（SPPF_LSKA 不支持 n/shortcut 参数）"
                    )

            # 统一插入 c1（此时 args[0] 已是缩放后的 c2）
            args = [c1, *args]

            # repeat_modules 处理
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1

            # C3K2_CLASS 特殊处理（M/L/X 尺寸）
            if m in C3K2_CLASS:  # for M/L/X sizes - 支持所有C3k2变体
                legacy = False
                if scale in "mlx":
                    args[3] = True

            # A2C2f 特殊处理（L/X 尺寸）
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))

            # C2fCIB 特殊处理
            if m is C2fCIB:
                legacy = False
        elif m in {
            AIFI,
            AIFI_LPE,
            AIFI_RepBN,
            AIFI_SEFN,
            AIFI_Mona,
            AIFI_DyT,
            AIFI_SEFFN,
            AIFI_EDFFN,
            TransformerEncoderLayer_LocalWindowAttention,
            TransformerEncoderLayer_DAttention,
            TransformerEncoderLayer_HiLo,
            TransformerEncoderLayer_EfficientAdditiveAttnetion,
            TransformerEncoderLayer_AdditiveTokenMixer,
            TransformerEncoderLayer_MSMHSA,
            TransformerEncoderLayer_DHSA,
            TransformerEncoderLayer_DPB,
            TransformerEncoderLayer_Pola,
            TransformerEncoderLayer_TSSA,
            TransformerEncoderLayer_ASSA,
            TransformerEncoderLayer_MSLA,
            TransformerEncoderLayer_Pola_SEFN,
            TransformerEncoderLayer_ASSA_SEFN,
            TransformerEncoderLayer_ASSA_SEFN_Mona,
            TransformerEncoderLayer_Pola_SEFN_Mona,
            TransformerEncoderLayer_ASSA_SEFN_Mona_DyT,
            TransformerEncoderLayer_Pola_SEFN_Mona_DyT,
            TransformerEncoderLayer_Pola_SEFFN_Mona_DyT,
            TransformerEncoderLayer_Pola_EDFFN_Mona_DyT,
        }:
            args = [ch[f], *args]
        elif m in {
            TimmBackbone,
            convnextv2_atto,
            convnextv2_femto,
            convnextv2_pico,
            convnextv2_nano,
            convnextv2_tiny,
            convnextv2_base,
            convnextv2_large,
            convnextv2_huge,
            repvit_m0_9,
            repvit_m1_0,
            repvit_m1_1,
            repvit_m1_5,
            repvit_m2_3,
            efficientformerv2_s0,
            efficientformerv2_s1,
            efficientformerv2_s2,
            efficientformerv2_l,
            EfficientViT_M0,
            EfficientViT_M1,
            EfficientViT_M2,
            EfficientViT_M3,
            EfficientViT_M4,
            EfficientViT_M5,
            SwinTransformer_Tiny,
        }:
            # 约束：单模块多输出主干必须出现在 backbone 首层，否则索引体系会产生歧义
            if i != 0:
                raise RuntimeError(
                    f"{m.__name__ if hasattr(m, '__name__') else m} 必须作为模型 backbone 的第 0 层（单模块多输出主干）。"
                    "请将模态输入投影Conv放在该模块内部（已内置），不要在其前面再插入额外层。"
                )
            if n != 1:
                raise ValueError(
                    f"{m.__name__ if hasattr(m, '__name__') else m} 不支持 repeats>1（单模块多输出主干）"
                )
            # 注入输入通道（优先使用 router 解析出的 c1，以支持 Dual=6ch 等多模态输入）
            in_ch = c1 if c1 is not None else ch[f]
            args = [in_ch, *args]
            m_instance = m(*args)
            m_created = True
            c2 = list(getattr(m_instance, "channel"))
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # ===== GOLD-YOLO / 提取类池化-融合模块 =====
        elif SPPF_EXTRACTION_AVAILABLE and m in frozenset({SimFusion_4in, AdvPoolFusion}):
            # 纯拼接型融合，输出通道为所有输入之和
            c2 = sum(ch[x] for x in f)
        elif SPPF_EXTRACTION_AVAILABLE and m is SimFusion_3in:
            # SimFusion_3in([c0,c1,c2], out)
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[f_] for f_ in f], c2]
        elif SPPF_EXTRACTION_AVAILABLE and m is IFM:
            # IFM(c1, [o1,o2,...], embed_dim_p=96, fuse_block_num=3)
            c1 = ch[f]
            c2 = sum(args[0])
            args = [c1, *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is InjectionMultiSum_Auto_pool:
            # InjectionMultiSum_Auto_pool(inp, oup, global_inp, flag), f=[local, global]
            c1 = ch[f[0]]
            c2 = args[0]
            args = [c1, *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is PyramidPoolAgg:
            # PyramidPoolAgg(inc=sum(inputs), ouc=args[0], stride, pool_mode='torch')
            c2 = args[0]
            args = [sum(ch[x] for x in f), *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is PyramidPoolAgg_PCE:
            # 输出为各输入池化后拼接，通道为输入之和
            c2 = sum(ch[x] for x in f)
        elif SPPF_EXTRACTION_AVAILABLE and m is WaveletPool:
            # 小波下采样，通道*4
            c2 = ch[f] * 4
        # DEA 专用分支：允许旧风格参数并归一化到新签名
        elif m is DEA:
            # Expect exactly two inputs; output channels follow the left branch
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            args = _validate_and_fill_dea_args(args, c_left)
            c2 = c_left
        elif m is MCFGatedFusion:
            if isinstance(f, int) or len(f) < 2:
                raise ValueError(f"{m.__name__} expects >=2 inputs, got {f} at layer {i}")
            mode = args[0] if len(args) > 0 else "add"
            k = args[1] if len(args) > 1 else 1
            c_out = args[2] if len(args) > 2 and args[2] else None
            main_idx = args[3] if len(args) > 3 else 0
            aux_idx = args[4] if len(args) > 4 else 1
            zero_init = args[5] if len(args) > 5 else True
            use_bn = args[6] if len(args) > 6 else False
            act = args[7] if len(args) > 7 else True

            c_main = ch[f[main_idx]]
            c_aux = ch[f[aux_idx]]
            c_out = c_out or c_main
            if c_out != nc:
                c_out = make_divisible(min(c_out, max_channels) * width, 8)
            args = [c_main, c_aux, c_out, mode, k, 1, None, 1, main_idx, aux_idx, zero_init, use_bn, act]
            c2 = c_main
        # 其他融合模块（与多模态路由无关的纯特征融合），要求两路输入空间尺寸一致
        elif m in frozenset({SMRF}):
            # Expect exactly two inputs; output channels follow the left branch
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            # Auto infer dim if not provided (None or missing)
            if len(args) == 0:
                args.insert(0, c_left)
            elif args[0] is None:
                args[0] = c_left
            c2 = c_left
        else:
            c2 = ch[f] if isinstance(f, int) else args[0]

        # ===== 单模块多输出主干：c2 为 list 时，将该模块标记为 backbone 并启用索引偏移 =====
        if isinstance(c2, list):
            is_backbone = True
            if not m_created:
                if n != 1:
                    raise ValueError(f"{m.__name__ if hasattr(m, '__name__') else m}: 多输出主干不支持 repeats>1")
                m_instance = m(*args)
            m_ = m_instance
            # 供 forward/predict 路径识别
            m_.backbone = True
        else:
            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = (i + 4 if is_backbone else i), f, t  # attach index, 'from' index, type

        # ===== MULTIMODAL EXTENSION START - 多模态属性设置 =====
        # Set multimodal attributes on the module if this layer has multimodal routing
        if mm_attributes and mm_router:
            mm_router.set_module_attributes(m_, mm_attributes)
        # ===== MULTIMODAL EXTENSION END =====
        if verbose:
            # Format args for display - convert class objects to simple names
            display_args = []
            for arg in args:
                if isinstance(arg, type):
                    display_args.append(arg.__name__)
                else:
                    display_args.append(arg)
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(display_args):<30}")  # print
        save_base = (i + 4) if is_backbone else i
        save.extend(x % save_base for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        # ===== HOOK EXTENSION START - 第六字段Hook注册 =====
        if MULTIMODAL_AVAILABLE and hook_manager is not None and hook_parser is not None:
            try:
                if len(layer_config) >= 6:
                    hook_field = layer_config[5]
                    specs = hook_parser.parse_hook_field(hook_field, layer_idx=i)
                    if specs:
                        # 禁止在 Dual 早期融合层上挂载CL钩子
                        if mm_input_source == 'Dual':
                            raise ValueError(
                                f"在第{i}层检测到 Dual 输入源，但同时配置了对比学习钩子。"
                                f"根据设计，禁止在早期融合(Dual)路径上挂载CL。"
                            )
                        for spec in specs:
                            # 注册时无需显式buffer；自动命名由HookManager完成
                            hook_manager.register(m_, spec)
            except Exception as e:
                # 明确失败，避免静默降级
                raise
        # ===== HOOK EXTENSION END =====
        if i == 0:
            ch = []
        if isinstance(c2, list):
            ch.extend(c2)
            # 对齐 RTDETR-main：补齐到 5 路（0-P1/2, 1-P2/4, 2-P3/8, 3-P4/16, 4-P5/32）
            for _ in range(5 - len(ch)):
                ch.insert(0, 0)
        else:
            ch.append(c2)

    model = torch.nn.Sequential(*layers)
    
    # 将multimodal_router附加到模型上，以便后续访问
    if mm_router:
        model.multimodal_router = mm_router
    # 将HookManager附加到模型上（仅当YAML实际注册了Hook时），并打印简明可视化摘要（无表情符号）
    if hook_manager is not None and hook_manager.has_hooks():
        model.mm_hook_manager = hook_manager
        try:
            summary = hook_manager.summary_text(max_items=20)
            for line in summary.splitlines():
                LOGGER.info(line)
        except Exception:
            pass
    
    return model, sorted(save)


def yaml_model_load(path):
    """
    Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)  # noqa
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
