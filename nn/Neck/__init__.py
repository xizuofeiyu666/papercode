# Ultralytics YOLOMM - Neck Module
# SOEP (Small Object Enhance Pyramid) 小目标增强金字塔模块集

from .auxiliary import GSConvE, MFM, SNI
from .soep import CSPOmniKernel, FGM, OmniKernel, SPDConv
# Neck 变体（AFPN/HSFPN/CFPT/融合等）
from .neck_variants import (
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

__all__ = [
    # SOEP核心模块
    'SPDConv',
    'FGM',
    'OmniKernel',
    'CSPOmniKernel',
    # SOEP辅助模块
    'SNI',
    'GSConvE',
    'MFM',
    # 颈部变体
    'AFPN_P345','AFPN_P345_Custom','AFPN_P2345','AFPN_P2345_Custom',
    'HFP','SDP','SDP_Improved',
    'ChannelAttention_HSFPN','ELA_HSFPN','CA_HSFPN','CAA_HSFPN',
    'CrossLayerSpatialAttention','CrossLayerChannelAttention',
    'FreqFusion','LocalSimGuidedSampler',
    'Fusion','SDI','CSPStage','BiFusion','OREPANCSPELAN4',
    'SBA','EUCB','MSDC','MSCB','CSP_MSCB',
]
