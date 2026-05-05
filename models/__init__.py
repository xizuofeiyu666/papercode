# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .fastsam import FastSAM
from .nas import NAS
from .rtdetr import RTDETR
from .sam import SAM
from .yolo import YOLO, YOLOE, YOLOWorld

# Multi-modal YOLO import (conditional)
try:
    from .yolo.model import YOLOMM
    _YOLOMM_AVAILABLE = True
except ImportError:
    _YOLOMM_AVAILABLE = False

# Multi-modal RTDETRMM import (conditional)
try:
    from .rtdetrmm.model import RTDETRMM
    _RTDETRMM_AVAILABLE = True
except ImportError:
    _RTDETRMM_AVAILABLE = False

if _YOLOMM_AVAILABLE:
    __all__ = "YOLO", "RTDETR", "SAM", "FastSAM", "NAS", "YOLOWorld", "YOLOE", "YOLOMM"  # allow simpler import
else:
    __all__ = "YOLO", "RTDETR", "SAM", "FastSAM", "NAS", "YOLOWorld", "YOLOE"  # allow simpler import

if _RTDETRMM_AVAILABLE:
    __all__ = (*__all__, "RTDETRMM")
