# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.163"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

# Import standard models
from ultralytics.models import NAS, RTDETR, SAM, YOLO, YOLOE, FastSAM, YOLOWorld

# Import YOLOMM (conditional)
try:
    from ultralytics.models.yolo.model import YOLOMM
    _YOLOMM_AVAILABLE = True
except ImportError:
    _YOLOMM_AVAILABLE = False
    YOLOMM = None  # Set to None if not available

# Import RTDETRMM (conditional)
try:
    from ultralytics.models.rtdetrmm.model import RTDETRMM
    _RTDETRMM_AVAILABLE = True
except ImportError:
    _RTDETRMM_AVAILABLE = False
    RTDETRMM = None  # Set to None if not available

# Import DepthGen (conditional)
try:
    from ultralytics.nn.mm import DepthGen
    _DEPTHGEN_AVAILABLE = True
except Exception:
    _DEPTHGEN_AVAILABLE = False
    DepthGen = None

# Import DEMGen (conditional)
try:
    from ultralytics.nn.mm import DEMGen
    _DEMGEN_AVAILABLE = True
except Exception:
    _DEMGEN_AVAILABLE = False
    DEMGen = None

# Import EdgeGen (conditional)
try:
    from ultralytics.nn.mm import EdgeGen
    _EDGEGEN_AVAILABLE = True
except Exception:
    _EDGEGEN_AVAILABLE = False
    EdgeGen = None
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS

# Dynamic __all__ based on availability
__all__ = [
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "YOLOE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
]

if _YOLOMM_AVAILABLE:
    __all__.append("YOLOMM")
    
if _RTDETRMM_AVAILABLE:
    __all__.append("RTDETRMM")
if _DEPTHGEN_AVAILABLE:
    __all__.append("DepthGen")
if _DEMGEN_AVAILABLE:
    __all__.append("DEMGen")
if _EDGEGEN_AVAILABLE:
    __all__.append("EdgeGen")

__all__ = tuple(__all__)
