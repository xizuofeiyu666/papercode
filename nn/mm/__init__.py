# Ultralytics Multimodal Router Module
# Universal RGB+X Data Routing System for YOLO and RTDETR
# Version: v1.0

"""
Ultralytics Multimodal Router Module

This module provides a comprehensive multimodal routing system for RGB+X architectures,
supporting YOLO and RTDETR with zero-copy tensor routing and configuration-driven data flow.

Core Components:
- MultiModalRouter: Universal RGB+X multimodal data router
- MultiModalConfigParser: YAML configuration parsing for multimodal architectures
- Utility functions: System status, model validation, and configuration helpers

Supported Modalities:
- RGB: 3-channel visible light images
- X: 3-channel unified other modality (depth/thermal/lidar/etc.)
- Dual: 6-channel RGB+X concatenated input

Features:
- Zero-copy tensor view routing
- Configuration-driven data flow
- Thread-safe caching mechanisms
- X modality new input start redirection
- Universal framework for RGB+X multimodal detection
"""

# Core multimodal router
from .router import MultiModalRouter

# Configuration parser
from .parser import MultiModalConfigParser
from .hook import HookManager, FeatureTap, BatchContext

# Utility functions
from .utils import (
    validate_mm_config_format,
    mm_system_status,
    check_mm_model_attributes,
    get_mm_system_info
)
from .generators import DepthGen, DEMGen, EdgeGen

# 版本信息（模块版本与项目展示版本分离）
__version__ = "v1.0"  # 模块内部版本
PROJECT_VERSION = "v0.1212"  # 多模态项目版本号（用于横幅展示）

# Export all components
__all__ = [
    # Core classes
    "MultiModalRouter",
    "MultiModalConfigParser",
    "HookManager",
    "FeatureTap",
    "BatchContext",
    
    # Utility functions
    "validate_mm_config_format",
    "mm_system_status", 
    "check_mm_model_attributes",
    "get_mm_system_info",
    # Generators (简化对外接口)
    "DepthGen",
    "DEMGen",
    "EdgeGen",
    
    # Version
    "__version__",
    "PROJECT_VERSION",
]

# Module metadata
__author__ = "YOLOMM Team"
__description__ = "Universal RGB+X Multimodal Routing System"
__supported_modalities__ = ["RGB", "X", "Dual"]
__supported_architectures__ = ["YOLO", "RTDETR"]
