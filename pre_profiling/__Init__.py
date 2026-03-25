"""
pre_profiling: 神经网络模块预分析工具
用于测量各层前向/反向传播时间，支持NPU/CUDA/CPU
"""

from .pre_profiling import (
    PreProfiler,
    run_pre_profiling,
    ProfilingResult,
    LayerInfo
)
from .layer_config_parser import LayerConfigParser

__version__ = "1.0.0"
__all__ = [
    "PreProfiler",
    "run_pre_profiling",
    "ProfilingResult",
    "LayerInfo",
    "LayerConfigParser"
]