"""
符号回归模块 - 基于Qwen2.5-VL框架
替换ViT为Set Transformer，支持数学表达式生成
"""

from .set_transformer import SetTransformerEncoder
from .data_processor import SymbolicRegressionDataProcessor, SymbolicRegressionConfig
from .model import SymbolicRegressionQwenModel
from .smart_dtype_manager import SmartDtypeManager

__all__ = [
    "SetTransformerEncoder",
    "SymbolicRegressionDataProcessor", 
    "SymbolicRegressionConfig",
    "SymbolicRegressionQwenModel",
    "SmartDtypeManager",
    "apply_smart_dtype_to_model",
] 