"""
Symbolic regression module based on the Qwen2.5-VL framework
Replaces ViT with Set Transformer and supports mathematical expression generation
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