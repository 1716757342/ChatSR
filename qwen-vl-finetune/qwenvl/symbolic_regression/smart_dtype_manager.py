"""
Fixed smart dtype manager
Resolve weight dtype conversion errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class SmartDtypeManager:
    """Smart dtype manager"""
    
    def __init__(self, default_dtype=torch.float32, fallback_dtype=torch.float32):
        # Fix: use float32 as default type to avoid bfloat16 issues
        self.default_dtype = default_dtype
        self.fallback_dtype = fallback_dtype
        self.force_float32_ops = {
            'softmax', 'layer_norm', 'cross_entropy', 'embedding'
        }
        
    @contextmanager 
    def force_dtype(self, dtype):
        """Context manager that forces the specified dtype"""
        old_dtype = self.default_dtype
        self.default_dtype = dtype
        try:
            yield
        finally:
            self.default_dtype = old_dtype
    
    def get_safe_dtype(self, tensor: torch.Tensor, operation: str = "default") -> torch.dtype:
        """Get safe dtype"""
        # Fix: use float32 for all operations to ensure stability
        return torch.float32
    
    def safe_cast(self, tensor: torch.Tensor, operation: str = "default") -> torch.Tensor:
        """Safely convert tensor dtype"""
        target_dtype = self.get_safe_dtype(tensor, operation)
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

# Global manager instance
dtype_manager = SmartDtypeManager()

def safe_linear(input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Safe linear transform without modifying original weights"""
    # Fix: ensure weight parameters are not modified directly
    if input_tensor.dtype != weight.dtype:
        # Convert input to the weight dtype, not the reverse
        input_tensor = input_tensor.to(weight.dtype)
    
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return F.linear(input_tensor, weight, bias)

def safe_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Safe softmax using float32 for numerical stability"""
    original_dtype = input_tensor.dtype
    
    # Convert to float32 for computation
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(torch.float32)
    
    result = F.softmax(input_tensor, dim=dim)
    
    # Convert back to original dtype
    if original_dtype != torch.float32:
        result = result.to(original_dtype)
    
    return result

def safe_layer_norm(input_tensor: torch.Tensor, normalized_shape, weight=None, bias=None, eps=1e-5) -> torch.Tensor:
    """Safe LayerNorm using float32 for numerical stability"""
    original_dtype = input_tensor.dtype
    
    # Convert to float32 for computation
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(torch.float32)
    
    # Convert weight and bias to float32 as well
    if weight is not None and weight.dtype != torch.float32:
        weight = weight.to(torch.float32)
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    
    result = F.layer_norm(input_tensor, normalized_shape, weight, bias, eps)
    
    # Convert back to original dtype
    if original_dtype != torch.float32:
        result = result.to(original_dtype)
    
    return result

class FixedLinear(nn.Linear):
    """Fixed Linear layer avoiding weight dtype issues"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Fix: use safe linear transform without modifying weights
        return safe_linear(input, self.weight, self.bias)

class FixedLayerNorm(nn.LayerNorm):
    """Fixed LayerNorm layer avoiding weight dtype issues"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Fix: use safe layer norm without modifying weights
        return safe_layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

def apply_safe_dtype_to_model(model: nn.Module, target_dtype: torch.dtype = torch.float32):
    """
    Safely convert the model to the specified dtype, avoiding weight modification issues
    
    Args:
        model: Model to convert
        target_dtype: Target dtype; float32 is recommended
    """
    # Fix: convert the entire model first
    model = model.to(target_dtype)
    
    # Replace potentially problematic layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, FixedLinear):
            # Create FixedLinear replacement
            fixed_linear = FixedLinear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            fixed_linear.weight.data = module.weight.data.to(target_dtype)
            if module.bias is not None:
                fixed_linear.bias.data = module.bias.data.to(target_dtype)
            
            # Replace module
            parent_module = model
            module_names = name.split('.')
            for module_name in module_names[:-1]:
                parent_module = getattr(parent_module, module_name)
            setattr(parent_module, module_names[-1], fixed_linear)
            
        elif isinstance(module, nn.LayerNorm) and not isinstance(module, FixedLayerNorm):
            # Create FixedLayerNorm replacement
            fixed_ln = FixedLayerNorm(module.normalized_shape, eps=module.eps)
            if module.weight is not None:
                fixed_ln.weight.data = module.weight.data.to(target_dtype)
            if module.bias is not None:
                fixed_ln.bias.data = module.bias.data.to(target_dtype)
            
            # Replace module
            parent_module = model
            module_names = name.split('.')
            for module_name in module_names[:-1]:
                parent_module = getattr(parent_module, module_name)
            setattr(parent_module, module_names[-1], fixed_ln)
    
    logger.info(f"Model safely converted to {target_dtype}")
    return model

# Backward-compatible alias
SmartLinear = FixedLinear
SmartLayerNorm = FixedLayerNorm
smart_linear = safe_linear
smart_layer_norm = safe_layer_norm
smart_softmax = safe_softmax

def disable_monkey_patch():
    """Disable potentially problematic monkey patch"""
    # Fix: do not apply any monkey patch; use native PyTorch functions
    logger.info("Monkey patch disabled; using native PyTorch functions")
    pass

# Fix: disable monkey patch by default
disable_monkey_patch() 