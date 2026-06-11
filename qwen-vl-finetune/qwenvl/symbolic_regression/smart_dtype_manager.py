"""
修复的智能数据类型管理器
解决权重类型转换错误问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class SmartDtypeManager:
    """智能数据类型管理器"""
    
    def __init__(self, default_dtype=torch.float32, fallback_dtype=torch.float32):
        # 修复：使用float32作为默认类型，避免bfloat16问题
        self.default_dtype = default_dtype
        self.fallback_dtype = fallback_dtype
        self.force_float32_ops = {
            'softmax', 'layer_norm', 'cross_entropy', 'embedding'
        }
        
    @contextmanager 
    def force_dtype(self, dtype):
        """强制使用指定dtype的上下文管理器"""
        old_dtype = self.default_dtype
        self.default_dtype = dtype
        try:
            yield
        finally:
            self.default_dtype = old_dtype
    
    def get_safe_dtype(self, tensor: torch.Tensor, operation: str = "default") -> torch.dtype:
        """获取安全的数据类型"""
        # 修复：所有操作都使用float32确保稳定性
        return torch.float32
    
    def safe_cast(self, tensor: torch.Tensor, operation: str = "default") -> torch.Tensor:
        """安全地转换tensor类型"""
        target_dtype = self.get_safe_dtype(tensor, operation)
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

# 全局管理器实例
dtype_manager = SmartDtypeManager()

def safe_linear(input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """安全的线性变换，不修改原权重"""
    # 修复：确保不直接修改权重参数
    if input_tensor.dtype != weight.dtype:
        # 将输入转换为权重的类型，而不是相反
        input_tensor = input_tensor.to(weight.dtype)
    
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return F.linear(input_tensor, weight, bias)

def safe_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """安全的softmax，使用float32保证数值稳定性"""
    original_dtype = input_tensor.dtype
    
    # 转换为float32进行计算
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(torch.float32)
    
    result = F.softmax(input_tensor, dim=dim)
    
    # 转回原始类型
    if original_dtype != torch.float32:
        result = result.to(original_dtype)
    
    return result

def safe_layer_norm(input_tensor: torch.Tensor, normalized_shape, weight=None, bias=None, eps=1e-5) -> torch.Tensor:
    """安全的LayerNorm，使用float32保证数值稳定性"""
    original_dtype = input_tensor.dtype
    
    # 转换为float32进行计算
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(torch.float32)
    
    # 权重和偏置也转换为float32
    if weight is not None and weight.dtype != torch.float32:
        weight = weight.to(torch.float32)
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    
    result = F.layer_norm(input_tensor, normalized_shape, weight, bias, eps)
    
    # 转回原始类型
    if original_dtype != torch.float32:
        result = result.to(original_dtype)
    
    return result

class FixedLinear(nn.Linear):
    """修复的Linear层，避免权重类型问题"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 修复：使用安全的线性变换，不修改权重
        return safe_linear(input, self.weight, self.bias)

class FixedLayerNorm(nn.LayerNorm):
    """修复的LayerNorm层，避免权重类型问题"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 修复：使用安全的layer norm，不修改权重
        return safe_layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

def apply_safe_dtype_to_model(model: nn.Module, target_dtype: torch.dtype = torch.float32):
    """
    将模型安全地转换为指定dtype，避免权重修改问题
    
    Args:
        model: 要转换的模型
        target_dtype: 目标数据类型，推荐float32
    """
    # 修复：首先转换整个模型
    model = model.to(target_dtype)
    
    # 替换可能有问题的层
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, FixedLinear):
            # 创建FixedLinear替换
            fixed_linear = FixedLinear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            fixed_linear.weight.data = module.weight.data.to(target_dtype)
            if module.bias is not None:
                fixed_linear.bias.data = module.bias.data.to(target_dtype)
            
            # 替换模块
            parent_module = model
            module_names = name.split('.')
            for module_name in module_names[:-1]:
                parent_module = getattr(parent_module, module_name)
            setattr(parent_module, module_names[-1], fixed_linear)
            
        elif isinstance(module, nn.LayerNorm) and not isinstance(module, FixedLayerNorm):
            # 创建FixedLayerNorm替换
            fixed_ln = FixedLayerNorm(module.normalized_shape, eps=module.eps)
            if module.weight is not None:
                fixed_ln.weight.data = module.weight.data.to(target_dtype)
            if module.bias is not None:
                fixed_ln.bias.data = module.bias.data.to(target_dtype)
            
            # 替换模块
            parent_module = model
            module_names = name.split('.')
            for module_name in module_names[:-1]:
                parent_module = getattr(parent_module, module_name)
            setattr(parent_module, module_names[-1], fixed_ln)
    
    logger.info(f"模型已安全转换为{target_dtype}")
    return model

# 向后兼容的别名
SmartLinear = FixedLinear
SmartLayerNorm = FixedLayerNorm
smart_linear = safe_linear
smart_layer_norm = safe_layer_norm
smart_softmax = safe_softmax

def disable_monkey_patch():
    """禁用可能有问题的monkey patch"""
    # 修复：不进行任何monkey patch，使用原生PyTorch函数
    logger.info("已禁用monkey patch，使用原生PyTorch函数")
    pass

# 修复：默认禁用monkey patch
disable_monkey_patch() 