"""
修复的Set Transformer for Symbolic Regression
解决权重类型转换错误问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# 使用修复的dtype管理器
from .smart_dtype_manager import (
    FixedLinear, FixedLayerNorm, 
    safe_softmax
)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        """
        d_model: 模型维度
        h: attention head数量
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # 使用修复的Linear层
        self.w_q = FixedLinear(d_model, d_model)
        self.w_k = FixedLinear(d_model, d_model)
        self.w_v = FixedLinear(d_model, d_model)
        self.w_o = FixedLinear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力，使用安全softmax"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 使用安全softmax保证数值稳定性
        attention_weights = safe_softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.w_o(attention_output)
        
        return output

class SetAttentionBlock(nn.Module):
    """Set Attention Block (SAB)"""
    def __init__(self, d_model, h, dropout=0.1):
        super(SetAttentionBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, h, dropout)
        self.layer_norm1 = FixedLayerNorm(d_model)
        self.layer_norm2 = FixedLayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            FixedLinear(d_model, 4 * d_model),
            nn.ReLU(),
            FixedLinear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        attn_output = self.multihead_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x

class InducedSetAttentionBlock(nn.Module):
    """Induced Set Attention Block (ISAB)"""
    def __init__(self, d_model, h, m, dropout=0.1):
        super(InducedSetAttentionBlock, self).__init__()
        self.m = m  # 引导点数量
        self.d_model = d_model
        
        # 可学习的引导点 - 使用float32初始化
        self.I = nn.Parameter(torch.randn(1, m, d_model, dtype=torch.float32))
        
        self.multihead_attention1 = MultiHeadAttention(d_model, h, dropout)
        self.multihead_attention2 = MultiHeadAttention(d_model, h, dropout)
        self.layer_norm1 = FixedLayerNorm(d_model)
        self.layer_norm2 = FixedLayerNorm(d_model)
        self.layer_norm3 = FixedLayerNorm(d_model)
        self.layer_norm4 = FixedLayerNorm(d_model)
        
        self.feed_forward1 = nn.Sequential(
            FixedLinear(d_model, 4 * d_model),
            nn.ReLU(),
            FixedLinear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.feed_forward2 = nn.Sequential(
            FixedLinear(d_model, 4 * d_model),
            nn.ReLU(),
            FixedLinear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 修复：确保引导点的dtype与输入一致，但不修改原参数
        I = self.I.expand(batch_size, -1, -1)
        if I.dtype != x.dtype:
            I = I.to(x.dtype)
        
        # 第一步：引导点attend to输入
        H = self.multihead_attention1(I, x, x)
        H = self.layer_norm1(I + H)
        H = self.layer_norm2(H + self.feed_forward1(H))
        
        # 第二步：输入attend to引导点
        output = self.multihead_attention2(x, H, H)
        output = self.layer_norm3(x + output)
        output = self.layer_norm4(output + self.feed_forward2(output))
        
        return output

class PoolingByMultiheadAttention(nn.Module):
    """Pooling by Multihead Attention (PMA)"""
    def __init__(self, d_model, h, k, dropout=0.1):
        super(PoolingByMultiheadAttention, self).__init__()
        self.k = k  # 输出序列长度
        self.d_model = d_model
        
        # 可学习的种子向量 - 使用float32初始化
        self.S = nn.Parameter(torch.randn(1, k, d_model, dtype=torch.float32))
        
        self.multihead_attention = MultiHeadAttention(d_model, h, dropout)
        self.layer_norm1 = FixedLayerNorm(d_model)
        self.layer_norm2 = FixedLayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            FixedLinear(d_model, 4 * d_model),
            nn.ReLU(),
            FixedLinear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 修复：确保种子向量的dtype与输入一致，但不修改原参数
        S = self.S.expand(batch_size, -1, -1)
        if S.dtype != x.dtype:
            S = S.to(x.dtype)
        
        # Pooling attention
        output = self.multihead_attention(S, x, x)
        output = self.layer_norm1(S + output)
        output = self.layer_norm2(output + self.feed_forward(output))
        
        return output

class FixedSetTransformerEncoder(nn.Module):
    """
    修复的Set Transformer编码器，用于符号回归
    解决权重类型转换错误问题
    """
    
    def __init__(self, config):
        super(FixedSetTransformerEncoder, self).__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads  
        self.num_hidden_layers = config.num_set_layers
        self.dropout = getattr(config, 'dropout', 0.1)
        self.num_inducing_points = config.inducing_points
        self.pooling_outputs = config.pooling_outputs
        
        # 输入投影层 - 使用修复的Linear
        self.input_projection = FixedLinear(config.input_dim, self.hidden_size)
        
        # ISAB层
        self.encoder_layers = nn.ModuleList([
            InducedSetAttentionBlock(
                self.hidden_size, 
                self.num_attention_heads, 
                self.num_inducing_points, 
                self.dropout
            ) for _ in range(self.num_hidden_layers)
        ])
        
        # PMA层用于池化
        self.pooling = PoolingByMultiheadAttention(
            self.hidden_size, 
            self.num_attention_heads, 
            self.pooling_outputs, 
            self.dropout
        )
        
    def forward(self, data_points: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            data_points: (batch_size, num_points, input_dim) 数据点
            attention_mask: 可选的注意力掩码
            
        Returns:
            torch.Tensor: (batch_size, pooling_outputs, hidden_size) 编码后的特征
        """
        batch_size, num_points, input_dim = data_points.shape
        
        # 输入投影
        x = self.input_projection(data_points)  # (batch_size, num_points, hidden_size)
        
        # 通过ISAB层
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 池化到固定长度
        output = self.pooling(x)  # (batch_size, pooling_outputs, hidden_size)
        
        return output
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Fixed Set Transformer - 可训练参数: {trainable_params:,} || 总参数: {all_param:,} || 可训练比例: {100 * trainable_params / all_param:.2f}%")

# 向后兼容的别名
SetTransformerEncoder = FixedSetTransformerEncoder 