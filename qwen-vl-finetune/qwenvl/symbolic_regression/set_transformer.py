"""
Fixed Set Transformer for symbolic regression
Resolve weight dtype conversion errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Use fixed dtype manager
from .smart_dtype_manager import (
    FixedLinear, FixedLayerNorm, 
    safe_softmax
)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        """
        d_model: Model dimension
        h: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # Use fixed Linear layer
        self.w_q = FixedLinear(d_model, d_model)
        self.w_k = FixedLinear(d_model, d_model)
        self.w_v = FixedLinear(d_model, d_model)
        self.w_o = FixedLinear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention using safe softmax"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Use safe softmax to ensure numerical stability
        attention_weights = safe_softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Apply linear transform and reshape into multiple heads
        Q = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate multiple heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear transform
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
        self.m = m  # Number of inducing points
        self.d_model = d_model
        
        # Learnable inducing points initialized with float32
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
        
        # Fix: ensure inducing point dtype matches input without modifying original parameters
        I = self.I.expand(batch_size, -1, -1)
        if I.dtype != x.dtype:
            I = I.to(x.dtype)
        
        # Step 1: inducing points attend to input
        H = self.multihead_attention1(I, x, x)
        H = self.layer_norm1(I + H)
        H = self.layer_norm2(H + self.feed_forward1(H))
        
        # Step 2: input attends to inducing points
        output = self.multihead_attention2(x, H, H)
        output = self.layer_norm3(x + output)
        output = self.layer_norm4(output + self.feed_forward2(output))
        
        return output

class PoolingByMultiheadAttention(nn.Module):
    """Pooling by Multihead Attention (PMA)"""
    def __init__(self, d_model, h, k, dropout=0.1):
        super(PoolingByMultiheadAttention, self).__init__()
        self.k = k  # Output sequence length
        self.d_model = d_model
        
        # Learnable seed vectors initialized with float32
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
        
        # Fix: ensure seed vector dtype matches input without modifying original parameters
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
    Fixed Set Transformer encoder for symbolic regression
    Resolve weight dtype conversion errors
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
        
        # Input projection layer - uses fixed Linear
        self.input_projection = FixedLinear(config.input_dim, self.hidden_size)
        
        # ISAB layers
        self.encoder_layers = nn.ModuleList([
            InducedSetAttentionBlock(
                self.hidden_size, 
                self.num_attention_heads, 
                self.num_inducing_points, 
                self.dropout
            ) for _ in range(self.num_hidden_layers)
        ])
        
        # PMA layer for pooling
        self.pooling = PoolingByMultiheadAttention(
            self.hidden_size, 
            self.num_attention_heads, 
            self.pooling_outputs, 
            self.dropout
        )
        
    def forward(self, data_points: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            data_points: (batch_size, num_points, input_dim) Data points
            attention_mask: Optional attention mask
            
        Returns:
            torch.Tensor: (batch_size, pooling_outputs, hidden_size) Encoded features
        """
        batch_size, num_points, input_dim = data_points.shape
        
        # Input projection
        x = self.input_projection(data_points)  # (batch_size, num_points, hidden_size)
        
        # Pass through ISAB layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Pool to fixed length
        output = self.pooling(x)  # (batch_size, pooling_outputs, hidden_size)
        
        return output
    
    def print_trainable_parameters(self):
        """Print trainable parameter information"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Fixed Set Transformer - trainable parameters: {trainable_params:,} || Total parameters: {all_param:,} || Trainable ratio: {100 * trainable_params / all_param:.2f}%")

# Backward-compatible alias
SetTransformerEncoder = FixedSetTransformerEncoder 