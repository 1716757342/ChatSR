"""
修复的符号回归Qwen模型
基于Qwen2.5-VL，用Set Transformer替换ViT
解决权重类型转换错误问题
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

# 使用修复的组件
from .set_transformer import FixedSetTransformerEncoder
from .smart_dtype_manager import apply_safe_dtype_to_model, FixedLinear
from .data_processor import SymbolicRegressionConfig

class FixedSymbolicRegressionQwenModel(Qwen2_5_VLForConditionalGeneration):
    """
    修复的符号回归版本的Qwen2.5-VL模型
    用Set Transformer替换视觉编码器，解决权重类型问题
    """
    
    def __init__(self, config, sr_config: Optional[SymbolicRegressionConfig] = None):
        super().__init__(config)
        
        # 设置符号回归配置
        if sr_config is None:
            sr_config = SymbolicRegressionConfig()
        self.sr_config = sr_config
        
        # 替换视觉模型为Set Transformer
        self.visual = FixedSymbolicRegressionVisualModel(sr_config)
        
        # 添加维度映射层，使用修复的Linear层
        self.feature_projector = FixedLinear(sr_config.hidden_size, config.hidden_size)
        
        # 修复：使用float32避免权重类型问题
        self = apply_safe_dtype_to_model(self, torch.float32)
        
    def get_data_features(
        self,
        data_points: Optional[torch.Tensor] = None,
        data_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        处理数据点，类似于原始的get_vision_features
        
        Args:
            data_points: (batch_size, num_points, input_dim) 数据点
            data_grid_thw: 数据网格信息
            
        Returns:
            torch.Tensor: 数据特征
        """
        if data_points is None:
            return None
            
        # 确保数据点是正确的类型
        data_points = data_points.float()
        
        # 使用Set Transformer编码
        data_features = self.visual(data_points)
        
        return data_features
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        data_points: Optional[torch.Tensor] = None,  # 新增：数据点输入
        data_grid_thw: Optional[torch.Tensor] = None,  # 数据网格信息
        pixel_values: Optional[torch.Tensor] = None,  # 保持兼容性
        image_grid_thw: Optional[torch.Tensor] = None,  # 保持兼容性
        pixel_values_videos: Optional[torch.FloatTensor] = None,  # 保持兼容性
        video_grid_thw: Optional[torch.Tensor] = None,  # 保持兼容性
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs  # 添加kwargs以兼容额外参数
    ):
        """
        修复的前向传播，支持数据点输入
        """
        # 过滤掉无效的kwargs
        filtered_kwargs = {}
        valid_keys = {
            'rope_deltas', 'rope_scaling_factor', 'use_sliding_window',
            'sliding_window', 'attention_mask', 'position_ids'
        }
        for key, value in kwargs.items():
            if key in valid_keys:
                filtered_kwargs[key] = value
        
        # 如果有数据点和input_ids，需要注入数据特征
        if data_points is not None and input_ids is not None:
            # 处理数据点
            data_features = self.get_data_features(data_points, data_grid_thw)
            
            # 获取输入嵌入
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            
            # 投影数据特征到语言模型维度
            projected_features = self.feature_projector(data_features)
            
            # 找到视觉token位置并注入数据特征
            vision_start_token_id = getattr(self.config, 'vision_start_token_id', 151652)
            vision_end_token_id = getattr(self.config, 'vision_end_token_id', 151653)
            
            batch_size = input_ids.shape[0]
            for batch_idx in range(batch_size):
                # 查找vision token位置
                try:
                    start_pos = (input_ids[batch_idx] == vision_start_token_id).nonzero(as_tuple=True)[0]
                    end_pos = (input_ids[batch_idx] == vision_end_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(start_pos) > 0 and len(end_pos) > 0:
                        start_idx = start_pos[0].item()
                        end_idx = end_pos[0].item()
                        
                        # 替换vision token之间的嵌入为数据特征
                        if end_idx > start_idx + 1:
                            feature_len = projected_features.shape[1]
                            available_len = end_idx - start_idx - 1
                            
                            if feature_len <= available_len:
                                inputs_embeds[batch_idx, start_idx + 1:start_idx + 1 + feature_len] = projected_features[batch_idx]
                            else:
                                # 如果特征太长，只使用前面的部分
                                inputs_embeds[batch_idx, start_idx + 1:end_idx] = projected_features[batch_idx, :available_len]
                                
                except (RuntimeError, IndexError):
                    # 如果找不到vision token，跳过这个batch
                    continue
            
            # 现在调用语言模型，使用已经注入数据特征的输入嵌入
            outputs = self.model(
                input_ids=None,  # 不使用input_ids，直接使用inputs_embeds
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **filtered_kwargs
            )
            
            # 添加语言模型头
            logits = self.lm_head(outputs.last_hidden_state)
            
            # 计算损失（如果提供了标签）
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 返回标准的CausalLMOutputWithPast格式
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None,
            )
        else:
            # 没有数据点或没有input_ids，调用原始forward
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **filtered_kwargs
            )

class FixedSymbolicRegressionVisualModel(nn.Module):
    """
    修复的符号回归视觉模型，用Set Transformer替换ViT
    解决权重类型转换错误问题
    """
    
    def __init__(self, config: SymbolicRegressionConfig):
        super().__init__()
        self.config = config
        
        # Set Transformer编码器 - 使用修复版本
        self.encoder = FixedSetTransformerEncoder(config)
        
        # MLP merger - 使用修复的Linear层
        self.merger = nn.Sequential(
            FixedLinear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            FixedLinear(config.hidden_size, config.hidden_size)
        )
        
        # 修复：使用float32确保稳定性
        self.to(torch.float32)
        
        # 添加dtype属性以兼容Qwen2.5-VL
        self.dtype = torch.float32
    
    def forward(self, data_points: torch.Tensor, grid_thw=None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data_points: (batch_size, num_points, input_dim)
            grid_thw: 网格信息（兼容性参数，暂时忽略）
            
        Returns:
            torch.Tensor: (batch_size, pooling_outputs, hidden_size)
        """
        # Set Transformer编码
        encoded_features = self.encoder(data_points)
        
        # MLP合并
        merged_features = self.merger(encoded_features)
        
        return merged_features
    
    def print_trainable_parameters(self):
        """打印可训练参数数量"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Fixed Set Transformer Encoder: trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

# 向后兼容的别名
SymbolicRegressionQwenModel = FixedSymbolicRegressionQwenModel
SymbolicRegressionVisualModel = FixedSymbolicRegressionVisualModel 






