"""
Fixed symbolic regression Qwen model
Based on Qwen2.5-VL, replacing ViT with Set Transformer
Resolve weight dtype conversion errors
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

# Use fixed components
from .set_transformer import FixedSetTransformerEncoder
from .smart_dtype_manager import apply_safe_dtype_to_model, FixedLinear
from .data_processor import SymbolicRegressionConfig

class FixedSymbolicRegressionQwenModel(Qwen2_5_VLForConditionalGeneration):
    """
    Fixed symbolic regression version of the Qwen2.5-VL model
    Replace the vision encoder with Set Transformer to resolve weight dtype issues
    """
    
    def __init__(self, config, sr_config: Optional[SymbolicRegressionConfig] = None):
        super().__init__(config)
        
        # Set symbolic regression config
        if sr_config is None:
            sr_config = SymbolicRegressionConfig()
        self.sr_config = sr_config
        
        # Replace vision model with Set Transformer
        self.visual = FixedSymbolicRegressionVisualModel(sr_config)
        
        # Add dimension mapping layer using the fixed Linear layer
        self.feature_projector = FixedLinear(sr_config.hidden_size, config.hidden_size)
        
        # Fix: use float32 to avoid weight dtype issues
        self = apply_safe_dtype_to_model(self, torch.float32)
        
    def get_data_features(
        self,
        data_points: Optional[torch.Tensor] = None,
        data_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process data points, similar to the original get_vision_features
        
        Args:
            data_points: (batch_size, num_points, input_dim) Data points
            data_grid_thw: Data grid information
            
        Returns:
            torch.Tensor: Data features
        """
        if data_points is None:
            return None
            
        # Ensure data points have the correct type
        data_points = data_points.float()
        
        # Encode with Set Transformer
        data_features = self.visual(data_points)
        
        return data_features
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        data_points: Optional[torch.Tensor] = None,  # New: data point input
        data_grid_thw: Optional[torch.Tensor] = None,  # Data grid information
        pixel_values: Optional[torch.Tensor] = None,  # Keep compatibility
        image_grid_thw: Optional[torch.Tensor] = None,  # Keep compatibility
        pixel_values_videos: Optional[torch.FloatTensor] = None,  # Keep compatibility
        video_grid_thw: Optional[torch.Tensor] = None,  # Keep compatibility
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs  # Add kwargs for compatibility with extra parameters
    ):
        """
        Fixed forward pass supporting data point input
        """
        # Filter invalid kwargs
        filtered_kwargs = {}
        valid_keys = {
            'rope_deltas', 'rope_scaling_factor', 'use_sliding_window',
            'sliding_window', 'attention_mask', 'position_ids'
        }
        for key, value in kwargs.items():
            if key in valid_keys:
                filtered_kwargs[key] = value
        
        # If data points and input_ids are present, inject data features
        if data_points is not None and input_ids is not None:
            # Process data points
            data_features = self.get_data_features(data_points, data_grid_thw)
            
            # Get input embeddings
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            
            # Project data features to language model dimension
            projected_features = self.feature_projector(data_features)
            
            # Find vision token positions and inject data features
            vision_start_token_id = getattr(self.config, 'vision_start_token_id', 151652)
            vision_end_token_id = getattr(self.config, 'vision_end_token_id', 151653)
            
            batch_size = input_ids.shape[0]
            for batch_idx in range(batch_size):
                # Find vision token positions
                try:
                    start_pos = (input_ids[batch_idx] == vision_start_token_id).nonzero(as_tuple=True)[0]
                    end_pos = (input_ids[batch_idx] == vision_end_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(start_pos) > 0 and len(end_pos) > 0:
                        start_idx = start_pos[0].item()
                        end_idx = end_pos[0].item()
                        
                        # Replace embeddings between vision tokens with data features
                        if end_idx > start_idx + 1:
                            feature_len = projected_features.shape[1]
                            available_len = end_idx - start_idx - 1
                            
                            if feature_len <= available_len:
                                inputs_embeds[batch_idx, start_idx + 1:start_idx + 1 + feature_len] = projected_features[batch_idx]
                            else:
                                # If features are too long, use only the leading part
                                inputs_embeds[batch_idx, start_idx + 1:end_idx] = projected_features[batch_idx, :available_len]
                                
                except (RuntimeError, IndexError):
                    # If vision tokens are not found, skip this batch
                    continue
            
            # Now call the language model using input embeddings with data features already injected
            outputs = self.model(
                input_ids=None,  # Do not use input_ids; use inputs_embeds directly
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
            
            # Add language model head
            logits = self.lm_head(outputs.last_hidden_state)
            
            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Return standard CausalLMOutputWithPast format
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None,
            )
        else:
            # If there are no data points or no input_ids, call original forward
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
    Fixed symbolic regression vision model replacing ViT with Set Transformer
    Resolve weight dtype conversion errors
    """
    
    def __init__(self, config: SymbolicRegressionConfig):
        super().__init__()
        self.config = config
        
        # Set Transformer encoder - fixed version
        self.encoder = FixedSetTransformerEncoder(config)
        
        # MLP merger - uses fixed Linear layer
        self.merger = nn.Sequential(
            FixedLinear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            FixedLinear(config.hidden_size, config.hidden_size)
        )
        
        # Fix: use float32 to ensure stability
        self.to(torch.float32)
        
        # Add dtype attribute for compatibility with Qwen2.5-VL
        self.dtype = torch.float32
    
    def forward(self, data_points: torch.Tensor, grid_thw=None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data_points: (batch_size, num_points, input_dim)
            grid_thw: Grid information (compatibility parameter, ignored for now)
            
        Returns:
            torch.Tensor: (batch_size, pooling_outputs, hidden_size)
        """
        # Set Transformer encoding
        encoded_features = self.encoder(data_points)
        
        # MLP merge
        merged_features = self.merger(encoded_features)
        
        return merged_features
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Fixed Set Transformer Encoder: trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

# Backward-compatible alias
SymbolicRegressionQwenModel = FixedSymbolicRegressionQwenModel
SymbolicRegressionVisualModel = FixedSymbolicRegressionVisualModel 






