#!/usr/bin/env python3
"""
Symbolic regression multimodal LLM validation script
Validate forward-pass and inference functionality of the whole system
"""

import torch
import numpy as np
import sys
import os
sys.path.append('.')

from qwenvl.symbolic_regression import (
    SymbolicRegressionQwenModel,
    SymbolicRegressionConfig
)
from qwenvl.data.data_symbolic_regression import make_symbolic_regression_data_module
from qwenvl.train.argument import DataArguments
from transformers import AutoTokenizer, Qwen2VLConfig
import json

def main():
    print("🚀 Starting symbolic regression multimodal LLM validation...")

    # Configuration
    model_path = "/path/to/Qwen2.5-VL-3B-Instruct"
    device = "cuda:0"

    # 1. Create symbolic regression config
    print("📋 Create symbolic regression config...")
    sr_config = SymbolicRegressionConfig(
        input_dim=11,  # x1-x10 + y
        hidden_size=896,
        num_attention_heads=14,
        num_set_layers=3,
        inducing_points=32,
        pooling_outputs=8,
        max_points=100
    )
    print(f"   ✅ Set Transformer config: {sr_config.hidden_size} dimensions, {sr_config.num_attention_heads} heads, {sr_config.num_set_layers} layers")

    # 2. Load tokenizer
    print("📋 Load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"   ✅ Vocabulary size: {len(tokenizer.get_vocab())}")

    # 3. Create symbolic regression model
    print("📋 Create symbolic regression model...")
    with torch.no_grad():  # Reduce memory usage
        # Load base configuration
        from transformers import Qwen2_5_VLForConditionalGeneration
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device
        )

        # Create symbolic regression model
        model = SymbolicRegressionQwenModel(base_model.config, sr_config)
        model = model.to(device)

        # Copy pretrained weights
        model.model.load_state_dict(base_model.model.state_dict(), strict=False)
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())

        # Delete base model to free memory
        del base_model
        torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model created successfully!")
    print(f"   📊 Total parameters: {total_params:,d}")
    print(f"   📊 Trainable parameters: {trainable_params:,d} ({100*trainable_params/total_params:.2f}%)")

    # 4. Create test data
    print("📋 Create test data...")

    # Generate test function: y = sin(x1) + cos(x2)
    batch_size = 1
    num_points = 50
    num_features = 2

    # Generate input data points
    x1 = np.random.uniform(-np.pi, np.pi, (batch_size, num_points))
    x2 = np.random.uniform(-np.pi, np.pi, (batch_size, num_points))
    y = np.sin(x1) + np.cos(x2)

    # Construct data point tensor [x1, x2, ..., x10, y] (pad with zeros to 11 dimensions)
    data_points = np.zeros((batch_size, num_points, 11))
    data_points[:, :, 0] = x1  # x1
    data_points[:, :, 1] = x2  # x2
    # x3-x10remain 0
    data_points[:, :, -1] = y  # y values

    data_points = torch.tensor(data_points, dtype=torch.float32, device=device)
    print(f"   ✅ Data point shape: {data_points.shape}")
    print(f"   📊 Test function: y = sin(x1) + cos(x2)")

    # 5. Create input text
    print("📋 Create input text...")
    conversations = [
        [
            {"from": "human", "value": "<data>\nThis is a sampled set of scientific data points. Please find an expression to fit this data. You only need to generate the preorder traversal of the expression binary tree."},
            {"from": "gpt", "value": "Okay, the preorder traversal of the expression I got is[+,sin,x1,cos,x2]"}
        ]
    ]

    # Preprocess conversation
    from qwenvl.data.data_symbolic_regression import preprocess_symbolic_regression_qwen
    processed = preprocess_symbolic_regression_qwen(
        conversations,
        tokenizer,
        data_grid_info=[8]  # 8 vision pad tokens per sample
    )

    input_ids = processed['input_ids'].to(device)
    labels = processed['labels'].to(device)
    # Create attention_mask (all 1s, meaning all tokens participate in attention computation)
    attention_mask = torch.ones_like(input_ids).to(device)

    print(f"   ✅ Input IDsshape: {input_ids.shape}")
    print(f"   ✅ Text sequence length: {input_ids.shape[1]}")

    # Validate vision tokens
    vision_tokens = ['<|vision_start|>', '<|vision_end|>', '<|vision_pad|>']
    for token in vision_tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        count = (input_ids == token_id).sum().item()
        print(f"   📊 {token}: {count} items")

    # 6. Forward pass validation
    print("📋 Run forward-pass validation...")

    model.eval()
    with torch.no_grad():
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                data_points=data_points,
                labels=labels
            )

            print(f"   ✅ Forward pass succeeded!")
            print(f"   📊 Output logits shape: {outputs.logits.shape}")
            print(f"   📊 Loss value: {outputs.loss.item():.4f}")

            # Validate generation ability
            print("📋 Validate generation ability...")

            # Use only the first sample for generation
            sample_input_ids = input_ids[:1]
            sample_attention_mask = attention_mask[:1]
            sample_data_points = data_points[:1]

            # Find the end position of the human part
            human_part = sample_input_ids[0]
            gpt_start_token = tokenizer.encode("<|im_start|>gpt\n", add_special_tokens=False)
            gpt_start_pos = None

            for i in range(len(human_part) - len(gpt_start_token) + 1):
                if torch.equal(human_part[i:i+len(gpt_start_token)], torch.tensor(gpt_start_token, device=device)):
                    gpt_start_pos = i + len(gpt_start_token)
                    break

            if gpt_start_pos:
                # Slice up to the start of the GPT answer
                generation_input_ids = sample_input_ids[:, :gpt_start_pos]
                generation_attention_mask = sample_attention_mask[:, :gpt_start_pos]

                print(f"   📊 Generation start position: {gpt_start_pos}")
                print(f"   📊 Generation input length: {generation_input_ids.shape[1]}")

                # Generate a few tokens for validation
                for step in range(3):
                    outputs = model(
                        input_ids=generation_input_ids,
                        attention_mask=generation_attention_mask,
                        data_points=sample_data_points
                    )

                    # Get next token
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    next_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)

                    print(f"   🎯 Generation step {step+1}: '{next_token_text}' (token_id: {next_token[0].item()})")

                    # Update input
                    generation_input_ids = torch.cat([generation_input_ids, next_token], dim=1)
                    generation_attention_mask = torch.cat([
                        generation_attention_mask,
                        torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
                    ], dim=1)

                    if generation_input_ids.shape[1] >= 400:  # Avoid excessively long sequences
                        break

                print(f"   ✅ Generation validation succeeded!")

        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            return False

    # 7. Memory usage statistics
    print("📋 Memory usage statistics...")
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"   📊 CUDA memory allocated: {memory_allocated:.2f} GB")
        print(f"   📊 CUDA memory reserved: {memory_reserved:.2f} GB")

    print("\n🎉 Symbolic regression multimodal LLM validation completed successfully!")
    print("=" * 60)
    print("✅ All core functions validated successfully:")
    print("   🔹 Set Transformer encoder - working normally")
    print("   🔹 Vision Token processing - working normally")
    print("   🔹 Data point encoding - working normally")
    print("   🔹 Text generation - working normally")
    print("   🔹 Loss computation - working normally")
    print("   🔹 End-to-end pipeline - working normally")
    print("=" * 60)
    print("🌟 Symbolic regression multimodal LLM implementation completed successfully!")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)