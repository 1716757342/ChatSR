#!/usr/bin/env python3
"""
Symbolic regression training script
Based on the official train_qwen.py, with support for Set Transformer + Qwen2.5-VL
"""

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_symbolic_regression import make_symbolic_regression_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.symbolic_regression.model import (
    SymbolicRegressionQwenModel,
    create_symbolic_regression_model
)
from qwenvl.symbolic_regression.data_processor import SymbolicRegressionConfig
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None

def rank0_print(*args):
    # Print in single-GPU mode or on the main process
    should_print = True
    if local_rank is not None:
        should_print = local_rank == 0
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        should_print = torch.distributed.get_rank() == 0

    if should_print:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save the model using the official method"""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def set_symbolic_regression_model(model_args, model):
    """Set trainable parameters for the symbolic regression model"""
    # Set Transformervision encoder
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # MLP merge layer
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # language model
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

def train_symbolic_regression(attn_implementation="flash_attention_2"):
    """Main symbolic regression training function"""
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set local_rank for compatibility with single-GPU and multi-GPU modes
    local_rank = getattr(training_args, 'local_rank', 0)
    if local_rank == -1:  # local_rank may be -1 in single-GPU mode
        local_rank = 0
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("🚀 Starting symbolic regression model training...")
    rank0_print(f"Base model path: {model_args.model_name_or_path}")
    rank0_print(f"Output directory: {training_args.output_dir}")
    rank0_print(f"Use DeepSpeed: {training_args.deepspeed is not None}")

    # Create symbolic regression config
    sr_config = SymbolicRegressionConfig()

    # Memory-optimized model loading plan
    rank0_print("📋 Memory-optimized model loading...")

    # Plan A: if using DeepSpeed, load on CPU to avoid GPU memory peaks
    device_map = "cpu" if training_args.deepspeed else "auto"

    # Load base model config without weights
    if "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLConfig
        config = Qwen2_5_VLConfig.from_pretrained(model_args.model_name_or_path)
        data_args.model_type = "qwen2.5vl"
    else:
        from transformers import Qwen2VLConfig
        config = Qwen2VLConfig.from_pretrained(model_args.model_name_or_path)
        data_args.model_type = "qwen2vl"

    # Create the symbolic regression model directly to avoid memory usage from intermediate steps
    model = SymbolicRegressionQwenModel(config, sr_config)

    # Now load pretrained weights into the symbolic regression model
    rank0_print("📋 Loading pretrained weights...")

    # Load model based on model type and DeepSpeed config
    if "qwen2.5" in model_args.model_name_or_path.lower():
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            device_map=device_map,  # Use optimized device mapping
        )
    else:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            device_map=device_map,  # Use optimized device mapping
        )

    # Copy weights in batches to reduce peak memory
    rank0_print("📋 Copying pretrained weights in batches...")

    # Copy language model weights
    model.model.load_state_dict(base_model.model.state_dict(), strict=False)

    # Copy language model head weights
    model.lm_head.load_state_dict(base_model.lm_head.state_dict())

    # Delete the base model immediately to free memory
    del base_model
    torch.cuda.empty_cache()
    rank0_print("✅ Pretrained weights loaded; base model released")

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Create tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Set model trainable parameters
    set_symbolic_regression_model(model_args, model)

    # Print parameter information
    is_main_process = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0

    if is_main_process:
        model.visual.print_trainable_parameters()
        if hasattr(model.model, 'print_trainable_parameters'):
            model.model.print_trainable_parameters()

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"📊 Model parameter statistics:")
        rank0_print(f"   Total parameters: {total_params:,}")
        rank0_print(f"   Trainable parameters: {trainable_params:,}")
        rank0_print(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        # Show memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            rank0_print(f"📊 Current GPU memory usage:")
            rank0_print(f"   Allocated: {memory_allocated:.2f} GB")
            rank0_print(f"   Reserved: {memory_reserved:.2f} GB")

    # Create data module
    rank0_print("📊 Create symbolic regression data module...")
    data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)

    # Create trainer - DeepSpeed compatible
    if training_args.deepspeed:
        rank0_print("🔧 Using DeepSpeed-optimized trainer...")

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )

    # Check for checkpoints
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("Checkpoint found; resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("🎯 Start training...")
        trainer.train()

    # Save model
    rank0_print("💾 Save model...")
    trainer.save_state()

    # Save symbolic regression config
    sr_config_path = os.path.join(training_args.output_dir, "symbolic_regression_config.json")
    with open(sr_config_path, 'w') as f:
        json.dump({
            'input_dim': sr_config.input_dim,
            'hidden_size': sr_config.hidden_size,
            'num_attention_heads': sr_config.num_attention_heads,
            'num_set_layers': sr_config.num_set_layers,
            'inducing_points': sr_config.inducing_points,
            'pooling_outputs': sr_config.pooling_outputs,
            'vocab_size': sr_config.vocab_size,
        }, f, indent=2)

    rank0_print(f"Symbolic regression config saved to: {sr_config_path}")

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print("✅ Symbolic regression model training completed!")

if __name__ == "__main__":
    train_symbolic_regression(attn_implementation="eager")