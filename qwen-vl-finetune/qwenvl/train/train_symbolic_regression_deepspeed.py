#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symbolic regression training script - DeepSpeed optimized version
Specifically handles memory optimization issues in multi-GPU training
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
import torch.nn as nn

# Set path
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

# Use the existing symbolic regression model directly


def train_symbolic_regression_deepspeed(attn_implementation="flash_attention_2"):
    """Main symbolic regression training function - DeepSpeed optimized version"""
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set local_rank for compatibility with single-GPU and multi-GPU modes
    local_rank = getattr(training_args, 'local_rank', 0)
    if local_rank == -1:
        local_rank = 0
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("🚀 Starting symbolic regression model training (DeepSpeed optimized version)...")
    rank0_print(f"Base model path: {model_args.model_name_or_path}")
    rank0_print(f"Output directory: {training_args.output_dir}")
    rank0_print(f"Use DeepSpeed: {training_args.deepspeed is not None}")

    # Create symbolic regression config
    sr_config = SymbolicRegressionConfig()
    
    # Use simplified model loading strategy
    rank0_print("📋 DeepSpeed mode: create symbolic regression model directly...")
    
    # Load base model config
    if "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLConfig
        config = Qwen2_5_VLConfig.from_pretrained(model_args.model_name_or_path)
        data_args.model_type = "qwen2.5vl"
    else:
        from transformers import Qwen2VLConfig  
        config = Qwen2VLConfig.from_pretrained(model_args.model_name_or_path)
        data_args.model_type = "qwen2vl"
    
    # Create symbolic regression model directly
    model = SymbolicRegressionQwenModel(config, sr_config)
    
    # In DeepSpeed mode, let the trainer handle weight loading automatically
    rank0_print("📋 Let DeepSpeed handle model initialization automatically...")

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
    
    # Create processor
    if "qwen2.5" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=training_args.cache_dir
        )
        image_processor = processor.image_processor
    else:
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=training_args.cache_dir
        )

    # Data module
    data_module = make_symbolic_regression_data_module(
        tokenizer=tokenizer,
        image_processor=image_processor, 
        data_args=data_args
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    rank0_print("✅ Symbolic regression model training completed!")


if __name__ == "__main__":
    import pathlib
    train_symbolic_regression_deepspeed(attn_implementation="eager") 