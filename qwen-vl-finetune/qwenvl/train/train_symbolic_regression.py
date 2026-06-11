#!/usr/bin/env python3
"""
符号回归训练脚本
基于官方train_qwen.py，支持Set Transformer + Qwen2.5-VL
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
    # 在单GPU模式下或主进程中打印
    should_print = True
    if local_rank is not None:
        should_print = local_rank == 0
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        should_print = torch.distributed.get_rank() == 0
    
    if should_print:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """保存模型，采用官方方法"""
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
    """设置符号回归模型的可训练参数"""
    # Set Transformer视觉编码器
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # MLP合并层
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # 语言模型
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

def train_symbolic_regression(attn_implementation="flash_attention_2"):
    """符号回归训练主函数"""
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置local_rank，兼容单GPU和多GPU模式
    local_rank = getattr(training_args, 'local_rank', 0)
    if local_rank == -1:  # 单GPU模式下local_rank可能是-1
        local_rank = 0
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("🚀 开始符号回归模型训练...")
    rank0_print(f"基础模型路径: {model_args.model_name_or_path}")
    rank0_print(f"输出目录: {training_args.output_dir}")
    rank0_print(f"是否使用DeepSpeed: {training_args.deepspeed is not None}")

    # 创建符号回归配置
    sr_config = SymbolicRegressionConfig()
    
    # 内存优化的模型加载方案
    rank0_print("📋 内存优化模型加载...")
    
    # 方案A: 如果使用DeepSpeed，使用CPU加载避免GPU内存峰值
    device_map = "cpu" if training_args.deepspeed else "auto"
    
    # 加载基础模型配置（不加载权重）
    if "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLConfig
        config = Qwen2_5_VLConfig.from_pretrained(model_args.model_name_or_path)
        data_args.model_type = "qwen2.5vl"
    else:
        from transformers import Qwen2VLConfig  
        config = Qwen2VLConfig.from_pretrained(model_args.model_name_or_path)
        data_args.model_type = "qwen2vl"
    
    # 直接创建符号回归模型（避免中间步骤的内存占用）
    model = SymbolicRegressionQwenModel(config, sr_config)
    
    # 现在加载预训练权重到符号回归模型
    rank0_print("📋 加载预训练权重...")
    
    # 根据模型类型和DeepSpeed配置加载模型
    if "qwen2.5" in model_args.model_name_or_path.lower():
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,  # 使用bfloat16节省内存
            device_map=device_map,  # 使用优化的设备映射
        )
    else:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,  # 使用bfloat16节省内存
            device_map=device_map,  # 使用优化的设备映射
        )

    # 分批复制权重以减少内存峰值
    rank0_print("📋 分批复制预训练权重...")
    
    # 复制语言模型权重
    model.model.load_state_dict(base_model.model.state_dict(), strict=False)
    
    # 复制语言模型头权重  
    model.lm_head.load_state_dict(base_model.lm_head.state_dict())
    
    # 立即删除基础模型释放内存
    del base_model
    torch.cuda.empty_cache()
    rank0_print("✅ 预训练权重加载完成，基础模型已释放")

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

    # 创建分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 设置模型可训练参数
    set_symbolic_regression_model(model_args, model)

    # 打印参数信息
    is_main_process = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    
    if is_main_process:
        model.visual.print_trainable_parameters()
        if hasattr(model.model, 'print_trainable_parameters'):
            model.model.print_trainable_parameters()
        
        # 统计总参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"📊 模型参数统计:")
        rank0_print(f"   总参数: {total_params:,}")
        rank0_print(f"   可训练参数: {trainable_params:,}")
        rank0_print(f"   可训练比例: {100 * trainable_params / total_params:.2f}%")
        
        # 显示内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            rank0_print(f"📊 当前GPU内存使用:")
            rank0_print(f"   已分配: {memory_allocated:.2f} GB")
            rank0_print(f"   已预留: {memory_reserved:.2f} GB")

    # 创建数据模块
    rank0_print("📊 创建符号回归数据模块...")
    data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 创建训练器 - 兼容DeepSpeed
    if training_args.deepspeed:
        rank0_print("🔧 使用DeepSpeed优化训练器...")
    
    trainer = Trainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        **data_module
    )

    # 检查是否有检查点
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("发现检查点，恢复训练")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("🎯 开始训练...")
        trainer.train()
    
    # 保存模型
    rank0_print("💾 保存模型...")
    trainer.save_state()
    
    # 保存符号回归配置
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
    
    rank0_print(f"符号回归配置已保存到: {sr_config_path}")

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    rank0_print("✅ 符号回归模型训练完成!")

if __name__ == "__main__":
    train_symbolic_regression(attn_implementation="eager") 