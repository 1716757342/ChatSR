# #!/usr/bin/env python3
# """
# 修复版本的符号回归训练脚本 - 分布式训练专用
# 解决DTensor混合错误的完整方案
# """

# import os
# import logging
# import pathlib
# import torch
# import transformers
# import json
# from typing import Dict
# import shutil
# import sys
# from pathlib import Path

# project_root = Path(__file__).parent
# sys.path.append(str(project_root / "qwen-vl-finetune"))

# from qwenvl.train.trainer import replace_qwen2_vl_attention_class

# from transformers import (
#     Qwen2VLForConditionalGeneration,
#     Qwen2_5_VLForConditionalGeneration,
# )
# from qwenvl.data.data_symbolic_regression import make_symbolic_regression_data_module
# from qwenvl.train.argument import (
#     ModelArguments,
#     DataArguments,
#     TrainingArguments,
# )
# from qwenvl.symbolic_regression.model import (
#     SymbolicRegressionQwenModel
# )
# from qwenvl.symbolic_regression.data_processor import SymbolicRegressionConfig
# from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

# local_rank = None

# def rank0_print(*args):
#     """在单GPU模式下或主进程中打印"""
#     should_print = True
#     if local_rank is not None:
#         should_print = local_rank == 0
#     elif torch.distributed.is_available() and torch.distributed.is_initialized():
#         should_print = torch.distributed.get_rank() == 0
    
#     if should_print:
#         print(*args)

# def set_symbolic_regression_model(model_args, model):
#     """设置符号回归模型的可训练参数"""
#     if model_args.tune_mm_llm:
#         for name, param in model.model.named_parameters():
#             param.requires_grad = True
#     else:
#         for name, param in model.model.named_parameters():
#             param.requires_grad = False

#     if model_args.tune_mm_vision:
#         for name, param in model.visual.named_parameters():
#             param.requires_grad = True
#     else:
#         for name, param in model.visual.named_parameters():
#             param.requires_grad = False

#     if model_args.tune_mm_mlp:
#         if hasattr(model, 'feature_projector'):
#             for name, param in model.feature_projector.named_parameters():
#                 param.requires_grad = True

# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
#     """HuggingFace Trainer的安全保存函数"""
#     try:
#         trainer.save_model(output_dir)
#     except Exception as e:
#         rank0_print(f"标准保存失败，尝试手动保存: {e}")
#         if hasattr(trainer.model, 'module'):
#             trainer.model.module.save_pretrained(output_dir)
#         else:
#             trainer.model.save_pretrained(output_dir)

# def train_symbolic_regression_distributed(attn_implementation="flash_attention_2"):
#     """符号回归分布式训练主函数 - 修复DTensor问题"""
#     global local_rank

#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # 设置local_rank，兼容单GPU和多GPU模式
#     local_rank = getattr(training_args, 'local_rank', 0)
#     if local_rank == -1:  # 单GPU模式下local_rank可能是-1
#         local_rank = 0
#     os.makedirs(training_args.output_dir, exist_ok=True)

#     rank0_print("🚀 开始符号回归模型训练 (分布式修复版本)...")
#     rank0_print(f"基础模型路径: {model_args.model_name_or_path}")
#     rank0_print(f"输出目录: {training_args.output_dir}")
#     rank0_print(f"Local Rank: {local_rank}")
#     rank0_print(f"是否使用DeepSpeed: {training_args.deepspeed is not None}")
#     rank0_print(f"是否使用FSDP: {'fsdp' in training_args.__dict__ and training_args.fsdp}")

#     # 创建符号回归配置
#     sr_config = SymbolicRegressionConfig()
    
#     # 🔧 关键修复：分布式环境下的模型加载策略
#     rank0_print("📋 分布式安全的模型加载...")
    
#     # 1. 首先加载基础模型配置（不加载权重）
#     if "qwen2.5" in model_args.model_name_or_path.lower():
#         from transformers import Qwen2_5_VLConfig
#         config = Qwen2_5_VLConfig.from_pretrained(model_args.model_name_or_path)
#         data_args.model_type = "qwen2.5vl"
#         base_model_class = Qwen2_5_VLForConditionalGeneration
#     else:
#         from transformers import Qwen2VLConfig  
#         config = Qwen2VLConfig.from_pretrained(model_args.model_name_or_path)
#         data_args.model_type = "qwen2vl"
#         base_model_class = Qwen2VLForConditionalGeneration
    
#     # 2. 先创建符号回归模型架构（不加载权重）
#     model = SymbolicRegressionQwenModel(config, sr_config)
    
#     # 3. 🔧 关键：分布式安全的权重加载
#     rank0_print("📋 分布式安全的权重加载...")
    
#     # 检测是否在分布式环境中
#     is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    
#     if is_distributed:
#         # 分布式环境：在主进程中加载权重，然后广播
#         if torch.distributed.get_rank() == 0:
#             rank0_print("🔧 主进程加载预训练权重...")
#             # 主进程加载权重（不使用device_map）
#             base_model = base_model_class.from_pretrained(
#                 model_args.model_name_or_path,
#                 cache_dir=training_args.cache_dir,
#                 torch_dtype=torch.bfloat16,
#                 # 🔧 关键：分布式环境下不使用device_map
#                 device_map=None,  
#                 low_cpu_mem_usage=True
#             )
            
#             # 复制权重到符号回归模型
#             model.model.load_state_dict(base_model.model.state_dict(), strict=False)
#             model.lm_head.load_state_dict(base_model.lm_head.state_dict())
            
#             # 释放基础模型
#             del base_model
#             torch.cuda.empty_cache()
#             rank0_print("✅ 主进程权重加载完成")
        
#         # 同步所有进程
#         torch.distributed.barrier()
#     else:
#         # 单GPU环境：正常加载
#         rank0_print("🔧 单GPU环境，正常加载权重...")
#         base_model = base_model_class.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             attn_implementation=attn_implementation,
#             torch_dtype=torch.bfloat16,
#             device_map="auto"  # 单GPU可以使用device_map
#         )
        
#         # 复制权重
#         model.model.load_state_dict(base_model.model.state_dict(), strict=False)
#         model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        
#         # 释放基础模型
#         del base_model
#         torch.cuda.empty_cache()
#         rank0_print("✅ 单GPU权重加载完成")

#     if data_args.data_flatten:
#         replace_qwen2_vl_attention_class()
#     model.config.use_cache = False

#     if training_args.gradient_checkpointing:
#         if hasattr(model, "enable_input_require_grads"):
#             model.enable_input_require_grads()
#         else:
#             def make_inputs_require_grad(module, input, output):
#                 output.requires_grad_(True)
#             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

#     # 创建分词器
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length,
#         padding_side="right",
#         use_fast=False,
#     )

#     # 设置模型可训练参数
#     set_symbolic_regression_model(model_args, model)

#     # 打印参数信息
#     is_main_process = True
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         is_main_process = torch.distributed.get_rank() == 0
    
#     if is_main_process:
#         if hasattr(model, 'visual') and hasattr(model.visual, 'print_trainable_parameters'):
#             model.visual.print_trainable_parameters()
#         if hasattr(model.model, 'print_trainable_parameters'):
#             model.model.print_trainable_parameters()
        
#         # 统计总参数
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         rank0_print(f"📊 模型参数统计:")
#         rank0_print(f"   总参数: {total_params:,}")
#         rank0_print(f"   可训练参数: {trainable_params:,}")
#         rank0_print(f"   可训练比例: {100 * trainable_params / total_params:.2f}%")
        
#         # 显示内存使用情况
#         if torch.cuda.is_available():
#             memory_allocated = torch.cuda.memory_allocated() / 1024**3
#             memory_reserved = torch.cuda.memory_reserved() / 1024**3
#             rank0_print(f"📊 当前GPU内存使用:")
#             rank0_print(f"   已分配: {memory_allocated:.2f} GB")
#             rank0_print(f"   已预留: {memory_reserved:.2f} GB")

#     # 创建数据模块
#     rank0_print("📊 创建符号回归数据模块...")
#     data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)
    
#     # 创建训练器
#     rank0_print("🔧 创建训练器...")
#     trainer = Trainer(
#         model=model, 
#         processing_class=tokenizer, 
#         args=training_args, 
#         **data_module
#     )

#     # 检查是否有检查点
#     if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#         rank0_print("🔄 发现检查点，恢复训练...")
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         rank0_print("🎯 开始训练...")
#         trainer.train()
    
#     # 保存模型
#     rank0_print("💾 保存模型...")
#     trainer.save_state()
    
#     # 保存符号回归配置
#     sr_config_path = os.path.join(training_args.output_dir, "symbolic_regression_config.json")
#     with open(sr_config_path, 'w') as f:
#         json.dump({
#             'input_dim': sr_config.input_dim,
#             'hidden_size': sr_config.hidden_size,
#             'num_attention_heads': sr_config.num_attention_heads,
#             'num_set_layers': sr_config.num_set_layers,
#             'inducing_points': sr_config.inducing_points,
#             'pooling_outputs': sr_config.pooling_outputs,
#             'vocab_size': sr_config.vocab_size,
#         }, f, indent=2)
    
#     rank0_print(f"符号回归配置已保存到: {sr_config_path}")

#     model.config.use_cache = True
#     safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
#     rank0_print("✅ 符号回归模型训练完成!")

# if __name__ == "__main__":
#     train_symbolic_regression_distributed() 



#### 版本2 ###

#!/usr/bin/env python3
"""
修复版本的符号回归训练脚本 - 分布式训练专用
在原始版本基础上集成 LoRA 支持以实现参数高效微调
"""

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import shutil
import sys
from pathlib import Path

# --- 项目路径设置 ---
project_root = Path(__file__).parent
sys.path.append(str(project_root / "qwen-vl-finetune"))

# --- 导入自定义模块 ---
from qwenvl.train.trainer import replace_qwen2_vl_attention_class
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer, 
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from qwenvl.data.data_symbolic_regression import make_symbolic_regression_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.symbolic_regression.model import (
    SymbolicRegressionQwenModel
)
from qwenvl.symbolic_regression.data_processor import SymbolicRegressionConfig


# --- 核心修改：导入 PEFT/LoRA 相关库 ---
from peft import LoraConfig, get_peft_model

# 全局变量，用于打印信息
local_rank = None

def rank0_print(*args):
    """仅在主进程中打印信息。"""
    should_print = True
    if local_rank is not None:
        should_print = local_rank == 0
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        should_print = torch.distributed.get_rank() == 0

    if should_print:
        print(*args)


def _is_rank0():
    return not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def evaluate_model_teacher_forcing_loss(model, data_module, device, title, max_samples=None, restore_train=True):
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    total = len(train_dataset) if max_samples is None else min(len(train_dataset), max_samples)
    losses = []
    model.eval()
    for idx in range(total):
        batch = data_collator([train_dataset[idx]])
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
        losses.append(float(outputs.loss.detach().cpu()))
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    if _is_rank0():
        print("\n" + "=" * 60)
        print(f"🧪 {title}: avg={avg_loss:.6f}, samples={len(losses)}")
        for idx, loss in enumerate(losses):
            sample_id = train_dataset.list_data_dict[idx].get("id", f"sample_{idx}")
            print(f"   {sample_id}: {loss:.6f}")
        print("=" * 60 + "\n")
    if restore_train:
        model.train()
    return avg_loss


def evaluate_training_loss_before_save(trainer, data_module, max_samples=None):
    return evaluate_model_teacher_forcing_loss(
        trainer.model,
        data_module,
        trainer.args.device,
        "训练结束保存前 teacher-forcing loss",
        max_samples=max_samples,
        restore_train=True,
    )


def print_reload_weight_diagnostics(reference_state_dict, reloaded_model, tokenizer, loading_info=None):
    if not _is_rank0():
        return

    print("\n" + "=" * 60)
    print("🔎 final_model 重载权重诊断")
    if loading_info is not None:
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        mismatched_keys = loading_info.get("mismatched_keys", [])
        print(f"   missing_keys: {len(missing_keys)}")
        print(f"   unexpected_keys: {len(unexpected_keys)}")
        print(f"   mismatched_keys: {len(mismatched_keys)}")
        for name, values in (("missing", missing_keys), ("unexpected", unexpected_keys), ("mismatched", mismatched_keys)):
            for key in values[:20]:
                print(f"   {name}: {key}")

    reloaded_state_dict = reloaded_model.state_dict()
    keys_to_compare = [
        "visual.encoder.input_projection.weight",
        "visual.encoder.input_projection.bias",
        "feature_projector.weight",
        "feature_projector.bias",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    if reference_state_dict:
        for key in keys_to_compare:
            if key not in reference_state_dict:
                print(f"   {key}: missing in reference state_dict")
                continue
            if key not in reloaded_state_dict:
                print(f"   {key}: missing in reloaded state_dict")
                continue
            reference_tensor = reference_state_dict[key].detach().cpu().float()
            reloaded_tensor = reloaded_state_dict[key].detach().cpu().float()
            if reference_tensor.shape != reloaded_tensor.shape:
                print(f"   {key}: shape mismatch {tuple(reference_tensor.shape)} vs {tuple(reloaded_tensor.shape)}")
                continue
            max_abs_diff = (reference_tensor - reloaded_tensor).abs().max().item()
            print(f"   {key}: max_abs_diff={max_abs_diff:.8f}")
    else:
        print("   reference state_dict 未提供，跳过逐权重 max_abs_diff 对比。")

    input_embeddings = reloaded_model.get_input_embeddings().weight
    lm_head = reloaded_model.lm_head.weight
    tied = input_embeddings.data_ptr() == lm_head.data_ptr()
    print(f"   lm_head 与 input_embeddings 是否共享 storage: {tied}")

    math_tokens = ["<|math_add|>", "<|math_x1|>", "<|math_C|>"]
    math_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in math_tokens]
    math_token_ids = [token_id for token_id in math_token_ids if token_id is not None and token_id >= 0]
    if math_token_ids:
        row_diff = (
            lm_head.detach().cpu()[math_token_ids].float()
            - input_embeddings.detach().cpu()[math_token_ids].float()
        ).abs().max().item()
        print(f"   math token rows lm_head/embed max_abs_diff: {row_diff:.8f}")
    print("=" * 60 + "\n")


def reload_final_model_and_evaluate(final_model_dir, full_state_dict, tokenizer, data_module, device):
    if not _is_rank0():
        return

    print(f"🔁 Reloading final_model for immediate save/load verification: {final_model_dir}", flush=True)
    reloaded_model, loading_info = SymbolicRegressionQwenModel.from_pretrained(
        final_model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        output_loading_info=True,
    )
    reloaded_model.to(device)
    reloaded_model.config.use_cache = False
    print_reload_weight_diagnostics(full_state_dict, reloaded_model, tokenizer, loading_info)
    evaluate_model_teacher_forcing_loss(
        reloaded_model,
        data_module,
        device,
        "final_model 保存后立即重载 teacher-forcing loss",
        restore_train=False,
    )
    del reloaded_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- 修复：移除有问题的 safe_save_model_for_hf_trainer 函数 ---
# 我们将直接使用 trainer.save_model()

def initialize_lm_head_from_embeddings(model):
    if not hasattr(model, "lm_head"):
        return
    input_embeddings = model.get_input_embeddings().weight
    lm_head = model.lm_head.weight
    if input_embeddings.shape != lm_head.shape:
        rank0_print(f"⚠️ 跳过 lm_head 初始化: embedding shape {tuple(input_embeddings.shape)} != lm_head shape {tuple(lm_head.shape)}")
        return
    with torch.no_grad():
        lm_head.copy_(input_embeddings)
    rank0_print("✅ Initialized independent lm_head from input embeddings.")


def train_symbolic_regression_distributed(attn_implementation="flash_attention_2"):
    """符号回归分布式训练主函数，集成LoRA"""
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = getattr(training_args, 'local_rank', 0)
    if local_rank == -1:
        local_rank = 0
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("🚀 Starting symbolic regression model training (Distributed Fixed Version)...")
    rank0_print(f"Base model path: {model_args.model_name_or_path}")
    rank0_print(f"Output directory: {training_args.output_dir}")
    rank0_print(f"Using FSDP: {'fsdp' in training_args.__dict__ and training_args.fsdp}")

    # 创建符号回归配置
    sr_config = SymbolicRegressionConfig()

    # FSDP 要求各 rank 在包装前拥有一致的参数初始化；这里确保自定义 SetTransformer/projector 初始化一致。
    transformers.set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    # --- 模型加载逻辑保持原始版本 ---
    rank0_print("📋 Loading model with FSDP-safe strategy...")
    if "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLConfig
        config = Qwen2_5_VLConfig.from_pretrained(model_args.model_name_or_path)
        base_model_class = Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLConfig  
        config = Qwen2VLConfig.from_pretrained(model_args.model_name_or_path)
        base_model_class = Qwen2VLForConditionalGeneration
    
    config.tie_word_embeddings = False
    rank0_print("🔧 Disabled tie_word_embeddings so lm_head reloads independently from input embeddings.")

    # 先创建符号回归模型架构（不加载权重）
    model = SymbolicRegressionQwenModel(config, sr_config)
    
    # 分布式安全的权重加载
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_distributed:
        rank0_print("🔧 Loading pretrained weights on every rank before FSDP wrapping...")
        base_model = base_model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model.load_state_dict(base_model.state_dict(), strict=False)
        initialize_lm_head_from_embeddings(model)
        del base_model
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        rank0_print("✅ Pretrained weights loaded on all ranks.")
    else:
        # 单GPU环境
        rank0_print("🔧 Loading pretrained weights for single GPU...")
        base_model = base_model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.load_state_dict(base_model.state_dict(), strict=False)
        initialize_lm_head_from_embeddings(model)
        del base_model
        torch.cuda.empty_cache()
        rank0_print("✅ Pretrained weights loaded.")

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # --- 核心修改：根据参数应用 LoRA 或进行全量微调 ---
    if model_args.lora_enable:
        rank0_print("🚀 Enabling LoRA for parameter-efficient fine-tuning...")
        
        # 首先冻结所有参数
        for name, param in model.named_parameters():
            param.requires_grad = False

        # 按需解冻需要训练的部分 (vision tower 和 mlp projector)
        if model_args.tune_mm_vision:
            model.visual.requires_grad_(True)
            rank0_print("   - Unfreezing Vision Tower (Set Transformer) for training.")
        if model_args.tune_mm_mlp:
            model.feature_projector.requires_grad_(True)
            rank0_print("   - Unfreezing MLP Projector for training.")

        # 创建 LoRA 配置
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(','),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用 LoRA 到模型
        model = get_peft_model(model, lora_config)
        rank0_print("✅ LoRA has been successfully applied to the model.")
        model.print_trainable_parameters() # 打印可训练参数信息
    else:
        # 如果不使用LoRA，则根据旧的参数设置
        rank0_print("🔧 Performing full or partial fine-tuning (LoRA is disabled).")
        model.model.requires_grad_(model_args.tune_mm_llm)
        if hasattr(model, 'lm_head'):
            model.lm_head.requires_grad_(model_args.tune_mm_llm)
            rank0_print(f"   - Setting LM head trainable: {model_args.tune_mm_llm}")
        model.visual.requires_grad_(model_args.tune_mm_vision)
        if hasattr(model, 'feature_projector'):
            model.feature_projector.requires_grad_(model_args.tune_mm_mlp)

    # 创建分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # # =================================================================
    # # ⬇️  关键：请将验证代码块移动到这里！ ⬇️
    # # =================================================================
    # rank0_print("🕵️  Verifying model state BEFORE training starts...")

    # # 注意：在使用FSDP或DDP时，模型被包裹了一层，需要访问 .module 来获取原始模型
    # model_to_check = model.module if hasattr(model, 'module') else model

    # try:
    #     tokenizer_vocab_size = len(tokenizer)
    #     model_embedding_size = model_to_check.get_input_embeddings().weight.shape[0]
    #     lm_head_size = model_to_check.lm_head.weight.shape[0]

    #     rank0_print(f"  - Tokenizer vocabulary size: {tokenizer_vocab_size}")
    #     rank0_print(f"  - Model input embedding size: {model_embedding_size}")
    #     rank0_print(f"  - Model LM head size: {lm_head_size}")

    #     if not (tokenizer_vocab_size == model_embedding_size == lm_head_size):
    #         rank0_print("🚨 FATAL ERROR: Vocabulary size mismatch detected before training!")
    #         # import sys
    #         # sys.exit(1) # 可以取消注释，在发现错误时直接终止程序
    #     else:
    #         rank0_print("✅ Vocabulary and model dimensions match. Starting training...")

    # except AttributeError as e:
    #     rank0_print(f"🚨 FATAL ERROR: Could not access a required model attribute. The model might be incomplete. Error: {e}")

    # # =================================================================

    # 创建数据模块
    rank0_print("📊 Creating symbolic regression data module...")
    data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)
    # 创建训练器
    rank0_print("🔧 Creating Trainer...")
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        **data_module
    )

    # # =================================================================
    # # ⬇️  请将下面的“最终验证”代码块完整地复制并粘贴到这里 ⬇️
    # # =================================================================

    # # --- 最终验证：检查第一个训练批次的分词情况 ---
    # # 只在主进程 (rank 0) 中执行此验证
    # if training_args.local_rank == 0:
    #     print("\n" + "="*60)
    #     print("🕵️  最终验证: 正在检查第一个实际训练批次的分词情况...")
        
    #     # 1. 获取训练数据加载器
    #     train_dataloader = trainer.get_train_dataloader()
        
    #     # 2. 从中取出一个批次的数据
    #     first_batch = next(iter(train_dataloader))
        
    #     # 3. 选择批次中的第一个样本进行检查
    #     sample_input_ids = first_batch['input_ids'][0]
    #     # --- 新增：直接打印Token ID列表 ---
    #     print("\n--- 第一个样本的Token ID列表 (送入模型前) ---")
    #     # 这就是模型嵌入层(Embedding Layer)接收到的最终输入
    #     print(sample_input_ids.tolist())
    #     # --- 新增结束 ---
    #     # 4. 使用 tokenizer.convert_ids_to_tokens 来查看最原始的分词结果
    #     # 这是最精确的验证方法
    #     token_list = tokenizer.convert_ids_to_tokens(sample_input_ids)
        
    #     print(f"\n--- 第一个样本的原始分词列表 (部分展示) ---")
    #     # 为了避免刷屏，我们只展示部分内容
    #     print(token_list[:100]) 
    #     print("...")

    #     # 5. 自动化检查我们关心的数学符号
    #     math_tokens_to_check = ["<|math_add|>", "<|math_log|>", "<|math_x1|>"]
    #     found_all_as_single = True
        
    #     for special_token in math_tokens_to_check:
    #         if special_token in token_list:
    #             print(f"  ✅ 验证成功: 在分词列表中找到了完整的 '{special_token}'")
    #         else:
    #             # 检查它是否被错误地切分了
    #             # 例如，检查 'math' 和 '_' 是否存在
    #             if 'math' in token_list and '_' in token_list:
    #                 print(f"  ❌ 验证失败: 未找到完整的 '{special_token}'，但可能已被切分！")
    #                 found_all_as_single = False

    #     print("\n--- 验证总结 ---")
    #     if found_all_as_single:
    #         print("🎉 最终确认: 数据在进入模型前的分词是正确的！")
    #     else:
    #         print("🚨 最终警告: 数据在进入模型前的分词是错误的！请检查您的Tokenizer创建和保存过程。")
        
    #     print("="*60 + "\n")


    # 检查是否有检查点并恢复训练
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    resume_from_checkpoint = None
    if last_checkpoint is not None:
        rank0_print(f"🔄 Checkpoint found at {last_checkpoint}, resuming training.")
        resume_from_checkpoint = last_checkpoint
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_training_loss_before_save(trainer, data_module)

    # 保存最终模型
    rank0_print("💾 Saving trainer state...")
    trainer.save_state()

    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    rank0_print(f"💾 Saving final model with Trainer.save_model() to: {final_model_dir}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        model_to_save.config.tie_word_embeddings = False
        trainer.save_model(final_model_dir)
        if torch.distributed.get_rank() == 0:
            tokenizer.save_pretrained(final_model_dir)
            rank0_print("✅ FSDP final model saved with Trainer.save_model().")
            reload_final_model_and_evaluate(final_model_dir, {}, tokenizer, data_module, trainer.args.device)
        torch.distributed.barrier()
    else:
        trainer.model.config.tie_word_embeddings = False
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        rank0_print("✅ Final model saved.")
        reload_final_model_and_evaluate(final_model_dir, trainer.model.state_dict(), tokenizer, data_module, trainer.args.device)


if __name__ == "__main__":
    train_symbolic_regression_distributed()




