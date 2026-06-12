# #!/usr/bin/env python3
# """
# Fixed symbolic regression training script - for distributed training
# Complete solution for DTensor mixing errors
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
#     """Print in single-GPU mode or on the main process"""
#     should_print = True
#     if local_rank is not None:
#         should_print = local_rank == 0
#     elif torch.distributed.is_available() and torch.distributed.is_initialized():
#         should_print = torch.distributed.get_rank() == 0
    
#     if should_print:
#         print(*args)

# def set_symbolic_regression_model(model_args, model):
#     """Set trainable parameters for the symbolic regression model"""
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
#     """Safe save function for HuggingFace Trainer"""
#     try:
#         trainer.save_model(output_dir)
#     except Exception as e:
#         rank0_print(f"Standard save failed; trying manual save: {e}")
#         if hasattr(trainer.model, 'module'):
#             trainer.model.module.save_pretrained(output_dir)
#         else:
#             trainer.model.save_pretrained(output_dir)

# def train_symbolic_regression_distributed(attn_implementation="flash_attention_2"):
#     """Main symbolic regression distributed training function - fixes DTensor issues"""
#     global local_rank

#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Set local_rank for compatibility with single-GPU and multi-GPU modes
#     local_rank = getattr(training_args, 'local_rank', 0)
#     if local_rank == -1:  # local_rank may be -1 in single-GPU mode
#         local_rank = 0
#     os.makedirs(training_args.output_dir, exist_ok=True)

#     rank0_print("🚀 Starting symbolic regression model training (distributed fixed version)...")
#     rank0_print(f"Base model path: {model_args.model_name_or_path}")
#     rank0_print(f"Output directory: {training_args.output_dir}")
#     rank0_print(f"Local Rank: {local_rank}")
#     rank0_print(f"Use DeepSpeed: {training_args.deepspeed is not None}")
#     rank0_print(f"Use FSDP: {'fsdp' in training_args.__dict__ and training_args.fsdp}")

#     # Create symbolic regression config
#     sr_config = SymbolicRegressionConfig()
    
#     # 🔧 Key fix: model loading strategy for distributed environments
#     rank0_print("📋 Distributed-safe model loading...")
    
#     # 1. First load the base model config without weights
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
    
#     # 2. First create the symbolic regression model architecture without weights
#     model = SymbolicRegressionQwenModel(config, sr_config)
    
#     # 3. 🔧 Key: distributed-safe weight loading
#     rank0_print("📋 Distributed-safe weight loading...")
    
#     # Detect whether running in a distributed environment
#     is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    
#     if is_distributed:
#         # Distributed environment: load weights in the main process, then broadcast
#         if torch.distributed.get_rank() == 0:
#             rank0_print("🔧 Main process loading pretrained weights...")
#             # Main process loads weights without using device_map
#             base_model = base_model_class.from_pretrained(
#                 model_args.model_name_or_path,
#                 cache_dir=training_args.cache_dir,
#                 torch_dtype=torch.bfloat16,
#                 # 🔧 Key: do not use device_map in distributed environments
#                 device_map=None,  
#                 low_cpu_mem_usage=True
#             )
            
#             # Copy weights to the symbolic regression model
#             model.model.load_state_dict(base_model.model.state_dict(), strict=False)
#             model.lm_head.load_state_dict(base_model.lm_head.state_dict())
            
#             # Release base model
#             del base_model
#             torch.cuda.empty_cache()
#             rank0_print("✅ Main process weight loading completed")
        
#         # Synchronize all processes
#         torch.distributed.barrier()
#     else:
#         # Single-GPU environment: load normally
#         rank0_print("🔧 Single-GPU environment, loading weights normally...")
#         base_model = base_model_class.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             attn_implementation=attn_implementation,
#             torch_dtype=torch.bfloat16,
#             device_map="auto"  # device_map can be used on single GPU
#         )
        
#         # Copy weights
#         model.model.load_state_dict(base_model.model.state_dict(), strict=False)
#         model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        
#         # Release base model
#         del base_model
#         torch.cuda.empty_cache()
#         rank0_print("✅ Single-GPU weight loading completed")

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

#     # Create tokenizer
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length,
#         padding_side="right",
#         use_fast=False,
#     )

#     # Set model trainable parameters
#     set_symbolic_regression_model(model_args, model)

#     # Print parameter information
#     is_main_process = True
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         is_main_process = torch.distributed.get_rank() == 0
    
#     if is_main_process:
#         if hasattr(model, 'visual') and hasattr(model.visual, 'print_trainable_parameters'):
#             model.visual.print_trainable_parameters()
#         if hasattr(model.model, 'print_trainable_parameters'):
#             model.model.print_trainable_parameters()
        
#         # Count total parameters
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         rank0_print(f"📊 Model parameter statistics:")
#         rank0_print(f"   Total parameters: {total_params:,}")
#         rank0_print(f"   Trainable parameters: {trainable_params:,}")
#         rank0_print(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
#         # Show memory usage
#         if torch.cuda.is_available():
#             memory_allocated = torch.cuda.memory_allocated() / 1024**3
#             memory_reserved = torch.cuda.memory_reserved() / 1024**3
#             rank0_print(f"📊 Current GPU memory usage:")
#             rank0_print(f"   Allocated: {memory_allocated:.2f} GB")
#             rank0_print(f"   Reserved: {memory_reserved:.2f} GB")

#     # Create data module
#     rank0_print("📊 Create symbolic regression data module...")
#     data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)
    
#     # Create trainer
#     rank0_print("🔧 Create trainer...")
#     trainer = Trainer(
#         model=model, 
#         processing_class=tokenizer, 
#         args=training_args, 
#         **data_module
#     )

#     # Check for checkpoints
#     if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#         rank0_print("🔄 Checkpoint found; resuming training...")
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         rank0_print("🎯 Start training...")
#         trainer.train()
    
#     # Save model
#     rank0_print("💾 Save model...")
#     trainer.save_state()
    
#     # Save symbolic regression config
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
    
#     rank0_print(f"Symbolic regression config saved to: {sr_config_path}")

#     model.config.use_cache = True
#     safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
#     rank0_print("✅ Symbolic regression model training completed!")

# if __name__ == "__main__":
#     train_symbolic_regression_distributed() 



#### Version2 ###

#!/usr/bin/env python3
"""
Fixed symbolic regression training script - for distributed training
Integrates LoRA support based on the original version for parameter-efficient fine-tuning
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

# --- Project path setup ---
project_root = Path(__file__).parent
sys.path.append(str(project_root / "qwen-vl-finetune"))

# --- Import custom modules ---
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


# --- Core change: import PEFT/LoRA related libraries ---
from peft import LoraConfig, get_peft_model

# Global variable for printing information
local_rank = None

def rank0_print(*args):
    """Print information only on the main process."""
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
        "teacher-forcing loss before saving at end of training",
        max_samples=max_samples,
        restore_train=True,
    )


def print_reload_weight_diagnostics(reference_state_dict, reloaded_model, tokenizer, loading_info=None):
    if not _is_rank0():
        return

    print("\n" + "=" * 60)
    print("🔎 final_model reloaded weight diagnostics")
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
        print("   reference state_dict not provided; skipping per-weight max_abs_diff comparison.")

    input_embeddings = reloaded_model.get_input_embeddings().weight
    lm_head = reloaded_model.lm_head.weight
    tied = input_embeddings.data_ptr() == lm_head.data_ptr()
    print(f"   lm_head shares storage with input_embeddings: {tied}")

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
        "teacher-forcing loss immediately after saving and reloading final_model",
        restore_train=False,
    )
    del reloaded_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Fix: remove the problematic safe_save_model_for_hf_trainer function ---
# We will use trainer.save_model() directly

def initialize_lm_head_from_embeddings(model):
    if not hasattr(model, "lm_head"):
        return
    input_embeddings = model.get_input_embeddings().weight
    lm_head = model.lm_head.weight
    if input_embeddings.shape != lm_head.shape:
        rank0_print(f"⚠️ Skipping lm_head initialization: embedding shape {tuple(input_embeddings.shape)} != lm_head shape {tuple(lm_head.shape)}")
        return
    with torch.no_grad():
        lm_head.copy_(input_embeddings)
    rank0_print("✅ Initialized independent lm_head from input embeddings.")


def train_symbolic_regression_distributed(attn_implementation="flash_attention_2"):
    """Main symbolic regression distributed training function with LoRA integration"""
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

    # Create symbolic regression config
    sr_config = SymbolicRegressionConfig()

    # FSDP requires all ranks to have consistent parameter initialization before wrapping; this ensures custom SetTransformer/projector initialization is consistent.
    transformers.set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    # --- Keep the original model loading logic ---
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

    # First create the symbolic regression model architecture without weights
    model = SymbolicRegressionQwenModel(config, sr_config)
    
    # Distributed-safe weight loading
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
        # Single-GPU environment
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

    # --- Core change: apply LoRA or perform full fine-tuning based on arguments ---
    if model_args.lora_enable:
        rank0_print("🚀 Enabling LoRA for parameter-efficient fine-tuning...")
        
        # First freeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze the parts that need training as needed (vision tower and MLP projector)
        if model_args.tune_mm_vision:
            model.visual.requires_grad_(True)
            rank0_print("   - Unfreezing Vision Tower (Set Transformer) for training.")
        if model_args.tune_mm_mlp:
            model.feature_projector.requires_grad_(True)
            rank0_print("   - Unfreezing MLP Projector for training.")

        # Create LoRA config
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(','),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        rank0_print("✅ LoRA has been successfully applied to the model.")
        model.print_trainable_parameters() # Print trainable parameter information
    else:
        # If LoRA is not used, set parameters according to the old logic
        rank0_print("🔧 Performing full or partial fine-tuning (LoRA is disabled).")
        model.model.requires_grad_(model_args.tune_mm_llm)
        if hasattr(model, 'lm_head'):
            model.lm_head.requires_grad_(model_args.tune_mm_llm)
            rank0_print(f"   - Setting LM head trainable: {model_args.tune_mm_llm}")
        model.visual.requires_grad_(model_args.tune_mm_vision)
        if hasattr(model, 'feature_projector'):
            model.feature_projector.requires_grad_(model_args.tune_mm_mlp)

    # Create tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # # =================================================================
    # # ⬇️  Key: move the validation code block here！ ⬇️
    # # =================================================================
    # rank0_print("🕵️  Verifying model state BEFORE training starts...")

    # # Note: when using FSDP or DDP, the model is wrapped, so access .module to get the original model
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
    #         # sys.exit(1) # Can be uncommented to terminate the program directly when an error is found
    #     else:
    #         rank0_print("✅ Vocabulary and model dimensions match. Starting training...")

    # except AttributeError as e:
    #     rank0_print(f"🚨 FATAL ERROR: Could not access a required model attribute. The model might be incomplete. Error: {e}")

    # # =================================================================

    # Create data module
    rank0_print("📊 Creating symbolic regression data module...")
    data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)
    # Create trainer
    rank0_print("🔧 Creating Trainer...")
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        **data_module
    )

    # # =================================================================
    # # ⬇️  Copy and paste the entire "final validation" code block below here ⬇️
    # # =================================================================

    # # --- Final validation: check tokenization of the first training batch ---
    # # Run this validation only on the main process (rank 0)
    # if training_args.local_rank == 0:
    #     print("\n" + "="*60)
    #     print("🕵️  Final validation: checking tokenization of the first real training batch...")
        
    #     # 1. Get the training data loader
    #     train_dataloader = trainer.get_train_dataloader()
        
    #     # 2. Take one batch of data from it
    #     first_batch = next(iter(train_dataloader))
        
    #     # 3. Select the first sample in the batch for inspection
    #     sample_input_ids = first_batch['input_ids'][0]
    #     # --- New: print the token ID list directly ---
    #     print("\n--- Token ID list of the first sample (before feeding into the model) ---")
    #     # This is the final input received by the model embedding layer
    #     print(sample_input_ids.tolist())
    #     # --- End new section ---
    #     # 4. Use tokenizer.convert_ids_to_tokens to view the raw tokenization result
    #     # This is the most precise validation method
    #     token_list = tokenizer.convert_ids_to_tokens(sample_input_ids)
        
    #     print(f"\n--- Raw token list of the first sample (partial display) ---")
    #     # To avoid flooding the screen, only part is shown
    #     print(token_list[:100]) 
    #     print("...")

    #     # 5. Automatically check the math symbols we care about
    #     math_tokens_to_check = ["<|math_add|>", "<|math_log|>", "<|math_x1|>"]
    #     found_all_as_single = True
        
    #     for special_token in math_tokens_to_check:
    #         if special_token in token_list:
    #             print(f"  ✅ Validation succeeded: found the complete '{special_token}'")
    #         else:
    #             # Check whether it was incorrectly split
    #             # For example, check 'math' and '_' exists
    #             if 'math' in token_list and '_' in token_list:
    #                 print(f"  ❌ Validation failed: did not find the complete '{special_token}'，but it may have been split！")
    #                 found_all_as_single = False

    #     print("\n--- Validation summary ---")
    #     if found_all_as_single:
    #         print("🎉 Final confirmation: tokenization is correct before data enters the model！")
    #     else:
    #         print("🚨 Final warning: tokenization is incorrect before data enters the model. Check your tokenizer creation and saving process.")
        
    #     print("="*60 + "\n")


    # Check for checkpoints and resume training
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    resume_from_checkpoint = None
    if last_checkpoint is not None:
        rank0_print(f"🔄 Checkpoint found at {last_checkpoint}, resuming training.")
        resume_from_checkpoint = last_checkpoint
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    evaluate_training_loss_before_save(trainer, data_module)

    # Save final model
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




