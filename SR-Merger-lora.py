#!/usr/bin/env python3
"""
Merge trained LoRA adapter weights into the base model and save a complete new model ready for inference.
"""

import torch
import sys
import os
from pathlib import Path
import argparse

# --- Key fix: add the custom code path to the system path ---
# This ensures the script can find the 'qwenvl' module.
# Assume this script is located in the project root directory (Qwen2.5-VL-main/).
try:
    project_root = Path(__file__).resolve().parent
    qwen_finetune_path = project_root / "qwen-vl-finetune"
    if str(qwen_finetune_path) not in sys.path:
        sys.path.append(str(qwen_finetune_path))
        print(f"✅ Added '{str(qwen_finetune_path)}' to system path.")
except NameError:
    # Provide a fallback for interactive environments such as Jupyter
    qwen_finetune_path = Path.cwd() / "qwen-vl-finetune"
    if str(qwen_finetune_path) not in sys.path:
        sys.path.append(str(qwen_finetune_path))
        print(f"✅ Added '{str(qwen_finetune_path)}' to system path (fallback).")

# Custom modules and PEFT can now be imported safely
from qwenvl.symbolic_regression.model import SymbolicRegressionQwenModel
from qwenvl.symbolic_regression.data_processor import SymbolicRegressionConfig
from peft import PeftModel
from transformers import AutoTokenizer

def merge_lora_model():
    """
    Main function that runs the model merging workflow.
    """
    parser = argparse.ArgumentParser(description="LoRA Model Merging Script")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model used for training.")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the trained LoRA adapter checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path for saving the merged complete model.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="Data type to use when loading the model (for example, bfloat16, float16, float32).")
    args = parser.parse_args()

    # --- 1. Load the base model and tokenizer ---
    print(f"📦 Loading base model from: {args.base_model_path}")
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    load_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    # Load the model with a fixed configuration
    sr_config = SymbolicRegressionConfig()
    base_model = SymbolicRegressionQwenModel.from_pretrained(
        args.base_model_path,
        sr_config=sr_config,
        torch_dtype=load_dtype,
        device_map="auto" # Automatically load the model onto available devices
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    print("✅ Base model and tokenizer loaded.")

    # --- 2. Load the LoRA adapter and apply it to the base model ---
    print(f"🚀 Loading LoRA adapter from: {args.lora_adapter_path}")
    # PeftModel automatically applies the adapter to the base model
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    print("✅ LoRA adapter loaded.")

    # --- 3. Merge weights ---
    print("🔄 Merging LoRA weights into the base model...")
    # .merge_and_unload() returns a regular model with PEFT wrapping removed and weights merged
    model = model.merge_and_unload()
    print("✅ Weights merged successfully.")

    # --- 4. Save the merged complete model ---
    os.makedirs(args.output_path, exist_ok=True)
    print(f"💾 Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"🎉 Merged model saved. You can now use '{args.output_path}' for inference as a standard model.")

if __name__ == "__main__":
    merge_lora_model()


# “““
# ### How to use this script

# 1.  **Save the script**：Save the code above as a new Python file, such as `merge_lora_model.py`, and place it under your project root `Qwen2.5-VL-main/`.

# 2.  **Run the command**：Open a terminal, enter the `Qwen2.5-VL-main/` directory, and run the following command:

#     ```bash
    # python SR-Merger-lora.py \
    #     --base_model_path /path/to/Qwen2.5-VL-3B-Instruct \
    #     --lora_adapter_path /path/to/ChatSR/checkpoints/symbolic-regression-qwen-multi-gpu-LEXICAL-2M-lora/checkpoint-19000 \
    #     --output_path /path/to/ChatSR/checkpoints/symbolic-regression-qwen-multi-gpu-2M-lora-merger
#     ```

#     **Replace the following paths**：
#     * `--base_model_path`: Path to the **base model** used during training (the one containing `config.json` and the expanded vocabulary).
#     * `--lora_adapter_path`: Path to the **LoRA checkpoint** produced by training (the directory containing `adapter_model.bin`, for example `./checkpoints/symbolic-regression-qwen-multi-gpu-1000/checkpoint-800`).
#     * `--output_path`: New directory path where you want to store the **final merged complete model**.

# This script generates a brand-new, complete model that can be used directly for inference

# ”””