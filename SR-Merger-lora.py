#!/usr/bin/env python3
"""
将训练好的LoRA适配器权重合并到基础模型中，并保存为一个完整的、可直接用于推理的新模型。
"""

import torch
import sys
import os
from pathlib import Path
import argparse

# --- 关键修复：将自定义代码路径添加到系统路径 ---
# 这确保脚本可以找到 'qwenvl' 模块。
# 假设此脚本位于项目根目录 (Qwen2.5-VL-main/)。
try:
    project_root = Path(__file__).resolve().parent
    qwen_finetune_path = project_root / "qwen-vl-finetune"
    if str(qwen_finetune_path) not in sys.path:
        sys.path.append(str(qwen_finetune_path))
        print(f"✅ Added '{str(qwen_finetune_path)}' to system path.")
except NameError:
    # 为交互式环境（如Jupyter）提供备用方案
    qwen_finetune_path = Path.cwd() / "qwen-vl-finetune"
    if str(qwen_finetune_path) not in sys.path:
        sys.path.append(str(qwen_finetune_path))
        print(f"✅ Added '{str(qwen_finetune_path)}' to system path (fallback).")

# 现在可以安全地导入自定义模块和PEFT
from qwenvl.symbolic_regression.model import SymbolicRegressionQwenModel
from qwenvl.symbolic_regression.data_processor import SymbolicRegressionConfig
from peft import PeftModel
from transformers import AutoTokenizer

def merge_lora_model():
    """
    主函数，执行模型合并流程。
    """
    parser = argparse.ArgumentParser(description="LoRA Model Merging Script")
    parser.add_argument("--base_model_path", type=str, required=True, help="指向基础模型（用于训练的那个）的路径。")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="指向训练好的LoRA适配器检查点（checkpoint）的路径。")
    parser.add_argument("--output_path", type=str, required=True, help="用于保存合并后完整模型的路径。")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="加载模型时使用的数据类型 (例如, bfloat16, float16, float32)。")
    args = parser.parse_args()

    # --- 1. 加载基础模型和Tokenizer ---
    print(f"📦 Loading base model from: {args.base_model_path}")
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    load_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    # 使用固定的配置加载模型
    sr_config = SymbolicRegressionConfig()
    base_model = SymbolicRegressionQwenModel.from_pretrained(
        args.base_model_path,
        sr_config=sr_config,
        torch_dtype=load_dtype,
        device_map="auto" # 自动将模型加载到可用设备
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    print("✅ Base model and tokenizer loaded.")

    # --- 2. 加载LoRA适配器并应用到基础模型 ---
    print(f"🚀 Loading LoRA adapter from: {args.lora_adapter_path}")
    # PeftModel会自动处理将适配器应用到基础模型上
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    print("✅ LoRA adapter loaded.")

    # --- 3. 合并权重 ---
    print("🔄 Merging LoRA weights into the base model...")
    # .merge_and_unload() 会返回一个去除了PEFT包装的、权重已合并的普通模型
    model = model.merge_and_unload()
    print("✅ Weights merged successfully.")

    # --- 4. 保存合并后的完整模型 ---
    os.makedirs(args.output_path, exist_ok=True)
    print(f"💾 Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"🎉 Merged model saved. You can now use '{args.output_path}' for inference as a standard model.")

if __name__ == "__main__":
    merge_lora_model()


# “““
# ### 如何使用这个脚本

# 1.  **保存脚本**：将上面的代码保存为一个新的 Python 文件，例如 `merge_lora_model.py`，并将其放置在您的项目根目录 `Qwen2.5-VL-main/`下。

# 2.  **运行命令**：打开终端，进入 `Qwen2.5-VL-main/` 目录，然后执行以下命令：

#     ```bash
    # python SR-Merger-lora.py \
    #     --base_model_path /oceanfs/liyanjie/Qwen2.5_vl_SR_all_cp/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct \
    #     --lora_adapter_path /oceanfs/liyanjie/Qwen2.5_vl_SR_all_cp/Qwen2.5-VL-main/checkpoints/symbolic-regression-qwen-multi-gpu-LEXICAL-2M-lora/checkpoint-19000 \
    #     --output_path /oceanfs/liyanjie/Qwen2.5_vl_SR_all_cp/Qwen2.5-VL-main/checkpoints/symbolic-regression-qwen-multi-gpu-2M-lora-merger
#     ```

#     **请替换以下路径**：
#     * `--base_model_path`: 您在训练时使用的**基础模型**的路径（即包含 `config.json` 和扩展后词汇表的那个）。
#     * `--lora_adapter_path`: 您训练输出的 **LoRA 检查点**的路径（即包含 `adapter_model.bin` 的那个目录，例如 `./checkpoints/symbolic-regression-qwen-multi-gpu-1000/checkpoint-800`）。
#     * `--output_path`: 您想用来存放**最终合并好的完整模型**的新目录路径。

# 这个脚本会为您生成一个全新的、可以直接用于推理的、完整的

# ”””