#!/bin/bash

python SR-Merger-lora.py \
        --base_model_path /oceanfs/liyanjie/Qwen2.5_vl_SR_all_cp/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct \
        --lora_adapter_path /oceanfs/liyanjie/Qwen2.5_vl_SR_all_cp/Qwen2.5-VL-main/checkpoints/symbolic-regression-qwen-multi-gpu-lexical-point200-lora/checkpoint-360 \
        --output_path /oceanfs/liyanjie/Qwen2.5_vl_SR_all_cp/Qwen2.5-VL-main/checkpoints/symbolic-regression-qwen-multi-gpu-lexical-point-200-lora-merger