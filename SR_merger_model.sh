#!/bin/bash

python SR-Merger-lora.py \
        --base_model_path /path/to/Qwen2.5-VL-3B-Instruct \
        --lora_adapter_path /path/to/ChatSR/checkpoints/symbolic-regression-qwen-multi-gpu-lexical-point200-lora/checkpoint-360 \
        --output_path /path/to/ChatSR/checkpoints/symbolic-regression-qwen-multi-gpu-lexical-point-200-lora-merger
