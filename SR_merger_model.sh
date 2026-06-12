#!/bin/bash

python SR-Merger-lora.py \
        --base_model_path ./ChatSR/Qwen/Qwen2.5-VL-3B-Instruct \
        --lora_adapter_path ./ChatSR/checkpoints/... \
        --output_path ./ChatSR/checkpoints/...r
