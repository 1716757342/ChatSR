#!/bin/bash
# Symbolic regression training launch script - fixes distributed training issues
# Optimized version that resolves DTensor mixing errors

# ======================
# Path configuration
# ======================
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="./checkpoints/symbolic-regression-qwen"
CACHE_DIR="./cache"
DATASETS="SYMBOLIC_REGRESSION%100"

# ======================
# Option 1: Try single-GPU training first (recommended)
# ======================
echo "🚀 Starting symbolic regression model training (single-GPU version, avoids distributed issues)..."
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

# Single-GPU configuration
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /path/to/ChatSR/train_symbolic_regression_fixed.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision True \
         --tune_mm_mlp True \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 True \
         --fp16 False \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 8 \
         --learning_rate 2e-5 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 2e-6 \
         --optim adamw_torch \
         --model_max_length 512 \
         --data_flatten False \
         --data_packing False \
         --num_train_epochs 50 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 1 \
         --save_steps 250 \
         --save_total_limit 5 \
         --eval_strategy "no" \
         --save_strategy "steps" \
         --load_best_model_at_end False \
         --metric_for_best_model "loss" \
         --greater_is_better False \
         --gradient_checkpointing True \
         --dataloader_num_workers 4 \
         --remove_unused_columns False \
         --report_to "none" \
         --save_safetensors True \
         --run_name "symbolic_regression_single_gpu_$(date +%Y%m%d_%H%M%S)"

echo "✅ Symbolic regression model training completed!"
