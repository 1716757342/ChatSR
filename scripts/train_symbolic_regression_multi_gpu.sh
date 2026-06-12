#!/bin/bash
# Symbolic regression training launch script - multi-GPU fixed version
# Use FSDP instead of DeepSpeed to avoid DTensor mixing errors

# ======================
# Path configuration
# ======================
MODEL_PATH="/path/to/ChatSR/Qwen/Qwen2.5-VL-3B-Instruct-expend-token"
OUTPUT_DIR="./checkpoints/symbolic-regression-qwen-multi-gpu-20"
CACHE_DIR="./cache"
DATASETS="SYMBOLIC_REGRESSION_LEXICAL_POINT_20_TEST%100"

# ======================
# Option 2: Multi-GPU training - use FSDP instead of DeepSpeed
# ======================
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
NPROC_PER_NODE=4
MASTER_ADDR="localhost"  # Changed to localhost
MASTER_PORT=$(shuf -i 20000-29999 -n 1)

echo "🚀 Starting symbolic regression model training (multi-GPU FSDP version)..."
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NPROC_PER_NODE"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Use torchrun + FSDP (without DeepSpeed)
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         /path/to/ChatSR/train_symbolic_regression_distributed_fixed.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision True \
         --tune_mm_mlp True \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 True \
         --fp16 False \
         --fsdp "full_shard auto_wrap" \
         --fsdp_transformer_layer_cls_to_wrap "Qwen2_5_VLDecoderLayer" \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 1 \
         --learning_rate 1e-5 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 512 \
         --data_flatten False \
         --data_packing False \
         --num_train_epochs 100 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 1 \
         --save_steps 200 \
         --save_total_limit 2 \
         --eval_strategy "no" \
         --save_strategy "steps" \
         --load_best_model_at_end False \
         --metric_for_best_model "loss" \
         --greater_is_better False \
         --gradient_checkpointing True \
         --dataloader_num_workers 8 \
         --remove_unused_columns False \
         --report_to "none" \
         --ddp_find_unused_parameters False \
         --save_safetensors True \
         --run_name "symbolic_regression_fsdp_$(date +%Y%m%d_%H%M%S)"

echo "✅ Symbolic regression model training completed!"
