#!/bin/bash
# 符号回归训练启动脚本 - 修复分布式训练问题
# 解决DTensor混合错误的优化版本

# ======================
# 路径配置
# ======================
MODEL_PATH="/oceanfs/liyanjie/Qwen2.5_vl/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="./checkpoints/symbolic-regression-qwen"
CACHE_DIR="./cache"
DATASETS="SYMBOLIC_REGRESSION%100"

# ======================
# 方案1: 先尝试单GPU训练（推荐）
# ======================
echo "🚀 开始符号回归模型训练 (单GPU版本，避免分布式问题)..."
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"

# 单GPU配置
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /oceanfs/liyanjie/Qwen2.5_vl_SR_all/Qwen2.5-VL-main/train_symbolic_regression_fixed.py \
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

echo "✅ 符号回归模型训练完成!" 