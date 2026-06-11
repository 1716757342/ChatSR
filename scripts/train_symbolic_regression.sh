#!/bin/bash
# 符号回归训练启动脚本 - 多GPU优化版本
# 基于Qwen2.5-VL官方fintuning.sh，添加DeepSpeed支持

# ======================
# 路径配置
# ======================
MODEL_PATH="/oceanfs/liyanjie/Qwen2.5_vl/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct"  # 预训练模型路径
OUTPUT_DIR="./checkpoints/symbolic-regression-qwen"          # 输出目录
CACHE_DIR="./cache"                                          # 缓存目录
DATASETS="SYMBOLIC_REGRESSION%100"                           # 数据集配置

# ======================
# 分布式训练配置 - 多GPU支持
# ======================
export CUDA_VISIBLE_DEVICES="0,1,2,3"                       # 使用4个GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO                                          # NCCL调试信息（可选）
# export NCCL_SOCKET_IFNAME=bond0                               # 网络接口（注释掉，让NCCL自动选择）
export OMP_NUM_THREADS=4                                        # OpenMP线程数限制
NPROC_PER_NODE=4                                                # 4GPU并行训练
MASTER_ADDR="10.3.2.4"                                      # 主节点 IP
MASTER_PORT=$(shuf -i 20000-29999 -n 1)                     # 随机端口，避免冲突

echo "🚀 开始符号回归模型训练 (4GPU分布式版本 + DeepSpeed)..."
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $NPROC_PER_NODE"
echo "主节点: $MASTER_ADDR:$MASTER_PORT"

# ======================
# 启动训练 - 添加DeepSpeed配置
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         /oceanfs/liyanjie/Qwen2.5_vl_SR_all/Qwen2.5-VL-main/train_symbolic_regression_fixed.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision True \
         --tune_mm_mlp True \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 True \
         --fp16 False \
         --deepspeed /oceanfs/liyanjie/Qwen2.5_vl_SR_all/Qwen2.5-VL-main/qwen-vl-finetune/scripts/zero2.json \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 2 \
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
         --dataloader_num_workers 8 \
         --remove_unused_columns False \
         --report_to "none" \
         --ddp_find_unused_parameters False \
         --save_safetensors False \
         --run_name "symbolic_regression_$(date +%Y%m%d_%H%M%S)"

echo "✅ 符号回归模型训练完成!" 