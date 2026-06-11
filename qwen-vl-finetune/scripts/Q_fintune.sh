#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# # ======================
# # Distributed Configuration
# # ======================
# MASTER_ADDR="10.3.2.7"                     # [Required] Master node IP for multi-GPU training
# MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="/home/liyanjie/Qwen2.5_vl/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints/Qwen-3b-Ex"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models
DATASETS="DEMO1%100"                  # [DataArguments] Dataset with sampling rate

# ======================
# 分布式训练配置
# ======================
# 指定使用 GPU 0, 1, 2, 3, 4, 5。这会告诉 CUDA 程序只有这几张卡可见，
# 并且它们会被 torchrun 重新编号为 0, 1, 2, 3, 4, 5
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NPROC_PER_NODE=4 # 进程数必须与 CUDA_VISIBLE_DEVICES 中的 GPU 数量一致

MASTER_ADDR="10.3.2.7"                     # [必需] 多卡训练的主节点 IP
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # 随机端口，避免冲突

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         /home/liyanjie/Qwen2.5_vl/Qwen2.5-VL-main/qwen-vl-finetune/qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --data_packing True \
         --max_pixels 451584 \
         --min_pixels 12544 \
         --base_interval 2 \
         --video_max_frames 8 \
         --video_min_frames 4 \
         --video_max_frame_pixels 1304576 \
         --video_min_frame_pixels 200704 \
         --num_train_epochs 3 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 500 \
         --save_total_limit 3 \
         --deepspeed ./zero3.json \