#!/bin/bash
# 符号回归训练环境验证脚本

echo "🔍 验证符号回归训练环境配置..."
echo "=" * 50

# 检查CUDA设备
echo "📱 检查CUDA设备..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --list-gpus
    echo "✅ CUDA设备检查完成"
else
    echo "❌ nvidia-smi未找到，请检查CUDA安装"
    exit 1
fi

# 检查Python环境
echo "🐍 检查Python环境..."
python --version
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 检查DeepSpeed
echo "🚀 检查DeepSpeed..."
if python -c "import deepspeed" 2>/dev/null; then
    echo "✅ DeepSpeed已安装: $(python -c 'import deepspeed; print(deepspeed.__version__)')"
else
    echo "❌ DeepSpeed未安装，请运行: pip install deepspeed"
    exit 1
fi

# 检查必要文件
echo "📁 检查必要文件..."

# 检查模型路径
MODEL_PATH="/oceanfs/liyanjie/Qwen2.5_vl/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct"
if [ -d "$MODEL_PATH" ]; then
    echo "✅ 模型路径存在: $MODEL_PATH"
else
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 检查DeepSpeed配置文件
DEEPSPEED_CONFIG="./qwen-vl-finetune/scripts/zero3.json"
if [ -f "$DEEPSPEED_CONFIG" ]; then
    echo "✅ DeepSpeed配置文件存在: $DEEPSPEED_CONFIG"
    # 验证JSON格式
    if python -c "import json; json.load(open('$DEEPSPEED_CONFIG'))" 2>/dev/null; then
        echo "✅ DeepSpeed配置JSON格式正确"
    else
        echo "❌ DeepSpeed配置JSON格式错误"
        exit 1
    fi
else
    echo "❌ DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
    exit 1
fi

# 检查训练脚本
TRAIN_SCRIPT="./qwen-vl-finetune/qwenvl/train/train_symbolic_regression.py"
if [ -f "$TRAIN_SCRIPT" ]; then
    echo "✅ 训练脚本存在: $TRAIN_SCRIPT"
else
    echo "❌ 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 检查内存使用
echo "💾 检查GPU内存..."
python << EOF
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        print(f"GPU {i}: {props.name}, 总内存: {total_memory:.1f} GB")
        
        # 检查当前内存使用
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  已分配: {allocated:.2f} GB, 已预留: {reserved:.2f} GB")
else:
    print("❌ CUDA不可用")
EOF

# 测试torchrun
echo "🚀 测试torchrun..."
if command -v torchrun &> /dev/null; then
    echo "✅ torchrun可用"
    # 简单测试
    torchrun --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ torchrun工作正常"
    else
        echo "❌ torchrun测试失败"
        exit 1
    fi
else
    echo "❌ torchrun未找到"
    exit 1
fi

# 环境变量检查
echo "🌍 检查环境变量..."
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-未设置}"

echo "=" * 50
echo "🎉 所有检查通过！环境配置正确。"
echo "现在可以运行训练脚本:"
echo "bash scripts/train_symbolic_regression.sh"
echo "=" * 50 