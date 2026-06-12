#!/bin/bash
# Symbolic regression training environment validation script

echo "🔍 Validating symbolic regression training environment configuration..."
echo "=" * 50

# Check CUDA devices
echo "📱 Checking CUDA devices..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --list-gpus
    echo "✅ CUDA device check completed"
else
    echo "❌ nvidia-smi not found; please check the CUDA installation"
    exit 1
fi

# Check Python environment
echo "🐍 Checking Python environment..."
python --version
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Check DeepSpeed
echo "🚀 Checking DeepSpeed..."
if python -c "import deepspeed" 2>/dev/null; then
    echo "✅ DeepSpeed installed: $(python -c 'import deepspeed; print(deepspeed.__version__)')"
else
    echo "❌ DeepSpeed is not installed; please run: pip install deepspeed"
    exit 1
fi

# Check required files
echo "📁 Checking required files..."

# Check model path
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Model path exists: $MODEL_PATH"
else
    echo "❌ Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Check DeepSpeed configuration file
DEEPSPEED_CONFIG="./qwen-vl-finetune/scripts/zero3.json"
if [ -f "$DEEPSPEED_CONFIG" ]; then
    echo "✅ DeepSpeed configuration file exists: $DEEPSPEED_CONFIG"
    # Validate JSON format
    if python -c "import json; json.load(open('$DEEPSPEED_CONFIG'))" 2>/dev/null; then
        echo "✅ DeepSpeed configuration JSON format is correct"
    else
        echo "❌ DeepSpeed configuration JSON format is invalid"
        exit 1
    fi
else
    echo "❌ DeepSpeed configuration file does not exist: $DEEPSPEED_CONFIG"
    exit 1
fi

# Check training script
TRAIN_SCRIPT="./qwen-vl-finetune/qwenvl/train/train_symbolic_regression.py"
if [ -f "$TRAIN_SCRIPT" ]; then
    echo "✅ Training script exists: $TRAIN_SCRIPT"
else
    echo "❌ Training script does not exist: $TRAIN_SCRIPT"
    exit 1
fi

# Check memory usage
echo "💾 Checking GPU memory..."
python << EOF
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        print(f"GPU {i}: {props.name}, total memory: {total_memory:.1f} GB")

        # Check current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
else:
    print("❌ CUDA is unavailable")
EOF

# Test torchrun
echo "🚀 Testing torchrun..."
if command -v torchrun &> /dev/null; then
    echo "✅ torchrun is available"
    # Simple test
    torchrun --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ torchrun is working normally"
    else
        echo "❌ torchrun test failed"
        exit 1
    fi
else
    echo "❌ torchrun not found"
    exit 1
fi

# Environment variable checks
echo "🌍 Checking environment variables..."
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-not set}"

echo "=" * 50
echo "🎉 All checks passed! The environment configuration is correct."
echo "You can now run the training script:"
echo "bash scripts/train_symbolic_regression.sh"
echo "=" * 50
