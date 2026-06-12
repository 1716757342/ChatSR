# Symbolic Regression Multi-GPU Training Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. DeepSpeed Configuration File Error

**Error message:**
```
ValueError: Expected a string path to an existing deepspeed config
```

**Solution:**
```bash
# Make sure to run the script from the correct directory
cd /path/to/ChatSR

# Check whether the configuration file exists
ls -la qwen-vl-finetune/scripts/zero3.json

# If the file does not exist, copy the existing configuration
cp qwen-vl-finetune/scripts/zero3.json ./zero3.json
```

### 2. Out-of-Memory (OOM) Issues

**Option A: Reduce the batch size**
```bash
# Modify the parameters in the training script
--per_device_train_batch_size 1    # Reduce from 2 to 1
--gradient_accumulation_steps 8    # Increase from 4 to 8
```

**Option B: Use memory offloading**
```bash
# Use the zero3_offload.json configuration
--deepspeed ./qwen-vl-finetune/scripts/zero3_offload.json
```

**Option C: Reduce the sequence length**
```bash
# Modify the model maximum length
--model_max_length 256    # Reduce from 512 to 256
```

### 3. NCCL Communication Error

**Error message:**
```
NCCL error: unhandled cuda error
```

**Solution:**
```bash
# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Adjust according to the actual network interface
export NCCL_IB_DISABLE=1        # Disable InfiniBand

# Or use single-node multi-GPU mode
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
```

### 4. Model Loading Failed

**Error message:**
```
OSError: [Errno 2] No such file or directory
```

**Solution:**
```bash
# Check the model path
ls -la /path/to/Qwen2.5-VL-3B-Instruct/

# If the model does not exist, download it again
python Model_download_HF.py
```

### 5. CUDA Version Compatibility Issues

**Check the CUDA version:**
```bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

**Reinstall PyTorch (if needed):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 Complete Run Steps

1. **Environment validation**
```bash
cd /path/to/ChatSR
bash scripts/validate_setup.sh
```

2. **Start training**
```bash
bash scripts/train_symbolic_regression.sh
```

3. **Monitor training**
```bash
# View GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f checkpoints/symbolic-regression-qwen/training.log
```

## 🚀 Performance Optimization Suggestions

### Memory Optimization
- Use DeepSpeed ZeRO-3 offloading
- Enable gradient checkpointing
- Reduce data loading worker processes

### Speed Optimization
- Use flash attention 2
- Enable bf16 mixed precision
- Optimize data preprocessing

### Stability Optimization
- Set an appropriate learning rate
- Use a warmup strategy
- Save checkpoints regularly

## 📊 Memory Usage Reference

| Configuration | GPU Memory Requirement | Recommended Settings |
|------|-------------|----------|
| Single GPU | 24GB+ | batch_size=1, grad_accum=4 |
| 2 GPUs | 16GB+ | batch_size=2, grad_accum=4 |
| 4 GPUs | 12GB+ | batch_size=2, grad_accum=4 |
| 8 GPUs | 8GB+ | batch_size=1, grad_accum=4 |

## 🛠️ Debugging Commands

```bash
# Check process status
ps aux | grep python

# Check port usage
netstat -tlnp | grep 29500

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Force-stop the training process
pkill -f "train_symbolic_regression"
```

## 📞 Getting Help

If the issue is still unresolved, provide the following information:
1. Complete error log
2. GPU model and memory size
3. CUDA and PyTorch versions
4. Training parameter configuration in use
