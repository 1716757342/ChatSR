# 符号回归多GPU训练故障排除指南

## 🚨 常见问题及解决方案

### 1. DeepSpeed配置文件错误

**错误信息：**
```
ValueError: Expected a string path to an existing deepspeed config
```

**解决方案：**
```bash
# 确保在正确的目录下运行脚本
cd /oceanfs/liyanjie/Qwen2.5_vl_SR_all/Qwen2.5-VL-main

# 检查配置文件是否存在
ls -la qwen-vl-finetune/scripts/zero3.json

# 如果文件不存在，复制现有配置
cp qwen-vl-finetune/scripts/zero3.json ./zero3.json
```

### 2. 内存不足 (OOM) 问题

**方案A: 减少批处理大小**
```bash
# 修改训练脚本中的参数
--per_device_train_batch_size 1    # 从2减到1
--gradient_accumulation_steps 8    # 从4增到8
```

**方案B: 使用内存卸载**
```bash
# 使用zero3_offload.json配置
--deepspeed ./qwen-vl-finetune/scripts/zero3_offload.json
```

**方案C: 减少序列长度**
```bash
# 修改模型最大长度
--model_max_length 256    # 从512减到256
```

### 3. NCCL通信错误

**错误信息：**
```
NCCL error: unhandled cuda error
```

**解决方案：**
```bash
# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口调整
export NCCL_IB_DISABLE=1        # 禁用InfiniBand

# 或者使用单机多卡模式
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
```

### 4. 模型加载失败

**错误信息：**
```
OSError: [Errno 2] No such file or directory
```

**解决方案：**
```bash
# 检查模型路径
ls -la /oceanfs/liyanjie/Qwen2.5_vl/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct/

# 如果模型不存在，重新下载
python Model_download_HF.py
```

### 5. CUDA版本兼容性问题

**检查CUDA版本：**
```bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

**重新安装PyTorch（如需要）：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 完整的运行步骤

1. **环境验证**
```bash
cd /oceanfs/liyanjie/Qwen2.5_vl_SR_all/Qwen2.5-VL-main
bash scripts/validate_setup.sh
```

2. **启动训练**
```bash
bash scripts/train_symbolic_regression.sh
```

3. **监控训练**
```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f checkpoints/symbolic-regression-qwen/training.log
```

## 🚀 性能优化建议

### 内存优化
- 使用DeepSpeed ZeRO-3卸载
- 启用梯度检查点
- 减少数据加载工作进程

### 速度优化
- 使用flash attention 2
- 启用bf16混合精度
- 优化数据预处理

### 稳定性优化
- 设置合适的学习率
- 使用warmup策略
- 定期保存检查点

## 📊 内存使用参考

| 配置 | GPU内存需求 | 推荐设置 |
|------|-------------|----------|
| 单GPU | 24GB+ | batch_size=1, grad_accum=4 |
| 2GPU | 16GB+ | batch_size=2, grad_accum=4 |
| 4GPU | 12GB+ | batch_size=2, grad_accum=4 |
| 8GPU | 8GB+ | batch_size=1, grad_accum=4 |

## 🛠️ 调试命令

```bash
# 检查进程状态
ps aux | grep python

# 检查端口占用
netstat -tlnp | grep 29500

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache()"

# 强制结束训练进程
pkill -f "train_symbolic_regression"
```

## 📞 获取帮助

如果问题仍未解决，请提供以下信息：
1. 完整的错误日志
2. GPU型号和内存大小
3. CUDA和PyTorch版本
4. 使用的训练参数配置 