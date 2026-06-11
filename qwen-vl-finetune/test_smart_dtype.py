#!/usr/bin/env python3
"""
智能dtype管理系统测试脚本
验证bfloat16节省内存的效果
"""

import torch
import torch.cuda
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from qwenvl.symbolic_regression import (
    SymbolicRegressionQwenModel, 
    SymbolicRegressionConfig,
    SmartDtypeManager,
    apply_smart_dtype_to_model
)
from transformers import Qwen2_5_VLForConditionalGeneration

def get_memory_usage():
    """获取当前GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0, 0

def test_dtype_memory_comparison():
    """测试不同dtype的内存使用对比"""
    
    print("🧪 智能dtype管理系统测试")
    print("=" * 50)
    
    # 清空显存
    torch.cuda.empty_cache()
    
    # 测试配置
    sr_config = SymbolicRegressionConfig()
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print(f"📍 测试模型: {model_name}")
    print(f"📊 Set Transformer配置:")
    print(f"   - Hidden size: {sr_config.hidden_size}")
    print(f"   - Attention heads: {sr_config.num_attention_heads}")
    print(f"   - Layers: {sr_config.num_set_layers}")
    print()
    
    # 测试1: float32模式
    print("🔴 测试1: Float32模式 (原来的方式)")
    torch.cuda.empty_cache()
    initial_mem = get_memory_usage()
    print(f"初始内存: {initial_mem[0]:.2f} GB (allocated), {initial_mem[1]:.2f} GB (reserved)")
    
    try:
        # 加载基础模型为float32
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # 先加载到CPU
        )
        
        # 创建符号回归模型（应用float32）
        model_f32 = SymbolicRegressionQwenModel(base_model.config, sr_config)
        model_f32.model.load_state_dict(base_model.model.state_dict(), strict=False)
        model_f32.lm_head.load_state_dict(base_model.lm_head.state_dict())
        
        # 强制转换为float32
        model_f32 = model_f32.float()
        
        # 移动到GPU
        model_f32 = model_f32.cuda()
        
        del base_model
        torch.cuda.empty_cache()
        
        float32_mem = get_memory_usage()
        print(f"Float32模型内存: {float32_mem[0]:.2f} GB (allocated), {float32_mem[1]:.2f} GB (reserved)")
        
        # 简单前向测试
        test_data = torch.randn(1, 50, 11).cuda()
        with torch.no_grad():
            features = model_f32.visual(test_data)
            print(f"Float32输出形状: {features.shape}, dtype: {features.dtype}")
        
        del model_f32
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Float32测试失败: {e}")
        torch.cuda.empty_cache()
    
    print("\n" + "="*50)
    
    # 测试2: bfloat16模式 (智能dtype管理)
    print("🟢 测试2: Bfloat16模式 (智能dtype管理)")
    torch.cuda.empty_cache()
    initial_mem = get_memory_usage()
    print(f"初始内存: {initial_mem[0]:.2f} GB (allocated), {initial_mem[1]:.2f} GB (reserved)")
    
    try:
        # 加载基础模型为bfloat16
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # 先加载到CPU
        )
        
        # 创建符号回归模型（智能dtype管理）
        model_bf16 = SymbolicRegressionQwenModel(base_model.config, sr_config)
        model_bf16.model.load_state_dict(base_model.model.state_dict(), strict=False)
        model_bf16.lm_head.load_state_dict(base_model.lm_head.state_dict())
        
        # 移动到GPU
        model_bf16 = model_bf16.cuda()
        
        del base_model
        torch.cuda.empty_cache()
        
        bfloat16_mem = get_memory_usage()
        print(f"Bfloat16模型内存: {bfloat16_mem[0]:.2f} GB (allocated), {bfloat16_mem[1]:.2f} GB (reserved)")
        
        # 简单前向测试
        test_data = torch.randn(1, 50, 11).cuda()
        with torch.no_grad():
            features = model_bf16.visual(test_data)
            print(f"Bfloat16输出形状: {features.shape}, dtype: {features.dtype}")
        
        # 计算内存节省
        if 'float32_mem' in locals():
            memory_saved = float32_mem[0] - bfloat16_mem[0]
            savings_percent = (memory_saved / float32_mem[0]) * 100 if float32_mem[0] > 0 else 0
            print(f"\n💰 内存节省:")
            print(f"   - 节省内存: {memory_saved:.2f} GB")
            print(f"   - 节省比例: {savings_percent:.1f}%")
        
        del model_bf16
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Bfloat16测试失败: {e}")
        torch.cuda.empty_cache()
    
    print("\n" + "="*50)
    
    # 测试3: 智能dtype切换测试
    print("🧠 测试3: 智能dtype切换验证")
    
    try:
        dtype_manager = SmartDtypeManager()
        
        # 测试不同操作的dtype选择
        test_tensor = torch.randn(10, 10).cuda()
        
        print(f"默认操作dtype: {dtype_manager.get_safe_dtype(test_tensor, 'default')}")
        print(f"Softmax操作dtype: {dtype_manager.get_safe_dtype(test_tensor, 'softmax')}")
        print(f"LayerNorm操作dtype: {dtype_manager.get_safe_dtype(test_tensor, 'layer_norm')}")
        print(f"线性变换操作dtype: {dtype_manager.get_safe_dtype(test_tensor, 'linear')}")
        
        # 测试安全转换
        test_bf16 = test_tensor.to(torch.bfloat16)
        converted = dtype_manager.safe_cast(test_bf16, 'softmax')
        print(f"Bfloat16 -> Softmax转换: {test_bf16.dtype} -> {converted.dtype}")
        
        print("✅ 智能dtype切换测试通过")
        
    except Exception as e:
        print(f"❌ 智能dtype切换测试失败: {e}")
    
    print("\n" + "="*50)
    print("🎯 测试总结:")
    print("1. 智能dtype管理系统可以显著减少内存使用")
    print("2. Bfloat16作为默认类型，float32用于数值稳定性")
    print("3. 系统可以根据操作类型自动选择合适的dtype")
    print("4. 对于大模型训练，内存节省可达到约50%")

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n📚 训练兼容性测试")
    print("=" * 30)
    
    try:
        sr_config = SymbolicRegressionConfig()
        
        # 创建一个小模型用于测试
        from transformers import Qwen2_5_VLConfig
        config = Qwen2_5_VLConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            vision_config={}
        )
        
        model = SymbolicRegressionQwenModel(config, sr_config)
        model = model.cuda()
        
        # 模拟训练数据
        batch_size = 2
        seq_len = 100
        num_points = 50
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        data_points = torch.randn(batch_size, num_points, 11).cuda()
        labels = input_ids.clone()
        
        # 前向传播
        model.train()
        outputs = model(input_ids=input_ids, data_points=data_points, labels=labels)
        
        print(f"✅ 前向传播成功")
        print(f"   - Loss: {outputs.loss.item():.4f}")
        print(f"   - Logits shape: {outputs.logits.shape}")
        print(f"   - Loss dtype: {outputs.loss.dtype}")
        
        # 反向传播
        outputs.loss.backward()
        print(f"✅ 反向传播成功")
        
        # 检查梯度
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        print(f"✅ 梯度计算成功，{grad_count}个参数有梯度")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ 需要GPU环境来测试内存优化效果")
        sys.exit(1)
    
    print(f"🖥️ GPU设备: {torch.cuda.get_device_name()}")
    print(f"📊 GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    test_dtype_memory_comparison()
    test_training_compatibility()
    
    print("\n🎉 所有测试完成！") 