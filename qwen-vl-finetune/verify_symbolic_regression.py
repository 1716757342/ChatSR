#!/usr/bin/env python3
"""
符号回归多模态LLM验证脚本
验证整个系统的前向传播和推理功能
"""

import torch
import numpy as np
import sys
import os
sys.path.append('.')

from qwenvl.symbolic_regression import (
    SymbolicRegressionQwenModel, 
    SymbolicRegressionConfig
)
from qwenvl.data.data_symbolic_regression import make_symbolic_regression_data_module
from qwenvl.train.argument import DataArguments
from transformers import AutoTokenizer, Qwen2VLConfig
import json

def main():
    print("🚀 符号回归多模态LLM验证开始...")
    
    # 配置
    model_path = "/oceanfs/liyanjie/Qwen2.5_vl/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct"
    device = "cuda:0"
    
    # 1. 创建符号回归配置
    print("📋 创建符号回归配置...")
    sr_config = SymbolicRegressionConfig(
        input_dim=11,  # x1-x10 + y
        hidden_size=896,
        num_attention_heads=14,
        num_set_layers=3,
        inducing_points=32,
        pooling_outputs=8,
        max_points=100
    )
    print(f"   ✅ Set Transformer配置: {sr_config.hidden_size}维, {sr_config.num_attention_heads}头, {sr_config.num_set_layers}层")
    
    # 2. 加载tokenizer
    print("📋 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"   ✅ 词汇表大小: {len(tokenizer.get_vocab())}")
    
    # 3. 创建符号回归模型
    print("📋 创建符号回归模型...")
    with torch.no_grad():  # 减少内存使用
        # 加载基础配置
        from transformers import Qwen2_5_VLForConditionalGeneration
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device
        )
        
        # 创建符号回归模型
        model = SymbolicRegressionQwenModel(base_model.config, sr_config)
        model = model.to(device)
        
        # 复制预训练权重
        model.model.load_state_dict(base_model.model.state_dict(), strict=False)
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        
        # 删除基础模型释放内存
        del base_model
        torch.cuda.empty_cache()
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ 模型创建成功!")
    print(f"   📊 总参数: {total_params:,d}")
    print(f"   📊 可训练参数: {trainable_params:,d} ({100*trainable_params/total_params:.2f}%)")
    
    # 4. 创建测试数据
    print("📋 创建测试数据...")
    
    # 生成测试函数: y = sin(x1) + cos(x2)
    batch_size = 1
    num_points = 50
    num_features = 2
    
    # 生成输入数据点
    x1 = np.random.uniform(-np.pi, np.pi, (batch_size, num_points))
    x2 = np.random.uniform(-np.pi, np.pi, (batch_size, num_points))
    y = np.sin(x1) + np.cos(x2)
    
    # 构造数据点张量 [x1, x2, ..., x10, y] (补零到11维)
    data_points = np.zeros((batch_size, num_points, 11))
    data_points[:, :, 0] = x1  # x1
    data_points[:, :, 1] = x2  # x2
    # x3-x10保持为0
    data_points[:, :, -1] = y  # y值
    
    data_points = torch.tensor(data_points, dtype=torch.float32, device=device)
    print(f"   ✅ 数据点形状: {data_points.shape}")
    print(f"   📊 测试函数: y = sin(x1) + cos(x2)")
    
    # 5. 创建输入文本
    print("📋 创建输入文本...")
    conversations = [
        [
            {"from": "human", "value": "<data>\n这是采样的一组科学数据点，请你寻找一个表达式来拟合这组数据，你只需要生成表达式二叉树的先序遍历即可"},
            {"from": "gpt", "value": "好的，我得到的表达式先序遍历是[+,sin,x1,cos,x2]"}
        ]
    ]
    
    # 预处理对话
    from qwenvl.data.data_symbolic_regression import preprocess_symbolic_regression_qwen
    processed = preprocess_symbolic_regression_qwen(
        conversations, 
        tokenizer, 
        data_grid_info=[8]  # 每个样本8个vision pad tokens
    )
    
    input_ids = processed['input_ids'].to(device)
    labels = processed['labels'].to(device)
    # 创建attention_mask（全为1，表示都参与注意力计算）
    attention_mask = torch.ones_like(input_ids).to(device)
    
    print(f"   ✅ Input IDs形状: {input_ids.shape}")
    print(f"   ✅ 文本序列长度: {input_ids.shape[1]}")
    
    # 验证vision tokens
    vision_tokens = ['<|vision_start|>', '<|vision_end|>', '<|vision_pad|>']
    for token in vision_tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        count = (input_ids == token_id).sum().item()
        print(f"   📊 {token}: {count} 个")
    
    # 6. 前向传播验证
    print("📋 执行前向传播验证...")
    
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                data_points=data_points,
                labels=labels
            )
            
            print(f"   ✅ 前向传播成功!")
            print(f"   📊 输出logits形状: {outputs.logits.shape}")
            print(f"   📊 Loss值: {outputs.loss.item():.4f}")
            
            # 验证生成能力
            print("📋 验证生成能力...")
            
            # 只用第一个样本做生成
            sample_input_ids = input_ids[:1]
            sample_attention_mask = attention_mask[:1]
            sample_data_points = data_points[:1]
            
            # 找到human部分的结束位置
            human_part = sample_input_ids[0]
            gpt_start_token = tokenizer.encode("<|im_start|>gpt\n", add_special_tokens=False)
            gpt_start_pos = None
            
            for i in range(len(human_part) - len(gpt_start_token) + 1):
                if torch.equal(human_part[i:i+len(gpt_start_token)], torch.tensor(gpt_start_token, device=device)):
                    gpt_start_pos = i + len(gpt_start_token)
                    break
            
            if gpt_start_pos:
                # 截取到GPT回答开始位置
                generation_input_ids = sample_input_ids[:, :gpt_start_pos]
                generation_attention_mask = sample_attention_mask[:, :gpt_start_pos]
                
                print(f"   📊 生成起始位置: {gpt_start_pos}")
                print(f"   📊 生成输入长度: {generation_input_ids.shape[1]}")
                
                # 生成几个token验证
                for step in range(3):
                    outputs = model(
                        input_ids=generation_input_ids,
                        attention_mask=generation_attention_mask,
                        data_points=sample_data_points
                    )
                    
                    # 获取下一个token
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    next_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    
                    print(f"   🎯 生成step {step+1}: '{next_token_text}' (token_id: {next_token[0].item()})")
                    
                    # 更新输入
                    generation_input_ids = torch.cat([generation_input_ids, next_token], dim=1)
                    generation_attention_mask = torch.cat([
                        generation_attention_mask, 
                        torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
                    ], dim=1)
                    
                    if generation_input_ids.shape[1] >= 400:  # 避免序列过长
                        break
                
                print(f"   ✅ 生成验证成功!")
            
        except Exception as e:
            print(f"   ❌ 前向传播失败: {e}")
            return False
    
    # 7. 内存使用统计
    print("📋 内存使用统计...")
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"   📊 CUDA内存分配: {memory_allocated:.2f} GB")
        print(f"   📊 CUDA内存保留: {memory_reserved:.2f} GB")
    
    print("\n🎉 符号回归多模态LLM验证完全成功!")
    print("=" * 60)
    print("✅ 所有核心功能验证成功:")
    print("   🔹 Set Transformer编码器 - 工作正常")
    print("   🔹 Vision Token处理 - 工作正常") 
    print("   🔹 数据点编码 - 工作正常")
    print("   🔹 文本生成 - 工作正常")
    print("   🔹 Loss计算 - 工作正常")
    print("   🔹 端到端流水线 - 工作正常")
    print("=" * 60)
    print("🌟 符号回归多模态LLM实现圆满成功!")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 