#!/usr/bin/env python3
"""
扩展模型词汇表，添加数学符号特殊token
用于符号回归任务的数学符号token化
"""

import torch
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import shutil

# 添加项目路径以导入自定义模型
project_root = Path(__file__).parent
sys.path.append(str(project_root / "qwen-vl-finetune"))

class MathTokenExtender:
    """数学符号Token扩展器"""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        
        # 数学符号映射
        self.math_tokens = {
            # 基本运算符
            "<|math_add|>": "+",
            "<|math_sub|>": "-", 
            "<|math_mul|>": "*",
            "<|math_div|>": "/",
            "<|math_pow|>": "^",
            
            # 三角函数
            "<|math_sin|>": "sin",
            "<|math_cos|>": "cos",
            "<|math_tan|>": "tan",
            
            # 其他函数
            "<|math_log|>": "log",
            "<|math_exp|>": "exp",
            "<|math_sqrt|>": "sqrt",
            "<|math_abs|>": "abs",
            
            # 变量
            "<|math_x1|>": "x1",
            "<|math_x2|>": "x2",
            "<|math_x3|>": "x3",
            "<|math_x4|>": "x4",
            "<|math_x5|>": "x5",
            "<|math_x6|>": "x6",
            "<|math_x7|>": "x7",
            "<|math_x8|>": "x8",
            "<|math_x9|>": "x9",
            "<|math_x10|>": "x10",
            
            # 常数
            "<|math_C|>": "C",
            "<|math_const_1|>": "1",
            "<|math_const_2|>": "2",
            "<|math_const_3|>": "3",
            "<|math_const_4|>": "4",
            "<|math_const_5|>": "5",
        }
        
        # 反向映射
        self.symbol_to_token = {v: k for k, v in self.math_tokens.items()}
        
    def load_model_and_tokenizer(self):
        """加载原始模型和tokenizer"""
        print(f"📦 加载原始模型: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 检查模型类型
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "")
            architectures = config.get("architectures", [])
            
            print(f"📋 模型信息:")
            print(f"   模型类型: {model_type}")
            print(f"   架构: {architectures}")
            
            # 根据模型类型选择加载方式
            if "SymbolicRegressionQwenModel" in architectures:
                print(f"🔧 检测到自定义符号回归模型，使用专用加载方式")
                try:
                    from qwenvl.symbolic_regression.model import SymbolicRegressionQwenModel
                    self.model = SymbolicRegressionQwenModel.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    print(f"✅ 使用SymbolicRegressionQwenModel加载成功")
                except Exception as e:
                    print(f"⚠️ 使用SymbolicRegressionQwenModel加载失败: {e}")
                    print(f"🔧 尝试使用AutoModel加载...")
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            elif model_type == "qwen2_5_vl" or any("Qwen2_5_VL" in arch for arch in architectures):
                print(f"🔧 检测到Qwen2.5-VL模型，使用AutoModel加载方式")
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # 标准模型
                print(f"🔧 使用标准模型加载方式")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
        else:
            # 没有config.json，尝试标准方式
            print(f"🔧 未找到config.json，使用标准模型加载方式")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        print(f"✅ 原始模型加载完成")
        print(f"   原始词汇表大小: {len(self.tokenizer.vocab)}")
        
    def check_existing_tokens(self):
        """检查已存在的数学符号token"""
        existing_tokens = []
        missing_tokens = []
        
        for token in self.math_tokens.keys():
            if token in self.tokenizer.vocab:
                existing_tokens.append(token)
            else:
                missing_tokens.append(token)
        
        print(f"📊 Token状态检查:")
        print(f"   已存在: {len(existing_tokens)} 个")
        print(f"   需要添加: {len(missing_tokens)} 个")
        
        if existing_tokens:
            print(f"   已存在的token: {existing_tokens[:5]}...")
        
        return existing_tokens, missing_tokens
    
    def add_math_tokens(self):
        """添加数学符号token到词汇表"""
        existing_tokens, missing_tokens = self.check_existing_tokens()
        
        if not missing_tokens:
            print(f"✅ 所有数学符号token已存在，无需添加")
            return
        
        print(f"➕ 添加 {len(missing_tokens)} 个数学符号token...")
        
        # 获取原始嵌入矩阵
        original_embeddings = self.model.get_input_embeddings()
        original_vocab_size = original_embeddings.weight.shape[0]
        embedding_dim = original_embeddings.weight.shape[1]
        
        # 添加新token到tokenizer
        new_tokens = [token for token in missing_tokens]
        num_added = self.tokenizer.add_tokens(new_tokens)
        
        print(f"✅ 成功添加 {num_added} 个token到词汇表")
        print(f"   新词汇表大小: {len(self.tokenizer.vocab)}")
        
        # 调整模型嵌入矩阵大小
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 初始化新token的嵌入
        self.initialize_new_embeddings(original_vocab_size, len(self.tokenizer))
        
        print(f"✅ 模型嵌入矩阵已调整")
        
    def initialize_new_embeddings(self, original_vocab_size: int, new_vocab_size: int):
        """初始化新token的嵌入向量"""
        if original_vocab_size == new_vocab_size:
            return
            
        print(f"🔧 初始化新token嵌入向量...")
        
        # 获取调整后的嵌入矩阵
        embeddings = self.model.get_input_embeddings()
        
        with torch.no_grad():
            # 对于每个新添加的数学符号token，使用相关符号的嵌入进行初始化
            for token, symbol in self.math_tokens.items():
                # 获取新token的ID
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if len(token_ids) == 1:
                    new_token_idx = token_ids[0]
                    
                    # 确保索引在有效范围内
                    if new_token_idx >= original_vocab_size and new_token_idx < new_vocab_size:
                        # 尝试找到相关符号的嵌入
                        related_embeddings = []
                        
                        # 搜索相关的token
                        for vocab_token, vocab_idx in self.tokenizer.vocab.items():
                            if vocab_idx < original_vocab_size:
                                if symbol in vocab_token.lower() or any(c in vocab_token for c in symbol):
                                    related_embeddings.append(embeddings.weight[vocab_idx])
                        
                        # 如果找到相关嵌入，使用它们的平均值
                        if related_embeddings:
                            avg_embedding = torch.stack(related_embeddings).mean(dim=0)
                            embeddings.weight[new_token_idx] = avg_embedding
                            print(f"   ✅ {token} (ID: {new_token_idx}) 使用相关嵌入初始化")
                        else:
                            # 否则使用随机初始化（小方差）
                            embeddings.weight[new_token_idx] = torch.randn(embeddings.weight.shape[1]) * 0.02
                            print(f"   🔧 {token} (ID: {new_token_idx}) 使用随机初始化")
                    else:
                        print(f"   ⚠️ {token} 索引超出范围: {new_token_idx}")
        
        print(f"✅ 新token嵌入向量初始化完成")
    
    def verify_tokens(self):
        """验证添加的token"""
        print(f"🔍 验证数学符号token...")
        
        success_count = 0
        
        for token, symbol in self.math_tokens.items():
            try:
                # 编码token
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                
                # 检查是否为单个token
                if len(token_ids) == 1:
                    # 解码验证
                    decoded = self.tokenizer.decode(token_ids[0])
                    if decoded == token:
                        success_count += 1
                        print(f"   ✅ {token} -> ID: {token_ids[0]} ({symbol})")
                    else:
                        print(f"   ❌ {token} 解码不匹配: {decoded}")
                else:
                    print(f"   ❌ {token} 被分割为多个token: {token_ids}")
                    
            except Exception as e:
                print(f"   ❌ {token} 验证失败: {e}")
        
        print(f"📊 验证结果: {success_count}/{len(self.math_tokens)} 个token成功")
        
        return success_count == len(self.math_tokens)
    
    def save_extended_model(self):
        """保存扩展后的模型"""
        print(f"💾 保存扩展后的模型到: {self.output_path}")
        
        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(self.output_path)
        
        # 保存模型
        self.model.save_pretrained(self.output_path)
        
        # 保存数学符号映射
        mapping_file = os.path.join(self.output_path, "math_token_mapping.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.math_tokens, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 模型保存完成")
        print(f"   词汇表大小: {len(self.tokenizer.vocab)}")
        print(f"   模型参数: {self.model.num_parameters():,}")
        
    def create_usage_guide(self):
        """创建使用说明"""
        guide_content = f"""# 数学符号Token扩展模型使用说明

## 模型信息
- 原始模型: {self.model_path}
- 扩展后模型: {self.output_path}
- 新增token数量: {len(self.math_tokens)}

## 数学符号映射

### 基本运算符
"""
        
        # 按类别组织token
        categories = {
            "基本运算符": ["+", "-", "*", "/", "^"],
            "三角函数": ["sin", "cos", "tan"],
            "其他函数": ["log", "exp", "sqrt", "abs"],
            "变量": [f"x{i}" for i in range(1, 11)],
            "常数": ["C", "1", "2", "3", "4", "5"]
        }
        
        for category, symbols in categories.items():
            guide_content += f"\n### {category}\n"
            for symbol in symbols:
                if symbol in self.symbol_to_token:
                    token = self.symbol_to_token[symbol]
                    guide_content += f"- {symbol} -> {token}\n"
        
        guide_content += f"""
## 使用示例

### Python代码示例
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载扩展后的模型
tokenizer = AutoTokenizer.from_pretrained('{self.output_path}')
model = AutoModelForCausalLM.from_pretrained('{self.output_path}')

# 示例：编码数学表达式
expression = ["<|math_add|>", "<|math_x1|>", "<|math_x2|>"]  # x1 + x2
tokens = tokenizer.encode(expression, add_special_tokens=False)
print(f"Token IDs: {{tokens}}")

# 解码验证
decoded = tokenizer.decode(tokens)
print(f"Decoded: {{decoded}}")
```

### 表达式格式转换
```python
# 标准格式: ["+", "x1", "x2"]
# 数学token格式: ["<|math_add|>", "<|math_x1|>", "<|math_x2|>"]

def convert_to_math_tokens(standard_tokens):
    mapping = {{mapping_str}}
    return [mapping.get(token, token) for token in standard_tokens]

# 使用示例
standard_expr = ["+", "x1", "x2"]
math_expr = convert_to_math_tokens(standard_expr)
print(f"Math tokens: {{math_expr}}")
```

## 注意事项

1. 训练数据需要使用数学符号token格式
2. 推理时确保使用正确的token格式
3. 模型需要在扩展后重新训练以学习新token的语义
4. 建议使用较小的学习率进行微调

## 测试验证

使用 `test_math_token_inference.py` 脚本验证模型功能：

```bash
python test_math_token_inference.py --checkpoint {self.output_path} --train_data path/to/train.json
```
"""
        
        # 保存使用说明
        guide_file = os.path.join(self.output_path, "README.md")
        with open(guide_file, 'w', encoding='utf-8') as f:
            mapping_str = json.dumps(self.symbol_to_token, indent=2)
            # 替换模板中的映射字符串
            final_content = guide_content.replace("{mapping_str}", mapping_str)
            f.write(final_content)
        
        print(f"📖 使用说明已保存到: {guide_file}")
    
    def run_extension(self):
        """运行完整的扩展流程"""
        print(f"🚀 开始扩展模型词汇表")
        print(f"=" * 60)
        
        # 加载模型
        self.load_model_and_tokenizer()
        
        # 添加数学符号token
        self.add_math_tokens()
        
        # 验证token
        if not self.verify_tokens():
            print(f"❌ Token验证失败")
            return False
        
        # 保存扩展后的模型
        self.save_extended_model()
        
        # 创建使用说明
        self.create_usage_guide()
        
        print(f"\n" + "=" * 60)
        print(f"🎉 模型词汇表扩展完成!")
        print(f"原始模型: {self.model_path}")
        print(f"扩展后模型: {self.output_path}")
        print(f"新增token: {len(self.math_tokens)} 个")
        print(f"=" * 60)
        
        return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='扩展模型词汇表，添加数学符号token')
    parser.add_argument('--model_path', type=str, required=True,
                      help='原始模型路径')
    parser.add_argument('--output_path', type=str, required=True,
                      help='扩展后模型保存路径')
    parser.add_argument('--verify_only', action='store_true',
                      help='仅验证已存在的数学符号token')
    
    args = parser.parse_args()
    
    # 创建扩展器
    extender = MathTokenExtender(args.model_path, args.output_path)
    
    if args.verify_only:
        print(f"🔍 仅验证模式")
        extender.load_model_and_tokenizer()
        extender.verify_tokens()
    else:
        # 运行完整扩展
        success = extender.run_extension()
        exit(0 if success else 1)


if __name__ == "__main__":
    main() 


"""
  python expend_tokens.py \
    --model_path /home/dataset-local/liyanjie/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-3B-Instruct \
    --output_path /home/dataset-local/liyanjie/Qwen-SR-V2/Qwen/Qwen2.5-VL-3B-Instruct-expend-token
"""



