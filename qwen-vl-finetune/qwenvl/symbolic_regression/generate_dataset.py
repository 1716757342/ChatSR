#!/usr/bin/env python3
"""
生成符号回归数据集
基于Qwen2.5-VL官方数据格式
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.symbolic_regression.data_processor import (
    SymbolicRegressionDataProcessor, 
    SymbolicRegressionConfig
)

def generate_symbolic_regression_data(num_samples, output_file):
    """生成符号回归数据的便捷函数"""
    config = SymbolicRegressionConfig()
    processor = SymbolicRegressionDataProcessor(config)
    processor.generate_dataset(num_samples, output_file)
    print(f"✅ 生成了 {num_samples} 个样本，保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Symbolic Regression Dataset')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./symbolic_regression_data', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test data ratio')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建配置
    config = SymbolicRegressionConfig()
    processor = SymbolicRegressionDataProcessor(config)
    
    print(f"🚀 开始生成符号回归数据集...")
    print(f"总样本数: {args.num_samples}")
    print(f"输出目录: {output_dir}")
    
    # 计算各个数据集的大小
    train_size = int(args.num_samples * args.train_ratio)
    val_size = int(args.num_samples * args.val_ratio)
    test_size = args.num_samples - train_size - val_size
    
    print(f"训练集: {train_size} 样本")
    print(f"验证集: {val_size} 样本")
    print(f"测试集: {test_size} 样本")
    
    # 生成训练集
    if train_size > 0:
        print("\n📊 生成训练集...")
        train_path = output_dir / "train.json"
        processor.generate_dataset(train_size, str(train_path))
    
    # 生成验证集
    if val_size > 0:
        print("\n📊 生成验证集...")
        val_path = output_dir / "val.json"
        processor.generate_dataset(val_size, str(val_path))
    
    # 生成测试集
    if test_size > 0:
        print("\n📊 生成测试集...")
        test_path = output_dir / "test.json"
        processor.generate_dataset(test_size, str(test_path))
    
    # 创建数据集配置文件
    dataset_config = {
        "name": "symbolic_regression",
        "description": "符号回归数据集，用于训练数学表达式生成模型",
        "total_samples": args.num_samples,
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": test_size,
        "input_dim": config.input_dim,
        "max_points": config.max_points,
        "min_points": config.min_points,
        "vocab_size": len(processor.vocab),
        "vocabulary": processor.vocab
    }
    
    import json
    config_path = output_dir / "dataset_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 数据集生成完成!")
    print(f"配置文件: {config_path}")
    print(f"\n📁 生成的文件:")
    for file_path in output_dir.glob("*.json"):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  {file_path.name}: {file_size:.1f} MB")

if __name__ == "__main__":
    main() 