#!/usr/bin/env python3
"""
Generate symbolic regression dataset
Based on official Qwen2.5-VL data format
"""

import os
import sys
import argparse
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.symbolic_regression.data_processor import (
    SymbolicRegressionDataProcessor, 
    SymbolicRegressionConfig
)

def generate_symbolic_regression_data(num_samples, output_file):
    """Convenience function for generating symbolic regression data"""
    config = SymbolicRegressionConfig()
    processor = SymbolicRegressionDataProcessor(config)
    processor.generate_dataset(num_samples, output_file)
    print(f"✅ Generated {num_samples} samples, saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Symbolic Regression Dataset')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./symbolic_regression_data', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test data ratio')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = SymbolicRegressionConfig()
    processor = SymbolicRegressionDataProcessor(config)
    
    print(f"🚀 Starting symbolic regression dataset generation...")
    print(f"Total samples: {args.num_samples}")
    print(f"Output directory: {output_dir}")
    
    # Calculate the size of each dataset
    train_size = int(args.num_samples * args.train_ratio)
    val_size = int(args.num_samples * args.val_ratio)
    test_size = args.num_samples - train_size - val_size
    
    print(f"Training set: {train_size} Sample")
    print(f"Validation set: {val_size} Sample")
    print(f"Test set: {test_size} Sample")
    
    # Generating training set
    if train_size > 0:
        print("\n📊 Generating training set...")
        train_path = output_dir / "train.json"
        processor.generate_dataset(train_size, str(train_path))
    
    # Generating validation set
    if val_size > 0:
        print("\n📊 Generating validation set...")
        val_path = output_dir / "val.json"
        processor.generate_dataset(val_size, str(val_path))
    
    # Generating test set
    if test_size > 0:
        print("\n📊 Generating test set...")
        test_path = output_dir / "test.json"
        processor.generate_dataset(test_size, str(test_path))
    
    # Create dataset config file
    dataset_config = {
        "name": "symbolic_regression",
        "description": "Symbolic regression dataset for training mathematical expression generation models",
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
    
    print(f"\n✅ Dataset generation completed!")
    print(f"Config file: {config_path}")
    print(f"\n📁 Generated files:")
    for file_path in output_dir.glob("*.json"):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  {file_path.name}: {file_size:.1f} MB")

if __name__ == "__main__":
    main() 