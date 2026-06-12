#!/usr/bin/env python3
"""
Expand the model vocabulary by adding math-symbol special tokens
Math-symbol tokenization for symbolic regression tasks
"""

import torch
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import shutil

# Add project path to import custom model
project_root = Path(__file__).parent
sys.path.append(str(project_root / "qwen-vl-finetune"))

class MathTokenExtender:
    """Math symbol token extender"""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        
        # Math symbol mapping
        self.math_tokens = {
            # Basic operators
            "<|math_add|>": "+",
            "<|math_sub|>": "-", 
            "<|math_mul|>": "*",
            "<|math_div|>": "/",
            "<|math_pow|>": "^",
            
            # Trigonometric functions
            "<|math_sin|>": "sin",
            "<|math_cos|>": "cos",
            "<|math_tan|>": "tan",
            
            # Other functions
            "<|math_log|>": "log",
            "<|math_exp|>": "exp",
            "<|math_sqrt|>": "sqrt",
            "<|math_abs|>": "abs",
            
            # Variables
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
            
            # Constants
            "<|math_C|>": "C",
            "<|math_const_1|>": "1",
            "<|math_const_2|>": "2",
            "<|math_const_3|>": "3",
            "<|math_const_4|>": "4",
            "<|math_const_5|>": "5",
        }
        
        # Reverse mapping
        self.symbol_to_token = {v: k for k, v in self.math_tokens.items()}
        
    def load_model_and_tokenizer(self):
        """Load the original model and tokenizer"""
        print(f"📦 Loading original model: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Check model type
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "")
            architectures = config.get("architectures", [])
            
            print(f"📋 Model information:")
            print(f"   Model type: {model_type}")
            print(f"   Architecture: {architectures}")
            
            # Choose loading method based on model type
            if "SymbolicRegressionQwenModel" in architectures:
                print(f"🔧 Detected custom symbolic regression model; using dedicated loading method")
                try:
                    from qwenvl.symbolic_regression.model import SymbolicRegressionQwenModel
                    self.model = SymbolicRegressionQwenModel.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    print(f"✅ Loaded successfully with SymbolicRegressionQwenModel")
                except Exception as e:
                    print(f"⚠️ Failed to load with SymbolicRegressionQwenModel: {e}")
                    print(f"🔧 Trying AutoModel loading...")
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            elif model_type == "qwen2_5_vl" or any("Qwen2_5_VL" in arch for arch in architectures):
                print(f"🔧 Detected Qwen2.5-VL model; using AutoModel loading")
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # Standard model
                print(f"🔧 Using standard model loading")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
        else:
            # config.json not found; trying standard loading
            print(f"🔧 config.json not found; using standard model loading")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        print(f"✅ Original model loaded")
        print(f"   Original vocabulary size: {len(self.tokenizer.vocab)}")
        
    def check_existing_tokens(self):
        """Check existing math-symbol tokens"""
        existing_tokens = []
        missing_tokens = []
        
        for token in self.math_tokens.keys():
            if token in self.tokenizer.vocab:
                existing_tokens.append(token)
            else:
                missing_tokens.append(token)
        
        print(f"📊 Token status check:")
        print(f"   Existing: {len(existing_tokens)} items")
        print(f"   Need to add: {len(missing_tokens)} items")
        
        if existing_tokens:
            print(f"   Existing tokens: {existing_tokens[:5]}...")
        
        return existing_tokens, missing_tokens
    
    def add_math_tokens(self):
        """Add math-symbol tokens to the vocabulary"""
        existing_tokens, missing_tokens = self.check_existing_tokens()
        
        if not missing_tokens:
            print(f"✅ All math-symbol tokens already exist; no need to add")
            return
        
        print(f"➕ Adding {len(missing_tokens)} math-symbol tokens...")
        
        # Get original embedding matrix
        original_embeddings = self.model.get_input_embeddings()
        original_vocab_size = original_embeddings.weight.shape[0]
        embedding_dim = original_embeddings.weight.shape[1]
        
        # Add new tokens to tokenizer
        new_tokens = [token for token in missing_tokens]
        num_added = self.tokenizer.add_tokens(new_tokens)
        
        print(f"✅ Successfully added {num_added} tokens to the vocabulary")
        print(f"   New vocabulary size: {len(self.tokenizer.vocab)}")
        
        # Resize model embedding matrix
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize embeddings for new tokens
        self.initialize_new_embeddings(original_vocab_size, len(self.tokenizer))
        
        print(f"✅ Model embedding matrix resized")
        
    def initialize_new_embeddings(self, original_vocab_size: int, new_vocab_size: int):
        """Initialize embedding vectors for new tokens"""
        if original_vocab_size == new_vocab_size:
            return
            
        print(f"🔧 Initializing new token embedding vectors...")
        
        # Get resized embedding matrix
        embeddings = self.model.get_input_embeddings()
        
        with torch.no_grad():
            # For each newly added math-symbol token, initialize with embeddings of related symbols
            for token, symbol in self.math_tokens.items():
                # Get the new token ID
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if len(token_ids) == 1:
                    new_token_idx = token_ids[0]
                    
                    # Ensure the index is in the valid range
                    if new_token_idx >= original_vocab_size and new_token_idx < new_vocab_size:
                        # Try to find embeddings of related symbols
                        related_embeddings = []
                        
                        # Search for related tokens
                        for vocab_token, vocab_idx in self.tokenizer.vocab.items():
                            if vocab_idx < original_vocab_size:
                                if symbol in vocab_token.lower() or any(c in vocab_token for c in symbol):
                                    related_embeddings.append(embeddings.weight[vocab_idx])
                        
                        # If related embeddings are found, use their average
                        if related_embeddings:
                            avg_embedding = torch.stack(related_embeddings).mean(dim=0)
                            embeddings.weight[new_token_idx] = avg_embedding
                            print(f"   ✅ {token} (ID: {new_token_idx}) initialized with related embeddings")
                        else:
                            # Otherwise use random initialization with small variance
                            embeddings.weight[new_token_idx] = torch.randn(embeddings.weight.shape[1]) * 0.02
                            print(f"   🔧 {token} (ID: {new_token_idx}) initialized randomly")
                    else:
                        print(f"   ⚠️ {token} index out of range: {new_token_idx}")
        
        print(f"✅ New token embedding vector initialization completed")
    
    def verify_tokens(self):
        """Validate added tokens"""
        print(f"🔍 Validating math-symbol tokens...")
        
        success_count = 0
        
        for token, symbol in self.math_tokens.items():
            try:
                # Encode token
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                
                # Check whether it is a single token
                if len(token_ids) == 1:
                    # Decode validation
                    decoded = self.tokenizer.decode(token_ids[0])
                    if decoded == token:
                        success_count += 1
                        print(f"   ✅ {token} -> ID: {token_ids[0]} ({symbol})")
                    else:
                        print(f"   ❌ {token} decode mismatch: {decoded}")
                else:
                    print(f"   ❌ {token} was split into multiple tokens: {token_ids}")
                    
            except Exception as e:
                print(f"   ❌ {token} validation failed: {e}")
        
        print(f"📊 Validation result: {success_count}/{len(self.math_tokens)} tokens succeeded")
        
        return success_count == len(self.math_tokens)
    
    def save_extended_model(self):
        """Save the expanded model"""
        print(f"💾 Saving expanded model to: {self.output_path}")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_path)
        
        # Save model
        self.model.save_pretrained(self.output_path)
        
        # Save math symbol mapping
        mapping_file = os.path.join(self.output_path, "math_token_mapping.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.math_tokens, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model saved")
        print(f"   Vocabulary size: {len(self.tokenizer.vocab)}")
        print(f"   Model parameters: {self.model.num_parameters():,}")
        
    def create_usage_guide(self):
        """Create usage guide"""
        guide_content = f"""# Math Symbol Token Expanded Model Usage Guide

## Model information
- Original model: {self.model_path}
- Expanded model: {self.output_path}
- Number of new tokens: {len(self.math_tokens)}

## Math symbol mapping

### Basic operators
"""
        
        # Organize tokens by category
        categories = {
            "Basic operators": ["+", "-", "*", "/", "^"],
            "Trigonometric functions": ["sin", "cos", "tan"],
            "Other functions": ["log", "exp", "sqrt", "abs"],
            "Variables": [f"x{i}" for i in range(1, 11)],
            "Constants": ["C", "1", "2", "3", "4", "5"]
        }
        
        for category, symbols in categories.items():
            guide_content += f"\n### {category}\n"
            for symbol in symbols:
                if symbol in self.symbol_to_token:
                    token = self.symbol_to_token[symbol]
                    guide_content += f"- {symbol} -> {token}\n"
        
        guide_content += f"""
## Usage example

### Python code example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the expanded model
tokenizer = AutoTokenizer.from_pretrained('{self.output_path}')
model = AutoModelForCausalLM.from_pretrained('{self.output_path}')

# Example: encode a math expression
expression = ["<|math_add|>", "<|math_x1|>", "<|math_x2|>"]  # x1 + x2
tokens = tokenizer.encode(expression, add_special_tokens=False)
print(f"Token IDs: {{tokens}}")

# Decode validation
decoded = tokenizer.decode(tokens)
print(f"Decoded: {{decoded}}")
```

### Expression format conversion
```python
# Standard format: ["+", "x1", "x2"]
# Math token format: ["<|math_add|>", "<|math_x1|>", "<|math_x2|>"]

def convert_to_math_tokens(standard_tokens):
    mapping = {{mapping_str}}
    return [mapping.get(token, token) for token in standard_tokens]

# Usage example
standard_expr = ["+", "x1", "x2"]
math_expr = convert_to_math_tokens(standard_expr)
print(f"Math tokens: {{math_expr}}")
```

## Notes

1. Training data must use the math-symbol token format
2. Ensure the correct token format is used during inference
3. The model needs to be retrained after expansion to learn the semantics of new tokens
4. Use a smaller learning rate for fine-tuning

## Test validation

Use the `test_math_token_inference.py` script to validate model functionality：

```bash
python test_math_token_inference.py --checkpoint {self.output_path} --train_data path/to/train.json
```
"""
        
        # Save usage guide
        guide_file = os.path.join(self.output_path, "README.md")
        with open(guide_file, 'w', encoding='utf-8') as f:
            mapping_str = json.dumps(self.symbol_to_token, indent=2)
            # Replace the mapping string in the template
            final_content = guide_content.replace("{mapping_str}", mapping_str)
            f.write(final_content)
        
        print(f"📖 Usage guide saved to: {guide_file}")
    
    def run_extension(self):
        """Run the complete expansion workflow"""
        print(f"🚀 Starting model vocabulary expansion")
        print(f"=" * 60)
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Add math-symbol tokens
        self.add_math_tokens()
        
        # Validate tokens
        if not self.verify_tokens():
            print(f"❌ Token validation failed")
            return False
        
        # Save the expanded model
        self.save_extended_model()
        
        # Create usage guide
        self.create_usage_guide()
        
        print(f"\n" + "=" * 60)
        print(f"🎉 Model vocabulary expansion completed!")
        print(f"Original model: {self.model_path}")
        print(f"Expanded model: {self.output_path}")
        print(f"New tokens: {len(self.math_tokens)} items")
        print(f"=" * 60)
        
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Expand the model vocabulary by adding math-symbol tokens')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Original model path')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Expanded model save path')
    parser.add_argument('--verify_only', action='store_true',
                      help='Only validate existing math-symbol tokens')
    
    args = parser.parse_args()
    
    # Create extender
    extender = MathTokenExtender(args.model_path, args.output_path)
    
    if args.verify_only:
        print(f"🔍 Validation-only mode")
        extender.load_model_and_tokenizer()
        extender.verify_tokens()
    else:
        # Run complete expansion
        success = extender.run_extension()
        exit(0 if success else 1)


if __name__ == "__main__":
    main() 


"""
  python expend_tokens.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --output_path /path/to/ChatSR/Qwen/Qwen2.5-VL-3B-Instruct-expend-token
"""



