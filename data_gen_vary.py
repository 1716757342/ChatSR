#!/usr/bin/env python3
"""
  2. data_gen_vary.py：multi-prompt version of data_gen.py
  
  Purpose: also generates formulas with special tokens, but uses more diverse prompts and answer templates.

  Example output:

  Based on the data, the derived formula is: <|math_add|>,<|math_x1|>,<|math_x2|>

  Features:

  - Also converts +, -, *, sin, and x1 into <|math_xxx|>
  - More diverse prompt templates
  - Helps improve model generalization to different question phrasings
  - The format is not necessarily fixed with [...] like data_gen.py

  If you want stable training first, use data_gen.py.
  If you later want the model to adapt to more prompts, use data_gen_vary.py.

  ---
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict
import math
import argparse

# ==============================================================================
# 1. Expression tree and generation logic
# ==============================================================================

class ExpressionNode:
    """Expression node class"""
    def __init__(self, value: str, children: List['ExpressionNode'] = None):
        self.value = value
        self.children = children or []

    def to_preorder(self) -> List[str]:
        result = [self.value]
        for child in self.children:
            result.extend(child.to_preorder())
        return result

    def evaluate(self, variables: Dict[str, float]) -> float:
        if self.value.startswith('x'):
            return variables.get(self.value, 0.0)
        elif self.value.replace('.', '').replace('-', '').isdigit():
            return float(self.value)
        elif self.value == 'C':
            return variables.get('C', 1.0)
        elif self.value == '+':
            return self.children[0].evaluate(variables) + self.children[1].evaluate(variables)
        elif self.value == '-':
            return self.children[0].evaluate(variables) - self.children[1].evaluate(variables)
        elif self.value == '*':
            return self.children[0].evaluate(variables) * self.children[1].evaluate(variables)
        elif self.value == '/':
            divisor = self.children[1].evaluate(variables)
            if abs(divisor) < 1e-8:
                divisor = 1e-8
            return self.children[0].evaluate(variables) / divisor
        elif self.value == '^':
            base = self.children[0].evaluate(variables)
            exp = self.children[1].evaluate(variables)
            exp = np.clip(exp, -5, 5)
            if base < 0 and exp % 1 != 0:
                base = abs(base)
            return base ** exp
        elif self.value == 'sin':
            return math.sin(self.children[0].evaluate(variables))
        elif self.value == 'cos':
            return math.cos(self.children[0].evaluate(variables))
        elif self.value == 'exp':
            arg = self.children[0].evaluate(variables)
            arg = np.clip(arg, -10, 10)
            return math.exp(arg)
        elif self.value == 'log':
            arg = self.children[0].evaluate(variables)
            if arg <= 0:
                arg = 1e-8
            return math.log(arg)
        elif self.value == 'sqrt':
            arg = self.children[0].evaluate(variables)
            if arg < 0:
                arg = abs(arg)
            return math.sqrt(arg)
        else:
            raise ValueError(f"Unknown operator: {self.value}")

class AdvancedExpressionGenerator:
    """Advanced expression generator"""
    def __init__(self, max_variables: int = 5):
        self.max_variables = max_variables
        self.binary_ops = ['+', '-', '*', '/', '^']
        self.unary_functions = ['sin', 'cos', 'exp', 'log', 'sqrt']
        self.constants = ['1', '2', '3', 'C']
        self.prob_binary_op = 0.4
        self.prob_unary_func = 0.3
        self.prob_variable = 0.2
        self.prob_constant = 0.1
        self.active_variables = []

    def set_active_variables(self, active_vars: List[str]):
        self.active_variables = active_vars

    def generate_expression(self, max_length: int, current_length: int = 0) -> ExpressionNode:
        if current_length >= max_length - 1:
            return self._generate_leaf()
        
        remaining_length = max_length - current_length
        if remaining_length >= 3:
            choices = ['binary', 'unary', 'leaf']
            weights = [self.prob_binary_op, self.prob_unary_func, self.prob_variable + self.prob_constant]
        elif remaining_length >= 2:
            choices = ['unary', 'leaf']
            weights = [self.prob_unary_func, self.prob_variable + self.prob_constant]
        else:
            choices = ['leaf']
            weights = [1.0]
        
        choice = random.choices(choices, weights=weights)[0]
        
        if choice == 'binary':
            return self._generate_binary_node(max_length, current_length)
        elif choice == 'unary':
            return self._generate_unary_node(max_length, current_length)
        else:
            return self._generate_leaf()

    def _generate_binary_node(self, max_length: int, current_length: int) -> ExpressionNode:
        op = random.choice(self.binary_ops)
        node = ExpressionNode(op)
        remaining = max_length - current_length - 1
        left_length = random.randint(1, max(1, remaining - 1))
        node.children = [
            self.generate_expression(current_length + 1 + left_length, current_length + 1),
            self.generate_expression(max_length, current_length + 1 + left_length)
        ]
        return node

    def _generate_unary_node(self, max_length: int, current_length: int) -> ExpressionNode:
        func = random.choice(self.unary_functions)
        node = ExpressionNode(func)
        node.children = [self.generate_expression(max_length, current_length + 1)]
        return node

    def _generate_leaf(self) -> ExpressionNode:
        if self.active_variables and random.random() < 0.7:
            return ExpressionNode(random.choice(self.active_variables))
        else:
            return ExpressionNode(random.choice(self.constants))

    def get_used_variables(self, node: ExpressionNode) -> List[str]:
        variables = set()
        def collect_vars(n):
            if n.value.startswith('x'):
                variables.add(n.value)
            for child in n.children:
                collect_vars(child)
        collect_vars(node)
        return sorted(list(variables))

# ==============================================================================
# 2. Special token mapping
# ==============================================================================

def get_special_token_map() -> Dict[str, str]:
    """Return a mapping dictionary from standard symbols to special tokens"""
    # This map should align with your `extend_tokenizer_math_tokens.py` script
    math_tokens_def = {
        "<|math_add|>": "+", "<|math_sub|>": "-", "<|math_mul|>": "*",
        "<|math_div|>": "/", "<|math_pow|>": "^", "<|math_sin|>": "sin",
        "<|math_cos|>": "cos", "<|math_tan|>": "tan", "<|math_log|>": "log",
        "<|math_exp|>": "exp", "<|math_sqrt|>": "sqrt", "<|math_abs|>": "abs",
        "<|math_x1|>": "x1", "<|math_x2|>": "x2", "<|math_x3|>": "x3",
        "<|math_x4|>": "x4", "<|math_x5|>": "x5", "<|math_x6|>": "x6",
        "<|math_x7|>": "x7", "<|math_x8|>": "x8", "<|math_x9|>": "x9",
        "<|math_x10|>": "x10", "<|math_const_1|>": "1", "<|math_const_2|>": "2",
        "<|math_const_3|>": "3", "<|math_const_4|>": "4", "<|math_const_5|>": "5",
        "<|math_C|>": "C" # Assuming you have a token for the general constant
    }
    return {v: k for k, v in math_tokens_def.items()}

# ==============================================================================
# 3. Integrated data generation function
# ==============================================================================

def generate_sr_data_with_special_tokens(num_samples: int = 1000,
                                         max_expr_length: int = 20,
                                         min_expr_length: int = 7,
                                         output_dir: str = "advanced_sr_data_final",
                                         max_vars: int = 5,
                                         max_dims: int = 10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = AdvancedExpressionGenerator(max_variables=max_vars)
    symbol_to_token_map = get_special_token_map()
    dataset = []
    
    # --- Core change: define diverse conversation templates ---
    human_templates = [
        "<data>\nMy task is to find a fitting formula for this data. Please help me generate the corresponding preorder traversal expression for the formula, using special tokens.",
        "<data>\nBased on the following sampled data, please perform symbolic regression and output the preorder traversal result of the expression binary tree.",
        "<data>\nPlease derive the fitting expression for this data and return its preorder traversal.",
        "<data>\nCan you find a mathematical expression for me based on these data points? I only need its preorder traversal."
    ]
    gpt_templates = [
        "Okay, the preorder traversal of the expression I obtained is: {}",
        "Of course, the analyzed expression is as follows: {}",
        "Based on the data, the derived formula is: {}",
        "Happy to help. Here is the preorder traversal of the expression I found: {}"
    ]

    # gpt_templates = [
    #     "{}",
    #     "{}",
    #     "{}",
    #     "{}"
    # ]
    
    print(f"🚀 Starting to generate {num_samples} high-quality symbolic regression samples...")
    
    success_count = 0
    attempt_count = 0
    
    while success_count < num_samples and attempt_count < num_samples * 20:
        attempt_count += 1
        
        try:
            num_active_vars = random.randint(1, generator.max_variables)
            active_vars = [f'x{i}' for i in range(1, num_active_vars + 1)]
            generator.set_active_variables(active_vars)

            effective_min_length = max(min_expr_length, 2 * num_active_vars)
            if effective_min_length > max_expr_length:
                continue

            expr_length = random.randint(effective_min_length, max_expr_length)
            expr_tree = generator.generate_expression(expr_length)
            
            used_vars = generator.get_used_variables(expr_tree)

            if set(used_vars) != set(active_vars):
                continue

            standard_tokens = expr_tree.to_preorder()
            special_tokens = [symbol_to_token_map.get(t, t) for t in standard_tokens]
            
            num_features = len(active_vars)
            
            num_points = random.randint(30, 60)
            X = np.random.uniform(-2, 2, (num_points, num_features))
            y_values = []
            constant_value = random.uniform(0.5, 3.0)
            
            for i in range(num_points):
                variables = {'C': constant_value}
                for j, var_name in enumerate(active_vars):
                    variables[var_name] = X[i, j]
                
                try:
                    y = expr_tree.evaluate(variables)
                    if not np.isfinite(y) or abs(y) > 100 or (abs(y) < 1e-10 and y != 0):
                        break
                    y_values.append(y)
                except (ValueError, OverflowError, ZeroDivisionError):
                    break
            
            if len(y_values) != num_points:
                continue
            
            y_array = np.array(y_values)
            noise_level = 0.01 * np.std(y_array) if np.std(y_array) > 1e-6 else 0.01
            y_array += np.random.normal(0, noise_level, len(y_array))
            
            if num_features < max_dims:
                padding = np.zeros((num_points, max_dims - num_features))
                X_padded = np.column_stack([X, padding])
            else:
                X_padded = X[:, :max_dims]
            
            data_points = np.column_stack([X_padded, y_array])
            
            # --- Core change: randomly select and format a conversation ---
            human_prompt = random.choice(human_templates)
            gpt_response_template = random.choice(gpt_templates)
            
            expression_str = ",".join(special_tokens)
            gpt_response = gpt_response_template.format(expression_str)
            
            conversations = [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": gpt_response}
            ]
            
            sample = {
                "id": f"advanced_sr_final_{success_count}",
                "conversations": conversations,
                "data_points": data_points.astype(float).tolist(),
                "expression_tokens": special_tokens,
                "standard_tokens": standard_tokens,
                "expression_length": len(special_tokens),
                "num_features": num_features,
                "num_points": num_points,
                "used_variables": used_vars
            }
            
            dataset.append(sample)
            success_count += 1
            
            if success_count % 100 == 0:
                print(f"✅ Generated {success_count}/{num_samples} samples (Attempts: {attempt_count})")
                
        except Exception:
            continue
    
    random.shuffle(dataset)
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 Data generation complete! All samples ensure variable continuity and completeness.")
    print(f"📊 Total samples: {len(dataset)} (from {attempt_count} attempts)")
    print(f"📁 Saved to directory: {output_dir}")
    
    print(f"\n📝 Sample Examples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  --- Sample {i+1} (#Features: {sample['num_features']}, Expr Length: {sample['expression_length']}) ---")
        print(f"  Special Tokens: [{','.join(sample['expression_tokens'])}]")
        print(f"  Used Variables: {sample['used_variables']}")


# ==============================================================================
# 4. Main execution module
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Advanced Symbolic Regression Dataset with Special Tokens (Final Version)')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to generate.')
    parser.add_argument('--max_length', type=int, default=10, help='Maximum preorder traversal length of the expression.')
    parser.add_argument('--min_length', type=int, default=5, help='Minimum preorder traversal length of the expression.')
    parser.add_argument('--output_dir', type=str, default='symbolic_regression_data_20', help='Output directory to save the data.')
    parser.add_argument('--max_vars', type=int, default=3, help='Maximum number of variables to use in a single sample.')
    parser.add_argument('--max_dims', type=int, default=3, help='Maximum dimension for data points (for padding).')
    
    args = parser.parse_args()
    
    generate_sr_data_with_special_tokens(
        num_samples=args.num_samples,
        max_expr_length=args.max_length,
        min_expr_length=args.min_length,
        output_dir=args.output_dir,
        max_vars=args.max_vars,
        max_dims=args.max_dims
    )
