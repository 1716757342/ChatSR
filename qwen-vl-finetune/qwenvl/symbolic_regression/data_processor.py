# """
# Symbolic regression data processor
# Replaces image processing in Qwen2.5-VL and processes mathematical data points
# """

# import torch
# import numpy as np
# import json
# from typing import List, Dict, Optional, Tuple, Union
# from dataclasses import dataclass
# import random
# import math
# import copy


# @dataclass
# class SymbolicRegressionConfig:
#     """Symbolic regression config"""
#     # Data config
#     # Fix: change input dimension from 11 to 6 to match the new data format of 5 independent variables (x1-x5) plus 1 dependent variable (y).
#     # This is the key to resolving the `mat1 and mat2 shapes cannot be multiplied` error.
#     input_dim: int = 6     # Dimension including y
#     max_points: int = 100  # Maximum number of data points per sample
#     min_points: int = 20   # Minimum number of data points per sample

#     # Set Transformer config
#     hidden_size: int = 896  # Match Qwen2.5-VL hidden_size
#     num_attention_heads: int = 14  # Match Qwen2.5-VL
#     num_set_layers: int = 3
#     inducing_points: int = 32
#     pooling_outputs: int = 128  # Number of output features

#     # Vocabulary config
#     vocab_size: int = 151936  # Match Qwen2.5-VL vocab_size
#     # vocab_size: int = 151692  # Match Qwen2.5-VL vocab_size

# class SymbolicRegressionDataProcessor:
#     """
#     Symbolic regression data processor
#     Similar to Qwen2.5-VL image processor, but processes mathematical data points
#     """

#     def __init__(self, config: SymbolicRegressionConfig):
#         self.config = config

#         # Define symbolic regression vocabulary
#         self.vocab = {
#             '<PAD>': 0, '<START>': 1, '<END>': 2, '<DATA>': 3,
#             '+': 4, '-': 5, '*': 6, '/': 7, '^': 8, '(': 9, ')': 10,
#             'sin': 11, 'cos': 12, 'exp': 13, 'log': 14, 'sqrt': 15,
#             'x1': 16, 'x2': 17, 'x3': 18, 'x4': 19, 'x5': 20,
#             # Fix: remove x6 to x10 because at most 5 dimensions are now supported
#             'C': 21,  # Constants
#             '1': 22, '2': 23, '3': 24, '4': 25, '5': 26,
#         }
#         self.idx_to_token = {v: k for k, v in self.vocab.items()}

#         # Predefined mathematical functions grouped by feature count
#         self.functions_by_features = {
#             1: [
#                 (lambda x: x[:, 0] + 2.5, ['+', 'x1', 'C']),
#                 (lambda x: 3.0 * x[:, 0], ['*', 'C', 'x1']),
#                 (lambda x: np.sin(x[:, 0]) + 1.5, ['+', 'sin', 'x1', 'C']),
#                 (lambda x: x[:, 0] ** 2 - 0.5, ['-', '^', 'x1', '2', 'C']),
#             ],
#             2: [
#                 (lambda x: x[:, 0] + x[:, 1], ['+', 'x1', 'x2']),
#                 (lambda x: x[:, 0] * x[:, 1], ['*', 'x1', 'x2']),
#                 (lambda x: np.sin(x[:, 0]) + x[:, 1], ['+', 'sin', 'x1', 'x2']),
#                 (lambda x: 2.0 * x[:, 0] + x[:, 1], ['+', '*', 'C', 'x1', 'x2']),
#             ],
#             3: [
#                 (lambda x: x[:, 0] + x[:, 1] * x[:, 2], ['+', 'x1', '*', 'x2', 'x3']),
#                 (lambda x: (x[:, 0] + x[:, 1]) * x[:, 2], ['*', '+', 'x1', 'x2', 'x3']),
#             ],
#             4: [
#                 (lambda x: x[:, 0] * x[:, 1] + x[:, 2] / (x[:, 3] + 1e-8), ['+', '*', 'x1', 'x2', '/', 'x3', 'x4']),
#             ],
#             5: [
#                 (lambda x: np.sin(x[:, 0] + x[:, 1]) + x[:, 2] * x[:, 3] - x[:, 4], ['-', '+', 'sin', '+', 'x1', 'x2', '*', 'x3', 'x4', 'x5']),
#             ]
#         }

#     def generate_data_sample(self) -> Dict:
#         """
#         Generate one symbolic regression training sample

#         Returns:
#             Dict: Dictionary containing data points and target expression
#         """
#         # Fix: randomly choose 1 to 5 features to match the new dimension setting
#         num_features = np.random.randint(1, 6)

#         # Randomly choose function
#         if num_features not in self.functions_by_features:
#             num_features = 1  # fall back to 1 feature

#         available_functions = self.functions_by_features[num_features]
#         func, expr_tokens = available_functions[np.random.randint(len(available_functions))]

#         # Generate number of data points
#         num_points = np.random.randint(self.config.min_points, self.config.max_points + 1)

#         # Generate random input points
#         X = np.random.uniform(-2, 2, (num_points, num_features))

#         try:
#             # Compute function values
#             y = func(X)

#             # Check validity
#             if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.max(np.abs(y)) > 100:
#                 return self.generate_data_sample()  # regenerate

#             # Pad to maximum number of features (5 dimensions)
#             if num_features < self.config.input_dim - 1:  # -1 because the last column is y
#                 padding = np.zeros((num_points, self.config.input_dim - 1 - num_features))
#                 X = np.column_stack([X, padding])

#             # Combine data points [x1, x2, ..., x5, y]
#             data_points = np.column_stack([X, y])

#             # Filter tokens in the vocabulary
#             expr_tokens_filtered = [token for token in expr_tokens if token in self.vocab]

#             if len(expr_tokens_filtered) == 0:
#                 return self.generate_data_sample()  # regenerate

#             return {
#                 'data_points': data_points.astype(np.float32),
#                 'expression_tokens': expr_tokens_filtered,
#                 'num_features': num_features,
#                 'num_points': num_points
#             }

#         except Exception as e:
#             return self.generate_data_sample()  # regenerate

#     def tokenize_expression(self, expression_tokens: List[str]) -> List[int]:
#         """
#         Convert expression token sequence to ID sequence

#         Args:
#             expression_tokens: Expression token list

#         Returns:
#             List[int]: token ID sequence
#         """
#         token_ids = [self.vocab['<START>']]
#         for token in expression_tokens:
#             if token in self.vocab:
#                 token_ids.append(self.vocab[token])
#         token_ids.append(self.vocab['<END>'])
#         return token_ids

#     def process_data_points(self, data_points: np.ndarray) -> torch.Tensor:
#         """
#         Process data points and convert to model input format

#         Args:
#             data_points: (num_points, input_dim) numpy array

#         Returns:
#             torch.Tensor: Processed data point tensor
#         """
#         # Convert to torch tensor
#         data_tensor = torch.from_numpy(data_points).float()

#         # Pad if there are not enough points
#         if data_tensor.shape[0] < self.config.max_points:
#             pad_size = self.config.max_points - data_tensor.shape[0]
#             padding = torch.zeros(pad_size, data_tensor.shape[1])
#             data_tensor = torch.cat([data_tensor, padding], dim=0)
#         elif data_tensor.shape[0] > self.config.max_points:
#             # Randomly sample if there are too many points
#             indices = torch.randperm(data_tensor.shape[0])[:self.config.max_points]
#             data_tensor = data_tensor[indices]

#         return data_tensor

#     def create_conversation_format(self, data_sample: Dict) -> Dict:
#         """
#         Convert data sample to Qwen2.5-VL conversation format

#         Args:
#             data_sample: Data sample dictionary

#         Returns:
#             Dict: Data in conversation format
#         """
#         # Build conversation
#         conversations = [
#             {
#                 "from": "human",
#                 "value": "<data>\nThis is a sampled set of scientific data points. Please find an expression to fit this data. You only need to generate the preorder traversal of the expression binary tree."
#             },
#             {
#                 "from": "gpt",
#                 "value": f"Okay, the preorder traversal of the expression I got is[{','.join(data_sample['expression_tokens'])}]"
#             }
#         ]

#         return {
#             "id": f"symbolic_regression_{np.random.randint(1000000)}",
#             "conversations": conversations,
#             "data_points": data_sample['data_points'].tolist(),
#             "num_features": data_sample['num_features'],
#             "num_points": data_sample['num_points']
#         }

#     def generate_dataset(self, num_samples: int, output_path: str):
#         """
#         Generate symbolic regression dataset

#         Args:
#             num_samples: Number of samples
#             output_path: Output file path
#         """
#         dataset = []

#         for i in range(num_samples):
#             try:
#                 data_sample = self.generate_data_sample()
#                 conversation_data = self.create_conversation_format(data_sample)
#                 dataset.append(conversation_data)

#                 if (i + 1) % 1000 == 0:
#                     print(f"Generated {i + 1}/{num_samples} samples")

#             except Exception as e:
#                 print(f"Error generating sample {i}: {e}")
#                 continue

#         # Save dataset
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(dataset, f, ensure_ascii=False, indent=2)

#         print(f"Dataset saved to {output_path} with {len(dataset)} samples")
#         return dataset

# # Data point token used to replace image tokens
# DATA_TOKEN_INDEX = 151657  # Use a new token index
# DEFAULT_DATA_TOKEN = "<data>"

# def preprocess_symbolic_regression(
#     sources,
#     tokenizer,
#     data_grid_info: List = [],
# ) -> Dict:
#     """
#     Preprocess symbolic regression data, similar to preprocess_qwen_2_visual

#     Args:
#         sources: Conversation data sources
#         tokenizer: Tokenizer
#         data_grid_info: Data grid information

#     Returns:
#         Dict: Preprocessed data
#     """
#     roles = {"human": "user", "gpt": "assistant"}
#     system_message = "You are a helpful assistant specialized in symbolic regression."

#     tokenizer = copy.deepcopy(tokenizer)
#     chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
#     tokenizer.chat_template = chat_template

#     data_replicate_index = 0
#     input_ids, targets = [], []

#     for i, source in enumerate(sources):
#         try:
#             if roles[source[0]["from"]] != roles["human"]:
#                 source = source[1:]
#         except:
#             print(sources)

#         input_id, target = [], []

#         input_id += tokenizer.apply_chat_template(
#             [{"role": "system", "content": system_message}]
#         )
#         target += [-100] * len(input_id)  # IGNORE_INDEX

#         for conv in source:
#             try:
#                 role = conv["role"]
#                 content = conv["content"]
#             except:
#                 role = conv["from"]
#                 content = conv["value"]

#             role = roles.get(role, role)
#             if role == "user":
#                 if "<data>" in content:
#                     # Replace <data> markers with special data tokens
#                     parts = content.split("<data>")
#                     new_parts = []
#                     for j in range(len(parts) - 1):
#                         new_parts.append(parts[j])
#                         replacement = (
#                             "<|vision_start|>"
#                             + f"<|data_pad|>" * data_grid_info[data_replicate_index]
#                             + "<|vision_end|>"
#                         )
#                         new_parts.append(replacement)
#                         data_replicate_index += 1
#                     new_parts.append(parts[-1])
#                     content = "".join(new_parts)

#             conv = [{"role": role, "content": content}]
#             encode_id = tokenizer.apply_chat_template(conv)
#             input_id += encode_id
#             if role in ["user", "system"]:
#                 target += [-100] * len(encode_id)  # IGNORE_INDEX
#             else:
#                 target_mask = encode_id.copy()
#                 target_mask[:3] = [-100] * 3  # IGNORE_INDEX for special tokens
#                 target += target_mask

#         assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
#         input_ids.append(input_id)
#         targets.append(target)

#     # input_ids and targets may need padding here, but Trainer usually handles it
#     # For safety, implement it manually or rely on Trainer DataCollator

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )

"""
Use this when LoRA merging fails.
Symbolic regression data processor
Replaces image processing in Qwen2.5-VL and processes mathematical data points
"""


import torch
import numpy as np
import json
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import random
import math
import copy

# --- Key fix: import PretrainedConfig ---
from transformers.configuration_utils import PretrainedConfig

Dim = 4 ####### This must match the synthetic data dimension (including y) ####################
# --- Key fix: make SymbolicRegressionConfig inherit from PretrainedConfig ---
@dataclass
class SymbolicRegressionConfig(PretrainedConfig):
    """Symbolic regression config"""
    # Data config
    # Fix: change input dimension from 11 to 6 to match the new data format of 5 independent variables (x1-x5) plus 1 dependent variable (y).
    # This is the key to resolving the `mat1 and mat2 shapes cannot be multiplied` error.
    input_dim: int = Dim     # Dimension including y
    max_points: int = 100  # Maximum number of data points per sample
    min_points: int = 20   # Minimum number of data points per sample

    # Set Transformer config
    hidden_size: int = 896  # Match Qwen2.5-VL hidden_size
    num_attention_heads: int = 14  # Match Qwen2.5-VL
    num_set_layers: int = 3
    inducing_points: int = 32
    pooling_outputs: int = 128  # Number of output features

    # Vocabulary config
    vocab_size: int = 151936  # Match Qwen2.5-VL vocab_size
    # vocab_size: int = 151692  # Match Qwen2.5-VL vocab_size

class SymbolicRegressionDataProcessor:
    """
    Symbolic regression data processor
    Similar to Qwen2.5-VL image processor, but processes mathematical data points
    """

    def __init__(self, config: SymbolicRegressionConfig):
        self.config = config

        # Define symbolic regression vocabulary
        self.vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2, '<DATA>': 3,
            '+': 4, '-': 5, '*': 6, '/': 7, '^': 8, '(': 9, ')': 10,
            'sin': 11, 'cos': 12, 'exp': 13, 'log': 14, 'sqrt': 15,
            'x1': 16, 'x2': 17, 'x3': 18, 'x4': 19, 'x5': 20,
            # Fix: remove x6 to x10 because at most 5 dimensions are now supported
            'C': 21,  # Constants
            '1': 22, '2': 23, '3': 24, '4': 25, '5': 26,
        }
        self.idx_to_token = {v: k for k, v in self.vocab.items()}

        # Predefined mathematical functions grouped by feature count
        self.functions_by_features = {
            1: [
                (lambda x: x[:, 0] + 2.5, ['+', 'x1', 'C']),
                (lambda x: 3.0 * x[:, 0], ['*', 'C', 'x1']),
                (lambda x: np.sin(x[:, 0]) + 1.5, ['+', 'sin', 'x1', 'C']),
                (lambda x: x[:, 0] ** 2 - 0.5, ['-', '^', 'x1', '2', 'C']),
            ],
            2: [
                (lambda x: x[:, 0] + x[:, 1], ['+', 'x1', 'x2']),
                (lambda x: x[:, 0] * x[:, 1], ['*', 'x1', 'x2']),
                (lambda x: np.sin(x[:, 0]) + x[:, 1], ['+', 'sin', 'x1', 'x2']),
                (lambda x: 2.0 * x[:, 0] + x[:, 1], ['+', '*', 'C', 'x1', 'x2']),
            ],
            3: [
                (lambda x: x[:, 0] + x[:, 1] * x[:, 2], ['+', 'x1', '*', 'x2', 'x3']),
                (lambda x: (x[:, 0] + x[:, 1]) * x[:, 2], ['*', '+', 'x1', 'x2', 'x3']),
            ],
            4: [
                (lambda x: x[:, 0] * x[:, 1] + x[:, 2] / (x[:, 3] + 1e-8), ['+', '*', 'x1', 'x2', '/', 'x3', 'x4']),
            ],
            5: [
                (lambda x: np.sin(x[:, 0] + x[:, 1]) + x[:, 2] * x[:, 3] - x[:, 4], ['-', '+', 'sin', '+', 'x1', 'x2', '*', 'x3', 'x4', 'x5']),
            ]
        }

    def generate_data_sample(self) -> Dict:
        """
        Generate one symbolic regression training sample

        Returns:
            Dict: Dictionary containing data points and target expression
        """
        # Fix: randomly choose 1 to 2 features to match the new dimension setting
        num_features = np.random.randint(1, Dim)

        # Randomly choose function
        if num_features not in self.functions_by_features:
            num_features = 1  # fall back to 1 feature

        available_functions = self.functions_by_features[num_features]
        func, expr_tokens = available_functions[np.random.randint(len(available_functions))]

        # Generate number of data points
        num_points = np.random.randint(self.config.min_points, self.config.max_points + 1)

        # Generate random input points
        X = np.random.uniform(-2, 2, (num_points, num_features))

        try:
            # Compute function values
            y = func(X)

            # Check validity
            if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.max(np.abs(y)) > 100:
                return self.generate_data_sample()  # regenerate

            # Pad to maximum number of features (5 dimensions)
            if num_features < self.config.input_dim - 1:  # -1 because the last column is y
                padding = np.zeros((num_points, self.config.input_dim - 1 - num_features))
                X = np.column_stack([X, padding])

            # Combine data points [x1, x2, ..., x5, y]
            data_points = np.column_stack([X, y])

            # Filter tokens in the vocabulary
            expr_tokens_filtered = [token for token in expr_tokens if token in self.vocab]

            if len(expr_tokens_filtered) == 0:
                return self.generate_data_sample()  # regenerate

            return {
                'data_points': data_points.astype(np.float32),
                'expression_tokens': expr_tokens_filtered,
                'num_features': num_features,
                'num_points': num_points
            }

        except Exception as e:
            return self.generate_data_sample()  # regenerate

    def tokenize_expression(self, expression_tokens: List[str]) -> List[int]:
        """
        Convert expression token sequence to ID sequence

        Args:
            expression_tokens: Expression token list

        Returns:
            List[int]: token ID sequence
        """
        token_ids = [self.vocab['<START>']]
        for token in expression_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
        token_ids.append(self.vocab['<END>'])
        return token_ids

    def process_data_points(self, data_points: np.ndarray) -> torch.Tensor:
        """
        Process data points and convert to model input format

        Args:
            data_points: (num_points, input_dim) numpy array

        Returns:
            torch.Tensor: Processed data point tensor
        """
        # Convert to torch tensor
        data_tensor = torch.from_numpy(data_points).float()

        # Pad if there are not enough points
        if data_tensor.shape[0] < self.config.max_points:
            pad_size = self.config.max_points - data_tensor.shape[0]
            padding = torch.zeros(pad_size, data_tensor.shape[1])
            data_tensor = torch.cat([data_tensor, padding], dim=0)
        elif data_tensor.shape[0] > self.config.max_points:
            # Randomly sample if there are too many points
            indices = torch.randperm(data_tensor.shape[0])[:self.config.max_points]
            data_tensor = data_tensor[indices]

        return data_tensor

    def create_conversation_format(self, data_sample: Dict) -> Dict:
        """
        Convert data sample to Qwen2.5-VL conversation format

        Args:
            data_sample: Data sample dictionary

        Returns:
            Dict: Data in conversation format
        """
        # Build conversation
        conversations = [
            {
                "from": "human",
                "value": "<data>\nThis is a sampled set of scientific data points. Please find an expression to fit this data. You only need to generate the preorder traversal of the expression binary tree."
            },
            {
                "from": "gpt",
                "value": f"Okay, the preorder traversal of the expression I got is[{','.join(data_sample['expression_tokens'])}]"
            }
        ]

        return {
            "id": f"symbolic_regression_{np.random.randint(1000000)}",
            "conversations": conversations,
            "data_points": data_sample['data_points'].tolist(),
            "num_features": data_sample['num_features'],
            "num_points": data_sample['num_points']
        }

    def generate_dataset(self, num_samples: int, output_path: str):
        """
        Generate symbolic regression dataset

        Args:
            num_samples: Number of samples
            output_path: Output file path
        """
        dataset = []

        for i in range(num_samples):
            try:
                data_sample = self.generate_data_sample()
                conversation_data = self.create_conversation_format(data_sample)
                dataset.append(conversation_data)

                if (i + 1) % 1000 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples")

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"Dataset saved to {output_path} with {len(dataset)} samples")
        return dataset

# Data point token used to replace image tokens
DATA_TOKEN_INDEX = 151657  # Use a new token index
DEFAULT_DATA_TOKEN = "<data>"

def preprocess_symbolic_regression(
    sources,
    tokenizer,
    data_grid_info: List = [],
) -> Dict:
    """
    Preprocess symbolic regression data, similar to preprocess_qwen_2_visual

    Args:
        sources: Conversation data sources
        tokenizer: Tokenizer
        data_grid_info: Data grid information

    Returns:
        Dict: Preprocessed data
    """
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant specialized in symbolic regression."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    data_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [-100] * len(input_id)  # IGNORE_INDEX

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<data>" in content:
                    # Replace <data> markers with special data tokens
                    parts = content.split("<data>")
                    new_parts = []
                    for j in range(len(parts) - 1):
                        new_parts.append(parts[j])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|data_pad|>" * data_grid_info[data_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        data_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)  # IGNORE_INDEX
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [-100] * 3  # IGNORE_INDEX for special tokens
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    # input_ids and targets may need padding here, but Trainer usually handles it
    # For safety, implement it manually or rely on Trainer DataCollator

    return dict(
        input_ids=input_ids,
        labels=targets,
    )