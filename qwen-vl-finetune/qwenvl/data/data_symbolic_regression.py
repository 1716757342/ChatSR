"""
符号回归数据处理模块
基于官方data_qwen.py，支持数据点输入而非图像输入
"""

import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

from . import data_list
from ..symbolic_regression.data_processor import (
    SymbolicRegressionDataProcessor, 
    SymbolicRegressionConfig,
    DATA_TOKEN_INDEX,
    DEFAULT_DATA_TOKEN
)

IGNORE_INDEX = -100

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def preprocess_symbolic_regression_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    data_grid_info: List = [],
) -> Dict:
    """
    预处理符号回归数据，替代原始的preprocess_qwen_2_visual
    """
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant specialized in symbolic regression and mathematical expression generation."

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
        target += [IGNORE_INDEX] * len(input_id)

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
                    # 替换<data>标记为特殊的数据token
                    parts = content.split("<data>")
                    new_parts = []
                    for j in range(len(parts) - 1):
                        new_parts.append(parts[j])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|vision_pad|>" * data_grid_info[data_replicate_index]
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
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class SymbolicRegressionDataset(Dataset):
    """符号回归数据集，基于官方LazySupervisedDataset"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(SymbolicRegressionDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading symbolic regression datasets: {dataset_list}")
        
        # 符号回归配置
        self.sr_config = SymbolicRegressionConfig()
        self.sr_processor = SymbolicRegressionDataProcessor(self.sr_config)
        
        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total symbolic regression training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)

        rank0_print("Formatting symbolic regression inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def process_data_points(self, data_points_list):
        """
        处理数据点，类似于process_image_unified
        
        Args:
            data_points_list: 数据点列表
            
        Returns:
            torch.Tensor: 处理后的数据点张量
        """
        # 转换为numpy数组
        data_points = np.array(data_points_list, dtype=np.float32)
        
        # 使用处理器处理数据点
        processed_data = self.sr_processor.process_data_points(data_points)
        
        return processed_data

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 获取数据样本
        data_sample = self.list_data_dict[i]
        
        # 提取对话和数据点
        conversations = data_sample.get("conversations", [])
        data_points = data_sample.get("data_points", [])
        # --- 关键检查：添加以下代码 ---
        temp_tensor = torch.tensor(data_points, dtype=torch.float32)
        if torch.isnan(temp_tensor).any() or torch.isinf(temp_tensor).any():
            print(f"🚨 WARNING: Invalid values (NaN or Inf) found in sample ID: {data_sample.get('id')}")
            # 您可以选择跳过这个样本或直接报错
            # 为了调试，可以先打印出来看看是哪些数据有问题
        # --- 检查结束 ---
        # 处理数据点
        if data_points:
            processed_data_points = self.process_data_points(data_points)
            data_grid_thw = torch.tensor([[1, self.sr_config.pooling_outputs, 1]], dtype=torch.long)
        else:
            processed_data_points = None
            data_grid_thw = None
        
        # 预处理对话
        data_dict = preprocess_symbolic_regression_qwen(
            [conversations], 
            self.tokenizer,
            data_grid_info=[self.sr_config.pooling_outputs] if data_points else []
        )
        
        if isinstance(i, int):
            data_dict = {
                key: value[0] if isinstance(value, torch.Tensor) and value.dim() > 1 else value
                for key, value in data_dict.items()
            }

        # 添加数据点信息
        if processed_data_points is not None:
            data_dict["data_points"] = processed_data_points
            data_dict["data_grid_thw"] = data_grid_thw
        
        # 添加位置ID（简化版本）
        seq_len = data_dict["input_ids"].shape[0]
        data_dict["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        
        return data_dict

def pad_and_cat_data(tensor_list):
    """padding和连接数据点张量"""
    if not tensor_list or tensor_list[0] is None:
        return None
    
    max_points = max(t.shape[0] for t in tensor_list)
    max_features = max(t.shape[1] for t in tensor_list)
    
    padded_tensors = []
    for tensor in tensor_list:
        pad_points = max_points - tensor.shape[0]
        pad_features = max_features - tensor.shape[1]
        
        if pad_points > 0 or pad_features > 0:
            padding = torch.zeros(max_points, max_features, dtype=tensor.dtype)
            padding[:tensor.shape[0], :tensor.shape[1]] = tensor
            padded_tensors.append(padding)
        else:
            padded_tensors.append(tensor)
    
    return torch.stack(padded_tensors)

@dataclass
class SymbolicRegressionDataCollator(object):
    """符号回归数据整理器"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) if ids.dim() > 1 else ids for ids in input_ids]
        labels = [ids.squeeze(0) if ids.dim() > 1 else ids for ids in labels]
        position_ids = [ids.squeeze(0) if ids.dim() > 1 else ids for ids in position_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # 处理位置ID
        position_ids = torch.nn.utils.rnn.pad_sequence(
            position_ids, batch_first=True, padding_value=0
        )
        
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids,
        )
        
        # 处理数据点
        data_points = [
            instance["data_points"] 
            for instance in instances 
            if "data_points" in instance
        ]
        data_grid_thw = [
            instance["data_grid_thw"]
            for instance in instances
            if "data_grid_thw" in instance
        ]
        
        if len(data_points) != 0:
            concat_data_points = pad_and_cat_data(data_points)
            concat_grid_thw = torch.cat(data_grid_thw, dim=0)
        else:
            concat_data_points = None
            concat_grid_thw = None

        batch["data_points"] = concat_data_points
        batch["data_grid_thw"] = concat_grid_thw
        
        return batch

def make_symbolic_regression_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """创建符号回归数据模块"""
    train_dataset = SymbolicRegressionDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = SymbolicRegressionDataCollator(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=None, 
        data_collator=data_collator
    ) 