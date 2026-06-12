# ChatSR: A Scientific Multimodal Large Language Model for Discovering Formulas from Scientific Data

![Status](https://img.shields.io/badge/status-active-brightgreen)
![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/license-MIT-blue)

![](ChatSR.png)
ChatSR 是一符号回归领域的一个多模态大语言模型。项目将Set Transformer作为模型的数据编码器，使模型可以接收并分析科学数据 `[x1, x2, ..., y]` ，并根据 prompt 生成拟合表达式的先序遍历 preorder来描述数据背后的规律。

模型目标输出示例：

```text
Based on the data, the derived formula is: <|math_add|>,<|math_x1|>,<|math_x2|>
```

本项目还提供推理脚本，可以将模型输出的 preorder 恢复成表达式，对表达式中的常数项使用 BFGS 进行优化，并输出拟合 `R2`。

---

## 功能特点

- 支持自动合成符号回归数据。
- 支持数学特殊 token，例如 `<|math_add|>`、`<|math_sin|>`、`<|math_x1|>`、`<|math_C|>`。
- 基于 Qwen2.5-VL 构建符号回归模型。
- 使用 Set Transformer 编码数值数据点。
- 支持 HuggingFace Trainer + FSDP 分布式训练。
- 提供交互式推理和调试脚本。
- 支持 preorder 恢复为数学表达式。
- 支持 BFGS 优化常数项并计算 R²。

---

## 项目结构

```text
.
├── data_gen_vary.py                         # 多 prompt 符号回归数据生成脚本
├── expend_tokens.py                         # 给基础模型添加 <|math_...|> 特殊 token
├── train_symbolic_regression_distributed_fixed.py
│                                             # 推荐使用的分布式训练脚本
├── train_symbolic_regression_fixed.py        # 非分布式训练脚本
├── interactive_inference_json_AAAA.py        # 交互式推理和调试脚本
├── interactive_inference_json_bfgs.py        # 推理 + 表达式恢复 + BFGS + R2
└── qwen-vl-finetune/
    └── qwenvl/
        ├── data/
        │   ├── __init__.py                   # 数据集注册入口
        │   └── data_symbolic_regression.py   # Dataset 和 Collator
        └── symbolic_regression/
            ├── model.py                      # Qwen-SR 模型定义
            └── data_processor.py             # 数据处理和模型配置
```

---

## 环境准备

从新建 conda 环境开始：

```bash
conda create -n chatsr python=3.10 -y
conda activate chatsr
```

进入项目目录：

```bash
cd /home/dataset-local/liyanjie/Qwen-SR-V2-copy-06-05
```

安装依赖：

```bash
pip install numpy scipy torch transformers accelerate peft safetensors
```

其中 `scipy` 用于 BFGS 常数优化。

---

## 数据格式

每条样本为 JSON 格式，示例：

```json
{
  "id": "advanced_sr_final_0",
  "conversations": [
    {
      "from": "human",
      "value": "<data>\nPlease derive the fitting expression for this data and return its preorder traversal."
    },
    {
      "from": "gpt",
      "value": "Based on the data, the derived formula is: <|math_add|>,<|math_x1|>,<|math_x2|>"
    }
  ],
  "data_points": [
    [0.1, -0.2, 0.0, -0.1],
    [0.5, 0.3, 0.0, 0.8]
  ],
  "expression_tokens": ["<|math_add|>", "<|math_x1|>", "<|math_x2|>"],
  "standard_tokens": ["+", "x1", "x2"]
}
```

`data_points` 的格式为：

```text
[x1, x2, ..., y]
```

当前默认配置在：

```text
qwen-vl-finetune/qwenvl/symbolic_regression/data_processor.py
```

关键参数：

```python
Dim = 4
```

这表示每个数据点是：

```text
[x1, x2, x3, y]
```

生成数据时需要满足：

```text
Dim = max_dims + 1
```

因此当 `Dim = 4` 时，数据生成应使用：

```bash
--max_dims 3
```

---

## 快速开始

### 1. 合成数据

使用 `data_gen_vary.py`：

```bash
python data_gen_vary.py \
  --num_samples 1000 \
  --max_length 10 \
  --min_length 5 \
  --output_dir symbolic_regression_data_vary_1000 \
  --max_vars 3 \
  --max_dims 3
```

生成结果：

```text
symbolic_regression_data_vary_1000/
├── train.json
├── val.json
└── test.json
```

### 2. 注册数据集

编辑：

```text
qwen-vl-finetune/qwenvl/data/__init__.py
```

添加数据集配置：

```python
SYMBOLIC_REGRESSION_VARY_1000 = {
    "annotation_path": "/home/dataset-local/liyanjie/Qwen-SR-V2-copy-06-05/symbolic_regression_data_vary_1000/train.json",
    "data_path": "",
}
```

并加入 `data_dict`：

```python
"SYMBOLIC_REGRESSION_VARY_1000": SYMBOLIC_REGRESSION_VARY_1000,
```

训练时通过以下参数选择该数据集：

```bash
--dataset_use SYMBOLIC_REGRESSION_VARY_1000%100
```

其中 `%100` 表示使用 100% 数据。

### 3. 扩展数学特殊 token

如果基础模型还没有 `<|math_...|>` token，需要先执行：

```bash
python expend_tokens.py \
  --model_path /path/to/Qwen2.5-VL-3B-Instruct \
  --output_path /home/dataset-local/liyanjie/Qwen-SR-V2-copy-06-05/Qwen/Qwen2.5-VL-3B-Instruct-expend-token
```

训练时 `--model_name_or_path` 应指向扩展后的模型目录。

### 4. 启动训练

推荐使用多卡 FSDP，直接运行训练脚本：

```bash
bash scripts/train_symbolic_regression_multi_gpu.sh
```

脚本开头包含训练相关配置，例如：

```bash
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct-expend-token"
OUTPUT_DIR="./checkpoints/symbolic-regression-qwen-multi-gpu-20"
DATASETS="SYMBOLIC_REGRESSION_LEXICAL_POINT_20_TEST%100"
CUDA_VISIBLE_DEVICES="0,1,2,3"
```

如需更换模型路径、输出目录、数据集或 GPU，请先修改：

```text
scripts/train_symbolic_regression_multi_gpu.sh
```

训练结束后，最终模型会保存到脚本中 `OUTPUT_DIR` 指定的目录。

### 5. 推理并计算 R²

编辑 `interactive_inference_json_bfgs.py` 顶部路径：

```python
MODEL_PATH = "/home/dataset-local/liyanjie/Qwen-SR-V2-copy-06-05/checkpoints/symbolic-regression-vary-1000/final_model"
JSON_PATH = "/home/dataset-local/liyanjie/Qwen-SR-V2-copy-06-05/symbolic_regression_data_vary_1000/test.json"
```

运行：

```bash
python interactive_inference_json_bfgs.py
```

进入交互界面后：

```text
sr-bfgs> list 20
sr-bfgs> infer advanced_sr_final_18
```

脚本会输出：

- 大模型原始预测；
- 提取到的 preorder；
- 恢复后的表达式；
- BFGS 优化后的常数；
- MSE；
- R²。

---

## 训练注意事项

### position_ids

Dataset 返回的 `input_ids` 和 `labels` 必须是 `[seq]`，不能是 `[1, seq]`。否则 `position_ids` 可能错误地变成长度 1，导致 teacher-forcing loss 假低。

`data_symbolic_regression.py` 中应保留类似逻辑：

```python
if isinstance(i, int):
    data_dict = {
        key: value[0] if isinstance(value, torch.Tensor) and value.dim() > 1 else value
        for key, value in data_dict.items()
    }
```

Collator 中也应确保：

```python
position_ids = [ids.squeeze(0) if ids.dim() > 1 else ids for ids in position_ids]
```

训练后可用 `interactive_inference_json_AAAA.py` 检查：

```text
sr-json> loss <sample_id>
```

正常情况下应满足：

```text
collator_loss ≈ raw_instance_loss
```

### lm_head

训练脚本中关闭了词嵌入共享：

```python
config.tie_word_embeddings = False
```

这是为了让 `lm_head` 中数学 token 对应的输出权重可以独立训练。

---

## 推理脚本

### 调试推理

```bash
python interactive_inference_json_AAAA.py
```

常用命令：

```text
list 20
loss <sample_id>
debug <sample_id>
gen <sample_id>
gen_prompt <sample_id>
target <sample_id>
quit
```

该脚本适合检查 loss、首 token 概率和生成行为。

### 表达式拟合推理

```bash
python interactive_inference_json_bfgs.py
```

常用命令：

```text
list 20
infer <sample_id>
fit <sample_id>
target <sample_id>
quit
```

其中 `infer` 和 `fit` 等价。

---

## 最小测试示例

如果只想快速跑通流程，可以生成 20 条样本：

```bash
python data_gen_vary.py \
  --num_samples 20 \
  --max_length 10 \
  --min_length 5 \
  --output_dir symbolic_regression_data_20 \
  --max_vars 3 \
  --max_dims 3
```

注册：

```python
SYMBOLIC_REGRESSION_LEXICAL_POINT_20 = {
    "annotation_path": "/home/dataset-local/liyanjie/Qwen-SR-V2-copy-06-05/symbolic_regression_data_20/train.json",
    "data_path": "",
}
```

训练时使用：

```bash
--dataset_use SYMBOLIC_REGRESSION_LEXICAL_POINT_20%100
```

---

## 常见问题

### 1. 模型输出普通英文，而不是 `<|math_...|>` token

先检查：

```text
sr-json> loss <sample_id>
sr-json> debug <sample_id>
```

如果 loss 高，说明模型还没学好。如果 loss 低但生成不稳定，可以尝试：

- 增加训练数据；
- 继续训练；
- 统一回答模板；
- 确保 `lm_head` 和 LLM 都参与训练。

### 2. 训练时报 shape mismatch

检查数据维度：

```text
Dim = max_dims + 1
```

例如：

```text
Dim = 4
--max_dims 3
```

### 3. BFGS 推理无法恢复表达式

检查 `skip_special_tokens=False` 的原始输出中是否包含：

```text
<|math_add|>,<|math_x1|>,<|math_C|>
```

如果没有，说明模型没有生成合法 preorder。

---

## License

本项目基于 Qwen2.5-VL 相关组件开发。使用时请遵守原始 Qwen 模型及相关第三方依赖的许可证要求。
