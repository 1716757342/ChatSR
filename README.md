# ChatSR

ChatSR is a symbolic regression project based on Qwen2.5-VL. The project replaces the original vision encoder with a Set Transformer, allowing the model to receive numerical sample points `[x1, x2, ..., y]` as multimodal input and generate the preorder traversal of the fitted expression according to the prompt.

Example target model output:

```text
Based on the data, the derived formula is: <|math_add|>,<|math_x1|>,<|math_x2|>
```

This project also provides inference scripts that can restore model-output preorder into expressions, optimize constants in the expressions with BFGS, and output the fitting `R2`.

---

## Features

- Supports automatic synthesis of symbolic regression data.
- Supports special mathematical tokens, such as `<|math_add|>`, `<|math_sin|>`, `<|math_x1|>`, and `<|math_C|>`.
- Builds a symbolic regression model based on Qwen2.5-VL.
- Uses Set Transformer to encode numerical data points.
- Supports HuggingFace Trainer + FSDP distributed training.
- Provides interactive inference and debugging scripts.
- Supports restoring preorder to mathematical expressions.
- Supports optimizing constants with BFGS and computing R².

---

## Project Structure

```text
.
├── data_gen_vary.py                         # Multi-prompt symbolic regression data generation script
├── expend_tokens.py                         # Add <|math_...|> special tokens to the base model
├── train_symbolic_regression_distributed_fixed.py
│                                             # Recommended distributed training script
├── train_symbolic_regression_fixed.py        # Non-distributed training script
├── interactive_inference_json_AAAA.py        # Interactive inference and debugging script
├── interactive_inference_json_bfgs.py        # Inference + expression restoration + BFGS + R2
└── qwen-vl-finetune/
    └── qwenvl/
        ├── data/
        │   ├── __init__.py                   # Dataset registration entry point
        │   └── data_symbolic_regression.py   # Dataset and Collator
        └── symbolic_regression/
            ├── model.py                      # Qwen-SR model definition
            └── data_processor.py             # Data processing and model configuration
```

---

## Environment Setup

Start from a new conda environment:

```bash
conda create -n chatsr python=3.10 -y
conda activate chatsr
```

Enter the project directory:

```bash
cd /path/to/ChatSR
```

Install dependencies:

```bash
pip install numpy scipy torch transformers accelerate peft safetensors
```

Here, `scipy` is used for BFGS constant optimization.

---

## Data Format

Each sample is in JSON format, for example:

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

The format of `data_points` is:

```text
[x1, x2, ..., y]
```

The current default configuration is in:

```text
qwen-vl-finetune/qwenvl/symbolic_regression/data_processor.py
```

Key parameter:

```python
Dim = 4
```

This means each data point is:

```text
[x1, x2, x3, y]
```

Data generation must satisfy:

```text
Dim = max_dims + 1
```

Therefore, when `Dim = 4`, data generation should use:

```bash
--max_dims 3
```

---

## Quick Start

### 1. Synthesize Data

Use `data_gen_vary.py`:

```bash
python data_gen_vary.py \
  --num_samples 1000 \
  --max_length 10 \
  --min_length 5 \
  --output_dir symbolic_regression_data_vary_1000 \
  --max_vars 3 \
  --max_dims 3
```

Generated results:

```text
symbolic_regression_data_vary_1000/
├── train.json
├── val.json
└── test.json
```

### 2. Register the Dataset

Edit:

```text
qwen-vl-finetune/qwenvl/data/__init__.py
```

Add the dataset configuration:

```python
SYMBOLIC_REGRESSION_VARY_1000 = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_vary_1000/train.json",
    "data_path": "",
}
```

And add it to `data_dict`:

```python
"SYMBOLIC_REGRESSION_VARY_1000": SYMBOLIC_REGRESSION_VARY_1000,
```

Select this dataset during training with the following parameter:

```bash
--dataset_use SYMBOLIC_REGRESSION_VARY_1000%100
```

Here `%100` means using 100% of the data.

### 3. Extend Special Mathematical Tokens

If the base model does not yet have `<|math_...|>` tokens, run this first:

```bash
python expend_tokens.py \
  --model_path /path/to/Qwen2.5-VL-3B-Instruct \
  --output_path /path/to/ChatSR/Qwen/Qwen2.5-VL-3B-Instruct-expend-token
```

During training, `--model_name_or_path` should point to the extended model directory.

### 4. Start Training

Multi-GPU FSDP is recommended. Run the training script directly:

```bash
bash scripts/train_symbolic_regression_multi_gpu.sh
```

The beginning of the script contains training-related configuration, for example:

```bash
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct-expend-token"
OUTPUT_DIR="./checkpoints/symbolic-regression-qwen-multi-gpu-20"
DATASETS="SYMBOLIC_REGRESSION_LEXICAL_POINT_20_TEST%100"
CUDA_VISIBLE_DEVICES="0,1,2,3"
```

To change the model path, output directory, dataset, or GPUs, modify this first:

```text
scripts/train_symbolic_regression_multi_gpu.sh
```

After training ends, the final model will be saved to the directory specified by `OUTPUT_DIR` in the script.

### 5. Run Inference and Compute R²

Edit the paths at the top of `interactive_inference_json_bfgs.py`:

```python
MODEL_PATH = "/path/to/ChatSR/checkpoints/symbolic-regression-vary-1000/final_model"
JSON_PATH = "/path/to/ChatSR/symbolic_regression_data_vary_1000/test.json"
```

Run:

```bash
python interactive_inference_json_bfgs.py
```

After entering the interactive interface:

```text
sr-bfgs> list 20
sr-bfgs> infer advanced_sr_final_18
```

The script will output:

- Raw prediction from the large model;
- Extracted preorder;
- Restored expression;
- Constants optimized by BFGS;
- MSE;
- R².

---

## Training Notes

### position_ids

The `input_ids` and `labels` returned by the Dataset must be `[seq]`, not `[1, seq]`. Otherwise, `position_ids` may incorrectly become length 1, causing the teacher-forcing loss to be falsely low.

Keep similar logic in `data_symbolic_regression.py`:

```python
if isinstance(i, int):
    data_dict = {
        key: value[0] if isinstance(value, torch.Tensor) and value.dim() > 1 else value
        for key, value in data_dict.items()
    }
```

The Collator should also ensure:

```python
position_ids = [ids.squeeze(0) if ids.dim() > 1 else ids for ids in position_ids]
```

After training, use `interactive_inference_json_AAAA.py` to check:

```text
sr-json> loss <sample_id>
```

Normally, this should satisfy:

```text
collator_loss ≈ raw_instance_loss
```

### lm_head

The training script disables word embedding sharing:

```python
config.tie_word_embeddings = False
```

This allows the output weights corresponding to mathematical tokens in `lm_head` to be trained independently.

---

## Inference Scripts

### Debug Inference

```bash
python interactive_inference_json_AAAA.py
```

Common commands:

```text
list 20
loss <sample_id>
debug <sample_id>
gen <sample_id>
gen_prompt <sample_id>
target <sample_id>
quit
```

This script is suitable for checking loss, first-token probabilities, and generation behavior.

### Expression-Fitting Inference

```bash
python interactive_inference_json_bfgs.py
```

Common commands:

```text
list 20
infer <sample_id>
fit <sample_id>
target <sample_id>
quit
```

Here, `infer` and `fit` are equivalent.

---

## Minimal Test Example

If you only want to quickly run through the workflow, generate 20 samples:

```bash
python data_gen_vary.py \
  --num_samples 20 \
  --max_length 10 \
  --min_length 5 \
  --output_dir symbolic_regression_data_20 \
  --max_vars 3 \
  --max_dims 3
```

Register:

```python
SYMBOLIC_REGRESSION_LEXICAL_POINT_20 = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_20/train.json",
    "data_path": "",
}
```

Use during training:

```bash
--dataset_use SYMBOLIC_REGRESSION_LEXICAL_POINT_20%100
```

---

## FAQ

### 1. The model outputs ordinary English instead of `<|math_...|>` tokens

Check first:

```text
sr-json> loss <sample_id>
sr-json> debug <sample_id>
```

If the loss is high, the model has not learned well yet. If the loss is low but generation is unstable, try:

- Increasing the training data;
- Continuing training;
- Standardizing the answer template;
- Ensuring both `lm_head` and the LLM participate in training.

### 2. Shape mismatch occurs during training

Check the data dimensions:

```text
Dim = max_dims + 1
```

For example:

```text
Dim = 4
--max_dims 3
```

### 3. BFGS inference cannot restore the expression

Check whether the raw output with `skip_special_tokens=False` contains:

```text
<|math_add|>,<|math_x1|>,<|math_C|>
```

If not, the model did not generate a valid preorder.

---

## License

This project is developed based on Qwen2.5-VL related components. Please comply with the license requirements of the original Qwen model and related third-party dependencies when using it.
