# ChatSR: A Scientific Multimodal Large Language Model for Discovering Formulas from Scientific Data

![Status](https://img.shields.io/badge/status-active-brightgreen)
![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/license-MIT-blue)

![](ChatSR.png)
ChatSR is a multimodal large language model for symbolic regression. The project uses Set Transformer as the data encoder, enabling the model to receive and analyze scientific data `[x1, x2, ..., y]` and generate the preorder traversal of a fitted expression from the prompt to describe the underlying law behind the data.

Example target model output:

```text
Based on the data, the derived formula is: <|math_add|>,<|math_x1|>,<|math_x2|>
```

This project also provides inference scripts that recover expressions from the model-generated preorder traversal, optimize constants in the expression with BFGS, and report the fitted `R2`.

---

## Features

- The first multimodal large language model for symbolic regression.
- Supports mathematical special tokens such as `<|math_add|>`, `<|math_sin|>`, `<|math_x1|>`, and `<|math_C|>`.
- Uses Set Transformer to encode numerical data points.
- Supports HuggingFace Trainer + FSDP distributed training.
- Provides interactive inference and debugging scripts.
- Supports recovering mathematical expressions from preorder traversals.
- Supports BFGS constant optimization and R² calculation.

---

## Project Structure

```text
.
├── data_gen_vary.py                         # Multi-prompt symbolic regression data generation script
├── expend_tokens.py                         # Add <|math_...|> special tokens to the base model
├── train_symbolic_regression_distributed_fixed.py
│                                             # Recommended distributed training script
├── train_symbolic_regression_fixed.py        # Non-distributed training script
├── interactive_inference_json.py        # Interactive inference and debugging script
├── interactive_inference_json_bfgs.py        # Inference + expression recovery + BFGS + R2
└── qwen-vl-finetune/
    └── qwenvl/
        ├── data/
        │   ├── __init__.py                   # Dataset registry entry point
        │   └── data_symbolic_regression.py   # Dataset and collator
        └── symbolic_regression/
            ├── model.py                      # ChatSR model definition
            └── data_processor.py             # Data processing and model configuration
```

---

## Environment Setup

Start by creating a new conda environment:

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

`scipy` is used for BFGS constant optimization.

---

## Data Format

Each sample is in JSON format. Example:

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

The `data_points` format is:

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

### 1. Generate Synthetic Data

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

Generated output:

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

Then add it to `data_dict`:

```python
"SYMBOLIC_REGRESSION_VARY_1000": SYMBOLIC_REGRESSION_VARY_1000,
```

Select this dataset during training with:

```bash
--dataset_use SYMBOLIC_REGRESSION_VARY_1000%100
```

`%100` means using 100% of the data.

### 3. Extend Mathematical Special Tokens

If the base model does not already contain `<|math_...|>` tokens, run:

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

To change the model path, output directory, dataset, or GPUs, edit:

```text
scripts/train_symbolic_regression_multi_gpu.sh
```

After training, the final model is saved under the directory specified by `OUTPUT_DIR` in the script.

### 5. Run Inference and Calculate R²

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

The script outputs:

- Raw model prediction;
- Extracted preorder traversal;
- Recovered expression;
- Constants optimized by BFGS;
- MSE；
- R²。

---

## Training Notes

### position_ids

The dataset must return `input_ids` and `labels` as `[seq]`, not `[1, seq]`. Otherwise, `position_ids` may incorrectly become length 1, causing a falsely low teacher-forcing loss.

Keep similar logic in `data_symbolic_regression.py`:

```python
if isinstance(i, int):
    data_dict = {
        key: value[0] if isinstance(value, torch.Tensor) and value.dim() > 1 else value
        for key, value in data_dict.items()
    }
```

The collator should also ensure:

```python
position_ids = [ids.squeeze(0) if ids.dim() > 1 else ids for ids in position_ids]
```

After training, use `interactive_inference_json.py` to check:

```text
sr-json> loss <sample_id>
```

Normally, the following should hold:

```text
collator_loss ≈ raw_instance_loss
```

### lm_head

The training script disables tied word embeddings:

```python
config.tie_word_embeddings = False
```

This allows the output weights corresponding to mathematical tokens in `lm_head` to be trained independently.

---

## Inference Scripts

### Debug Inference

```bash
python interactive_inference_json.py
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

This script is useful for checking loss, first-token probabilities, and generation behavior.

### Expression Fitting Inference

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

`infer` and `fit` are equivalent.

---

## Minimal Test Example

To quickly run through the workflow, generate 20 samples:

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

### 1. The model outputs plain English instead of `<|math_...|>` tokens

Check first:

```text
sr-json> loss <sample_id>
sr-json> debug <sample_id>
```

If the loss is high, the model has not learned well yet. If the loss is low but generation is unstable, try:

- Increase the training data;
- Continue training;
- Use a consistent answer template;
- Ensure both `lm_head` and the LLM participate in training.

### 2. Shape mismatch during training

Check the data dimension:

```text
Dim = max_dims + 1
```

For example:

```text
Dim = 4
--max_dims 3
```

### 3. BFGS inference cannot recover the expression

Check whether the raw output with `skip_special_tokens=False` contains:

```text
<|math_add|>,<|math_x1|>,<|math_C|>
```

If not, the model did not generate a valid preorder traversal.

---

## License

This project is developed based on Qwen-related components. Please comply with the licenses of the original Qwen models and related third-party dependencies when using it.
