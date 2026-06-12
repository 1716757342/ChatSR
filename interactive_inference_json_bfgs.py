#!/usr/bin/env python3

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

# =====================
# Change paths directly here
# =====================
MODEL_PATH = "/path/to/ChatSR/checkpoints/symbolic-regression-qwen-positionids-fix/final_model/final_model"
JSON_PATH = "/path/to/ChatSR/symbolic_regression_data_20/test.json"
DEVICE = "cuda"
TORCH_DTYPE = "float32"
MAX_NEW_TOKENS = 120
SEED = 42
AUTO_DATASET_NAME = "__INTERACTIVE_JSON_BFGS__"

DEFAULT_PROMPT = "<data>\nPlease derive the fitting expression for this data and return its preorder traversal."
BFGS_RESTARTS = 8
BFGS_MAXITER = 1000
EPS = 1e-8
CLIP_VALUE = 1e6

project_root = Path(__file__).parent
qwen_finetune_path = project_root / "qwen-vl-finetune"
if str(qwen_finetune_path) not in sys.path:
    sys.path.append(str(qwen_finetune_path))
    print(f"Added '{qwen_finetune_path}' to sys.path")

try:
    from qwenvl.symbolic_regression.model import SymbolicRegressionQwenModel
    from qwenvl.symbolic_regression.data_processor import SymbolicRegressionConfig, SymbolicRegressionDataProcessor
    from qwenvl.data import data_dict as DATASET_REGISTRY
    from qwenvl.data.data_symbolic_regression import make_symbolic_regression_data_module
except ImportError as e:
    print("❌ Unable to import symbolic regression module.")
    print(f"   Please confirm the path exists: {qwen_finetune_path}")
    print(f"   Original error: {e}")
    sys.exit(1)


@dataclass
class ExprNode:
    op: str
    children: tuple = ()
    index: int | None = None


BINARY_OPS = {"add", "sub", "mul", "div", "pow"}
UNARY_OPS = {"sin", "cos", "exp", "log", "sqrt"}
MATH_TOKEN_RE = re.compile(r"<\|math_([A-Za-z0-9_]+)\|>")


def select_device():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is unavailable; switching to CPU.")
        return torch.device("cpu")
    return torch.device(DEVICE)


def get_expected_response(sample):
    return next((conv["value"] for conv in sample.get("conversations", []) if conv.get("from") == "gpt"), "")


def get_human_prompt(sample):
    return next((conv["value"] for conv in sample.get("conversations", []) if conv.get("from") == "human"), DEFAULT_PROMPT)


def build_samples_from_dataset_module(json_path, tokenizer):
    DATASET_REGISTRY[AUTO_DATASET_NAME] = {
        "annotation_path": json_path,
        "data_path": "",
    }
    data_args = SimpleNamespace(dataset_use=f"{AUTO_DATASET_NAME}%100")
    data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["train_dataset"]
    samples = []
    for idx in range(len(dataset)):
        raw = dataset.list_data_dict[idx]
        samples.append(
            {
                "id": raw.get("id", f"sample_{idx}"),
                "raw": raw,
                "instance": dataset[idx],
                "expected_response": get_expected_response(raw),
                "source_index": idx,
            }
        )
    return samples


def find_sample(samples, sample_id):
    for sample in samples:
        if sample["id"] == sample_id:
            return sample
    return None


def print_model_diagnostics(model, tokenizer):
    input_embeddings = model.get_input_embeddings().weight
    lm_head = model.lm_head.weight
    print("\n🔎 Model diagnostics")
    print(f"   tie_word_embeddings: {model.config.tie_word_embeddings}")
    print(f"   lm_head/input_embeddings shared storage: {lm_head.data_ptr() == input_embeddings.data_ptr()}")
    math_ids = [tokenizer.convert_tokens_to_ids(token) for token in ["<|math_add|>", "<|math_x1|>", "<|math_C|>"]]
    math_ids = [token_id for token_id in math_ids if token_id is not None and token_id >= 0]
    if math_ids:
        row_diff = (lm_head.detach()[math_ids].float() - input_embeddings.detach()[math_ids].float()).abs().max().item()
        print(f"   math token rows lm_head/embed max_abs_diff: {row_diff:.8f}")


def prepare_generation_inputs(sample, tokenizer, sr_config, sr_processor, device, prompt):
    raw = sample["raw"]
    data_points = raw.get("data_points")
    if data_points is None:
        raise ValueError(f"Sample {sample['id']} is missing data_points。")

    if "<data>" not in prompt:
        prompt = "<data>\n" + prompt

    processed_data_points = sr_processor.process_data_points(np.array(data_points, dtype=np.float32))
    data_points_tensor = processed_data_points.unsqueeze(0).to(device)
    data_grid_thw = torch.tensor([[1, sr_config.pooling_outputs, 1]], dtype=torch.long, device=device)

    replacement = "<|vision_start|>" + "<|vision_pad|>" * sr_config.pooling_outputs + "<|vision_end|>"
    prompt = prompt.replace("<data>", replacement)

    system_message = "You are a helpful assistant specialized in symbolic regression and mathematical expression generation."
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device) if tokenizer.pad_token_id is not None else torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask, data_points_tensor, data_grid_thw


def greedy_generate_with_inputs(model, input_ids, data_points_tensor, data_grid_thw, tokenizer):
    generated = input_ids.clone()
    for _ in range(MAX_NEW_TOKENS):
        position_ids = torch.arange(generated.shape[1], dtype=torch.long, device=generated.device).unsqueeze(0)
        attention_mask = generated.ne(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else torch.ones_like(generated)
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                position_ids=position_ids,
                data_points=data_points_tensor,
                data_grid_thw=data_grid_thw,
                use_cache=False,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
    return generated


def generate_prediction(model, tokenizer, sample, sr_config, sr_processor, device, prompt):
    input_ids, _, data_points_tensor, data_grid_thw = prepare_generation_inputs(
        sample, tokenizer, sr_config, sr_processor, device, prompt
    )
    outputs = greedy_generate_with_inputs(model, input_ids, data_points_tensor, data_grid_thw, tokenizer)
    generated_ids = outputs[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    response_raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    tokens = tokenizer.convert_ids_to_tokens(generated_ids[:120].tolist())
    return response, response_raw, tokens


def extract_preorder_tokens(text):
    return [match.group(1) for match in MATH_TOKEN_RE.finditer(text)]


def parse_preorder_tokens(tokens):
    const_counter = 0

    def parse_at(pos):
        nonlocal const_counter
        if pos >= len(tokens):
            raise ValueError("Incomplete preorder tokens: expression ended early.")
        token = tokens[pos]
        if token in BINARY_OPS:
            left, next_pos = parse_at(pos + 1)
            right, next_pos = parse_at(next_pos)
            return ExprNode(token, (left, right)), next_pos
        if token in UNARY_OPS:
            child, next_pos = parse_at(pos + 1)
            return ExprNode(token, (child,)), next_pos
        if re.fullmatch(r"x\d+", token):
            return ExprNode("var", index=int(token[1:]) - 1), pos + 1
        if token == "C":
            node = ExprNode("const", index=const_counter)
            const_counter += 1
            return node, pos + 1
        raise ValueError(f"Unknown math token: <|math_{token}|>")

    root, end_pos = parse_at(0)
    unused = tokens[end_pos:]
    return root, const_counter, unused


def safe_array(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=CLIP_VALUE, neginf=-CLIP_VALUE)
    return np.clip(values, -CLIP_VALUE, CLIP_VALUE)


def evaluate_expr(node, x, constants):
    if node.op == "var":
        if node.index is None or node.index < 0 or node.index >= x.shape[1]:
            raise ValueError(f"Expression references a nonexistent variable x{(node.index or 0) + 1}，current X has only {x.shape[1]} dimensions.")
        return x[:, node.index]
    if node.op == "const":
        return np.full(x.shape[0], constants[node.index], dtype=np.float64)

    child_values = [evaluate_expr(child, x, constants) for child in node.children]
    with np.errstate(all="ignore"):
        if node.op == "add":
            result = child_values[0] + child_values[1]
        elif node.op == "sub":
            result = child_values[0] - child_values[1]
        elif node.op == "mul":
            result = child_values[0] * child_values[1]
        elif node.op == "div":
            denom = np.where(np.abs(child_values[1]) < EPS, np.sign(child_values[1]) * EPS + EPS, child_values[1])
            result = child_values[0] / denom
        elif node.op == "pow":
            base = np.clip(np.abs(child_values[0]) + EPS, EPS, CLIP_VALUE)
            exponent = np.clip(child_values[1], -8.0, 8.0)
            result = np.power(base, exponent)
        elif node.op == "sin":
            result = np.sin(child_values[0])
        elif node.op == "cos":
            result = np.cos(child_values[0])
        elif node.op == "exp":
            result = np.exp(np.clip(child_values[0], -50.0, 50.0))
        elif node.op == "log":
            result = np.log(np.abs(child_values[0]) + EPS)
        elif node.op == "sqrt":
            result = np.sqrt(np.abs(child_values[0]) + EPS)
        else:
            raise ValueError(f"Unsupported op: {node.op}")
    return safe_array(result)


def expr_to_string(node, constants=None):
    if node.op == "var":
        return f"x{node.index + 1}"
    if node.op == "const":
        if constants is None:
            return f"C{node.index}"
        return f"{constants[node.index]:.10g}"

    args = [expr_to_string(child, constants) for child in node.children]
    if node.op == "add":
        return f"({args[0]} + {args[1]})"
    if node.op == "sub":
        return f"({args[0]} - {args[1]})"
    if node.op == "mul":
        return f"({args[0]} * {args[1]})"
    if node.op == "div":
        return f"({args[0]} / {args[1]})"
    if node.op == "pow":
        return f"pow({args[0]}, {args[1]})"
    if node.op in UNARY_OPS:
        return f"{node.op}({args[0]})"
    raise ValueError(f"Unsupported op: {node.op}")


def get_xy(sample):
    data_points = np.asarray(sample["raw"].get("data_points"), dtype=np.float64)
    if data_points.ndim != 2 or data_points.shape[1] < 2:
        raise ValueError("data_points must be a 2D array and include at least one independent variable and y.")
    return data_points[:, :-1], data_points[:, -1]


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < EPS:
        return 1.0 if ss_res < EPS else float("-inf")
    return 1.0 - ss_res / ss_tot


def fit_constants_bfgs(root, n_constants, x, y):
    if n_constants == 0:
        y_pred = evaluate_expr(root, x, np.zeros(0, dtype=np.float64))
        return np.zeros(0, dtype=np.float64), y_pred, r2_score(y, y_pred), float(np.mean((y - y_pred) ** 2))
    if minimize is None:
        raise RuntimeError("scipy is not installed in the current Python environment, so scipy.optimize.minimize(method='BFGS') cannot be run.")

    rng = np.random.default_rng(SEED)
    starts = [np.zeros(n_constants), np.ones(n_constants), -np.ones(n_constants)]
    for _ in range(max(0, BFGS_RESTARTS - len(starts))):
        starts.append(rng.normal(loc=0.0, scale=2.0, size=n_constants))

    best_result = None

    def objective(constants):
        try:
            y_pred = evaluate_expr(root, x, constants)
            if not np.all(np.isfinite(y_pred)):
                return 1e30
            residual = y_pred - y
            return float(np.mean(residual ** 2))
        except Exception:
            return 1e30

    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="BFGS",
            options={"maxiter": BFGS_MAXITER, "gtol": 1e-10},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    constants = np.asarray(best_result.x, dtype=np.float64)
    y_pred = evaluate_expr(root, x, constants)
    return constants, y_pred, r2_score(y, y_pred), float(np.mean((y - y_pred) ** 2))


def analyze_prediction(sample, response_raw):
    tokens = extract_preorder_tokens(response_raw)
    if not tokens:
        print("❌ No <|math_...|> preorder tokens were found in the model output; cannot recover expression.")
        return

    print("\n📌 Extracted preorder:")
    print(",".join(f"<|math_{token}|>" for token in tokens))

    try:
        root, n_constants, unused = parse_preorder_tokens(tokens)
    except ValueError as e:
        print(f"❌ Preorder parsing failed: {e}")
        return

    if unused:
        print("⚠️ Unused extra tokens in preorder:")
        print(",".join(f"<|math_{token}|>" for token in unused))

    x, y = get_xy(sample)
    print("\n🧮 Recovered expression structure:")
    print(expr_to_string(root))
    print(f"Number of constants: {n_constants}")

    try:
        constants, y_pred, r2, mse = fit_constants_bfgs(root, n_constants, x, y)
    except Exception as e:
        print(f"❌ BFGS constant optimization failed: {e}")
        return

    print("\n✅ Expression after BFGS optimization:")
    print(expr_to_string(root, constants))
    if n_constants:
        print("Constants:")
        for idx, value in enumerate(constants):
            print(f"  C{idx} = {value:.12g}")
    print(f"MSE: {mse:.12g}")
    print(f"R2: {r2:.12g}")


def run_inference_and_fit(model, tokenizer, sample, sr_config, sr_processor, device, prompt):
    response, response_raw, generated_tokens = generate_prediction(
        model, tokenizer, sample, sr_config, sr_processor, device, prompt
    )

    print("\n" + "=" * 70)
    print(f"Sample: {sample['id']}")
    print(f"Prompt: {prompt}")
    if sample["expected_response"]:
        print(f"Training/test target: {sample['expected_response']}")
    print("\nLLM prediction skip_special_tokens=True:")
    print(response)
    print("\nLLM prediction skip_special_tokens=False:")
    print(response_raw)
    print("\nFirst 120 generated tokens:")
    print(generated_tokens)

    analyze_prediction(sample, response_raw)
    print("=" * 70 + "\n")


def print_help():
    print(
        """
Available commands:
  help                         Show help
  list [n]                     List the first n samples; default 20
  use <sample_id>              Select current sample
  show                         Show current sample information
  prompt                       Manually enter/modify the current prompt
  infer                        Generate preorder + BFGS fit + R2 for current sample
  infer <sample_id>            Generate preorder + BFGS fit + R2 for specified sample
  fit                          Alias for infer
  fit <sample_id>              Alias for infer
  target                       Show target answer for current sample
  target <sample_id>           Show target answer for specified sample
  quit / exit                  Exit
""".strip()
    )


def main():
    transformers.set_seed(SEED)
    np.random.seed(SEED)
    device = select_device()
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model_dtype = dtype_map[TORCH_DTYPE]

    print(f"🤖 Using device: {device}")
    print(f"🔢 Model loading dtype: {model_dtype}")
    print(f"📦 Model path: {MODEL_PATH}")
    print(f"📖 JSON path: {JSON_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        model_max_length=512,
        padding_side="right",
    )
    model = SymbolicRegressionQwenModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    ).to(device).eval()
    model.config.use_cache = False

    sr_config = SymbolicRegressionConfig()
    sr_processor = SymbolicRegressionDataProcessor(sr_config)
    print(f"📊 Internal temporary dataset_use: {AUTO_DATASET_NAME}%100 -> {JSON_PATH}")
    samples = build_samples_from_dataset_module(JSON_PATH, tokenizer)

    print(f"✅ Number of samples: {len(samples)}")
    print_model_diagnostics(model, tokenizer)
    print_help()

    current_sample = samples[0] if samples else None
    current_prompt = get_human_prompt(current_sample["raw"]) if current_sample else DEFAULT_PROMPT
    if current_sample:
        print(f"\nCurrent sample: {current_sample['id']}")
        print("The current prompt uses the human prompt from this sample JSON.")

    while True:
        try:
            command = input("sr-bfgs> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit。")
            break

        if not command:
            continue
        parts = command.split()
        op = parts[0].lower()

        if op in {"quit", "exit", "q"}:
            break
        if op == "help":
            print_help()
            continue
        if op == "list":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            for sample in samples[:n]:
                print(sample["id"])
            continue
        if op == "use":
            if len(parts) < 2:
                print("❌ Usage: use <sample_id>")
                continue
            sample = find_sample(samples, parts[1])
            if sample is None:
                print(f"❌ Sample not found: {parts[1]}")
                continue
            current_sample = sample
            current_prompt = get_human_prompt(current_sample["raw"]) or DEFAULT_PROMPT
            print(f"✅ Current sample: {current_sample['id']}")
            print("✅ Current prompt switched to the human prompt from this sample JSON.")
            continue
        if op == "show":
            if current_sample is None:
                print("❌ There is no current sample.")
                continue
            points = current_sample["raw"].get("data_points", [])
            print(f"ID: {current_sample['id']}")
            print(f"Number/dimension of data points: {len(points)} x {len(points[0]) if points else 0}")
            print(f"Current prompt: {current_prompt}")
            if current_sample["expected_response"]:
                print(f"Target: {current_sample['expected_response']}")
            continue
        if op == "prompt":
            print("Enter a prompt; press Enter directly to use DEFAULT_PROMPT.")
            text = input("prompt> ").strip()
            current_prompt = text if text else DEFAULT_PROMPT
            print(f"✅ Current prompt: {current_prompt}")
            continue

        target_sample = current_sample
        if len(parts) > 1:
            sample = find_sample(samples, parts[1])
            if sample is None:
                print(f"❌ Sample not found: {parts[1]}")
                continue
            target_sample = sample
            current_sample = sample
            current_prompt = get_human_prompt(current_sample["raw"]) or DEFAULT_PROMPT

        if target_sample is None:
            print("❌ There is no current sample.")
            continue

        if op in {"infer", "fit"}:
            run_inference_and_fit(model, tokenizer, target_sample, sr_config, sr_processor, device, current_prompt)
        elif op == "target":
            print(target_sample["expected_response"])
        else:
            print(f"❌ Unknown command: {op}")


if __name__ == "__main__":
    main()
