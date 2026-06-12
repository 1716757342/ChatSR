#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

# =====================
# Change paths directly here
# =====================
MODEL_PATH = "/path/to/ChatSR/checkpoints/symbolic-regression-qwen-positionids-fix/final_model/final_model"
JSON_PATH = "/path/to/ChatSR/symbolic_regression_data_20/test.json"
DEVICE = "cuda"
TORCH_DTYPE = "float32"  # float32 / float16 / bfloat16
MAX_NEW_TOKENS = 100
SEED = 42
AUTO_DATASET_NAME = "__INTERACTIVE_JSON__"

DEFAULT_PROMPT = "<data>\nPlease derive the fitting expression for this data and return its preorder traversal."

project_root = Path(__file__).parent
qwen_finetune_path = project_root / "qwen-vl-finetune"
if str(qwen_finetune_path) not in sys.path:
    sys.path.append(str(qwen_finetune_path))
    print(f"Added '{qwen_finetune_path}' to sys.path")

try:
    from qwenvl.symbolic_regression.model import SymbolicRegressionQwenModel
    from qwenvl.symbolic_regression.data_processor import SymbolicRegressionConfig, SymbolicRegressionDataProcessor
    from qwenvl.data import data_dict as DATASET_REGISTRY
    from qwenvl.data.data_symbolic_regression import (
        preprocess_symbolic_regression_qwen,
        SymbolicRegressionDataCollator,
        make_symbolic_regression_data_module,
    )
except ImportError as e:
    print("❌ Unable to import symbolic regression module.")
    print(f"   Please confirm the path exists: {qwen_finetune_path}")
    print(f"   Original error: {e}")
    sys.exit(1)


def select_device():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is unavailable; switching to CPU.")
        return torch.device("cpu")
    return torch.device(DEVICE)


def load_json_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("The JSON top level must be a list.")
    return data


def build_samples_from_dataset_module(json_path, tokenizer):
    DATASET_REGISTRY[AUTO_DATASET_NAME] = {
        "annotation_path": json_path,
        "data_path": "",
    }
    data_args = SimpleNamespace(dataset_use=f"{AUTO_DATASET_NAME}%100")
    data_module = make_symbolic_regression_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    samples = []
    for idx in range(len(train_dataset)):
        raw = train_dataset.list_data_dict[idx]
        samples.append(
            {
                "id": raw.get("id", f"sample_{idx}"),
                "raw": raw,
                "instance": train_dataset[idx],
                "expected_response": get_expected_response(raw),
                "source_index": idx,
            }
        )
    return samples, data_module["data_collator"]


def get_expected_response(sample):
    return next((conv["value"] for conv in sample.get("conversations", []) if conv.get("from") == "gpt"), "")


def get_human_prompt(sample):
    return next((conv["value"] for conv in sample.get("conversations", []) if conv.get("from") == "human"), DEFAULT_PROMPT)


def build_instance_from_json_sample(sample, tokenizer, sr_config, sr_processor):
    conversations = sample.get("conversations", [])
    data_points = sample.get("data_points")
    if data_points is None:
        raise ValueError(f"Sample {sample.get('id', 'unknown')} is missing data_points。")
    if not conversations:
        conversations = [
            {"from": "human", "value": DEFAULT_PROMPT},
            {"from": "gpt", "value": get_expected_response(sample)},
        ]

    processed_data_points = sr_processor.process_data_points(np.array(data_points, dtype=np.float32))
    instance = preprocess_symbolic_regression_qwen(
        [conversations],
        tokenizer,
        data_grid_info=[sr_config.pooling_outputs],
    )
    instance = {
        key: value[0] if isinstance(value, torch.Tensor) and value.dim() > 1 else value
        for key, value in instance.items()
    }
    instance["data_points"] = processed_data_points
    instance["data_grid_thw"] = torch.tensor([[1, sr_config.pooling_outputs, 1]], dtype=torch.long)
    instance["position_ids"] = torch.arange(instance["input_ids"].shape[0], dtype=torch.long)
    return instance


def build_samples(raw_samples, tokenizer, sr_config, sr_processor):
    samples = []
    for idx, raw in enumerate(raw_samples):
        sample_id = raw.get("id", f"sample_{idx}")
        try:
            instance = build_instance_from_json_sample(raw, tokenizer, sr_config, sr_processor)
        except Exception as e:
            print(f"⚠️ Skipping sample {sample_id}: {e}")
            continue
        samples.append(
            {
                "id": sample_id,
                "raw": raw,
                "instance": instance,
                "expected_response": get_expected_response(raw),
                "source_index": idx,
            }
        )
    return samples


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


def batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_loss(model, data_collator, sample, device, bf16=False):
    batch = batch_to_device(data_collator([sample["instance"]]), device)
    if bf16 and device.type == "cuda":
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch, use_cache=False)
    else:
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
    return float(outputs.loss.detach().cpu())


def compute_raw_instance_loss(model, sample, tokenizer, device):
    instance = sample["instance"]
    input_ids = instance["input_ids"]
    labels = instance["labels"]
    if input_ids.dim() > 1:
        input_ids = input_ids.squeeze(0)
    if labels.dim() > 1:
        labels = labels.squeeze(0)
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    data_points = instance["data_points"].unsqueeze(0).to(device)
    data_grid_thw = instance["data_grid_thw"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else torch.ones_like(input_ids),
            position_ids=torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0),
            data_points=data_points,
            data_grid_thw=data_grid_thw,
            labels=labels,
            use_cache=False,
        )
    return float(outputs.loss.detach().cpu())


def evaluate_all(model, data_collator, samples, device, bf16=False):
    rows = []
    for sample in samples:
        loss = compute_loss(model, data_collator, sample, device, bf16=bf16)
        rows.append((sample["id"], loss, sample["expected_response"]))
        suffix = " bf16" if bf16 else ""
        print(f"   {sample['id']}: {loss:.6f}{suffix} | {sample['expected_response'][:100]}")

    if not rows:
        print("❌ No evaluable samples.")
        return

    avg = sum(loss for _, loss, _ in rows) / len(rows)
    print("\n" + "=" * 60)
    print(f"📊 Number of samples: {len(rows)}")
    print(f"📊 Average teacher-forcing loss: {avg:.6f}")
    print("📊 Loss from high to low:")
    for sample_id, loss, expected in sorted(rows, key=lambda item: item[1], reverse=True):
        print(f"  {sample_id}: {loss:.6f} | {expected[:100]}")
    print("=" * 60 + "\n")


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


def greedy_generate_with_inputs(model, input_ids, attention_mask, data_points_tensor, data_grid_thw, tokenizer):
    generated = input_ids.clone()
    for _ in range(MAX_NEW_TOKENS):
        position_ids = torch.arange(generated.shape[1], dtype=torch.long, device=generated.device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=generated.ne(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else torch.ones_like(generated),
                position_ids=position_ids,
                data_points=data_points_tensor,
                data_grid_thw=data_grid_thw,
                use_cache=False,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return generated


def generate_with_inputs(model, tokenizer, sample, input_ids, attention_mask, data_points_tensor, data_grid_thw):
    outputs = greedy_generate_with_inputs(
        model,
        input_ids,
        attention_mask,
        data_points_tensor,
        data_grid_thw,
        tokenizer,
    )
    generated_ids = outputs[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    response_raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    tokens = tokenizer.convert_ids_to_tokens(generated_ids[:120].tolist())

    print("\n" + "=" * 60)
    print(f"Sample: {sample['id']}")
    if sample["expected_response"]:
        print(f"Training/test target: {sample['expected_response']}")
    print("\nPrediction result skip_special_tokens=True:")
    print(response)
    print("\nRaw decoding skip_special_tokens=False:")
    print(response_raw)
    print("\nFirst 120 generated tokens:")
    print(tokens)
    print("=" * 60 + "\n")


def generate_for_sample(model, tokenizer, sample, sr_config, sr_processor, device, prompt):
    input_ids, attention_mask, data_points_tensor, data_grid_thw = prepare_generation_inputs(
        sample, tokenizer, sr_config, sr_processor, device, prompt
    )
    print(f"Prompt: {prompt}")
    generate_with_inputs(model, tokenizer, sample, input_ids, attention_mask, data_points_tensor, data_grid_thw)


def generate_from_training_prefix(model, tokenizer, sample, device):
    instance = sample["instance"]
    input_ids = instance["input_ids"]
    labels = instance["labels"]
    if input_ids.dim() > 1:
        input_ids = input_ids.squeeze(0)
    if labels.dim() > 1:
        labels = labels.squeeze(0)

    label_positions = (labels != -100).nonzero(as_tuple=True)[0]
    if label_positions.numel() == 0:
        raise ValueError(f"Sample {sample['id']} has no assistant labels available for generation.")

    prefix_len = int(label_positions[0].item())
    prefix_ids = input_ids[:prefix_len].unsqueeze(0).to(device)
    attention_mask = prefix_ids.ne(tokenizer.pad_token_id).to(device) if tokenizer.pad_token_id is not None else torch.ones_like(prefix_ids, device=device)
    data_points_tensor = instance["data_points"].unsqueeze(0).to(device)
    data_grid_thw = instance["data_grid_thw"].to(device)

    print("✅ Generating with the real prefix from the teacher-forcing sample.")
    print(f"prefix_len={prefix_len}, target_first_token={tokenizer.decode(input_ids[prefix_len:prefix_len + 1])!r}")
    generate_with_inputs(model, tokenizer, sample, prefix_ids, attention_mask, data_points_tensor, data_grid_thw)


def debug_next_token_logits(model, tokenizer, sample, device):
    instance = sample["instance"]
    input_ids = instance["input_ids"]
    labels = instance["labels"]
    if input_ids.dim() > 1:
        input_ids = input_ids.squeeze(0)
    if labels.dim() > 1:
        labels = labels.squeeze(0)

    label_positions = (labels != -100).nonzero(as_tuple=True)[0]
    if label_positions.numel() == 0:
        print(f"❌ Sample {sample['id']} has no assistant labels.")
        return

    first_label_pos = int(label_positions[0].item())
    target_id = int(input_ids[first_label_pos].item())
    prefix_ids = input_ids[:first_label_pos].unsqueeze(0).to(device)
    full_ids = input_ids.unsqueeze(0).to(device)
    data_points_tensor = instance["data_points"].unsqueeze(0).to(device)
    data_grid_thw = instance["data_grid_thw"].to(device)

    with torch.no_grad():
        prefix_outputs = model(
            input_ids=prefix_ids,
            attention_mask=prefix_ids.ne(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else torch.ones_like(prefix_ids),
            position_ids=torch.arange(prefix_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0),
            data_points=data_points_tensor,
            data_grid_thw=data_grid_thw,
            use_cache=False,
        )
        full_outputs = model(
            input_ids=full_ids,
            attention_mask=full_ids.ne(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else torch.ones_like(full_ids),
            position_ids=torch.arange(full_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0),
            data_points=data_points_tensor,
            data_grid_thw=data_grid_thw,
            labels=labels.unsqueeze(0).to(device),
            use_cache=False,
        )

    prefix_logits = prefix_outputs.logits[0, -1]
    full_logits = full_outputs.logits[0, first_label_pos - 1]
    prefix_probs = torch.softmax(prefix_logits.float(), dim=-1)
    full_probs = torch.softmax(full_logits.float(), dim=-1)
    prefix_top = torch.topk(prefix_probs, k=10)
    full_top = torch.topk(full_probs, k=10)

    print("\n" + "=" * 60)
    print(f"Sample: {sample['id']}")
    print(f"first_label_pos={first_label_pos}")
    print(f"Target first token id={target_id}, token={tokenizer.convert_ids_to_tokens([target_id])}, text={tokenizer.decode([target_id])!r}")
    print(f"teacher-forcing loss={float(full_outputs.loss.detach().cpu()):.6f}")
    print(f"prefix-only target token probability={float(prefix_probs[target_id].detach().cpu()):.8f}")
    print(f"full-input  target token probability={float(full_probs[target_id].detach().cpu()):.8f}")
    print("prefix-only top10:")
    for prob, token_id in zip(prefix_top.values.tolist(), prefix_top.indices.tolist()):
        print(f"  {prob:.8f} | {token_id} | {tokenizer.convert_ids_to_tokens([token_id])} | {tokenizer.decode([token_id])!r}")
    print("full-input top10 at same prediction position:")
    for prob, token_id in zip(full_top.values.tolist(), full_top.indices.tolist()):
        print(f"  {prob:.8f} | {token_id} | {tokenizer.convert_ids_to_tokens([token_id])} | {tokenizer.decode([token_id])!r}")
    print("=" * 60 + "\n")


def find_sample(samples, sample_id):
    for sample in samples:
        if sample["id"] == sample_id:
            return sample
    return None


def print_help():
    print(
        """
Available commands:
  help                         Show help
  list [n]                     List the first n samples; default 20
  use <sample_id>              Select current sample
  show                         Show current sample information
  prompt                       Manually enter/modify the current prompt
  gen                          Generate with the real prefix from the teacher-forcing sample
  gen <sample_id>              Select a sample and generate with the real teacher-forcing prefix
  gen_prompt                   Rebuild input using the current sample JSON raw prompt and generate
  gen_prompt <sample_id>       Select a sample, rebuild input with the JSON raw prompt, and generate
  gen_manual                   Manually enter a prompt for the current sample and generate
  gen_manual <sample_id>       Select a sample, manually enter a prompt, and generate
  gen_default                  Generate for the current sample using DEFAULT_PROMPT
  debug <sample_id>            Compare first-token logits and teacher-forcing loss
  loss                         Compute teacher-forcing loss for the current sample
  loss <sample_id>             Compute teacher-forcing loss for the specified sample
  all                          Compute teacher-forcing loss for all samples
  target                       Show target answer for current sample
  target <sample_id>           Show target answer for specified sample
  quit / exit                  Exit
""".strip()
    )


def main():
    transformers.set_seed(SEED)
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
    samples, data_collator = build_samples_from_dataset_module(JSON_PATH, tokenizer)

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
            command = input("sr-json> ").strip()
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
        if op == "all":
            evaluate_all(model, data_collator, samples, device)
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
            print(f"✅ Current prompt switched to the human prompt from this sample JSON.")
            continue
        if op == "show":
            if current_sample is None:
                print("❌ There is no current sample.")
                continue
            raw = current_sample["raw"]
            points = raw.get("data_points", [])
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

        if op == "loss":
            collator_loss = compute_loss(model, data_collator, target_sample, device)
            raw_loss = compute_raw_instance_loss(model, target_sample, tokenizer, device)
            print(f"{target_sample['id']}: collator_loss={collator_loss:.6f}, raw_instance_loss={raw_loss:.6f}")
        elif op == "target":
            print(target_sample["expected_response"])
        elif op == "debug":
            debug_next_token_logits(model, tokenizer, target_sample, device)
        elif op == "gen":
            generate_from_training_prefix(model, tokenizer, target_sample, device)
        elif op == "gen_prompt":
            current_prompt = get_human_prompt(target_sample["raw"]) or DEFAULT_PROMPT
            print("✅ Rebuilding input with the human prompt from this sample JSON and generating.")
            generate_for_sample(model, tokenizer, target_sample, sr_config, sr_processor, device, current_prompt)
        elif op == "gen_manual":
            print("Enter a prompt; press Enter directly to use the current prompt.")
            text = input("prompt> ").strip()
            if text:
                current_prompt = text
            generate_for_sample(model, tokenizer, target_sample, sr_config, sr_processor, device, current_prompt)
        elif op == "gen_default":
            current_prompt = DEFAULT_PROMPT
            generate_for_sample(model, tokenizer, target_sample, sr_config, sr_processor, device, current_prompt)
        else:
            print(f"❌ Unknown command: {op}")


if __name__ == "__main__":
    main()
