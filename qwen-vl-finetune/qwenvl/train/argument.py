# import transformers
# from dataclasses import dataclass, field
# from typing import Dict, Optional, Sequence, List


# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
#     tune_mm_llm: bool = field(default=False)
#     tune_mm_mlp: bool = field(default=False)
#     tune_mm_vision: bool = field(default=False)

# @dataclass
# class DataArguments:
#     dataset_use: str = field(default="")
#     video_max_frames: Optional[int] = field(default=8)
#     video_min_frames: Optional[int] = field(default=4)
#     data_flatten: bool = field(default=False)
#     data_packing: bool = field(default=False)
#     base_interval: int = field(default=2)
#     max_pixels: int = field(default=28 * 28 * 576)
#     min_pixels: int = field(default=28 * 28 * 16)
#     video_max_frame_pixels: int = field(default=32 * 28 * 28)
#     video_min_frame_pixels: int = field(default=4 * 28 * 28)


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     model_max_length: int = field(
#         default=512,
#         metadata={
#             "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )
#     mm_projector_lr: Optional[float] = None
#     vision_tower_lr: Optional[float] = None


import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    # --- Fix: add LoRA-related parameters ---
    lora_enable: bool = field(default=False, metadata={"help": "Enable LoRA fine-tuning."})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma separated list of modules to target for LoRA."}
    )
    # --- Existing parameters ---
    tune_mm_mlp: bool = field(default=False, metadata={"help": "Tune the multi-modal projector."})
    tune_mm_vision: bool = field(default=False, metadata={"help": "Tune the vision tower."})
    tune_mm_llm: bool = field(default=False, metadata={"help": "Tune the LLM."})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    dataset_use: str = field(default="SYMBOLIC_REGRESSION", metadata={"help": "Dataset to use."})
    lazy_preprocess: bool = False
    model_type: str = field(default="qwen2.5vl", metadata={"help": "qwen2vl or qwen2.5vl"})
    data_flatten: bool = field(default=False, metadata={"help": "data flatten"})
    data_packing: bool = field(default=False, metadata={"help": "data packing"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)

