from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments as HFTrainingArguments


def parse_layers(spec: str) -> List[int]:
    """"0-10" -> [0, ..., 10]; "0,2,4" -> [0, 2, 4]."""
    layers = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            layers.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            layers.append(int(part))
    return layers


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    attn_implementation: str = field(default="flash_attention_2")


@dataclass
class DataArguments:
    train_annotation: str = field(metadata={"help": "Training annotation json (with key_concepts)."})
    video_dir: str = field(metadata={"help": "Directory with the videos referenced by the annotation."})
    dino_feature_dir: Optional[str] = field(
        default=None, metadata={"help": "DINOv2 patch features (tools/extract_dino_features.py) for the EBD loss."}
    )
    num_frames: int = field(default=20)
    timestamp_format: str = field(
        default="relative",
        metadata={"help": "'relative' (integer bins in [0, 100]; Charades-STA) or 'absolute' (integer seconds; QVHighlights)."},
    )
    frame_size: int = field(default=224, metadata={"help": "224 -> 8x8 = 64 visual tokens per frame."})


@dataclass
class TrainingArguments(HFTrainingArguments):
    # EB Adapter
    adapter_layers: str = field(default="0-10")
    bottleneck_dim: int = field(default=896)
    num_slots: int = field(default=4)
    num_iterations: int = field(default=3)
    tokens_per_frame: int = field(default=64)
    use_e2v_gate: bool = field(default=True)
    ebd_loss_weight: float = field(default=0.1, metadata={"help": "lambda of the EBD loss; 0 disables it."})

    # LoRA on the decoder layers without an EB Adapter
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)

    lr_schedule_epochs: Optional[float] = field(
        default=None,
        metadata={
            "help": "Build the warmup/cosine schedule as if training for this many epochs while stopping at "
                    "num_train_epochs. The paper uses a 5-epoch schedule stopped after 2 epochs."
        },
    )
    remove_unused_columns: bool = field(default=False)
