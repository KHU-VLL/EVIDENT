"""Build an EVIDENT model for inference from a checkpoint.

A checkpoint is a local directory or a folder of a Hugging Face Hub repository, e.g.
`load_evident("mreraser/EVIDENT", subfolder="charades")`. It contains:
    adapter_config.json / adapter_model.safetensors   LoRA weights (PEFT)
    eb_adapter.safetensors                            EB Adapter weights
    evident_config.json                               EB Adapter hyper-parameters
"""
import json
import os
from typing import Optional

import torch
from safetensors.torch import load_file
from transformers import Qwen2_5_VLForConditionalGeneration

from .inject import inject_eb_adapters, load_eb_adapter_state_dict

EB_WEIGHTS_NAME = "eb_adapter.safetensors"
EB_CONFIG_NAME = "evident_config.json"


def resolve_checkpoint(checkpoint: str, subfolder: Optional[str] = None) -> str:
    """Local directory of a checkpoint, downloading it from the Hub if `checkpoint` is a repo id."""
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, subfolder) if subfolder else checkpoint
    from huggingface_hub import snapshot_download
    local = snapshot_download(checkpoint, allow_patterns=[f"{subfolder}/*"] if subfolder else None)
    return os.path.join(local, subfolder) if subfolder else local


def load_evident_config(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, EB_CONFIG_NAME)) as f:
        config = json.load(f)
    config.setdefault("timestamp_format", "relative")
    return config


def load_evident(
    checkpoint: str,
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    subfolder: Optional[str] = None,
):
    checkpoint_dir = resolve_checkpoint(checkpoint, subfolder)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=dtype, device_map={"": device}, attn_implementation=attn_implementation
    )

    # LoRA only touches layers outside the EB Adapter layers, so merging it before
    # inserting the adapters is equivalent to the training-time composition.
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir).merge_and_unload()

    cfg = load_evident_config(checkpoint_dir)
    inject_eb_adapters(
        model,
        layer_indices=cfg["layers"],
        bottleneck_dim=cfg["bottleneck_dim"],
        num_slots=cfg["num_slots"],
        num_iterations=cfg["num_iterations"],
        tokens_per_frame=cfg["tokens_per_frame"],
        use_e2v_gate=cfg["use_e2v_gate"],
    )
    model.to(device=device, dtype=dtype)
    load_eb_adapter_state_dict(model, load_file(os.path.join(checkpoint_dir, EB_WEIGHTS_NAME)))
    return model.eval()
