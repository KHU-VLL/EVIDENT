"""Convert a checkpoint of the internal research code into the EVIDENT format.

Legacy layout:  adapter_model.safetensors, adapter_config.json, non_lora_state_dict.bin
New layout:     adapter_model.safetensors, adapter_config.json, eb_adapter.safetensors, evident_config.json

    python tools/convert_legacy_checkpoint.py --src <legacy checkpoint-776> --dst checkpoints/evident_charades \
        --timestamp_format relative
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

RENAMES = [
    ("base_model.model.model.", "model."),
    (".slot_adapter.attn_transpose_out_proj.", ".eb_adapter.recon_out_proj."),
    (".slot_adapter.", ".eb_adapter."),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--layers", default="0-10")
    parser.add_argument("--bottleneck_dim", type=int, default=896)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--timestamp_format", choices=["relative", "absolute"], required=True,
                        help="Prompt format the checkpoint was trained with (Charades-STA: relative, QVHighlights: absolute).")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    shutil.copy(os.path.join(args.src, "adapter_model.safetensors"), os.path.join(args.dst, "adapter_model.safetensors"))
    with open(os.path.join(args.src, "adapter_config.json")) as f:
        lora_config = json.load(f)
    lora_config["base_model_name_or_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"  # drop the local path
    with open(os.path.join(args.dst, "adapter_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)

    legacy = torch.load(os.path.join(args.src, "non_lora_state_dict.bin"), map_location="cpu", weights_only=True)
    state = {}
    for key, value in legacy.items():
        if ".slot_adapter." not in key:
            raise KeyError(f"Unexpected non-LoRA parameter {key}")
        for old, new in RENAMES:
            key = key.replace(old, new)
        state[key] = value.contiguous()
    save_file(state, os.path.join(args.dst, "eb_adapter.safetensors"))

    lo, hi = (int(x) for x in args.layers.split("-"))
    config = dict(
        layers=list(range(lo, hi + 1)),
        bottleneck_dim=args.bottleneck_dim,
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        tokens_per_frame=64,
        use_e2v_gate=True,
        timestamp_format=args.timestamp_format,
    )
    with open(os.path.join(args.dst, "evident_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Converted {len(state)} EB Adapter tensors -> {args.dst}")


if __name__ == "__main__":
    main()
