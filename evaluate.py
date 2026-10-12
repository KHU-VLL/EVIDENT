"""Evaluate an EVIDENT checkpoint on a temporal grounding benchmark.

    torchrun --nproc_per_node=8 evaluate.py \
        --checkpoint checkpoints/evident_charades \
        --annotation data/annotations/qvhighlights_test.json \
        --video_dir /path/to/qvhighlights/videos --num_frames 60 \
        --output results/cha2qvh.json

Metrics follow the MR R1@IoU protocol: the first predicted window is scored against
the best-matching ground-truth window.
"""
import argparse
import json
import logging
import os
import re
from ast import literal_eval

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoProcessor

from evident.data import (
    build_prompt,
    find_concept_tokens,
    format_timestamps,
    instruction_from_sample,
    load_video,
    parse_key_concepts,
    to_seconds,
)
from evident.model import load_evident, load_evident_config, resolve_checkpoint, set_eb_context

logger = logging.getLogger(__name__)
WINDOW_PATTERN = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")


def iou(pred, gt):
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def predict(model, processor, sample, video_dir, num_frames, timestamp_format="relative", max_new_tokens=512):
    clip = load_video(os.path.join(video_dir, sample["video"]), num_frames)
    subject, obj = parse_key_concepts(sample.get("key_concepts", []))
    timestamps = format_timestamps(clip.timestamps, clip.duration, timestamp_format)
    prompt = build_prompt(timestamps, subject, obj, instruction_from_sample(sample))
    inputs = processor(text=[prompt], images=clip.frames, padding=True, do_resize=False, return_tensors="pt").to(model.device)

    ids = inputs["input_ids"][0].tolist()
    subject_idx = find_concept_tokens(ids, processor.tokenizer, "Subject", subject)
    object_idx = find_concept_tokens(ids, processor.tokenizer, "Object", obj)
    set_eb_context(
        model,
        subject_token_indices=[subject_idx],
        object_token_indices=[object_idx],
        concept_valid_mask=torch.tensor([[bool(subject_idx), bool(object_idx)]]),
    )

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    windows = [
        [to_seconds(float(s), clip.duration, timestamp_format), to_seconds(float(e), clip.duration, timestamp_format)]
        for s, e in WINDOW_PATTERN.findall(output.strip())
    ]
    return output, windows


def compute_metrics(results, thresholds=(0.3, 0.5, 0.7)):
    ious = []
    for r in results:
        if not r["predictions"] or not r["gt_windows"]:
            ious.append(0.0)
            continue
        ious.append(max(iou(r["predictions"][0], gt) for gt in r["gt_windows"]))
    ious = np.array(ious)
    metrics = {f"R1@{t}": float((ious >= t).mean() * 100) for t in thresholds}
    metrics["mIoU"] = float(ious.mean() * 100)
    metrics["num_samples"] = len(results)
    metrics["num_invalid"] = sum(1 for r in results if not r["predictions"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or Hugging Face repo id.")
    parser.add_argument("--subfolder", default=None, help="Folder inside the repo, e.g. charades or qvhighlights.")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()

    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world_size = 0, 1
    device = f"cuda:{torch.cuda.current_device()}"

    checkpoint = resolve_checkpoint(args.checkpoint, args.subfolder)
    config = load_evident_config(checkpoint)
    model = load_evident(checkpoint, args.base_model, device=device, attn_implementation=args.attn_implementation)
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)

    with open(args.annotation) as f:
        samples = json.load(f)

    results = []
    for i in tqdm(range(rank, len(samples), world_size), disable=rank != 0):
        sample = samples[i]
        gt = literal_eval(next(c["value"] for c in sample["conversations"] if c["from"] == "gpt"))
        try:
            output, windows = predict(
                model, processor, sample, args.video_dir, args.num_frames, config["timestamp_format"]
            )
        except Exception as e:
            logger.warning(f"Sample {i} ({sample['video']}) failed: {e}")
            output, windows = f"ERROR: {e}", []
        results.append(dict(index=i, video=sample["video"], output=output, predictions=windows, gt_windows=gt))

    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, results, group=dist.new_group(backend="gloo"))
        results = sorted((r for part in gathered for r in part), key=lambda r: r["index"])

    if rank == 0:
        metrics = compute_metrics(results)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(json.dumps(metrics, indent=2))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
