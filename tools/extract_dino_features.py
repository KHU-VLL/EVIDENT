"""Extract DINOv2 patch features used as the teacher of the EBD loss.

Output: <output_dir>/<video_name>.pt, a float16 tensor [num_frames, 256, 1024]
(DINOv2 ViT-L/14 at 224x224 -> 16x16 patches). The loss pools them 2x2 to the
8x8 visual-token grid of the EB Adapter.

Use the same number of frames as training (20 for Charades-STA, 60 for QVHighlights).
Shard across GPUs with --num_shards / --shard:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python tools/extract_dino_features.py \
          --annotation data/annotations/charades_train.json --video_dir $CHARADES \
          --output_dir $FEATS --num_frames 20 --num_shards 4 --shard $i &
    done; wait
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import decord  # must come after torch: importing decord first segfaults at CUDA init

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.annotation) as f:
        videos = sorted({s["video"] for s in json.load(f)})[args.shard::args.num_shards]

    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").cuda().eval()
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    for video in tqdm(videos, desc=f"shard {args.shard}"):
        out_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(video))[0] + ".pt")
        if os.path.exists(out_path):
            continue
        try:
            vr = decord.VideoReader(os.path.join(args.video_dir, video))
            idx = np.linspace(0, len(vr) - 1, args.num_frames, dtype=np.int32)
            frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).cuda()
        except Exception as e:
            print(f"[skip] {video}: {e}")
            continue

        x = (frames.float() / 255.0 - mean) / std
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feats = encoder.forward_features(x)["x_norm_patchtokens"]  # [T, 256, 1024]
        torch.save(feats.cpu().half(), out_path)


if __name__ == "__main__":
    main()
