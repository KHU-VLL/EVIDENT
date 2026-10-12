"""Uniform frame sampling, matching qwen-vl-utils' `fetch_video` with a fixed frame count."""
import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

FRAME_FACTOR = 2  # Qwen2.5-VL samples an even number of frames
PATCH_FACTOR = 28  # patch size 14 x spatial merge 2


@dataclass
class VideoClip:
    frames: torch.Tensor      # [T, 3, H, W] float in [0, 255]
    timestamps: np.ndarray    # [T] seconds
    duration: float           # seconds (total_frames / fps)


def _num_frames(requested: int, total_frames: int) -> int:
    n = round(requested / FRAME_FACTOR) * FRAME_FACTOR
    if n > total_frames:
        n = math.floor(total_frames / FRAME_FACTOR) * FRAME_FACTOR
    if not FRAME_FACTOR <= n <= total_frames:
        raise ValueError(f"Cannot sample {requested} frames from a video with {total_frames} frames.")
    return n


def _read_decord(path: str, num_frames: int):
    import decord
    vr = decord.VideoReader(path)
    total, fps = len(vr), vr.get_avg_fps()
    idx = torch.linspace(0, total - 1, _num_frames(num_frames, total)).round().long().tolist()
    frames = torch.tensor(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
    return frames, idx, total, fps


def _read_torchvision(path: str, num_frames: int):
    from torchvision import io
    video, _, info = io.read_video(path, pts_unit="sec", output_format="TCHW")
    total, fps = video.size(0), info["video_fps"]
    idx = torch.linspace(0, total - 1, _num_frames(num_frames, total)).round().long()
    return video[idx], idx.tolist(), total, fps


def load_video(path: str, num_frames: int, size: int = 224) -> VideoClip:
    try:
        frames, idx, total, fps = _read_decord(path, num_frames)
    except Exception as e:
        logger.warning(f"decord failed on {path} ({e}); falling back to torchvision.")
        frames, idx, total, fps = _read_torchvision(path, num_frames)

    size = max(PATCH_FACTOR, round(size / PATCH_FACTOR) * PATCH_FACTOR)
    frames = TF.resize(frames, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True).float()

    duration = total / fps
    timestamps = np.array(idx) / total * duration
    return VideoClip(frames=frames, timestamps=timestamps, duration=duration)
