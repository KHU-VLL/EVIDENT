"""Training dataset and collator for temporal grounding with EVIDENT."""
import json
import logging
import os
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .prompt import (
    INTERVAL_PATTERN,
    build_prompt,
    find_concept_tokens,
    format_answer,
    format_timestamps,
    instruction_from_sample,
    parse_key_concepts,
)
from .video import load_video

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
# GT end times may overshoot the decoded duration slightly due to label rounding.
GT_OVERSHOOT_TOLERANCE = 2.0


class GroundingDataset(Dataset):
    """
    Annotation format (one entry per query):
        {
          "video": "AO8RW.mp4",
          "conversations": [
            {"from": "human", "value": "<video>\\nThe video lasts 33.67 seconds.\\nQuery: ...\\nGiven the video ..."},
            {"from": "gpt", "value": "[[0.0, 6.9]]"}
          ],
          "key_concepts": ["subject: person", "object: book"]
        }
    """

    def __init__(
        self,
        annotation_path: str,
        video_dir: str,
        processor,
        num_frames: int,
        dino_feature_dir: Optional[str] = None,
        frame_size: int = 224,
        timestamp_format: str = "relative",
    ):
        with open(annotation_path) as f:
            self.samples = json.load(f)
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames
        self.dino_feature_dir = dino_feature_dir
        self.frame_size = frame_size
        self.timestamp_format = timestamp_format

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict:
        sample = self.samples[i]
        video_path = os.path.join(self.video_dir, sample["video"])
        try:
            clip = load_video(video_path, self.num_frames, self.frame_size)
        except Exception as e:
            logger.warning(f"Failed to load {video_path}: {e}")
            return self[(i + 1) % len(self)]
        if len(clip.frames) != self.num_frames:
            # The EB Adapter batches frames, so every sample needs the same count.
            logger.warning(f"{video_path}: got {len(clip.frames)} frames, expected {self.num_frames}; skipping.")
            return self[(i + 1) % len(self)]

        answer = next(c["value"] for c in sample["conversations"] if c["from"] == "gpt")
        for start, end in INTERVAL_PATTERN.findall(answer):
            start, end = float(start), float(end)
            if start < 0 or end < start or end > clip.duration + GT_OVERSHOOT_TOLERANCE:
                logger.warning(f"Invalid GT [{start}, {end}] (duration={clip.duration:.2f}) for sample {i}; resampling.")
                j = random.randrange(len(self) - 1)
                return self[j + 1 if j >= i else j]

        subject, obj = parse_key_concepts(sample.get("key_concepts", []))
        prompt = build_prompt(
            format_timestamps(clip.timestamps, clip.duration, self.timestamp_format),
            subject, obj, instruction_from_sample(sample),
        )
        response = format_answer(answer, clip.duration, self.timestamp_format) + "<|im_end|>\n"

        inputs = self.processor(text=[prompt], images=clip.frames, padding=False, do_resize=False, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0]
        response_ids = self.processor.tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

        item = dict(
            input_ids=torch.cat([prompt_ids, response_ids]),
            labels=torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), response_ids]),
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            subject=subject,
            object=obj,
        )
        dino = self._load_dino_features(sample["video"], len(clip.frames))
        if dino is not None:
            item["dino_features"] = dino
        return item

    def _load_dino_features(self, video: str, num_frames: int) -> Optional[torch.Tensor]:
        if self.dino_feature_dir is None:
            return None
        path = os.path.join(self.dino_feature_dir, os.path.splitext(os.path.basename(video))[0] + ".pt")
        if not os.path.exists(path):
            logger.warning(f"Missing DINO features {path}; the EBD loss is skipped for this batch.")
            return None
        feats = torch.load(path, map_location="cpu", weights_only=True)  # [T_dino, P, D]
        if feats.shape[0] != num_frames:
            # Nearest-neighbour resampling to the decoded frames.
            t = feats.shape[0]
            idx = [min(max(round(i * (t - 1) / max(num_frames - 1, 1)), 0), t - 1) for i in range(num_frames)]
            feats = feats[idx]
        return feats.float()


class GroundingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: List[Dict]) -> Dict:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [e["input_ids"] for e in examples], batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [e["labels"] for e in examples], batch_first=True, padding_value=IGNORE_INDEX
        )

        subject_idx, object_idx = [], []
        for e in examples:
            ids = e["input_ids"].tolist()
            subject_idx.append(find_concept_tokens(ids, self.tokenizer, "Subject", e["subject"]))
            object_idx.append(find_concept_tokens(ids, self.tokenizer, "Object", e["object"]))

        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids != self.pad_token_id,
            labels=labels,
            pixel_values=torch.cat([e["pixel_values"] for e in examples]),
            image_grid_thw=torch.cat([e["image_grid_thw"] for e in examples]),
            subject_token_indices=subject_idx,
            object_token_indices=object_idx,
            concept_valid_mask=torch.tensor([[bool(s), bool(o)] for s, o in zip(subject_idx, object_idx)]),
        )
        if all("dino_features" in e for e in examples):
            batch["dino_features"] = torch.stack([e["dino_features"] for e in examples])  # [B, T, P, D]
        return batch
