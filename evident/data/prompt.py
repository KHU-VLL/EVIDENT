"""Prompt construction shared by training and evaluation.

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    timestamp: 0; feature: <|vision_start|><|image_pad|><|vision_end|>
    timestamp: 5; feature: <|vision_start|><|image_pad|><|vision_end|>
    ...

    Subject: person
    Object: light
    The video lasts 30.96 seconds.
    Query: person turn a light on.
    Given the video and the query, find the relevant windows [start_time, end_time].<|im_end|>
    <|im_start|>assistant

Two timestamp formats are supported (Appendix C.3):
    relative  integer bins in [0, 100] of the video duration (Chrono-BLIP), used for Charades-STA
              "timestamp: 47; feature: ..."      answer "[[5, 25]]"
    absolute  integer seconds, used for QVHighlights
              "timestamp: 47 seconds; feature: ..." answer "[[3, 15]]"
"""
import re
from typing import List, Sequence, Tuple

SYSTEM_PROMPT = "You are a helpful assistant."
FRAME_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
RESOLUTION = 100
INTERVAL_PATTERN = re.compile(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]")


TIMESTAMP_FORMATS = ("relative", "absolute")


def to_bin(seconds: float, duration: float) -> int:
    """Absolute seconds -> relative integer bin in [0, RESOLUTION]."""
    if duration <= 0:
        return 0
    return max(0, min(RESOLUTION, int(round(seconds / duration * RESOLUTION))))


def format_timestamps(timestamps: Sequence[float], duration: float, fmt: str) -> List[str]:
    """Frame timestamps (seconds) as written in the prompt, for training and evaluation."""
    if fmt == "relative":
        return [str(to_bin(t, duration)) for t in timestamps]
    if fmt == "absolute":
        return [f"{int(round(t))} seconds" for t in timestamps]
    raise ValueError(f"Unknown timestamp format {fmt!r}; expected one of {TIMESTAMP_FORMATS}.")


def format_answer(answer: str, duration: float, fmt: str) -> str:
    """Ground-truth windows in seconds -> target text, e.g. "[[3.0, 15.0]]" -> "[[5, 25]]" (relative, 60 s)."""
    if fmt == "relative":
        convert = lambda x: to_bin(x, duration)
    elif fmt == "absolute":
        convert = lambda x: int(round(x))
    else:
        raise ValueError(f"Unknown timestamp format {fmt!r}; expected one of {TIMESTAMP_FORMATS}.")
    return INTERVAL_PATTERN.sub(lambda m: f"[{convert(float(m.group(1)))}, {convert(float(m.group(2)))}]", answer)


def to_seconds(value: float, duration: float, fmt: str) -> float:
    """A predicted timestamp -> seconds."""
    return value * (duration / RESOLUTION) if fmt == "relative" else value


def parse_key_concepts(key_concepts: Sequence[str]) -> Tuple[str, str]:
    """["subject: person", "object: book"] -> ("person", "book"). Missing entries are ""."""
    subject, obj = "", ""
    for concept in key_concepts:
        if concept.startswith("subject:"):
            subject = concept[len("subject:"):].strip()
        elif concept.startswith("object:"):
            obj = concept[len("object:"):].strip()
    return subject, obj


def instruction_from_sample(sample: dict) -> str:
    """The human turn of the annotation without the <video> placeholder."""
    human = next(c["value"] for c in sample["conversations"] if c["from"] == "human")
    return re.sub(r"\n?<video>\n?", "", human).strip()


def build_prompt(timestamps: Sequence[str], subject: str, obj: str, instruction: str) -> str:
    frames = "".join(f"timestamp: {t}; feature: {FRAME_TOKEN}\n" for t in timestamps)
    concepts = (f"Subject: {subject}\n" if subject else "") + (f"Object: {obj}\n" if obj else "")
    user = f"{frames}\n{concepts}{instruction}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def find_concept_tokens(input_ids: List[int], tokenizer, label: str, text: str) -> List[int]:
    """Positions of the tokens after "Subject:"/"Object:" up to the line break.

    Returns [] if the concept is empty or the decoded span does not contain it.
    """
    if not text:
        return []
    pattern = tokenizer.encode(f"{label}:", add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    for i in range(len(input_ids) - len(pattern)):
        if input_ids[i:i + len(pattern)] != pattern:
            continue
        start = end = i + len(pattern)
        while end < len(input_ids) and input_ids[end] != newline:
            end += 1
        if start < end and text.strip().lower() in tokenizer.decode(input_ids[start:end]).strip().lower():
            return list(range(start, end))
        return []
    return []
