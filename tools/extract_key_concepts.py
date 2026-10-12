"""Add subject/object concepts to an annotation file with Qwen3-4B-Instruct-2507.

Each entry gets "key_concepts", e.g. ["subject: person", "object: book"]; the
object is omitted when the query has none.

    torchrun --nproc_per_node=4 tools/extract_key_concepts.py \
        --input data/annotations/charades_train_raw.json \
        --output data/annotations/charades_train.json
"""
import argparse
import json
import os
import re

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """You are an expert in semantic decomposition for video temporal grounding.
Extract the most important Subject and Object from the query for frame-level grounding.

Roles & Prefixes you MUST use:
- "Subject: [noun]" -> The primary living actor or main entity.
- "Object: [physical noun]" -> The primary tangible target object being interacted with. If none exists, omit this.

Rules:
1. Return a JSON list containing up to 2 strings (one Subject, one Object).
2. Each string MUST start with "Subject:" or "Object:".
3. If the Subject is interacting with a physical item (e.g., door, shoes, cup), you MUST extract it as an Object. Do not miss it even if the grammar is awkward.
4. Ignore verbs, scenes, abstract concepts, and meta-descriptions. Focus ONLY on WHO and WHAT.
5. Keep the nouns extremely concise (1-2 words).
6. Return ONLY a valid JSON list.

Examples:
Query: "a person opens the refrigerator in the kitchen"
JSON list: ["Subject: person", "Object: refrigerator"]

Query: "person they open the door."
JSON list: ["Subject: person", "Object: door"]

Query: "person start putting shoes on."
JSON list: ["Subject: person", "Object: shoes"]

Query: "The girl dances around the room."
JSON list: ["Subject: girl"]

Query: "{query}"
JSON list:"""


def parse_concepts(response: str):
    for candidate in (response.strip(), *re.findall(r"\[.*?\]", response, re.DOTALL)):
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return [c.strip().lower() for c in result if isinstance(c, str) and 0 < len(c.strip()) < 50]
        except json.JSONDecodeError:
            continue
    return [c.strip().lower() for c in re.findall(r'"([^"]+)"', response)]


@torch.no_grad()
def extract(query: str, tokenizer, model) -> list:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(query=query)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return parse_concepts(tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)

    with open(args.input) as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}
    ).eval()

    concepts = {}
    for i in tqdm(range(rank, len(data), world), disable=rank != 0):
        match = re.search(r"Query:\s*(.+?)(?:\n|$)", data[i]["conversations"][0]["value"], re.IGNORECASE)
        if match:
            concepts[i] = extract(match.group(1).strip(), tokenizer, model)

    if distributed:
        parts = [None] * world
        dist.all_gather_object(parts, concepts)
        concepts = {k: v for part in parts for k, v in part.items()}

    if rank == 0:
        for i, c in concepts.items():
            data[i]["key_concepts"] = c
        with open(args.output, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Wrote {len(concepts)}/{len(data)} entries with key_concepts to {args.output}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
