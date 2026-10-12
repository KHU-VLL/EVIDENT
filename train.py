"""Train EVIDENT (EB Adapter + LoRA) on Qwen2.5-VL. See scripts/train_*.sh."""
import logging
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, Qwen2_5_VLForConditionalGeneration, set_seed

from evident.data import GroundingCollator, GroundingDataset
from evident.model import get_eb_adapters, inject_eb_adapters, lora_target_modules
from evident.train import (
    DataArguments,
    EVIDENTTrainer,
    ModelArguments,
    TrainingArguments,
    evident_config,
    parse_layers,
    save_eb_adapter,
)

logger = logging.getLogger(__name__)


def build_model(model_args: ModelArguments, args: TrainingArguments):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id, torch_dtype=dtype, attn_implementation=model_args.attn_implementation
    )
    model.config.use_cache = False
    model.requires_grad_(False)  # vision tower, merger and LLM stay frozen

    if args.gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        model.enable_input_require_grads()

    adapter_layers = parse_layers(args.adapter_layers)
    inject_eb_adapters(
        model,
        layer_indices=adapter_layers,
        bottleneck_dim=args.bottleneck_dim,
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        tokens_per_frame=args.tokens_per_frame,
        use_e2v_gate=args.use_e2v_gate,
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules(model, skip_layers=adapter_layers),
        bias="none",
    )
    model.to(dtype)
    model = get_peft_model(model, lora_config)
    # get_peft_model freezes everything except LoRA; re-enable the EB Adapters.
    for adapter in get_eb_adapters(model):
        adapter.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO if args.local_rank in (-1, 0) else logging.WARNING)
    set_seed(args.seed)

    model = build_model(model_args, args)
    processor = AutoProcessor.from_pretrained(model_args.model_id, padding_side="left")

    dataset = GroundingDataset(
        annotation_path=data_args.train_annotation,
        video_dir=data_args.video_dir,
        processor=processor,
        num_frames=data_args.num_frames,
        dino_feature_dir=data_args.dino_feature_dir,
        frame_size=data_args.frame_size,
        timestamp_format=data_args.timestamp_format,
    )
    config = evident_config(args, data_args)
    trainer = EVIDENTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=GroundingCollator(processor.tokenizer),
        processing_class=processor,
        evident_config=config,
    )

    resume = any(pathlib.Path(args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_state()

    trainer.save_model(args.output_dir)  # LoRA weights
    save_eb_adapter(model, args.output_dir, config, args.should_save)


if __name__ == "__main__":
    main()
