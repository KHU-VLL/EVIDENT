#!/bin/bash
# EVIDENT on Charades-STA (Table 4): Qwen2.5-VL-7B, EB Adapter on layers 0-10,
# LoRA on layers 11-27, 20 frames, relative timestamps, lambda=0.1, global batch 32 on 8 GPUs.
set -e

VIDEO_DIR=${VIDEO_DIR:?set VIDEO_DIR to the Charades-STA videos}
DINO_DIR=${DINO_DIR:?set DINO_DIR to features from tools/extract_dino_features.py (20 frames)}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/evident_charades}
NUM_GPUS=${NUM_GPUS:-8}
SEED=${SEED:-42}

deepspeed --num_gpus $NUM_GPUS train.py \
    --deepspeed scripts/zero2.json \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --train_annotation data/annotations/charades_train.json \
    --video_dir $VIDEO_DIR \
    --dino_feature_dir $DINO_DIR \
    --num_frames 20 \
    --timestamp_format relative \
    --adapter_layers 0-10 \
    --bottleneck_dim 896 \
    --num_slots 4 \
    --num_iterations 3 \
    --ebd_loss_weight 0.1 \
    --lora_rank 16 --lora_alpha 64 --lora_dropout 0.05 \
    --num_train_epochs 2 \
    --lr_schedule_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $((32 / NUM_GPUS)) \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --bf16 True --tf32 True \
    --gradient_checkpointing True \
    --use_liger_kernel True \
    --dataloader_num_workers 8 \
    --logging_steps 50 \
    --save_strategy epoch \
    --report_to none \
    --seed $SEED \
    --output_dir $OUTPUT_DIR
