#!/bin/bash
# Evaluate one checkpoint on Charades-STA (20 frames), QVHighlights (60 frames) and DiDeMo (20 frames).
set -e

CHECKPOINT=${1:?usage: scripts/eval_cross_domain.sh <checkpoint_dir> [result_dir]}
RESULT_DIR=${2:-results/$(basename $CHECKPOINT)}
NUM_GPUS=${NUM_GPUS:-8}
CHARADES_VIDEOS=${CHARADES_VIDEOS:?set CHARADES_VIDEOS}
QVH_VIDEOS=${QVH_VIDEOS:?set QVH_VIDEOS}
DIDEMO_VIDEOS=${DIDEMO_VIDEOS:?set DIDEMO_VIDEOS}

run() {  # name annotation video_dir num_frames
    torchrun --nproc_per_node=$NUM_GPUS evaluate.py \
        --checkpoint $CHECKPOINT \
        --annotation data/annotations/$2 \
        --video_dir $3 \
        --num_frames $4 \
        --output $RESULT_DIR/$1.json
}

run charades     charades_test.json     $CHARADES_VIDEOS 20
run qvhighlights qvhighlights_test.json $QVH_VIDEOS      60
run didemo       didemo_test.json       $DIDEMO_VIDEOS   20
