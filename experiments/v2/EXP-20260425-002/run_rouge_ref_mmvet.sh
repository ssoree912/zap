#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

REPO=/workspace/zap
MODEL_PATH=$REPO/ckpts/llava-1.5-7b-hf
STUDENT=$REPO/ckpts/student_v2_traj
DATA_PATH=/workspace/data/mm-vet/mm-vet.json
IMAGE_PATH=/workspace/data/mm-vet
OUT_BASE=$REPO/artifacts/EXP-20260425-002/mm-vet

for K in 0.1 0.4 0.6 0.8 0.9; do
    echo "=== k=${K} ==="
    python $REPO/eval_rouge.py \
        --model-path $MODEL_PATH \
        --data-path $DATA_PATH \
        --image-path $IMAGE_PATH \
        --eval-samples 218 \
        --method visual_utility_student \
        --student-model-name $STUDENT \
        --total-keep-ratio $K \
        --output-dir $OUT_BASE/traj_rouge_total${K} \
        --device cuda:0 \
        --attn-implementation sdpa
done

echo "=== Computing ROUGE vs full-cache ref ==="
python $REPO/scripts/compute_rouge_ref.py \
    --rouge-ref $REPO/data/rouge_ref/our_full_mm-vet.json \
    --result-jsons \
        $OUT_BASE/traj_rouge_total0.1/result.json \
        $OUT_BASE/traj_rouge_total0.2/result.json \
        $OUT_BASE/traj_rouge_total0.3/result.json \
        $OUT_BASE/traj_rouge_total0.4/result.json \
        $OUT_BASE/traj_rouge_total0.5/result.json \
        $OUT_BASE/traj_rouge_total0.6/result.json \
        $OUT_BASE/traj_rouge_total0.7/result.json \
        $OUT_BASE/traj_rouge_total0.8/result.json \
        $OUT_BASE/traj_rouge_total0.9/result.json \
    --output $REPO/result/traj_mmvet_rouge_ref.csv

echo "DONE"
