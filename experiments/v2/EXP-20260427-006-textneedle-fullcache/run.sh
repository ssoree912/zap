#!/usr/bin/env bash
set -euo pipefail

cd /workspace/zap

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK=TextNeedleInAHaystack
KEEP=1.0
DATA_ROOT=/workspace/zap/data/MileBench
OUTDIR=/workspace/zap/artifacts/EXP-20260427-006-textneedle-fullcache/milebench_fullcache/TextNeedleInAHaystack

mkdir -p "$OUTDIR"

/opt/conda/envs/kv/bin/python /workspace/zap/evaluate_image_teacher_pruning.py \
  --mode random_all_token \
  --total_keep_ratio "$KEEP" \
  --dataset_path "$DATA_ROOT/$TASK/$TASK.json" \
  --image_root "$DATA_ROOT/$TASK/images" \
  --look_dataset_name "$TASK" \
  --look_model_name "zap_${TASK}_llava15_fullcache" \
  --output_dir "$OUTDIR" \
  --attn_implementation sdpa \
  --max_new_tokens 32 \
  --truncate
