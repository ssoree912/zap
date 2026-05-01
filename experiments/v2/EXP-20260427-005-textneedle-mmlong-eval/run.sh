#!/usr/bin/env bash
set -euo pipefail

cd /workspace/zap

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK=TextNeedleInAHaystack
KEEP=0.2
DATA_ROOT=/workspace/zap/data/MileBench
CKPT=/workspace/zap/ckpts/student_v2_A_mmlongbench_lr1e4_20ep
OUTDIR=/workspace/zap/artifacts/EXP-20260427-005-textneedle-mmlong-eval/milebench_mmlong_total0.2/TextNeedleInAHaystack

mkdir -p "$OUTDIR"

/opt/conda/envs/kv/bin/python /workspace/zap/evaluate_image_teacher_pruning.py \
  --mode visual_utility_student \
  --student_model_name "$CKPT" \
  --total_keep_ratio "$KEEP" \
  --dataset_path "$DATA_ROOT/$TASK/$TASK.json" \
  --image_root "$DATA_ROOT/$TASK/images" \
  --look_dataset_name "$TASK" \
  --look_model_name "zap_${TASK}_mmlong_lr1e4_20ep_total${KEEP}" \
  --output_dir "$OUTDIR" \
  --attn_implementation sdpa \
  --max_new_tokens 32 \
  --truncate
