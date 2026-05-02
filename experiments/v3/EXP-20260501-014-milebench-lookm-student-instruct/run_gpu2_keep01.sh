#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

EXP_DIR=/workspace/zap/experiments/EXP-20260501-014-milebench-lookm-student-instruct
LOG="${EXP_DIR}/logs/gpu2_keep01_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== GPU2 keep_ratio=0.1 START $(date -Is) ==="
/opt/conda/envs/vflowopt_chartqa_eval/bin/python3 /workspace/zap/foresight/eval/milebench_lookm_student.py \
  --dataset all \
  --student_path /workspace/zap/ckpts/student_llava15_instruct_2000_lr1e4 \
  --keep_ratio 0.1 \
  --output_dir "${EXP_DIR}/outputs" \
  --device cuda:0 \
  --overwrite
echo "=== GPU2 keep_ratio=0.1 DONE $(date -Is) ==="
