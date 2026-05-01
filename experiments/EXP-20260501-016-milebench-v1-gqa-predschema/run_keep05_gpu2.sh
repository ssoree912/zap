#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

EXP_DIR=/workspace/zap/experiments/EXP-20260501-016-milebench-v1-gqa-predschema
mkdir -p "${EXP_DIR}/logs"
LOG="${EXP_DIR}/logs/keep05_gpu2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== v1 GQA student keep=0.5 fresh MileBench generation START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

/opt/conda/envs/vflowopt_chartqa_eval/bin/python3 /workspace/zap/foresight/eval/milebench_llava15_student.py \
  --dataset all \
  --pretrained /workspace/zap/ckpts/llava-1.5-7b-hf \
  --student_path /workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4 \
  --total_keep_ratio 0.5 \
  --output_dir "${EXP_DIR}/outputs" \
  --device cuda:0 \
  --image_column combined_1_images \
  --prompt_style look_milebench \
  --generation_backend press \
  --eval_backend lookm \
  --look_model_name foresight_hf_llava15_student_v1_gqa_predschema \
  --overwrite

echo "=== v1 GQA student keep=0.5 fresh MileBench generation DONE $(date -Is) ==="
