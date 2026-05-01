#!/usr/bin/env bash
set -euo pipefail

cd /workspace/zap

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="/opt/conda/envs/vflowopt_chartqa_eval/bin/python3"
LLAVA15_CKPT="/workspace/zap/ckpts/llava-1.5-7b-hf"
STUDENT_CKPT="/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4"
EXP_DIR="/workspace/zap/experiments/EXP-20260501-012-milebench-llava15-v1-gqa-lr1e4"
OUT_DIR="${EXP_DIR}/outputs"
LOG_DIR="${EXP_DIR}/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${RUN_ID}.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== EXP-20260501-012 START $(date -Is) ====="
echo "[config] physical_gpu=0 visible_device=cuda:0"
echo "[config] base=${LLAVA15_CKPT}"
echo "[config] student=${STUDENT_CKPT}"
echo "[config] output=${OUT_DIR}"
echo "[config] dataset=all keep_ratio=0.5 0.1"

"$PYTHON" /workspace/zap/foresight/eval/milebench_llava15_student.py \
  --dataset all \
  --pretrained "$LLAVA15_CKPT" \
  --student_path "$STUDENT_CKPT" \
  --keep_ratio 0.5 0.1 \
  --output_dir "$OUT_DIR" \
  --device cuda:0 \
  --image_column combined_1_images \
  --prompt_style look_milebench

echo "===== EXP-20260501-012 DONE $(date -Is) ====="
