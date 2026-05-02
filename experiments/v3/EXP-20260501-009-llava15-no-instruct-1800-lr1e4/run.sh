#!/usr/bin/env bash
set -euo pipefail

cd /workspace/zap

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="/opt/conda/envs/vflowopt_chartqa_eval/bin/python"
LLAVA15_CKPT="/workspace/zap/ckpts/llava-1.5-7b-hf"
TEACHER_ROOT="/workspace/zap/data/train/teacher_llava15"
STUDENT_CKPT="/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4"
EXP_DIR="/workspace/zap/experiments/EXP-20260501-009-llava15-no-instruct-1800-lr1e4"
LOG_DIR="${EXP_DIR}/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${RUN_ID}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== EXP-20260501-009 START $(date -Is) ====="
echo "[config] python=${PYTHON}"
echo "[config] llava=${LLAVA15_CKPT}"
echo "[config] teacher_root=${TEACHER_ROOT}"
echo "[config] student_ckpt=${STUDENT_CKPT}"
echo "[config] datasets=textvqa scienceqa gqa"
echo "[config] n_per_dataset=600 epochs=20 lr=1e-4 device=cuda:0"

for dataset in textvqa scienceqa gqa; do
  n_files="$(find "${TEACHER_ROOT}/${dataset}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  echo "[check] ${dataset}: ${n_files} teacher files"
  if [ "$n_files" -lt 600 ]; then
    echo "===== COLLECT ${dataset} n=600 START $(date -Is) ====="
    "$PYTHON" -m foresight.teacher.collect_llava15 \
      --dataset "$dataset" \
      --n-samples 600 \
      --model "$LLAVA15_CKPT" \
      --output-root "$TEACHER_ROOT" \
      --seed 0 \
      --max-new-tokens 64 \
      --trajectory-m 1 \
      --device cuda:0
    echo "===== COLLECT ${dataset} DONE $(date -Is) ====="
  fi
done

for dataset in textvqa scienceqa gqa; do
  n_files="$(find "${TEACHER_ROOT}/${dataset}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  echo "[check] ${dataset}: ${n_files} teacher files"
  if [ "$n_files" -lt 600 ]; then
    echo "[error] ${dataset} has fewer than 600 teacher files"
    exit 1
  fi
done

if [ -f "${STUDENT_CKPT}/config.json" ]; then
  echo "[error] checkpoint already exists: ${STUDENT_CKPT}"
  echo "[error] refusing to overwrite existing student checkpoint"
  exit 2
fi

echo "===== TRAIN no-instruct 1800 lr1e4 START $(date -Is) ====="
"$PYTHON" -m foresight.train.llava_15 \
  --teacher-root "$TEACHER_ROOT" \
  --datasets textvqa scienceqa gqa \
  --n-per-dataset 600 \
  --llava-path "$LLAVA15_CKPT" \
  --output-dir "$STUDENT_CKPT" \
  --epochs 20 \
  --lr 1e-4 \
  --device cuda:0 \
  --log-every 50
echo "===== TRAIN no-instruct 1800 lr1e4 DONE $(date -Is) ====="

echo "===== EXP-20260501-009 DONE $(date -Is) ====="
