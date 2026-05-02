#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXP_DIR="/workspace/zap/experiments/EXP-20260501-017-prefixkv-mmvet-detail-v1-gqa"
OUT_DIR="${EXP_DIR}/outputs/keep05"
LOG_DIR="${EXP_DIR}/logs"
PYTHON="/opt/conda/envs/vflowopt_chartqa_eval/bin/python3"
STUDENT_CKPT="/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4"
KEEP="0.5"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "=== PrefixKV mm-vet/detail_1k v1 GQA keep=${KEEP} START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "STUDENT_CKPT=${STUDENT_CKPT}"

run_rouge() {
  local dataset="$1"
  local data_path="$2"
  local image_path="$3"
  local n_samples="$4"
  local output_dir="${OUT_DIR}/${dataset}/rouge_gt"

  mkdir -p "${output_dir}"
  echo "=== ${dataset} keep=${KEEP} START $(date -Is) ==="
  "${PYTHON}" /workspace/zap/scripts/eval_rouge.py \
    --method visual_utility_student \
    --student-model-name "${STUDENT_CKPT}" \
    --total-keep-ratio "${KEEP}" \
    --data-path "${data_path}" \
    --image-path "${image_path}" \
    --eval-samples "${n_samples}" \
    --output-dir "${output_dir}"
  echo "=== ${dataset} keep=${KEEP} DONE $(date -Is) ==="
}

run_rouge "mm-vet" "/workspace/data/mm-vet/mm-vet.json" "/workspace/data/mm-vet" 218
run_rouge "detail_1k" "/workspace/data/detail_1k.json" "/workspace/data" 1000

echo "=== PrefixKV mm-vet/detail_1k v1 GQA keep=${KEEP} DONE $(date -Is) ==="
