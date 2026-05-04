#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/VFlowOpt/.conda/VFlowOpt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/workspace/zap:/workspace/VFlowOpt/src/LLaVA-OneVision:${PYTHONPATH:-}"

EXP_DIR="/workspace/zap/experiments/EXP-20260503-026-onevision-original-teacher-train"
LOG_DIR="${EXP_DIR}/logs"
REF_ROOT="${EXP_DIR}/outputs/mmvet_detail_fullcache_refs"
OUT_ROOT="${EXP_DIR}/outputs/mmvet_detail_student"
STUDENT_DIR="/workspace/zap/artifacts/original_onevision_teacher/student_onevision_original_future_1800_lr1e4_15ep"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

if [[ ! -f "${STUDENT_DIR}/pytorch_model.bin" ]]; then
  echo "Missing student checkpoint: ${STUDENT_DIR}/pytorch_model.bin" >&2
  exit 1
fi
if [[ ! -f "${REF_ROOT}/mmvet/fullcache_rouge_ref.json" || ! -f "${REF_ROOT}/detail_1k/fullcache_rouge_ref.json" ]]; then
  echo "Missing full-cache ROUGE refs under ${REF_ROOT}; run run_eval_fullcache_gpu0.sh first." >&2
  exit 1
fi

LOG_FILE="${LOG_DIR}/eval_student_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== OneVision MMVet/detail student eval START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "STUDENT_DIR=${STUDENT_DIR}"
echo "REF_ROOT=${REF_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_FILE=${LOG_FILE}"

for KEEP in 0.2 0.5 0.7; do
  KEEP_TAG="${KEEP/./}"
  OUT_DIR="${OUT_ROOT}/keep_${KEEP_TAG}"
  mkdir -p "${OUT_DIR}"
  echo "--- keep_ratio=${KEEP} OUT_DIR=${OUT_DIR} ---"
  python /workspace/zap/scripts/eval_onevision_mmvet_detail.py \
    --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
    --model-name llava_qwen \
    --conv-template qwen_1_5 \
    --device cuda:0 \
    --device-map auto \
    --datasets mmvet detail_1k \
    --student-path "${STUDENT_DIR}" \
    --keep-ratio "${KEEP}" \
    --rouge-ref-path "${REF_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --overwrite
done 2>&1 | tee "${LOG_FILE}"

python "${EXP_DIR}/summarize_mmvet_detail_results.py" \
  --input-root "${OUT_ROOT}" \
  --output-csv "${OUT_ROOT}/keep_ratio_dataset_performance.csv"

echo "=== OneVision MMVet/detail student eval DONE $(date -Is) ==="
