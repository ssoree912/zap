#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/VFlowOpt/.conda/VFlowOpt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/workspace/zap:/workspace/VFlowOpt/src/LLaVA-OneVision:${PYTHONPATH:-}"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_DIR="/workspace/zap/experiments/EXP-20260504-001-onevision-student-milebench"
LOG_DIR="${EXP_DIR}/logs"
OUT_ROOT="${OUT_ROOT:-${EXP_DIR}/outputs/student_milebench_max64_onevision}"
STUDENT_DIR="/workspace/zap/artifacts/original_onevision_teacher/student_onevision_original_future_1800_lr1e4_15ep"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

LOG_FILE="${LOG_DIR}/onevision_original_student_rouge_max64_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== OneVision original student ROUGE max64 START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "STUDENT_DIR=${STUDENT_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "LOG_FILE=${LOG_FILE}"

python -u "${EXP_DIR}/milebench_onevision_original_student_sweep.py" \
  --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
  --model-name llava_qwen \
  --student-path "${STUDENT_DIR}" \
  --datasets ALFRED CLEVR-Change IEdit Spot-the-Diff \
  --keep-ratios 0.5 0.1 0.05 \
  --output-root "${OUT_ROOT}" \
  --device cuda:0 \
  --device-map auto \
  --conv-template qwen_1_5 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --overwrite 2>&1 | tee "${LOG_FILE}"

echo "=== OneVision original student ROUGE max64 DONE $(date -Is) ==="
