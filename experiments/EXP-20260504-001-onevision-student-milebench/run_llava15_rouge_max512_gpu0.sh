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
OUT_ROOT="${OUT_ROOT:-${EXP_DIR}/outputs/student_milebench_max512_llava15}"
MODEL_DIR="/workspace/zap/ckpts/llava-v1.5-7b"
STUDENT_DIR="/workspace/zap/artifacts/original_llava_teacher/student_llava15_original_future_1800_lr1e4_15ep"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

LOG_FILE="${LOG_DIR}/llava15_original_student_rouge_max512_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== LLaVA-1.5 original student ROUGE max512 START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "STUDENT_DIR=${STUDENT_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "LOG_FILE=${LOG_FILE}"

python -u "${EXP_DIR}/milebench_llava15_original_student_sweep.py" \
  --model-path "${MODEL_DIR}" \
  --model-name llava-v1.5-7b \
  --student-path "${STUDENT_DIR}" \
  --datasets ALFRED CLEVR-Change IEdit Spot-the-Diff \
  --keep-ratios 0.5 0.1 0.05 \
  --output-root "${OUT_ROOT}" \
  --device cuda:0 \
  --device-map cuda:0 \
  --conv-template vicuna_v1 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --overwrite 2>&1 | tee "${LOG_FILE}"

echo "=== LLaVA-1.5 original student ROUGE max512 DONE $(date -Is) ==="
