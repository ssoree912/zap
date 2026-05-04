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
TEACHER_ROOT="/workspace/zap/artifacts/original_onevision_teacher/future_decode_qwen2_7b"
OUT_DIR="/workspace/zap/artifacts/original_onevision_teacher/student_onevision_original_future_1800_lr1e4_15ep"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

LOG_FILE="${LOG_DIR}/train_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== original OneVision student train START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TEACHER_ROOT=${TEACHER_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

python "${EXP_DIR}/train_original_onevision_student.py" \
  --teacher-root "${TEACHER_ROOT}" \
  --datasets textvqa gqa scienceqa \
  --per-ds-limit 600 \
  --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
  --model-name llava_qwen \
  --device cuda:0 \
  --device-map auto \
  --epochs 15 \
  --lr 1e-4 \
  --output-dir "${OUT_DIR}" \
  --seed 42 2>&1 | tee "${LOG_FILE}"

echo "=== original OneVision student train DONE $(date -Is) ==="
