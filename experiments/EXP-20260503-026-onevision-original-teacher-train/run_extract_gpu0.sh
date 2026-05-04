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
OUT_ROOT="/workspace/zap/artifacts/original_onevision_teacher/future_decode_qwen2_7b"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

LOG_FILE="${LOG_DIR}/extract_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== original OneVision teacher extract START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_FILE=${LOG_FILE}"

python "${EXP_DIR}/collect_original_onevision_teacher.py" \
  --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
  --model-name llava_qwen \
  --conv-template qwen_1_5 \
  --device cuda:0 \
  --device-map auto \
  --datasets textvqa gqa scienceqa \
  --n-samples 600 \
  --max-new-tokens 64 \
  --seed 42 \
  --output-root "${OUT_ROOT}" \
  --overwrite 2>&1 | tee "${LOG_FILE}"

echo "=== original OneVision teacher extract DONE $(date -Is) ==="
