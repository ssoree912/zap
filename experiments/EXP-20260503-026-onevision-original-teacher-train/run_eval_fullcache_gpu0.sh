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
OUT_DIR="${EXP_DIR}/outputs/mmvet_detail_fullcache_refs"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

LOG_FILE="${LOG_DIR}/eval_fullcache_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== OneVision MMVet/detail full-cache eval START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

python /workspace/zap/scripts/eval_onevision_mmvet_detail.py \
  --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
  --model-name llava_qwen \
  --conv-template qwen_1_5 \
  --device cuda:0 \
  --device-map auto \
  --datasets mmvet detail_1k \
  --output-dir "${OUT_DIR}" \
  --keep-ratio 1.0 \
  --overwrite 2>&1 | tee "${LOG_FILE}"

echo "=== OneVision MMVet/detail full-cache eval DONE $(date -Is) ==="
