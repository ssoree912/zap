#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/VFlowOpt/.conda/VFlowOpt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/workspace/zap:/workspace/VFlowOpt/src/LLaVA-OneVision:${PYTHONPATH:-}"

EXP_DIR="/workspace/zap/experiments/EXP-20260503-025-onevision-mmvet-detail-fullcache"
OUT_DIR="${EXP_DIR}/outputs/fullcache_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EXP_DIR}/logs"
MODEL_PATH="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/fullcache_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== OneVision MMVet/detail_1k full-cache START $(date -Is) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

python /workspace/zap/scripts/eval_onevision_mmvet_detail.py \
  --model-path "${MODEL_PATH}" \
  --datasets mmvet detail_1k \
  --output-dir "${OUT_DIR}" \
  --device cuda:0 \
  --torch-dtype float16 \
  --attn-implementation sdpa \
  --overwrite 2>&1 | tee "${LOG_FILE}"

echo "=== OneVision MMVet/detail_1k full-cache DONE $(date -Is) ==="
