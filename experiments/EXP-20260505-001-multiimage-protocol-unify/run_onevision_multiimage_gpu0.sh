#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# OneVision-7B + Ours student under multi-image protocol (images/, video modality)
# 3 MileBench datasets (no Spot-the-Diff) x {keep=0.5, 0.1, 0.05} x 200 samples
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/VFlowOpt/.conda/VFlowOpt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/workspace/VFlowOpt/.conda/VFlowOpt/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/workspace/zap:/workspace/VFlowOpt/src/LLaVA-OneVision:${PYTHONPATH:-}"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

EXP_DIR=/workspace/zap/experiments/EXP-20260505-001-multiimage-protocol-unify
LOG_DIR="${EXP_DIR}/logs"
OUT_ROOT="${EXP_DIR}/outputs/onevision_multiimage_$(date +%Y%m%d_%H%M%S)"
STUDENT_DIR=/workspace/zap/artifacts/original_onevision_teacher/student_onevision_original_future_1800_lr1e4_15ep
DATASETS=(ALFRED CLEVR-Change IEdit)
KEEP_RATIOS=(0.5 0.1 0.05)
LIMIT=200
MAX_NEW_TOKENS=64

mkdir -p "$LOG_DIR" "$OUT_ROOT"
LOG_FILE="${LOG_DIR}/onevision_multiimage_$(date +%Y%m%d_%H%M%S).log"

echo "=== OneVision multi-image START $(date -Is) ===" | tee -a "$LOG_FILE"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG_FILE"
echo "DATASETS=${DATASETS[*]}" | tee -a "$LOG_FILE"
echo "KEEP_RATIOS=${KEEP_RATIOS[*]}" | tee -a "$LOG_FILE"
echo "LIMIT=$LIMIT  MAX_NEW_TOKENS=$MAX_NEW_TOKENS" | tee -a "$LOG_FILE"

python -u /workspace/zap/foresight/eval/milebench_onevision_student_multiimage.py \
    --student-path "$STUDENT_DIR" \
    --datasets "${DATASETS[@]}" \
    --keep-ratios "${KEEP_RATIOS[@]}" \
    --output-root "$OUT_ROOT" \
    --device cuda:0 \
    --device-map cuda:0 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --limit "$LIMIT" \
    --overwrite 2>&1 | tee -a "$LOG_FILE"

echo "=== OneVision multi-image DONE $(date -Is) ===" | tee -a "$LOG_FILE"
echo "scoring: pred.json files under $OUT_ROOT"
