#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LLaVA-1.5-7B + Ours student under multi-image protocol (images/)
# 4 MileBench datasets x {keep=1.0, 0.5, 0.2} x 200 samples
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/VFlowOpt/.conda/VFlowOpt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LD_LIBRARY_PATH="/workspace/VFlowOpt/.conda/VFlowOpt/lib:${LD_LIBRARY_PATH:-}"
# look-m's LLaVA fork must come BEFORE VFlowOpt's LLaVA-OneVision so `llava`
# resolves to the LOOK-M repo (matches the runner's expected fork).
export PYTHONPATH="/workspace/look-m/LLaVA-mix_merge_v1:/workspace/look-m:/workspace/zap:${PYTHONPATH:-}"
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
OUT_ROOT="${EXP_DIR}/outputs/llava15_multiimage_$(date +%Y%m%d_%H%M%S)"
MODEL_DIR=/workspace/zap/ckpts/llava-v1.5-7b
STUDENT_DIR=/workspace/zap/artifacts/original_llava_teacher/student_llava15_original_future_1800_lr1e4_15ep
DATASETS=(ALFRED CLEVR-Change IEdit Spot-the-Diff)
KEEP_RATIOS=(1.0 0.5 0.2)
LIMIT=200

mkdir -p "$LOG_DIR" "$OUT_ROOT"
LOG_FILE="${LOG_DIR}/llava15_multiimage_$(date +%Y%m%d_%H%M%S).log"

echo "=== LLaVA-1.5 multi-image START $(date -Is) ===" | tee -a "$LOG_FILE"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG_FILE"
echo "DATASETS=${DATASETS[*]}" | tee -a "$LOG_FILE"
echo "KEEP_RATIOS=${KEEP_RATIOS[*]}" | tee -a "$LOG_FILE"
echo "LIMIT=$LIMIT" | tee -a "$LOG_FILE"

for ds in "${DATASETS[@]}"; do
  echo "---- dataset=$ds ----" | tee -a "$LOG_FILE"
  python /workspace/zap/foresight/eval/milebench_lookm_student.py \
      --dataset "$ds" \
      --model_dir "$MODEL_DIR" \
      --student_path "$STUDENT_DIR" \
      --keep_ratio "${KEEP_RATIOS[@]}" \
      --output_dir "$OUT_ROOT" \
      --device cuda:0 \
      --limit "$LIMIT" \
      --overwrite \
      --look_model_name llava15_lookm_student 2>&1 | tee -a "$LOG_FILE"
done

echo "=== LLaVA-1.5 multi-image DONE $(date -Is) ===" | tee -a "$LOG_FILE"
echo "scoring next: pred.json files under $OUT_ROOT"
