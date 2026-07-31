#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
MODEL="${WORKSPACE}/models/llava-onevision-qwen2-7b-ov"
STUDENT="${WORKSPACE}/zap/artifacts/original_onevision_teacher/student_onevision_answer_n1800_e15_seed0"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260726-004-onevision-answer-n600"
OUT_ROOT="${EXP_DIR}/outputs/chartqa_onevision_answer_imgkeep0p1"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/chartqa_imgkeep0p1_gpu3_${RUN_STAMP}.log"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

if [[ ! -f "${STUDENT}/pytorch_model.bin" ]]; then
    echo "Missing student checkpoint: ${STUDENT}/pytorch_model.bin" >&2
    exit 1
fi

echo "[config] task=ChartQA full test split"
echo "[config] keep_ratio=0.1 basis=image-token"
echo "[config] physical_gpu=3"
echo "[config] student=${STUDENT}"
echo "[config] output=${OUT_ROOT}"

cd "${QVIK_ROOT}"
CUDA_VISIBLE_DEVICES=3 \
QVIK_DATA_ROOT="${WORKSPACE}/data/eval" \
PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
"${PYTHON}" qvik/eval/run_lmms_eval.py \
    --model lmms_onevision_student \
    --model_args "pretrained=${MODEL},student_path=${STUDENT},keep_ratio=0.1,device=cuda:0,attn_implementation=sdpa,conv_template=qwen_1_5" \
    --tasks chartqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix onevision_answer_imgkeep0p1 \
    --output_path "${OUT_ROOT}" \
    --seed 0 \
    2>&1 | tee "${LOG_FILE}"

echo "[complete] ChartQA OneVision answer-only student image keep 0.1"
