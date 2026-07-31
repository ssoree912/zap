#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

GPU="${1:?usage: run_train_case.sh GPU scratch|warm TOTAL_SAMPLES}"
MODE="${2:?usage: run_train_case.sh GPU scratch|warm TOTAL_SAMPLES}"
TOTAL_SAMPLES="${3:?usage: run_train_case.sh GPU scratch|warm TOTAL_SAMPLES}"

if [[ "${MODE}" != "scratch" && "${MODE}" != "warm" ]]; then
    echo "MODE must be scratch or warm, got ${MODE}" >&2
    exit 2
fi
if (( TOTAL_SAMPLES % 3 != 0 )); then
    echo "TOTAL_SAMPLES must be divisible by 3, got ${TOTAL_SAMPLES}" >&2
    exit 2
fi

WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
ZAP_ROOT="${WORKSPACE}/zap"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MODEL="${WORKSPACE}/models/llava-onevision-qwen2-7b-ov"
TEACHER="${WORKSPACE}/data/train/teacher/zap_onevision_answer_n600_seed0"
LLAVA15_STUDENT="${QVIK_ROOT}/ckpts/student_llava15_fixed_gpu0"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260730-001-onevision-transfer-efficiency"
PER_DATASET=$((TOTAL_SAMPLES / 3))
TAG="${MODE}_n${TOTAL_SAMPLES}_e15"
OUT_DIR="${EXP_DIR}/checkpoints/${TAG}"
LOG_DIR="${EXP_DIR}/logs"
LOG_FILE="${LOG_DIR}/train_${TAG}_gpu${GPU}.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

WARM_ARGS=()
if [[ "${MODE}" == "warm" ]]; then
    WARM_ARGS=(
        --warm-start-llava15 "${LLAVA15_STUDENT}"
        --warm-start-mode adapt_all
    )
fi

echo "[train-config] gpu=${GPU} mode=${MODE} total=${TOTAL_SAMPLES} per_dataset=${PER_DATASET}"
echo "[train-config] output=${OUT_DIR}"

cd "${QVIK_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" \
PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PYTHON}" -m qvik.train.llava_onevision \
    --teacher-root "${TEACHER}" \
    --datasets textvqa gqa scienceqa \
    --per-ds-limit "${PER_DATASET}" \
    --llava-path "${MODEL}" \
    --epochs 15 \
    --lr 1e-4 \
    --seed 0 \
    --device cuda:0 \
    --output-dir "${OUT_DIR}" \
    --save-epochs 1 3 5 10 15 \
    "${WARM_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo "[train-complete] ${TAG}"
