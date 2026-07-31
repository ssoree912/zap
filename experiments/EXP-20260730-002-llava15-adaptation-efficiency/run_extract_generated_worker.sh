#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

GPU="${1:?usage: run_extract_generated_worker.sh GPU DATASET}"
DATASET="${2:?usage: run_extract_generated_worker.sh GPU DATASET}"

WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MODEL="${WORKSPACE}/models/llava-v1.5-7b"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-002-llava15-adaptation-efficiency"
LOG_DIR="${EXP_DIR}/logs"

mkdir -p "${LOG_DIR}"

run_extract() {
    local per_dataset="$1"
    local output_root="$2"
    local log_file="${LOG_DIR}/teacher_generated_${DATASET}_n${per_dataset}_gpu${GPU}.log"

    echo "[teacher] gpu=${GPU} source=generated dataset=${DATASET} n=${per_dataset}"
    cd "${QVIK_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${QVIK_ROOT}:${WORKSPACE}/zap:${PYTHONPATH:-}" \
    TOKENIZERS_PARALLELISM=false \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" -m qvik.teacher.extract_llava15 \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --n-samples "${per_dataset}" \
        --max-new-tokens 32 \
        --seed 0 \
        --device cuda:0 \
        --output-root "${output_root}" \
        --problems-json "${WORKSPACE}/data/train/scienceqa/problems.json" \
        --images-root "${WORKSPACE}/data/train/scienceqa/images" \
        --gqa-questions-json "${WORKSPACE}/data/train/gqa/val_balanced_questions.json" \
        --gqa-images-root "${WORKSPACE}/data/train/gqa/images" \
        --textvqa-json "${WORKSPACE}/data/train/textvqa/train/data.json" \
        --textvqa-data-root "${WORKSPACE}/data/train" \
        --response-source generated \
        2>&1 | tee "${log_file}"
}

run_extract 300 "${WORKSPACE}/data/train/teacher/llava15_rff2_generated_n300perds_seed0"
run_extract 100 "${WORKSPACE}/data/train/teacher/llava15_rff2_generated_n100perds_seed0"
