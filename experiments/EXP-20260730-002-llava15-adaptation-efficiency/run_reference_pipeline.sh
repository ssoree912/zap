#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

GPU="${1:?usage: run_reference_pipeline.sh GPU}"
WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MODEL="${WORKSPACE}/models/llava-v1.5-7b"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-002-llava15-adaptation-efficiency"
MANIFEST_ROOT="${WORKSPACE}/data/train/teacher/zap_llava15_n600_seed0"
TEACHER_ROOT="${WORKSPACE}/data/train/teacher/llava15_rff3_reference_vicuna_n600perds_seed0"
LOG_DIR="${EXP_DIR}/logs"

mkdir -p "${LOG_DIR}"

for dataset in textvqa scienceqa gqa; do
    mkdir -p "${TEACHER_ROOT}/${dataset}"
    if [[ "$(find "${TEACHER_ROOT}/${dataset}" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" -ge 600 ]]; then
        echo "[skip] reference teacher exists dataset=${dataset}"
        continue
    fi
    echo "[teacher] gpu=${GPU} source=reference dataset=${dataset} n=600"
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
        --dataset "${dataset}" \
        --n-samples 600 \
        --max-new-tokens 32 \
        --seed 0 \
        --device cuda:0 \
        --output-root "${TEACHER_ROOT}" \
        --problems-json "${WORKSPACE}/data/train/scienceqa/problems.json" \
        --images-root "${WORKSPACE}/data/train/scienceqa/images" \
        --gqa-questions-json "${WORKSPACE}/data/train/gqa/val_balanced_questions.json" \
        --gqa-images-root "${WORKSPACE}/data/train/gqa/images" \
        --textvqa-json "${WORKSPACE}/data/train/textvqa/train/data.json" \
        --textvqa-data-root "${WORKSPACE}/data/train" \
        --sample-manifest-root "${MANIFEST_ROOT}" \
        --response-source reference \
        2>&1 | tee "${LOG_DIR}/teacher_reference_${dataset}_n600_gpu${GPU}.log"
done

"${EXP_DIR}/run_train_eval_condition.sh" \
    "${GPU}" \
    vicuna_reference_n1800_e15 \
    "${TEACHER_ROOT}" \
    600 \
    15
