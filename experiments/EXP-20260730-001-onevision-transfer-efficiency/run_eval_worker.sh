#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

GPU="${1:?usage: run_eval_worker.sh GPU scratch|warm TOTAL_SAMPLES}"
MODE="${2:?usage: run_eval_worker.sh GPU scratch|warm TOTAL_SAMPLES}"
TOTAL_SAMPLES="${3:?usage: run_eval_worker.sh GPU scratch|warm TOTAL_SAMPLES}"

WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
ZAP_ROOT="${WORKSPACE}/zap"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MODEL="${WORKSPACE}/models/llava-onevision-qwen2-7b-ov"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260730-001-onevision-transfer-efficiency"
TAG="${MODE}_n${TOTAL_SAMPLES}_e15"
CKPT_ROOT="${EXP_DIR}/checkpoints/${TAG}"
LOG_DIR="${EXP_DIR}/logs"

mkdir -p "${LOG_DIR}"

for EPOCH in 1 3 5 15; do
    EPOCH_PADDED="$(printf '%03d' "${EPOCH}")"
    STUDENT="${CKPT_ROOT}/epoch_${EPOCH_PADDED}"
    CASE_TAG="${MODE}_n${TOTAL_SAMPLES}_epoch${EPOCH_PADDED}"
    OUT_ROOT="${EXP_DIR}/outputs/chartqa/${CASE_TAG}"
    LOG_FILE="${LOG_DIR}/eval_${CASE_TAG}_gpu${GPU}.log"

    if [[ ! -f "${STUDENT}/pytorch_model.bin" ]]; then
        echo "Missing milestone checkpoint: ${STUDENT}" >&2
        exit 1
    fi

    echo "[eval-config] gpu=${GPU} case=${CASE_TAG} image_keep_ratio=0.1"
    cd "${QVIK_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
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
        --log_samples_suffix "${CASE_TAG}" \
        --output_path "${OUT_ROOT}" \
        --seed 0 \
        2>&1 | tee "${LOG_FILE}"
done

echo "[eval-complete] ${TAG}"
