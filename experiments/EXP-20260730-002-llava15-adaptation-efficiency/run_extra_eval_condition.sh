#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

GPU="${1:?usage: run_extra_eval_condition.sh GPU WAIT_PID TAG}"
WAIT_PID="${2:?usage: run_extra_eval_condition.sh GPU WAIT_PID TAG}"
TAG="${3:?usage: run_extra_eval_condition.sh GPU WAIT_PID TAG}"

WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MODEL="${WORKSPACE}/models/llava-v1.5-7b"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-002-llava15-adaptation-efficiency"
STUDENT="${EXP_DIR}/checkpoints/${TAG}"
OUT_ROOT="${EXP_DIR}/outputs/${TAG}"
LOG_DIR="${EXP_DIR}/logs"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "[wait] gpu=${GPU} tag=${TAG} pid=${WAIT_PID}"
while [[ -r "/proc/${WAIT_PID}/cmdline" ]]; do
    cmdline="$(tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline" 2>/dev/null || true)"
    if [[ "${cmdline}" != *"${EXP_DIR}/run_train_eval_condition.sh"* &&
          "${cmdline}" != *"${EXP_DIR}/run_reference_pipeline.sh"* &&
          "${cmdline}" != *"${EXP_DIR}/run_extra_eval_condition.sh"* ]]; then
        break
    fi
    sleep 30
done

if [[ ! -f "${STUDENT}/pytorch_model.bin" ]]; then
    echo "[error] checkpoint missing after upstream worker: ${STUDENT}" >&2
    exit 1
fi

status=0
for task in gqa coco_cap nocaps textcaps; do
    task_out="${OUT_ROOT}/${task}"
    task_log="${LOG_DIR}/eval_${TAG}_${task}_gpu${GPU}.log"
    if find "${task_out}" -name '*_results.json' -print -quit 2>/dev/null | grep -q .; then
        echo "[skip] result exists tag=${TAG} task=${task}"
        continue
    fi

    echo "[eval] gpu=${GPU} tag=${TAG} task=${task} total_keep_ratio=0.2"
    cd "${QVIK_ROOT}"
    if ! CUDA_VISIBLE_DEVICES="${GPU}" \
    QVIK_DATA_ROOT="${WORKSPACE}/data/eval" \
    PATH="${WORKSPACE}/.conda/envs/qvik/lib/jvm/bin:${PATH}" \
    PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" qvik/eval/run_lmms_eval.py \
        --model lmms_llava15_student \
        --model_args "pretrained=${MODEL},student_path=${STUDENT},keep_ratio=0.2,keep_ratio_basis=total,device=cuda:0,device_map=cuda:0,conv_template=vicuna_v1" \
        --tasks "${task}" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "${TAG}_${task}" \
        --output_path "${task_out}" \
        --seed 0 \
        2>&1 | tee "${task_log}"; then
        echo "[error] evaluation failed tag=${TAG} task=${task}" >&2
        status=1
    fi
done

if (( status != 0 )); then
    exit 1
fi

echo "[complete] extra tasks ${TAG}"
