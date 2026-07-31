#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE="/workspace/nips"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-002-llava15-adaptation-efficiency"
TEACHER_ROOT="${WORKSPACE}/data/train/teacher/zap_llava15_n600_seed0"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"

run_condition() {
    local gpu="$1"
    local tag="$2"
    local epochs="$3"

    "${EXP_DIR}/run_train_eval_condition.sh" \
        "${gpu}" \
        "${tag}" \
        "${TEACHER_ROOT}" \
        600 \
        "${epochs}" \
        final

    "${EXP_DIR}/run_extra_eval_condition.sh" "${gpu}" 0 "${tag}"
}

run_condition 0 vicuna_generated_n1800_e7 7 \
    > "${LOG_DIR}/pipeline_vicuna_generated_n1800_e7_gpu0.log" 2>&1 &
epoch7_pid=$!
echo "[launch] gpu=0 epochs=7 pid=${epoch7_pid}"

run_condition 1 vicuna_generated_n1800_e10 10 \
    > "${LOG_DIR}/pipeline_vicuna_generated_n1800_e10_gpu1.log" 2>&1 &
epoch10_pid=$!
echo "[launch] gpu=1 epochs=10 pid=${epoch10_pid}"

status=0
wait "${epoch7_pid}" || status=1
wait "${epoch10_pid}" || status=1

if (( status != 0 )); then
    echo "[error] one or more epoch conditions failed" >&2
    exit 1
fi

echo "[complete] 1,800-sample 7/10-epoch training and all evaluations"
