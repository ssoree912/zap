#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WAIT_PID="${1:-0}"
GPU="${2:-0}"
WORKSPACE="/workspace/nips"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-002-llava15-adaptation-efficiency"
TAG="vicuna_generated_n1800_e15"
TEACHER_ROOT="${WORKSPACE}/data/train/teacher/zap_llava15_n600_seed0"

echo "[wait] corrected full baseline gpu=${GPU} pid=${WAIT_PID}"
while [[ -r "/proc/${WAIT_PID}/cmdline" ]]; do
    cmdline="$(tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline" 2>/dev/null || true)"
    if [[ "${cmdline}" != *"${EXP_DIR}/run_extra_eval_condition.sh"* ]]; then
        break
    fi
    sleep 30
done

echo "[start] corrected answer-only full baseline"
"${EXP_DIR}/run_train_eval_condition.sh" \
    "${GPU}" \
    "${TAG}" \
    "${TEACHER_ROOT}" \
    600 \
    15

"${EXP_DIR}/run_extra_eval_condition.sh" "${GPU}" 0 "${TAG}"
echo "[complete] corrected answer-only full baseline and all evaluations"
