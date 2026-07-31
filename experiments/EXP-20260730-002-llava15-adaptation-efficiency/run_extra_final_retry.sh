#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="/workspace/nips/zap/experiments/EXP-20260730-002-llava15-adaptation-efficiency"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"

declare -a workers=(
    "0 369939 vicuna_generated_n900_e15"
    "1 369940 vicuna_generated_n300_e15"
    "3 369941 vicuna_generated_n1800_e3"
    "2 369942 vicuna_reference_n1800_e15"
)

pids=()
for spec in "${workers[@]}"; do
    read -r gpu wait_pid tag <<< "${spec}"
    "${EXP_DIR}/run_extra_eval_condition.sh" "${gpu}" "${wait_pid}" "${tag}" \
        > "${LOG_DIR}/extra_final_retry_${tag}_gpu${gpu}.log" 2>&1 &
    worker_pid="$!"
    pids+=("${worker_pid}")
    echo "[launch] gpu=${gpu} wait_pid=${wait_pid} tag=${tag} worker_pid=${worker_pid}"
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

if (( status != 0 )); then
    echo "[error] one or more final retry workers failed" >&2
    exit 1
fi

echo "[complete] all extra evaluations verified"
