#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${EXP_DIR}/vqa_queue_state"
LOG_DIR="${EXP_DIR}/logs/vqa_scratch"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

if [[ ! -f "${STATE_DIR}/pending.tsv" ]]; then
    cp "${EXP_DIR}/vqa_jobs.tsv" "${STATE_DIR}/pending.tsv"
fi

echo "[queue] pending=$(wc -l < "${STATE_DIR}/pending.tsv")"

"${EXP_DIR}/run_vqa_queue_worker.sh" 0 351439 > "${LOG_DIR}/worker_gpu0.log" 2>&1 &
pid0=$!
"${EXP_DIR}/run_vqa_queue_worker.sh" 1 > "${LOG_DIR}/worker_gpu1.log" 2>&1 &
pid1=$!
"${EXP_DIR}/run_vqa_queue_worker.sh" 2 351441 > "${LOG_DIR}/worker_gpu2.log" 2>&1 &
pid2=$!
"${EXP_DIR}/run_vqa_queue_worker.sh" 3 > "${LOG_DIR}/worker_gpu3.log" 2>&1 &
pid3=$!

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
wait "${pid2}" || status=1
wait "${pid3}" || status=1

echo "[complete] scratch VQA queue status=${status}"
exit "${status}"
