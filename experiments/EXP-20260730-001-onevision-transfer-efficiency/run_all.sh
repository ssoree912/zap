#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${EXP_DIR}/logs"
PIPELINE_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[stage] four-GPU training"
"${EXP_DIR}/run_train_case.sh" 0 scratch 450 &
pid0=$!
"${EXP_DIR}/run_train_case.sh" 1 warm 450 &
pid1=$!
"${EXP_DIR}/run_train_case.sh" 2 scratch 900 &
pid2=$!
"${EXP_DIR}/run_train_case.sh" 3 warm 900 &
pid3=$!

status0=0
status1=0
status2=0
status3=0
wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
wait "${pid2}" || status2=$?
wait "${pid3}" || status3=$?
echo "[train-status] gpu0=${status0} gpu1=${status1} gpu2=${status2} gpu3=${status3}"
if (( status0 != 0 || status1 != 0 || status2 != 0 || status3 != 0 )); then
    exit 1
fi

echo "[stage] four-GPU ChartQA evaluation"
"${EXP_DIR}/run_eval_worker.sh" 0 scratch 450 &
pid0=$!
"${EXP_DIR}/run_eval_worker.sh" 1 warm 450 &
pid1=$!
"${EXP_DIR}/run_eval_worker.sh" 2 scratch 900 &
pid2=$!
"${EXP_DIR}/run_eval_worker.sh" 3 warm 900 &
pid3=$!

status0=0
status1=0
status2=0
status3=0
wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
wait "${pid2}" || status2=$?
wait "${pid3}" || status3=$?
echo "[eval-status] gpu0=${status0} gpu1=${status1} gpu2=${status2} gpu3=${status3}"
if (( status0 != 0 || status1 != 0 || status2 != 0 || status3 != 0 )); then
    exit 1
fi

/workspace/nips/.conda/envs/qvik/bin/python "${EXP_DIR}/summarize_results.py"
echo "[complete] OneVision transfer and data-efficiency experiment"
