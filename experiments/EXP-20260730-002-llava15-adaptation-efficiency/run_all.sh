#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/workspace/nips"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "[stage] scratch training/evaluation with shared Vicuna-prompt teacher cache"
"${EXP_DIR}/run_train_eval_condition.sh" \
    0 \
    vicuna_generated_n900_e15 \
    "${WORKSPACE}/data/train/teacher/zap_llava15_n600_seed0" \
    300 \
    15 &
half=$!

"${EXP_DIR}/run_train_eval_condition.sh" \
    1 \
    vicuna_generated_n300_e15 \
    "${WORKSPACE}/data/train/teacher/zap_llava15_n600_seed0" \
    100 \
    15 &
sixth=$!

"${EXP_DIR}/run_train_eval_condition.sh" \
    3 \
    vicuna_generated_n1800_e3 \
    "${WORKSPACE}/data/train/teacher/zap_llava15_n600_seed0" \
    600 \
    3 &
full3=$!

"${EXP_DIR}/run_reference_pipeline.sh" 2 &
reference=$!

wait "${full3}"
wait "${half}"
wait "${sixth}"
wait "${reference}"

"${WORKSPACE}/.conda/envs/qvik/bin/python" "${EXP_DIR}/summarize_results.py"
echo "[complete] LLaVA-1.5 adaptation-efficiency experiment"
