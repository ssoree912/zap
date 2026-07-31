#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260726-003-h2o-q2"
OUTPUT="${OUTPUT:-${ZAP_ROOT}/artifacts/rebuttal_h2o_q2_llava15_base_n600_total0p2}"
PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
RUNNER="${EXP_DIR}/run_llava15_h2o_q2.py"
mkdir -p "${OUTPUT}/logs"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export QVIK_ROOT=/workspace/nips/Q-ViK
export PYTHONPATH="${ZAP_ROOT}:${QVIK_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

datasets=("textvqa" "scienceqa" "gqa")
for gpu in 0 1 2; do
  dataset="${datasets[$gpu]}"
  session="h2o_q2_${dataset}"
  log="${OUTPUT}/logs/gpu${gpu}_${dataset}.log"
  screen -dmS "${session}" bash -lc \
    "CUDA_VISIBLE_DEVICES=${gpu} \
${PYTHON} -u ${RUNNER} \
--datasets ${dataset} \
--device cuda:0 \
--total-keep-ratio 0.2 \
--output-dir ${OUTPUT} \
--extract-only >${log} 2>&1"
  echo "launched gpu=${gpu} dataset=${dataset} screen=${session}"
done
