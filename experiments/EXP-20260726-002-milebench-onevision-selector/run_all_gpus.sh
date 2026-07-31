#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260726-002-milebench-onevision-selector"
OUTPUT_ROOT="${ZAP_ROOT}/artifacts/rebuttal_milebench_onevision_selector_total0p2"
PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
RUNNER="${EXP_DIR}/run_selector_comparison.py"
mkdir -p "${OUTPUT_ROOT}/logs"

datasets=("ALFRED" "CLEVR-Change" "IEdit" "Spot-the-Diff")
for gpu in 0 1 2 3; do
  dataset="${datasets[$gpu]}"
  session="milebench_selector_gpu${gpu}"
  log="${OUTPUT_ROOT}/logs/gpu${gpu}_${dataset}.log"
  screen -dmS "${session}" bash -lc \
    "CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/workspace/nips/Q-ViK:/workspace/nips/zap \
${PYTHON} -u ${RUNNER} \
--dataset ${dataset} \
--device cuda:0 \
--device-map cuda:0 \
--total-keep-ratio 0.2 \
--max-new-tokens 128 \
--output-root ${OUTPUT_ROOT} \
>${log} 2>&1"
  echo "${session}" >"${OUTPUT_ROOT}/logs/gpu${gpu}_${dataset}.session"
  echo "launched gpu=${gpu} dataset=${dataset} screen=${session}"
done
