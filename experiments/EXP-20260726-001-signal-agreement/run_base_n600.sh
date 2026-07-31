#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260726-001-signal-agreement"
PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
OUTPUT_DIR="${ZAP_ROOT}/artifacts/rebuttal_signal_agreement_llava15_base_n600"
SCRIPT="${EXP_DIR}/analyze_llava15_signal_agreement.py"

mkdir -p "${OUTPUT_DIR}/logs"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export QVIK_ROOT=/workspace/nips/Q-ViK

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u "${SCRIPT}" \
  --datasets textvqa scienceqa gqa \
  --device cuda:0 \
  --output-dir "${OUTPUT_DIR}" \
  >"${OUTPUT_DIR}/logs/extract.log" 2>&1

"${PYTHON}" -u "${SCRIPT}" \
  --summarize \
  --output-dir "${OUTPUT_DIR}" \
  >"${OUTPUT_DIR}/logs/summarize.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u \
  "${EXP_DIR}/run_llava15_selector_intervention.py" \
  --device cuda:0 \
  --total-keep-ratio 0.2 \
  --output-dir "${OUTPUT_DIR}/intervention_total_keep_0p2" \
  >"${OUTPUT_DIR}/logs/intervention_total_keep_0p2.log" 2>&1

echo "Signal-agreement results: ${OUTPUT_DIR}/summary"
echo "Selector intervention: ${OUTPUT_DIR}/intervention_total_keep_0p2"
