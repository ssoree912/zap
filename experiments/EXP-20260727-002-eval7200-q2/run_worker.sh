#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 PHYSICAL_GPU MILEBENCH_TASK..." >&2
  exit 2
fi

physical_gpu=$1
shift

exp_dir=/workspace/nips/zap/experiments/EXP-20260727-002-eval7200-q2
python_bin=/workspace/nips/.conda/envs/qvik/bin/python
output_dir=${OUTPUT_DIR:-/workspace/nips/zap/artifacts/rebuttal_eval7200_q2_llava15_keep0p2}
budget_mode=${MILEBENCH_BUDGET_MODE:-visual}
keep_ratio=${KEEP_RATIO:-0.2}
max_prompt_tokens=${MAX_PROMPT_TOKENS:-4096}
max_new_tokens=${MILEBENCH_MAX_NEW_TOKENS:-128}

mkdir -p "${output_dir}/logs"
echo "[worker] gpu=${physical_gpu} tasks=$* mode=${budget_mode} ratio=${keep_ratio} max_new_tokens=${max_new_tokens}"

CUDA_VISIBLE_DEVICES="${physical_gpu}" \
PYTHONPATH=/workspace/nips/Q-ViK:/workspace/nips/zap \
"${python_bin}" -u "${exp_dir}/run_eval7200_q2.py" \
  --datasets "$@" \
  --n-samples 200 \
  --sample-seed 42 \
  --device cuda:0 \
  --worker-id "physical_gpu${physical_gpu}" \
  --keep-ratio "${keep_ratio}" \
  --milebench-budget-mode "${budget_mode}" \
  --max-prompt-tokens "${max_prompt_tokens}" \
  --milebench-max-new-tokens "${max_new_tokens}" \
  --output-dir "${output_dir}"
