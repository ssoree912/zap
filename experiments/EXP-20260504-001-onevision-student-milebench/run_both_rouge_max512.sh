#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="/workspace/zap/experiments/EXP-20260504-001-onevision-student-milebench"

screen -dmS llava15_rouge512_gpu0 bash -lc \
  "cd /workspace/zap && CUDA_VISIBLE_DEVICES=0 ${EXP_DIR}/run_llava15_rouge_max512_gpu0.sh"

screen -dmS onevision_rouge512_gpu1 bash -lc \
  "cd /workspace/zap && CUDA_VISIBLE_DEVICES=1 ${EXP_DIR}/run_rouge_extra_gpu1.sh"

echo "started: llava15_rouge512_gpu0"
echo "started: onevision_rouge512_gpu1"
echo "logs: ${EXP_DIR}/logs"
