#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="/workspace/zap/experiments/EXP-20260504-001-onevision-student-milebench"

echo "=== GPU0 sequential ROUGE run START $(date -Is) ==="
echo "Step 1/2: LLaVA-1.5 original, max_new_tokens=512"
CUDA_VISIBLE_DEVICES=0 MAX_NEW_TOKENS=512 "${EXP_DIR}/run_llava15_rouge_max512_gpu0.sh"

echo "Step 2/2: LLaVA-OneVision, max_new_tokens=64"
CUDA_VISIBLE_DEVICES=0 MAX_NEW_TOKENS=64 "${EXP_DIR}/run_onevision_rouge_max64_gpu0.sh"

echo "=== GPU0 sequential ROUGE run DONE $(date -Is) ==="
