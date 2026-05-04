#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="/workspace/zap/experiments/EXP-20260504-001-onevision-student-milebench"
PYTHON="/workspace/look-m/.conda/look-m/bin/python"

"${PYTHON}" "${EXP_DIR}/score_rouge_only_milebench.py" \
  --output-root "${EXP_DIR}/outputs/student_milebench_max512_llava15" \
  --keep-tags keep050 keep010 keep005 \
  --summary-dir "${EXP_DIR}/outputs/summary_max512/llava15"

"${PYTHON}" "${EXP_DIR}/score_rouge_only_milebench.py" \
  --output-root "${EXP_DIR}/outputs/student_milebench_max512_onevision" \
  --keep-tags keep050 keep010 keep005 \
  --summary-dir "${EXP_DIR}/outputs/summary_max512/onevision"
