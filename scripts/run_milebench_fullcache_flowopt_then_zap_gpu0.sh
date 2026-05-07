#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd /workspace/zap

GPU="${GPU:-0}"
DATASETS="${DATASETS:-CLEVR-Change,IEdit,Spot-the-Diff,ALFRED}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

FLOWOPT_OUT="${FLOWOPT_OUT:-/workspace/VFlowOpt/experiments/EXP-20260507-001-milebench-vflowopt-onevision/outputs/fullcache_explicit4_max64_gpu${GPU}}"
FLOWOPT_LOG="${FLOWOPT_LOG:-${FLOWOPT_OUT}/run.log}"
ZAP_STUDENT_PATH="${ZAP_STUDENT_PATH:-/workspace/zap/ckpts/student_onevision_A_ep20}"

mkdir -p "${FLOWOPT_OUT}"

echo "===== START FlowOpt fullcache GPU=${GPU} $(date --iso-8601=seconds) ====="
CUDA_VISIBLE_DEVICES="${GPU}" /opt/conda/envs/vflowopt_chartqa_eval/bin/python \
  /workspace/zap/scripts/milebench_vflowopt_onevision.py \
  --datasets "${DATASETS}" \
  --keep-ratio 1.0 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device cuda:0 \
  --output-dir "${FLOWOPT_OUT}" \
  --overwrite \
  2>&1 | tee "${FLOWOPT_LOG}"
echo "===== DONE FlowOpt fullcache GPU=${GPU} $(date --iso-8601=seconds) ====="

echo "===== START ZAP fullcache GPU=${GPU} $(date --iso-8601=seconds) ====="
DATASET="${DATASETS}" \
GPU="${GPU}" \
KEEP_RATIO=1.0 \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
MODEL_FORMAT=hf \
STUDENT_PATH="${ZAP_STUDENT_PATH}" \
bash scripts/run_milebench_zap_student_onevision.sh
echo "===== DONE ZAP fullcache GPU=${GPU} $(date --iso-8601=seconds) ====="
