#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Distribute Future Oracle eval jobs across multiple GPUs in parallel.
#
# Tasks: chartqa, textvqa_val, docvqa_val, gqa, coco2017_cap_val, nocaps_val, textcaps_val
# Keep ratios: 0.5, 0.2 -> 14 jobs total.
# Default split (4 GPUs):
#   gpu 2: docvqa_val:0.5, docvqa_val:0.2, gqa:0.5
#   gpu 3: chartqa:0.5, chartqa:0.2, textvqa_val:0.5, textvqa_val:0.2
#   gpu 4: coco2017_cap_val:0.5, coco2017_cap_val:0.2, gqa:0.2
#   gpu 5: nocaps_val:0.5, nocaps_val:0.2, textcaps_val:0.5, textcaps_val:0.2

set -euo pipefail

EXP_DIR="/mnt/srv/home/dlpc.3842/zap/experiments/EXP-20260505-001-llava15-future-oracle"
SHARD="${EXP_DIR}/run_eval_oracle_shard.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${EXP_DIR}/outputs/oracle_${STAMP}}"

mkdir -p "${OUT_ROOT}"
echo "Output root: ${OUT_ROOT}"

declare -A GPU_JOBS=(
  [2]="docvqa_val:0.5 docvqa_val:0.2 gqa:0.5"
  [3]="chartqa:0.5 chartqa:0.2 textvqa_val:0.5 textvqa_val:0.2"
  [4]="coco2017_cap_val:0.5 coco2017_cap_val:0.2 gqa:0.2"
  [5]="nocaps_val:0.5 nocaps_val:0.2 textcaps_val:0.5 textcaps_val:0.2"
)

declare -a PIDS=()
declare -A PID_TO_GPU=()
for gpu in "${!GPU_JOBS[@]}"; do
  jobs="${GPU_JOBS[$gpu]}"
  log="${OUT_ROOT}/shard_gpu${gpu}.log"
  echo "[launch] gpu=${gpu} jobs=[${jobs}] log=${log}"
  GPU_ID="${gpu}" OUT_ROOT="${OUT_ROOT}" bash "${SHARD}" ${jobs} >"${log}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  PID_TO_GPU[${pid}]="${gpu}"
done

echo "[launch] waiting on ${#PIDS[@]} shards (pids: ${PIDS[*]})"
status=0
for pid in "${PIDS[@]}"; do
  gpu="${PID_TO_GPU[${pid}]}"
  if wait "${pid}"; then
    echo "[done] gpu=${gpu} pid=${pid} ok"
  else
    rc=$?
    echo "[done] gpu=${gpu} pid=${pid} FAILED rc=${rc}"
    status=${rc}
  fi
done

echo "All shards finished. Output: ${OUT_ROOT}"
exit "${status}"
