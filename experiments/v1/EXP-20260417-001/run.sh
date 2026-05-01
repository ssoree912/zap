#!/usr/bin/env bash
# EXP-20260417-001: True Iterative Probe Pruning (n_rounds=4)
#
# Runs the same datasets/ratios as probe_mlp sweep, but saves results
# under iterative_4/ to avoid overwriting existing probe_mlp results.
#
# Usage:
#   bash experiments/EXP-20260417-001/run.sh
#
# Overrides:
#   KEEP_RATIOS="0.20"       (default: match existing probe_mlp sweep)
#   GPU_INDEX=1              (default: 0)
#   LIMIT=50                 (default: full dataset)
#   DATASETS="DocVQA IEdit"  (default: all available)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZAP_ROOT="${ZAP_ROOT:-/workspace/zap}"

GPU_INDEX="${GPU_INDEX:-0}"
KEEP_RATIOS="${KEEP_RATIOS:-0.20}"
LIMIT="${LIMIT:-}"
DATASETS="${DATASETS:-}"

N_ITERATIVE_ROUNDS=4

N_ITERATIVE_ROUNDS="${N_ITERATIVE_ROUNDS}" \
GPU_INDEX="${GPU_INDEX}" \
DATA_ROOT="${ZAP_ROOT}/data/MileBench" \
PROB_ARTIFACT_ROOT="${ZAP_ROOT}/artifacts/combine_prob" \
KEEP_RATIOS="${KEEP_RATIOS}" \
TEACHERS="att_only_postvision" \
METHODS="mlp" \
SKIP_EXISTING="1" \
TRUNCATE_LIKE_LOOKM="1" \
PROMPT_STYLE="look_milebench" \
MAX_NEW_TOKENS="32" \
PROBE_LABEL="combined" \
DATASETS="${DATASETS}" \
LIMIT="${LIMIT}" \
bash "${ZAP_ROOT}/scripts/run_milebench_probe_all.sh"

echo "EXP-20260417-001 done. Results under: ${ZAP_ROOT}/artifacts/combine_prob/*/iterative_${N_ITERATIVE_ROUNDS}/"
