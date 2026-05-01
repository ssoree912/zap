#!/usr/bin/env bash
set -euo pipefail

ZAP_ROOT="${ZAP_ROOT:-/workspace/zap}"
GPU_INDEX="${GPU_INDEX:-1}"
KEEP_RATIOS="${KEEP_RATIOS:-0.20}"
LIMIT="${LIMIT:-}"
DATASETS="${DATASETS:-}"

N_ITERATIVE_ROUNDS=4
LAYERWISE_ITERATIVE=1

N_ITERATIVE_ROUNDS="${N_ITERATIVE_ROUNDS}" \
LAYERWISE_ITERATIVE="${LAYERWISE_ITERATIVE}" \
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

echo "iterative_ver2_4 done. Results: ${ZAP_ROOT}/artifacts/combine_prob/*/iterative_ver2_4/"
