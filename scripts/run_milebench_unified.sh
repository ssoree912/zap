#!/usr/bin/env bash
# Run unified MileBench full-cache eval. Identical script in zap/, look-m/,
# VFlowOpt_llava1.5/. The folder this is launched from determines only the
# default OUTPUT_DIR — pred.json must match across folders for the same model.
#
# Usage:
#   GPU=0 MODEL=llava15  ./run_milebench_unified.sh
#   GPU=1 MODEL=onevision ./run_milebench_unified.sh
#   GPU=0 MODEL=llava15 LIMIT=2 ./run_milebench_unified.sh   # smoke test
set -euo pipefail

GPU="${GPU:-0}"
MODEL="${MODEL:?MODEL must be llava15 or onevision}"
DATASET="${DATASET:-all}"
LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then LIMIT_FLAG="--limit ${LIMIT}"; fi
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then OVERWRITE_FLAG="--overwrite"; fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
FOLDER_TAG="$(basename "${FOLDER_ROOT}")"

OUTPUT_DIR="${OUTPUT_DIR:-${FOLDER_ROOT}/logs/milebench_unified_${MODEL}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

ENV_PYTHON="${ENV_PYTHON:-/mnt/srv/home/dlpc.3842/look-m/.conda/lookm/bin/python}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src/LLaVA-OneVision:/mnt/srv/home/dlpc.3842/look-m:/mnt/srv/home/dlpc.3842/zap:${PYTHONPATH:-}"

echo "[launch] folder=${FOLDER_TAG} model=${MODEL} dataset=${DATASET} gpu=${GPU}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] python=${ENV_PYTHON}"

cd "${FOLDER_ROOT}"
exec "${ENV_PYTHON}" "${THIS_DIR}/milebench_unified.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    ${LIMIT_FLAG} \
    ${OVERWRITE_FLAG} \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
