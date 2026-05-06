#!/usr/bin/env bash
# Run ZAP student KV-pruning eval on LLaVA-1.5 7B over the full MileBench (28).
#
# Defaults match the local request: keep_ratio=0.1, max_new_tokens=64, GPU 0.
# Override by env: GPU, KEEP_RATIO, MAX_NEW_TOKENS, DATASET, OUTPUT_DIR, LIMIT.
#
# Usage:
#   ./run_milebench_zap_student_llava15.sh
#   GPU=1 KEEP_RATIO=0.05 ./run_milebench_zap_student_llava15.sh
set -euo pipefail

GPU="${GPU:-0}"
KEEP_RATIO="${KEEP_RATIO:-0.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DATASET="${DATASET:-full}"
LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then LIMIT_FLAG="--limit ${LIMIT}"; fi
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then OVERWRITE_FLAG="--overwrite"; fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZAP_ROOT="$(cd "${THIS_DIR}/.." && pwd)"

KEEP_TAG="$(printf '%s' "${KEEP_RATIO}" | tr '.' 'p')"
OUTPUT_DIR="${OUTPUT_DIR:-${ZAP_ROOT}/logs/milebench_zap_student_llava15_keep${KEEP_TAG}_max${MAX_NEW_TOKENS}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

ENV_PYTHON="${ENV_PYTHON:-/mnt/srv/home/dlpc.3842/look-m/.conda/lookm/bin/python}"
STUDENT_PATH="${STUDENT_PATH:-/mnt/srv/home/dlpc.3842/zap/artifacts/student_llava15_original_future_1800_lr1e4_15ep}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/mnt/srv/home/dlpc.3842/look-m/LLaVA-mix_merge_v1:/mnt/srv/home/dlpc.3842/look-m:/mnt/srv/home/dlpc.3842/zap:${PYTHONPATH:-}"

echo "[launch] model=llava15 keep_ratio=${KEEP_RATIO} max_new_tokens=${MAX_NEW_TOKENS} dataset=${DATASET} gpu=${GPU}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] python=${ENV_PYTHON} student=${STUDENT_PATH}"

cd "${ZAP_ROOT}"
exec "${ENV_PYTHON}" "${THIS_DIR}/milebench_zap_student.py" \
    --keep_ratio "${KEEP_RATIO}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --student_path "${STUDENT_PATH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${LIMIT_FLAG} \
    ${OVERWRITE_FLAG} \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
