#!/usr/bin/env bash
# Run ZAP student KV-pruning eval on LLaVA-OneVision 7B over full MileBench (28).
#
# Defaults: keep_ratio=0.2, combine_image=1 (single stitched grid), GPU 0.
# Override by env: GPU, KEEP_RATIO, COMBINE_IMAGE, MAX_NEW_TOKENS, DATASET, OUTPUT_DIR, LIMIT.
#
# Usage:
#   ./run_milebench_zap_student_onevision.sh
#   GPU=1 KEEP_RATIO=0.5 ./run_milebench_zap_student_onevision.sh
set -euo pipefail

GPU="${GPU:-0}"
KEEP_RATIO="${KEEP_RATIO:-0.2}"
COMBINE_IMAGE="${COMBINE_IMAGE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DATASET="${DATASET:-full}"
LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then LIMIT_FLAG="--limit ${LIMIT}"; fi
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then OVERWRITE_FLAG="--overwrite"; fi
COMBINE_FLAG=""
if [[ -n "${COMBINE_IMAGE}" && "${COMBINE_IMAGE}" != "0" ]]; then
    COMBINE_FLAG="--combine_image ${COMBINE_IMAGE}"
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZAP_ROOT="$(cd "${THIS_DIR}/.." && pwd)"

KEEP_TAG="$(printf '%s' "${KEEP_RATIO}" | tr '.' 'p')"
COMBINE_TAG="${COMBINE_IMAGE:+_combine${COMBINE_IMAGE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ZAP_ROOT}/logs/milebench_zap_student_onevision_keep${KEEP_TAG}${COMBINE_TAG}_max${MAX_NEW_TOKENS}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

ENV_PYTHON="${ENV_PYTHON:-/mnt/srv/home/dlpc.3842/look-m/.conda/lookm/bin/python}"
STUDENT_PATH="${STUDENT_PATH:-/mnt/srv/home/dlpc.3842/zap/artifacts/student_onevision_original_future_1800_lr1e4_15ep}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src/LLaVA-OneVision:/mnt/srv/home/dlpc.3842/look-m:/mnt/srv/home/dlpc.3842/zap:${PYTHONPATH:-}"

echo "[launch] model=onevision keep_ratio=${KEEP_RATIO} combine_image=${COMBINE_IMAGE} max_new_tokens=${MAX_NEW_TOKENS} dataset=${DATASET} gpu=${GPU}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] python=${ENV_PYTHON} student=${STUDENT_PATH}"

cd "${ZAP_ROOT}"
exec "${ENV_PYTHON}" "${THIS_DIR}/milebench_zap_student_onevision.py" \
    --keep_ratio "${KEEP_RATIO}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --student_path "${STUDENT_PATH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${COMBINE_FLAG} \
    ${LIMIT_FLAG} \
    ${OVERWRITE_FLAG} \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
