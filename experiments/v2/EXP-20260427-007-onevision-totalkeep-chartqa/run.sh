#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/zap
CASE="${1:?usage: bash run.sh total50|total10|total05}"

case "${CASE}" in
  total50)
    GPU=0
    CONFIG="${ROOT}/eval_configs/onevision_student_total50_chartqa.json"
    WORK_DIR="${ROOT}/eval_results/student_total50_chartqa"
    LOG="${ROOT}/eval_results/student_total_logs/chartqa_total50.log"
    ;;
  total10)
    GPU=1
    CONFIG="${ROOT}/eval_configs/onevision_student_total10_chartqa.json"
    WORK_DIR="${ROOT}/eval_results/student_total10_chartqa"
    LOG="${ROOT}/eval_results/student_total_logs/chartqa_total10.log"
    ;;
  total05)
    GPU=2
    CONFIG="${ROOT}/eval_configs/onevision_student_total05_chartqa.json"
    WORK_DIR="${ROOT}/eval_results/student_total05_chartqa"
    LOG="${ROOT}/eval_results/student_total_logs/chartqa_total05.log"
    ;;
  *)
    echo "unknown case: ${CASE}" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "${LOG}")" "${WORK_DIR}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" /opt/conda/envs/kv/bin/python \
  "${ROOT}/scripts/run_vlmeval_student.py" -- \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  2>&1 | tee "${LOG}"
