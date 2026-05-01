#!/usr/bin/env bash
# Run missing OCR-doc student, keep_ratio=0.5 evaluations:
# GPU 1 → ocrdoc img50 ChartQA
# GPU 2 → ocrdoc img50 InfoVQA
set -euo pipefail

ROOT=/workspace/zap
PYTHON=/opt/conda/envs/kv/bin/python
RUNNER="${ROOT}/scripts/run_vlmeval_student.py"
LOGDIR="${ROOT}/experiments/EXP-20260428-003-onevision-ocrdoc-eval/outputs"
mkdir -p "${LOGDIR}"

run_eval() {
    local gpu=$1 tag=$2 config=$3 workdir=$4
    echo "[$(date '+%H:%M:%S')] GPU${gpu} START ${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${RUNNER}" -- \
        --config "${config}" \
        --work-dir "${workdir}" \
        2>&1 | tee "${LOGDIR}/${tag}.log"
    echo "[$(date '+%H:%M:%S')] GPU${gpu} DONE  ${tag}"
}

# GPU 1: ocrdoc img50 ChartQA
run_eval 1 "ocrdoc_img50_chartqa" \
    "${ROOT}/eval_configs/onevision_ocrdoc_img50_chartqa.json" \
    "${ROOT}/eval_results/ocrdoc_img50_chartqa" &
PID1=$!

# GPU 2: ocrdoc img50 InfoVQA
run_eval 2 "ocrdoc_img50_infovqa" \
    "${ROOT}/eval_configs/onevision_ocrdoc_img50_infovqa.json" \
    "${ROOT}/eval_results/ocrdoc_img50_infovqa" &
PID2=$!

echo "2 GPU workers launched. PIDs: ${PID1} ${PID2}"
wait "${PID1}" && echo "GPU1 done" || echo "GPU1 FAILED (exit $?)"
wait "${PID2}" && echo "GPU2 done" || echo "GPU2 FAILED (exit $?)"

echo "=== ALL DONE ==="
