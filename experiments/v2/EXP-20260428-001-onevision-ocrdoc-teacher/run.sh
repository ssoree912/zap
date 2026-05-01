#!/usr/bin/env bash
# EXP-20260428-001: OneVision teacher extraction for OCR/chart/doc datasets
# chartqa (200) → GPU 0
# docvqa  (200) → GPU 1
# infovqa (200) → GPU 2
set -euo pipefail

ROOT=/workspace/zap
LOGDIR="${ROOT}/experiments/EXP-20260428-001-onevision-ocrdoc-teacher/outputs"
PYTHON=/opt/conda/envs/kv/bin/python
COLLECTOR="${ROOT}/collect_future_teacher_v2_onevision.py"
OUT="${ROOT}/data/teacher_v2_onevision"

mkdir -p "${LOGDIR}"

run_collect() {
    local gpu=$1 dataset=$2
    echo "[$(date '+%H:%M:%S')] GPU${gpu} START ${dataset}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${COLLECTOR}" \
        --dataset "${dataset}" \
        --n-samples 200 \
        --output-root "${OUT}" \
        --device "cuda:0" \
        2>&1 | tee "${LOGDIR}/${dataset}.log"
    echo "[$(date '+%H:%M:%S')] GPU${gpu} DONE  ${dataset}"
}

run_collect 0 chartqa &
PID0=$!
run_collect 1 docvqa  &
PID1=$!
run_collect 2 infovqa &
PID2=$!

echo "PIDs: chartqa=${PID0} docvqa=${PID1} infovqa=${PID2}"
wait "${PID0}" && echo "chartqa done" || echo "chartqa FAILED"
wait "${PID1}" && echo "docvqa done"  || echo "docvqa FAILED"
wait "${PID2}" && echo "infovqa done" || echo "infovqa FAILED"

echo "=== ALL DONE ==="
echo "Results in ${OUT}/{chartqa,docvqa,infovqa}/"
