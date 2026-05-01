#!/usr/bin/env bash
# Retry DocVQA (0/200) and InfoVQA (45/200) with --max-image-size 1024
set -euo pipefail

ROOT=/workspace/zap
LOGDIR="${ROOT}/experiments/EXP-20260428-001-onevision-ocrdoc-teacher/outputs"
PYTHON=/opt/conda/envs/kv/bin/python
COLLECTOR="${ROOT}/collect_future_teacher_v2_onevision.py"
OUT="${ROOT}/data/teacher_v2_onevision"

run_collect() {
    local gpu=$1 dataset=$2
    echo "[$(date '+%H:%M:%S')] GPU${gpu} START ${dataset}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${COLLECTOR}" \
        --dataset "${dataset}" \
        --n-samples 200 \
        --max-image-size 1024 \
        --output-root "${OUT}" \
        --device "cuda:0" \
        2>&1 | tee "${LOGDIR}/${dataset}_retry.log"
    echo "[$(date '+%H:%M:%S')] GPU${gpu} DONE  ${dataset}"
}

# docvqa → GPU 1,  infovqa → GPU 2  (GPU 0 free for other work)
run_collect 1 docvqa  &
PID1=$!
run_collect 2 infovqa &
PID2=$!

echo "PIDs: docvqa=${PID1} infovqa=${PID2}"
wait "${PID1}" && echo "docvqa done"  || echo "docvqa FAILED"
wait "${PID2}" && echo "infovqa done" || echo "infovqa FAILED"
echo "=== DONE ==="
