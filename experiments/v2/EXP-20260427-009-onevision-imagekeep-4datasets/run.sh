#!/usr/bin/env bash
# EXP-20260427-009: OneVision image-keep sweep × 4 datasets
# keep_ratio: 1.0(full) / 0.5 / 0.1 / 0.05 / 0.01
# datasets:   ChartQA_TEST, InfoVQA_VAL, DocVQA_VAL, TextVQA_VAL
#
# GPU mapping:
#   GPU 0 → full (1.0) then 0.01   (sequential)
#   GPU 1 → 0.5        then 0.05   (sequential)
#   GPU 2 → 0.1                    (single)
set -euo pipefail

ROOT=/workspace/zap
LOGDIR="${ROOT}/experiments/EXP-20260427-009-onevision-imagekeep-4datasets/outputs"
PYTHON=/opt/conda/envs/kv/bin/python
RUNNER="${ROOT}/scripts/run_vlmeval_student.py"

mkdir -p "${LOGDIR}"

run_ratio() {
    local gpu=$1 ratio_tag=$2 config=$3 workdir=$4 logfile=$5
    echo "[$(date '+%H:%M:%S')] GPU${gpu} START ${ratio_tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${RUNNER}" -- \
        --config "${config}" \
        --work-dir "${workdir}" \
        2>&1 | tee "${logfile}"
    echo "[$(date '+%H:%M:%S')] GPU${gpu} DONE  ${ratio_tag}"
}

# GPU 0: full → 0.01
(
    run_ratio 0 "img_full" \
        "${ROOT}/eval_configs/onevision_imgfull_4datasets.json" \
        "${ROOT}/eval_results/img_full_4datasets" \
        "${LOGDIR}/gpu0_full.log"
    run_ratio 0 "img_01" \
        "${ROOT}/eval_configs/onevision_img01_4datasets.json" \
        "${ROOT}/eval_results/img01_4datasets" \
        "${LOGDIR}/gpu0_01.log"
) &
PID0=$!

# GPU 1: 0.5 → 0.05
(
    run_ratio 1 "img_50" \
        "${ROOT}/eval_configs/onevision_img50_4datasets.json" \
        "${ROOT}/eval_results/img50_4datasets" \
        "${LOGDIR}/gpu1_50.log"
    run_ratio 1 "img_05" \
        "${ROOT}/eval_configs/onevision_img05_4datasets.json" \
        "${ROOT}/eval_results/img05_4datasets" \
        "${LOGDIR}/gpu1_05.log"
) &
PID1=$!

# GPU 2: 0.1
(
    run_ratio 2 "img_10" \
        "${ROOT}/eval_configs/onevision_img10_4datasets.json" \
        "${ROOT}/eval_results/img10_4datasets" \
        "${LOGDIR}/gpu2_10.log"
) &
PID2=$!

echo "All 3 GPU workers launched. PIDs: ${PID0} ${PID1} ${PID2}"
wait "${PID0}" && echo "GPU0 chain done" || echo "GPU0 chain FAILED (exit $?)"
wait "${PID1}" && echo "GPU1 chain done" || echo "GPU1 chain FAILED (exit $?)"
wait "${PID2}" && echo "GPU2 done"       || echo "GPU2 FAILED (exit $?)"

echo "=== ALL DONE ==="
