#!/usr/bin/env bash
# ChartQA evaluation via lmms-eval with student KV pruning.
# GPU 2 → ocrdoc student, keep_ratio=0.25
set -euo pipefail

PYTHON=/opt/conda/envs/kv/bin/python
RUNNER=/workspace/zap/scripts/run_lmms_eval_student.py
BASE_MODEL=/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf
OCRDOC_STUDENT=/workspace/zap/ckpts/student_onevision_ocrdoc_lr1e4_20ep
OUT_ROOT=/workspace/zap/eval_results/lmms_chartqa_student
LOGDIR=/workspace/zap/experiments/EXP-20260429-lmms-chartqa-student/logs
mkdir -p "${LOGDIR}" "${OUT_ROOT}/ocrdoc_img25"

echo "[$(date '+%H:%M:%S')] GPU2 START ocrdoc_img25"
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${RUNNER}" -- \
    --model llava_onevision_student \
    --model_args "pretrained=${BASE_MODEL},student_path=${OCRDOC_STUDENT},keep_ratio=0.25" \
    --tasks chartqa_local \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ocrdoc_img25 \
    --output_path "${OUT_ROOT}/ocrdoc_img25" \
    2>&1 | tee "${LOGDIR}/ocrdoc_img25.log"
echo "[$(date '+%H:%M:%S')] GPU2 DONE  ocrdoc_img25"
