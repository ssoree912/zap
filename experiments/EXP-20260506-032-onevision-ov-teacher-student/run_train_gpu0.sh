#!/usr/bin/env bash
# EXP-20260506-032 : OneVision student 훈련 (teacher: ov format, 1800 samples, 20 epochs)
# run_extract_gpu0.sh 완료 후 실행할 것
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/workspace/zap:/workspace/VFlowOpt/src/LLaVA-OneVision:/workspace/VFlowOpt/src/transformers-4.46.0/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/conda/envs/vflowopt_chartqa_eval/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="/opt/conda/envs/vflowopt_chartqa_eval/bin/python"

EXP_DIR="/workspace/zap/experiments/EXP-20260506-032-onevision-ov-teacher-student"
TEACHER_ROOT="/workspace/zap/data/train/teacher_onevision_ov"
STUDENT_OUT="/workspace/zap/ckpts/student_onevision_B_ep10"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}" "${STUDENT_OUT}"

LOG_FILE="${LOG_DIR}/train_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== EXP-032 student train START $(date -Is) ===" | tee "${LOG_FILE}"
echo "TEACHER_ROOT=${TEACHER_ROOT}" | tee -a "${LOG_FILE}"
echo "STUDENT_OUT=${STUDENT_OUT}" | tee -a "${LOG_FILE}"

${PYTHON} /workspace/zap/experiments/EXP-20260503-026-onevision-original-teacher-train/train_original_onevision_student.py \
  --teacher-root "${TEACHER_ROOT}" \
  --datasets textvqa gqa scienceqa \
  --per-ds-limit 300 \
  --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
  --model-name llava_qwen \
  --device cuda:0 \
  --device-map "cuda:0" \
  --epochs 10 \
  --lr 1e-4 \
  --output-dir "${STUDENT_OUT}" \
  --seed 42 \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== EXP-032 student train DONE $(date -Is) ===" | tee -a "${LOG_FILE}"
