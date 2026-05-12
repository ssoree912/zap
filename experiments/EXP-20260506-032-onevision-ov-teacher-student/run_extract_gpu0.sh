#!/usr/bin/env bash
# EXP-20260506-032 : OneVision teacher 추출 (llava-onevision-qwen2-7b-ov, LLaVA format)
# textvqa 600 + gqa 600 + scienceqa 600 = 1800 samples
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/workspace/zap:/workspace/VFlowOpt/src/LLaVA-OneVision:/workspace/VFlowOpt/src/transformers-4.46.0/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/conda/envs/vflowopt_chartqa_eval/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="/opt/conda/envs/vflowopt_chartqa_eval/bin/python"

EXP_DIR="/workspace/zap/experiments/EXP-20260506-032-onevision-ov-teacher-student"
OUT_ROOT="/workspace/zap/data/train/teacher_onevision_ov"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

LOG_FILE="${LOG_DIR}/extract_gpu0_$(date +%Y%m%d_%H%M%S).log"

echo "=== EXP-032 teacher extract START $(date -Is) ===" | tee "${LOG_FILE}"
echo "OUT_ROOT=${OUT_ROOT}" | tee -a "${LOG_FILE}"

${PYTHON} "${EXP_DIR}/collect_teacher.py" \
  --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
  --model-name llava_qwen \
  --conv-template qwen_1_5 \
  --device cuda:0 \
  --device-map "cuda:0" \
  --datasets textvqa gqa scienceqa \
  --n-samples 300 \
  --max-new-tokens 64 \
  --seed 42 \
  --output-root "${OUT_ROOT}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== EXP-032 teacher extract DONE $(date -Is) ===" | tee -a "${LOG_FILE}"
