#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/workspace/VFlowOpt/.conda/VFlowOpt"
EXP_DIR="/workspace/zap/experiments/EXP-20260502-024-llava15-original-teacher-extract"
PYTHON_SCRIPT="${EXP_DIR}/collect_original_llava15_teacher.py"
OUT_ROOT="/workspace/zap/artifacts/original_llava_teacher/future_decode_llava15_7b"
LOG_DIR="${EXP_DIR}/logs"
LOG_FILE="${LOG_DIR}/extract_scienceqa_gpu0_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_DIR}"

cd /workspace/zap

echo "[log] ${LOG_FILE}"
python "${PYTHON_SCRIPT}" \
  --model-path /workspace/zap/ckpts/llava-v1.5-7b \
  --model-name llava-v1.5-7b \
  --conv-template vicuna_v1 \
  --device cuda:0 \
  --device-map cuda:0 \
  --datasets scienceqa \
  --n-samples 600 \
  --max-new-tokens 64 \
  --scienceqa-prompt-style choices_only \
  --seed 42 \
  --overwrite \
  --output-root "${OUT_ROOT}" \
  2>&1 | tee "${LOG_FILE}"
