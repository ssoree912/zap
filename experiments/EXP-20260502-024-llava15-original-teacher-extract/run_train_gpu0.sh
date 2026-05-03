#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/workspace/VFlowOpt/.conda/VFlowOpt"
EXP_DIR="/workspace/zap/experiments/EXP-20260502-024-llava15-original-teacher-extract"
PYTHON_SCRIPT="${EXP_DIR}/train_original_llava15_student.py"
OUT_DIR="/workspace/zap/artifacts/original_llava_teacher/student_llava15_original_future_1800_lr1e4_15ep"
LOG_DIR="${EXP_DIR}/logs"
LOG_FILE="${LOG_DIR}/train_gpu0_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

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
  --teacher-root /workspace/zap/artifacts/original_llava_teacher/future_decode_llava15_7b \
  --datasets gqa textvqa scienceqa \
  --llava-path /workspace/zap/ckpts/llava-v1.5-7b \
  --model-name llava-v1.5-7b \
  --device cuda:0 \
  --device-map cuda:0 \
  --epochs 15 \
  --lr 1e-4 \
  --n-per-dataset 600 \
  --seed 42 \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
