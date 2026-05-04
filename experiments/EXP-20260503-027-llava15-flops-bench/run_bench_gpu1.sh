#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/workspace/VFlowOpt/.conda/VFlowOpt"
EXP_DIR="/workspace/zap/experiments/EXP-20260503-027-llava15-flops-bench"
PYTHON_SCRIPT="${EXP_DIR}/bench_original_llava15_flops.py"
OUT_DIR="${EXP_DIR}/outputs/run_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EXP_DIR}/logs"
LOG_FILE="${LOG_DIR}/bench_gpu1_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_DIR}"

cd /workspace/zap

echo "[log] ${LOG_FILE}"
echo "[out] ${OUT_DIR}"
python "${PYTHON_SCRIPT}" \
  --milebench-root /workspace/zap/data/MileBench \
  --sample-size 100 \
  --seed 42 \
  --output-dir "${OUT_DIR}" \
  --model-path /workspace/zap/ckpts/llava-v1.5-7b \
  --model-name llava-v1.5-7b \
  --conv-template vicuna_v1 \
  --ratios 1.0 0.5 0.2 \
  --device cuda:0 \
  --device-map cuda:0 \
  --attn-implementation sdpa \
  --max-new-tokens 32 \
  --max-prompt-mm-tokens 3900 \
  --warmup-samples 1 \
  --continue-on-error \
  2>&1 | tee "${LOG_FILE}"
