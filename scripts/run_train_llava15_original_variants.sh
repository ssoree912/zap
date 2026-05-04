#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/M2026107/zap"
PYTHON_BIN="/home/M2026107/.conda/envs/kv/bin/python"
TRAIN_SCRIPT="${ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract/train_original_llava15_student.py"

export MPLCONFIGDIR=/tmp/mplconfig
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export ZAP_REPO_ROOT="${ROOT}"
export VFLOWOPT_LLAVA_ROOT="/home/M2026107/VFlowOpt_llava1.5/src/LLaVA-OneVision"
export VFLOWOPT_TRANSFORMERS_ROOT="/home/M2026107/VFlowOpt_llava1.5/src/transformers-4.46.0/src"

COMMON_ARGS=(
  --teacher-root "${ROOT}/artifacts/future_decode_llava15_7b"
  --datasets gqa textvqa scienceqa
  --llava-path "${ROOT}/ckpts/llava-v1.5-7b"
  --model-name llava-v1.5-7b
  --vision-tower-path "${ROOT}/ckpts/clip-vit-large-patch14-336"
  --device cuda:0
  --device-map cuda:0
  --epochs 15
  --lr 1e-4
  --n-per-dataset 600
  --seed 42
  --log-every 25
)

run_variant() {
  local variant="$1"
  local out_dir="${ROOT}/artifacts/student_llava15_original_future_1800_lr1e4_15ep_${variant}"
  mkdir -p "${out_dir}"
  echo "[run] variant=${variant} out=${out_dir}"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --student-variant "${variant}" \
    --output-dir "${out_dir}"
}

run_variant cnn_only
run_variant mlp_only
