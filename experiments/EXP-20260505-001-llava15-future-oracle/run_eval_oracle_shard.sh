#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Worker shard: run a list of (task, keep_ratio) jobs on a single GPU.
#
# Usage:
#   GPU_ID=2 OUT_ROOT=/path/oracle_<stamp> \
#     run_eval_oracle_shard.sh chartqa:0.5 chartqa:0.2 textvqa_val:0.5 ...
#
# Each job is encoded as "<task>:<keep_ratio>".

set -euo pipefail

if [[ -z "${GPU_ID:-}" ]]; then
  echo "GPU_ID not set" >&2
  exit 2
fi
if [[ -z "${OUT_ROOT:-}" ]]; then
  echo "OUT_ROOT not set" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "no jobs given" >&2
  exit 2
fi

ZAP_ROOT="/mnt/srv/home/dlpc.3842/zap"
ENV_DIR="/home/M2026107/.conda/envs/kv-vflowopt"
MODEL_DIR="${ZAP_ROOT}/ckpts/llava-v1.5-7b"
WRAPPER="${ZAP_ROOT}/experiments/EXP-20260505-001-llava15-future-oracle/lmms_eval_oracle_local_run.py"

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${ZAP_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${ZAP_ROOT}/.cache/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export ZAP_REPO_ROOT="${ZAP_ROOT}"
export VFLOWOPT_ROOT="/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5/src"
export VFLOWOPT_LLAVA_ROOT="${VFLOWOPT_ROOT}/LLaVA-OneVision"
export VFLOWOPT_TRANSFORMERS_ROOT="${VFLOWOPT_ROOT}/transformers-4.46.0/src"
export LMMS_EVAL_ROOT="${VFLOWOPT_ROOT}/lmms_eval-0.2.4"
export LMMS_LOCAL_EVAL_ROOT="${ZAP_ROOT}/data/eval"

source /usr/gatoai/mamba/26.1.0-0/etc/profile.d/conda.sh
conda activate "${ENV_DIR}"

PYTHON="${ENV_DIR}/bin/python"

mkdir -p "${OUT_ROOT}"

echo "[shard gpu=${GPU_ID}] OUT_ROOT=${OUT_ROOT}"
echo "[shard gpu=${GPU_ID}] jobs: $*"

for job in "$@"; do
  task="${job%%:*}"
  keep="${job##*:}"
  keep_tag="${keep/./}"
  keep_out="${OUT_ROOT}/keep_${keep_tag}"
  mkdir -p "${keep_out}/keep_stats"
  suffix="llava15_orig_oracle_keep${keep_tag}"
  task_log="${keep_out}/${task}.gpu${GPU_ID}.log"
  echo "[shard gpu=${GPU_ID}] ---- task=${task} keep=${keep} log=${task_log}"
  if "${PYTHON}" "${WRAPPER}" \
      --model llava15_original_oracle \
      --model_args "pretrained=${MODEL_DIR},keep_ratio=${keep},conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,attn_implementation=eager,stats_output_dir=${keep_out}/keep_stats" \
      --tasks "${task}" \
      --batch_size 1 \
      --log_samples \
      --log_samples_suffix "${suffix}" \
      --output_path "${keep_out}" \
      2>&1 | tee "${task_log}"; then
    echo "[shard gpu=${GPU_ID}] [ok] task=${task} keep=${keep}"
  else
    echo "[shard gpu=${GPU_ID}] [fail] task=${task} keep=${keep} -- see ${task_log}" \
      | tee -a "${OUT_ROOT}/FAILURES.gpu${GPU_ID}.log"
  fi
done

echo "[shard gpu=${GPU_ID}] done"
