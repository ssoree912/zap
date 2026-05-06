#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/srv/home/dlpc.3842/zap}"
VFLOW_ROOT="${VFLOW_ROOT:-/mnt/srv/home/dlpc.3842/VFlowOpt_llava1.5}"
CONDA_BIN="${CONDA_BIN:-/usr/gatoai/bin/conda}"
CONDA_ENV="${CONDA_ENV:-kv}"
GPU_ID="${GPU_ID:-0}"
KEEP_RATIO="${KEEP_RATIO:?KEEP_RATIO is required}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/ckpts/llava-onevision-qwen2-7b-ov}"
STUDENT_DIR="${STUDENT_DIR:-${REPO_ROOT}/artifacts/student_onevision_original_future_1800_lr1e4_15ep}"
EXP_DIR="${REPO_ROOT}/experiments/EXP-20260503-026-onevision-original-teacher-train"
OUT_ROOT="${OUT_ROOT:-${EXP_DIR}/outputs/data_eval_original_onevision_student_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"

TASKS=(
  docvqa_val
  chartqa
  coco2017_cap_val
  gqa
  nocaps_val
  textcaps_val
  textvqa_val
)

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export ZAP_REPO_ROOT="${REPO_ROOT}"
export LMMS_EVAL_ROOT="${VFLOW_ROOT}/src/lmms_eval-0.2.4"
export VFLOWOPT_LLAVA_ROOT="${VFLOW_ROOT}/src/LLaVA-OneVision"
export VFLOWOPT_TRANSFORMERS_ROOT="${VFLOW_ROOT}/src/transformers-4.46.0/src"
export LMMS_LOCAL_EVAL_ROOT="${REPO_ROOT}/data/eval"
export PYTHONPATH="${REPO_ROOT}:${LMMS_EVAL_ROOT}:${VFLOWOPT_TRANSFORMERS_ROOT}:${VFLOWOPT_LLAVA_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${REPO_ROOT}/ckpts/.hf_home"
export HF_DATASETS_CACHE="${REPO_ROOT}/data/.hf_datasets_cache"
export MPLCONFIGDIR="/tmp/mpl-${USER:-zap}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

KEEP_TAG="${KEEP_RATIO/./}"
KEEP_OUT="${OUT_ROOT}/keep_${KEEP_TAG}"
STATS_DIR="${KEEP_OUT}/keep_stats"
mkdir -p "${KEEP_OUT}" "${STATS_DIR}" "${MPLCONFIGDIR}"

LOG_FILE="${LOG_DIR}/data_eval_onevision_keep${KEEP_TAG}_gpu${GPU_ID}_${RUN_ID}.log"

{
  echo "=== OneVision original student data/eval START $(date -Is) ==="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "KEEP_RATIO=${KEEP_RATIO}"
  echo "MODEL_DIR=${MODEL_DIR}"
  echo "STUDENT_DIR=${STUDENT_DIR}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "LOG_FILE=${LOG_FILE}"

  for task in "${TASKS[@]}"; do
    task_log="${KEEP_OUT}/${task}.log"
    suffix="onevision_orig_student_keep${KEEP_TAG}"
    echo "---- task=${task} log=${task_log}"
    if "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${REPO_ROOT}/eval.py" \
        --model onevision \
        --framework lmms \
        -- \
        --model llava_onevision_original_student \
        --model_args "pretrained=${MODEL_DIR},student_path=${STUDENT_DIR},keep_ratio=${KEEP_RATIO},conv_template=qwen_1_5,model_name=llava_qwen,device=cuda:0,device_map=cuda:0,stats_output_dir=${STATS_DIR}" \
        --tasks "${task}" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "${suffix}" \
        --output_path "${KEEP_OUT}" \
        2>&1 | tee "${task_log}"; then
      echo "[ok] keep_ratio=${KEEP_RATIO} task=${task}"
    else
      echo "[fail] keep_ratio=${KEEP_RATIO} task=${task} -- see ${task_log}" | tee -a "${OUT_ROOT}/FAILURES.log"
    fi
  done

  echo "=== OneVision original student data/eval DONE $(date -Is) ==="
} 2>&1 | tee "${LOG_FILE}"
