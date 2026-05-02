#!/usr/bin/env bash
# GPU 1: docvqa full cache + all datasets keep_ratio=0.1
set -uo pipefail
cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

PYTHON="/opt/conda/envs/kv/bin/python3"
STUDENT="/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4"
MODEL="/workspace/zap/ckpts/llava-1.5-7b-hf"
LOG_DIR="/workspace/zap/experiments/EXP-20260501-020-llava15-noinst-lmms-eval/logs"

run_task() {
  local keep="$1"
  local task="$2"
  local out_dir="$3"
  local model_args="$4"
  local logfile="${LOG_DIR}/gpu1_${keep}_${task}.log"

  if find "${out_dir}" -name "*_results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP ${task} keep=${keep} (exists) ====="
    return
  fi
  mkdir -p "${out_dir}"
  echo "===== START keep=${keep} task=${task} $(date -Is) =====" | tee -a "${logfile}"
  if ${PYTHON} /workspace/zap/eval.py --model llava15 --framework lmms -- \
      --model llava15_student \
      --model_args "${model_args}" \
      --tasks "${task}" \
      --batch_size 1 \
      --output_path "${out_dir}" \
      2>&1 | tee -a "${logfile}"; then
    echo "===== DONE ${task} keep=${keep} $(date -Is) =====" | tee -a "${logfile}"
  else
    echo "===== FAILED ${task} keep=${keep} $(date -Is) =====" | tee -a "${logfile}"
  fi
}

FULL_BASE="/workspace/zap/experiments/EXP-20260501-020-llava15-noinst-lmms-eval/outputs/full"
FULL_ARGS="pretrained=${MODEL},student_path=${STUDENT},keep_ratio=1.0,device=cuda:0"

K01_BASE="/workspace/zap/experiments/EXP-20260501-020-llava15-noinst-lmms-eval/outputs/keep010"
K01_ARGS="pretrained=${MODEL},student_path=${STUDENT},keep_ratio=0.1,device=cuda:0,stats_output_dir=${K01_BASE}"

mkdir -p "${LOG_DIR}"

echo "===== [1/7] docvqa full cache ====="
run_task "full" "docvqa_val_local" "${FULL_BASE}/docvqa_val_local" "${FULL_ARGS}"

echo "===== [2/7] keep=0.1 textvqa ====="
run_task "0.1" "textvqa_val"       "${K01_BASE}/textvqa_val"       "${K01_ARGS}"

echo "===== [3/7] keep=0.1 gqa ====="
run_task "0.1" "gqa_local"         "${K01_BASE}/gqa_local"         "${K01_ARGS}"

echo "===== [4/7] keep=0.1 docvqa ====="
run_task "0.1" "docvqa_val_local"  "${K01_BASE}/docvqa_val_local"  "${K01_ARGS}"

echo "===== [5/7] keep=0.1 mme ====="
run_task "0.1" "mme_local"         "${K01_BASE}/mme_local"         "${K01_ARGS}"

echo "===== [6/7] keep=0.1 chartqa ====="
run_task "0.1" "chartqa_local"     "${K01_BASE}/chartqa_local"     "${K01_ARGS}"

echo "===== [7/7] keep=0.1 scienceqa ====="
run_task "0.1" "scienceqa_img_local" "${K01_BASE}/scienceqa_img_local" "${K01_ARGS}"

echo "===== ALL DONE gpu1 $(date -Is) ====="
