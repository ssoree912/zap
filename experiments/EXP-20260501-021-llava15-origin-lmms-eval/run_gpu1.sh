#!/usr/bin/env bash
# EXP-20260501-021 : LLaVA-1.5-7B origin (no compression) lmms-eval
# datasets : textvqa_val, gqa_local, docvqa_val_local, mme_local, chartqa_local, scienceqa_img_local
# keep_ratio: 1.0 (no eviction) / GPU 1
set -uo pipefail
cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

PYTHON="/opt/conda/envs/kv/bin/python3"
MODEL="/workspace/zap/ckpts/llava-1.5-7b-hf"
STUDENT="/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4"
OUT_DIR="/workspace/zap/experiments/EXP-20260501-021-llava15-origin-lmms-eval/outputs"
LOG_DIR="/workspace/zap/experiments/EXP-20260501-021-llava15-origin-lmms-eval/logs"

MODEL_ARGS="pretrained=${MODEL},student_path=${STUDENT},keep_ratio=1.0,device=cuda:0"
# Note: CUDA_VISIBLE_DEVICES=1 remaps physical GPU1 → logical cuda:0

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

run_task() {
  local task="$1"
  local task_out="${OUT_DIR}/${task}"

  if find "${task_out}" -name "*_results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP ${task} (already done) ====="
    return
  fi
  mkdir -p "${task_out}"
  echo "===== START task=${task} $(date -Is) =====" | tee -a "${LOG_DIR}/${task}.log"

  if ${PYTHON} /workspace/zap/eval.py --model llava15 --framework lmms -- \
      --model llava15_student \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --batch_size 1 \
      --output_path "${task_out}" \
      2>&1 | tee -a "${LOG_DIR}/${task}.log"; then
    echo "===== DONE ${task} $(date -Is) =====" | tee -a "${LOG_DIR}/${task}.log"
  else
    echo "===== FAILED ${task} $(date -Is) =====" | tee -a "${LOG_DIR}/${task}.log"
  fi
}

run_task "textvqa_val"
run_task "gqa_local"
run_task "docvqa_val_local"
run_task "mme_local"
run_task "chartqa_local"
run_task "scienceqa_img_local"

echo "===== ALL DONE $(date -Is) ====="
