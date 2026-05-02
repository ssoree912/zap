#!/usr/bin/env bash
# EXP-20260501-020 : student_llava15_no_instruct_1800_lr1e4 lmms-eval
# datasets : textvqa(5k), gqa(5k), docvqa_val, MME, chartqa, scienceqa
# keep_ratio: 0.3 / GPU 2
set -uo pipefail
cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

PYTHON="/opt/conda/envs/kv/bin/python3"
KEEP="0.3"
STUDENT="/workspace/zap/ckpts/student_llava15_no_instruct_1800_lr1e4"
MODEL="/workspace/zap/ckpts/llava-1.5-7b-hf"
OUT_DIR="/workspace/zap/experiments/EXP-20260501-020-llava15-noinst-lmms-eval/outputs/keep030"
LOG_DIR="/workspace/zap/experiments/EXP-20260501-020-llava15-noinst-lmms-eval/logs"

MODEL_ARGS="pretrained=${MODEL},student_path=${STUDENT},keep_ratio=${KEEP},device=cuda:0,stats_output_dir=${OUT_DIR}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

run_task() {
  local task="$1"
  local limit_args="${2:-}"
  local task_out="${OUT_DIR}/${task}"

  if find "${task_out}" -name "*_results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP ${task} (exists) ====="
    return
  fi
  mkdir -p "${task_out}"
  echo "===== START keep=${KEEP} task=${task} $(date -Is) =====" | tee -a "${LOG_DIR}/${task}.log"

  if ${PYTHON} /workspace/zap/eval.py --model llava15 --framework lmms -- \
      --model llava15_student \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --batch_size 1 \
      ${limit_args} \
      --output_path "${task_out}" \
      2>&1 | tee -a "${LOG_DIR}/${task}.log"; then
    echo "===== DONE ${task} $(date -Is) =====" | tee -a "${LOG_DIR}/${task}.log"
  else
    echo "===== FAILED ${task} $(date -Is) =====" | tee -a "${LOG_DIR}/${task}.log"
  fi
}

run_task "textvqa_val"          ""
run_task "gqa_local"            ""
run_task "docvqa_val_local"     ""
run_task "mme_local"            ""
run_task "chartqa_local"        ""
run_task "scienceqa_img_local"  ""

echo "===== ALL DONE gpu2 keep=${KEEP} $(date -Is) ====="
