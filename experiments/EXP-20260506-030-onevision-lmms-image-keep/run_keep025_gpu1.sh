#!/usr/bin/env bash
# EXP-20260506-030 : OneVision student lmms-eval, image_keep_ratio=0.50
# keep_ratio_basis=image (keep 50% of image tokens)
set -uo pipefail
cd /workspace/zap

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true
export LD_LIBRARY_PATH=/opt/conda/envs/vflowopt_chartqa_eval/lib:${LD_LIBRARY_PATH:-}
export LMMS_EVAL_ROOT=/workspace/VFlowOpt/src/lmms_eval-0.2.4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

PYTHON="/opt/conda/envs/vflowopt_chartqa_eval/bin/python"
MODEL="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf"
STUDENT="/workspace/zap/ckpts/student_onevision_A_ep20"
KEEP=0.25
OUT_DIR="/workspace/zap/experiments/EXP-20260506-030-onevision-lmms-image-keep/outputs/keep025"

MODEL_ARGS="pretrained=${MODEL},student_path=${STUDENT},keep_ratio=${KEEP},device=cuda:1,stats_output_dir=${OUT_DIR}"

mkdir -p "${OUT_DIR}"

run_task() {
  local task="$1"
  local task_out="${OUT_DIR}/${task}"

  if find "${task_out}" -name "*_results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP ${task} (already done) ====="
    return
  fi
  mkdir -p "${task_out}"
  echo "===== START task=${task} keep=${KEEP} $(date -Is) =====" | tee -a "${OUT_DIR}/run_gpu1.log"

  if ${PYTHON} /workspace/zap/eval.py --model onevision --framework lmms -- \
      --model llava_onevision_student \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --batch_size 1 \
      --log_samples \
      --output_path "${task_out}" \
      2>&1 | tee -a "${OUT_DIR}/run_gpu1.log"; then
    echo "===== DONE ${task} $(date -Is) =====" | tee -a "${OUT_DIR}/run_gpu1.log"
  else
    echo "===== FAILED ${task} $(date -Is) =====" | tee -a "${OUT_DIR}/run_gpu1.log"
  fi
}

run_task "textvqa_val"
run_task "chartqa_local"
run_task "docvqa_val_local"
run_task "nocaps_val"
run_task "textcaps_val"

echo "===== ALL DONE keep=${KEEP} $(date -Is) ====="
