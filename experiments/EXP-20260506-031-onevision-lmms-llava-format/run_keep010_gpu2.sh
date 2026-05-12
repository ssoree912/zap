#!/usr/bin/env bash
# EXP-20260506-031 : OneVision student lmms-eval, llava-format ckpt, image_keep_ratio=0.10
set -uo pipefail
cd /workspace/zap

export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/opt/conda/envs/vflowopt_chartqa_eval/lib:${LD_LIBRARY_PATH:-}
export LMMS_EVAL_ROOT=/workspace/VFlowOpt/src/lmms_eval-0.2.4
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

PYTHON="/opt/conda/envs/vflowopt_chartqa_eval/bin/python"
MODEL="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov"
STUDENT="/workspace/zap/ckpts/student_onevision_A_ep20"
KEEP=0.1
OUT_DIR="/workspace/zap/experiments/EXP-20260506-031-onevision-lmms-llava-format/outputs/keep010"

MODEL_ARGS="pretrained=${MODEL},student_path=${STUDENT},keep_ratio=${KEEP},device=cuda:0,stats_output_dir=${OUT_DIR},model_format=llava,conv_template=qwen_1_5"

mkdir -p "${OUT_DIR}"

run_task() {
  local task="$1"
  local task_out="${OUT_DIR}/${task}"

  if find "${task_out}" -name "*_results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP ${task} (already done) ====="
    return
  fi
  mkdir -p "${task_out}"
  echo "===== START task=${task} keep=${KEEP} $(date -Is) =====" | tee -a "${OUT_DIR}/run_gpu2.log"

  if ${PYTHON} /workspace/zap/eval.py --model onevision --framework lmms -- \
      --model llava_onevision_student \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --batch_size 1 \
      --log_samples \
      --output_path "${task_out}" \
      2>&1 | tee -a "${OUT_DIR}/run_gpu2.log"; then
    echo "===== DONE ${task} $(date -Is) =====" | tee -a "${OUT_DIR}/run_gpu2.log"
  else
    echo "===== FAILED ${task} $(date -Is) =====" | tee -a "${OUT_DIR}/run_gpu2.log"
  fi
}

run_task "textvqa_val"
run_task "chartqa_local"
run_task "docvqa_val_local"
run_task "nocaps_val"
run_task "textcaps_val"

echo "===== ALL DONE keep=${KEEP} $(date -Is) ====="
