#!/usr/bin/env bash
set -uo pipefail
cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval_hf/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

KEEP_RATIO="0.30"
OUT_DIR="/workspace/zap/experiments/EXP-20260430-003-progressive/outputs/keep030"
MODEL_ARGS="pretrained=/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf,student_path=/workspace/zap/ckpts/student_onevision_A_ep20,keep_ratio=${KEEP_RATIO},device=cuda:0,stats_output_dir=${OUT_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAIL_LOG="${OUT_DIR}/failed_tasks_${RUN_ID}.csv"
mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/run_${RUN_ID}.log"
printf "task,exit_code,ended_at\n" > "$FAIL_LOG"
exec > >(tee -a "$LOG_FILE") 2>&1

TASKS=(mme_local mmbench_en_dev_local scienceqa_img_local vizwiz_vqa_val_local gqa_local pope_local textvqa_val chartqa_local docvqa_val_local mmstar_local)

for task in "${TASKS[@]}"; do
  task_out="${OUT_DIR}/${task}"
  if find "$task_out" -name "*_results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP ${task} (exists) ====="
    continue
  fi
  mkdir -p "$task_out"
  echo "===== START keep_ratio=${KEEP_RATIO} task=${task} $(date -Is) ====="
  if /opt/conda/envs/vflowopt_chartqa_eval/bin/python /workspace/zap/eval.py \
    --model onevision --framework lmms -- \
    --model llava_onevision_student \
    --model_args "$MODEL_ARGS" \
    --tasks "$task" \
    --batch_size 1 \
    --output_path "$task_out"; then
    status=0
  else
    status=$?
  fi
  new_result="$(find "$task_out" -path '*/ckpts__*/*_results.json' -print -quit 2>/dev/null)"
  if [ "$status" -eq 0 ] && [ -n "$new_result" ]; then
    echo "===== DONE task=${task} $(date -Is) ====="
  else
    [ "$status" -eq 0 ] && status="0_no_result_json"
    echo "===== FAILED task=${task} exit_code=${status} $(date -Is) ====="
    printf "%s,%s,%s\n" "$task" "$status" "$(date -Is)" >> "$FAIL_LOG"
  fi
done
echo "===== ALL DONE gpu2 ratio=${KEEP_RATIO} $(date -Is) ====="
