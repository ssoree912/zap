#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/workspace/VFlowOpt/.conda/VFlowOpt"
MODEL_DIR="/workspace/zap/ckpts/llava-v1.5-7b"
STUDENT_DIR="/workspace/zap/artifacts/original_llava_teacher/student_llava15_original_future_1800_lr1e4_15ep"
WRAPPER="/workspace/zap/experiments/EXP-20260504-002-llava15-joint-eviction/lmms_eval_joint_local_run.py"
OUT_ROOT="/workspace/zap/experiments/EXP-20260504-002-llava15-joint-eviction/outputs/lmms_joint_eviction_$(date +%Y%m%d_%H%M%S)"

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$OUT_ROOT"

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

TASKS=(
  textvqa_val
  gqa
  docvqa_val
  chartqa
  mme
  scienceqa
  coco2017_cap_val
  nocaps_val
  textcaps_val
)

KEEP_RATIOS=(0.5 0.2)

echo "Output root: $OUT_ROOT"
echo "Model:       $MODEL_DIR"
echo "Student:     $STUDENT_DIR"
echo "Variant:     llava15_original_student_joint (text+image, rank-norm, eager attn)"

for keep in "${KEEP_RATIOS[@]}"; do
  keep_tag="${keep/./}"
  keep_out="$OUT_ROOT/keep_${keep_tag}"
  mkdir -p "$keep_out/keep_stats"
  suffix="llava15_orig_joint_keep${keep_tag}"
  echo "========== keep_ratio=$keep =========="

  for task in "${TASKS[@]}"; do
    task_log="$keep_out/${task}.log"
    echo "---- task=$task log=$task_log"
    if python "$WRAPPER" \
        --model llava15_original_student_joint \
        --model_args "pretrained=${MODEL_DIR},student_path=${STUDENT_DIR},keep_ratio=${keep},conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,attn_implementation=eager,score_normalize=rank,stats_output_dir=${keep_out}/keep_stats" \
        --tasks "$task" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "$suffix" \
        --output_path "$keep_out" \
        2>&1 | tee "$task_log"; then
      echo "[ok] keep_ratio=$keep task=$task"
    else
      echo "[fail] keep_ratio=$keep task=$task -- see $task_log" | tee -a "$OUT_ROOT/FAILURES.log"
    fi
  done
done

echo "Done. Results under $OUT_ROOT"
