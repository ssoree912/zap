#!/usr/bin/env bash
set -euo pipefail

ZAP_ROOT="${ZAP_ROOT:-/home/M2026107/zap}"
PYTHON_BIN="${PYTHON_BIN:-/home/M2026107/.conda/envs/kv/bin/python}"
MODEL_DIR="${MODEL_DIR:-${ZAP_ROOT}/ckpts/llava-v1.5-7b}"
VISION_TOWER_DIR="${VISION_TOWER_DIR:-${ZAP_ROOT}/ckpts/clip-vit-large-patch14-336}"
WRAPPER="${WRAPPER:-${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract/lmms_eval_original_llava15_local_run.py}"
TASK="${TASK:-docvqa_val}"
OUT_ROOT="${OUT_ROOT:-${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract/outputs/${TASK}_original_variants_$(date +%Y%m%d_%H%M%S)}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export HF_HOME="${HF_HOME:-${ZAP_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ZAP_ROOT}/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ZAP_REPO_ROOT="${ZAP_REPO_ROOT:-${ZAP_ROOT}}"
export VFLOWOPT_ROOT="${VFLOWOPT_ROOT:-/home/M2026107/VFlowOpt_llava1.5/src}"
export VFLOWOPT_LLAVA_ROOT="${VFLOWOPT_LLAVA_ROOT:-${VFLOWOPT_ROOT}/LLaVA-OneVision}"
export VFLOWOPT_TRANSFORMERS_ROOT="${VFLOWOPT_TRANSFORMERS_ROOT:-${VFLOWOPT_ROOT}/transformers-4.46.0/src}"
export LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${VFLOWOPT_ROOT}/lmms_eval-0.2.4}"
export LMMS_ALT_LOCAL_EVAL_ROOT="${LMMS_ALT_LOCAL_EVAL_ROOT:-/home/M2026107/VFlowOpt_llava1.5/zap/data/eval}"

declare -A STUDENTS=(
  [cnn_mlp]="${ZAP_ROOT}/artifacts/student_llava15_original_future_1800_lr1e4_15ep"
  [cnn_only]="${ZAP_ROOT}/artifacts/student_llava15_original_future_1800_lr1e4_15ep_cnn_only"
  [mlp_only]="${ZAP_ROOT}/artifacts/student_llava15_original_future_1800_lr1e4_15ep_mlp_only"
)

if [[ -n "${VARIANT_LIST:-}" ]]; then
  read -r -a VARIANTS <<< "$VARIANT_LIST"
else
  VARIANTS=(cnn_mlp cnn_only mlp_only)
fi
KEEP_RATIOS=(0.5 0.2)
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

mkdir -p "$OUT_ROOT"

echo "Output root: $OUT_ROOT"
echo "Model: $MODEL_DIR"
echo "Vision tower: $VISION_TOWER_DIR"
echo "Task: $TASK"
echo "Limit: ${LIMIT:-full}"
echo "Keep ratios: ${KEEP_RATIOS[*]}"
echo "Variants: ${VARIANTS[*]}"

for variant in "${VARIANTS[@]}"; do
  student="${STUDENTS[$variant]}"
  echo "========== variant=$variant student=$student =========="

  for keep in "${KEEP_RATIOS[@]}"; do
    keep_tag="${keep/./}"
    run_out="$OUT_ROOT/${variant}/keep_${keep_tag}"
    stats_out="$run_out/keep_stats"
    task_log="$run_out/${TASK}.log"
    suffix="llava15_orig_${variant}_keep${keep_tag}"

    mkdir -p "$stats_out"
    echo "---- variant=$variant keep_ratio=$keep log=$task_log"

    if "$PYTHON_BIN" "$WRAPPER" \
        --model llava15_original_student \
        --model_args "pretrained=${MODEL_DIR},student_path=${student},vision_tower_path=${VISION_TOWER_DIR},keep_ratio=${keep},conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,stats_output_dir=${stats_out}" \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "$suffix" \
        --output_path "$run_out" \
        "${LIMIT_ARGS[@]}" \
        2>&1 | tee "$task_log"; then
      echo "[ok] variant=$variant keep_ratio=$keep task=$TASK"
    else
      echo "[fail] variant=$variant keep_ratio=$keep task=$TASK -- see $task_log" | tee -a "$OUT_ROOT/FAILURES.log"
    fi
  done
done

echo "Done. Results under $OUT_ROOT"
