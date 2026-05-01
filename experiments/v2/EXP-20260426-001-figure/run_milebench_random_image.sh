#!/bin/bash
# MileBench S-1 ~ S-5 (13 tasks) random_image_only @ keep=0.2.
# Default GPU=1.
set -uo pipefail

GPU=${GPU:-1}
KEEP=0.2
DATA_ROOT=/workspace/zap/data/MileBench
OUT_ROOT=/workspace/zap/artifacts/EXP-20260426-001-figure/milebench_random_image
mkdir -p "$OUT_ROOT"

# S-1: Knowledge Grounded QA
# S-2: Text-Rich Images QA
# S-3: Visual Relation Inference (ROUGE-L)
# S-4: Dialogue
# S-5: Space Understanding
TASKS=(
  WebQA MultiModalQA TQA WikiVQA
  DocVQA OCR-VQA SlideVQA
  Spot-the-Diff CLEVR-Change IEdit
  MMCoQA ALFRED
  nuscenes
)

run_task() {
  local task=$1
  local data="$DATA_ROOT/$task/$task.json"
  local outdir="$OUT_ROOT/$task"
  mkdir -p "$outdir"

  if [ ! -f "$data" ]; then
    echo "[skip] $task: no $data"
    return
  fi
  if [ ! -d "$DATA_ROOT/$task/images" ]; then
    echo "[skip] $task: no $DATA_ROOT/$task/images"
    return
  fi
  if [ -f "$outdir/eval.json" ] || [ -f "$outdir/result.json" ]; then
    echo "[skip] $task: already done"
    return
  fi

  echo "[gpu $GPU] $task starting (random_image_only, keep=$KEEP)"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/evaluate_image_teacher_pruning.py \
    --mode random_image_only \
    --total_keep_ratio "$KEEP" \
    --dataset_path "$data" \
    --image_root "$DATA_ROOT/$task/images" \
    --look_dataset_name "$task" \
    --look_model_name "zap_${task}_random_image_total${KEEP}" \
    --output_dir "$outdir" \
    --attn_implementation sdpa \
    --max_new_tokens 32 \
    --truncate \
    >"$outdir/eval.log" 2>&1
  local rc=$?
  echo "[gpu $GPU] $task done (rc=$rc)"
}

for task in "${TASKS[@]}"; do
  run_task "$task"
done

echo "[done] random_image_only @ keep=$KEEP on MileBench S-1..S-5"
