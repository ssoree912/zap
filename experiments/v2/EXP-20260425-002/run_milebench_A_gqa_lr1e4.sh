#!/bin/bash
# MileBench eval with new lr=1e-4 ckpt at total_keep ∈ {0.1, 0.2}.
# Multi-image + truncate. Sequential on GPU 1.
set -uo pipefail

GPU=${GPU:-1}
DATA_ROOT=/workspace/zap/data/MileBench
CKPT=/workspace/zap/ckpts/student_v2_A_gqa_lr1e4
OUT_BASE=/workspace/zap/artifacts/EXP-20260425-002

TASKS=(
  ActionLocalization ActionPrediction ActionSequence ALFRED CLEVR-Change
  CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200
  IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection
  MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle
  SceneTransition SlideVQA Spot-the-Diff StateChange TQA
  TextNeedleInAHaystack WebQA WikiVQA
)

run_task() {
  local task=$1 keep=$2
  local data="$DATA_ROOT/$task/$task.json"
  local img="$DATA_ROOT/$task/images"
  local outdir="$OUT_BASE/milebench_A_gqa_lr1e4_total${keep}/$task"
  mkdir -p "$outdir"
  if [ ! -f "$data" ] || [ ! -d "$img" ]; then
    echo "[skip] $task: missing data or images"
    return
  fi
  if [ -f "$outdir/metrics.json" ]; then
    echo "[skip] $task k=$keep: already done"
    return
  fi
  echo "[gpu $GPU] $task k=$keep starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/evaluate_image_teacher_pruning.py \
    --mode visual_utility_student \
    --student_model_name "$CKPT" \
    --total_keep_ratio "$keep" \
    --dataset_path "$data" \
    --image_root "$img" \
    --look_dataset_name "$task" \
    --look_model_name "zap_${task}_A_gqa_lr1e4_total${keep}" \
    --output_dir "$outdir" \
    --attn_implementation sdpa \
    --max_new_tokens 32 \
    --truncate \
    >"$outdir/eval.log" 2>&1
  echo "[gpu $GPU] $task k=$keep done (rc=$?)"
}

for keep in 0.2 0.1; do
  for task in "${TASKS[@]}"; do
    run_task "$task" "$keep"
  done
done

echo "[done] all MileBench tasks at keep ∈ {0.2, 0.1} finished"
