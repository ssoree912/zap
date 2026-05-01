#!/bin/bash
# MileBench: student_v2_traj, total_keep_ratio=0.2
set -uo pipefail

GPU=0
DATA_ROOT=/workspace/zap/data/MileBench
CKPT=/workspace/zap/ckpts/student_v2_traj
OUT_ROOT=/workspace/zap/artifacts/EXP-20260425-002/milebench_traj_total02
KEEP=0.2
mkdir -p "$OUT_ROOT"

TASKS=(
  ActionLocalization ActionPrediction ActionSequence ALFRED CLEVR-Change
  CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200
  IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection
  MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle
  SceneTransition SlideVQA Spot-the-Diff StateChange TQA
  TextNeedleInAHaystack WebQA WikiVQA
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
    echo "[skip] $task: no images"
    return
  fi
  if [ -f "$outdir/metrics.json" ] || [ -f "$outdir/result.json" ]; then
    echo "[skip] $task: already done"
    return
  fi

  echo "[gpu $GPU] $task starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/evaluate_image_teacher_pruning.py \
    --mode visual_utility_student \
    --student_model_name "$CKPT" \
    --total_keep_ratio "$KEEP" \
    --dataset_path "$data" \
    --image_root "$DATA_ROOT/$task/images" \
    --look_dataset_name "$task" \
    --look_model_name "zap_${task}_traj_total${KEEP}" \
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

echo "[done] all MileBench tasks finished"
