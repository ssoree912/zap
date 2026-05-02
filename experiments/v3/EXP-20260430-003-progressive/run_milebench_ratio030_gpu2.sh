#!/usr/bin/env bash
set -uo pipefail
cd /workspace/zap

export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

KEEP_RATIO="0.30"
OUT_DIR="/workspace/zap/experiments/EXP-20260430-003-progressive/outputs/keep030"
STUDENT="/workspace/zap/ckpts/student_onevision_A_ep20"
PRETRAINED="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/milebench_run_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

TASKS=(
  ActionLocalization ActionPrediction ActionSequence ALFRED CLEVR-Change
  CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200
  IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection
  MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle
  SceneTransition SlideVQA Spot-the-Diff StateChange TQA
  TextNeedleInAHaystack WebQA WikiVQA
)

for task in "${TASKS[@]}"; do
  pred_file="${OUT_DIR}/${task}/pred.json"
  if [ -f "$pred_file" ]; then
    echo "===== SKIP ${task} (pred.json exists) ====="
    continue
  fi
  echo "===== START keep_ratio=${KEEP_RATIO} task=${task} $(date -Is) ====="
  /opt/conda/envs/vflowopt_chartqa_eval/bin/python /workspace/zap/foresight/eval/milebench_onevision_student.py \
    --dataset "$task" \
    --pretrained "$PRETRAINED" \
    --student_path "$STUDENT" \
    --keep_ratio "$KEEP_RATIO" \
    --output_dir "$OUT_DIR" \
    --device cuda:0
  echo "===== DONE ${task} $(date -Is) ====="
done

# Aggregate scores using look-m scorer
echo "===== SCORING ====="
/opt/conda/envs/vflowopt_chartqa_eval/bin/python /workspace/look-m/evaluate.py \
  --data_dir /workspace/zap/data/MileBench \
  --pred_dir "$OUT_DIR" \
  --output_dir "$OUT_DIR" 2>/dev/null || true

echo "===== ALL DONE ratio=${KEEP_RATIO} $(date -Is) ====="
