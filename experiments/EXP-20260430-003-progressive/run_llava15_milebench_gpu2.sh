#!/usr/bin/env bash
set -uo pipefail
cd /workspace/zap

export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/opt/conda/envs/vflowopt_chartqa_eval/bin/python
LLAVA15_CKPT="/workspace/zap/ckpts/llava-1.5-7b-hf"
TEACHER_ROOT="/workspace/zap/data/train/teacher_llava15"
STUDENT_CKPT="/workspace/zap/ckpts/student_llava15_mmvet_instruct"
DATA_DIR="/workspace/zap/data/MileBench"
EXP_DIR="/workspace/zap/experiments/EXP-20260430-003-progressive/outputs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$EXP_DIR"
LOG_FILE="${EXP_DIR}/llava15_run_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

TASKS=(
  ActionLocalization ActionPrediction ActionSequence ALFRED CLEVR-Change
  CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200
  IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection
  MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle
  SceneTransition SlideVQA Spot-the-Diff StateChange TQA
  TextNeedleInAHaystack WebQA WikiVQA
)

# ── Step 1: Collect mmvet teacher ────────────────────────────────────────────
echo "===== STEP 1: Collect mmvet teacher $(date -Is) ====="
if [ -d "${TEACHER_ROOT}/mmvet" ] && [ "$(ls -A ${TEACHER_ROOT}/mmvet/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[skip] teacher already collected at ${TEACHER_ROOT}/mmvet"
else
  $PYTHON -m foresight.teacher.collect_llava15 \
    --dataset mmvet \
    --model "$LLAVA15_CKPT" \
    --output-root "$TEACHER_ROOT" \
    --device cuda:0
fi
echo "===== STEP 1 DONE $(date -Is) ====="

# ── Step 2: Train LLaVA-1.5 student ─────────────────────────────────────────
echo "===== STEP 2: Train student $(date -Is) ====="
if [ -f "${STUDENT_CKPT}/config.json" ]; then
  echo "[skip] student already trained at ${STUDENT_CKPT}"
else
  $PYTHON -m foresight.train.llava_15 \
    --teacher-root "$TEACHER_ROOT" \
    --datasets mmvet llava_instruct \
    --llava-path "$LLAVA15_CKPT" \
    --output-dir "$STUDENT_CKPT" \
    --epochs 20 \
    --device cuda:0
fi
echo "===== STEP 2 DONE $(date -Is) ====="

# ── Step 3: MileBench eval at each keep_ratio ────────────────────────────────
for KEEP_RATIO in 0.50 0.20 0.10; do
  RATIO_TAG="$(echo $KEEP_RATIO | tr -d '.' | sed 's/^0*//')"
  # zero-pad to 3 digits: 050, 020, 010
  RATIO_TAG="$(printf '%03d' $((10#${RATIO_TAG})))"
  OUT_DIR="${EXP_DIR}/llava15_milebench_keep${RATIO_TAG}"
  mkdir -p "$OUT_DIR"

  echo "===== STEP 3 keep_ratio=${KEEP_RATIO} START $(date -Is) ====="

  for task in "${TASKS[@]}"; do
    pred_file="${OUT_DIR}/${task}/pred.json"
    if [ -f "$pred_file" ]; then
      echo "===== SKIP ${task} (pred.json exists) ====="
    else
      echo "===== START keep_ratio=${KEEP_RATIO} task=${task} $(date -Is) ====="
      $PYTHON /workspace/zap/foresight/eval/milebench_llava15_student.py \
        --dataset "$task" \
        --pretrained "$LLAVA15_CKPT" \
        --student_path "$STUDENT_CKPT" \
        --keep_ratio "$KEEP_RATIO" \
        --output_dir "$OUT_DIR" \
        --device cuda:0
      echo "===== DONE ${task} $(date -Is) ====="
    fi

    eval_file="${OUT_DIR}/${task}/eval.json"
    if [ ! -f "$eval_file" ] && [ -f "$pred_file" ]; then
      $PYTHON /workspace/look-m/evaluate.py \
        --data-dir "$DATA_DIR" \
        --dataset "$task" \
        --result-dir "$OUT_DIR"
    fi
  done

  echo "===== STEP 3 keep_ratio=${KEEP_RATIO} DONE $(date -Is) ====="
done

echo "===== ALL DONE $(date -Is) ====="
