#!/bin/bash
# Trajectory teacher collection for EXP-20260425-002.
# Collects 200 LLaVA-Instruct-150K samples with M=3 trajectories on GPU 2,
# then re-collects existing datasets (scienceqa/textvqa/docvqa) with M=3
# to produce a uniform trajectory teacher cache for all 600 training samples.
#
# LLaVA-Instruct-150K JSON must be downloaded first:
#   mkdir -p /workspace/zap/data/llava_instruct
#   wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json \
#        -O /workspace/zap/data/llava_instruct/llava_instruct_150k.json
#
# Images are COCO train2017, already at /workspace/data/coco/train2017.
set -uo pipefail

DEVICE=${DEVICE:-cuda:2}
TRAJ_M=${TRAJ_M:-3}
TRAJ_TEMP=${TRAJ_TEMP:-0.7}
TRAJ_TOP_P=${TRAJ_TOP_P:-0.9}
N_LLAVA=${N_LLAVA:-200}
N_OTHERS=${N_OTHERS:-200}
MAX_NEW_TOKENS=64
OUTPUT_ROOT=/workspace/zap/data/teacher_v2_traj
COLLECT=/workspace/zap/collect_future_teacher_v2.py

echo "[trajectory collect] M=$TRAJ_M temp=$TRAJ_TEMP top_p=$TRAJ_TOP_P device=$DEVICE"
echo "[trajectory collect] output_root=$OUTPUT_ROOT"

# ── LLaVA-Instruct-150K (200 samples) ───────────────────────────────────────
echo "[step 1/4] llava_instruct n=$N_LLAVA"
python "$COLLECT" \
  --dataset llava_instruct \
  --n-samples "$N_LLAVA" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --llava-instruct-json /workspace/zap/data/llava_instruct/llava_instruct_150k.json \
  --llava-instruct-images-root /workspace/data/coco/train2017 \
  --trajectory-m "$TRAJ_M" \
  --trajectory-temperature "$TRAJ_TEMP" \
  --trajectory-top-p "$TRAJ_TOP_P" \
  --seed 0

# ── ScienceQA (200 samples) ──────────────────────────────────────────────────
echo "[step 2/4] scienceqa n=$N_OTHERS"
python "$COLLECT" \
  --dataset scienceqa \
  --n-samples "$N_OTHERS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --trajectory-m "$TRAJ_M" \
  --trajectory-temperature "$TRAJ_TEMP" \
  --trajectory-top-p "$TRAJ_TOP_P" \
  --seed 0

# ── TextVQA (200 samples) ────────────────────────────────────────────────────
echo "[step 3/4] textvqa n=$N_OTHERS"
python "$COLLECT" \
  --dataset textvqa \
  --n-samples "$N_OTHERS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --trajectory-m "$TRAJ_M" \
  --trajectory-temperature "$TRAJ_TEMP" \
  --trajectory-top-p "$TRAJ_TOP_P" \
  --seed 0

# ── DocVQA (optional, skip if not enough data) ──────────────────────────────
echo "[step 4/4] docvqa n=$N_OTHERS (may be fewer if dataset is small)"
python "$COLLECT" \
  --dataset docvqa \
  --n-samples "$N_OTHERS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --trajectory-m "$TRAJ_M" \
  --trajectory-temperature "$TRAJ_TEMP" \
  --trajectory-top-p "$TRAJ_TOP_P" \
  --seed 0

echo "[done] trajectory teacher cache written to $OUTPUT_ROOT"
echo "To train with this cache, pass --teacher-root $OUTPUT_ROOT to train_visual_utility_student.py"
