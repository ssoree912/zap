#!/usr/bin/env bash
set -euo pipefail

EXP=/workspace/nips/zap/experiments/EXP-20260727-001-eval700-q2
OUT=${OUTPUT_DIR:-/workspace/nips/zap/artifacts/rebuttal_eval700_q2_total0p2}
PY=/workspace/nips/.conda/envs/qvik/bin/python

mkdir -p "$OUT/logs"

screen -dmS eval700_gpu0 bash -lc \
  "cd /workspace/nips/zap && CUDA_VISIBLE_DEVICES=0 $PY $EXP/run_eval700_q2.py \
  --device cuda:0 --output-dir $OUT --datasets gqa textvqa chartqa \
  > $OUT/logs/gpu0.log 2>&1"

screen -dmS eval700_gpu1 bash -lc \
  "cd /workspace/nips/zap && CUDA_VISIBLE_DEVICES=1 $PY $EXP/run_eval700_q2.py \
  --device cuda:0 --output-dir $OUT --datasets docvqa coco_caption \
  > $OUT/logs/gpu1.log 2>&1"

screen -dmS eval700_gpu2 bash -lc \
  "cd /workspace/nips/zap && CUDA_VISIBLE_DEVICES=2 $PY $EXP/run_eval700_q2.py \
  --device cuda:0 --output-dir $OUT --datasets nocaps textcaps \
  > $OUT/logs/gpu2.log 2>&1"

echo "Launched eval700_gpu0, eval700_gpu1, and eval700_gpu2."
