#!/usr/bin/env bash
set -euo pipefail

cd /workspace/zap

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

/opt/conda/envs/kv/bin/python train_visual_utility_student.py \
  --teacher-root /workspace/zap/data/teacher_v2 \
  --datasets mmlongbench_doc \
  --scope A \
  --epochs 20 \
  --lr 1e-4 \
  --output-dir /workspace/zap/ckpts/student_v2_A_mmlongbench_lr1e4_20ep \
  --device cuda:0 \
  --log-every 25
