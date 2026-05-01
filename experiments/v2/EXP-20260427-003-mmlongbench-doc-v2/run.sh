#!/usr/bin/env bash
set -euo pipefail

cd /workspace/zap

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

/opt/conda/envs/kv/bin/python \
  experiments/EXP-20260427-003-mmlongbench-doc-v2/collect_mmlongbench_doc_teacher_v2.py \
  --n-samples 500 \
  --dataset-name mmlongbench_doc \
  --output-root /workspace/zap/data/teacher_v2 \
  --device cuda:0
