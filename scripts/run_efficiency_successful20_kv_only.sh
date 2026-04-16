#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_DIR=${OUT_DIR:-/workspace/hd/artifacts/prob/efficiency_successful_random20_zap}
MILEBENCH_ROOT=${MILEBENCH_ROOT:-/workspace/hd/data/MileBench}
SAMPLE_SIZE=${SAMPLE_SIZE:-20}
SEED=${SEED:-42}
DATASETS=(ALFRED CLEVR-Change CounterfactualInference DocVQA IEdit MovingAttribute MovingDirection ObjectExistence OCR-VQA Spot-the-Diff)

mkdir -p "$(dirname "$OUT_DIR")"

/opt/conda/envs/kv/bin/python /workspace/zap/scripts/measure_milebench_efficiency.py   --milebench_root "$MILEBENCH_ROOT"   --datasets "${DATASETS[@]}"   --sample_size "$SAMPLE_SIZE"   --seed "$SEED"   --output_dir "$OUT_DIR"   --device cuda:0   --probe_image_keep_ratio 0.20   --no-include_oracle   --include_zap   --no-include_lookm
