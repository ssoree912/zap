#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT_DIR=${OUT_DIR:-/workspace/hd/artifacts/prob/efficiency_successful_random20_lookm}
MILEBENCH_ROOT=${MILEBENCH_ROOT:-/workspace/hd/data/MileBench}
MANIFEST_PATH=${MANIFEST_PATH:-/workspace/hd/artifacts/prob/efficiency_successful_random20_zap/sample_manifest.json}
SAMPLE_SIZE=${SAMPLE_SIZE:-20}
SEED=${SEED:-42}
DATASETS=(ALFRED CLEVR-Change CounterfactualInference DocVQA IEdit MovingAttribute MovingDirection ObjectExistence OCR-VQA Spot-the-Diff)

mkdir -p "$(dirname "$OUT_DIR")"

/opt/conda/envs/look/bin/python /workspace/zap/scripts/measure_milebench_efficiency.py   --milebench_root "$MILEBENCH_ROOT"   --datasets "${DATASETS[@]}"   --sample_size "$SAMPLE_SIZE"   --seed "$SEED"   --output_dir "$OUT_DIR"   --device cuda:0   --sample_manifest_path "$MANIFEST_PATH"   --no-include_oracle   --no-include_zap   --include_lookm   --look_kv_mode text_prior_pivot_merge   --look_hh_ratio 0.10   --look_recent_ratio 0.10
