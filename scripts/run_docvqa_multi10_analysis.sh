#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-1}"

DATASET_PATH="${DATASET_PATH:-/workspace/hd/data/MileBench/DocVQA/DocVQA.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/workspace/hd/data/MileBench/DocVQA/images}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/hd/artifacts/zap}"
EXTRACTOR_DIR="${EXTRACTOR_DIR:-${ARTIFACT_ROOT}/llava_docvqa_full_multi}"
TEACHER_DIR="${TEACHER_DIR:-${ARTIFACT_ROOT}/llava_docvqa_full_multi_10_image_teacher}"
SPLUS_DIR="${SPLUS_DIR:-${ARTIFACT_ROOT}/llava_docvqa_full_multi_10_splus}"

LIMIT="${LIMIT:-10}"
MAX_FULL_ATTENTION_SAMPLES="${MAX_FULL_ATTENTION_SAMPLES:-10}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

echo "[1/3] extract_llava (limit=${LIMIT}, multi-image)"
"${PYTHON_BIN}" /workspace/zap/kvzap/train.py extract_llava \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${EXTRACTOR_DIR}" \
  --image_root "${IMAGE_ROOT}" \
  --image_column images_path \
  --id_column sample_id \
  --implementation_model_name llava-hf/llava-1.5-7b-hf \
  --device_map auto \
  --torch_dtype float16 \
  --save_full_attentions True \
  --max_full_attention_samples "${MAX_FULL_ATTENTION_SAMPLES}" \
  --limit "${LIMIT}" \
  --continue_on_error True \
  --overwrite True

echo "[2/3] analyze_image_teachers"
"${PYTHON_BIN}" /workspace/zap/analyze_image_teachers.py \
  --input_dir "${EXTRACTOR_DIR}" \
  --out_dir "${TEACHER_DIR}" \
  --save_teacher_records

echo "[3/3] analyze_splus_distributions"
"${PYTHON_BIN}" /workspace/zap/analyze_splus_distributions.py \
  --input_dir "${EXTRACTOR_DIR}" \
  --out_dir "${SPLUS_DIR}"

echo "Done"
