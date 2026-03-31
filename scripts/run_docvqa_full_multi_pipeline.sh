#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-2}"

DATASET_PATH="${DATASET_PATH:-/workspace/hd/data/MileBench/DocVQA/DocVQA.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/workspace/hd/data/MileBench/DocVQA/images}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/hd/artifacts/zap}"
EXTRACTOR_DIR="${EXTRACTOR_DIR:-${ARTIFACT_ROOT}/llava_docvqa_full_multi}"
TEACHER_DIR="${TEACHER_DIR:-${ARTIFACT_ROOT}/llava_docvqa_full_multi_image_teacher}"
PROBE_DIR="${PROBE_DIR:-${ARTIFACT_ROOT}/image_probe_splus_postvision_full_multi}"

MODEL_NAME="${MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"
TARGET_SCORE_NAME="${TARGET_SCORE_NAME:-splus_postvision}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
MAX_IMAGE_TOKENS_PER_SAMPLE="${MAX_IMAGE_TOKENS_PER_SAMPLE:-128}"
MAX_FULL_ATTENTION_SAMPLES="${MAX_FULL_ATTENTION_SAMPLES:-200}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

echo "[1/3] extract_llava: full DocVQA multi-image"
"${PYTHON_BIN}" /workspace/zap/kvzap/train.py extract_llava \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${EXTRACTOR_DIR}" \
  --image_root "${IMAGE_ROOT}" \
  --image_column images_path \
  --id_column sample_id \
  --implementation_model_name "${MODEL_NAME}" \
  --device_map auto \
  --torch_dtype float16 \
  --save_full_attentions True \
  --max_full_attention_samples "${MAX_FULL_ATTENTION_SAMPLES}" \
  --continue_on_error True \
  --overwrite True

echo "[2/3] analyze_image_teachers"
"${PYTHON_BIN}" /workspace/zap/analyze_image_teachers.py \
  --input_dir "${EXTRACTOR_DIR}" \
  --out_dir "${TEACHER_DIR}" \
  --save_teacher_records

echo "[3/3] train_image_teacher_probe"
"${PYTHON_BIN}" /workspace/zap/train_image_teacher_probe.py \
  --extractor_dir "${EXTRACTOR_DIR}" \
  --teacher_dir "${TEACHER_DIR}" \
  --output_dir "${PROBE_DIR}" \
  --target_score_name "${TARGET_SCORE_NAME}" \
  --methods linear mlp \
  --train_fraction "${TRAIN_FRACTION}" \
  --max_image_tokens_per_sample "${MAX_IMAGE_TOKENS_PER_SAMPLE}" \
  --device cuda:0

echo "Pipeline complete"
