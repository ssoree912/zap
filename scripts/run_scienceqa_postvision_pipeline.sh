#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"

SCIENCEQA_DIR="${SCIENCEQA_DIR:-/workspace/hd/data/scienceqa}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/hd/artifacts/zap}"
MODEL_NAME="${MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"
TEACHER_TYPE="${TEACHER_TYPE:-splus_postvision}"

XY_ROOT="${XY_ROOT:-${ARTIFACT_ROOT}/scienceqa_xy/${TEACHER_TYPE}}"
TRAIN_SHARD_DIR="${TRAIN_SHARD_DIR:-${XY_ROOT}/train}"
VAL_SHARD_DIR="${VAL_SHARD_DIR:-${XY_ROOT}/val}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${ARTIFACT_ROOT}/image_probe_scienceqa_${TEACHER_TYPE}}"

SAMPLE_IMAGE_TOKENS="${SAMPLE_IMAGE_TOKENS:-all}"
VAL_SAMPLE_IMAGE_TOKENS="${VAL_SAMPLE_IMAGE_TOKENS:-${SAMPLE_IMAGE_TOKENS}}"
TOP_FRACTION="${TOP_FRACTION:-0.20}"
RANDOM_FRACTION="${RANDOM_FRACTION:-0.20}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
TRAIN_METHODS="${TRAIN_METHODS:-linear mlp}"
MLP_MAX_EPOCHS="${MLP_MAX_EPOCHS:-10}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

TRAIN_LIMIT_ARG=()
if [[ -n "${TRAIN_LIMIT:-${LIMIT:-}}" ]]; then
  TRAIN_LIMIT_ARG=(--limit "${TRAIN_LIMIT:-${LIMIT:-}}")
fi

VAL_LIMIT_ARG=()
if [[ -n "${VAL_LIMIT:-${LIMIT:-}}" ]]; then
  VAL_LIMIT_ARG=(--limit "${VAL_LIMIT:-${LIMIT:-}}")
fi

MAX_TOKENS_ARG=()
if [[ -n "${MAX_TOKENS_PER_LAYER}" ]]; then
  MAX_TOKENS_ARG=(--max_tokens_per_layer "${MAX_TOKENS_PER_LAYER}")
fi

DEVICE_ARG=()
if [[ -n "${DEVICE}" ]]; then
  DEVICE_ARG=(--device "${DEVICE}")
fi

DEVICE_MAP_ARG=()
if [[ -n "${DEVICE_MAP}" ]]; then
  DEVICE_MAP_ARG=(--device_map "${DEVICE_MAP}")
fi

echo "[1/3] collect train shards (${TEACHER_TYPE})"
"${PYTHON_BIN}" /workspace/zap/collect_scienceqa_teacher_xy.py \
  --base_dir "${SCIENCEQA_DIR}" \
  --split train \
  --out_dir "${TRAIN_SHARD_DIR}" \
  --teacher_type "${TEACHER_TYPE}" \
  --implementation_model_name "${MODEL_NAME}" \
  --sample_image_tokens "${SAMPLE_IMAGE_TOKENS}" \
  --top_fraction "${TOP_FRACTION}" \
  --random_fraction "${RANDOM_FRACTION}" \
  "${MAX_TOKENS_ARG[@]}" \
  --shard_size "${SHARD_SIZE}" \
  --target_transform log \
  --torch_dtype float16 \
  --storage_dtype float16 \
  --target_dtype float32 \
  --teacher_head_reduction mean \
  "${DEVICE_ARG[@]}" \
  "${DEVICE_MAP_ARG[@]}" \
  --overwrite \
  --continue_on_error \
  "${TRAIN_LIMIT_ARG[@]}"

echo "[2/3] collect val shards (${TEACHER_TYPE})"
"${PYTHON_BIN}" /workspace/zap/collect_scienceqa_teacher_xy.py \
  --base_dir "${SCIENCEQA_DIR}" \
  --split val \
  --out_dir "${VAL_SHARD_DIR}" \
  --teacher_type "${TEACHER_TYPE}" \
  --implementation_model_name "${MODEL_NAME}" \
  --sample_image_tokens "${VAL_SAMPLE_IMAGE_TOKENS}" \
  --top_fraction "${TOP_FRACTION}" \
  --random_fraction "${RANDOM_FRACTION}" \
  "${MAX_TOKENS_ARG[@]}" \
  --shard_size "${SHARD_SIZE}" \
  --target_transform log \
  --torch_dtype float16 \
  --storage_dtype float16 \
  --target_dtype float32 \
  --teacher_head_reduction mean \
  "${DEVICE_ARG[@]}" \
  "${DEVICE_MAP_ARG[@]}" \
  --overwrite \
  --continue_on_error \
  "${VAL_LIMIT_ARG[@]}"

echo "[3/3] train probe from shards"
"${PYTHON_BIN}" /workspace/zap/train_image_teacher_probe_shards.py \
  --train_shard_dir "${TRAIN_SHARD_DIR}" \
  --val_shard_dir "${VAL_SHARD_DIR}" \
  --output_dir "${PROBE_OUTPUT_DIR}" \
  --methods ${TRAIN_METHODS} \
  --mlp_max_epochs "${MLP_MAX_EPOCHS}" \
  --device cuda:0

echo "Done"
