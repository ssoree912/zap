#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-1}"

DATASET_PATH="${DATASET_PATH:-/workspace/hd/data/MileBench/DocVQA/DocVQA.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/workspace/hd/data/MileBench/DocVQA/images}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/hd/artifacts/zap}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/llava_docvqa_full_multi_postvision_minimal}"

MODEL_NAME="${MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"

# Lean defaults: compute teacher on GPU and only save compact outputs.
SAVE_HIDDEN_IMAGE="${SAVE_HIDDEN_IMAGE:-False}"
SAVE_ATTN_POSTVISION_TO_IMAGE="${SAVE_ATTN_POSTVISION_TO_IMAGE:-False}"
SAVE_WOV_NORM_IMAGE="${SAVE_WOV_NORM_IMAGE:-False}"
SAVE_H_NORM_POSTVISION="${SAVE_H_NORM_POSTVISION:-False}"
SAVE_TEACHER_SCORES="${SAVE_TEACHER_SCORES:-True}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float16}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

echo "[postvision-minimal] extract_llava_postvision (multi-image)"
"${PYTHON_BIN}" /workspace/zap/kvzap/train.py extract_llava_postvision \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_root "${IMAGE_ROOT}" \
  --image_column images_path \
  --id_column sample_id \
  --implementation_model_name "${MODEL_NAME}" \
  --device_map auto \
  --torch_dtype float16 \
  --capture_vproj True \
  --save_hidden_image "${SAVE_HIDDEN_IMAGE}" \
  --save_attn_postvision_to_image "${SAVE_ATTN_POSTVISION_TO_IMAGE}" \
  --save_wov_norm_image "${SAVE_WOV_NORM_IMAGE}" \
  --save_h_norm_postvision "${SAVE_H_NORM_POSTVISION}" \
  --save_teacher_scores "${SAVE_TEACHER_SCORES}" \
  --storage_dtype "${STORAGE_DTYPE}" \
  --continue_on_error True \
  --overwrite True \
  "${LIMIT_ARG[@]}"

echo "Done"
