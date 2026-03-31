#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-1}"

BASE_TEACHER_DIR="${BASE_TEACHER_DIR:-/workspace/hd/artifacts/zap/llava_docvqa_full_multi_postvision_minimal}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/hd/artifacts/zap/llava_docvqa_full_multi_teacher4}"
MODEL_NAME="${MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float16}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

echo "[teacher4] build_docvqa_teacher4"
"${PYTHON_BIN}" /workspace/zap/build_docvqa_teacher4.py \
  --base_teacher_dir "${BASE_TEACHER_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --implementation_model_name "${MODEL_NAME}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --torch_dtype float16 \
  --device_map auto \
  --attn_implementation eager \
  --storage_dtype "${STORAGE_DTYPE}" \
  --overwrite \
  --continue_on_error \
  "${LIMIT_ARG[@]}"

echo "Done"
