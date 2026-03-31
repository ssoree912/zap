#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-probe}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
DATASET_PATH="${DATASET_PATH:-/workspace/hd/data/MileBench/DocVQA/DocVQA.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/workspace/hd/data/MileBench/DocVQA/images}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/hd/artifacts/zap}"
LOOK_DATASET_NAME="${LOOK_DATASET_NAME:-DocVQA}"
LOOK_MODEL_NAME="${LOOK_MODEL_NAME:-zap_docvqa_${MODE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/${LOOK_MODEL_NAME}}"
IMPLEMENTATION_MODEL_NAME="${IMPLEMENTATION_MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"
IMAGE_KEEP_RATIO="${IMAGE_KEEP_RATIO:-0.10}"
HEAD_REDUCE="${HEAD_REDUCE:-amax}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PROMPT_STYLE="${PROMPT_STYLE:-look_milebench}"

CMD=(
  "${PYTHON_BIN}" /workspace/zap/evaluate_image_teacher_pruning.py
  --mode "${MODE}"
  --dataset_path "${DATASET_PATH}"
  --image_root "${IMAGE_ROOT}"
  --image_column images_path
  --output_dir "${OUTPUT_DIR}"
  --implementation_model_name "${IMPLEMENTATION_MODEL_NAME}"
  --image_keep_ratio "${IMAGE_KEEP_RATIO}"
  --head_reduce "${HEAD_REDUCE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --prompt_style "${PROMPT_STYLE}"
  --look_dataset_name "${LOOK_DATASET_NAME}"
  --look_model_name "${LOOK_MODEL_NAME}"
  --look_result_root "${ARTIFACT_ROOT}"
  --save_look_files
  --evaluate_with_look_metrics
)

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "${LIMIT}" --allow_partial_look_eval)
fi

if [[ "${MODE}" == "oracle" ]]; then
  TEACHER_DIR="${TEACHER_DIR:-${ARTIFACT_ROOT}/llava_docvqa_image_teacher_analysis}"
  TEACHER_SCORE_NAME="${TEACHER_SCORE_NAME:-splus_postvision}"
  CMD+=(
    --teacher_dir "${TEACHER_DIR}"
    --teacher_score_name "${TEACHER_SCORE_NAME}"
  )
elif [[ "${MODE}" == "probe" ]]; then
  PROBE_MODEL_NAME="${PROBE_MODEL_NAME:-${ARTIFACT_ROOT}/image_probe_splus_postvision/linear}"
  CMD+=(--probe_model_name "${PROBE_MODEL_NAME}")
else
  echo "Unsupported mode: ${MODE}. Use 'oracle' or 'probe'." >&2
  exit 1
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
