#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-1}"

DATASET_PATH="${DATASET_PATH:-/workspace/hd/data/MileBench/DocVQA/DocVQA.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/workspace/hd/data/MileBench/DocVQA/images}"
TEACHER_DIR="${TEACHER_DIR:-/workspace/hd/artifacts/zap/llava_docvqa_full_multi_teacher4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/hd/artifacts/zap/docvqa_teacher_oracle_sweep}"
LOOK_RESULT_ROOT="${LOOK_RESULT_ROOT:-}"
LOOK_MODEL_PREFIX="${LOOK_MODEL_PREFIX:-zap_docvqa_teacher_oracle}"
LOOK_DATASET_NAME="${LOOK_DATASET_NAME:-DocVQA}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# space-separated
TEACHER_SCORES="${TEACHER_SCORES:-att_only_answer splus_answer att_only_postvision splus_postvision}"
KEEP_RATIOS="${KEEP_RATIOS:-0.02 0.05 0.10 0.20}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
mkdir -p "${OUTPUT_ROOT}"

echo "[oracle-sweep] teacher_dir=${TEACHER_DIR}"
for score in ${TEACHER_SCORES}; do
  for ratio in ${KEEP_RATIOS}; do
    ratio_tag="${ratio//./p}"
    out_dir="${OUTPUT_ROOT}/${score}/keep_${ratio_tag}"
    look_model_name="${LOOK_MODEL_PREFIX}_${score}_k${ratio_tag}"
    mkdir -p "${out_dir}"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/metrics.json" ]]; then
      echo "[oracle-sweep] skip existing score=${score} keep=${ratio} -> ${out_dir}"
      continue
    fi

    echo "[oracle-sweep] score=${score} keep=${ratio} -> ${out_dir}"

    "${PYTHON_BIN}" /workspace/zap/evaluate_image_teacher_pruning.py \
      --mode oracle \
      --dataset_path "${DATASET_PATH}" \
      --image_root "${IMAGE_ROOT}" \
      --image_column images_path \
      --teacher_dir "${TEACHER_DIR}" \
      --teacher_score_name "${score}" \
      --image_keep_ratio "${ratio}" \
      --output_dir "${out_dir}" \
      --implementation_model_name llava-hf/llava-1.5-7b-hf \
      --torch_dtype float16 \
      --device_map auto \
      --attn_implementation eager \
      --look_dataset_name "${LOOK_DATASET_NAME}" \
      --look_model_name "${look_model_name}" \
      --look_result_root "${LOOK_RESULT_ROOT}" \
      --save_look_files \
      --evaluate_with_look_metrics \
      --continue_on_error
  done
done

echo "Done"
