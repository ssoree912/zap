#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"

DATASET_PATH="${DATASET_PATH:-/workspace/hd/data/MileBench/DocVQA/DocVQA.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/workspace/hd/data/MileBench/DocVQA/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/hd/artifacts/prob/docvqa_scienceqa_probe_sweep}"
LOOK_RESULT_ROOT="${LOOK_RESULT_ROOT:-}"
LOOK_MODEL_PREFIX="${LOOK_MODEL_PREFIX:-zap_docvqa_scienceqa_probe}"
LOOK_DATASET_NAME="${LOOK_DATASET_NAME:-DocVQA}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

IMPLEMENTATION_MODEL_NAME="${IMPLEMENTATION_MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
HEAD_REDUCE="${HEAD_REDUCE:-amax}"
PROMPT_STYLE="${PROMPT_STYLE:-look_milebench}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
LIMIT="${LIMIT:-}"
TRUNCATE_LIKE_LOOKM="${TRUNCATE_LIKE_LOOKM:-0}"
LOOK_MAX_CONTEXT_LEN="${LOOK_MAX_CONTEXT_LEN:-}"
LOOK_N_TOKENS_PER_IMAGE="${LOOK_N_TOKENS_PER_IMAGE:-}"
COMBINE_IMAGE="${COMBINE_IMAGE:-}"

KEEP_RATIOS="${KEEP_RATIOS:-0.02 0.05 0.10 0.20}"
TEACHERS="${TEACHERS:-att_only_postvision splus_postvision}"
METHODS="${METHODS:-linear mlp}"
USE_TOTAL_KEEP_RATIO="${USE_TOTAL_KEEP_RATIO:-0}"

ATT_ONLY_PROBE_ROOT="${ATT_ONLY_PROBE_ROOT:-/workspace/hd/artifacts/sq_teacher/image_probe_scienceqa_att_only_postvision}"
SPLUS_PROBE_ROOT="${SPLUS_PROBE_ROOT:-/workspace/hd/artifacts/sq_teacher/image_probe_scienceqa_splus_postvision_small}"

resolve_probe_root() {
  local teacher="$1"
  case "${teacher}" in
    att_only_postvision)
      echo "${ATT_ONLY_PROBE_ROOT}"
      ;;
    splus_postvision)
      echo "${SPLUS_PROBE_ROOT}"
      ;;
    *)
      echo "Unknown teacher: ${teacher}" >&2
      exit 1
      ;;
  esac
}

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
mkdir -p "${OUTPUT_ROOT}"

echo "[probe-sweep] output_root=${OUTPUT_ROOT}"
for teacher in ${TEACHERS}; do
  probe_root="$(resolve_probe_root "${teacher}")"
  for method in ${METHODS}; do
    probe_model_name="${probe_root}/${method}"
    if [[ ! -f "${probe_model_name}/config.json" ]]; then
      echo "[probe-sweep] skip missing ${teacher}/${method}: ${probe_model_name}"
      continue
    fi

    for ratio in ${KEEP_RATIOS}; do
      ratio_tag="${ratio//./p}"
      out_dir="${OUTPUT_ROOT}/${teacher}/${method}/keep_${ratio_tag}"
      look_model_name="${LOOK_MODEL_PREFIX}_${teacher}_${method}_k${ratio_tag}"
      mkdir -p "${out_dir}"

      if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/metrics.json" ]]; then
        echo "[probe-sweep] skip existing teacher=${teacher} method=${method} keep=${ratio} -> ${out_dir}"
        continue
      fi

      echo "[probe-sweep] teacher=${teacher} method=${method} keep=${ratio} -> ${out_dir}"
      cmd=(
        "${PYTHON_BIN}" /workspace/zap/evaluate_image_teacher_pruning.py
        --mode probe
        --dataset_path "${DATASET_PATH}"
        --image_root "${IMAGE_ROOT}"
        --image_column images_path
        --probe_model_name "${probe_model_name}"
        $([ "${USE_TOTAL_KEEP_RATIO}" == "1" ] && echo "--total_keep_ratio" || echo "--image_keep_ratio") "${ratio}"
        --output_dir "${out_dir}"
        --implementation_model_name "${IMPLEMENTATION_MODEL_NAME}"
        --torch_dtype "${TORCH_DTYPE}"
        --device "${DEVICE}"
        --device_map "${DEVICE_MAP}"
        --attn_implementation "${ATTN_IMPLEMENTATION}"
        --head_reduce "${HEAD_REDUCE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --prompt_style "${PROMPT_STYLE}"
        --look_dataset_name "${LOOK_DATASET_NAME}"
        --look_model_name "${look_model_name}"
        --look_result_root "${LOOK_RESULT_ROOT}"
        --save_look_files
        --evaluate_with_look_metrics
        --continue_on_error
      )

      if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}" --allow_partial_look_eval)
      fi
      if [[ "${TRUNCATE_LIKE_LOOKM}" == "1" ]]; then
        cmd+=(--truncate_like_lookm)
      fi
      if [[ -n "${LOOK_MAX_CONTEXT_LEN}" ]]; then
        cmd+=(--look_max_context_len "${LOOK_MAX_CONTEXT_LEN}")
      fi
      if [[ -n "${LOOK_N_TOKENS_PER_IMAGE}" ]]; then
        cmd+=(--look_n_tokens_per_image "${LOOK_N_TOKENS_PER_IMAGE}")
      fi
      if [[ -n "${COMBINE_IMAGE}" ]]; then
        cmd+=(--combine_image "${COMBINE_IMAGE}")
      fi

      "${cmd[@]}"
    done
  done
done

echo "Done"
