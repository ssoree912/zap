#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"

DATASET_PATH="${DATASET_PATH:-/workspace/zap/data/MileBench/DocVQA/DocVQA.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/workspace/zap/data/MileBench/DocVQA/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/zap/artifacts/combine_prob/docvqa}"
LOOK_RESULT_ROOT="${LOOK_RESULT_ROOT:-}"
LOOK_MODEL_PREFIX="${LOOK_MODEL_PREFIX:-zap_docvqa_scienceqa_probe}"
LOOK_DATASET_NAME="${LOOK_DATASET_NAME:-DocVQA}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

IMPLEMENTATION_MODEL_NAME="${IMPLEMENTATION_MODEL_NAME:-/workspace/zap/ckpts/llava-1.5-7b-hf}"
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
N_ITERATIVE_ROUNDS="${N_ITERATIVE_ROUNDS:-1}"
LAYERWISE_ITERATIVE="${LAYERWISE_ITERATIVE:-0}"
METRICS_PYTHON_BIN="${METRICS_PYTHON_BIN:-python3}"

ATT_ONLY_PROBE_ROOT="${ATT_ONLY_PROBE_ROOT:-/workspace/zap/ckpts/image_probe_combined_v1}"
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

resolve_method_dir_name() {
  local teacher="$1"
  local method="$2"
  if [[ "${N_ITERATIVE_ROUNDS}" -gt 1 ]]; then
    if [[ "${LAYERWISE_ITERATIVE}" == "1" ]]; then
      echo "iterative_ver2_${N_ITERATIVE_ROUNDS}"
    else
      echo "iterative_${N_ITERATIVE_ROUNDS}"
    fi
    return
  fi
  if [[ "${teacher}" == "att_only_postvision" ]]; then
    echo "probe_${method}"
  elif [[ "${teacher}" == "splus_postvision" ]]; then
    echo "probe_splus_${method}"
  else
    echo "${teacher}_${method}"
  fi
}

metrics_is_success() {
  local metrics_path="$1"
  [[ -f "${metrics_path}" ]] || return 1
  "${METRICS_PYTHON_BIN}" - "${metrics_path}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

n_failures = data.get("n_failures")
if n_failures is None:
    sys.exit(0)

try:
    n_failures = int(n_failures)
except Exception:
    sys.exit(1)

sys.exit(0 if n_failures == 0 else 1)
PY
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

    method_dir_name="$(resolve_method_dir_name "${teacher}" "${method}")"

    for ratio in ${KEEP_RATIOS}; do
      ratio_tag="${ratio//./p}"
      out_dir="${OUTPUT_ROOT}/${method_dir_name}/keep_${ratio_tag}"
      look_model_name="${LOOK_MODEL_PREFIX}_${teacher}_${method}_k${ratio_tag}"
      mkdir -p "${out_dir}"

      metrics_path="${out_dir}/metrics.json"
      if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics_path}" ]]; then
        if metrics_is_success "${metrics_path}"; then
          echo "[probe-sweep] skip existing teacher=${teacher} method=${method} keep=${ratio} -> ${out_dir}"
          continue
        fi
        echo "[probe-sweep] re-run failed/incomplete output teacher=${teacher} method=${method} keep=${ratio} -> ${out_dir}"
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
      if [[ "${N_ITERATIVE_ROUNDS}" -gt 1 ]]; then
        cmd+=(--n_iterative_rounds "${N_ITERATIVE_ROUNDS}")
        if [[ "${LAYERWISE_ITERATIVE}" == "1" ]]; then
          cmd+=(--layerwise_iterative)
        fi
      fi

      "${cmd[@]}"
    done
  done
done

echo "Done"
