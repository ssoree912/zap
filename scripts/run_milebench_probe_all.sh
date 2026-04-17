#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"
DATA_ROOT="${DATA_ROOT:-/workspace/zap/data/MileBench}"
PROB_ARTIFACT_ROOT="${PROB_ARTIFACT_ROOT:-/workspace/zap/artifacts/combine_prob}"
LOOK_RESULT_ROOT="${LOOK_RESULT_ROOT:-${PROB_ARTIFACT_ROOT}/_look_runs}"
KEEP_RATIOS="${KEEP_RATIOS:-0.02 0.05 0.10 0.20}"
TEACHERS="${TEACHERS:-att_only_postvision}"
METHODS="${METHODS:-mlp}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PROMPT_STYLE="${PROMPT_STYLE:-look_milebench}"
DATASETS="${DATASETS:-}"
TRUNCATE_LIKE_LOOKM="${TRUNCATE_LIKE_LOOKM:-0}"
LOOK_MAX_CONTEXT_LEN="${LOOK_MAX_CONTEXT_LEN:-}"
LOOK_N_TOKENS_PER_IMAGE="${LOOK_N_TOKENS_PER_IMAGE:-}"
COMBINE_IMAGE="${COMBINE_IMAGE:-}"
PROBE_LABEL="${PROBE_LABEL:-combined}"

slugify() {
  local text="$1"
  text="${text//-/_}"
  text="${text// /_}"
  echo "${text}" | tr '[:upper:]' '[:lower:]'
}

count_done() {
  local root="$1"
  if [[ -d "${root}" ]]; then
    find "${root}" -maxdepth 4 -type f -name metrics.json | wc -l
  else
    echo 0
  fi
}

count_words() {
  set -- $1
  echo $#
}

run_one() {
  local dataset_name="$1"
  local slug="$2"
  local dataset_path="${DATA_ROOT}/${dataset_name}/${dataset_name}.json"
  local image_root="${DATA_ROOT}/${dataset_name}/images"
  local output_root="${PROB_ARTIFACT_ROOT}/${slug}"

  if [[ ! -f "${dataset_path}" ]]; then
    echo "[${dataset_name}] skip missing dataset json: ${dataset_path}"
    return
  fi
  if [[ ! -d "${image_root}" ]]; then
    echo "[${dataset_name}] skip missing image root: ${image_root}"
    return
  fi

  local n_teachers n_methods n_ratios expected
  n_teachers="$(count_words "${TEACHERS}")"
  n_methods="$(count_words "${METHODS}")"
  n_ratios="$(count_words "${KEEP_RATIOS}")"
  expected=$(( n_teachers * n_methods * n_ratios ))

  local done
  done="$(count_done "${output_root}")"
  if [[ "${SKIP_EXISTING}" == "1" && "${done}" -ge "${expected}" ]]; then
    echo "[${dataset_name}] skip existing probe sweep (${done}/${expected})"
    return
  fi

  echo "[${dataset_name}] probe sweep -> ${output_root}"
  DATASET_PATH="${dataset_path}" \
  IMAGE_ROOT="${image_root}" \
  OUTPUT_ROOT="${output_root}" \
  LOOK_RESULT_ROOT="${LOOK_RESULT_ROOT}" \
  LOOK_MODEL_PREFIX="zap_${slug}_${PROBE_LABEL}_probe" \
  LOOK_DATASET_NAME="${dataset_name}" \
  KEEP_RATIOS="${KEEP_RATIOS}" \
  TEACHERS="${TEACHERS}" \
  METHODS="${METHODS}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  PROMPT_STYLE="${PROMPT_STYLE}" \
  TRUNCATE_LIKE_LOOKM="${TRUNCATE_LIKE_LOOKM}" \
  LOOK_MAX_CONTEXT_LEN="${LOOK_MAX_CONTEXT_LEN}" \
  LOOK_N_TOKENS_PER_IMAGE="${LOOK_N_TOKENS_PER_IMAGE}" \
  COMBINE_IMAGE="${COMBINE_IMAGE}" \
  GPU_INDEX="${GPU_INDEX}" \
  LIMIT="${LIMIT}" \
  bash /workspace/zap/scripts/run_docvqa_probe_teacher_sweep.sh
}

mkdir -p "${PROB_ARTIFACT_ROOT}"
mkdir -p "${LOOK_RESULT_ROOT}"

if [[ -n "${DATASETS}" ]]; then
  for dataset_name in ${DATASETS}; do
    slug="$(slugify "${dataset_name}")"
    run_one "${dataset_name}" "${slug}"
  done
else
  while IFS= read -r dataset_dir; do
    dataset_name="$(basename "${dataset_dir}")"
    slug="$(slugify "${dataset_name}")"
    run_one "${dataset_name}" "${slug}"
  done < <(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

echo "All MileBench probe datasets completed"
