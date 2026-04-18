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
N_ITERATIVE_ROUNDS="${N_ITERATIVE_ROUNDS:-1}"
LAYERWISE_ITERATIVE="${LAYERWISE_ITERATIVE:-0}"
METRICS_PYTHON_BIN="${METRICS_PYTHON_BIN:-python3}"
EXPORT_RESULTS_AFTER_RUN="${EXPORT_RESULTS_AFTER_RUN:-1}"
COMBINE_VARIANT="${COMBINE_VARIANT:-probe_mlp}"
ITERATIVE_VARIANT="${ITERATIVE_VARIANT:-auto}"
BASE_COMPARE_CSV="${BASE_COMPARE_CSV:-/workspace/zap/artifacts/probe_global/probe_vs_lookm_r020_truncated.csv}"
RESULTS_SOURCE_CSV="${RESULTS_SOURCE_CSV:-/workspace/zap/artifacts/results/probe_vs_lookm_r020_truncated_with_combine.csv}"
RESULTS_OUTPUT_CSV="${RESULTS_OUTPUT_CSV:-/workspace/zap/artifacts/results/results.csv}"

slugify() {
  local text="$1"
  text="${text//-/_}"
  text="${text// /_}"
  echo "${text}" | tr '[:upper:]' '[:lower:]'
}

count_done() {
  local root="$1"
  # When running iterative, only count within the iterative_N subdir to avoid
  # false-positive skip caused by pre-existing probe_mlp results.
  if [[ "${N_ITERATIVE_ROUNDS}" -gt 1 ]]; then
    if [[ "${LAYERWISE_ITERATIVE}" == "1" ]]; then
      root="${root}/iterative_ver2_${N_ITERATIVE_ROUNDS}"
    else
      root="${root}/iterative_${N_ITERATIVE_ROUNDS}"
    fi
  fi
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi

  local count=0 metrics_path
  while IFS= read -r metrics_path; do
    if "${METRICS_PYTHON_BIN}" - "${metrics_path}" <<'PY'
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
    then
      count=$((count + 1))
    fi
  done < <(find "${root}" -maxdepth 4 -type f -name metrics.json)

  echo "${count}"
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
  N_ITERATIVE_ROUNDS="${N_ITERATIVE_ROUNDS}" \
  LAYERWISE_ITERATIVE="${LAYERWISE_ITERATIVE}" \
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

if [[ "${EXPORT_RESULTS_AFTER_RUN}" == "1" && -z "${DATASETS}" ]]; then
  echo "Exporting aggregated MileBench results"
  python3 /workspace/zap/scripts/export_probe_vs_lookm_with_combine.py \
    --base_csv "${BASE_COMPARE_CSV}" \
    --combine_root "${PROB_ARTIFACT_ROOT}" \
    --combine_variant "${COMBINE_VARIANT}" \
    --iterative_variant "${ITERATIVE_VARIANT}" \
    --output_csv "${RESULTS_SOURCE_CSV}" \
    --results_csv "${RESULTS_OUTPUT_CSV}"
fi
