#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/hd/artifacts/zap}"
DATA_ROOT="${DATA_ROOT:-/workspace/hd/data/MileBench}"
LOOK_RESULT_ROOT="${LOOK_RESULT_ROOT:-}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-200}"

# Must match sweep defaults unless overridden from env.
TEACHER_SCORES="${TEACHER_SCORES:-att_only_answer splus_answer att_only_postvision splus_postvision}"
KEEP_RATIOS="${KEEP_RATIOS:-0.02 0.05 0.10 0.20}"

count_records() {
  local root="$1"
  if [[ -d "${root}/records" ]]; then
    find "${root}/records" -maxdepth 1 -type f | wc -l
  else
    echo 0
  fi
}

count_oracle_done() {
  local root="$1"
  if [[ -d "${root}" ]]; then
    find "${root}" -maxdepth 3 -type f -name metrics.json | wc -l
  else
    echo 0
  fi
}

count_words() {
  # shellcheck disable=SC2086
  set -- $1
  echo $#
}

run_one() {
  local dataset_name="$1"
  local slug="$2"
  local dataset_path="${DATA_ROOT}/${dataset_name}/${dataset_name}.json"
  local image_root="${DATA_ROOT}/${dataset_name}/images"

  local postvision_dir="${ARTIFACT_ROOT}/llava_${slug}_full_multi_postvision_minimal"
  local teacher4_dir="${ARTIFACT_ROOT}/llava_${slug}_full_multi_teacher4"
  local oracle_root="${ARTIFACT_ROOT}/${slug}_teacher_oracle_sweep"

  local n_scores n_ratios expected_oracle
  n_scores="$(count_words "${TEACHER_SCORES}")"
  n_ratios="$(count_words "${KEEP_RATIOS}")"
  expected_oracle=$(( n_scores * n_ratios ))

  echo "============================================================"

  local postvision_count
  postvision_count="$(count_records "${postvision_dir}")"
  if [[ "${postvision_count}" -ge "${EXPECTED_RECORDS}" ]]; then
    echo "[${dataset_name}] stage 1/3: skip (postvision exists ${postvision_count}/${EXPECTED_RECORDS})"
  else
    echo "[${dataset_name}] stage 1/3: extract postvision minimal"
    DATASET_PATH="${dataset_path}" \
    IMAGE_ROOT="${image_root}" \
    OUTPUT_DIR="${postvision_dir}" \
    GPU_INDEX="${GPU_INDEX}" \
    bash /workspace/zap/scripts/run_docvqa_full_multi_postvision_minimal.sh
  fi

  local teacher4_count
  teacher4_count="$(count_records "${teacher4_dir}")"
  if [[ "${teacher4_count}" -ge "${EXPECTED_RECORDS}" ]]; then
    echo "[${dataset_name}] stage 2/3: skip (teacher4 exists ${teacher4_count}/${EXPECTED_RECORDS})"
  else
    echo "[${dataset_name}] stage 2/3: build teacher4"
    BASE_TEACHER_DIR="${postvision_dir}" \
    OUTPUT_DIR="${teacher4_dir}" \
    GPU_INDEX="${GPU_INDEX}" \
    bash /workspace/zap/scripts/run_docvqa_build_teacher4.sh
  fi

  local oracle_done
  oracle_done="$(count_oracle_done "${oracle_root}")"
  if [[ "${oracle_done}" -ge "${expected_oracle}" ]]; then
    echo "[${dataset_name}] stage 3/3: skip (oracle exists ${oracle_done}/${expected_oracle})"
  else
    echo "[${dataset_name}] stage 3/3: oracle sweep (4 teachers x 4 keep ratios)"
    DATASET_PATH="${dataset_path}" \
    IMAGE_ROOT="${image_root}" \
    TEACHER_DIR="${teacher4_dir}" \
    OUTPUT_ROOT="${oracle_root}" \
    LOOK_MODEL_PREFIX="zap_${slug}_teacher_oracle" \
    LOOK_DATASET_NAME="${dataset_name}" \
    LOOK_RESULT_ROOT="${LOOK_RESULT_ROOT}" \
    TEACHER_SCORES="${TEACHER_SCORES}" \
    KEEP_RATIOS="${KEEP_RATIOS}" \
    SKIP_EXISTING=1 \
    GPU_INDEX="${GPU_INDEX}" \
    bash /workspace/zap/scripts/run_docvqa_oracle_teacher_sweep.sh
  fi

  echo "[${dataset_name}] done"
}

run_one "Spot-the-Diff" "spot_the_diff"
run_one "CLEVR-Change" "clevr_change"
run_one "IEdit" "iedit"

echo "All datasets completed"
