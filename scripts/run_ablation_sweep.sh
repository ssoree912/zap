#!/usr/bin/env bash
# Sweep: probe / oracle / h2o_image_only / oracle_all_token
# All use --total_keep_ratio so r_eff_prompt is directly comparable to LOOK-M.
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"
DATA_ROOT="${DATA_ROOT:-/workspace/hd/data/MileBench}"
ABLATION_ARTIFACT_ROOT="${ABLATION_ARTIFACT_ROOT:-/workspace/hd/artifacts/ablation}"
TEACHER_ROOT="${TEACHER_ROOT:-/workspace/hd/artifacts}"
TOTAL_KEEP_RATIOS="${TOTAL_KEEP_RATIOS:-0.05 0.10 0.20}"
MODES="${MODES:-h2o_image_only oracle_all_token}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DATASETS="${DATASETS:-}"
PROMPT_STYLE="${PROMPT_STYLE:-look_milebench}"
# probe mode settings
PROBE_MODEL_MLP="${PROBE_MODEL_MLP:-/workspace/hd/artifacts/sq_teacher/image_probe_scienceqa_att_only_postvision/mlp}"
PROBE_MODEL_LINEAR="${PROBE_MODEL_LINEAR:-/workspace/hd/artifacts/sq_teacher/image_probe_scienceqa_att_only_postvision/linear}"
PROBE_ARCH="${PROBE_ARCH:-mlp}"  # mlp or linear
# EXP-20260412-003: positional forced-keep parameters
N_INITIAL_KEEP="${N_INITIAL_KEEP:-0}"
N_RECENT_KEEP="${N_RECENT_KEEP:-0}"
N_RANDOM_KEEP="${N_RANDOM_KEEP:-0}"
PER_IMAGE_FORCED="${PER_IMAGE_FORCED:-0}"  # 1 to enable per-image mode
# LOOK-M style truncation (for long-sequence datasets like ActionLocalization)
TRUNCATE_LIKE_LOOKM="${TRUNCATE_LIKE_LOOKM:-0}"  # 1 to enable
LOOK_MAX_CONTEXT_LEN="${LOOK_MAX_CONTEXT_LEN:-4096}"
LOOK_N_TOKENS_PER_IMAGE="${LOOK_N_TOKENS_PER_IMAGE:-576}"

PYTHON_BIN="python"
SCRIPT="$(dirname "$0")/../evaluate_image_teacher_pruning.py"

slugify() {
  local text="$1"
  text="${text//-/_}"
  text="${text// /_}"
  echo "${text}" | tr '[:upper:]' '[:lower:]'
}

resolve_teacher_dir() {
  local dataset_name="$1"
  local slug
  slug="$(slugify "${dataset_name}")"
  # Prefer per-dataset teacher dir; fall back to ScienceQA teacher only if nothing else found
  local candidates=(
    "${TEACHER_ROOT}/teacher/${slug}/records"
    "${TEACHER_ROOT}/teacher/${slug}"
    "${TEACHER_ROOT}/llava_${slug}_full_multi_teacher4/records"
    "${TEACHER_ROOT}/sq_teacher/image_probe_scienceqa_att_only_postvision/teacher_records"
  )
  for dir in "${candidates[@]}"; do
    if [[ -d "${dir}" ]]; then
      echo "${dir}"
      return
    fi
  done
  echo ""
}

run_one_dataset_mode() {
  local dataset_name="$1"
  local mode="$2"
  local ratio="$3"
  local slug
  slug="$(slugify "${dataset_name}")"

  local dataset_path="${DATA_ROOT}/${dataset_name}/${dataset_name}.json"
  local image_root="${DATA_ROOT}/${dataset_name}/images"

  if [[ ! -f "${dataset_path}" ]]; then
    echo "[${dataset_name}] skip missing json: ${dataset_path}"
    return
  fi
  if [[ ! -d "${image_root}" ]]; then
    echo "[${dataset_name}] skip missing image root: ${image_root}"
    return
  fi

  local ratio_tag
  ratio_tag="$(echo "${ratio}" | sed 's/0\./0p/')"
  # For probe mode, include arch in directory name to avoid mlp/linear collision
  local mode_slug="${mode}"
  if [[ "${mode}" == "probe" ]]; then
    mode_slug="probe_${PROBE_ARCH}"
  fi
  # Build forced-keep suffix for EXP-20260412-003 (empty when all zero)
  local forced_suffix=""
  if [[ "${N_INITIAL_KEEP}" -gt 0 ]]; then
    forced_suffix="${forced_suffix}_init${N_INITIAL_KEEP}"
  fi
  if [[ "${N_RECENT_KEEP}" -gt 0 ]]; then
    forced_suffix="${forced_suffix}_rec${N_RECENT_KEEP}"
  fi
  if [[ "${N_RANDOM_KEEP}" -gt 0 ]]; then
    forced_suffix="${forced_suffix}_rnd${N_RANDOM_KEEP}"
  fi
  if [[ "${PER_IMAGE_FORCED}" == "1" ]]; then
    forced_suffix="${forced_suffix}_perimg"
  fi
  local output_dir="${ABLATION_ARTIFACT_ROOT}/${slug}/${mode_slug}/keep_${ratio_tag}${forced_suffix}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/metrics.json" ]]; then
    echo "[${dataset_name}][${mode}][${ratio}] skip existing"
    return
  fi

  echo "[${dataset_name}][${mode}][${ratio}] -> ${output_dir}"

  local extra_args=()
  extra_args+=(--attn_implementation eager)

  # oracle and oracle_all_token need teacher dir
  if [[ "${mode}" == "oracle_all_token" || "${mode}" == "oracle" ]]; then
    local teacher_dir
    teacher_dir="$(resolve_teacher_dir "${dataset_name}")"
    if [[ -z "${teacher_dir}" ]]; then
      echo "[${dataset_name}][${mode}] skip — no teacher dir found"
      return
    fi
    extra_args+=(--teacher_dir "${teacher_dir}")
    extra_args+=(--teacher_score_name att_only_postvision)
  fi

  # probe mode: select model by PROBE_ARCH
  if [[ "${mode}" == "probe" ]]; then
    if [[ "${PROBE_ARCH}" == "linear" ]]; then
      extra_args+=(--probe_model_name "${PROBE_MODEL_LINEAR}")
    else
      extra_args+=(--probe_model_name "${PROBE_MODEL_MLP}")
    fi
  fi

  # LOOK-M style truncation (for long-sequence datasets)
  if [[ "${TRUNCATE_LIKE_LOOKM}" == "1" ]]; then
    extra_args+=(--truncate_like_lookm)
    extra_args+=(--look_max_context_len "${LOOK_MAX_CONTEXT_LEN}")
    extra_args+=(--look_n_tokens_per_image "${LOOK_N_TOKENS_PER_IMAGE}")
  fi

  if [[ -n "${LIMIT}" ]]; then
    extra_args+=(--limit "${LIMIT}")
  fi

  # EXP-20260412-003: positional forced-keep
  if [[ "${N_INITIAL_KEEP}" -gt 0 ]]; then
    extra_args+=(--n_initial_keep "${N_INITIAL_KEEP}")
  fi
  if [[ "${N_RECENT_KEEP}" -gt 0 ]]; then
    extra_args+=(--n_recent_keep "${N_RECENT_KEEP}")
  fi
  if [[ "${N_RANDOM_KEEP}" -gt 0 ]]; then
    extra_args+=(--n_random_keep "${N_RANDOM_KEEP}")
  fi
  if [[ "${PER_IMAGE_FORCED}" == "1" ]]; then
    extra_args+=(--per_image_forced)
  fi

  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
    --mode "${mode}" \
    --dataset_path "${dataset_path}" \
    --image_root "${image_root}" \
    --output_dir "${output_dir}" \
    --total_keep_ratio "${ratio}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --prompt_style "${PROMPT_STYLE}" \
    --look_dataset_name "${dataset_name}" \
    --look_model_name "ablation_${mode_slug}_${slug}" \
    --evaluate_with_look_metrics \
    --save_look_files \
    --allow_partial_look_eval \
    --continue_on_error \
    "${extra_args[@]}"
}

mkdir -p "${ABLATION_ARTIFACT_ROOT}"

collect_datasets() {
  if [[ -n "${DATASETS}" ]]; then
    echo "${DATASETS}"
  else
    find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | xargs -I{} basename {}
  fi
}

for dataset_name in $(collect_datasets); do
  for mode in ${MODES}; do
    for ratio in ${TOTAL_KEEP_RATIOS}; do
      run_one_dataset_mode "${dataset_name}" "${mode}" "${ratio}"
    done
  done
done

echo "Ablation sweep completed."
