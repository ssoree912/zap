#!/usr/bin/env bash
set -euo pipefail

readonly ZAP_ROOT="/workspace/nips/zap"
readonly EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260727-007-full-vqa-pa-qa-paired"
readonly DEFAULT_TRAINING_RUN_ROOT="${ZAP_ROOT}/artifacts/paired_prefill_question_answer/training_runs/20260727_070541"
readonly TRAINING_RUN_ROOT="${TRAINING_RUN_ROOT:-${DEFAULT_TRAINING_RUN_ROOT}}"
readonly TRAINING_STATE_ROOT="${TRAINING_RUN_ROOT}/state"
readonly TRAINING_VERIFICATION="${TRAINING_RUN_ROOT}/training_verification.json"
readonly POLL_SECONDS="${POLL_SECONDS:-20}"
readonly HANDOFF_ROOT="${TRAINING_RUN_ROOT}/full_vqa_exp007_handoff"

if [[ ! -d "${TRAINING_RUN_ROOT}" || ! -d "${TRAINING_STATE_ROOT}" ]]; then
  echo "Training run/state directory is missing: ${TRAINING_RUN_ROOT}" >&2
  exit 1
fi
if [[ -e "${HANDOFF_ROOT}" ]]; then
  echo "Refusing to reuse existing evaluation handoff: ${HANDOFF_ROOT}" >&2
  exit 1
fi
mkdir "${HANDOFF_ROOT}"

readonly orchestration_log="${HANDOFF_ROOT}/launch_after_training.log"
readonly launch_stdout="${HANDOFF_ROOT}/run_4gpu.stdout"
readonly evaluation_run_root_record="${HANDOFF_ROOT}/evaluation_run_root.txt"
readonly failed_marker="${HANDOFF_ROOT}/launch_after_training.failed"
readonly done_marker="${HANDOFF_ROOT}/launch_after_training.done"

exec > >(tee -a "${orchestration_log}") 2>&1

current_stage="wait_for_training_verification"
record_failure() {
  local status=$?
  if ((status != 0)); then
    printf 'stage=%s\nexit_code=%s\ntraining_run_root=%s\n' \
      "${current_stage}" "${status}" "${TRAINING_RUN_ROOT}" >"${failed_marker}"
  fi
}
trap record_failure EXIT

echo "[wait] training_run_root=${TRAINING_RUN_ROOT}"
echo "[wait] polling every ${POLL_SECONDS}s for state/verification.done"
while true; do
  failure="$(find "${TRAINING_STATE_ROOT}" -maxdepth 1 -type f -name '*.failed' -print -quit)"
  if [[ -n "${failure}" ]]; then
    echo "[wait] paired training failed: ${failure}" >&2
    exit 1
  fi
  if [[ -f "${TRAINING_STATE_ROOT}/verification.done" ]]; then
    if [[ ! -s "${TRAINING_VERIFICATION}" ]]; then
      echo "[wait] verification.done exists but the verification JSON is missing" >&2
      exit 1
    fi
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[wait] paired training verification passed"
current_stage="wait_for_all_evaluation_gpus"
while true; do
  busy=()
  for gpu in 0 1 2 3; do
    used_mb="$(nvidia-smi --id="${gpu}" --query-compute-apps=used_memory \
      --format=csv,noheader,nounits | awk '{sum += $1} END {print sum + 0}')"
    if ((used_mb > 512)); then
      busy+=("gpu${gpu}:${used_mb}MiB")
    fi
  done
  if ((${#busy[@]} == 0)); then
    break
  fi
  echo "[wait] evaluation GPUs still busy: ${busy[*]}"
  sleep "${POLL_SECONDS}"
done

echo "[wait] physical GPUs 0-3 are available"
current_stage="launch_full_vqa"
set +e
launcher_output="$("${EXP_DIR}/run_4gpu.sh" 2>&1)"
launcher_status=$?
set -e
printf '%s\n' "${launcher_output}" | tee "${launch_stdout}"
if ((launcher_status != 0)); then
  echo "[launch] run_4gpu.sh failed with exit code ${launcher_status}" >&2
  exit "${launcher_status}"
fi

mapfile -t run_root_lines < <(
  printf '%s\n' "${launcher_output}" | sed -n 's/^run_root=//p'
)
if ((${#run_root_lines[@]} != 1)); then
  echo "[launch] expected exactly one run_root line, found ${#run_root_lines[@]}" >&2
  exit 1
fi
evaluation_run_root="${run_root_lines[0]}"
if [[ "${evaluation_run_root}" != /* ||
      ! -f "${evaluation_run_root}/run_manifest.json" ]]; then
  echo "[launch] invalid prepared evaluation run root: ${evaluation_run_root}" >&2
  exit 1
fi

printf '%s\n' "${evaluation_run_root}" >"${evaluation_run_root_record}"
printf 'training_run_root=%s\nevaluation_run_root=%s\nlaunched_at=%s\n' \
  "${TRAINING_RUN_ROOT}" "${evaluation_run_root}" "$(date --iso-8601=seconds)" \
  >"${HANDOFF_ROOT}/launch_metadata.txt"
touch "${done_marker}"
current_stage="complete"
echo "[launch] evaluation_run_root=${evaluation_run_root}"
