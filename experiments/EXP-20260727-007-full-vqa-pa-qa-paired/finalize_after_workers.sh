#!/usr/bin/env bash
set -euo pipefail

readonly ZAP_ROOT="/workspace/nips/zap"
readonly EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260727-007-full-vqa-pa-qa-paired"
readonly PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must point to a prepared run directory" >&2
  exit 2
fi

readonly final_log="${RUN_ROOT}/logs/finalizer.log"
exec >>"${final_log}" 2>&1

expected=(
  prefill_answer_50_50__gqa_local
  question_answer_50_50__gqa_local
  prefill_answer_50_50__textvqa_local
  question_answer_50_50__textvqa_local
  prefill_answer_50_50__docvqa_local
  question_answer_50_50__docvqa_local
  prefill_answer_50_50__chartqa_local
  question_answer_50_50__chartqa_local
)

echo "[finalizer] waiting for ${#expected[@]} jobs"
while true; do
  failure="$(find "${RUN_ROOT}/state" -maxdepth 1 -type f -name '*.failed' -print -quit)"
  if [[ -n "${failure}" ]]; then
    echo "[finalizer] aborting because a worker/job failed: ${failure}" >&2
    sed -n '1,120p' "${failure}" >&2
    exit 1
  fi
  done_count=0
  for job in "${expected[@]}"; do
    if [[ -f "${RUN_ROOT}/state/${job}.done" ]]; then
      done_count=$((done_count + 1))
    fi
  done
  echo "[finalizer] completed=${done_count}/${#expected[@]}"
  if ((done_count == ${#expected[@]})); then
    break
  fi
  sleep 20
done

"${PYTHON}" "${EXP_DIR}/validate_paired_outputs.py" --run-root "${RUN_ROOT}"
echo "[finalizer] paired validation complete"
