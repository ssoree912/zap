#!/usr/bin/env bash
set -euo pipefail

readonly ZAP_ROOT="/workspace/nips/zap"
readonly EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260727-006-paired-prefill-question-answer"
readonly PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
readonly SOURCE_ROOT="/workspace/nips/data/train/teacher/zap_llava15_qa50_n600_seed0"
readonly PA_ROOT="/workspace/nips/data/train/teacher/zap_llava15_prefill_answer50_paired_n600_seed0"
readonly QA_ROOT="/workspace/nips/data/train/teacher/zap_llava15_question_answer50_paired_n600_seed0"
readonly RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"

required=(
  prefill_gqa.done
  prefill_textvqa.done
  prefill_scienceqa_shard0.done
  prefill_scienceqa_shard1.done
)
while true; do
  if compgen -G "${RUN_ROOT}/state/*.failed" >/dev/null; then
    touch "${RUN_ROOT}/state/finalization.failed"
    exit 1
  fi
  complete=true
  for marker in "${required[@]}"; do
    if [[ ! -e "${RUN_ROOT}/state/${marker}" ]]; then
      complete=false
      break
    fi
  done
  if [[ "${complete}" == true ]]; then
    break
  fi
  sleep 10
done

"${PYTHON}" "${EXP_DIR}/recompose_saved_targets.py" \
  --pa-root "${PA_ROOT}" \
  --qa-root "${QA_ROOT}" \
  >"${RUN_ROOT}/logs/recomposition.log" 2>&1
touch "${RUN_ROOT}/state/recomposition.done"

"${PYTHON}" "${EXP_DIR}/verify_paired_teachers.py" \
  --source-root "${SOURCE_ROOT}" \
  --pa-root "${PA_ROOT}" \
  --qa-root "${QA_ROOT}" \
  --summary-path "${RUN_ROOT}/teacher_verification.json" \
  >"${RUN_ROOT}/logs/verification.log" 2>&1
touch "${RUN_ROOT}/state/verification.done"
