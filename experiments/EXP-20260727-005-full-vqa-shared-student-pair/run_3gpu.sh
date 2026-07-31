#!/usr/bin/env bash
set -euo pipefail

readonly ZAP_ROOT="/workspace/nips/zap"
readonly EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260727-005-full-vqa-shared-student-pair"
readonly PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-${ZAP_ROOT}/artifacts/rebuttal_full_vqa_shared_student_pair_llava15_total0p2}"
readonly RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly RUN_ROOT="${OUTPUT_ROOT}/runs/${RUN_ID}"
readonly session_tag="vqa_pair_${RUN_ID}"

for gpu in 0 1 2; do
  used_mb="$(nvidia-smi --id="${gpu}" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{sum += $1} END {print sum + 0}')"
  if ((used_mb > 512)); then
    echo "Physical GPU ${gpu} is already using ${used_mb} MiB; refusing to launch." >&2
    exit 1
  fi
done

"${PYTHON}" "${EXP_DIR}/prepare_run.py" --run-root "${RUN_ROOT}"

screen -dmS "${session_tag}_gpu0" env RUN_ROOT="${RUN_ROOT}" \
  "${EXP_DIR}/run_worker.sh" 0 \
  last_prompt_plus_answer_steps:gqa_local \
  last_prompt_plus_answer_steps:docvqa_local

screen -dmS "${session_tag}_gpu1" env RUN_ROOT="${RUN_ROOT}" \
  "${EXP_DIR}/run_worker.sh" 1 \
  question_answer_50_50:gqa_local \
  question_answer_50_50:docvqa_local

screen -dmS "${session_tag}_gpu2" env RUN_ROOT="${RUN_ROOT}" \
  "${EXP_DIR}/run_worker.sh" 2 \
  last_prompt_plus_answer_steps:textvqa_local \
  question_answer_50_50:textvqa_local \
  last_prompt_plus_answer_steps:chartqa_local \
  question_answer_50_50:chartqa_local

screen -dmS "${session_tag}_final" env RUN_ROOT="${RUN_ROOT}" \
  "${EXP_DIR}/finalize_after_workers.sh"

echo "run_root=${RUN_ROOT}"
echo "sessions=${session_tag}_{gpu0,gpu1,gpu2,final}"
echo "Only physical GPUs 0, 1, and 2 were assigned."
