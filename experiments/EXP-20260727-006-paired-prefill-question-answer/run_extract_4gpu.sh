#!/usr/bin/env bash
set -euo pipefail

readonly ZAP_ROOT="/workspace/nips/zap"
readonly EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260727-006-paired-prefill-question-answer"
readonly PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
readonly SOURCE_ROOT="/workspace/nips/data/train/teacher/zap_llava15_qa50_n600_seed0"
readonly PA_ROOT="/workspace/nips/data/train/teacher/zap_llava15_prefill_answer50_paired_n600_seed0"
readonly QA_ROOT="/workspace/nips/data/train/teacher/zap_llava15_question_answer50_paired_n600_seed0"
readonly RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly RUN_ROOT="${ZAP_ROOT}/artifacts/paired_prefill_question_answer/runs/${RUN_ID}"
readonly SESSION_PREFIX="paired_teacher_${RUN_ID}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/state"

for gpu in 0 1 2 3; do
  used_mb="$(nvidia-smi --id="${gpu}" --query-compute-apps=used_memory \
    --format=csv,noheader,nounits | awk '{sum += $1} END {print sum + 0}')"
  if ((used_mb > 512)); then
    echo "GPU ${gpu} is already using ${used_mb} MiB; refusing to launch." >&2
    exit 1
  fi
done

"${PYTHON}" "${EXP_DIR}/derive_paired_teachers.py" \
  --mode question_answer \
  --datasets gqa textvqa scienceqa \
  --source-root "${SOURCE_ROOT}" \
  --qa-output-root "${QA_ROOT}" \
  >"${RUN_ROOT}/logs/question_answer_recompose.log" 2>&1
touch "${RUN_ROOT}/state/question_answer.done"

screen -dmS "${SESSION_PREFIX}_gpu0" bash -lc \
  "CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${EXP_DIR}/derive_paired_teachers.py \
  --mode prefill_answer --datasets gqa --device cuda:0 \
  --source-root ${SOURCE_ROOT} --pa-output-root ${PA_ROOT} \
  >${RUN_ROOT}/logs/prefill_gqa.log 2>&1 && \
  touch ${RUN_ROOT}/state/prefill_gqa.done || \
  touch ${RUN_ROOT}/state/prefill_gqa.failed"

screen -dmS "${SESSION_PREFIX}_gpu1" bash -lc \
  "CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${EXP_DIR}/derive_paired_teachers.py \
  --mode prefill_answer --datasets textvqa --device cuda:0 \
  --source-root ${SOURCE_ROOT} --pa-output-root ${PA_ROOT} \
  >${RUN_ROOT}/logs/prefill_textvqa.log 2>&1 && \
  touch ${RUN_ROOT}/state/prefill_textvqa.done || \
  touch ${RUN_ROOT}/state/prefill_textvqa.failed"

for shard in 0 1; do
  gpu=$((shard + 2))
  screen -dmS "${SESSION_PREFIX}_gpu${gpu}" bash -lc \
    "CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${EXP_DIR}/derive_paired_teachers.py \
    --mode prefill_answer --datasets scienceqa --device cuda:0 \
    --shard-index ${shard} --num-shards 2 \
    --source-root ${SOURCE_ROOT} --pa-output-root ${PA_ROOT} \
    >${RUN_ROOT}/logs/prefill_scienceqa_shard${shard}.log 2>&1 && \
    touch ${RUN_ROOT}/state/prefill_scienceqa_shard${shard}.done || \
    touch ${RUN_ROOT}/state/prefill_scienceqa_shard${shard}.failed"
done

screen -dmS "${SESSION_PREFIX}_verify" bash -lc \
  "while [[ ! -e ${RUN_ROOT}/state/prefill_gqa.done || \
  ! -e ${RUN_ROOT}/state/prefill_textvqa.done || \
  ! -e ${RUN_ROOT}/state/prefill_scienceqa_shard0.done || \
  ! -e ${RUN_ROOT}/state/prefill_scienceqa_shard1.done ]]; do \
    if compgen -G '${RUN_ROOT}/state/*.failed' >/dev/null; then exit 1; fi; \
    sleep 10; \
  done; \
  ${PYTHON} ${EXP_DIR}/verify_paired_teachers.py \
  --source-root ${SOURCE_ROOT} --pa-root ${PA_ROOT} --qa-root ${QA_ROOT} \
  --summary-path ${RUN_ROOT}/teacher_verification.json \
  >${RUN_ROOT}/logs/verification.log 2>&1 && \
  touch ${RUN_ROOT}/state/verification.done || \
  touch ${RUN_ROOT}/state/verification.failed"

echo "run_root=${RUN_ROOT}"
echo "sessions=${SESSION_PREFIX}_{gpu0,gpu1,gpu2,gpu3,verify}"
