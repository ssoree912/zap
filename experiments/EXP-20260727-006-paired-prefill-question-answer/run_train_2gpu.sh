#!/usr/bin/env bash
set -euo pipefail

readonly ZAP_ROOT="/workspace/nips/zap"
readonly EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260727-006-paired-prefill-question-answer"
readonly TRAINER="${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract/train_original_llava15_student.py"
readonly PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
readonly PA_ROOT="/workspace/nips/data/train/teacher/zap_llava15_prefill_answer50_paired_n600_seed0"
readonly QA_ROOT="/workspace/nips/data/train/teacher/zap_llava15_question_answer50_paired_n600_seed0"
readonly PA_STUDENT="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_prefill_answer50_paired_1800_e15_seed0"
readonly QA_STUDENT="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_question_answer50_paired_1800_e15_seed0"
readonly TEACHER_RUN_ROOT="${TEACHER_RUN_ROOT:?Set TEACHER_RUN_ROOT to the completed extraction run}"
readonly RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly RUN_ROOT="${ZAP_ROOT}/artifacts/paired_prefill_question_answer/training_runs/${RUN_ID}"
readonly SESSION_PREFIX="paired_train_${RUN_ID}"

if [[ ! -e "${TEACHER_RUN_ROOT}/state/verification.done" ||
      ! -s "${TEACHER_RUN_ROOT}/teacher_verification.json" ]]; then
  echo "Teacher verification has not completed: ${TEACHER_RUN_ROOT}" >&2
  exit 1
fi
for path in "${PA_STUDENT}" "${QA_STUDENT}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing checkpoint directory: ${path}" >&2
    exit 1
  fi
done
for gpu in 0 1; do
  used_mb="$(nvidia-smi --id="${gpu}" --query-compute-apps=used_memory \
    --format=csv,noheader,nounits | awk '{sum += $1} END {print sum + 0}')"
  if ((used_mb > 512)); then
    echo "GPU ${gpu} is already using ${used_mb} MiB; refusing to launch." >&2
    exit 1
  fi
done
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing training run root: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/state"

common_args="--datasets gqa textvqa scienceqa \
--llava-path /workspace/nips/models/llava-v1.5-7b \
--model-name llava-v1.5-7b --epochs 15 --seed 0 \
--n-per-dataset 600 --val-ratio 0.1 --student-variant full \
--conv-dim 256 --proj-dim 256 --mlp-dim 512 \
--num-conv-blocks 2 --kernel-size 7 --grid-h 24 --grid-w 24 \
--lr 0.0001 --weight-decay 0.0 --lambda-rank 0.1 \
--rank-margin 0.05 --rank-top-ratio 0.2 --rank-bottom-ratio 0.4 \
--max-grad-norm 1.0 --log-every 25"

screen -dmS "${SESSION_PREFIX}_pa_gpu0" bash -lc \
  "CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  TOKENIZERS_PARALLELISM=false QVIK_ROOT=/workspace/nips/Q-ViK \
  ZAP_REPO_ROOT=${ZAP_ROOT} ${PYTHON} ${TRAINER} \
  --teacher-root ${PA_ROOT} --device cuda:0 --device-map cuda:0 \
  --output-dir ${PA_STUDENT} ${common_args} \
  >${RUN_ROOT}/logs/prefill_answer_train.log 2>&1 && \
  touch ${RUN_ROOT}/state/prefill_answer.done || \
  { touch ${RUN_ROOT}/state/prefill_answer.failed; exit 1; }"

screen -dmS "${SESSION_PREFIX}_qa_gpu1" bash -lc \
  "CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  TOKENIZERS_PARALLELISM=false QVIK_ROOT=/workspace/nips/Q-ViK \
  ZAP_REPO_ROOT=${ZAP_ROOT} ${PYTHON} ${TRAINER} \
  --teacher-root ${QA_ROOT} --device cuda:0 --device-map cuda:0 \
  --output-dir ${QA_STUDENT} ${common_args} \
  >${RUN_ROOT}/logs/question_answer_train.log 2>&1 && \
  touch ${RUN_ROOT}/state/question_answer.done || \
  { touch ${RUN_ROOT}/state/question_answer.failed; exit 1; }"

screen -dmS "${SESSION_PREFIX}_verify" bash -lc \
  "while [[ ! -e ${RUN_ROOT}/state/prefill_answer.done || \
  ! -e ${RUN_ROOT}/state/question_answer.done ]]; do \
    if compgen -G '${RUN_ROOT}/state/*.failed' >/dev/null; then \
      touch ${RUN_ROOT}/state/verification.failed; exit 1; fi; \
    sleep 20; \
  done; \
  ${PYTHON} ${EXP_DIR}/verify_paired_training.py \
  --pa-dir ${PA_STUDENT} --qa-dir ${QA_STUDENT} \
  --summary-path ${RUN_ROOT}/training_verification.json \
  >${RUN_ROOT}/logs/verification.log 2>&1 && \
  touch ${RUN_ROOT}/state/verification.done || \
  touch ${RUN_ROOT}/state/verification.failed"

echo "run_root=${RUN_ROOT}"
echo "students=${PA_STUDENT},${QA_STUDENT}"
echo "sessions=${SESSION_PREFIX}_{pa_gpu0,qa_gpu1,verify}"
