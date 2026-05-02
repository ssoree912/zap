#!/usr/bin/env bash
# EXP-20260501-019 : student_llava15_no_instruct_1800_lr1e4 on mm-vet
# ROUGE  : vs our_full (full-cache reference)
# PPL    : GT-based teacher-forcing
# keep   : 0.1 → 0.2 → 0.5
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

PYTHON="/opt/conda/envs/kv/bin/python3"
REPO="/workspace/zap"
STUDENT="${REPO}/ckpts/student_llava15_no_instruct_1800_lr1e4"
DATA_PATH="/workspace/data/mm-vet/mm-vet.json"
IMAGE_PATH="/workspace/data/mm-vet"
ROUGE_REF="${REPO}/data/rouge_ref/our_full_mm-vet.json"
N=218

EXP_DIR="${REPO}/experiments/EXP-20260501-019-student-llava15-noinst-mmvet"
OUT_DIR="${EXP_DIR}/outputs"
LOG_DIR="${EXP_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

run_rouge() {
  local keep="$1"
  local outdir="${OUT_DIR}/keep${keep/./}/rouge_vsfull"
  mkdir -p "${outdir}"
  if [ -f "${outdir}/result.json" ]; then
    echo "[skip] ROUGE keep=${keep} already done"; return
  fi
  echo "=== ROUGE keep=${keep} START $(date -Is) ==="
  "${PYTHON}" "${REPO}/scripts/eval_rouge.py" \
    --method visual_utility_student \
    --student-model-name "${STUDENT}" \
    --total-keep-ratio "${keep}" \
    --data-path "${ROUGE_REF}" \
    --image-path "${IMAGE_PATH}" \
    --eval-samples "${N}" \
    --output-dir "${outdir}" \
    2>&1 | tee "${LOG_DIR}/rouge_keep${keep/./}.log"
  echo "=== ROUGE keep=${keep} DONE $(date -Is) ==="
}

run_ppl() {
  local keep="$1"
  local outdir="${OUT_DIR}/keep${keep/./}/ppl_gt"
  mkdir -p "${outdir}"
  if [ -f "${outdir}/result.json" ]; then
    echo "[skip] PPL keep=${keep} already done"; return
  fi
  echo "=== PPL keep=${keep} START $(date -Is) ==="
  "${PYTHON}" "${REPO}/scripts/eval_ppl.py" \
    --method visual_utility_student \
    --student-model-name "${STUDENT}" \
    --total-keep-ratio "${keep}" \
    --data-path "${DATA_PATH}" \
    --image-path "${IMAGE_PATH}" \
    --eval-samples "${N}" \
    --output-dir "${outdir}" \
    2>&1 | tee "${LOG_DIR}/ppl_keep${keep/./}.log"
  echo "=== PPL keep=${keep} DONE $(date -Is) ==="
}

for K in 0.1 0.2 0.5; do
  run_rouge "${K}"
  run_ppl   "${K}"
done

echo ""
echo "=== ALL DONE ==="
echo "Results:"
for K in 0.1 0.2 0.5; do
  R=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/keep${K/./}/rouge_vsfull/result.json')); print(round(d['rouge_l'],4))" 2>/dev/null || echo "N/A")
  P=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/keep${K/./}/ppl_gt/result.json')); print(round(d['ppl'],3))" 2>/dev/null || echo "N/A")
  echo "  keep=${K}: ROUGE-L=${R}  PPL=${P}"
done
