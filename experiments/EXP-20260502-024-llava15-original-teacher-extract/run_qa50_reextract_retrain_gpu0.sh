#!/usr/bin/env bash
set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract"
PYTHON="/opt/conda/envs/qvik/bin/python"
MODEL="/workspace/nips/models/llava-v1.5-7b"
SAMPLE_MANIFEST="/workspace/nips/data/train/teacher/llava15_qa50_all"
TEACHER_ROOT="${ZAP_ROOT}/artifacts/original_llava_teacher/qa50_llava15_900_seed0"
STUDENT_DIR="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_qa50_900_e15_gpu0"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qa50_reextract_retrain_gpu0_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}" "${TEACHER_ROOT}" "${STUDENT_DIR}"
exec >>"${LOG_FILE}" 2>&1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export QVIK_ROOT=/workspace/nips/Q-ViK
export ZAP_REPO_ROOT="${ZAP_ROOT}"

cd "${ZAP_ROOT}"

echo "[run] log=${LOG_FILE}"
echo "[run] teacher=${TEACHER_ROOT}"
echo "[run] student=${STUDENT_DIR}"

"${PYTHON}" "${EXP_DIR}/collect_original_llava15_teacher.py" \
  --model-path "${MODEL}" \
  --model-name llava-v1.5-7b \
  --conv-template vicuna_v1 \
  --device cuda:0 \
  --device-map cuda:0 \
  --datasets gqa textvqa scienceqa \
  --n-samples 300 \
  --max-new-tokens 64 \
  --seed 0 \
  --question-weight 0.5 \
  --sample-manifest-root "${SAMPLE_MANIFEST}" \
  --output-root "${TEACHER_ROOT}"

"${PYTHON}" - "${TEACHER_ROOT}" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
for dataset in ("gqa", "textvqa", "scienceqa"):
    paths = sorted((root / dataset).glob("*.pt"))
    if len(paths) != 300:
        raise RuntimeError(f"{dataset}: expected 300 teacher files, found {len(paths)}")
    for path in paths:
        record = torch.load(path, weights_only=False, map_location="cpu")
        teacher = record["teacher_norm"].float()
        if tuple(teacher.shape) != (32, 576):
            raise RuntimeError(f"{path}: unexpected teacher shape {tuple(teacher.shape)}")
        if not torch.allclose(
            teacher.sum(dim=-1),
            torch.ones(32),
            atol=1e-3,
            rtol=0,
        ):
            raise RuntimeError(f"{path}: teacher distribution is not normalized")
        if record.get("teacher_question_weight") != 0.5:
            raise RuntimeError(f"{path}: incorrect question weight")
        if record.get("teacher_answer_weight") != 0.5:
            raise RuntimeError(f"{path}: incorrect answer weight")
print("[verify] 900 mixed teacher records passed")
PY

"${PYTHON}" "${EXP_DIR}/train_original_llava15_student.py" \
  --teacher-root "${TEACHER_ROOT}" \
  --datasets gqa textvqa scienceqa \
  --llava-path "${MODEL}" \
  --model-name llava-v1.5-7b \
  --vision-tower-path "" \
  --device cuda:0 \
  --device-map cuda:0 \
  --epochs 15 \
  --lr 1e-4 \
  --n-per-dataset 300 \
  --seed 0 \
  --output-dir "${STUDENT_DIR}"

echo "[complete] teacher extraction and scorer training finished"
