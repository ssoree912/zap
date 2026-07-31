#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/workspace/nips"
ZAP_ROOT="${WORKSPACE}/zap"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
DATA_ROOT="${WORKSPACE}/data/train"
MODEL="${WORKSPACE}/models/llava-onevision-qwen2-7b-ov"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
TEACHER_ROOT="${DATA_ROOT}/teacher/zap_onevision_answer_n600_seed0"
STUDENT_DIR="${ZAP_ROOT}/artifacts/original_onevision_teacher/student_onevision_answer_n1800_e15_seed0"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="${LOG_DIR}/pipeline_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}" "${TEACHER_ROOT}" "${STUDENT_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[config] teacher signal: generated answer decode attention only"
echo "[config] correctness filtering: disabled"
echo "[config] samples: textvqa=600 gqa=600 scienceqa=600 seed=0"
echo "[config] student: epochs=15 lr=1e-4"
echo "[config] teacher_root=${TEACHER_ROOT}"
echo "[config] student_dir=${STUDENT_DIR}"

extract_dataset() {
    local dataset="$1"
    local physical_gpu="$2"
    local log_file="${LOG_DIR}/extract_${dataset}_${RUN_STAMP}.log"

    echo "[extract-start] dataset=${dataset} physical_gpu=${physical_gpu} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" -m qvik.teacher.extract_llava_onevision \
        --model "${MODEL}" \
        --dataset "${dataset}" \
        --n-samples 600 \
        --max-new-tokens 64 \
        --seed 0 \
        --device cuda:0 \
        --output-root "${TEACHER_ROOT}" \
        --problems-json "${DATA_ROOT}/scienceqa/problems.json" \
        --images-root "${DATA_ROOT}/scienceqa/images" \
        --split train \
        --gqa-questions-json "${DATA_ROOT}/gqa/val_balanced_questions.json" \
        --gqa-images-root "${DATA_ROOT}/gqa/images" \
        --textvqa-json "${DATA_ROOT}/textvqa/train/data.json" \
        --textvqa-data-root "${DATA_ROOT}" \
        2>&1 | tee "${log_file}"
}

extract_dataset textvqa 0 &
pid_textvqa=$!
extract_dataset gqa 1 &
pid_gqa=$!
extract_dataset scienceqa 2 &
pid_scienceqa=$!

set +e
wait "${pid_textvqa}"
status_textvqa=$?
wait "${pid_gqa}"
status_gqa=$?
wait "${pid_scienceqa}"
status_scienceqa=$?
set -e

echo "[extract-status] textvqa=${status_textvqa} gqa=${status_gqa} scienceqa=${status_scienceqa}"
if (( status_textvqa != 0 || status_gqa != 0 || status_scienceqa != 0 )); then
    echo "[error] one or more teacher extraction jobs failed"
    exit 1
fi

TEACHER_ROOT="${TEACHER_ROOT}" PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import torch

root = Path(os.environ["TEACHER_ROOT"])
datasets = ("textvqa", "gqa", "scienceqa")
for dataset in datasets:
    files = sorted((root / dataset).glob("*.pt"))
    if len(files) != 600:
        raise RuntimeError(f"{dataset}: expected exactly 600 shards, found {len(files)}")
    for path in files:
        record = torch.load(path, weights_only=False, map_location="cpu")
        teacher = record["teacher_norm"].float()
        if teacher.ndim != 2 or not torch.isfinite(teacher).all():
            raise RuntimeError(f"{path}: invalid teacher_norm")
        if (teacher.sum(dim=-1) <= 0).any():
            raise RuntimeError(f"{path}: zero-sum teacher row")
        if "teacher_question_norm" in record or "teacher_question_raw" in record:
            raise RuntimeError(f"{path}: question teacher signal was unexpectedly stored")
    print(f"[verify] {dataset}: 600 finite answer-only teacher shards")
print("[verify] total: 1800 teacher shards")
PY

TRAIN_LOG="${LOG_DIR}/train_${RUN_STAMP}.log"
echo "[train-start] physical_gpu=3 log=${TRAIN_LOG}"
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" \
"${PYTHON}" -m qvik.train.llava_onevision \
    --teacher-root "${TEACHER_ROOT}" \
    --datasets textvqa gqa scienceqa \
    --per-ds-limit 600 \
    --llava-path "${MODEL}" \
    --epochs 15 \
    --lr 1e-4 \
    --seed 0 \
    --device cuda:0 \
    --output-dir "${STUDENT_DIR}" \
    2>&1 | tee "${TRAIN_LOG}"

echo "[complete] answer-only OneVision teacher extraction and student training finished"
