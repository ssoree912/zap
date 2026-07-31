#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE_ROOT="/workspace/nips"
REPO_ROOT="${WORKSPACE_ROOT}/zap"
EXP_DIR="${REPO_ROOT}/experiments/EXP-20260503-026-onevision-original-teacher-train"
MODEL_PATH="${WORKSPACE_ROOT}/models/llava-onevision-qwen2-7b-ov"
TEACHER_ROOT="${REPO_ROOT}/artifacts/original_onevision_teacher/future_answer_agnostic_1800_seed42"
STUDENT_DIR="${REPO_ROOT}/artifacts/original_onevision_teacher/student_onevision_zap_future_answer_agnostic_1800_lr1e4_15ep_2gpu"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"

mkdir -p "${TEACHER_ROOT}" "${STUDENT_DIR}" "${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}:${WORKSPACE_ROOT}/VFlowOpt/src/LLaVA-OneVision:${PYTHONPATH:-}"
PYTHON_BIN="/opt/conda/envs/vflowopt/bin/python"
TORCHRUN_BIN="/opt/conda/envs/vflowopt/bin/torchrun"
COLLECTOR="${EXP_DIR}/collect_original_onevision_teacher.py"
TRAINER="${EXP_DIR}/train_original_onevision_student.py"

echo "=== answer-agnostic OneVision ZAP pipeline START $(date -Is) ==="
echo "teacher_root=${TEACHER_ROOT}"
echo "student_dir=${STUDENT_DIR}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${COLLECTOR}" \
  --model-path "${MODEL_PATH}" \
  --model-name llava_qwen \
  --conv-template qwen_1_5 \
  --device cuda:0 \
  --device-map auto \
  --datasets scienceqa gqa \
  --n-samples 600 \
  --max-candidates 720 \
  --max-new-tokens 64 \
  --seed 42 \
  --output-root "${TEACHER_ROOT}" \
  > "${LOG_DIR}/answer_agnostic_extract_gpu0_${RUN_STAMP}.log" 2>&1 &
PID_GPU0=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${COLLECTOR}" \
  --model-path "${MODEL_PATH}" \
  --model-name llava_qwen \
  --conv-template qwen_1_5 \
  --device cuda:0 \
  --device-map auto \
  --datasets textvqa \
  --n-samples 600 \
  --max-candidates 720 \
  --max-new-tokens 64 \
  --seed 42 \
  --output-root "${TEACHER_ROOT}" \
  > "${LOG_DIR}/answer_agnostic_extract_gpu1_${RUN_STAMP}.log" 2>&1 &
PID_GPU1=$!

echo "teacher extraction pids: gpu0=${PID_GPU0}, gpu1=${PID_GPU1}"
wait "${PID_GPU0}"
wait "${PID_GPU1}"

"${PYTHON_BIN}" - "${TEACHER_ROOT}" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
for dataset in ("scienceqa", "gqa", "textvqa"):
    paths = sorted((root / dataset).glob("*.pt"))
    if len(paths) != 600:
        raise SystemExit(f"{dataset}: expected 600 teacher files, found {len(paths)}")
    for path in paths:
        rec = torch.load(path, weights_only=False, map_location="cpu")
        if rec.get("require_correct") is not False:
            raise SystemExit(f"{path}: require_correct must be False")
        if rec.get("teacher_signal") != "future_attention_generated_answer_tokens":
            raise SystemExit(f"{path}: unexpected teacher_signal={rec.get('teacher_signal')!r}")
        if tuple(rec["teacher_norm"].shape) != tuple(rec["teacher_raw"].shape):
            raise SystemExit(f"{path}: teacher shape mismatch")
    print(f"[verified] {dataset}: {len(paths)} answer-agnostic future-attention records")
print("[verified] total teacher records: 1800")
PY

echo "=== 2-GPU student training START $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0,1 "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=2 \
  "${TRAINER}" \
  --teacher-root "${TEACHER_ROOT}" \
  --datasets textvqa gqa scienceqa \
  --per-ds-limit 600 \
  --model-path "${MODEL_PATH}" \
  --model-name llava_qwen \
  --device-map auto \
  --epochs 15 \
  --lr 1e-4 \
  --optimizer adamw8bit \
  --output-dir "${STUDENT_DIR}" \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/answer_agnostic_train_2gpu_${RUN_STAMP}.log"

test -s "${STUDENT_DIR}/pytorch_model.bin"
test -s "${STUDENT_DIR}/config.json"
test -s "${STUDENT_DIR}/last_checkpoint.pt"
echo "=== answer-agnostic OneVision ZAP pipeline DONE $(date -Is) ==="
