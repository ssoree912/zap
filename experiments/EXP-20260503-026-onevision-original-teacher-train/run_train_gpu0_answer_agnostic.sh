#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE_ROOT="/workspace/nips"
REPO_ROOT="${WORKSPACE_ROOT}/zap"
EXP_DIR="${REPO_ROOT}/experiments/EXP-20260503-026-onevision-original-teacher-train"
TEACHER_ROOT="${REPO_ROOT}/artifacts/original_onevision_teacher/future_answer_agnostic_1800_seed42"
STUDENT_DIR="${REPO_ROOT}/artifacts/original_onevision_teacher/student_onevision_zap_future_answer_agnostic_1800_lr1e4_15ep_gpu0"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"

mkdir -p "${STUDENT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${WORKSPACE_ROOT}/VFlowOpt/src/LLaVA-OneVision:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/vflowopt/bin/python \
  "${EXP_DIR}/train_original_onevision_student.py" \
  --teacher-root "${TEACHER_ROOT}" \
  --datasets textvqa gqa scienceqa \
  --per-ds-limit 600 \
  --model-path "${WORKSPACE_ROOT}/models/llava-onevision-qwen2-7b-ov" \
  --model-name llava_qwen \
  --device cuda:0 \
  --device-map auto \
  --epochs 15 \
  --lr 1e-4 \
  --optimizer adamw8bit \
  --output-dir "${STUDENT_DIR}" \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/answer_agnostic_train_gpu0_${RUN_STAMP}.log"
