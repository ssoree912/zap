#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

GPU="${1:?usage: run_train_eval_condition.sh GPU TAG TEACHER_ROOT N_PER_DATASET EPOCHS [best|final]}"
TAG="${2:?usage: run_train_eval_condition.sh GPU TAG TEACHER_ROOT N_PER_DATASET EPOCHS [best|final]}"
TEACHER_ROOT="${3:?usage: run_train_eval_condition.sh GPU TAG TEACHER_ROOT N_PER_DATASET EPOCHS [best|final]}"
N_PER_DATASET="${4:?usage: run_train_eval_condition.sh GPU TAG TEACHER_ROOT N_PER_DATASET EPOCHS [best|final]}"
EPOCHS="${5:?usage: run_train_eval_condition.sh GPU TAG TEACHER_ROOT N_PER_DATASET EPOCHS [best|final]}"
CHECKPOINT_SELECTION="${6:-best}"

if [[ "${CHECKPOINT_SELECTION}" != "best" && "${CHECKPOINT_SELECTION}" != "final" ]]; then
    echo "[error] checkpoint selection must be 'best' or 'final': ${CHECKPOINT_SELECTION}" >&2
    exit 2
fi

WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MODEL="${WORKSPACE}/models/llava-v1.5-7b"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-002-llava15-adaptation-efficiency"
STUDENT="${EXP_DIR}/checkpoints/${TAG}"
OUT_ROOT="${EXP_DIR}/outputs/${TAG}"
LOG_DIR="${EXP_DIR}/logs"

mkdir -p "${STUDENT}" "${OUT_ROOT}" "${LOG_DIR}"

if [[ ! -f "${STUDENT}/pytorch_model.bin" ]]; then
    echo "[train] gpu=${GPU} tag=${TAG} n_per_dataset=${N_PER_DATASET} epochs=${EPOCHS}"
    cd "${QVIK_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    ZAP_REPO_ROOT="${WORKSPACE}/zap" \
    PYTHONPATH="${QVIK_ROOT}:${WORKSPACE}/zap:${PYTHONPATH:-}" \
    TOKENIZERS_PARALLELISM=false \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" -m qvik.train.llava15 \
        --teacher-root "${TEACHER_ROOT}" \
        --datasets textvqa scienceqa gqa \
        --llava-path "${MODEL}" \
        --model-name llava-v1.5-7b \
        --device cuda:0 \
        --device-map cuda:0 \
        --epochs "${EPOCHS}" \
        --lr 1e-4 \
        --n-per-dataset "${N_PER_DATASET}" \
        --seed 0 \
        --output-dir "${STUDENT}" \
        2>&1 | tee "${LOG_DIR}/train_${TAG}_gpu${GPU}.log"
else
    echo "[skip] training result exists: ${STUDENT}"
fi

if [[ "${CHECKPOINT_SELECTION}" == "final" ]]; then
    if [[ ! -f "${STUDENT}/last_checkpoint.pt" ]]; then
        echo "[error] final checkpoint missing: ${STUDENT}/last_checkpoint.pt" >&2
        exit 1
    fi
    "${PYTHON}" - "${STUDENT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

student_dir = Path(sys.argv[1])
checkpoint = torch.load(
    student_dir / "last_checkpoint.pt",
    map_location="cpu",
    weights_only=False,
)
torch.save(checkpoint["student"], student_dir / "pytorch_model.bin")
(student_dir / "checkpoint_selection.json").write_text(
    json.dumps(
        {
            "selection": "final",
            "epoch": int(checkpoint["epoch"]),
            "metrics": checkpoint["metrics"],
        },
        indent=2,
    )
)
print(
    f"[ckpt] exported final epoch={checkpoint['epoch']} "
    f"to {student_dir / 'pytorch_model.bin'}",
    flush=True,
)
PY
fi

for task in chartqa docvqa textvqa; do
    task_out="${OUT_ROOT}/${task}"
    task_log="${LOG_DIR}/eval_${TAG}_${task}_gpu${GPU}.log"
    if find "${task_out}" -name '*_results.json' -print -quit 2>/dev/null | grep -q .; then
        echo "[skip] result exists tag=${TAG} task=${task}"
        continue
    fi

    echo "[eval] gpu=${GPU} tag=${TAG} task=${task} total_keep_ratio=0.2"
    cd "${QVIK_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    QVIK_DATA_ROOT="${WORKSPACE}/data/eval" \
    PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" qvik/eval/run_lmms_eval.py \
        --model lmms_llava15_student \
        --model_args "pretrained=${MODEL},student_path=${STUDENT},keep_ratio=0.2,keep_ratio_basis=total,device=cuda:0,device_map=cuda:0,conv_template=vicuna_v1" \
        --tasks "${task}" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "${TAG}_${task}" \
        --output_path "${task_out}" \
        --seed 0 \
        2>&1 | tee "${task_log}"
done

echo "[complete] ${TAG}"
