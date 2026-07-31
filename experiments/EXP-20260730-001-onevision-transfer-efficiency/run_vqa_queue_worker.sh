#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

GPU="${1:?usage: run_vqa_queue_worker.sh GPU [WAIT_PID]}"
WAIT_PID="${2:-0}"

WORKSPACE="/workspace/nips"
QVIK_ROOT="${WORKSPACE}/Q-ViK"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MODEL="${WORKSPACE}/models/llava-onevision-qwen2-7b-ov"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-001-onevision-transfer-efficiency"
STATE_DIR="${EXP_DIR}/vqa_queue_state"
PENDING="${STATE_DIR}/pending.tsv"
QUEUE_LOCK="${STATE_DIR}/queue.lock"
LOG_DIR="${EXP_DIR}/logs/vqa_scratch"

mkdir -p "${STATE_DIR}" "${LOG_DIR}"

if (( WAIT_PID > 0 )); then
    echo "[wait] gpu=${GPU} waiting for existing worker pid=${WAIT_PID}"
    while [[ -r "/proc/${WAIT_PID}/cmdline" ]] && \
          tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline" | grep -q "run_eval_worker.sh"; do
        sleep 30
    done
fi

pop_job() {
    local line=""
    exec 9>"${QUEUE_LOCK}"
    flock 9
    if [[ -s "${PENDING}" ]]; then
        line="$(head -n 1 "${PENDING}")"
        sed -i '1d' "${PENDING}"
    fi
    flock -u 9
    exec 9>&-
    printf '%s' "${line}"
}

while true; do
    JOB="$(pop_job)"
    if [[ -z "${JOB}" ]]; then
        echo "[complete] gpu=${GPU} queue empty"
        break
    fi

    IFS='|' read -r TASK TOTAL LABEL <<< "${JOB}"
    CKPT_ROOT="${EXP_DIR}/checkpoints/scratch_n${TOTAL}_e15"
    case "${LABEL}" in
        epoch001) STUDENT="${CKPT_ROOT}/epoch_001" ;;
        epoch003) STUDENT="${CKPT_ROOT}/epoch_003" ;;
        epoch005) STUDENT="${CKPT_ROOT}/epoch_005" ;;
        best) STUDENT="${CKPT_ROOT}" ;;
        *)
            echo "[fail] unknown checkpoint label: ${LABEL}" >&2
            exit 2
            ;;
    esac

    if [[ ! -f "${STUDENT}/pytorch_model.bin" ]]; then
        echo "[fail] missing checkpoint: ${STUDENT}" >&2
        exit 1
    fi

    CASE_TAG="scratch_n${TOTAL}_${LABEL}"
    OUT_ROOT="${EXP_DIR}/outputs/vqa_scratch/${TASK}/${CASE_TAG}"
    LOG_FILE="${LOG_DIR}/eval_${TASK}_${CASE_TAG}_gpu${GPU}.log"

    if find "${OUT_ROOT}" -name '*_results.json' -print -quit 2>/dev/null | grep -q .; then
        echo "[skip] gpu=${GPU} task=${TASK} case=${CASE_TAG} result exists"
        continue
    fi

    echo "[start] gpu=${GPU} task=${TASK} case=${CASE_TAG} image_keep_ratio=0.1"
    cd "${QVIK_ROOT}"
    if CUDA_VISIBLE_DEVICES="${GPU}" \
       QVIK_DATA_ROOT="${WORKSPACE}/data/eval" \
       PYTHONPATH="${QVIK_ROOT}:${PYTHONPATH:-}" \
       HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false \
       PYTHONUNBUFFERED=1 \
       "${PYTHON}" qvik/eval/run_lmms_eval.py \
           --model lmms_onevision_student \
           --model_args "pretrained=${MODEL},student_path=${STUDENT},keep_ratio=0.1,device=cuda:0,attn_implementation=sdpa,conv_template=qwen_1_5" \
           --tasks "${TASK}" \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix "${TASK}_${CASE_TAG}" \
           --output_path "${OUT_ROOT}" \
           --seed 0 \
           2>&1 | tee "${LOG_FILE}"; then
        echo "[done] gpu=${GPU} task=${TASK} case=${CASE_TAG}"
    else
        echo "${JOB}" >> "${STATE_DIR}/failed.tsv"
        echo "[fail] gpu=${GPU} task=${TASK} case=${CASE_TAG}; continuing" >&2
    fi
done
