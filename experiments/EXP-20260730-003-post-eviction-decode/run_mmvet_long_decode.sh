#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE="/workspace/nips"
EXP_DIR="${WORKSPACE}/zap/experiments/EXP-20260730-003-post-eviction-decode"
PYTHON="${WORKSPACE}/.conda/envs/qvik/bin/python"
MANIFEST="${EXP_DIR}/mmvet_50_seed42/manifest.json"
mkdir -p "${EXP_DIR}/results_mmvet_long/llava15" \
    "${EXP_DIR}/results_mmvet_long/onevision" "${EXP_DIR}/logs"

run_llava15() {
    CUDA_VISIBLE_DEVICES=2 \
    PYTHONPATH="${WORKSPACE}/Q-ViK:${WORKSPACE}/zap:${PYTHONPATH:-}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" "${EXP_DIR}/benchmark_post_eviction_decode.py" \
        --backbone llava15 \
        --device cuda:0 \
        --manifest "${MANIFEST}" \
        --sample-limit 50 \
        --decode-steps 256 \
        --warmup-samples 2 \
        --llava15-total-keep-ratio 0.2 \
        --output-dir "${EXP_DIR}/results_mmvet_long/llava15"
}

run_onevision() {
    CUDA_VISIBLE_DEVICES=3 \
    PYTHONPATH="${WORKSPACE}/Q-ViK:${WORKSPACE}/zap:${PYTHONPATH:-}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" "${EXP_DIR}/benchmark_post_eviction_decode.py" \
        --backbone onevision \
        --device cuda:0 \
        --manifest "${MANIFEST}" \
        --sample-limit 50 \
        --decode-steps 256 \
        --warmup-samples 2 \
        --onevision-image-keep-ratio 0.1 \
        --output-dir "${EXP_DIR}/results_mmvet_long/onevision"
}

run_llava15 > "${EXP_DIR}/logs/mmvet_long_llava15_gpu2.log" 2>&1 &
llava15_pid=$!
echo "[launch] MM-Vet long decode LLaVA-1.5 GPU 2 pid=${llava15_pid}"

while [[ ! -f "${EXP_DIR}/results/onevision/summary.json" ]]; do
    sleep 15
done
run_onevision > "${EXP_DIR}/logs/mmvet_long_onevision_gpu3.log" 2>&1 &
onevision_pid=$!
echo "[launch] MM-Vet long decode OneVision GPU 3 pid=${onevision_pid}"

status=0
wait "${llava15_pid}" || status=1
wait "${onevision_pid}" || status=1
if (( status != 0 )); then
    echo "[error] one or more MM-Vet long-decode benchmarks failed" >&2
    exit 1
fi
echo "[complete] MM-Vet 256-step post-eviction decode benchmarks"
