#!/usr/bin/env bash
# EXP-20260428-002: OneVision student training on OCR/chart/doc + general data
#
# Datasets (200 samples each, 1200 total):
#   OCR/doc: chartqa, docvqa, infovqa
#   general: gqa, scienceqa, st_vqa
#
# Comparable to student_onevision_A_lr1e4_20ep (scienceqa+gqa+st_vqa, 500/ds)
# but domain-shifted toward OCR/document tasks.
set -euo pipefail

ROOT=/workspace/zap
LOGDIR="${ROOT}/experiments/EXP-20260428-002-onevision-ocrdoc-train/outputs"
PYTHON=/opt/conda/envs/kv/bin/python
CKPT="${ROOT}/ckpts/student_onevision_ocrdoc_lr1e4_20ep"

mkdir -p "${LOGDIR}" "${CKPT}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${ROOT}/train_visual_utility_student_onevision.py" \
    --scope A \
    --datasets chartqa docvqa infovqa gqa scienceqa st_vqa \
    --per-ds-limit 200 \
    --teacher-root "${ROOT}/data/teacher_v2_onevision" \
    --llava-path "${ROOT}/ckpts/llava-onevision-qwen2-7b-ov-hf" \
    --epochs 20 \
    --lr 1e-4 \
    --output-dir "${CKPT}" \
    --device cuda:0 \
    --log-every 50 \
    2>&1 | tee "${LOGDIR}/train.log"

echo "=== DONE: ${CKPT} ==="
