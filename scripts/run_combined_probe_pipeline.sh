#!/usr/bin/env bash
# EXP-20260415-002: Combined probe training (ScienceQA + TextVQA + NLVR2)
#
# Stage 1: collect XY shards for TextVQA and NLVR2
#   (ScienceQA shards assumed to already exist at SQ_SHARD_DIR)
# Stage 2: train probe from all combined shards
#
# Usage:
#   TEACHER_TYPE=att_only_postvision \
#   bash scripts/run_combined_probe_pipeline.sh
#
# Override any variable via env:
#   DEVICE=cuda:1 LIMIT=500 bash scripts/run_combined_probe_pipeline.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/kv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

DATA_ROOT="${DATA_ROOT:-/workspace/hd/data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/hd/artifacts/sq_teacher}"
MODEL_NAME="${MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"
TEACHER_TYPE="${TEACHER_TYPE:-att_only_postvision}"

# Existing ScienceQA shards (train split only)
SQ_SHARD_DIR="${SQ_SHARD_DIR:-/workspace/hd/artifacts/sq_teacher/scienceqa_xy/${TEACHER_TYPE}/train}"

# New XY shard output dirs
TEXTVQA_SHARD_DIR="${TEXTVQA_SHARD_DIR:-${ARTIFACT_ROOT}/textvqa_xy/${TEACHER_TYPE}/train}"
NLVR2_SHARD_DIR="${NLVR2_SHARD_DIR:-${ARTIFACT_ROOT}/nlvr2_xy/${TEACHER_TYPE}/train}"

# Combined probe output
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${ARTIFACT_ROOT}/image_probe_combined_v1}"

SHARD_SIZE="${SHARD_SIZE:-50000}"
TRAIN_METHODS="${TRAIN_METHODS:-mlp}"
MLP_MAX_EPOCHS="${MLP_MAX_EPOCHS:-10}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

# ── Stage 1a: TextVQA XY collection ──────────────────────────────────────────

echo "============================================================"
echo "[Stage 1a] Collecting TextVQA XY shards (${TEACHER_TYPE})"
echo "  data: ${DATA_ROOT}/textvqa/train"
echo "  out:  ${TEXTVQA_SHARD_DIR}"
echo "============================================================"

"${PYTHON_BIN}" /workspace/zap/collect_vqa_teacher_xy.py \
  --dataset textvqa \
  --data_dir "${DATA_ROOT}/textvqa/train" \
  --out_dir "${TEXTVQA_SHARD_DIR}" \
  --teacher_type "${TEACHER_TYPE}" \
  --implementation_model_name "${MODEL_NAME}" \
  --sample_image_tokens all \
  --shard_size "${SHARD_SIZE}" \
  --target_transform log \
  --torch_dtype float16 \
  --storage_dtype float16 \
  --target_dtype float32 \
  --teacher_head_reduction mean \
  --device "${DEVICE}" \
  --overwrite \
  --continue_on_error \
  "${LIMIT_ARG[@]}"

echo "[Stage 1a] TextVQA XY collection done."

# ── Stage 1b: NLVR2 XY collection ────────────────────────────────────────────

echo "============================================================"
echo "[Stage 1b] Collecting NLVR2 XY shards (${TEACHER_TYPE})"
echo "  data: ${DATA_ROOT}/nlvr2/train"
echo "  out:  ${NLVR2_SHARD_DIR}"
echo "============================================================"

"${PYTHON_BIN}" /workspace/zap/collect_vqa_teacher_xy.py \
  --dataset nlvr2 \
  --data_dir "${DATA_ROOT}/nlvr2/train" \
  --out_dir "${NLVR2_SHARD_DIR}" \
  --teacher_type "${TEACHER_TYPE}" \
  --implementation_model_name "${MODEL_NAME}" \
  --sample_image_tokens all \
  --shard_size "${SHARD_SIZE}" \
  --target_transform log \
  --torch_dtype float16 \
  --storage_dtype float16 \
  --target_dtype float32 \
  --teacher_head_reduction mean \
  --device "${DEVICE}" \
  --overwrite \
  --continue_on_error \
  "${LIMIT_ARG[@]}"

echo "[Stage 1b] NLVR2 XY collection done."

# ── Stage 2: Combined probe training ─────────────────────────────────────────

echo "============================================================"
echo "[Stage 2] Training combined probe"
echo "  ScienceQA shards: ${SQ_SHARD_DIR}"
echo "  TextVQA shards:   ${TEXTVQA_SHARD_DIR}"
echo "  NLVR2 shards:     ${NLVR2_SHARD_DIR}"
echo "  output:           ${PROBE_OUTPUT_DIR}"
echo "============================================================"

"${PYTHON_BIN}" /workspace/zap/train_image_teacher_probe_shards.py \
  --train_shard_dir "${SQ_SHARD_DIR}" "${TEXTVQA_SHARD_DIR}" "${NLVR2_SHARD_DIR}" \
  --output_dir "${PROBE_OUTPUT_DIR}" \
  --methods ${TRAIN_METHODS} \
  --mlp_max_epochs "${MLP_MAX_EPOCHS}" \
  --device "${DEVICE}"

echo "============================================================"
echo "All done. Probe saved to: ${PROBE_OUTPUT_DIR}"
echo "============================================================"
