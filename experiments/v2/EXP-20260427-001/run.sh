#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="/workspace/zap/experiments/EXP-20260427-001"
CONFIG="/workspace/zap/eval_configs/onevision_baseline_chartqa.json"
ROOT="${ROOT:-/workspace/zap/eval_results/baseline_chartqa_default}"
LOG_DIR="$ROOT/logs"
TARGET="${TARGET:-80.3}"

GPU="${GPU:-0}"

work_dir="$ROOT/default"
log="$LOG_DIR/default.log"
acc="$work_dir/ov7b_baseline_local/ov7b_baseline_local_ChartQA_TEST_acc.csv"

mkdir -p "$LOG_DIR" "$work_dir"

if [[ -f "$acc" ]]; then
  echo "[gpu=$GPU] existing metric found, skipping: $acc"
  exit 0
fi

echo "EXP_DIR=$EXP_DIR"
echo "ROOT=$ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "TARGET=$TARGET"
echo "GPU=$GPU"
echo "No explicit seed is set for this run."

echo "[gpu=$GPU] start $(date -Is)"
CUDA_VISIBLE_DEVICES="$GPU" \
  python /workspace/zap/scripts/run_vlmeval_student.py -- \
    --config "$CONFIG" \
    --work-dir "$work_dir" \
  >"$log" 2>&1
echo "[gpu=$GPU] done $(date -Is)"
