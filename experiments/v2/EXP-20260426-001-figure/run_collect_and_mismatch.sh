#!/bin/bash
# Collect prefill+future attention dump and compute mismatch metrics.
# GPU 1 = DocVQA / OCR-VQA   GPU 2 = SlideVQA / mm-vet  (parallel)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT="$SCRIPT_DIR/collect_attn_dump.py"
MISMATCH="$SCRIPT_DIR/compute_mismatch.py"

N_SAMPLES=${N_SAMPLES:-30}

collect_and_measure() {
  local dataset=$1
  local gpu=$2
  echo "[gpu $gpu] collecting $dataset (n=$N_SAMPLES)..."
  CUDA_VISIBLE_DEVICES=$gpu python "$COLLECT" \
    --dataset "$dataset" \
    --n-samples "$N_SAMPLES" \
    --device "cuda:$gpu" \
    > "/tmp/collect_${dataset}.log" 2>&1
  local rc=$?
  echo "[gpu $gpu] $dataset collect done (rc=$rc)"

  echo "[gpu $gpu] computing mismatch for $dataset..."
  python "$MISMATCH" \
    --dataset "$dataset" \
    --keep-ratios 0.2 0.5 \
    >> "/tmp/collect_${dataset}.log" 2>&1
  echo "[gpu $gpu] $dataset mismatch done"
  cat "/tmp/collect_${dataset}.log" | tail -20
}

# run sequentially per GPU (one model per GPU to avoid OOM)
# GPU 1: DocVQA → OCR-VQA
( collect_and_measure DocVQA   1 && collect_and_measure OCR-VQA  1 ) &
# GPU 2: SlideVQA → mmvet
( collect_and_measure SlideVQA 2 && collect_and_measure mmvet    2 ) &
wait

echo "[done] all datasets collected and mismatch computed"
echo "Candidates CSVs:"
ls /workspace/zap/artifacts/EXP-20260426-001-figure/candidates_*.csv 2>/dev/null
