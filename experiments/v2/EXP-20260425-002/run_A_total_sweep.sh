#!/bin/bash
# scope A (32-layer student) total_keep sweep on mm-vet + detail_1k.
# Uses GPU 0 + GPU 1 only. GPU 2 left free for other work.
set -uo pipefail

OUT=/workspace/zap/artifacts/EXP-20260425-002
CKPT=/workspace/zap/ckpts/student_v2_A
mkdir -p "$OUT"

run_student() {
  local gpu=$1 dataset=$2 keep=$3
  local outdir
  case "$dataset" in
    mm-vet) outdir="$OUT/mm-vet/A_total${keep}"; data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218 ;;
    detail_1k) outdir="$OUT/detail_1k/A_total${keep}"; data=/workspace/data/detail_1k.json; img=/workspace/data; n=1000 ;;
  esac
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[gpu $gpu] $dataset total_k=$keep already done — skip"
    return
  fi
  echo "[gpu $gpu] $dataset total_k=$keep starting"
  CUDA_VISIBLE_DEVICES=$gpu python /workspace/zap/eval_ppl.py \
    --method visual_utility_student \
    --student-model-name "$CKPT" \
    --total-keep-ratio "$keep" \
    --data-path "$data" --image-path "$img" \
    --eval-samples "$n" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[gpu $gpu] $dataset total_k=$keep done"
}

run_full() {
  local gpu=$1 dataset=$2
  local outdir
  case "$dataset" in
    mm-vet) outdir="$OUT/mm-vet/full"; data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218 ;;
    detail_1k) outdir="$OUT/detail_1k/full"; data=/workspace/data/detail_1k.json; img=/workspace/data; n=1000 ;;
  esac
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[gpu $gpu] full $dataset already done — skip"
    return
  fi
  echo "[gpu $gpu] full $dataset starting"
  CUDA_VISIBLE_DEVICES=$gpu python /workspace/zap/eval_ppl.py \
    --method full --image-keep-ratio 1.0 \
    --data-path "$data" --image-path "$img" \
    --eval-samples "$n" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[gpu $gpu] full $dataset done"
}

# GPU 0: all mm-vet (cheap) + detail_1k {0.1, 0.3, 0.5, 0.7, 0.9}
(
  for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    run_student 0 mm-vet $k
  done
  for k in 0.1 0.3 0.5 0.7 0.9; do
    run_student 0 detail_1k $k
  done
) &

# GPU 1: detail_1k {0.2, 0.4, 0.6, 0.8} + full detail_1k baseline
(
  for k in 0.2 0.4 0.6 0.8; do
    run_student 1 detail_1k $k
  done
  run_full 1 detail_1k
) &

wait
echo "[done] all A scope total-budget sweep runs finished"
