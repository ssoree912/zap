#!/bin/bash
# B-only sweep: total_keep_ratio ∈ {0.1..0.9} on mm-vet + detail_1k
# Distributes remaining 16 runs + full detail_1k baseline across 3 GPUs.
set -uo pipefail

OUT=/workspace/zap/artifacts/EXP-20260425-002
CKPT=/workspace/zap/ckpts/student_v2_B
mkdir -p "$OUT"

run_student() {
  local gpu=$1 dataset=$2 keep=$3
  local outdir
  case "$dataset" in
    mm-vet) outdir="$OUT/mm-vet/B_total${keep}"; data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218 ;;
    detail_1k) outdir="$OUT/detail_1k/B_total${keep}"; data=/workspace/data/detail_1k.json; img=/workspace/data; n=1000 ;;
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

# GPU 0: all mm-vet (cheap, ~37s each) + detail_1k {0.4, 0.7, 0.9}
(
  for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    run_student 0 mm-vet $k
  done
  for k in 0.4 0.7 0.9; do
    run_student 0 detail_1k $k
  done
) &

# GPU 1: detail_1k 0.2 already running externally (skip if done) + {0.5, 0.8}
(
  run_student 1 detail_1k 0.2
  for k in 0.5 0.8; do
    run_student 1 detail_1k $k
  done
) &

# GPU 2: full baseline detail_1k + detail_1k {0.1, 0.3, 0.6}
(
  run_full 2 detail_1k
  for k in 0.1 0.3 0.6; do
    run_student 2 detail_1k $k
  done
) &

wait
echo "[done] all B total-budget sweep runs finished"
