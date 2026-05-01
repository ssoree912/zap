#!/bin/bash
# scope A ROUGE sweep on mm-vet + detail_1k. Sequential on GPU 0.
set -uo pipefail

OUT=/workspace/zap/artifacts/EXP-20260425-002
CKPT=/workspace/zap/ckpts/student_v2_A
GPU=0
mkdir -p "$OUT"

run_rouge() {
  local dataset=$1 keep=$2
  local outdir
  case "$dataset" in
    mm-vet) outdir="$OUT/mm-vet/A_total${keep}_rouge"; data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218 ;;
    detail_1k) outdir="$OUT/detail_1k/A_total${keep}_rouge"; data=/workspace/data/detail_1k.json; img=/workspace/data; n=1000 ;;
  esac
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[rouge] $dataset total_k=$keep already done — skip"
    return
  fi
  echo "[rouge] $dataset total_k=$keep starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_rouge.py \
    --method visual_utility_student \
    --student-model-name "$CKPT" \
    --total-keep-ratio "$keep" \
    --data-path "$data" --image-path "$img" \
    --eval-samples "$n" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[rouge] $dataset total_k=$keep done"
}

run_rouge_full() {
  local dataset=$1
  local outdir
  case "$dataset" in
    mm-vet) outdir="$OUT/mm-vet/full_rouge"; data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218 ;;
    detail_1k) outdir="$OUT/detail_1k/full_rouge"; data=/workspace/data/detail_1k.json; img=/workspace/data; n=1000 ;;
  esac
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[rouge] full $dataset already done — skip"
    return
  fi
  echo "[rouge] full $dataset starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_rouge.py \
    --method full --image-keep-ratio 1.0 \
    --data-path "$data" --image-path "$img" \
    --eval-samples "$n" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[rouge] full $dataset done"
}

# mm-vet first (cheap), then detail_1k. Plus full baselines.
for ds in mm-vet detail_1k; do
  run_rouge_full "$ds"
  for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    run_rouge "$ds" "$k"
  done
done

echo "[done] all ROUGE runs finished"
