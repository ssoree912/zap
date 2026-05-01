#!/bin/bash
# Trajectory student ROUGE-L eval sweep — GPU 2.
set -uo pipefail

OUT=/workspace/zap/artifacts/EXP-20260425-002
CKPT=/workspace/zap/ckpts/student_v2_traj
mkdir -p "$OUT"

run_rouge() {
  local gpu=$1 dataset=$2 keep=$3
  local outdir data img n
  case "$dataset" in
    mm-vet)    outdir="$OUT/mm-vet/traj_rouge_total${keep}";    data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218  ;;
    detail_1k) outdir="$OUT/detail_1k/traj_rouge_total${keep}"; data=/workspace/data/detail_1k.json;    img=/workspace/data;          n=1000 ;;
  esac
  mkdir -p "$outdir"
  [ -f "$outdir/result.json" ] && { echo "[skip] $dataset traj rouge k=$keep"; return; }
  echo "[gpu $gpu] $dataset traj rouge k=$keep starting"
  CUDA_VISIBLE_DEVICES=$gpu python /workspace/zap/eval_rouge.py \
    --method visual_utility_student \
    --student-model-name "$CKPT" \
    --total-keep-ratio "$keep" \
    --data-path "$data" --image-path "$img" \
    --eval-samples "$n" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[gpu $gpu] $dataset traj rouge k=$keep done"
}

for k in 0.3 0.5 0.7; do
  run_rouge 2 mm-vet $k
  run_rouge 2 detail_1k $k
done

echo "[done] traj ROUGE eval finished"
