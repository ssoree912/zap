#!/bin/bash
# Re-run ROUGE with corrected DEFAULT_PROMPT_TEMPLATE (now includes vicuna system prompt).
# Uses new lr=1e-4 ckpt. Sequential on GPU 2.
set -uo pipefail

GPU=${GPU:-2}
CKPT=/workspace/zap/ckpts/student_v2_A_gqa_lr1e4
OUT=/workspace/zap/artifacts/EXP-20260425-002

run_rouge() {
  local dataset=$1 keep=$2
  local outdir
  case "$dataset" in
    mm-vet) outdir="$OUT/mm-vet/A_gqa_lr1e4_total${keep}_rouge"; data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218 ;;
    detail_1k) outdir="$OUT/detail_1k/A_gqa_lr1e4_total${keep}_rouge_v2"; data=/workspace/data/detail_1k.json; img=/workspace/data; n=1000 ;;
  esac
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[skip] $dataset k=$keep already done"
    return
  fi
  echo "[gpu $GPU] $dataset k=$keep starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_rouge.py \
    --method visual_utility_student \
    --student-model-name "$CKPT" \
    --total-keep-ratio "$keep" \
    --data-path "$data" --image-path "$img" \
    --eval-samples "$n" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[gpu $GPU] $dataset k=$keep done"
}

run_full() {
  local dataset=$1
  local outdir
  case "$dataset" in
    mm-vet) outdir="$OUT/mm-vet/full_rouge_v2"; data=/workspace/data/mm-vet/mm-vet.json; img=/workspace/data/mm-vet; n=218 ;;
    detail_1k) outdir="$OUT/detail_1k/full_rouge_v2"; data=/workspace/data/detail_1k.json; img=/workspace/data; n=1000 ;;
  esac
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[skip] full $dataset already done"
    return
  fi
  echo "[gpu $GPU] full $dataset starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_rouge.py \
    --method full --image-keep-ratio 1.0 \
    --data-path "$data" --image-path "$img" \
    --eval-samples "$n" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[gpu $GPU] full $dataset done"
}

# mm-vet first (cheap, verify prompt fix)
run_full mm-vet
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  run_rouge mm-vet $k
done

# detail_1k full + key ratios
run_full detail_1k
for k in 0.2 0.1 0.3 0.5; do
  run_rouge detail_1k $k
done

echo "[done] ROUGE re-run with vicuna prompt finished"
