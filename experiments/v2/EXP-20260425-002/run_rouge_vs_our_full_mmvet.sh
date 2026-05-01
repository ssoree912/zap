#!/bin/bash
# mm-vet ROUGE vs OUR full-cache reference (vs-full mode), all keep ratios.
# Uses reference at /workspace/zap/data/rouge_ref/our_full_mm-vet.json.
# Sequential on GPU 0.
set -uo pipefail

GPU=${GPU:-0}
CKPT=/workspace/zap/ckpts/student_v2_A_gqa_lr1e4
OUT=/workspace/zap/artifacts/EXP-20260425-002/mm-vet
REF=/workspace/zap/data/rouge_ref/our_full_mm-vet.json
IMG=/workspace/zap/data/rouge_ref   # reference's relative paths start with "images/" — but image files live under /workspace/data/mm-vet
# Reference was generated with --image-path /workspace/data/mm-vet; reuse it.
IMG=/workspace/data/mm-vet
N=218

run_keep() {
  local keep=$1
  local outdir="$OUT/A_gqa_lr1e4_total${keep}_rouge_vsourfull"
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[skip] keep=$keep already done"; return
  fi
  echo "[gpu $GPU] keep=$keep starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_rouge.py \
    --method visual_utility_student \
    --student-model-name "$CKPT" \
    --total-keep-ratio "$keep" \
    --data-path "$REF" --image-path "$IMG" \
    --eval-samples "$N" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[gpu $GPU] keep=$keep done"
}

run_full() {
  local outdir="$OUT/full_rouge_vsourfull"
  mkdir -p "$outdir"
  if [ -f "$outdir/result.json" ]; then
    echo "[skip] full already done"; return
  fi
  echo "[gpu $GPU] full vs full sanity starting"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_rouge.py \
    --method full --image-keep-ratio 1.0 \
    --data-path "$REF" --image-path "$IMG" \
    --eval-samples "$N" \
    --output-dir "$outdir" >"$outdir/eval.log" 2>&1
  echo "[gpu $GPU] full done"
}

# Full vs full sanity (should be ~1.0 since same model + greedy)
run_full
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  run_keep $k
done
echo "[done] mm-vet vs-our-full ROUGE finished"
