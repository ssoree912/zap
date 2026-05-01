#!/bin/bash
# detail_1k ROUGE vs OUR full-cache reference (vs-full mode), all keep ratios.
# Sequential on GPU 2.
set -uo pipefail

GPU=${GPU:-2}
CKPT=/workspace/zap/ckpts/student_v2_A_gqa_lr1e4
OUT=/workspace/zap/artifacts/EXP-20260425-002/detail_1k
REF=/workspace/zap/data/rouge_ref/our_full_detail_1k.json
IMG=/workspace/data
N=1000

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

run_full
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  run_keep $k
done
echo "[done] detail_1k vs-our-full ROUGE finished"
