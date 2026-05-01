#!/bin/bash
# mm-vet PPL + ROUGE-L vs full-cache output, random_all_token @ keep=0.2.
# Default GPU=2.
set -uo pipefail

GPU=${GPU:-2}
KEEP=0.2
DATA_GT=/workspace/data/mm-vet/mm-vet.json                    # real dataset GT (for PPL)
DATA_FULL=/workspace/zap/data/rouge_ref/our_full_mm-vet.json  # full-cache output (for ROUGE-vs-full)
IMG_ROOT=/workspace/data/mm-vet
OUT_ROOT=/workspace/zap/artifacts/EXP-20260426-001-figure/mmvet_random_all
mkdir -p "$OUT_ROOT"

PPL_DIR="$OUT_ROOT/ppl"
ROUGE_DIR="$OUT_ROOT/rouge_vsourfull"
mkdir -p "$PPL_DIR" "$ROUGE_DIR"

# ── PPL ───────────────────────────────────────────────────────────────────────
if [ -f "$PPL_DIR/result.json" ]; then
  echo "[skip] PPL already done"
else
  echo "[gpu $GPU] PPL random_all_token keep=$KEEP"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_ppl.py \
    --method random_all_token \
    --total-keep-ratio "$KEEP" \
    --data-path "$DATA_GT" \
    --image-path "$IMG_ROOT" \
    --eval-samples 218 \
    --output-dir "$PPL_DIR" \
    --attn-implementation sdpa \
    > "$PPL_DIR/run.log" 2>&1
  echo "[gpu $GPU] PPL done (rc=$?)"
fi

# ── ROUGE-L vs full-cache reference ──────────────────────────────────────────
if [ -f "$ROUGE_DIR/result.json" ]; then
  echo "[skip] ROUGE already done"
else
  echo "[gpu $GPU] ROUGE random_all_token keep=$KEEP"
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/zap/eval_rouge.py \
    --method random_all_token \
    --total-keep-ratio "$KEEP" \
    --data-path "$DATA_FULL" \
    --image-path "$IMG_ROOT" \
    --eval-samples 218 \
    --output-dir "$ROUGE_DIR" \
    --max-new-tokens 512 \
    --attn-implementation sdpa \
    > "$ROUGE_DIR/run.log" 2>&1
  echo "[gpu $GPU] ROUGE done (rc=$?)"
fi

echo "[done] random_all_token @ keep=$KEEP on mm-vet"
