#!/bin/bash
# ROUGE re-run with PrefixKV's full-cache prediction reference (vs-full mode).
# Uses /workspace/PrefixKV/data/{mm-vet,detail_1k}/rouge-llava-v1.5-7b-*.json as references.
# Sequential on GPU 2.
set -uo pipefail

GPU=${GPU:-2}
CKPT=/workspace/zap/ckpts/student_v2_A_gqa_lr1e4
OUT=/workspace/zap/artifacts/EXP-20260425-002

run_rouge() {
  local dataset=$1 keep=$2
  local outdir
  case "$dataset" in
    mm-vet)
      outdir="$OUT/mm-vet/A_gqa_lr1e4_total${keep}_rouge_vsfull"
      data=/workspace/PrefixKV/data/mm-vet/rouge-llava-v1.5-7b-mm-vet.json
      img=/workspace/PrefixKV/data/mm-vet
      n=218
      ;;
    detail_1k)
      outdir="$OUT/detail_1k/A_gqa_lr1e4_total${keep}_rouge_vsfull"
      data=/workspace/PrefixKV/data/detail_1k/rouge-llava-v1.5-7b-detail_1k.json
      img=/workspace/PrefixKV/data
      n=1000
      ;;
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
    mm-vet)
      outdir="$OUT/mm-vet/full_rouge_vsfull"
      data=/workspace/PrefixKV/data/mm-vet/rouge-llava-v1.5-7b-mm-vet.json
      img=/workspace/PrefixKV/data/mm-vet
      n=218
      ;;
    detail_1k)
      outdir="$OUT/detail_1k/full_rouge_vsfull"
      data=/workspace/PrefixKV/data/detail_1k/rouge-llava-v1.5-7b-detail_1k.json
      img=/workspace/PrefixKV/data
      n=1000
      ;;
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

# mm-vet first (cheap)
run_full mm-vet
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  run_rouge mm-vet $k
done

# detail_1k (expensive)
run_full detail_1k
for k in 0.1 0.2 0.3 0.5 0.7 0.9; do
  run_rouge detail_1k $k
done

echo "[done] vs-full ROUGE finished"
