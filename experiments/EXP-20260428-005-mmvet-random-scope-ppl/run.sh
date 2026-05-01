#!/bin/bash
# mm-vet PPL sweep for random image-only vs all-token eviction.
set -uo pipefail

EXP_DIR=/workspace/zap/experiments/EXP-20260428-005-mmvet-random-scope-ppl
OUT_DIR="$EXP_DIR/outputs"
LOG_DIR="$EXP_DIR/logs"
PY=${PY:-/opt/conda/envs/kv/bin/python}
GPU=${GPU:-2}
SEED=${SEED:-0}

DATA_GT=/workspace/data/mm-vet/mm-vet.json
IMG_ROOT=/workspace/data/mm-vet
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf

RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
METHODS=(random_image_only random_all_token)

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_full() {
  local out="$OUT_DIR/full/ppl"
  mkdir -p "$out"
  if [ -f "$out/result.json" ]; then
    echo "[skip] full cache PPL already exists: $out/result.json"
    return
  fi

  echo "[gpu $GPU] full cache PPL"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" /workspace/zap/eval_ppl.py \
    --model-path "$MODEL" \
    --method full \
    --data-path "$DATA_GT" \
    --image-path "$IMG_ROOT" \
    --eval-samples 218 \
    --output-dir "$out" \
    --attn-implementation sdpa \
    --seed "$SEED" \
    > "$out/run.log" 2>&1
  echo "[gpu $GPU] full cache done (rc=$?)"
}

run_method_ratio() {
  local method=$1
  local ratio=$2
  local tag=${ratio/./p}
  local out="$OUT_DIR/$method/keep_$tag/ppl"
  mkdir -p "$out"
  if [ -f "$out/result.json" ]; then
    echo "[skip] $method keep=$ratio already exists: $out/result.json"
    return
  fi

  echo "[gpu $GPU] PPL $method keep=$ratio seed=$SEED"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" /workspace/zap/eval_ppl.py \
    --model-path "$MODEL" \
    --method "$method" \
    --total-keep-ratio "$ratio" \
    --data-path "$DATA_GT" \
    --image-path "$IMG_ROOT" \
    --eval-samples 218 \
    --output-dir "$out" \
    --attn-implementation sdpa \
    --seed "$SEED" \
    > "$out/run.log" 2>&1
  echo "[gpu $GPU] $method keep=$ratio done (rc=$?)"
}

run_full

for ratio in "${RATIOS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_method_ratio "$method" "$ratio"
  done
  "$PY" "$EXP_DIR/aggregate_and_plot.py" --exp-dir "$EXP_DIR"
done

"$PY" "$EXP_DIR/aggregate_and_plot.py" --exp-dir "$EXP_DIR"
echo "[done] mm-vet random PPL sweep"
