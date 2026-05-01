#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope
REPO=/workspace/zap
PYTHON=${PYTHON:-/opt/conda/envs/kv/bin/python}
GPU=${GPU:-2}
DEVICE=${DEVICE:-cuda:0}
MODEL=${MODEL:-/workspace/zap/ckpts/llava-1.5-7b-hf}
IMAGE_ROOT=${IMAGE_ROOT:-/workspace/zap/data/MileBench/DocVQA/combined_1_images}
EVAL_SAMPLES=${EVAL_SAMPLES:-200}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
SEED=${SEED:-0}
RATIOS=${RATIOS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"}

INPUT_DIR="$EXP_DIR/inputs"
OUTPUT_DIR="$EXP_DIR/outputs"
LOG_DIR="$EXP_DIR/logs"
GT_MANIFEST="$INPUT_DIR/docvqa_${EVAL_SAMPLES}_gt.json"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

ratio_tag() {
  python - "$1" <<'PY'
import sys
print(sys.argv[1].replace(".", "p"))
PY
}

run_rouge_gt() {
  local method="$1"
  local ratio="$2"
  local out="$3"
  local log="$out/run.log"

  if [[ -f "$out/result.json" ]]; then
    echo "[SKIP] ROUGE-GT $method keep=$ratio"
    return
  fi

  mkdir -p "$out"
  echo "[RUN] ROUGE-GT $method keep=$ratio -> $out"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$REPO/eval_rouge.py" \
    --model-path "$MODEL" \
    --data-path "$GT_MANIFEST" \
    --image-path "$IMAGE_ROOT" \
    --eval-samples "$EVAL_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --method "$method" \
    --total-keep-ratio "$ratio" \
    --torch-dtype bfloat16 \
    --device "$DEVICE" \
    --attn-implementation sdpa \
    --seed "$SEED" \
    --output-dir "$out" 2>&1 | tee "$log"
}

cd "$REPO"

if [[ ! -f "$GT_MANIFEST" ]]; then
  echo "[ERROR] Missing GT manifest: $GT_MANIFEST" >&2
  echo "Run $EXP_DIR/run.sh once, or regenerate inputs/docvqa_${EVAL_SAMPLES}_gt.json." >&2
  exit 1
fi

echo "[start] DocVQA random-scope ROUGE vs GT"
echo "[config] gpu=$GPU seed=$SEED samples=$EVAL_SAMPLES max_new_tokens=$MAX_NEW_TOKENS"

for method in random_image_only random_all_token; do
  for ratio in $RATIOS; do
    tag=$(ratio_tag "$ratio")
    run_rouge_gt "$method" "$ratio" "$OUTPUT_DIR/$method/keep_$tag/rouge_gt"
    "$PYTHON" "$EXP_DIR/plot_original_scores.py" --exp-dir "$EXP_DIR"
  done
done

"$PYTHON" "$EXP_DIR/plot_original_scores.py" --exp-dir "$EXP_DIR"
echo "[DONE] DocVQA random-scope ROUGE vs GT"
