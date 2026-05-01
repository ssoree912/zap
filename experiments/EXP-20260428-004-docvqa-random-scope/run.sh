#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/workspace/zap/experiments/EXP-20260428-004-docvqa-random-scope
REPO=/workspace/zap
PYTHON=${PYTHON:-/opt/conda/envs/kv/bin/python}
GPU=${GPU:-2}
DEVICE=${DEVICE:-cuda:0}
MODEL=${MODEL:-/workspace/zap/ckpts/llava-1.5-7b-hf}
DATA_SRC=${DATA_SRC:-/workspace/zap/data/MileBench/DocVQA/DocVQA.json}
IMAGE_ROOT=${IMAGE_ROOT:-/workspace/zap/data/MileBench/DocVQA/combined_1_images}
EVAL_SAMPLES=${EVAL_SAMPLES:-200}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
RATIOS=${RATIOS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"}

INPUT_DIR="$EXP_DIR/inputs"
OUTPUT_DIR="$EXP_DIR/outputs"
LOG_DIR="$EXP_DIR/logs"
GT_MANIFEST="$INPUT_DIR/docvqa_${EVAL_SAMPLES}_gt.json"
FULL_REF_MANIFEST="$INPUT_DIR/docvqa_${EVAL_SAMPLES}_full_ref.json"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$LOG_DIR"

ratio_tag() {
  python - "$1" <<'PY'
import sys
print(sys.argv[1].replace(".", "p"))
PY
}

run_ppl() {
  local method="$1"
  local ratio="$2"
  local out="$3"
  local ratio_args=()
  if [[ "$method" != "full" ]]; then
    ratio_args+=(--total-keep-ratio "$ratio")
  fi
  if [[ -f "$out/result.json" ]]; then
    echo "[SKIP] PPL $method keep=$ratio"
    return
  fi
  echo "[RUN] PPL $method keep=$ratio -> $out"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$REPO/eval_ppl.py" \
    --model-path "$MODEL" \
    --data-path "$GT_MANIFEST" \
    --image-path "$IMAGE_ROOT" \
    --eval-samples "$EVAL_SAMPLES" \
    --method "$method" \
    "${ratio_args[@]}" \
    --torch-dtype bfloat16 \
    --device "$DEVICE" \
    --attn-implementation sdpa \
    --output-dir "$out"
}

run_rouge() {
  local method="$1"
  local ratio="$2"
  local data_path="$3"
  local out="$4"
  local ratio_args=()
  if [[ "$method" != "full" ]]; then
    ratio_args+=(--total-keep-ratio "$ratio")
  fi
  if [[ -f "$out/result.json" ]]; then
    echo "[SKIP] ROUGE $method keep=$ratio"
    return
  fi
  echo "[RUN] ROUGE $method keep=$ratio -> $out"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$REPO/eval_rouge.py" \
    --model-path "$MODEL" \
    --data-path "$data_path" \
    --image-path "$IMAGE_ROOT" \
    --eval-samples "$EVAL_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --method "$method" \
    "${ratio_args[@]}" \
    --torch-dtype bfloat16 \
    --device "$DEVICE" \
    --attn-implementation sdpa \
    --output-dir "$out"
}

cd "$REPO"

echo "[prepare] DocVQA flat manifest"
"$PYTHON" "$EXP_DIR/prepare_docvqa_manifest.py" \
  --input "$DATA_SRC" \
  --output "$GT_MANIFEST" \
  --limit "$EVAL_SAMPLES"

run_ppl full none "$OUTPUT_DIR/full/ppl_gt"
run_rouge full none "$GT_MANIFEST" "$OUTPUT_DIR/full/rouge_gt"

echo "[prepare] full-cache ROUGE reference manifest"
"$PYTHON" "$EXP_DIR/make_full_ref_manifest.py" \
  --manifest "$GT_MANIFEST" \
  --full-result "$OUTPUT_DIR/full/rouge_gt/result.json" \
  --output "$FULL_REF_MANIFEST"

for method in random_image_only random_all_token; do
  for ratio in $RATIOS; do
    tag=$(ratio_tag "$ratio")
    base="$OUTPUT_DIR/$method/keep_$tag"
    run_ppl "$method" "$ratio" "$base/ppl_gt"
    run_rouge "$method" "$ratio" "$FULL_REF_MANIFEST" "$base/rouge_vs_full"
    "$PYTHON" "$EXP_DIR/aggregate_results.py" --exp-dir "$EXP_DIR"
  done
done

"$PYTHON" "$EXP_DIR/aggregate_results.py" --exp-dir "$EXP_DIR"
echo "[DONE] EXP-20260428-004-docvqa-random-scope"

