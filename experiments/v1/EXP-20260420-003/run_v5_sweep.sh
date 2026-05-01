#!/usr/bin/env bash
# v5 probe (all-token, 10 epoch) PPL sweep on GPU 0.
# 12 runs: 2 datasets × 6 ratios, future-only. full-cache baseline excluded
# (shared across probe versions — run separately via run_sweep.sh).

set -u
set -o pipefail

REPO=/workspace/zap
PROBE=/workspace/zap/ckpts/future_probe_v5_all_token_10ep
PROBE_TAG=future_v5
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf
ARTIFACT_ROOT=/workspace/zap/artifacts/EXP-20260420-003
LOG_DIR="$ARTIFACT_ROOT/logs_v5"
FAIL_LOG="$ARTIFACT_ROOT/failures_v5.log"
GPU="${GPU:-0}"

mkdir -p "$LOG_DIR"
echo "# v5 sweep started $(date -Iseconds) on cuda:${GPU}" >> "$FAIL_LOG"

RATIOS=(0.1 0.2 0.3 0.5 0.7 0.9)
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

run_one () {
    local ds_tag="$1" data_path="$2" image_path="$3" eval_samples="$4" ratio="$5"
    local tag="${PROBE_TAG}_k${ratio//./p}"
    local out_dir="$ARTIFACT_ROOT/${ds_tag}/${tag}"
    local log_file="$LOG_DIR/${ds_tag}_${tag}.log"
    if [[ -s "$out_dir/result.json" ]]; then
        echo "[SKIP] $ds_tag/$tag"
        return 0
    fi
    mkdir -p "$out_dir"
    echo "[RUN ] $ds_tag/$tag  → $out_dir"
    CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/eval_ppl.py" \
        --method future \
        --model-path "$MODEL" \
        --data-path "$data_path" \
        --image-path "$image_path" \
        --eval-samples "$eval_samples" \
        --output-dir "$out_dir" \
        --device "cuda:0" \
        --image-keep-ratio "$ratio" \
        --future-probe-name "$PROBE" \
        --selected-layer-indices "${LAYERS[@]}" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo "[FAIL] $ds_tag/$tag rc=$rc (see $log_file)" | tee -a "$FAIL_LOG"
    fi
    return 0
}

for r in "${RATIOS[@]}"; do
    run_one mmvet    /workspace/data/mm-vet/mm-vet.json /workspace/data/mm-vet      218  "$r"
done
for r in "${RATIOS[@]}"; do
    run_one detail1k /workspace/data/detail_1k.json      /workspace/data             1000 "$r"
done

echo "=========="
echo "[DONE] v5 sweep finished $(date -Iseconds)"
if grep -q '^\[FAIL\]' "$FAIL_LOG" 2>/dev/null; then
    echo "Failures:"
    grep '^\[FAIL\]' "$FAIL_LOG"
fi
