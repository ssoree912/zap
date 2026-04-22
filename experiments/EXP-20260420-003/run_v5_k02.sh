#!/usr/bin/env bash
# v5 probe @ k=0.2 only — mm-vet + detail_1k, GPU 0.
set -u
set -o pipefail

REPO=/workspace/zap
PROBE=/workspace/zap/ckpts/future_probe_v5_all_token_10ep
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf
ART=/workspace/zap/artifacts/EXP-20260420-003
LOG_DIR="$ART/logs_v5"
mkdir -p "$LOG_DIR"
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

run_one () {
    local ds_tag="$1" data_path="$2" image_path="$3" eval_samples="$4"
    local out_dir="$ART/${ds_tag}/future_v5_k0p2"
    local log_file="$LOG_DIR/${ds_tag}_future_v5_k0p2.log"
    if [[ -s "$out_dir/result.json" ]]; then
        echo "[SKIP] $ds_tag already has result.json"; return 0
    fi
    mkdir -p "$out_dir"
    echo "[RUN ] $ds_tag  → $out_dir"
    CUDA_VISIBLE_DEVICES=0 python "$REPO/eval_ppl.py" \
        --method future \
        --model-path "$MODEL" \
        --data-path "$data_path" \
        --image-path "$image_path" \
        --eval-samples "$eval_samples" \
        --output-dir "$out_dir" \
        --device cuda:0 \
        --image-keep-ratio 0.2 \
        --future-probe-name "$PROBE" \
        --selected-layer-indices "${LAYERS[@]}" 2>&1 | tee "$log_file"
}

run_one mmvet    /workspace/data/mm-vet/mm-vet.json /workspace/data/mm-vet 218
run_one detail1k /workspace/data/detail_1k.json      /workspace/data        1000

echo "[DONE] v5 k=0.2 sweep finished $(date -Iseconds)"
