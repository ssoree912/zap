#!/usr/bin/env bash
# EXP-20260420-003 — PPL sweep on PrefixKV protocol (detail_1k + mm-vet)
# Usage:
#   bash run_sweep.sh                 # all 26 runs on cuda:0
#   GPU=1 bash run_sweep.sh mmvet     # only mm-vet, on cuda:1
#   GPU=0 bash run_sweep.sh detail1k  # only detail_1k, on cuda:0

set -u
set -o pipefail

REPO=/workspace/zap
PROBE=/workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf
ARTIFACT_ROOT=/workspace/zap/artifacts/EXP-20260420-003
LOG_DIR="$ARTIFACT_ROOT/logs"
FAIL_LOG="$ARTIFACT_ROOT/failures.log"
GPU="${GPU:-0}"

mkdir -p "$LOG_DIR"
echo "# sweep started $(date -Iseconds)" >> "$FAIL_LOG"

DS_FILTER="${1:-all}"   # all | mmvet | detail1k
RATIOS=(0.1 0.2 0.3 0.5 0.7 0.9)

run_one () {
    local method="$1" ds_tag="$2" data_path="$3" image_path="$4" eval_samples="$5" ratio="$6"
    local tag="${method}"
    [[ "$method" != "full" ]] && tag="${method}_k${ratio//./p}"
    local out_dir="$ARTIFACT_ROOT/${ds_tag}/${tag}"
    local log_file="$LOG_DIR/${ds_tag}_${tag}.log"
    if [[ -s "$out_dir/result.json" ]]; then
        echo "[SKIP] $ds_tag/$tag already has result.json"
        return 0
    fi
    mkdir -p "$out_dir"
    echo "[RUN ] $ds_tag/$tag  → $out_dir"
    local cmd=(
        python "$REPO/eval_ppl.py"
        --method "$method"
        --model-path "$MODEL"
        --data-path "$data_path"
        --image-path "$image_path"
        --eval-samples "$eval_samples"
        --output-dir "$out_dir"
        --device "cuda:${GPU}"
    )
    if [[ "$method" != "full" ]]; then
        cmd+=(--image-keep-ratio "$ratio")
    fi
    if [[ "$method" == "future" ]]; then
        cmd+=(--future-probe-name "$PROBE"
              --selected-layer-indices 24 25 26 27 28 29 30 31)
    fi
    if ! "${cmd[@]}" 2>&1 | tee "$log_file"; then
        echo "[FAIL] $ds_tag/$tag  (see $log_file)" | tee -a "$FAIL_LOG"
        return 1
    fi
    return 0
}

run_dataset () {
    local ds_tag="$1" data_path="$2" image_path="$3" eval_samples="$4"
    run_one full "$ds_tag" "$data_path" "$image_path" "$eval_samples" 0.0
    for r in "${RATIOS[@]}"; do
        run_one future         "$ds_tag" "$data_path" "$image_path" "$eval_samples" "$r"
        run_one h2o_image_only "$ds_tag" "$data_path" "$image_path" "$eval_samples" "$r"
    done
}

if [[ "$DS_FILTER" == "all" || "$DS_FILTER" == "mmvet" ]]; then
    # mm-vet.json `image` field already has the `images/` prefix
    run_dataset mmvet /workspace/data/mm-vet/mm-vet.json /workspace/data/mm-vet 218
fi

if [[ "$DS_FILTER" == "all" || "$DS_FILTER" == "detail1k" ]]; then
    run_dataset detail1k /workspace/data/detail_1k.json /workspace/data 1000
fi

echo "==========="
if [[ -s "$FAIL_LOG" ]]; then
    echo "[DONE] sweep finished with failures:"
    cat "$FAIL_LOG"
else
    echo "[DONE] sweep finished cleanly, all runs wrote result.json"
fi
