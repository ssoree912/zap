#!/usr/bin/env bash
# v5 probe — total_keep_ratio sweep. Ratios passed as positional args.
# Usage:
#   GPU=0 bash run_v5_total_sweep.sh 0.1 0.3
#   GPU=1 bash run_v5_total_sweep.sh 0.5 0.7 0.9
set -u
set -o pipefail

REPO=/workspace/zap
PROBE=/workspace/zap/ckpts/future_probe_v5_all_token_10ep
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf
ART=/workspace/zap/artifacts/EXP-20260420-003
GPU="${GPU:-0}"
LOG_DIR="$ART/logs_v5_total_gpu${GPU}"
FAIL_LOG="$ART/failures_v5_total.log"
mkdir -p "$LOG_DIR"
echo "# v5 total sweep gpu${GPU} started $(date -Iseconds) ratios=$*" >> "$FAIL_LOG"

if (( $# == 0 )); then
    echo "usage: $0 <ratio> [<ratio> ...]"; exit 2
fi
RATIOS=("$@")
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

run_one () {
    local ds_tag="$1" data_path="$2" image_path="$3" eval_samples="$4" ratio="$5"
    local tag="future_v5_tot${ratio//./p}"
    local out_dir="$ART/${ds_tag}/${tag}"
    local log_file="$LOG_DIR/${ds_tag}_${tag}.log"
    if [[ -s "$out_dir/result.json" ]]; then
        echo "[SKIP] gpu${GPU} $ds_tag/$tag"; return 0
    fi
    mkdir -p "$out_dir"
    echo "[RUN ] gpu${GPU} $ds_tag/$tag"
    CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/eval_ppl.py" \
        --method future \
        --model-path "$MODEL" \
        --data-path "$data_path" \
        --image-path "$image_path" \
        --eval-samples "$eval_samples" \
        --output-dir "$out_dir" \
        --device cuda:0 \
        --total-keep-ratio "$ratio" \
        --future-probe-name "$PROBE" \
        --selected-layer-indices "${LAYERS[@]}" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    (( rc != 0 )) && echo "[FAIL] gpu${GPU} $ds_tag/$tag rc=$rc" | tee -a "$FAIL_LOG"
    return 0
}

for r in "${RATIOS[@]}"; do
    run_one mmvet    /workspace/data/mm-vet/mm-vet.json /workspace/data/mm-vet 218  "$r"
done
for r in "${RATIOS[@]}"; do
    run_one detail1k /workspace/data/detail_1k.json      /workspace/data        1000 "$r"
done
echo "[DONE] gpu${GPU} v5 total sweep finished $(date -Iseconds)"
