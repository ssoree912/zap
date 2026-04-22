#!/usr/bin/env bash
# H2O-all-token + Hybrid(H2O+Future, α=0.5 flat) — total-keep sweep.
# 5 ratios × 2 methods × 2 datasets = 20 runs. GPU 1.
set -u
set -o pipefail

REPO=/workspace/zap
PROBE=/workspace/zap/ckpts/future_probe_v5_all_token_10ep
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf
ART=/workspace/zap/artifacts/EXP-20260420-003
GPU="${GPU:-1}"
LOG_DIR="$ART/logs_h2o_hybrid"
FAIL_LOG="$ART/failures_h2o_hybrid.log"
mkdir -p "$LOG_DIR"
echo "# h2o+hybrid sweep gpu${GPU} started $(date -Iseconds)" >> "$FAIL_LOG"

RATIOS=(0.1 0.3 0.5 0.7 0.9)

run_one () {
    local method="$1" ds_tag="$2" data_path="$3" image_path="$4" eval_samples="$5" ratio="$6"
    local short
    case "$method" in
        h2o_all_token)               short="h2oall" ;;
        hybrid_h2o_future_all_token) short="hybrid_a050" ;;
        *) short="$method" ;;
    esac
    local tag="${short}_tot${ratio//./p}"
    local out_dir="$ART/${ds_tag}/${tag}"
    local log_file="$LOG_DIR/${ds_tag}_${tag}.log"
    if [[ -s "$out_dir/result.json" ]]; then
        echo "[SKIP] gpu${GPU} $ds_tag/$tag"; return 0
    fi
    mkdir -p "$out_dir"
    echo "[RUN ] gpu${GPU} $ds_tag/$tag"
    local cmd=(
        CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/eval_ppl.py"
        --method "$method"
        --model-path "$MODEL"
        --data-path "$data_path"
        --image-path "$image_path"
        --eval-samples "$eval_samples"
        --output-dir "$out_dir"
        --device cuda:0
        --total-keep-ratio "$ratio"
        --attn-implementation eager
    )
    if [[ "$method" == "hybrid_h2o_future_all_token" ]]; then
        cmd+=(--future-probe-name "$PROBE" --alpha 0.5)
    fi
    env "${cmd[@]}" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    (( rc != 0 )) && echo "[FAIL] gpu${GPU} $ds_tag/$tag rc=$rc" | tee -a "$FAIL_LOG"
    return 0
}

# Interleave methods within each dataset so failures surface early per-method.
for r in "${RATIOS[@]}"; do
    run_one h2o_all_token               mmvet /workspace/data/mm-vet/mm-vet.json /workspace/data/mm-vet 218 "$r"
    run_one hybrid_h2o_future_all_token mmvet /workspace/data/mm-vet/mm-vet.json /workspace/data/mm-vet 218 "$r"
done
for r in "${RATIOS[@]}"; do
    run_one h2o_all_token               detail1k /workspace/data/detail_1k.json /workspace/data 1000 "$r"
    run_one hybrid_h2o_future_all_token detail1k /workspace/data/detail_1k.json /workspace/data 1000 "$r"
done
echo "[DONE] gpu${GPU} h2o+hybrid sweep finished $(date -Iseconds)"
