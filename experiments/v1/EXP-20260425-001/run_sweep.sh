#!/usr/bin/env bash
# EXP-20260425-001: Quadrant eviction ablation
# Probe: future_probe_allL_limit250_all_token
# GPU: cuda:2

set -euo pipefail

DEVICE="cuda:2"
PROBE="/workspace/zap/ckpts/future_probe_allL_limit250_all_token"
MODEL="/workspace/zap/ckpts/llava-1.5-7b-hf"
OUT_BASE="/workspace/zap/artifacts/EXP-20260425-001"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p "${LOG_DIR}"

MMVET_DATA="/workspace/data/mm-vet/mm-vet.json"
MMVET_IMG="/workspace/data/mm-vet"
DETAIL_DATA="/workspace/data/detail_1k.json"
DETAIL_IMG="/workspace/data"

EVICT_RATIO=0.25

run_ppl() {
    local METHOD=$1; local QUADRANT=$2; local DS_NAME=$3
    local DATA=$4; local IMG=$5; local N=$6
    local TAG="${METHOD}${QUADRANT:+_${QUADRANT}}"
    local OUT="${OUT_BASE}/${DS_NAME}/${TAG}/ppl"
    local LOG="${LOG_DIR}/${DS_NAME}_${TAG}_ppl.log"
    if [ -f "${OUT}/result.json" ]; then
        echo "[SKIP] ${DS_NAME}/${TAG}/ppl already done"; return
    fi
    echo "[RUN ] ppl ${DS_NAME}/${TAG}"
    local EXTRA_ARGS=""
    if [ "${METHOD}" = "quadrant_eviction" ]; then
        EXTRA_ARGS="--quadrant ${QUADRANT} --evict-ratio ${EVICT_RATIO} --attn-implementation eager"
    elif [ "${METHOD}" = "h2o_all_token" ]; then
        EXTRA_ARGS="--total-keep-ratio 0.75 --attn-implementation eager"
    elif [ "${METHOD}" = "future_all_token" ]; then
        EXTRA_ARGS="--total-keep-ratio 0.75"
    elif [ "${METHOD}" = "hybrid_h2o_future_all_token" ]; then
        EXTRA_ARGS="--total-keep-ratio 0.75 --alpha 0.5 --attn-implementation eager"
    fi
    python /workspace/zap/eval_ppl.py \
        --method "${METHOD}" \
        --model-path "${MODEL}" \
        --data-path "${DATA}" \
        --image-path "${IMG}" \
        --eval-samples "${N}" \
        --future-probe-name "${PROBE}" \
        --device "${DEVICE}" \
        --output-dir "${OUT}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG}"
}

run_rouge() {
    local METHOD=$1; local QUADRANT=$2; local DS_NAME=$3
    local DATA=$4; local IMG=$5; local N=$6
    local TAG="${METHOD}${QUADRANT:+_${QUADRANT}}"
    local OUT="${OUT_BASE}/${DS_NAME}/${TAG}/rouge"
    local LOG="${LOG_DIR}/${DS_NAME}_${TAG}_rouge.log"
    if [ -f "${OUT}/result.json" ]; then
        echo "[SKIP] ${DS_NAME}/${TAG}/rouge already done"; return
    fi
    echo "[RUN ] rouge ${DS_NAME}/${TAG}"
    local EXTRA_ARGS=""
    if [ "${METHOD}" = "quadrant_eviction" ]; then
        EXTRA_ARGS="--quadrant ${QUADRANT} --evict-ratio ${EVICT_RATIO} --attn-implementation eager"
    elif [ "${METHOD}" = "h2o_all_token" ]; then
        EXTRA_ARGS="--total-keep-ratio 0.75 --attn-implementation eager"
    elif [ "${METHOD}" = "future_all_token" ]; then
        EXTRA_ARGS="--total-keep-ratio 0.75"
    elif [ "${METHOD}" = "hybrid_h2o_future_all_token" ]; then
        EXTRA_ARGS="--total-keep-ratio 0.75 --alpha 0.5 --attn-implementation eager"
    fi
    python /workspace/zap/eval_rouge.py \
        --method "${METHOD}" \
        --model-path "${MODEL}" \
        --data-path "${DATA}" \
        --image-path "${IMG}" \
        --eval-samples "${N}" \
        --future-probe-name "${PROBE}" \
        --device "${DEVICE}" \
        --output-dir "${OUT}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG}"
}

# ── full baseline ──────────────────────────────────────────────────────────────
# full does not need --image-keep-ratio or --total-keep-ratio
run_full_ppl() {
    local DS_NAME=$1; local DATA=$2; local IMG=$3; local N=$4
    local OUT="${OUT_BASE}/${DS_NAME}/full/ppl"
    local LOG="${LOG_DIR}/${DS_NAME}_full_ppl.log"
    [ -f "${OUT}/result.json" ] && { echo "[SKIP] ${DS_NAME}/full/ppl"; return; }
    echo "[RUN ] ppl ${DS_NAME}/full"
    python /workspace/zap/eval_ppl.py \
        --method full \
        --model-path "${MODEL}" \
        --data-path "${DATA}" \
        --image-path "${IMG}" \
        --eval-samples "${N}" \
        --device "${DEVICE}" \
        --output-dir "${OUT}" \
        --image-keep-ratio 1.0 \
        2>&1 | tee "${LOG}"
}

run_full_rouge() {
    local DS_NAME=$1; local DATA=$2; local IMG=$3; local N=$4
    local OUT="${OUT_BASE}/${DS_NAME}/full/rouge"
    local LOG="${LOG_DIR}/${DS_NAME}_full_rouge.log"
    [ -f "${OUT}/result.json" ] && { echo "[SKIP] ${DS_NAME}/full/rouge"; return; }
    echo "[RUN ] rouge ${DS_NAME}/full"
    python /workspace/zap/eval_rouge.py \
        --method full \
        --model-path "${MODEL}" \
        --data-path "${DATA}" \
        --image-path "${IMG}" \
        --eval-samples "${N}" \
        --device "${DEVICE}" \
        --output-dir "${OUT}" \
        --image-keep-ratio 1.0 \
        2>&1 | tee "${LOG}"
}

# ── mm-vet (218 samples) ───────────────────────────────────────────────────────
echo "=== mm-vet ==="
run_full_ppl   "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
run_full_rouge "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218

for Q in HH HL LH LL; do
    run_ppl   quadrant_eviction "${Q}" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
    run_rouge quadrant_eviction "${Q}" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
done

run_ppl   future_all_token   "" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
run_rouge future_all_token   "" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
run_ppl   h2o_all_token      "" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
run_rouge h2o_all_token      "" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
run_ppl   hybrid_h2o_future_all_token "" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218
run_rouge hybrid_h2o_future_all_token "" "mm-vet" "${MMVET_DATA}" "${MMVET_IMG}" 218

# ── detail_1k (1000 samples) ──────────────────────────────────────────────────
echo "=== detail_1k ==="
run_full_ppl   "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
run_full_rouge "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000

for Q in HH HL LH LL; do
    run_ppl   quadrant_eviction "${Q}" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
    run_rouge quadrant_eviction "${Q}" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
done

run_ppl   future_all_token   "" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
run_rouge future_all_token   "" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
run_ppl   h2o_all_token      "" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
run_rouge h2o_all_token      "" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
run_ppl   hybrid_h2o_future_all_token "" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000
run_rouge hybrid_h2o_future_all_token "" "detail_1k" "${DETAIL_DATA}" "${DETAIL_IMG}" 1000

echo ""
echo "=== ALL DONE ==="
echo "Results in: ${OUT_BASE}"
