#!/usr/bin/env bash
# Run ZAP student KV-pruning eval on LLaVA-OneVision 7B over MileBench.
#
# Defaults: HF-format OneVision, image keep_ratio=0.2, combine_image=1
# (single stitched grid), GPU 0.
# Override by env: GPU, KEEP_RATIO, COMBINE_IMAGE, MAX_NEW_TOKENS, DATASET,
# OUTPUT_DIR, LIMIT, OVERWRITE, NO_SCORE, MODEL_FORMAT, MODEL_PATH.
#
# Usage:
#   ./run_milebench_zap_student_onevision.sh
#   GPU=1 KEEP_RATIO=0.5 ./run_milebench_zap_student_onevision.sh
set -euo pipefail

GPU="${GPU:-0}"
KEEP_RATIO="${KEEP_RATIO:-0.2}"
COMBINE_IMAGE="${COMBINE_IMAGE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DATASET="${DATASET:-full}"
MODEL_FORMAT="${MODEL_FORMAT:-hf}"
NO_SCORE="${NO_SCORE:-0}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "${LIMIT}"); fi
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then OVERWRITE_ARGS=(--overwrite); fi
NO_SCORE_ARGS=()
if [[ "${NO_SCORE}" == "1" ]]; then NO_SCORE_ARGS=(--no-score); fi
COMBINE_ARGS=()
if [[ -n "${COMBINE_IMAGE}" && "${COMBINE_IMAGE}" != "0" ]]; then
    COMBINE_ARGS=(--combine_image "${COMBINE_IMAGE}")
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZAP_ROOT="$(cd "${THIS_DIR}/.." && pwd)"

KEEP_TAG="$(printf '%s' "${KEEP_RATIO}" | tr '.' 'p')"
COMBINE_TAG="${COMBINE_IMAGE:+_combine${COMBINE_IMAGE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ZAP_ROOT}/logs/milebench_zap_student_onevision_${MODEL_FORMAT}_keep${KEEP_TAG}${COMBINE_TAG}_max${MAX_NEW_TOKENS}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

ENV_PYTHON="${ENV_PYTHON:-/opt/conda/envs/vflowopt_chartqa_eval/bin/python}"
STUDENT_PATH="${STUDENT_PATH:-/workspace/zap/ckpts/student_onevision_A_ep20}"
if [[ "${MODEL_FORMAT}" == "hf" ]]; then
    MODEL_PATH="${MODEL_PATH:-/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf}"
else
    MODEL_PATH="${MODEL_PATH:-/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-/workspace/VFlowOpt/src/lmms_eval-0.2.4}"
export PYTHONPATH="/workspace/VFlowOpt/src/LLaVA-OneVision:${LMMS_EVAL_ROOT}:/workspace/look-m:/workspace/zap:${PYTHONPATH:-}"

DATASETS_4=(ALFRED CLEVR-Change IEdit Spot-the-Diff)
DATASETS_FULL=(
    ALFRED ActionLocalization ActionPrediction ActionSequence
    CLEVR-Change CharacterOrder CounterfactualInference DocVQA
    EgocentricNavigation GPR1200 IEdit ImageNeedleInAHaystack
    MMCoQA MovingAttribute MovingDirection MultiModalQA
    OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle
    SceneTransition SlideVQA Spot-the-Diff StateChange
    TQA TextNeedleInAHaystack WebQA WikiVQA
)

if [[ "${DATASET}" == "all" ]]; then
    DATASETS_RESOLVED=("${DATASETS_4[@]}")
elif [[ "${DATASET}" == "full" ]]; then
    DATASETS_RESOLVED=("${DATASETS_FULL[@]}")
elif [[ "${DATASET}" == *","* ]]; then
    IFS=',' read -r -a DATASETS_RESOLVED <<< "${DATASET}"
else
    DATASETS_RESOLVED=("${DATASET}")
fi

echo "[launch] model=onevision format=${MODEL_FORMAT} keep_ratio=${KEEP_RATIO} keep_ratio_basis=image combine_image=${COMBINE_IMAGE} max_new_tokens=${MAX_NEW_TOKENS} dataset=${DATASET} gpu=${GPU}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] python=${ENV_PYTHON} model=${MODEL_PATH} student=${STUDENT_PATH}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/run.log}"

cd "${ZAP_ROOT}"

if [[ "${MODEL_FORMAT}" == "hf" ]]; then
    if [[ "${COMBINE_IMAGE}" != "1" ]]; then
        echo "[error] MODEL_FORMAT=hf currently uses MileBench combined_1_images; set COMBINE_IMAGE=1."
        exit 2
    fi
    {
        for task in "${DATASETS_RESOLVED[@]}"; do
            task="${task//[[:space:]]/}"
            [[ -z "${task}" ]] && continue
            pred_file="${OUTPUT_DIR}/${task}/pred.json"
            if [[ -f "${pred_file}" && "${OVERWRITE:-0}" != "1" ]]; then
                echo "===== SKIP ${task} (pred.json exists) ====="
            else
                echo "===== START task=${task} keep=${KEEP_RATIO} $(date -Is) ====="
                "${ENV_PYTHON}" "${ZAP_ROOT}/foresight/eval/milebench_onevision_student.py" \
                    --dataset "${task}" \
                    --pretrained "${MODEL_PATH}" \
                    --student_path "${STUDENT_PATH}" \
                    --keep_ratio "${KEEP_RATIO}" \
                    --output_dir "${OUTPUT_DIR}" \
                    --device cuda:0 \
                    --max_new_tokens "${MAX_NEW_TOKENS}" \
                    "${LIMIT_ARGS[@]}" \
                    "${OVERWRITE_ARGS[@]}"
                echo "===== DONE ${task} $(date -Is) ====="
            fi

            eval_file="${OUTPUT_DIR}/${task}/eval.json"
            if [[ "${NO_SCORE}" != "1" && ! -f "${eval_file}" && -f "${pred_file}" ]]; then
                "${ENV_PYTHON}" /workspace/look-m/evaluate.py \
                    --data-dir /workspace/zap/data/MileBench \
                    --dataset "${task}" \
                    --result-dir "${OUTPUT_DIR}"
            fi
        done
    } 2>&1 | tee -a "${RUN_LOG}"
else
    exec "${ENV_PYTHON}" "${THIS_DIR}/milebench_zap_student_onevision.py" \
        --keep_ratio "${KEEP_RATIO}" \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda:0 \
        --student_path "${STUDENT_PATH}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        "${COMBINE_ARGS[@]}" \
        "${LIMIT_ARGS[@]}" \
        "${OVERWRITE_ARGS[@]}" \
        "${NO_SCORE_ARGS[@]}" \
        2>&1 | tee -a "${RUN_LOG}"
fi
