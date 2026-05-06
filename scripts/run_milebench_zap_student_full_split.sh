#!/usr/bin/env bash
# ZAP student eval on full MileBench (28) at keep_ratio=0.1.
# 23 datasets at max_new_tokens=64; 5 long-form/needle datasets at 128.
# All predictions land in one OUTPUT_DIR; scoring runs once at the end over all 28.
set -euo pipefail

GPU="${GPU:-0}"
KEEP_RATIO="${KEEP_RATIO:-0.1}"
MAX_SHORT="${MAX_SHORT:-64}"
MAX_LONG="${MAX_LONG:-128}"

LONG_DATASETS="IEdit,Spot-the-Diff,ImageNeedleInAHaystack,TextNeedleInAHaystack,MMCoQA"
SHORT_DATASETS="ALFRED,ActionLocalization,ActionPrediction,ActionSequence,CLEVR-Change,CharacterOrder,CounterfactualInference,DocVQA,EgocentricNavigation,GPR1200,MovingAttribute,MovingDirection,MultiModalQA,OCR-VQA,ObjectExistence,ObjectInteraction,ObjectShuffle,SceneTransition,SlideVQA,StateChange,TQA,WebQA,WikiVQA"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZAP_ROOT="$(cd "${THIS_DIR}/.." && pwd)"

KEEP_TAG="$(printf '%s' "${KEEP_RATIO}" | tr '.' 'p')"
OUTPUT_DIR="${OUTPUT_DIR:-${ZAP_ROOT}/logs/milebench_zap_student_llava15_keep${KEEP_TAG}_split${MAX_SHORT}_${MAX_LONG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run.log"

ENV_PYTHON="/mnt/srv/home/dlpc.3842/look-m/.conda/lookm/bin/python"
SCRIPT="${THIS_DIR}/milebench_zap_student.py"
STUDENT_PATH="/mnt/srv/home/dlpc.3842/zap/artifacts/student_llava15_original_future_1800_lr1e4_15ep"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/mnt/srv/home/dlpc.3842/look-m/LLaVA-mix_merge_v1:/mnt/srv/home/dlpc.3842/look-m:/mnt/srv/home/dlpc.3842/zap:${PYTHONPATH:-}"

cd "${ZAP_ROOT}"

{
    echo "[launch] method=zap_student keep_ratio=${KEEP_RATIO} gpu=${GPU}"
    echo "[launch] short_max=${MAX_SHORT} long_max=${MAX_LONG}"
    echo "[launch] output_dir=${OUTPUT_DIR}"
    echo

    echo "=== STEP 1/3: 23 datasets @ max_new_tokens=${MAX_SHORT} ==="
    "${ENV_PYTHON}" "${SCRIPT}" \
        --keep_ratio "${KEEP_RATIO}" \
        --dataset "${SHORT_DATASETS}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda:0 \
        --student_path "${STUDENT_PATH}" \
        --max_new_tokens "${MAX_SHORT}" \
        --no-score \
        --overwrite

    echo
    echo "=== STEP 2/3: 5 datasets @ max_new_tokens=${MAX_LONG} ==="
    "${ENV_PYTHON}" "${SCRIPT}" \
        --keep_ratio "${KEEP_RATIO}" \
        --dataset "${LONG_DATASETS}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda:0 \
        --student_path "${STUDENT_PATH}" \
        --max_new_tokens "${MAX_LONG}" \
        --no-score \
        --overwrite

    echo
    echo "=== STEP 3/3: scoring all 28 datasets ==="
    "${ENV_PYTHON}" -c "
import sys
sys.path.insert(0, '/mnt/srv/home/dlpc.3842/look-m')
sys.path.insert(0, '/mnt/srv/home/dlpc.3842/zap')
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location('milebench_zap_student', '${SCRIPT}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.score(Path('${OUTPUT_DIR}'), mod.ALL_MILEBENCH)
print('[done] _summary.json written for all 28 datasets')
"
} 2>&1 | tee "${LOG_FILE}"
