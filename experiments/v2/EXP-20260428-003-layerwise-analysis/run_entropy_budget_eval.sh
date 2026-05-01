#!/bin/bash
# Sub-exp D: Entropy-based budget allocation vs Uniform — ChartQA
# Compares entropy_budget=True vs uniform at keep_ratio = 0.1, 0.2, 0.3
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash run_entropy_budget_eval.sh

set -e
cd /workspace/zap

WORK_DIR="experiments/EXP-20260428-003-layerwise-analysis/outputs/entropy_eval"
mkdir -p "$WORK_DIR"

run_eval() {
    local config=$1
    local out_dir=$2
    echo "[run] config=$config → $out_dir"
    python3 scripts/run_vlmeval_student.py -- \
        --config "$config" \
        --work-dir "$out_dir" \
        2>&1 | tee "${out_dir}/run.log"
}

# Entropy budget
run_eval eval_configs/onevision_entropy_10_chartqa.json "${WORK_DIR}/entropy10"
run_eval eval_configs/onevision_entropy_20_chartqa.json "${WORK_DIR}/entropy20"
run_eval eval_configs/onevision_entropy_30_chartqa.json "${WORK_DIR}/entropy30"

# Uniform baseline
run_eval eval_configs/onevision_student10_chartqa.json  "${WORK_DIR}/uniform10"
run_eval eval_configs/onevision_student20_chartqa.json  "${WORK_DIR}/uniform20"
run_eval eval_configs/onevision_student30_chartqa.json  "${WORK_DIR}/uniform30"

echo "All runs done. Results in $WORK_DIR"
