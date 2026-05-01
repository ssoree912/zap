#!/bin/bash
# Entropy budget vs Uniform — ChartQA, keep_ratio = 0.1 and 0.05 only
set -e
cd /workspace/zap

OUT="experiments/EXP-20260428-003-layerwise-analysis/outputs/entropy_eval"
mkdir -p "$OUT"/{entropy10,entropy05,uniform05}

run() {
    echo "[run] $1 → $2"
    python3 scripts/run_vlmeval_student.py -- --config "$1" --work-dir "$2" 2>&1 | tee "$2/run.log"
}

run eval_configs/onevision_entropy_10_chartqa.json "$OUT/entropy10"
run eval_configs/onevision_entropy_05_chartqa.json "$OUT/entropy05"

echo "Done."
