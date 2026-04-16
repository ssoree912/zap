#!/bin/bash
# LOOK-M efficiency measurement — run in look conda env
# Waits for ZAP run to finish (run_zap.log must contain "ZAP DONE"), then runs.

echo "[$(date +%H:%M:%S)] Waiting for ZAP to finish..."
until grep -q "ZAP DONE" /workspace/hd/artifacts/probe_global/efficiency_all_datasets/run_zap.log 2>/dev/null; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] ZAP done. Starting LOOK-M measurement..."

cd /workspace/zap
conda run -n look --no-capture-output python scripts/measure_milebench_efficiency_lookm.py \
  --manifest /workspace/hd/artifacts/probe_global/efficiency_all_datasets/sample_manifest.json \
  --output_dir /workspace/hd/artifacts/probe_global/efficiency_all_datasets \
  --milebench_root /workspace/hd/data/MileBench \
  --device cuda:0 \
  2>&1 | tee /workspace/hd/artifacts/probe_global/efficiency_all_datasets/run_lookm.log

echo "[$(date +%H:%M:%S)] LOOK-M DONE"
