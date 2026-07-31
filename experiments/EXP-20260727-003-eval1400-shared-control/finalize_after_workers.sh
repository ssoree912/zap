#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
exp_dir=/workspace/nips/zap/experiments/EXP-20260727-003-eval1400-shared-control
python_bin=/workspace/nips/.conda/envs/qvik/bin/python
runner=$exp_dir/run_eval1400_shared_control.py
status_dir=$output_dir/status
workers=(gpu0 gpu1 gpu2)

while true; do
  all_complete=1
  for worker in "${workers[@]}"; do
    if [[ -f "$status_dir/$worker.failed" ]]; then
      echo "Refusing summary because $worker failed; see $output_dir/logs/$worker.log" \
        >"$output_dir/logs/summary.log"
      touch "$status_dir/summary.failed"
      exit 1
    fi
    if [[ ! -f "$status_dir/$worker.ok" ]]; then
      all_complete=0
    fi
  done
  if [[ "$all_complete" -eq 1 ]]; then
    break
  fi
  sleep 15
done

cd /workspace/nips/zap
"$python_bin" "$runner" \
  --output-dir "$output_dir" \
  --datasets gqa textvqa docvqa chartqa coco_caption nocaps textcaps \
  --n-samples 200 \
  --total-keep-ratio 0.2 \
  --summarize-only \
  >"$output_dir/logs/summary.log" 2>&1
touch "$status_dir/summary.ok"
