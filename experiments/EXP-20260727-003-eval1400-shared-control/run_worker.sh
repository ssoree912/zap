#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 WORKER_ID PHYSICAL_GPU OUTPUT_DIR DATASET..." >&2
  exit 2
fi

worker_id=$1
physical_gpu=$2
output_dir=$3
shift 3
datasets=("$@")

case "$physical_gpu" in
  0|1|2) ;;
  *)
    echo "Only physical GPUs 0, 1, and 2 are allowed; got $physical_gpu" >&2
    exit 2
    ;;
esac

exp_dir=/workspace/nips/zap/experiments/EXP-20260727-003-eval1400-shared-control
python_bin=/workspace/nips/.conda/envs/qvik/bin/python
runner=$exp_dir/run_eval1400_shared_control.py
status_dir=$output_dir/status
log_path=$output_dir/logs/$worker_id.log

mkdir -p "$status_dir" "$output_dir/logs"
rm -f "$status_dir/$worker_id.ok" "$status_dir/$worker_id.failed"

set +e
cd /workspace/nips/zap
CUDA_VISIBLE_DEVICES="$physical_gpu" "$python_bin" "$runner" \
  --device cuda:0 \
  --worker-id "$worker_id" \
  --output-dir "$output_dir" \
  --datasets "${datasets[@]}" \
  --n-samples 200 \
  --total-keep-ratio 0.2 \
  --extract-only \
  >"$log_path" 2>&1
exit_code=$?
set -e

if [[ "$exit_code" -eq 0 ]]; then
  touch "$status_dir/$worker_id.ok"
else
  printf '%s\n' "$exit_code" >"$status_dir/$worker_id.failed"
fi
exit "$exit_code"
