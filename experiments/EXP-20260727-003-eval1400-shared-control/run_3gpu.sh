#!/usr/bin/env bash
set -euo pipefail

exp_dir=/workspace/nips/zap/experiments/EXP-20260727-003-eval1400-shared-control
output_dir=${OUTPUT_DIR:-/workspace/nips/zap/artifacts/rebuttal_eval1400_shared_control_llava15_total0p2}

# This launcher is intentionally fresh-only.  Resume an interrupted worker by
# rerunning its exact run_worker.sh command against the existing output.
if [[ -e "$output_dir" ]]; then
  echo "Refusing non-fresh OUTPUT_DIR: $output_dir" >&2
  exit 2
fi

for session in shared1400_gpu0 shared1400_gpu1 shared1400_gpu2 shared1400_summary; do
  if screen -ls 2>/dev/null | grep -q "[.]$session"; then
    echo "Refusing to reuse live screen session: $session" >&2
    exit 2
  fi
done

# Query all target devices in one snapshot immediately before launch.
gpu_snapshot=$(nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits)
for physical_gpu in 0 1 2; do
  row=$(awk -F',' -v target="$physical_gpu" \
    '$1 + 0 == target {gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3}' \
    <<<"$gpu_snapshot")
  if [[ -z "$row" ]]; then
    echo "GPU $physical_gpu is not visible; refusing launch" >&2
    exit 2
  fi
  read -r memory_used utilization <<<"$row"
  if (( memory_used > 512 || utilization > 10 )); then
    echo "GPU $physical_gpu is occupied: memory=${memory_used}MiB util=${utilization}%" >&2
    exit 2
  fi
done

mkdir -p "$output_dir/logs" "$output_dir/status"

# Measured-runtime balancing on physical GPUs 0, 1, and 2 only. GPU 3 is
# explicitly outside the authorized target set.
screen -dmS shared1400_gpu0 \
  "$exp_dir/run_worker.sh" gpu0 0 "$output_dir" coco_caption docvqa
screen -dmS shared1400_gpu1 \
  "$exp_dir/run_worker.sh" gpu1 1 "$output_dir" textcaps chartqa
screen -dmS shared1400_gpu2 \
  "$exp_dir/run_worker.sh" gpu2 2 "$output_dir" nocaps gqa textvqa
screen -dmS shared1400_summary \
  "$exp_dir/finalize_after_workers.sh" "$output_dir"

echo "Launched shared1400_gpu0, shared1400_gpu1, and shared1400_gpu2."
echo "shared1400_summary will run one summarize-only pass after all three succeed."
echo "GPU 3 is not used. Output: $output_dir"
