#!/bin/bash
set -uo pipefail

cd /workspace/zap/look_rebuttal/onevision

RESULT_DIR=/workspace/zap/look_rebuttal/outputs/onevision_text_prior_h2o_0.1
KEEP_RATIO=0.1
MERGE=none

mkdir -p "${RESULT_DIR}/Video-MME" "${RESULT_DIR}/SEEDBench_video"

echo "=== START Video-MME $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/kv/bin/python generate_onevision.py \
  --annotation /workspace/zap/look_rebuttal/data_video/Video-MME/Video-MME.json \
  --output_dir "${RESULT_DIR}/Video-MME" \
  --image_keep_ratio ${KEEP_RATIO} --merge_strategy ${MERGE} \
  --max_new_tokens 32 --log_every 50
echo "=== DONE Video-MME $(date -Is) exit=$? ==="

echo "=== START SEEDBench_video $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/kv/bin/python generate_onevision.py \
  --annotation /workspace/zap/look_rebuttal/data_video/SEEDBench_video/SEEDBench_video.json \
  --output_dir "${RESULT_DIR}/SEEDBench_video" \
  --image_keep_ratio ${KEEP_RATIO} --merge_strategy ${MERGE} \
  --max_new_tokens 32 --log_every 50
echo "=== DONE SEEDBench_video $(date -Is) exit=$? ==="

echo "=== ALL RUNS COMPLETE $(date -Is) ==="
