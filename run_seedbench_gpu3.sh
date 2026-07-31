#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="/workspace/nips/zap:/workspace/nips/Q-ViK:/workspace/nips/VFlowOpt:/workspace/nips/VFlowOpt/src/lmms_eval-0.2.4:/workspace/nips/VFlowOpt/src/transformers-4.46.0/src"
# The HF `datasets` loader must materialize parquet into an Arrow build cache
# before it can iterate. Its default location (~/.cache) is on the full root
# overlay, so point it at nips/zap/data, which lives on the roomy sdb1 volume.
export HF_DATASETS_CACHE="/workspace/nips/zap/data/hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"

RESULT_DIR="/workspace/nips/zap/results/lmms_zap_onevision_keep0.1/seedbench_local/full"
TASK_DIR="/workspace/nips/zap/onevision_zap/lmms_tasks/seedbench_local"

mkdir -p "${RESULT_DIR}"

LIMIT_ARG=""
if [[ "${SMOKE_LIMIT:-}" != "" ]]; then
  LIMIT_ARG="--limit ${SMOKE_LIMIT}"
fi

exec /opt/conda/envs/VFlowOpt/bin/python -u -m onevision_zap.run_lmms \
  --model zap_onevision_student \
  --model_args "pretrained=/workspace/nips/Q-ViK/model/llava-onevision-qwen2-7b-ov,student_path=/workspace/nips/zap/student_onevision_answer_n1800_e15_seed0,image_keep_ratio=0.1,max_frames_num=32,conv_template=qwen_1_5" \
  --tasks seedbench_local \
  --include_path "${TASK_DIR}" \
  --batch_size 1 \
  --device cuda:0 \
  ${LIMIT_ARG} \
  --log_samples \
  --output_path "${RESULT_DIR}" \
  --verbosity INFO
