#!/usr/bin/env bash
set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
QVIK_ROOT="/workspace/nips/Q-ViK"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract"
PYTHON="/opt/conda/envs/qvik/bin/python"
MODEL="/workspace/nips/models/llava-v1.5-7b"
QVIK_STUDENT="${QVIK_ROOT}/ckpts/student_llava15_qa50_all_900_e15_gpu0"
ZAP_STUDENT="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_qa50_900_e15_gpu0"
QVIK_OUT="${QVIK_ROOT}/results/llava15_qa50_all"
ZAP_OUT="${EXP_DIR}/outputs/question_answer_chartqa_keep01/zap"
LOG="${EXP_DIR}/logs/chartqa_qvik_zap_keep01_gpu0_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "${LOG}")" "${QVIK_OUT}" "${ZAP_OUT}/keep_stats"
exec >>"${LOG}" 2>&1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export QVIK_ROOT
export ZAP_REPO_ROOT="${ZAP_ROOT}"

echo "[qvik] ChartQA total keep_ratio=0.1"
cd "${QVIK_ROOT}"
"${PYTHON}" qvik/eval/run_lmms_eval.py \
  --model lmms_llava15_student \
  --model_args "pretrained=${MODEL},student_path=${QVIK_STUDENT},keep_ratio=0.1,keep_ratio_basis=total,device=cuda:0,device_map=cuda:0,attn_implementation=sdpa,max_new_tokens=32" \
  --tasks chartqa \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix qvik_qa50_keep01 \
  --output_path "${QVIK_OUT}"

echo "[zap] ChartQA total keep_ratio=0.1"
cd "${ZAP_ROOT}"
"${PYTHON}" "${EXP_DIR}/lmms_eval_original_llava15_local_run.py" \
  --include_path "${EXP_DIR}/tasks" \
  --model llava15_original_student \
  --model_args "pretrained=${MODEL},student_path=${ZAP_STUDENT},keep_ratio=0.1,conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,stats_output_dir=${ZAP_OUT}/keep_stats" \
  --tasks chartqa_local \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix zap_qa50_keep01 \
  --output_path "${ZAP_OUT}"

echo "[complete] Q-ViK and zap ChartQA keep_ratio=0.1 finished"
