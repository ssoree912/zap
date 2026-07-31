#!/usr/bin/env bash
set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract"
PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
MODEL="/workspace/nips/models/llava-v1.5-7b"
STUDENT="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_correct_qa1200_e15_gpu2"
OUT_ROOT="${EXP_DIR}/outputs/correct_qa_n600_vqa"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/correct_qa_docvqa_textvqa_gpu2_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
exec >>"${MAIN_LOG}" 2>&1

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/nips/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/nips/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export QVIK_ROOT=/workspace/nips/Q-ViK
export ZAP_REPO_ROOT="${ZAP_ROOT}"

cd "${ZAP_ROOT}"

run_task() {
  local task="$1"
  local output_name="$2"
  local out_dir="${OUT_ROOT}/${output_name}_keep02"

  mkdir -p "${out_dir}/keep_stats"
  echo "[eval] physical_gpu=2 task=${task} output=${out_dir}"
  "${PYTHON}" "${EXP_DIR}/lmms_eval_original_llava15_local_run.py" \
    --include_path "${EXP_DIR}/tasks" \
    --model llava15_original_student \
    --model_args "pretrained=${MODEL},student_path=${STUDENT},keep_ratio=0.2,conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,stats_output_dir=${out_dir}/keep_stats" \
    --tasks "${task}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "zap_correct_qa_n600_${output_name}_keep02" \
    --output_path "${out_dir}"
  echo "[done] task=${task}"
}

echo "[run] main_log=${MAIN_LOG}"
echo "[run] student=${STUDENT}"
run_task docvqa_local docvqa
run_task textvqa_local textvqa
echo "[complete] DocVQA and TextVQA inference finished"
