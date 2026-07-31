#!/usr/bin/env bash
set -euo pipefail

readonly ZAP_ROOT="/workspace/nips/zap"
readonly EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260727-007-full-vqa-pa-qa-paired"
readonly LOCAL_EXP="${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract"
readonly PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
readonly MODEL="/workspace/nips/models/llava-v1.5-7b"
readonly PA_STUDENT="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_prefill_answer50_paired_1800_e15_seed0"
readonly QA_STUDENT="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_question_answer50_paired_1800_e15_seed0"

if (($# < 2)); then
  echo "usage: $0 PHYSICAL_GPU LABEL:TASK [LABEL:TASK ...]" >&2
  exit 2
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must point to a prepared run directory" >&2
  exit 2
fi
if [[ ! -f "${RUN_ROOT}/run_manifest.json" ]]; then
  echo "Prepared run manifest is missing: ${RUN_ROOT}/run_manifest.json" >&2
  exit 2
fi

readonly physical_gpu="$1"
shift
case "${physical_gpu}" in
  0|1|2|3) ;;
  *)
    echo "Physical GPU must be one of 0, 1, 2, or 3; got ${physical_gpu}" >&2
    exit 2
    ;;
esac

readonly worker_log="${RUN_ROOT}/logs/worker_gpu${physical_gpu}.log"
readonly worker_failed="${RUN_ROOT}/state/worker_gpu${physical_gpu}.failed"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/outputs" "${RUN_ROOT}/state"
exec >>"${worker_log}" 2>&1

current_job="worker_setup"
worker_exit() {
  local status=$?
  if ((status != 0)); then
    printf 'physical_gpu=%s\njob=%s\nexit_code=%s\n' \
      "${physical_gpu}" "${current_job}" "${status}" >"${worker_failed}"
  fi
}
trap worker_exit EXIT

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/nips/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/nips/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export LMMS_LOCAL_EVAL_ROOT=/workspace/nips/data/eval
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export QVIK_ROOT=/workspace/nips/Q-ViK
export ZAP_REPO_ROOT="${ZAP_ROOT}"

cd "${ZAP_ROOT}"
echo "[worker] physical_gpu=${physical_gpu} run_root=${RUN_ROOT}"
echo "[worker] local_dataset_root=${LMMS_LOCAL_EVAL_ROOT}"

for job in "$@"; do
  if [[ "${job}" != *:* ]]; then
    echo "Invalid job ${job}; expected LABEL:TASK" >&2
    exit 2
  fi
  label="${job%%:*}"
  task="${job#*:}"
  current_job="${label}:${task}"

  case "${label}" in
    prefill_answer_50_50) student="${PA_STUDENT}" ;;
    question_answer_50_50) student="${QA_STUDENT}" ;;
    *)
      echo "Unknown student label: ${label}" >&2
      exit 2
      ;;
  esac
  case "${task}" in
    gqa_local|textvqa_local|docvqa_local|chartqa_local) ;;
    *)
      echo "Unknown or out-of-scope task: ${task}" >&2
      exit 2
      ;;
  esac

  out_dir="${RUN_ROOT}/outputs/${label}/${task}"
  stats_dir="${out_dir}/keep_stats"
  done_file="${RUN_ROOT}/state/${label}__${task}.done"
  failed_file="${RUN_ROOT}/state/${label}__${task}.failed"
  if [[ -e "${done_file}" || -e "${failed_file}" || -e "${out_dir}" ]]; then
    echo "Refusing to overwrite existing job state/output for ${current_job}" >&2
    exit 2
  fi
  mkdir -p "${stats_dir}"

  echo "[eval] start physical_gpu=${physical_gpu} label=${label} task=${task}"
  set +e
  "${PYTHON}" "${LOCAL_EXP}/lmms_eval_original_llava15_local_run.py" \
    --include_path "${LOCAL_EXP}/tasks" \
    --model llava15_original_student \
    --model_args "pretrained=${MODEL},student_path=${student},keep_ratio=0.2,keep_budget_mode=exact_total_ceil,student_failure_policy=raise,conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,stats_output_dir=${stats_dir}" \
    --tasks "${task}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${label}__${task}__exact_total0p2" \
    --output_path "${out_dir}"
  status=$?
  set -e
  if ((status != 0)); then
    printf 'physical_gpu=%s\nlabel=%s\ntask=%s\nexit_code=%s\n' \
      "${physical_gpu}" "${label}" "${task}" "${status}" >"${failed_file}"
    echo "[eval] failed label=${label} task=${task} exit_code=${status}" >&2
    exit "${status}"
  fi
  printf 'physical_gpu=%s\nlabel=%s\ntask=%s\ncompleted_at=%s\n' \
    "${physical_gpu}" "${label}" "${task}" "$(date --iso-8601=seconds)" >"${done_file}"
  echo "[eval] done label=${label} task=${task}"
done

current_job="worker_complete"
echo "[worker] complete physical_gpu=${physical_gpu}"
