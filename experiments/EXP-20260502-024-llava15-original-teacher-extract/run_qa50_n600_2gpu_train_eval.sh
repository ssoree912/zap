#!/usr/bin/env bash
set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract"
PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
MODEL="/workspace/nips/models/llava-v1.5-7b"
SAMPLE_MANIFEST="/workspace/nips/data/train/teacher/zap_llava15_n600_seed0"
TEACHER_ROOT="/workspace/nips/data/train/teacher/zap_llava15_qa50_n600_seed0"
STUDENT_DIR="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_qa50_1800_e15_gpu0"
EVAL_OUT="${EXP_DIR}/outputs/question_answer_n600_vqa/chartqa_keep02"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/qa50_n600_2gpu_train_eval_${RUN_STAMP}.log"
GPU0_EXTRACT_LOG="${LOG_DIR}/qa50_n600_extract_gpu0_${RUN_STAMP}.log"
GPU1_EXTRACT_LOG="${LOG_DIR}/qa50_n600_extract_gpu1_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}" "${TEACHER_ROOT}" "${STUDENT_DIR}" "${EVAL_OUT}/keep_stats"
exec >>"${MAIN_LOG}" 2>&1

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
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

echo "[run] main_log=${MAIN_LOG}"
echo "[run] teacher=${TEACHER_ROOT}"
echo "[run] student=${STUDENT_DIR}"
echo "[run] eval=${EVAL_OUT}"
echo "[extract] GPU 0: scienceqa (600)"
echo "[extract] GPU 1: gqa + textvqa (600 each)"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${EXP_DIR}/collect_original_llava15_teacher.py" \
  --model-path "${MODEL}" \
  --model-name llava-v1.5-7b \
  --conv-template vicuna_v1 \
  --device cuda:0 \
  --device-map cuda:0 \
  --datasets scienceqa \
  --n-samples 600 \
  --max-new-tokens 64 \
  --seed 0 \
  --question-weight 0.5 \
  --sample-manifest-root "${SAMPLE_MANIFEST}" \
  --output-root "${TEACHER_ROOT}" \
  >"${GPU0_EXTRACT_LOG}" 2>&1 &
gpu0_pid=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${EXP_DIR}/collect_original_llava15_teacher.py" \
  --model-path "${MODEL}" \
  --model-name llava-v1.5-7b \
  --conv-template vicuna_v1 \
  --device cuda:0 \
  --device-map cuda:0 \
  --datasets gqa textvqa \
  --n-samples 600 \
  --max-new-tokens 64 \
  --seed 0 \
  --question-weight 0.5 \
  --sample-manifest-root "${SAMPLE_MANIFEST}" \
  --output-root "${TEACHER_ROOT}" \
  >"${GPU1_EXTRACT_LOG}" 2>&1 &
gpu1_pid=$!

echo "[extract] gpu0_pid=${gpu0_pid} log=${GPU0_EXTRACT_LOG}"
echo "[extract] gpu1_pid=${gpu1_pid} log=${GPU1_EXTRACT_LOG}"

gpu0_status=0
gpu1_status=0
wait "${gpu0_pid}" || gpu0_status=$?
wait "${gpu1_pid}" || gpu1_status=$?
if ((gpu0_status != 0 || gpu1_status != 0)); then
  echo "[error] extraction failed: gpu0_status=${gpu0_status} gpu1_status=${gpu1_status}"
  exit 1
fi

"${PYTHON}" - "${TEACHER_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
summaries = {}
for dataset in ("gqa", "textvqa", "scienceqa"):
    paths = sorted((root / dataset).glob("*.pt"))
    if len(paths) != 600:
        raise RuntimeError(f"{dataset}: expected 600 teacher files, found {len(paths)}")
    for path in paths:
        record = torch.load(path, weights_only=False, map_location="cpu")
        teacher = record["teacher_norm"].float()
        if tuple(teacher.shape) != (32, 576):
            raise RuntimeError(f"{path}: unexpected teacher shape {tuple(teacher.shape)}")
        if not torch.allclose(
            teacher.sum(dim=-1),
            torch.ones(32),
            atol=1e-3,
            rtol=0,
        ):
            raise RuntimeError(f"{path}: teacher distribution is not normalized")
        if record.get("teacher_question_weight") != 0.5:
            raise RuntimeError(f"{path}: incorrect question weight")
        if record.get("teacher_answer_weight") != 0.5:
            raise RuntimeError(f"{path}: incorrect answer weight")
        if record.get("teacher_signal") != "question_answer_normalized_mix":
            raise RuntimeError(f"{path}: incorrect teacher signal")
    summary_path = root / dataset / "_summary.json"
    summaries[dataset] = json.loads(summary_path.read_text())
(root / "_summary.json").write_text(json.dumps(summaries, indent=2))
print("[verify] 1,800 question+answer teacher records passed")
PY

echo "[train] GPU 0: 1,800 records, 15 epochs"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${EXP_DIR}/train_original_llava15_student.py" \
  --teacher-root "${TEACHER_ROOT}" \
  --datasets gqa textvqa scienceqa \
  --llava-path "${MODEL}" \
  --model-name llava-v1.5-7b \
  --vision-tower-path "" \
  --device cuda:0 \
  --device-map cuda:0 \
  --epochs 15 \
  --lr 1e-4 \
  --n-per-dataset 600 \
  --seed 0 \
  --output-dir "${STUDENT_DIR}"

echo "[eval] GPU 0: ChartQA VQA, total keep_ratio=0.2"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${EXP_DIR}/lmms_eval_original_llava15_local_run.py" \
  --include_path "${EXP_DIR}/tasks" \
  --model llava15_original_student \
  --model_args "pretrained=${MODEL},student_path=${STUDENT_DIR},keep_ratio=0.2,conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,stats_output_dir=${EVAL_OUT}/keep_stats" \
  --tasks chartqa_local \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix zap_qa50_n600_keep02 \
  --output_path "${EVAL_OUT}"

echo "[complete] extraction, training, and ChartQA VQA inference finished"
