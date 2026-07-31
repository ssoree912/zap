#!/usr/bin/env bash
set -euo pipefail

ZAP_ROOT="/workspace/nips/zap"
EXP_DIR="${ZAP_ROOT}/experiments/EXP-20260502-024-llava15-original-teacher-extract"
PYTHON="/workspace/nips/.conda/envs/qvik/bin/python"
MODEL="/workspace/nips/models/llava-v1.5-7b"
TEACHER_ROOT="/workspace/nips/data/train/teacher/zap_llava15_correct_n600_seed0"
STUDENT_DIR="${ZAP_ROOT}/artifacts/original_llava_teacher/student_llava15_correct_qa1200_e15_gpu2"
EVAL_OUT="${EXP_DIR}/outputs/correct_qa_n600_vqa/chartqa_keep02"
LOG_DIR="${EXP_DIR}/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/correct_qa_n600_gpu2_train_eval_${RUN_STAMP}.log"
GQA_EXTRACT_LOG="${LOG_DIR}/correct_qa_n600_gqa_extract_gpu1_${RUN_STAMP}.log"
SCIENCEQA_EXTRACT_LOG="${LOG_DIR}/correct_qa_n600_scienceqa_extract_gpu2_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}" "${TEACHER_ROOT}" "${STUDENT_DIR}" "${EVAL_OUT}/keep_stats"
exec >>"${MAIN_LOG}" 2>&1

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

echo "[run] extraction_gpus=1,2 training_gpu=2 inference_gpu=2"
echo "[run] main_log=${MAIN_LOG}"
echo "[run] teacher=${TEACHER_ROOT}"
echo "[run] student=${STUDENT_DIR}"
echo "[run] eval=${EVAL_OUT}"

echo "[extract] GPU 1: GQA, 600 correct predictions"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${EXP_DIR}/collect_original_llava15_teacher.py" \
  --model-path "${MODEL}" \
  --model-name llava-v1.5-7b \
  --conv-template vicuna_v1 \
  --device cuda:0 \
  --device-map cuda:0 \
  --datasets gqa \
  --n-samples 600 \
  --max-candidates 0 \
  --max-new-tokens 64 \
  --seed 0 \
  --question-weight 0.5 \
  --require-correct \
  --sample-manifest-root "" \
  --gqa-questions-json /workspace/nips/data/train/gqa/val_balanced_questions.json \
  --gqa-images-root /workspace/nips/data/train/gqa/images \
  --output-root "${TEACHER_ROOT}" \
  >"${GQA_EXTRACT_LOG}" 2>&1 &
gqa_pid=$!

echo "[extract] GPU 2: ScienceQA, 600 correct predictions"
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${EXP_DIR}/collect_original_llava15_teacher.py" \
  --model-path "${MODEL}" \
  --model-name llava-v1.5-7b \
  --conv-template vicuna_v1 \
  --device cuda:0 \
  --device-map cuda:0 \
  --datasets scienceqa \
  --n-samples 600 \
  --max-candidates 0 \
  --max-new-tokens 64 \
  --seed 0 \
  --question-weight 0.5 \
  --require-correct \
  --sample-manifest-root "" \
  --scienceqa-problems-json /workspace/nips/data/train/scienceqa/problems.json \
  --scienceqa-images-root /workspace/nips/data/train/scienceqa/images \
  --scienceqa-split train \
  --scienceqa-prompt-style direct_letter \
  --output-root "${TEACHER_ROOT}" \
  >"${SCIENCEQA_EXTRACT_LOG}" 2>&1 &
scienceqa_pid=$!

echo "[extract] gqa_pid=${gqa_pid} log=${GQA_EXTRACT_LOG}"
echo "[extract] scienceqa_pid=${scienceqa_pid} log=${SCIENCEQA_EXTRACT_LOG}"
gqa_status=0
scienceqa_status=0
wait "${gqa_pid}" || gqa_status=$?
wait "${scienceqa_pid}" || scienceqa_status=$?
if ((gqa_status != 0 || scienceqa_status != 0)); then
  echo "[error] extraction failed: gqa_status=${gqa_status} scienceqa_status=${scienceqa_status}"
  exit 1
fi

"${PYTHON}" - "${TEACHER_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
summaries = {}
for dataset in ("gqa", "scienceqa"):
    paths = sorted((root / dataset).glob("*.pt"))
    if len(paths) != 600:
        raise RuntimeError(f"{dataset}: expected 600 teacher files, found {len(paths)}")
    for path in paths:
        record = torch.load(path, weights_only=False, map_location="cpu")
        if record.get("prediction_correct") is not True:
            raise RuntimeError(f"{path}: teacher prediction is not marked correct")
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
        if record.get("teacher_signal") != "question_answer_normalized_mix":
            raise RuntimeError(f"{path}: incorrect teacher signal")
    summary_path = root / dataset / "_summary.json"
    summaries[dataset] = json.loads(summary_path.read_text())
(root / "_summary.json").write_text(json.dumps(summaries, indent=2))
print("[verify] 1,200 correct-answer teacher records passed")
PY

echo "[train] GPU 2: 1,200 records, 15 epochs"
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${EXP_DIR}/train_original_llava15_student.py" \
  --teacher-root "${TEACHER_ROOT}" \
  --datasets gqa scienceqa \
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

echo "[eval] GPU 2: ChartQA VQA, total keep_ratio=0.2"
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${EXP_DIR}/lmms_eval_original_llava15_local_run.py" \
  --include_path "${EXP_DIR}/tasks" \
  --model llava15_original_student \
  --model_args "pretrained=${MODEL},student_path=${STUDENT_DIR},keep_ratio=0.2,conv_template=vicuna_v1,model_name=llava-v1.5-7b,device=cuda:0,device_map=cuda:0,stats_output_dir=${EVAL_OUT}/keep_stats" \
  --tasks chartqa_local \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix zap_correct_qa_n600_keep02 \
  --output_path "${EVAL_OUT}"

echo "[complete] correct-teacher extraction, training, and ChartQA inference finished"
