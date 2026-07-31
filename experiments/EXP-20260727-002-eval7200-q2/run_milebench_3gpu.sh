#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

exp_dir=/workspace/nips/zap/experiments/EXP-20260727-002-eval7200-q2
output_dir=${OUTPUT_DIR:-/workspace/nips/zap/artifacts/rebuttal_eval7200_q2_llava15_keep0p2}
if [[ -d "${output_dir}" ]] && \
   find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing non-fresh OUTPUT_DIR: ${output_dir}" >&2
  echo "Choose a new OUTPUT_DIR; this launcher never mixes runs." >&2
  exit 1
fi
mkdir -p "${output_dir}/logs"

# TextNeedle and WikiVQA each contain many prompts that hit the 4096-token
# cap, so they are placed on separate GPUs. MMCoQA is the third heaviest text
# task. Every physical GPU runs exactly one model process and iterates its
# assigned tasks sequentially, avoiding both GPU OOM and output collisions.
gpu0_tasks=(
  TextNeedleInAHaystack
  ActionPrediction
  CharacterOrder
  MovingAttribute
  ObjectExistence
  MultiModalQA
  SlideVQA
  ALFRED
  CLEVR-Change
  nuscenes
)
gpu1_tasks=(
  WikiVQA
  ActionLocalization
  ObjectShuffle
  CounterfactualInference
  MovingDirection
  OCR-VQA
  TQA
  IEdit
  Spot-the-Diff
  GPR1200
)
gpu2_tasks=(
  MMCoQA
  EgocentricNavigation
  ObjectInteraction
  StateChange
  ActionSequence
  ImageNeedleInAHaystack
  SceneTransition
  DocVQA
  WebQA
)

for gpu in 0 1 2; do
  declare -n tasks="gpu${gpu}_tasks"
  session="eval7200_mb_gpu${gpu}"
  log="${output_dir}/logs/gpu${gpu}_milebench.log"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "screen session already exists: ${session}" >&2
    exit 1
  fi
  screen -dmS "${session}" bash -lc \
    "OUTPUT_DIR='${output_dir}' MILEBENCH_MAX_NEW_TOKENS=128 bash '${exp_dir}/run_worker.sh' '${gpu}' ${tasks[*]} >'${log}' 2>&1"
  echo "launched ${session}: ${tasks[*]}"
done

echo "MileBench jobs write to ${output_dir}."
echo "Run summary only after all three screen sessions finish."
