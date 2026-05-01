#!/bin/bash
# Stage 3 sweep: 3 scope × 3 keep_ratio × 2 dataset = 18 student runs + 2 full baselines.
# Each GPU handles one scope across all (keep, dataset) combos. Logs in artifacts dir.
set -uo pipefail

OUT=/workspace/zap/artifacts/EXP-20260425-002
mkdir -p "$OUT"

SCOPES=(A Bprime B)
GPUS=(0 1 2)
KEEPS=(0.3 0.5 0.7)
DATASETS=("mm-vet:/workspace/data/mm-vet/mm-vet.json:/workspace/data/mm-vet:218"
          "detail_1k:/workspace/data/detail_1k.json:/workspace/data:1000")

run_eval() {
  local gpu=$1 method=$2 outdir=$3
  shift 3
  CUDA_VISIBLE_DEVICES=$gpu python /workspace/zap/eval_ppl.py \
    --method "$method" --output-dir "$outdir" "$@" \
    >"$outdir/eval.log" 2>&1
}

for i in 0 1 2; do
  scope=${SCOPES[$i]}
  gpu=${GPUS[$i]}
  ckpt=/workspace/zap/ckpts/student_v2_${scope}
  (
    for ds_spec in "${DATASETS[@]}"; do
      IFS=":" read -r dsname dspath imgpath nsamp <<<"$ds_spec"
      for keep in "${KEEPS[@]}"; do
        outdir="$OUT/${dsname}/${scope}_k${keep}"
        mkdir -p "$outdir"
        echo "[gpu $gpu] $scope $dsname k=$keep starting"
        run_eval "$gpu" visual_utility_student "$outdir" \
          --student-model-name "$ckpt" \
          --image-keep-ratio "$keep" \
          --data-path "$dspath" \
          --image-path "$imgpath" \
          --eval-samples "$nsamp"
        echo "[gpu $gpu] $scope $dsname k=$keep done"
      done
    done
  ) &
done

wait
echo "[done] all student eval runs finished. Run full baselines separately."
