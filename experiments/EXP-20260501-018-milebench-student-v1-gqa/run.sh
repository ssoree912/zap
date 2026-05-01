#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/workspace/zap/experiments/EXP-20260501-018-milebench-student-v1-gqa
DATA_ROOT=/workspace/zap/data/MileBench
KEEP_RATIO=0.5
OUTPUT_BASE=${EXP_DIR}/outputs/keep050
LOG_FILE=${EXP_DIR}/logs/run_$(date +%Y%m%d_%H%M%S).log
STUDENT=/workspace/zap/ckpts/v1/student_v2_A_gqa_lr1e4
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf

mkdir -p "${OUTPUT_BASE}" "${EXP_DIR}/logs"

SKIP_DIRS="images preview"

{
echo "=== EXP-20260501-018 START $(date -Iseconds) ==="
echo "[config] gpu=2 (CUDA_VISIBLE_DEVICES=2 → cuda:0)"
echo "[config] model=${MODEL}"
echo "[config] student=${STUDENT}"
echo "[config] keep_ratio=${KEEP_RATIO}"

for dataset_dir in "${DATA_ROOT}"/*/; do
    dataset_name=$(basename "${dataset_dir}")

    if echo "${SKIP_DIRS}" | grep -qw "${dataset_name}"; then
        continue
    fi

    json_path="${DATA_ROOT}/${dataset_name}/${dataset_name}.json"
    combined_root="${DATA_ROOT}/${dataset_name}/combined_1_images"
    image_root="${DATA_ROOT}/${dataset_name}/images"

    if [[ ! -f "${json_path}" ]]; then
        echo "[skip] ${dataset_name}: no json"
        continue
    fi

    # prefer combined_1_images (single-image, avoids OOM on multi-image datasets)
    if [[ -d "${combined_root}" ]]; then
        use_image_root="${combined_root}"
        use_image_column="combined_1_images"
    elif [[ -d "${image_root}" ]]; then
        use_image_root="${image_root}"
        use_image_column="images_path"
    else
        echo "[skip] ${dataset_name}: no images dir"
        continue
    fi

    out_dir="${OUTPUT_BASE}/${dataset_name}"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[skip] ${dataset_name}: already done"
        continue
    fi

    mkdir -p "${out_dir}"
    echo "[run] ${dataset_name} → ${out_dir}"

    echo "[config] ${dataset_name}: image_column=${use_image_column}"
    CUDA_VISIBLE_DEVICES=2 conda run -n kv python3 /workspace/zap/evaluate_image_teacher_pruning.py \
        --dataset_path "${json_path}" \
        --image_root "${use_image_root}" \
        --image_column "${use_image_column}" \
        --implementation_model_name "${MODEL}" \
        --student_model_name "${STUDENT}" \
        --total_keep_ratio "${KEEP_RATIO}" \
        --output_dir "${out_dir}" \
        --look_dataset_name "${dataset_name}" \
        --look_model_name "student_v1_gqa_keep050" \
        --device cuda:0

    echo "[done] ${dataset_name}"
done

echo "=== EXP-20260501-018 DONE $(date -Iseconds) ==="
} 2>&1 | tee "${LOG_FILE}"
