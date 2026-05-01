#!/usr/bin/env bash
# EXP-20260416-001: full_cache / combine_probe / LOOK-M 3-way efficiency
# 70 samples, stratified across 29 MileBench datasets
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MILEBENCH_ROOT=/workspace/zap/data/MileBench
OUT_DIR=/workspace/zap/artifacts/efficiency_70_3methods
MANIFEST=$OUT_DIR/manifest_70.json
MODEL=/workspace/zap/ckpts/llava-1.5-7b-hf
PROBE=/workspace/zap/ckpts/image_probe_combined_v1/mlp
LOG=$OUT_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log

mkdir -p "$OUT_DIR"

DATASETS=(
  ALFRED ActionLocalization ActionPrediction ActionSequence
  CLEVR-Change CharacterOrder CounterfactualInference DocVQA
  EgocentricNavigation GPR1200 IEdit ImageNeedleInAHaystack
  MMCoQA MovingAttribute MovingDirection MultiModalQA
  OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle
  SceneTransition SlideVQA Spot-the-Diff StateChange
  TQA TextNeedleInAHaystack WebQA WikiVQA nuscenes
)

echo "[$(date -u +%H:%M:%SZ)] Step 1: stratified manifest 생성 (70개)" | tee -a "$LOG"

/opt/conda/envs/kv/bin/python - <<'PYEOF'
import json, random, sys
from pathlib import Path

milebench_root = Path("/workspace/zap/data/MileBench")
out_path = Path("/workspace/zap/artifacts/efficiency_70_3methods/manifest_70.json")
seed = 42
target_total = 70

datasets = [
    "ALFRED","ActionLocalization","ActionPrediction","ActionSequence",
    "CLEVR-Change","CharacterOrder","CounterfactualInference","DocVQA",
    "EgocentricNavigation","GPR1200","IEdit","ImageNeedleInAHaystack",
    "MMCoQA","MovingAttribute","MovingDirection","MultiModalQA",
    "OCR-VQA","ObjectExistence","ObjectInteraction","ObjectShuffle",
    "SceneTransition","SlideVQA","Spot-the-Diff","StateChange",
    "TQA","TextNeedleInAHaystack","WebQA","WikiVQA","nuscenes",
]

rng = random.Random(seed)
manifest = []
pool_extra = []

# 1단계: 각 데이터셋에서 2개씩
for ds in datasets:
    json_path = milebench_root / ds / f"{ds}.json"
    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        print(f"  [SKIP] {ds}: {e}", file=sys.stderr)
        continue
    samples = data.get("data", [])
    if not samples:
        print(f"  [SKIP] {ds}: no samples", file=sys.stderr)
        continue
    chosen = rng.sample(samples, min(2, len(samples)))
    for s in chosen:
        manifest.append({"dataset": ds, "sample_id": str(s["sample_id"])})
    # 나머지를 extra pool에
    remaining = [s for s in samples if str(s["sample_id"]) not in {str(c["sample_id"]) for c in chosen}]
    for s in remaining:
        pool_extra.append({"dataset": ds, "sample_id": str(s["sample_id"])})

# 2단계: 부족분 보충 (target 70)
n_extra = target_total - len(manifest)
if n_extra > 0 and pool_extra:
    extras = rng.sample(pool_extra, min(n_extra, len(pool_extra)))
    manifest.extend(extras)

manifest.sort(key=lambda x: (x["dataset"], x["sample_id"]))

# 데이터셋별 분포 출력
from collections import Counter
dist = Counter(r["dataset"] for r in manifest)
print(f"\n[manifest] 총 {len(manifest)}개 샘플, {len(dist)}개 데이터셋", file=sys.stderr)
for ds, n in sorted(dist.items()):
    print(f"  {ds}: {n}", file=sys.stderr)

out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"\n저장: {out_path}", file=sys.stderr)
PYEOF

echo "[$(date -u +%H:%M:%SZ)] Step 2: efficiency 측정 시작" | tee -a "$LOG"

/opt/conda/envs/kv/bin/python /workspace/zap/scripts/measure_milebench_efficiency.py \
  --milebench_root  "$MILEBENCH_ROOT" \
  --datasets        "${DATASETS[@]}" \
  --sample_manifest_path "$MANIFEST" \
  --output_dir      "$OUT_DIR" \
  --implementation_model_name "$MODEL" \
  --probe_model_name "$PROBE" \
  --device          cuda:0 \
  --total_keep_ratio 0.20 \
  --look_hh_ratio   0.10 \
  --look_recent_ratio 0.10 \
  --truncate_like_lookm \
  --max_new_tokens  32 \
  --include_zap \
  --include_lookm \
  --no-include_oracle \
  --no-include_h2o_ablation \
  --no-include_oracle_all_token \
  2>&1 | tee -a "$LOG"

echo "[$(date -u +%H:%M:%SZ)] 완료. 결과: $OUT_DIR" | tee -a "$LOG"
