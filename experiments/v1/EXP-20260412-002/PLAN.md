## Experiment Plan

**ID**: EXP-20260412-002
**Date**: 2026-04-12
**Status**: [ ] Planned  [ ] Running  [ ] Done  [ ] Abandoned

---

### 제목
연산량 측정 재실행 — 전체 데이터셋 기준, probe vs LOOK-M, total_keep_ratio 통일

---

### 1. 동기

기존 효율성 측정(`efficiency_random20_combined_summary.csv`)은:
- 8개 데이터셋만 포함
- probe는 `probe_image_keep_ratio` 기준 (image-only ratio)
- LOOK-M은 `hh+recent=0.20` 기준
- **r_eff_prompt 기준이 다름 → 부당 비교**

전체 29개 데이터셋에서 `total_keep_ratio=0.20` 통일 기준으로 재측정.
측정 지표도 기존과 동일하게: Prefill, TBT, GPU 메모리, r_eff_prompt, evicted token 수.

---

### 2. 측정 대상 (method × ratio)

| Method | Ratio 기준 | 설명 |
|---|---|---|
| Full Cache (baseline) | — | 압축 없음 |
| Probe (mlp, total r=0.20) | total_keep_ratio=0.20 | 우리 방법 (배포 버전) |
| Oracle (total r=0.20) | total_keep_ratio=0.20 | 우리 방법 upper bound |
| LOOK-M (hh=0.10+rec=0.10) | ≈ total r=0.20 | 비교 대상 |

→ probe와 LOOK-M 모두 r_eff_prompt ≈ 0.20으로 통일

---

### 3. 측정 지표

- **Prefill latency (ms)**: 첫 token 생성까지 시간
- **TBT (ms/token)**: decode 속도 (Time Between Tokens)
- **Peak GPU memory (GiB)**
- **r_eff_prompt**: 실제 유지된 전체 token 비율
- **n_image_kept / n_image_total**: evicted image token 수 (per sample)
- **n_text_kept**: 보존된 text token 수 (probe는 항상 100%)

---

### 4. 고정 조건

- 데이터셋: MileBench 전체 29개
- sample_size: 20 per dataset (기존 방식 유지)
- seed: 42
- 모델: LLaVA-1.5-7B
- GPU: 0 (probe), 1 (LOOK-M) — 동시 실행 가능

---

### 5. 실행 계획

```bash
# GPU 0: probe + oracle + full_cache
screen -dmS eff_probe bash -c "
source $(conda info --base)/etc/profile.d/conda.sh && conda activate kv
cd /workspace/zap
CUDA_VISIBLE_DEVICES=0 python scripts/measure_milebench_efficiency.py \
  --datasets ALFRED ActionLocalization ActionPrediction ActionSequence \
             CLEVR-Change CharacterOrder CounterfactualInference DocVQA \
             EgocentricNavigation GPR1200 IEdit ImageNeedleInAHaystack \
             MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA \
             ObjectExistence ObjectInteraction ObjectShuffle SceneTransition \
             SlideVQA Spot-the-Diff StateChange TQA TextNeedleInAHaystack \
             WebQA WikiVQA nuscenes \
  --total_keep_ratio 0.20 \
  --include_zap --include_oracle \
  --output_dir /workspace/hd/artifacts/efficiency_global \
  2>&1 | tee /workspace/hd/artifacts/efficiency_global/eff_probe.log
"

# GPU 1: LOOK-M
screen -dmS eff_lookm bash -c "
source $(conda info --base)/etc/profile.d/conda.sh && conda activate kv
cd /workspace/zap
CUDA_VISIBLE_DEVICES=1 python scripts/measure_milebench_efficiency.py \
  --datasets [전체 29개] \
  --total_keep_ratio 0.20 \
  --include_lookm \
  --look_hh_ratio 0.10 --look_recent_ratio 0.10 \
  --output_dir /workspace/hd/artifacts/efficiency_global \
  2>&1 | tee /workspace/hd/artifacts/efficiency_global/eff_lookm.log
"
```

---

### 6. 예상 결과

- TBT: probe < LOOK-M (2~3× 빠를 것, 기존 28 vs 78ms/tok 경향 유지)
- Prefill: probe ≈ LOOK-M (probe forward 포함해도 유사)
- GPU 메모리: probe ≤ LOOK-M
- r_eff_prompt: 둘 다 ≈ 0.20 (통일됨)

---

### 7. 아티팩트 위치

`/workspace/hd/artifacts/efficiency_global/`
- `summary.csv`: 전체 요약
- `per_sample.csv`: 샘플별 상세

---

### 8. 예상 런타임

~2시간 (20 samples × 29 datasets × 3 methods, GPU 2개 병렬)
