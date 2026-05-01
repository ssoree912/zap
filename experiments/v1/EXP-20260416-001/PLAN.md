## Experiment Plan

**ID**: EXP-20260416-001
**Author**: ZAP Team
**Date**: 2026-04-16
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 목적

Oracle OOM 없는 조건에서 **full_cache / combine_probe / LOOK-M** 3가지 방법의 효율성을 재측정한다.

기존 common-69 효율성 측정(EXP-20260415-001)은 oracle을 포함하여 OOM 발생 샘플이 있었고, oracle 제외 기준으로 정리된 수치가 없었음. 본 실험은 oracle 없이 깔끔하게 3-way 비교를 제공한다.

출력:
- `per_sample_70_3methods.csv` — 샘플별 prefill_ms, tbt_ms, kv_cache_gib, peak_gpu_gib, r_eff_prompt
- `summary_70_3methods.csv` — 방법별 평균/std 요약

---

### 2. 고정 실행 조건

| 항목 | 값 |
|---|---|
| Conda env | `kv` |
| 모델 | `/workspace/zap/ckpts/llava-1.5-7b-hf` |
| Probe ckpt | `/workspace/zap/ckpts/image_probe_combined_v1/mlp` |
| Data root | `/workspace/zap/data/MileBench` |
| Keep ratio | `total_keep_ratio=0.20` |
| LOOK-M budget | `hh=0.10 + recent=0.10` (r_eff ≈ 0.20 동일) |
| Truncation | `--truncate_like_lookm` (max_context=4096, n_tokens_per_image=576) |
| 샘플 수 | **70개** (29개 데이터셋 × 2~3개씩, stratified) |
| Seed | 42 |
| max_new_tokens | 32 |

---

### 3. 포함/제외 방법

| 방법 | 포함 | 비고 |
|---|---|---|
| full_cache | ✓ | baseline (no compression) |
| combine_probe | ✓ | ZAP our method |
| LOOK-M | ✓ | prior work baseline |
| oracle | ✗ | OOM 위험으로 제외 |
| h2o_ablation | ✗ | 불필요 |

---

### 4. 샘플링 전략

- 전체 29개 데이터셋에서 각 2개씩 우선 샘플 (=58개)
- 나머지 12개는 다양성 확보를 위해 임의 선택 (다중 이미지 비중이 높은 데이터셋 우선)
- 매니페스트 파일: `artifacts/efficiency_70_3methods/manifest_70.json`

---

### 5. 실행 명령

```bash
screen -S eff_70_3methods
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n kv --no-capture-output \
bash /workspace/zap/experiments/EXP-20260416-001/run.sh
```

---

### 6. 성공 기준

- 3개 방법 모두 `n_samples >= 65` (70개 중 일부 OOM 허용)
- full_cache / probe / LOOK-M 비교 가능한 매칭 샘플 기준 TBT, prefill_ms, kv_cache_gib 산출
- `RESULTS.md` 작성

---

### 7. 체크리스트

- [x] PLAN.md 작성
- [ ] manifest_70.json 생성 확인
- [ ] run.sh 실행 시작
- [ ] per_sample CSV 확인
- [ ] RESULTS.md 작성
