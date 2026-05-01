## Experiment Plan

**ID**: EXP-20260428-003  
**Author**: ssoree912  
**Date**: 2026-04-28  
**Status**: [ ] Planned  [ ] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)

현재 `ImageTokenTopKPress`는 모든 layer에 동일한 `image_keep_ratio`를 적용한다 (`k_l = r * N_I`, uniform).  
그러나 layer별 image token future utility 분포가 다를 수 있으며, 이를 무시하면 낮은 keep ratio에서 성능이 불필요하게 떨어질 수 있다.  
AirCache가 낮은 ratio에서 우리 방법보다 좋다면 layer-wise budget allocation이 원인일 수 있다.  
본 실험은 이를 4개 sub-exp로 체계적으로 검증한다.

---

### 2. 가설 (Hypothesis)

- **H1**: teacher score entropy가 layer별로 균일하지 않다 (일부 layer는 utility 집중, 일부는 분산)
- **H2**: student-teacher Spearman correlation이 layer별로 다르다 (불안정한 layer 존재)
- **H3**: 특정 layer만 aggressive prune 시 성능 하락이 크다 ("민감한 layer" 존재)
- **H4**: entropy 또는 sensitivity 기반 layer-wise budget allocation이 uniform보다 낮은 keep ratio에서 성능이 높다

---

### 3. 독립변수

| Sub-exp | 변수 |
|---------|------|
| A. Teacher Entropy | layer index → normalized entropy of `y^(l)` |
| B. Student-Teacher Alignment | layer index → Spearman / Overlap@K |
| C. Single-Layer Sensitivity | pruned layer index (해당 layer만 keep=0.1, 나머지 full) |
| D. Budget Allocation | strategy: uniform / entropy-prop / entropy-inv / sensitivity |

---

### 4. 종속변수

- **A/B**: normalized entropy per layer, Spearman(student, teacher) per layer, Overlap@K per layer
- **C**: accuracy delta vs full cache (ChartQA)
- **D**: ChartQA accuracy at keep_ratio = 0.1 / 0.2 / 0.3

---

### 5. 고정 조건

- 모델: LLaVA-OneVision
- 데이터셋: **ChartQA** (단일)
- Teacher score: 기존 수집된 onevision teacher shard 재사용
- Student probe: 기존 trained probe (onevision)
- seed: 42
- 하드웨어: A100 80GB × 1

---

### 6. 베이스라인

- Full cache (upper bound)
- Uniform keep ratio — 현재 `future_all_token` / `hybrid_h2o_future_all_token`

---

### 7. 예상 결과

- A: entropy는 layer별로 std > 0.05 (normalized) 수준의 편차 존재
- B: 초반 또는 후반 layer에서 Spearman 저하 구간 존재
- C: 중후반 layer 1~2개가 sensitivity 상위 집중
- D: keep=0.1에서 entropy/sensitivity-based allocation이 uniform 대비 ChartQA +1%p 이상

---

### 8. 판단 기준 (Success Criteria)

- A/B/C: layer entropy std > 0.05 (normalized) 또는 sensitivity top-3 layer가 나머지 대비 2× 이상 delta
- D: keep=0.1 기준 ChartQA accuracy uniform 대비 +1%p 이상 → 메인 방법에 통합 결정

---

### 9. Caveat

- Single-layer sensitivity는 multi-layer simultaneous pruning의 상호작용 미반영
- Budget allocation 최적값이 다른 모델/데이터셋에 일반화된다는 보장 없음
- Global cache 압축 (text token 포함) 방법과 직접 비교 불가

---

### 10. 실험 순서

```
Step 1: Sub-exp A — teacher entropy per layer        [오프라인, ~1h]
Step 2: Sub-exp B — student-teacher alignment        [오프라인, ~1h]
Step 3: Sub-exp C — single-layer sensitivity eval    [ChartQA eval × n_layers, ~4h]
Step 4: Sub-exp D — budget allocation 비교 eval      [3 ratio × 4 strategy, ~6h]
```

Step 1-2는 기존 teacher shard 재사용, GPU 불필요.  
Step 3-4는 실제 generation 필요.

---

### 11. 구현 포인트

**Step 1-2** — `analyze_layer_entropy_alignment.py` (신규):
```python
# teacher shard 로드 → per-layer y^(l) 계산
# normalized entropy: H^(l) / log(N_I)
# student score 로드 → Spearman, Overlap@K(K=0.2*N_I)
# 결과 CSV + matplotlib 시각화
```

**Step 3** — `ImageTokenTopKPress`에 `per_layer_keep_ratios: Dict[int, float]` 옵션 추가  
또는 `single_layer_prune_idx` + `single_layer_keep_ratio` 파라미터로 경량 구현.

**Step 4** — `layer_budget_mode: Literal["uniform", "entropy_prop", "entropy_inv", "sensitivity"]` 추가  
entropy는 prefill 시점 student score에서 online 계산 (teacher shard 불필요).

---

### 12. 예상 런타임 / 리소스

- Sub-exp A/B: ~2h (CPU, teacher shard 파싱)
- Sub-exp C: ~4h (A100 × 1, n_layers × ChartQA partial eval)
- Sub-exp D: ~6h (A100 × 1, 12 run × ChartQA)
- 추가 디스크: ~500MB (결과 CSV + 시각화)
