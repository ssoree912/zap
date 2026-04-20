# EXP-20260417-001 — Iterative Pruning 결과 분석

**ID**: EXP-20260417-001  
**Completed**: 2026-04-18  
**Status**: Done — 가설 기각 (Iterative ≡ One-shot, 집계 방식 무관)

---

## 전체 실험 흐름 요약 (Context)

| 실험 | 핵심 발견 |
|------|-----------|
| EXP-20260410-001 | `att_only_postvision` 선정, probe ≈ oracle |
| EXP-20260412-001 | probe 17W/10L/2T vs LOOK-M (MileBench 29) |
| EXP-20260412-002 | image-only scope **+9.6%p** — 결정적 요인 |
| EXP-20260412-003 | forced-keep 불필요 (score가 positional importance 내재) |
| EXP-20260412-004 | ratio 무감각성 — r=0.05도 r=0.20과 동일 17W/10L/2T |
| EXP-20260415-001 | VizCapture 시각화 + 효율성 최종 정리 |
| EXP-20260415-002 | 학습 데이터 다양화 (Running) |
| **EXP-20260417-001** | **Iterative pruning: 집계 방식과 무관하게 one-shot과 동일** |

---

## 결과 요약

### 5-way 비교 (29 datasets, r=0.20, MileBench LOOK-M eval 기준)

| Dataset | One-shot | Δ iter2 | Δ iter4 | Δ iter2-v2 | Δ iter4-v2 |
|---------|----------|---------|---------|------------|------------|
| actionlocalization | 0.2650 | 0 | 0 | 0 | 0 |
| actionprediction | 0.5400 | 0 | 0 | 0 | 0 |
| actionsequence | 0.4600 | 0 | 0 | 0 | 0 |
| alfred | 0.2000 | 0 | 0 | 0 | 0 |
| characterorder | 0.4900 | 0 | 0 | 0 | 0 |
| clevr_change | 0.2000 | 0 | 0 | 0 | 0 |
| counterfactualinference | 0.3250 | **-0.0050** | **-0.0050** | **-0.0050** | **-0.0050** |
| docvqa | 0.5100 | 0 | 0 | 0 | 0 |
| egocentricnavigation | 0.3200 | 0 | 0 | 0 | 0 |
| gpr1200 | 0.1167 | 0 | 0 | +0.0017 | **-0.0017** |
| iedit | 0.2000 | 0 | 0 | 0 | 0 |
| imageneedleinahaystack | 0.2000 | 0 | 0 | 0 | 0 |
| mmcoqa | 0.2000 | 0 | 0 | 0 | 0 |
| movingattribute | 0.5150 | 0 | 0 | 0 | 0 |
| movingdirection | 0.3350 | 0 | 0 | 0 | 0 |
| multimodalqa | 0.7700 | **-0.0050** | **-0.0100** | **-0.0150** | **-0.0150** |
| nuscenes | 0.6100 | 0 | 0 | 0 | 0 |
| objectexistence | 0.4900 | 0 | 0 | 0 | 0 |
| objectinteraction | 0.5200 | 0 | 0 | 0 | 0 |
| objectshuffle | 0.3150 | 0 | 0 | 0 | 0 |
| ocr_vqa | 0.3200 | 0 | 0 | 0 | 0 |
| scenetransition | 0.7750 | 0 | 0 | 0 | 0 |
| slidevqa | 0.4750 | 0 | 0 | 0 | 0 |
| spot_the_diff | 0.2000 | 0 | 0 | 0 | 0 |
| statechange | 0.4100 | 0 | 0 | 0 | 0 |
| textneedleinahaystack | 0.2000 | 0 | 0 | 0 | 0 |
| tqa | 0.4700 | 0 | 0 | 0 | 0 |
| webqa | 0.6150 | 0 | 0 | 0 | 0 |
| wikivqa | 0.7100 | 0 | 0 | 0 | 0 |
| **AVG (N=29)** | **0.4054** | **-0.0003** | **-0.0005** | **-0.0006** | **-0.0007** |
| **W / L / T** | | 0/2/27 | 0/2/27 | 1/2/26 | 0/3/26 |

**iter2** = iterative-2 (global amax 집계)  
**iter4** = iterative-4 (global amax 집계)  
**iter2-v2** = iterative-2 (vote-based per-layer 집계, `iterative_ver2_2`)  
**iter4-v2** = iterative-4 (vote-based per-layer 집계, `iterative_ver2_4`)

---

## 가설 검증

- [x] 가설이 틀렸다

| 예측 | 실제 |
|------|------|
| iterative avg ≥ one-shot (+0.3~0.8%p) | 모든 변형에서 avg Δ ≤ -0.0003 (사실상 tie) |
| per-layer 집계(ver2)가 global보다 유리 | iter2-v2 MultiModalQA -0.0150 (더 나쁨) |
| 라운드 수 증가가 도움 | iter4 ≤ iter2 성능 (라운드 증가할수록 소폭 하락) |

---

## 왜 Iterative ≡ One-shot인가

### 1. A-option 구현 확인 및 구조 차이

현재 구현(`_iterative_probe_preselect()`)은 라운드마다 LLaVA forward를 재실행해 hidden state를 갱신하는 **A-option**으로 동작한다. hidden_states[0]은 merged multimodal embedding이고, 각 라운드에서 좁혀진 image token pool로 재계산된다.

그러나 iterative와 one-shot 사이에는 **구조적 차이**가 있다:

- **One-shot (`ProbeImageTeacherPress`)**: 레이어별로 독립적인 top-k를 수행 → 각 레이어가 서로 다른 image token 집합을 유지
- **Iterative (`PreselectedImagePress` + `_iterative_probe_preselect`)**: 최종 선택은 **모든 레이어에 공통된 단일 global mask** → 레이어별 선택의 다양성이 없음

즉 iterative pruning의 마지막 단계 자체가 one-shot의 per-layer 구조와 다르다. 이 두 변수(iterative re-scoring vs global mask)가 함께 변하기 때문에 본 실험의 비교는 re-scoring 효과만을 순수하게 격리하지 못한다.

### 2. Per-layer 집계(ver2)를 도입해도 개선 없음

`iterative_ver2` 시리즈는 각 라운드에서 레이어별로 top-k를 뽑고 vote count + max-score tiebreak로 pool을 좁히는 방식이다. 이는 re-scoring과 집계 방식을 둘 다 변경한 실험이다.

결과: **오히려 MultiModalQA에서 더 큰 하락** (-0.0150 vs iter4-global의 -0.0100). 다른 모든 데이터셋은 동일. vote 기반 집계도 ranking을 바꾸지 못했다.

### 3. Round 간 ranking stability

A-option을 사용하더라도 이득이 없는 가장 그럴듯한 설명은 **하위 image token 제거 후에도 남은 토큰의 hidden state가 충분히 바뀌지 않아 probe score ranking이 거의 유지**된다는 것이다. H2O(Zhang et al., 2023)와 SnapKV(Li et al., 2024)가 텍스트 LLM에서 attention 기반 중요도 ranking의 시간적 안정성을 보고한 것과 유사한 현상일 수 있다. 다만 두 논문 모두 multimodal image-token 재pruning 시나리오를 직접 다루지는 않으므로 원인 증명이 아닌 analogy 수준이다.

이 설명을 검증하려면 round-wise token overlap ratio나 rank correlation을 측정해야 하는데, 현재 실험에서는 수행하지 않았다.

### 4. `position_ids` 불일치 (known risk)

Round 2+ 에서 `position_ids`를 명시하지 않아 preselect 단계와 generate 단계 간 RoPE positional encoding이 불일치할 수 있다 (implementation.md KNOWN RISK 1). counterfactualinference와 multimodalqa의 소폭 하락이 이와 관련될 가능성이 있으나, 이 원인과 ranking stability를 현재 데이터만으로 분리해서 증명할 수는 없다.

---

## 결론

네 가지 iterative 변형(2/4 라운드 × global/per-layer 집계) 모두 one-shot 대비 유의한 이득을 주지 못했다. 평균 Δ는 최대 -0.0007로 사실상 tie이며, 일부 변형에서는 소폭 하락했다. re-scoring 효과보다 round 간 ranking stability가 지배적이거나, `position_ids` 처리 같은 구현 artifact가 개입한 것으로 추정되나 직접 검증되지 않았다.

집계 방식(global vs vote-based per-layer)의 차이도 결과를 바꾸지 못했다. 이는 aggregation이 병목이 아님을 시사한다.

성능 개선의 방향은 iterative pruning 전략이 아닌 **probe 학습 데이터 다양화** (EXP-20260415-002)가 올바른 경로다.

---

## 다음 실험 제안

| 우선순위 | 실험 | 근거 |
|---|---|---|
| 🔴 | EXP-20260415-002 마무리 (학습 데이터 다양화) | probe-oracle gap 감소가 진짜 개선 경로 |
| 🟡 | FastV 비교 (image-only, layer-2 attention scoring) | probe vs rule-based scoring 비교 완성 |
| 🟡 | SparseVLM 비교 (task-conditioned 유사 방법) | novelty 포지셔닝 보완 |

---

## 재현 커맨드

```bash
# One-shot baseline
GPU_INDEX=0 bash scripts/run_milebench_probe_all.sh

# Iterative-2 (global)
GPU_INDEX=0 KEEP_RATIOS="0.20" bash experiments/EXP-20260417-001/run_iter2.sh

# Iterative-4 (global)
GPU_INDEX=0 KEEP_RATIOS="0.20" bash experiments/EXP-20260417-001/run.sh

# Iterative-2 layerwise vote (ver2)
GPU_INDEX=2 KEEP_RATIOS="0.20" bash experiments/EXP-20260417-001/run_iter2_ver2.sh

# Iterative-4 layerwise vote (ver2)
GPU_INDEX=1 KEEP_RATIOS="0.20" bash experiments/EXP-20260417-001/run_iter4_ver2.sh
```

결과 위치:
- One-shot: `artifacts/combine_prob/*/probe_mlp/keep_0p20/metrics.json`
- Iterative-2: `artifacts/combine_prob/*/iterative_2/keep_0p20/metrics.json`
- Iterative-4: `artifacts/combine_prob/*/iterative_4/keep_0p20/metrics.json`
- Iterative-2 ver2: `artifacts/combine_prob/*/iterative_ver2_2/keep_0p20/metrics.json`
- Iterative-4 ver2: `artifacts/combine_prob/*/iterative_ver2_4/keep_0p20/metrics.json`
