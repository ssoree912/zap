# EXP-20260417-001 — Iterative Pruning 결과 분석

**ID**: EXP-20260417-001  
**Completed**: 2026-04-17  
**Status**: Done — 가설 기각 (Iterative ≡ One-shot)

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
| **EXP-20260417-001** | **Iterative pruning: one-shot과 동일 → 가설 기각** |

---

## 결과 요약

### 수치 비교 (22 datasets, r=0.20, MileBench LOOK-M eval 기준)

| Dataset | One-shot | Iter-4 | Δ |
|---------|----------|--------|---|
| actionlocalization | 0.2650 | 0.2650 | 0.0000 |
| actionprediction | 0.5400 | 0.5400 | 0.0000 |
| actionsequence | 0.4600 | 0.4600 | 0.0000 |
| characterorder | 0.4900 | 0.4900 | 0.0000 |
| counterfactualinference | 0.3250 | 0.3200 | **-0.0050** |
| docvqa | 0.5100 | 0.5100 | 0.0000 |
| egocentricnavigation | 0.3200 | 0.3200 | 0.0000 |
| gpr1200 | 0.1167 | 0.1167 | 0.0000 |
| movingattribute | 0.5150 | 0.5150 | 0.0000 |
| movingdirection | 0.3350 | 0.3350 | 0.0000 |
| multimodalqa | 0.7700 | 0.7600 | **-0.0100** |
| nuscenes | 0.6100 | 0.6100 | 0.0000 |
| objectexistence | 0.4900 | 0.4900 | 0.0000 |
| objectinteraction | 0.5200 | 0.5200 | 0.0000 |
| objectshuffle | 0.3150 | 0.3150 | 0.0000 |
| ocr_vqa | 0.3200 | 0.3200 | 0.0000 |
| scenetransition | 0.7750 | 0.7750 | 0.0000 |
| slidevqa | 0.4750 | 0.4750 | 0.0000 |
| statechange | 0.4100 | 0.4100 | 0.0000 |
| tqa | 0.4700 | 0.4700 | 0.0000 |
| webqa | 0.6150 | 0.6150 | 0.0000 |
| wikivqa | 0.7100 | 0.7100 | 0.0000 |
| **AVERAGE** | **0.4708** | **0.4701** | **-0.0007** |

**Win / Loss / Tie: 0 / 2 / 20**

> **LOOK-M eval 없는 7개 데이터셋** (alfred, clevr_change, iedit, imageneedleinahaystack, mmcoqa, spot_the_diff, textneedleinahaystack): exact_match 기준에서도 두 방법이 완전히 동일한 점수를 기록함. MMCoQA 기준 both=0.295.

---

## 가설 검증

- [x] 가설이 틀렸다

| 예측 | 실제 |
|------|------|
| iterative-4 avg ≥ one-shot (+0.3~0.8%p) | avg Δ = **-0.0007 (사실상 tie)** |
| oracle-probe gap의 20~50% 개선 | **0% 개선** |

**판정: 실패 — iterative-4는 one-shot 대비 유의한 이득을 주지 못했다. 평균 차이 -0.0007은 사실상 동일한 수준이다.**

---

## 예상과 달랐던 점

PLAN.md는 두 가지 failure 시나리오를 사전에 명시했다:

1. **역효과**: iterative < one-shot — 2개 데이터셋에서 소폭 하락 (각 1~2 샘플 차이 수준)
2. **동치 시나리오**: round 간 score rank가 너무 안정적이어서 iterative ≡ one-shot — **실제로 발생**

---

## 왜 Iterative ≡ One-shot인가

### 1. 고정 score 반복 방식 (B-option)에서의 동치

초기 구현 로그에는 "모든 라운드에서 동일한 probe score를 재사용"하는 방식이 기록되어 있다. 이 경우는 원리적으로 one-shot과 동치다. 고정된 score 위에서 중첩 top-k를 반복해도 최종 top-k 집합은 달라지지 않는다.

이 설명은 현재 공유된 구현 로그(Phase 1 pseudocode 기준)로 검증 가능하다.

### 2. Full forward re-run 방식 (A-option)에 대해서

implementation.md는 고정 score 방식의 한계를 인식하고 "라운드마다 LLaVA forward를 재실행해 hidden state를 갱신하는 A-option"으로 전환했다고 기록하고 있다. 그러나 현재 공유된 구현 코드만으로는 A-option이 실제로 이 실험에 적용됐는지 완전히 확인된 상태는 아니다.

만약 A-option이 실행됐다면, 이득이 없었던 가능한 원인은 다음이다:

- **round 간 ranking이 empirically 안정적이었을 가능성**: 하위 토큰 제거 후 남은 토큰의 hidden state가 충분히 바뀌지 않아 re-scoring의 ranking 순서가 거의 유지됐을 수 있다. H2O(Zhang et al., 2023)와 SnapKV(Li et al., 2024)는 텍스트 LLM에서 attention 기반 중요도 ranking이 시간적으로 안정적임을 보이는데, 이것이 이 실험의 유사한 현상에 대한 analogy가 될 수 있다. 다만 두 논문 모두 multimodal image-token 재pruning 시나리오를 직접 다루지는 않으므로 원인 증명이 아닌 참고 수준이다.

- **`position_ids` 불일치 (known risk)**: implementation.md에도 명시된 KNOWN RISK 1 — round 2+에서 `position_ids`를 명시하지 않아 preselect 단계와 generate 단계 간 RoPE positional encoding이 불일치할 수 있다. 이것이 counterfactualinference(-0.005)와 multimodalqa(-0.010) 소폭 하락의 원인일 가능성이 있다. 두 데이터셋 모두 다중 이미지 간 reasoning이 중요한 태스크이며, 손실은 각각 1~2 샘플 차이 수준으로 noise와 양립 가능하다.

이 원인들은 아직 round-wise overlap ratio나 rank correlation 측정으로 직접 검증되지 않았다. 현재 데이터만으로는 "ranking이 안 바뀌어서 gain이 없었다"와 "`position_ids` artifact 때문에 소폭 손해 봤다"를 분리해서 증명한 상태가 아니다.

### 3. Implementation sanity check 필요

Phase 1 pseudocode 기준으로, 마지막 라운드는 이미 `final_k` 크기로 좁혀진 pool에 per-head `topk(..., k=final_k)`를 돌린다. 이 단계는 pool membership을 바꾸지 않고 순서만 재정렬한다. Baseline one-shot의 token selection geometry와 정확히 같은 결과를 내는지 추가 확인이 필요하다.

---

## 결론

iterative-4는 one-shot 대비 유의한 이득을 주지 못했고, 평균 차이 -0.0007은 사실상 tie다. 고정 score 반복 방식에서는 one-shot과 수학적으로 동치이며, full forward re-run 방식에서도 round 간 ranking이 거의 변하지 않았거나 `position_ids` 처리 같은 구현 artifact가 개입했을 가능성이 있다. 다만 이 원인들은 아직 overlap, rank correlation, position ablation으로 직접 검증되지 않았다.

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
# Iterative-4 probe
GPU_INDEX=0 KEEP_RATIOS="0.20" bash experiments/EXP-20260417-001/run.sh

# Iterative-2 probe
GPU_INDEX=0 KEEP_RATIOS="0.20" bash experiments/EXP-20260417-001/run_iter2.sh

# One-shot baseline (비교용)
GPU_INDEX=0 bash scripts/run_milebench_probe_all.sh
```

결과 위치:
- Iterative-4: `/workspace/zap/artifacts/combine_prob/*/iterative_4/keep_0p20/metrics.json`
- Iterative-2: `/workspace/zap/artifacts/combine_prob/*/iterative_2/keep_0p20/metrics.json`
- One-shot: `/workspace/zap/artifacts/combine_prob/*/probe_mlp/keep_0p20/metrics.json`
