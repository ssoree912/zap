# ZAP 실험 인덱스

**프로젝트**: Task-conditioned Image Token KV Cache Eviction (ZAP)  
**모델**: LLaVA-1.5-7B | **Baseline**: LOOK-M | **평가**: MileBench 29 datasets

---

## 실험 목록

| ID | 날짜 | 핵심 질문 | 결론 | 상태 |
|----|------|-----------|------|------|
| [EXP-20260410-001](EXP-20260410-001/) | 2026-04-10 | Teacher score 선택 + Probe distillation 검증 | `att_only_postvision` 선정, probe ≈ oracle | ✅ Done |
| [EXP-20260412-001](EXP-20260412-001/) | 2026-04-12 | MileBench 29 datasets 전체 비교 | probe **17W/10L/2T** vs LOOK-M | ✅ Done |
| [EXP-20260412-002](EXP-20260412-002/) | 2026-04-12 | 2×2 Ablation (Score × Scope) | image-only scope **+9.6%p** 결정적 | ✅ Done |
| [EXP-20260412-003](EXP-20260412-003/) | 2026-04-12 | Position forced-keep 필요성 분석 | 불필요 — score가 이미 내재 | ✅ Done |
| [EXP-20260412-004](EXP-20260412-004/) | 2026-04-14 | Ratio sensitivity (r=0.05도 충분한가?) | ✓ 17W/10L/2T 유지, ratio 무감각성 | ✅ Done |
| [EXP-20260415-001](EXP-20260415-001/) | 2026-04-15 | 시각화 파이프라인 + 효율성 최종 정리 | VizCapture 구현, 전체 결과 통합 | ✅ Done |
| [EXP-20260415-002](EXP-20260415-002/) | 2026-04-15 | 학습 데이터 다양화 (+ TextVQA + NLVR2) | 진행 중 | 🔄 Running |
| [EXP-20260417-001](EXP-20260417-001/) | 2026-04-17 | Inference-time iterative pruning (one-shot → N-round) | 계획 중 | 📋 Planned |

---

## 핵심 발견 요약

### 성능 (r=0.20, MileBench 29 datasets)

```
probe vs LOOK-M:
  승/패/타이 = 17 / 10 / 2
  평균 Δ     = +0.023 (+2.3%p)
  
probe vs LOOK-M (ratio 1/4, r=0.05):
  승/패/타이 = 17 / 10 / 2  ← 동일! ratio 무감각성
  평균 Δ     = +0.021
```

### 효율성 (common 69 샘플, r_eff=0.200 통일)

```
          Prefill    TBT        KV cache   r_eff
full_cache   496ms    29.0ms   0.887 GiB   1.000
probe        673ms    27.6ms   0.129 GiB   0.200  ← 우리 방법
LOOK-M     1,597ms    74.0ms   0.182 GiB   0.200
oracle       856ms    27.4ms   0.129 GiB   0.200  ※ Pass2만 측정

probe vs LOOK-M:
  TBT 2.63× 빠름 (73.5ms → 27.9ms, matched 77 샘플 기준)
  Prefill 39% 단축 (1,080ms → 657ms)
  KV cache full_cache 대비 72% 절감
```

### 2×2 Ablation

```
                image-only eviction   all-token eviction
H2O score             0.228                LOOK-M ref
att_only_pv           0.235 ← Ours         0.139

Factor 1 (Score):  +0.007 (+0.7%p)  — 부차적
Factor 2 (Scope):  +0.096 (+9.6%p)  — 결정적
```

---

## 다음 실험 후보

| 우선순위 | 실험 | 상태 | 예상 가치 |
|---|---|---|---|
| ✅ | Eviction map 시각화 (probe/oracle/H2O 3-way, Spot-the-Diff) | **Done** — EXP-20260415-001 | qualitative 증거 |
| 🔴 | FastV 비교 (image-only, scoring 방식만 다름) | 미구현 | Phase 2 ablation 보완 |
| 🟡 | SparseVLM 비교 (task-conditioned 유사 방법) | 미구현 | novelty 검증 |
| 🟡 | 학습 데이터 다양화 (+ NExT-QA + TextVQA + NLVR2) | **Running** — EXP-20260415-002 | weak 데이터셋 개선 |
| 🟢 | Oracle 2-pass 공정 비교 (Pass1 포함) | 재실행 필요 | 논문 efficiency 표 보완 |

---

## 아티팩트 위치

| 항목 | 경로 |
|---|---|
| 전체 Ablation 보고서 | `experiments/ablation_report.md` |
| 성능 결과 (MileBench 29) | `/workspace/hd/artifacts/probe_global/probe_vs_lookm_r020_truncated.csv` |
| 성능 결과 (common 69, 17 datasets) | `/workspace/hd/artifacts/probe_global/efficiency_all_datasets/performance_common69_datasets_keep0p20.csv` |
| 효율성 CSV (common 69, 4-way) | `/workspace/hd/artifacts/probe_global/efficiency_all_datasets/per_sample_common69_4methods.csv` |
| Probe 모델 체크포인트 | `/workspace/hd/artifacts/zap/probe_model_scienceqa` |
| 시각화 코드 | `kvpress/presses/image_token_press.py` (VizCapture) |
| 시각화 렌더러 | `scripts/render_eviction_visualizations.py` |
