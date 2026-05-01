# Ablation Study Report
**작성일**: 2026-04-15  
**최종 수정일**: 2026-04-16  
**대상 실험**: EXP-20260410-001 ~ EXP-20260415-002  
**방법**: Image token selective eviction using task-conditioned attention probe  
**비교 대상**: LOOK-M (primary baseline)

---

## 실험 전체 흐름

```
[Phase 0] Oracle score 선택           → att_only_postvision 선정
              ↓
[Phase 1] Eviction scope ablation      → image-only >> all-token
              ↓
[Phase 2] Scoring signal ablation      → oracle vs H2O (image-only scope 고정)
              ↓
[Phase 3] Probe 파이프라인 확장        → ScienceQA-only vs Combined 비교 축 확정
              ↓
[Phase 4] MileBench 전체 추론          → truncate 통일, 결과 구조 통일
              ↓
[Phase 5] 동일 조건 연산량 비교         → common69 (n=69) 기준 재측정
              ↓
[Phase 6] 전체 성능 비교               → LOOK-M / Oracle / SQ-only / Combine
```

---

## 우리 방법 개요

**핵심 아이디어**: Prefill 단계에서 image token의 KV cache를 task-conditioned attention score로 선택적으로 제거.

```
입력: 이미지 + 텍스트 질문
  → Vision Encoder → Image tokens (576개/이미지, LLaVA-1.5 기준)
  → [우리 방법] Probe MLP로 각 image token 중요도 scoring
  → Top-k image token만 KV cache에 유지 (text token은 항상 전부 유지)
  → Decode: 압축된 KV cache로 고속 생성
```

**LOOK-M과의 구조적 차이**:

| 항목 | 우리 방법 | LOOK-M |
|---|---|---|
| Eviction 대상 | image token만 | text + image 전체 KV cache |
| Score 방식 | att_only_postvision (task-conditioned) | H2O (heavy-hitter + recent, task-agnostic) |
| Eviction 시점 | Prefill 1회 고정 | Decode 중 온라인 업데이트 |
| Text token 보존 | 항상 100% | ratio에 따라 제거 가능 |
| Flash Attention | 사용 가능 (표준 HF path) | 불가 (커스텀 attention 구현) |

---

## Phase 0: Oracle Score 선택 (EXP-20260410-001 일부)

### 변수
- **독립변수**: scoring 방법 (att_only_postvision / att_only_answer / splus_postvision / splus_answer)
- **고정**: eviction scope (image-only), ratio (r=0.02~0.20), 데이터셋 4개

### 결과 (r=0.10, image_keep_ratio 기준)

| Dataset | att_only_pv | att_only_ans | splus_pv | splus_ans |
|---|---|---|---|---|
| Spot-the-Diff (ROUGE-L) | **0.2107** | 0.2041 | 0.2095 | 0.2110 |
| IEdit (ROUGE-L) | 0.0955 | **0.0979** | 0.0940 | 0.0956 |
| CLEVR-Change (ROUGE-L) | **0.1294** | 0.1254 | 0.1279 | 0.1239 |
| DocVQA (Accuracy) | 0.515 | 0.515 | 0.515 | 0.515 |

---

## Phase 1+2: Eviction Scope × Scoring 2×2 Ablation (EXP-20260410-001 / EXP-20260412-002)

### 설계

| | image-only eviction | all-token eviction |
|---|---|---|
| **H2O score** | **Cell A**: H2O + img-only | LOOK-M (유사) |
| **oracle score** | **Ours**: att_only_pv + img-only | **Cell B**: att_only_pv + all-token |

- **Factor 1 (Score)**: att_only_postvision vs H2O — eviction scope 고정으로 분리
- **Factor 2 (Scope)**: image-only vs all-token — score 고정으로 분리
- **통일 기준**: `total_keep_ratio=0.20` (r_eff_prompt 동일)

### 핵심 결론

**Image-only eviction이 압도적으로 중요**.  
All-token eviction은 낮은 ratio에서 text token(질문/선택지)을 제거하여 성능이 붕괴함.  
Scoring signal(att_only_pv vs H2O)의 효과는 부차적이며 task에 따라 다름.

---

## Phase 3: 최신 파이프라인 정렬 (ScienceQA-only vs Combined)

### 목적
- 비교 축을 아래 4개 방법으로 고정:
  - LOOK-M
  - Oracle (on-the-fly)
  - Probe (ScienceQA-only)
  - Probe (Combined: ScienceQA+TextVQA+NLVR2)

### 고정 조건 (최신)
- Backbone: `llava-1.5-7b-hf`
- Keep: `0.20`
- Prompt style: `look_milebench`
- Truncation: `--truncate_like_lookm`
- `look_max_context_len=4096`
- `look_n_tokens_per_image=576`

### 체크포인트/결과 경로
- Probe SQ-only: `TBD`
- Probe Combined: `/workspace/zap/ckpts/image_probe_combined_v1/mlp`
- Probe inference root: `/workspace/zap/artifacts/combine_prob`
- Oracle inference root: `/workspace/zap/artifacts/oracle`

---

## Phase 4: MileBench 전체 추론 (포맷 통일)

### 출력 구조
- Probe: `/workspace/zap/artifacts/combine_prob/{dataset_slug}/probe_mlp/keep_0p20`
- Oracle: `/workspace/zap/artifacts/oracle/{dataset_slug}/oracle_onthefly/keep_0p20`

### 상태 (업데이트용)
| 항목 | 상태 | 비고 |
|---|---|---|
| Probe Combined 전체 실행 | `TBD` | |
| Probe 실패 데이터셋 재실행 | `Running` | GPU0/GPU2 분할 실행 중 |
| Oracle on-the-fly 전체 실행 | `Running` | GPU1 실행 중 |

---

## Phase 5: 연산량 비교 (동일 조건, common69 n=69)

> 재측정 예정. 아래 표는 최신 파이프라인 기준 입력 포맷만 고정.

### 조건
- 동일 샘플: common69 (`n=69`)
- 동일 keep budget: `r_eff_prompt=0.20`
- 동일 모델/토크나이저/프롬프트 체인

### 결과 표 (채우기 전)
| Method | n | Prefill (ms) | TTFT (ms) | TBT (ms/token) | KV Cache (GiB) | r_eff_prompt | 비고 |
|---|---:|---:|---:|---:|---:|---:|---|
| Full Cache | 69 | `TBD` | `TBD` | `TBD` | `TBD` | 1.00 | |
| LOOK-M | 69 | `TBD` | `TBD` | `TBD` | `TBD` | 0.20 | |
| Oracle (on-the-fly) | 69 | `TBD` | `TBD` | `TBD` | `TBD` | 0.20 | |
| Probe (ScienceQA-only) | 69 | `TBD` | `TBD` | `TBD` | `TBD` | 0.20 | |
| Probe (Combined) | 69 | `TBD` | `TBD` | `TBD` | `TBD` | 0.20 | |

---

## Phase 6: 전체 성능 비교 (LOOK-M / Oracle / SQ-only / Combined)

> 실험 진행 중이므로 결과값은 비워둠. keep=0.20 기준.

### 6.1 요약표
| Method | Win | Lose | Tie | Weighted Mean | 비고 |
|---|---:|---:|---:|---:|---|
| LOOK-M | `TBD` | `TBD` | `TBD` | `TBD` | |
| Oracle (on-the-fly) | `TBD` | `TBD` | `TBD` | `TBD` | |
| Probe (ScienceQA-only) | `TBD` | `TBD` | `TBD` | `TBD` | |
| Probe (Combined) | `TBD` | `TBD` | `TBD` | `TBD` | |

### 6.2 데이터셋별 성능표
| Dataset | Metric | LOOK-M | Oracle | Probe (SQ-only) | Probe (Combined) | Winner |
|---|---|---:|---:|---:|---:|---|
| actionlocalization | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| actionprediction | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| actionsequence | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| alfred | ROUGE-L | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| characterorder | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| clevr_change | ROUGE-L | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| counterfactualinference | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| docvqa | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| egocentricnavigation | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| gpr1200 | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| iedit | ROUGE-L | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| imageneedleinahaystack | ROUGE-L | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| mmcoqa | ROUGE-L | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| movingattribute | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| movingdirection | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| multimodalqa | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| nuscenes | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| objectexistence | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| objectinteraction | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| objectshuffle | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| ocr_vqa | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| scenetransition | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| slidevqa | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| spot_the_diff | ROUGE-L | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| statechange | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| textneedleinahaystack | ROUGE-L | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| tqa | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| webqa | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| wikivqa | Accuracy | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

---

## 최신 아티팩트 경로

- Probe Combined 결과: `/workspace/zap/artifacts/combine_prob`
- Oracle on-the-fly 결과: `/workspace/zap/artifacts/oracle`
- Probe vs LOOK-M + Combine 확장 CSV: `/workspace/zap/artifacts/results/probe_vs_lookm_r020_truncated_with_combine.csv`
- Oracle 실행 로그: `/workspace/zap/artifacts/oracle/_run_logs/`
- Probe 재실행 로그: `/workspace/zap/artifacts/combine_prob/_run_logs/`

---

## TODO (문서 채우기)

- [ ] common69 연산량 재측정 값 입력
- [ ] 4방법(LOOK-M / Oracle / SQ-only / Combined) 성능표 값 입력
- [ ] W/L/T 및 weighted mean 계산 반영
- [ ] 최종 결론 업데이트
