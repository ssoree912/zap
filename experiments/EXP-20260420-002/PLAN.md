# Experiment Plan

**ID:** EXP-20260420-002
**Author:** ssoree912
**Date:** 2026-04-20
**Status:** [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned
**Parent:** EXP-20260418-001 (Future-supervised probe + Hybrid), EXP-20260420-001 (Per-layer Hybrid)

---

## 1. Motivation

논문용 비교 실험 설계. 현재까지의 실험은 PostVision probe와 Future probe, 그리고 둘의 Hybrid만 비교하고 있어 **가장 단순한 prefill-only attention baseline**이 빠져 있다. 논문에서 다음 3축 비교를 완성해야 한다:

| 축 | 방법 | 의미 |
|---|---|---|
| **Generic prefill saliency** | H2O-prefill | prefill 전체 query에서 image token이 얼마나 참조되었는가 (heavy hitter) |
| **Task-conditioned relevance** | PostVision probe | post-vision text query에서 image token으로의 attention (질문 관련성) |
| **Future-conditioned utility** | Future probe | decode query에서 image token으로의 attention (미래 활용도) |

H2O-prefill이 빠지면 "단순 prefill attention으로는 부족하고, task/future conditioning이 필요하다"는 핵심 claim의 근거가 약해진다.

### 논문 서사 구조

> **Claim 1**: Generic prefill accumulation (H2O-prefill) < Task-conditioned prefill (PostVision)
> **Claim 2**: Task-conditioned prefill만으로는 부족 → Future-conditioned supervision이 decode-time usefulness를 더 잘 포착
> **Claim 3**: Relevance(PV)와 Utility(Future)는 상보적 → Hybrid가 추가 개선 가능

---

## 2. Hypothesis

**H1 (Generic < Task-conditioned)**: H2O-prefill은 PostVision보다 MileBench 평균 ROUGE-L/Accuracy 기준 낮은 성능을 보인다.
- 이유: H2O는 task-agnostic하므로 질문과 무관하게 자주 참조된 토큰(sink 등)을 과대평가.

**H2 (Task-conditioned < Future-conditioned)**: PostVision-only는 Future-only보다 일부 dataset에서 열세.
- 이유: prefill 시점에서는 모델이 아직 답변을 생성하지 않아, 실제로 decode에서 필요한 토큰을 완벽히 예측 불가.

**H3 (Hybrid > 단독)**: Hybrid(PV+Future)가 PV-only, Future-only 각각보다 최소 동등 이상.
- EXP-20260420-001 per-layer 결과와 결합하여 검증.

### Success Target
- 6 dataset 중 **4개 이상에서 H2O-prefill < PostVision** (ROUGE-L 기준)
- 6 dataset 중 **2개 이상에서 PostVision < Future or Hybrid**
- H2O-prefill이 전 dataset에서 가장 낮거나 두 번째로 낮은 성능

---

## 3. Independent Variables

본 실험에서 변경하는 것:

- **Scoring method**: {H2O-prefill, PostVision, Future, Hybrid(per-layer)}
- **Keep ratio**: {0.2, 0.1}

### 방법별 정의

#### H2O-prefill (Generic prefill saliency)
```
s_i^{H2O} = (1/H) * sum_h sum_{q in Q_prefill} A^{(l,h)}[q, i]
```
- 모든 prefill query position (image + text 포함)에서 각 image token(i)으로의 attention 합
- Layer별 독립 계산, head 평균
- `--mode h2o_image_only` (이미 구현됨)

#### PostVision (Task-conditioned relevance)
```
s_i^{PV} = MLP_pv(h_i)    // trained on post-vision text→image attention
```
- `--mode probe`

#### Future (Decode-conditioned utility)
```
s_i^{Future} = MLP_fu(h_i)    // trained on future decode→image attention
```
- `--mode future --selected_layer_indices 24..31`

#### Hybrid per-layer (PV + Future blend)
```
Layer 0-23:  s_i = softmax(MLP_pv(h_i))
Layer 24-31: s_i = α * softmax(MLP_pv(h_i)) + (1-α) * softmax(MLP_fu(h_i))
```
- `--mode hybrid --future_blend_layers 24..31 --alpha <α>`
- EXP-20260420-001 결과에서 dataset별 best α 사용

---

## 4. Dependent Variables

### Primary metrics
- **ROUGE-L** (generative: clevr_change, alfred, iedit, mmcoqa, spot_the_diff)
- **Accuracy** (choice: webqa)

### Secondary metrics
- 방법 간 순위 (rank ordering)
- H2O-prefill 대비 각 방법의 delta

---

## 5. Fixed Conditions

- **Model**: `/home/M2026107/zap/ckpts/llava-1.5-7b-hf` (LLaVA-1.5-7B)
- **PV probe**: `/home/M2026107/zap/ckpts/postvision_probe_v4_20ep`
- **Future probe**: `/home/M2026107/zap/ckpts/future_probe_v4_last8_20ep_bcast31`
- **Dataset**: 6개 — spot_the_diff, clevr_change, webqa, alfred, iedit, mmcoqa
- **Prompt style**: `look_milebench`
- **Attention impl**: `eager` (output_attentions=True 필요)
- **Truncation**: `--truncate_like_lookm`
- **Hardware**: RTX 4090 × 1 (순차 실행)

---

## 6. Baselines & Comparisons

| 방법 | 축 | mode | 출처 |
|---|---|---|---|
| H2O-prefill | Generic saliency | `h2o_image_only` | **본 실험 (신규)** |
| PostVision-only | Task-conditioned relevance | `probe` | EXP-20260418-001 Phase 3 결과 재사용 |
| Future-only | Future utility | `future` | EXP-20260418-001 Phase 3 결과 재사용 |
| Hybrid per-layer | PV+Future blend | `hybrid` | EXP-20260420-001 결과 재사용 |

### 재사용 가능 결과 확인

| 방법 | keep_ratio | 재사용 가능? | 경로 |
|---|---|---|---|
| PostVision k=0.2 | 0.2 | 확인 필요 | `artifacts/EXP-20260418-001/v4_eval/<ds>/pv_k0p2/` |
| PostVision k=0.1 | 0.1 | 확인 필요 | `artifacts/EXP-20260418-001/v4_eval/<ds>/pv_k0p1/` |
| Future k=0.2 | 0.2 | 확인 필요 | `artifacts/EXP-20260418-001/v4_eval/<ds>/future_k0p2/` |
| Future k=0.1 | 0.1 | 확인 필요 | `artifacts/EXP-20260418-001/v4_eval/<ds>/future_k0p1/` |
| Hybrid per-layer | 0.2, 0.1 | EXP-20260420-001 진행 중 | `artifacts/EXP-20260420-001/v5_perlayer_sweep/` |

**신규 실행 필요**: H2O-prefill만 (6 dataset × 2 keep_ratio = **12 runs**)

---

## 7. Expected Results

### 정성적 예측

- **H2O-prefill**: 가장 낮은 성능. Sink token bias로 실제 task-relevant image token을 놓칠 가능성 높음.
- **PostVision**: H2O보다 유의미하게 높음. 질문과 관련된 시각 정보를 선별.
- **Future**: PostVision과 비슷하거나 일부 dataset에서 우세. Decode 시점 필요 토큰 반영.
- **Hybrid per-layer**: 최고 또는 Future와 동등. Late layer에서 Future signal 추가.

### 정량적 예측 (ROUGE-L, k=0.1 기준)

| Dataset | H2O-prefill | PostVision | Future | Hybrid |
|---|---|---|---|---|
| clevr_change | ~0.25 | ~0.27 | ~0.29 | ~0.29 |
| alfred | ~0.27 | ~0.28 | ~0.29 | ~0.30 |
| spot_the_diff | ~0.17 | ~0.20 | ~0.19 | ~0.20 |
| iedit | ~0.23 | ~0.25 | ~0.24 | ~0.25 |
| mmcoqa | ~0.30 | ~0.35 | ~0.33 | ~0.38 |
| webqa (Acc) | ~0.40 | ~0.42 | ~0.41 | ~0.42 |

---

## 8. Success Criteria

### Primary
- **H1 지지**: 6 dataset 중 4개 이상에서 H2O-prefill < PostVision (유의미한 차이 Δ > 0.005)
- 논문 Table 1에 4-method 비교 grid 완성

### Secondary
- H2O → PV → Future/Hybrid 순서가 대부분 dataset에서 유지
- 최소 1개 dataset에서 Hybrid > PV ∧ Hybrid > Future (상보성 증거)

### Failure Mode
- H2O-prefill ≈ PostVision인 경우: "generic attention도 충분히 좋다"는 반론 가능 → claim 1 약화
- 이 경우 H2O와 PV의 keep mask overlap 분석으로 원인 진단

---

## 9. What This Experiment Cannot Prove

- **H2O-prefill의 최적 variant**: sum vs mean, all-query vs text-only query 비교는 범위 밖
- **다른 모델에서의 일반화**: LLaVA-1.5-7B 단일 모델 기준
- **더 큰 keep_ratio에서의 차이**: k=0.5 이상에서는 방법 간 차이가 축소될 가능성
- **Random / Recency / Sink baseline**: 추후 추가 가능하나 본 실험에서는 미포함

---

## 10. Runtime / Resources

- **신규 runs**: H2O-prefill만 12 runs (6 dataset × 2 keep_ratio)
- **Run당 시간**: ~3-5 min (H2O는 probe loading 없이 attention만 사용, 비슷하거나 약간 빠름)
- **총 시간**: ~40-60 min (순차, GPU 1개)
- **디스크**: ~12 MB (metrics.json + pred.jsonl)
- **출력 경로**: `/home/M2026107/zap/artifacts/EXP-20260420-002/h2o_prefill/<ds>/h2o_k{0p2,0p1}/`

---

## 11. Implementation Details

### 11.1 H2O-prefill 실행

기존 `evaluate_image_teacher_pruning.py`의 `--mode h2o_image_only` 사용. 추가 코드 수정 불필요.

### 11.2 재현 커맨드 예시

```bash
conda run --no-capture-output -n kv python /home/M2026107/zap/evaluate_image_teacher_pruning.py \
  --mode h2o_image_only \
  --dataset_path /home/M2026107/zap/data/MileBench/CLEVR-Change/CLEVR-Change.json \
  --image_root /home/M2026107/zap/data/MileBench/CLEVR-Change/images \
  --image_column images_path \
  --output_dir /home/M2026107/zap/artifacts/EXP-20260420-002/h2o_prefill/clevr_change/h2o_k0p1 \
  --implementation_model_name /home/M2026107/zap/ckpts/llava-1.5-7b-hf \
  --image_keep_ratio 0.1 \
  --prompt_style look_milebench \
  --attn_implementation eager \
  --truncate_like_lookm \
  --look_dataset_name DocVQA \
  --look_model_name zap_docvqa \
  --look_result_root /home/M2026107/zap/artifacts/combine_prob \
  --device cuda:0
```

### 11.3 Sweep 스크립트

`artifacts/EXP-20260420-002/h2o_sweep_chain.sh` — 12 runs 순차 실행

### 11.4 결과 통합

EXP-20260418-001 (PV, Future), EXP-20260420-001 (Hybrid per-layer), 본 실험 (H2O-prefill) 결과를 합쳐 논문용 Table 생성.

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| H2O-prefill이 예상보다 좋아서 PV와 차이 없음 | Claim 1 약화 | Keep mask overlap 분석으로 "왜 비슷한지" 진단. k=0.1에서 차이 극대화 기대. |
| EXP-20260420-001 (per-layer Hybrid) 아직 미완료 | Hybrid 비교 불가 | H2O sweep 먼저 완료, Hybrid는 EXP-20260420-001 완료 후 합산 |
| H2O mode에서 OOM | 일부 dataset 실패 | `--truncate_like_lookm`으로 방어 (기존과 동일) |
| 기존 PV/Future 결과 경로 불일치 | 비교 테이블 누락 | 결과 수집 스크립트에서 경로 매핑 확인 |

---

## 13. Analysis Plan

### 13.1 논문용 Main Table (Table 1)

| Dataset | k | H2O-prefill | PostVision | Future | Hybrid (best) |
|---|---|---|---|---|---|
| clevr_change | 0.2 | | | | |
| clevr_change | 0.1 | | | | |
| alfred | 0.2 | | | | |
| ... | ... | ... | ... | ... | ... |
| **Average** | | | | | |

### 13.2 Claim 검증

1. **Claim 1**: H2O vs PV paired comparison (Wilcoxon signed-rank or per-dataset delta)
2. **Claim 2**: PV vs Future per-dataset comparison
3. **Claim 3**: Hybrid vs max(PV, Future) per-dataset comparison

### 13.3 추가 분석 (optional)

- H2O와 PV의 keep mask IoU: "얼마나 다른 토큰을 선택하는가"
- Per-layer attention pattern visualization (H2O가 어떤 layer에서 PV와 크게 다른지)

---

## 14. Checklist

실험 시작 전:
- [ ] `--mode h2o_image_only` smoke test (`--limit 5`) 통과
- [ ] Sweep 스크립트 작성 및 실행권한 부여
- [ ] 6 dataset 경로 정확한지 확인
- [ ] EXP-20260418-001 기존 PV/Future 결과 경로 확인

실험 후:
- [ ] 12 runs 전부 `rc=0`
- [ ] `metrics.json`에 ROUGE-L / Accuracy 정상 기록
- [ ] EXP-20260418-001 PV/Future + EXP-20260420-001 Hybrid + 본 실험 H2O 통합 테이블 생성
- [ ] 논문용 Table 1 초안 작성
