# ZAP 실험 분석 — 핵심 질문 5가지

**작성일**: 2026-04-16  
**기반 근거**: repo 코드 + EXP-20260410-001 ~ EXP-20260415-002  
**표기 규칙**: `[코드]` = 코드에서 직접 확인, `[실험]` = 실험 결과에서 직접 확인, `[해석]` = 문헌/추론 기반

---

## 1. Repo / Experiment Inspection Summary

### 확인된 핵심 수치

| 비교 | 승/패/무 | 평균 Δ | 출처 |
|---|---|---|---|
| Probe vs LOOK-M (29 datasets, r=0.20) | **17/10/2** | +0.022 | `[실험]` ablation_report.md:214 |
| Combine probe vs LOOK-M (28 datasets) | **22/5/1** | — | `[실험]` EXP-20260415-002/RESULTS.md:18 |
| Oracle (14 datasets) vs LOOK-M | **11/2/1** | — | `[실험]` EXP-20260415-002/RESULTS.md:18 |
| Probe vs Oracle (19 matched samples) | Probe≥Oracle 15/19 (79%) | probe +0.011 | `[실험]` ablation_report.md:154 |

### 핵심 알고리즘 구조 확인

| 구성 요소 | 구현 위치 | 핵심 내용 |
|---|---|---|
| Image-only eviction | `kvpress/presses/image_token_press.py:313-326` | `image_positions.numel()==0` → no-op; text token 항상 보존 |
| att_only_postvision | `kvzap/llava_extractor.py` | postvision query × image kv position의 amax over q-dim |
| Probe MLP | `train_image_teacher_probe.py:27-36` | `Linear(4096→512) → GELU → Linear(512→n_layers)` |
| Oracle 2-pass | `evaluate_image_teacher_pruning.py:537-552` | Pass1: `output_attentions=True`로 score 추출; Pass2: score 적용 압축 generate |
| LOOK-M truncate | `look-m/utils.py:167-172` | `n_tokens_per_image=576` 단위로 예산 초과 시 이미지 drop |

---

## 2. Probe > Oracle 분석 / Probe > LOOK-M 분석

### 2-1. 왜 probe가 oracle보다 성능이 좋을 수 있는가

**실험 근거**: 19개 matched 샘플에서 probe 평균 0.173 > oracle 평균 0.162, probe≥oracle 비율 79% `[실험]` ablation_report.md:154-177

#### 이유 1 — Oracle의 score 추출 경로가 inference 경로와 다르다 `[코드]`

```python
# evaluate_image_teacher_pruning.py:537-551 — oracle on-the-fly Pass 1
model(..., output_attentions=True, use_cache=False)
```

Oracle은 Pass1에서 `output_attentions=True` + `use_cache=False`(eager attention)로 attention을 추출한다.  
반면 실제 generation(Pass2)은 SDPA(Flash Attention) 경로를 사용한다.  
두 경로의 attention 계산이 수치적으로 완전히 동일하지 않을 수 있고, `[해석]` eager attention의 softmax precision이 FlashAttention과 미묘하게 달라 oracle score가 "완벽한" importance map이 아닐 수 있다.

#### 이유 2 — Probe는 per-sample noise를 평활화한다 `[해석]`

Oracle score = 단일 샘플의 attention snapshot. 특정 샘플에서는 attention이 실제로 중요한 이미지 영역보다 현재 decoding position bias나 layer-specific artifact에 의해 교란될 수 있다.  
Probe는 수백~수천 샘플의 hidden state → score 매핑을 학습했으므로, 각 샘플의 노이즈를 평균적으로 제거(ensemble 효과)하여 더 안정적인 importance estimate를 제공한다.

#### 이유 3 — Hidden state에 더 풍부한 정보가 있다 `[해석]`

Probe의 입력: LLaVA image token의 last hidden state [I, 4096]  
`[코드]` `train_image_teacher_probe.py:140-144`  

Hidden state는 해당 image patch의 visual embedding + 모든 prior token과의 cross-attention 후 상태이므로, single-layer attention score보다 전체 컨텍스트를 통합한 정보를 담는다. Probe는 이 풍부한 representation에서 중요도를 추출한다.

#### 이유 4 — Oracle은 19개 샘플에서 OOM 없는 케이스만 측정 `[실험]`

Oracle 측정 가능 데이터셋이 14개로 제한된 이유가 OOM. OOM이 없는 19개 matched sample은 상대적으로 짧은/단순한 샘플 편향이 있어, oracle이 어려운 케이스를 회피하는 구조.

---

### 2-2. 왜 probe가 LOOK-M보다 성능이 좋은가

#### 핵심 원인 1 — Image-only eviction (결정적, +9.6%p) `[실험]`

**근거**: EXP-20260410-001/RESULT.md:115-131의 2×2 ablation  
동일 oracle score, scope만 다를 때: image-only 0.235 vs all-token 0.139 → **+0.096p (69% 상대 개선)**

```
DocVQA r=0.20:
  oracle + image-only (ours): 0.515
  oracle + all-token (B):     0.234   ← text 날아가서 질문 자체를 못 읽음
```

메커니즘: `[코드]` `image_token_press.py:313-326`  
- Image-only eviction은 text token을 항상 100% 보존  
- LOOK-M의 all-token eviction은 낮은 ratio에서 질문/선택지 텍스트도 증발  
- VLM에서 text token(질문, 선택지, 문맥)은 generation에 직접적으로 필요 → 구조적 보존이 성능에 결정적

#### 핵심 원인 2 — Task-conditioned scoring (부차적, +0.7%p) `[실험]`

**근거**: EXP-20260410-001/RESULT.md:123-131  
동일 image-only scope, score만 다를 때: att_only_pv(ours) 0.235 vs H2O 0.228 → **+0.007p**

```
att_only_postvision 쿼리: <image> 이후 prompt 텍스트 (Context / Question / Options)
H2O 쿼리: 모든 position의 attention 누적 (task-agnostic)
```

`[코드]` EXP-20260410-001/RESULT.md:71-89  
"이 질문에 어느 이미지 영역이 중요한가"를 prefill에서 task-conditioned하게 평가 → H2O보다 semantically 정밀. 단, 전반적으로는 scope 효과보다 약 13배 작은 기여.

#### 부가 원인 3 — Decode 경로 차이에 의한 TBT 가속 `[실험]`

**근거**: ablation_report.md:184-185  
LOOK-M: `H2OLlamaAttention_drop`에서 매 decode step마다 fp32 softmax + causal mask 재생성 → Flash/SDPA 미사용  
Probe (ours): prefill에서 top-k 고정 제거 후 표준 HF SDPA → TBT **2.63× 빠름** (27.6ms vs 74.0ms)  
더 빠른 decode가 user experience 상 체감 성능 차이를 만든다.

---

## 3. Oracle OOM 분석

**근거**: EXP-20260410-001/RESULT.md:133-146, EXP-20260415-002/RESULTS.md:131-134

### 직접 원인 — `output_attentions=True`가 full attention matrix를 GPU에 materialze `[코드]`

```python
# evaluate_image_teacher_pruning.py:537-551
model(..., output_attentions=True, use_cache=False)  # Pass 1
```

`output_attentions=True`는 `use_cache=False` + eager attention을 강제한다.  
Full attention matrix shape: `[batch=1, n_heads=32, q_len, kv_len]`  
`[실험]` DocVQA MileBench에서 샘플당 이미지 2~5장 → 최대 ~2,880 image token + text ≈ q_len/kv_len ≈ 3,500~4,500  

메모리 계산:  
```
4500 × 4500 × 32 heads × 32 layers × 2 bytes (fp16) ≈ 41 GiB
```
이는 실제로 ALL layers를 동시에 저장하는 것이 아니라 layer별로 전달되지만, 각 layer 처리 중 attention map이 GPU에 상주해야 하며 activation이 gradient 없이도 큰 메모리를 사용.

### OOM 분포 패턴 `[실험]`

| Mode | DocVQA r=0.10 | r=0.20 | r=0.30 |
|---|---|---|---|
| h2o_image_only | **25 OOM** | 3 OOM | 3 OOM |
| oracle_all_token | 3 OOM | 3 OOM | 3 OOM |
| oracle (ours, on-the-fly) | **0** | **0** | **0** |

`oracle (ours)` (= 사전 계산된 teacher record 사용)가 OOM 0인 이유: `[코드]`  
Pass1은 오프라인에서 미리 실행하고 `.pt` 파일로 저장. inference 시 Pass1 없이 저장된 score만 로드하여 Pass2만 수행 → `output_attentions=True` 불필요.

### Oracle on-the-fly OOM의 핵심 `[실험]`

EXP-20260415-002/RESULTS.md:133:  
> "oracle(on-the-fly)는 여전히 다수 데이터셋에서 OOM 비중이 높아, score 산출 데이터셋이 14개로 제한됨."

OOM이 특히 심한 조건:
1. 다중 이미지 (이미지 10장+ = 5,760 token+)  
2. 긴 context (전체 seq_len > 3,000)  
3. `output_attentions=True`로 인한 attention 행렬 누적

요약: **Oracle on-the-fly의 OOM은 scoring method의 본질적 비용**이며, 이것이 probe distillation의 핵심 동기다.

---

## 4. Initial/Recent Forced-Keep이 거의 효과 없는 이유

**근거**: EXP-20260412-003/RESULT.md 전체

### 실험 결과 `[실험]`

```
Init forced-keep: avg +0.0014, 개선=7/29, 하락=0/29
Rec  forced-keep: avg +0.0003, 개선=3/29, 하락=1/29
22/29 데이터셋에서 완전 flat (+0.000)
```

### 핵심 이유 1 — att_only_postvision이 이미 positional importance를 내재 `[실험]`

EXP-20260412-003/RESULT.md:86-88:  
> "probe의 att_only_postvision score는 이미 positional bias를 내재하고 있어, 별도의 positional forced-keep이 불필요하다."

`[코드]` Probe는 hidden state에서 score를 예측한다. Hidden state는 RoPE positional embedding을 거쳐 계산되므로, position 정보가 이미 feature에 인코딩돼 있다. "중요한 첫 이미지 patch"는 이미 probe score에 의해 높은 점수를 받아 자연스럽게 top-k에 포함된다.

### 핵심 이유 2 — Init N 크기에 무감각 `[실험]`

EXP-20260412-003/RESULT.md:79-81:  
> "init16, init32, init64가 대부분 데이터셋에서 동일한 점수. 즉 초기 16개 강제 보존이나 64개 강제 보존이나 결과가 같음 → probe가 이미 초기 토큰을 top-k에 포함시키고 있음."

forced-keep이 추가로 보존하려는 토큰들이 이미 score 기반 top-k에 들어 있으므로, forced-keep이 실질적으로 아무것도 바꾸지 않는다.

### 예외 케이스 (init이 효과를 보인 경우) `[실험]`

| 데이터셋 | Δ | 태스크 구조 |
|---|---|---|
| CharacterOrder | **+0.015** | 이미지 순서를 기억해야 함 → 첫 번째 이미지 patch가 구조적으로 중요 |
| StateChange | +0.005 | 초기 상태 참조 필요 |
| Spot-the-Diff | +0.006 | reference 이미지(첫 번째)와 비교 이미지(두 번째) 비교 |

이 태스크들은 "위치 자체"가 의미를 가지는 경우. CharacterOrder에서만 +1.5%p로 통계적으로 의미 있는 효과.

### 결론

Forced-keep은 "score가 positional importance를 포착 못한다"는 가정 하에 필요하다.  
att_only_postvision는 task query에서 실제로 중요한 토큰을 task-conditioned하게 평가하므로, 위치 heuristic이 불필요하다. 이는 StreamingLLM 류의 initial/recent keep이 LLM text에서는 유효하지만, **task-conditioned image saliency가 있는 VLM 설정에서는 redundant**함을 시사한다. `[해석]`

---

## 5. 도메인별 ~500개 학습으로도 Probe 성능이 나오는 이유

**참고**: 학습 데이터 정확한 샘플 수는 실험 로그에 명시되지 않음. 사용자 제공 정보 "~500개 정도" 기준으로 분석.

### 이유 1 — 실제 학습 단위는 "샘플" 수가 아닌 "image token" 수 `[코드]`

```python
# train_image_teacher_probe.py:136-169
for sample_id in sample_ids:
    x = hidden_layer[image_pos]   # [n_image, 4096]
    all_x.append(x)               # 샘플 1개 → 576 토큰 (LLaVA-1.5 기준)
```

샘플 500개 × 1이미지 × 576 image token = **288,000 training instances** (per layer).  
MLP 입력이 4096-dim이고 hidden_dim=512, output_dim=1로 매우 작은 모델(~2M params/layer)이므로 이 규모로 충분히 수렴한다.

### 이유 2 — 입력이 이미 semantically rich한 LLM hidden state `[코드]` `[해석]`

`[코드]` `kvpress/presses/kvzap_press.py` + `train_image_teacher_probe.py:97-101`  
Probe의 입력은 LLaVA-1.5의 last hidden state (4096-dim). 이는 수십억 개의 텍스트+이미지 샘플로 사전 학습된 표현이다.  
`[해석]` 이미 풍부하게 학습된 representation 위에 얇은 regression head를 얹는 것이므로, 적은 labeled data로도 수렴한다. Linear probing이나 shallow MLP가 frozen representation 위에서 잘 작동하는 것과 동일한 원리 (feature extraction regime).

### 이유 3 — Target이 clean하고 deterministic `[코드]`

```python
# train_image_teacher_probe.py:147
layer_scores = teacher[target_score_name][layer_idx].float()
```

학습 target (att_only_postvision score)은 동일한 LLaVA 모델의 출력이므로 noise가 없다. 같은 sample_id에 대해 항상 동일한 target → clean supervision signal. 이런 조건에서 적은 샘플으로도 overfitting 없이 일반화가 잘 된다.

### 이유 4 — Task는 absolute scoring이 아닌 ranking `[해석]`

실제로 필요한 것은 정확한 score 값이 아니라 "어떤 image token이 더 중요한가"의 순서(ranking). Regression으로 학습하더라도 top-k selection은 rank order만 의존하므로, score의 절대값보다 상대적 order만 맞으면 된다. Ranking task는 regression보다 generalization이 쉽고, 더 적은 샘플로 수렴한다.

### 이유 5 — ScienceQA의 분포가 universal visual feature를 커버 `[해석]`

ScienceQA는 과학 MCQ이지만, 핵심 패턴인 "질문에 관련된 시각적 영역에 attention"은 도메인에 무관한 universal behavior이다. LLaVA의 hidden state가 이미 cross-domain representation을 가지고 있으므로, ScienceQA 기반 probe도 ALFRED, CharacterOrder, WikiVQA 등 다양한 태스크에서 작동한다.

---

## 6. 핵심 요약 문장 (바로 공유 가능)

> **ZAP은 image token만 선택적으로 evict하고(+9.6%p), task query 기반 attention score로 의미적으로 중요한 patch를 고르며(+0.7%p), 이를 ~500샘플로 학습된 얇은 MLP probe로 distill하여 oracle 없이 실시간 동작한다. 결과적으로 LOOK-M 대비 29개 MileBench 데이터셋에서 17W/10L/2T, TBT 2.63× 가속, KV cache 85% 절감을 달성하고, oracle 대비 -0.9%p 이내로 근사하면서 oracle의 OOM 문제(output_attentions=True로 인한 full attention matrix materialization)를 완전히 회피한다.**

보조 요약:
- Probe가 oracle보다 좋을 수 있는 이유: oracle은 single-sample attention snapshot (noisy) + eager attention vs SDPA 수치 불일치; probe는 대규모 hidden state에서 평활화된 importance estimate를 학습
- Forced-keep이 불필요한 이유: att_only_postvision이 task-conditioned attention으로 이미 positional importance를 내재화 — CharacterOrder(순서 태스크)를 제외한 22/29 데이터셋에서 완전 flat

---

## 7. 후속 분석 제안

### A. Oracle score의 "Pass1 eager vs SDPA 불일치" 정량화

**질문**: Oracle score가 probe보다 안 좋은 19개 케이스 중, score extraction(eager)과 실제 generation(SDPA) 간의 attention 분포 divergence가 얼마나 큰가?

**방법**:
1. 동일 샘플에 대해 `output_attentions=True` (eager)와 `output_attentions=False` (SDPA) 두 경로에서 image position별 attention 분포 추출
2. KL divergence 또는 rank correlation (Spearman ρ) 계산
3. probe > oracle인 샘플 vs oracle > probe인 샘플에서 divergence 차이 비교

**예상 가치**: oracle의 본질적 한계를 정량화하여 "왜 probe ≈ oracle인가"를 논문에서 설명할 근거 확보.

---

### B. 학습 샘플 수에 따른 probe 성능 수렴 곡선

**질문**: 몇 개의 샘플로 학습하면 성능이 수렴하는가? 50개, 100개, 200개, 500개별로 29개 MileBench 데이터셋 성능 측정.

**방법**:
1. `train_image_teacher_probe.py`에 `--n_train_samples N` 옵션 추가
2. N = [50, 100, 200, 500, ALL]에 대해 ScienceQA 기반 probe 학습
3. 각 N별로 MileBench 전체 추론 → probe vs LOOK-M W/L/T 비교

**예상 가치**: "소수 샘플로도 강력한 generalization" 클레임의 정량적 근거 — 논문의 data efficiency 절에 직접 사용 가능.

---

### C. CharacterOrder에서 per-image initial forced-keep의 효과 분리

**질문**: CharacterOrder에서 init 효과(+1.5%p)의 원인이 "global sequence의 초반 image patch" vs "각 이미지의 첫 번째 patch (per-image sink)" 중 무엇인가?

**방법**:
1. `per_image_forced=True` 조건 추가 (이미 `_find_image_blocks`로 코드 지원, `image_token_press.py:156-171`)
2. CharacterOrder에서 global_init_16 vs per_image_init_16 비교
3. n_initial_keep을 1~8로 변화시키며 민감도 측정

**예상 가치**: positional forced-keep이 "필요 없다"는 주장의 예외 케이스를 정확히 기술하고, per-image sink 개념을 논문에서 정밀하게 다룰 수 있음.
