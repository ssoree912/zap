# Experiment Plan

**ID:** EXP-20260418-001  
**Author:** ssoree912  
**Date:** 2026-04-18 (plan) / 2026-04-19 (updated)  
**Status:** [ ] Planned  [x] Running (Phase 3 full MileBench evaluation in progress)  [ ] Done  [ ] Abandoned

---

## 구현 실적 (업데이트)

원안 대비 변경 사항:
- **Teacher label 수집**: 독립적인 Future / PostVision 수집이 아니라, **한 번의 LLaVA full pass에서 두 label 동시 추출** (`collect_unified_teacher_shards.py`).
- **Student 학습**: 레이어별 독립 학습 → **one-pass multi-layer, per-sample softmax MSE** (`train_unified_probe_onepass.py`). PV·Future가 동일 loss family로 통일됨.
- **Selected layers**: Future는 last 8 (24-31)로 고정 (last 4보다 조금 더 넓힘). Untrained layer는 layer 31 broadcast 체크포인트로 대체.
- **MLP 구조**: LayerNorm 제거 (`Linear → GELU → Linear`) — KVzapModel 포맷 호환을 위해.
- **v4 학습 결과**: PV best Spearman **0.7512**, Future best **0.7124**.
- **상세 구현**: `implementation.md`, 결과: `results.md` 참고.

---

---

## 1. Motivation

기존 ZAP의 prefill-only probe는 prefill 단계에서 관측 가능한 self-attention 신호만을 기반으로 token 중요도를 추정한다. 그러나 실제 decode 단계에서 어떤 prefill token이 얼마나 다시 참조되는지는 prefill 시점에는 직접 관측할 수 없다. 따라서 prefill-only importance는 실제 미래 사용 패턴과 괴리될 수 있다.

본 실험의 목적은 teacher의 full run attention으로부터 **future decode가 각 prefill token을 얼마나 참조했는지**를 teacher label로 만들고, student MLP가 **prefill hidden state만으로 그 token-wise importance를 예측**할 수 있는지 검증하는 것이다.

핵심 아이디어는 다음과 같다.

- teacher는 full run에서 `decode query → prefill key` attention을 관측할 수 있다.
- 이 attention block을 future 축으로 집계하면, prefill token별 future-conditioned importance score를 얻을 수 있다.
- student는 future를 직접 볼 수 없지만, prefill hidden state만으로 이 token importance score를 근사하도록 학습할 수 있다.

중요하게도, 본 실험에서 student가 학습하는 것은 `T × N` attention matrix 자체가 아니라, 이를 집계하여 얻은 **길이 N의 importance vector**이다.

---

## 2. Hypothesis

teacher의 `decode → prefill` attention 기반 score는 prefill token의 미래 유용도를 나타내는 유의미한 proxy이며, 이 score는 prefill hidden state만으로 일정 수준 예측 가능하다.

구체적 가설은 다음과 같다.

1. student prediction과 teacher label 사이의 rank correlation이 유의미하게 나타난다.
2. student score 기반 top-k pruning은 random보다 downstream accuracy가 높다.
3. student score 기반 pruning은 기존 prefill-only heuristic보다 downstream accuracy가 높다.
4. student는 oracle과 동일하지 않더라도 oracle ranking을 부분적으로 근사할 수 있다.

### Success target
- Spearman ρ ≥ 0.5
- Top-k overlap @ keep ratio 50% ≥ 60%

### Minimum viability
- Random baseline 대비 downstream accuracy 개선
- Prefill-only heuristic 대비 downstream accuracy 개선

---

## 3. Problem Formulation

### 3.1 Sequence setup

prefill token 수를 `N`, future decode token 수를 `T`라고 두고, 전체 시퀀스 길이는 다음과 같다.

$$L = N + T$$

전체 시퀀스는 다음과 같이 표현한다.

$$[P_1, P_2, \dots, P_N \;|\; D_1, D_2, \dots, D_T]$$

여기서

- $P_i$: prefill token
- $D_t$: future decode token

이다.

---

### 3.2 Teacher attention block

teacher의 layer $l$, head $h$에서의 self-attention은

$$A^{(l,h)} \in \mathbb{R}^{L \times L}$$

이다.

이 중에서 우리가 관심 있는 부분은 **future decode token이 query이고, prefill token이 key인 attention block**이다.

$$A_{DP}^{(l,h)} = A^{(l,h)}[N:N+T,\; 0:N] \in \mathbb{R}^{T \times N}$$

즉,

- 행(row): future decode query
- 열(column): prefill key

를 의미한다.

이 `T × N` block은 teacher의 원신호(raw signal)이며, student가 직접 이 행렬 전체를 예측하는 것은 아니다.

---

### 3.3 Teacher label

우리가 최종적으로 필요한 것은 **prefill token마다 score 하나**이므로, `T × N` attention block을 future 방향으로 집계하여 길이 `N`의 vector를 만든다.

선택한 layer 집합을 $\mathcal{L}$, head 수를 $H$라 하면 teacher label은 다음과 같이 정의한다.

$$y_i = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \frac{1}{H} \sum_{h=1}^{H} \frac{1}{T_i} \sum_{t \in \mathcal{T}_i} A_{DP}^{(l,h)}[t,i]$$

여기서

- $y_i$: prefill token $i$의 teacher importance score
- $\mathcal{T}_i$: 유효한 decode step 집합
- $T_i = |\mathcal{T}_i|$

이다.

따라서 최종 teacher target은

$$y = [y_1, y_2, \dots, y_N] \in \mathbb{R}^{N}$$

이다.

즉 teacher는 `T × N` attention matrix를 직접 supervision으로 주는 것이 아니라, 이를 집계한 **N-dimensional token importance vector**를 supervision으로 제공한다.

---

### 3.4 Student prediction

student는 prefill hidden state만 사용한다.

$$H_{\text{prefill}} \in \mathbb{R}^{N \times d}$$

각 prefill token hidden $h_i \in \mathbb{R}^d$에 shared MLP를 적용하여 scalar logit 하나를 예측한다.

$$z_i = g_\theta(h_i), \qquad g_\theta : \mathbb{R}^d \rightarrow \mathbb{R}$$

이를 모든 prefill token에 대해 모으면

$$z = [z_1, z_2, \dots, z_N] \in \mathbb{R}^{N}$$

이 된다.

teacher label과 scale을 맞추기 위해 student output을 softmax로 정규화한다.

$$\hat{y}_i = \frac{\exp(z_i)}{\sum_{j=1}^{N}\exp(z_j)}$$

따라서 최종 student prediction은

$$\hat{y} \in \mathbb{R}^{N}$$

이다.

---

### 3.5 Key clarification

본 실험에서 학습되는 대상은 다음과 같다.

$$A_{DP} \in \mathbb{R}^{T \times N} \;\;\longrightarrow\;\; y \in \mathbb{R}^{N} \;\;\longrightarrow\;\; \hat{y} \in \mathbb{R}^{N}$$

즉,

- teacher raw signal: `T × N`
- teacher supervision target: `N`
- student output: `N`

이다.

**student는 attention matrix 자체를 학습하는 것이 아니라, attention-derived token importance vector를 학습한다.**

---

## 4. Token Scope

1차 실험에서는 pruning 대상 token scope를 다음과 같이 고정한다.

### Default target scope
- **image tokens only**

그 이유는 다음과 같다.

- 실제 pruning 관심 대상이 image-side prefix representation일 가능성이 높다.
- system prompt, template token, formatting token을 포함하면 label 해석이 흐려질 수 있다.
- text template 쪽에 attention mass가 쏠릴 가능성을 줄일 수 있다.

이후 ablation에서 아래 범위를 추가 비교한다.

- image only
- image + instruction text
- all prefill tokens

---

## 5. Future Source Definition

teacher label을 만들기 위해 future decode token이 필요하다.

### Default future source
- teacher LLaVA의 **self-generated future**
- greedy decode 결과를 future token sequence로 사용

이 설정은 inference-time과 더 가까운 supervision을 제공한다.

다만 self-generated future는 teacher의 generation error에 영향을 받을 수 있으므로, 필요 시 아래 비교를 추가한다.

### Optional comparison
- teacher-forced future
- reference answer 기반 future token sequence 사용

1차 실험은 self-generated future를 기본으로 한다.

---

## 6. Overall Pipeline

### Phase A: Teacher label construction

$$[P_1,\dots,P_N \;|\; D_1,\dots,D_T] \rightarrow A^{(l,h)} \in \mathbb{R}^{L \times L} \rightarrow A_{DP}^{(l,h)} \in \mathbb{R}^{T \times N} \rightarrow y \in \mathbb{R}^{N}$$

### Phase B: Student training

$$H_{\text{prefill}} \in \mathbb{R}^{N \times d} \rightarrow \text{MLP} \rightarrow z \in \mathbb{R}^{N} \rightarrow \hat{y} \in \mathbb{R}^{N} \rightarrow \mathcal{L}(\hat{y}, y)$$

### Phase C: Inference pruning

두 가지 scoring 버전으로 분리해서 실험한다.

**Version 1 — Future-only:**

$$H_{\text{prefill}} \rightarrow \text{MLP}_F \rightarrow s^{(F)} \rightarrow \text{Top-k keep} \rightarrow \text{KV prune} \rightarrow \text{decode}$$

**Version 2 — Hybrid (PostVision + Future):**

$$s^{(\text{hybrid})} = \alpha \cdot \text{softmax}(s^{(P)}) + (1-\alpha) \cdot \text{softmax}(s^{(F)}) \rightarrow \text{Top-k keep} \rightarrow \text{KV prune} \rightarrow \text{decode}$$

여기서 $s^{(P)}$는 기존 PostVision MLP(`splus_postvision`) 출력이다.

---

## 7. Teacher Label Design

teacher label은 selected layers $\mathcal{L}$에 대해 다음과 같이 정의한다.

$$y_i = \text{Normalize}\left( \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \frac{1}{H} \sum_{h=1}^{H} \frac{1}{T_i} \sum_{t \in \mathcal{T}_i} A_{DP}^{(l,h)}[t,i] \right)$$

### Default aggregation
- Teacher layers: last 4 layers (`-4, -3, -2, -1`)
- Head aggregation: mean
- Future-step aggregation: mean
- Target-token normalization: sum-to-1 over target token dimension

### Decode horizon rule
- 최대 decode horizon은 `T_max = 64`
- 실제 생성 길이가 더 짧으면 유효 decode step만 사용
- EOS 이후 padding step은 집계에서 제외

---

## 8. Student Model

student는 prefill hidden state를 입력으로 받는 shared token-wise MLP이다.

### Default input
- final hidden state

$$H_{\text{prefill}} \in \mathbb{R}^{N \times d}$$

### Default scorer

$$z_i = g_\theta(h_i)$$

여기서 $g_\theta$는 모든 token에 공유되는 MLP이다.

예시 구조:

```python
class TokenImportanceMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):   # [B, N, D]
        return self.net(h).squeeze(-1)   # [B, N]
```

### Prediction normalization

student logits를 softmax로 정규화하여 importance distribution으로 변환한다.

$$\hat{y}_i = \frac{\exp(z_i)}{\sum_{j=1}^{N}\exp(z_j)}$$

### Input ablation priority

1. final hidden
2. penultimate hidden
3. final + penultimate concat

---

## 9. Loss Function

teacher target ($y$)와 student prediction ($\hat{y}$)는 모두 길이 $N$의 vector이다.

기본 loss는 masked MSE를 사용한다.

$$\mathcal{L}_{\text{mse}} = \frac{\sum_{i=1}^{N} m_i (\hat{y}_i - y_i)^2}{\sum_{i=1}^{N} m_i + \epsilon}$$

여기서

- $m_i \in \{0,1\}$: valid target token mask
- $\epsilon$: numerical stability term

이다.

즉 student는 다음을 학습한다.

$$\hat{y} \approx y$$

초기 안정화를 위해 MSE only로 시작하되, 실제 주요 판단 기준은 rank 보존이다.

### Optional extension

필요 시 pairwise ranking loss를 추가한다.

$$\mathcal{L} = \lambda_{\text{reg}} \mathcal{L}_{\text{mse}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}}$$

---

## 10. Offline Dataset Construction

teacher label 생성과 student 학습을 분리하기 위해 offline dataset construction 단계를 둔다.

### Per-sample stored fields

- sample id
- prefill input ids
- prefill valid mask
- target token mask
- teacher generated ids
- teacher label ($y \in \mathbb{R}^{N}$)
- decode valid-step mask
- metadata for rebuilding hidden states

### Default storage policy

- teacher label은 offline으로 미리 생성
- prefill hidden은 student training 시 online 추출

---

## 11. Inference-time Scoring Versions

추론 시 사용할 score의 출처에 따라 두 버전을 비교한다.

두 버전 모두 학습된 MLP를 사용하며, 추론 시 future token을 직접 참조하지 않는다.
future-supervised MLP는 학습 label만 future-derived일 뿐, 추론 입력은 prefill hidden 전용이다.

### Two scoring sources

| 기호 | 출처 | 설명 |
|------|------|------|
| $s^{(P)}$ | PostVision MLP (기존) | prefill 내 post-vision text → image attention 기반 학습 |
| $s^{(F)}$ | Future-supervised MLP (신규) | decode query → prefill image attention 기반 학습 |

### Version 1: Future-only

$$s = s^{(F)}$$

$$\mathcal{K}_{\text{keep}} = \text{Top-k}(s^{(F)})$$

decode에서 많이 참조될 image token만 남기자는 관점.

### Version 2: Hybrid (PostVision + Future)

두 score를 softmax로 정규화한 뒤 weighted sum으로 결합한다.

$$s^{(\text{hybrid})} = \alpha \cdot \tilde{s}^{(P)} + (1-\alpha) \cdot \tilde{s}^{(F)}$$

여기서

$$\tilde{s}^{(P)} = \text{softmax}(s^{(P)}), \qquad \tilde{s}^{(F)} = \text{softmax}(s^{(F)})$$

$$\mathcal{K}_{\text{keep}} = \text{Top-k}(s^{(\text{hybrid})})$$

$\alpha$ sweep: `0.25, 0.5, 0.75`

현재 중요도($s^{(P)}$)와 미래 유용도($s^{(F)}$)를 함께 반영하는 구조.

### Score normalization 이유

두 MLP의 출력 분포(scale, range)가 다를 수 있으므로 softmax로 먼저 확률 분포로 변환한 뒤 합산한다.

---

## 12. Experimental Conditions

### Main conditions (이번 실험에서 비교할 대상)

| 조건 | 설명 |
|------|------|
| **Future-only** | $s = s^{(F)}$ |
| **Hybrid** | $s = \alpha \tilde{s}^{(P)} + (1-\alpha)\tilde{s}^{(F)}$, $\alpha \in \{0.25, 0.5, 0.75\}$ |

### Baselines

| 조건 | 설명 |
|------|------|
| **PostVision-only** | $s = s^{(P)}$ (기존 ZAP probe, 재학습 없이 그대로 사용) |
| **Random** | 동일 keep ratio에서 무작위 keep |
| **Oracle** | teacher label $y \in \mathbb{R}^{N}$ 직접 사용 (상한선) |

Oracle은 아래 조건을 main conditions와 동일하게 고정한다.

- selected layers
- head aggregation
- future aggregation
- token scope
- keep ratio

---

## 13. Metrics

### 12.1 Label Prediction Quality

| 지표 | 설명 |
|------|------|
| MSE / MAE | 절대 수치 오차 |
| **Spearman ρ** | rank 보존 여부 (핵심) |
| Kendall τ | rank 보존 여부 (보조) |
| **Top-k overlap** | keep ratio 50% 기준 overlap (핵심) |

### 12.2 Downstream Pruning Quality

keep ratio:

- 90%
- 70%
- 50%
- 30%

평가 항목:

- task accuracy
- memory usage
- latency

### Benchmark

- MileBench
- 기존 ZAP 평가 설정과 통일

---

## 14. Ablation Plan

### A. Teacher label layer range
- last 1
- last 4
- all layers

### B. Student input layer
- final
- penultimate
- final + penultimate concat

### C. Loss
- MSE only
- MSE + ranking

### D. Target token scope
- image only
- image + instruction
- all prefill

### E. Decode horizon
- fixed 64
- variable valid length
- early decode only

### Default first experiment

- A = last 4
- B = final
- C = MSE
- D = image only
- E = fixed `T_max = 64`, valid-step masking applied

---

## 15. Implementation Phases

### Phase 1. Teacher label pipeline verification

- [ ] `build_teacher_labels()` 구현
- [ ] shape 검증: `N=10`, `T=5`
  - attention shape: `[B, H, 15, 15]`
  - slice: `[B, H, 5, 10]`
  - aggregate 후 label: `[B, 10]`
- [ ] normalization 검증
- [ ] label distribution 시각화

### Phase 2. Student training

- [ ] `TokenImportanceMLP` 구현
- [ ] prefill-only hidden 추출
- [ ] MLP prediction
- [ ] masked MSE 계산
- [ ] validation에서 Spearman / top-k overlap 로깅

### Phase 3. Pruning + downstream evaluation

- [ ] predicted vector $\hat{y}$로 top-k keep
- [ ] KV prune → decode 연결
- [ ] Ours / Random / Prefill-heuristic / Oracle 비교

---

## 16. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| attention score가 noisy함 | single-layer vs multi-layer aggregation 비교 |
| 절대 score보다 순위가 더 중요함 | Spearman, top-k overlap 필수 측정 |
| text token 쪽으로 label이 편중될 수 있음 | image-only scope를 기본값으로 사용 |
| decode 길이에 민감할 수 있음 | valid-step masking 적용 |
| self-generated future가 noisy할 수 있음 | teacher-forced future와 label 안정성 비교 |
| student가 future dynamics를 충분히 못 담을 수 있음 | oracle과의 gap을 정량화 |

---

## 17. Checklist

- [ ] teacher raw attention block shape 검증: `T × N`
- [ ] teacher label vector shape 검증: `N`
- [ ] student output vector shape 검증: `N`
- [ ] decode valid-step masking 검증
- [ ] target token mask 검증
- [ ] label normalization 검증
- [ ] Spearman 측정 파이프라인 구축
- [ ] top-k overlap 측정 파이프라인 구축
- [ ] random baseline 비교 가능 상태 확보
- [ ] prefill-heuristic baseline 비교 가능 상태 확보

---

## 18. One-line Summary

본 실험은 `future decode → prefill attention`으로부터 얻은 raw signal

$$A_{DP} \in \mathbb{R}^{T \times N}$$

을 직접 학습하는 것이 아니라, 이를 집계하여 만든 prefill token importance vector

$$y \in \mathbb{R}^{N}$$

를 student가 prefill hidden만으로 예측하도록 학습하는 실험이다.

$$A_{DP} \in \mathbb{R}^{T \times N} \rightarrow y \in \mathbb{R}^{N} \rightarrow \hat{y} \in \mathbb{R}^{N}$$
