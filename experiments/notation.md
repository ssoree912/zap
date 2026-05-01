네, 현재 본문을 기준으로 수식 변수 표를 다시 정리하겠습니다. 이전 표에서 다이어트로 사라진 항목들을 빼고, 새로 추가된 항목들을 반영했습니다.

---

# 수식 변수 정의표 (수식 다이어트 후 최종본)

## A. 모델 / 입력 관련 (§3.1 Preliminaries)

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $L$ | scalar (int) | LVLM의 transformer layer 수 | §3.1 |
| $D$ | scalar (int) | LVLM의 hidden dimension | §3.1 |
| $N_h$ | scalar (int) | LVLM의 attention head 수 | §3.1 |
| $d_h$ | scalar (int) | Per-head dimension; $d_h = D/N_h$ | §3.1 |
| $N_{\mathrm{prompt}}$ | scalar (int) | Multimodal prompt 토큰 시퀀스 길이 | §3.1 |
| $X^{(l)}$ | $\mathbb{R}^{N_{\mathrm{prompt}} \times D}$ | 레이어 $l$의 prefill hidden state 행렬 | §3.1 |
| $\mathcal{I}_{\mathrm{img}}$ | index set | Image token이 차지하는 prompt 위치 | §3.1 |
| $\mathcal{I}_{\mathrm{text}}$ | index set | Text token이 차지하는 prompt 위치; $\mathcal{I}_{\mathrm{img}} \cup \mathcal{I}_{\mathrm{text}}$가 전체 prompt | §3.1 |
| $\mathcal{I}_q$ | index set $\subseteq \mathcal{I}_{\mathrm{text}}$ | Question conditioning에 사용되는 user-question token 위치 | §3.1 |
| $N_I$ | scalar (int) | $|\mathcal{I}_{\mathrm{img}}|$, image token 개수 | §3.1 |
| $N_Q$ | scalar (int) | $|\mathcal{I}_q|$, question token 개수 | §3.1 |
| $N_{\mathrm{text}}$ | scalar (int) | $|\mathcal{I}_{\mathrm{text}}|$, text token 개수 | §3.1 |
| $\pi_i$ | $\in \mathcal{I}_{\mathrm{img}}$ | 로컬 image-token 인덱스 $i \in \{1, \ldots, N_I\}$에 대응되는 prompt 위치 | §3.1 |
| $X_{\mathrm{img}}^{(l)}$ | $\mathbb{R}^{N_I \times D}$ | 레이어 $l$의 image-token hidden state 행렬 | §3.1 |
| $X_q^{(l)}$ | $\mathbb{R}^{N_Q \times D}$ | 레이어 $l$의 question-token hidden state 행렬 | §3.1 |
| $x_i^{v,(l)}$ | $\mathbb{R}^{D}$ | $i$-번째 image token의 hidden state ($X_{\mathrm{img}}^{(l)}$의 $i$-번째 행) | §3.1 |
| $T$ | scalar (int) | Teacher 계산에 사용되는 답변 길이 | §3.1 |
| $d$ | scalar (int) | Student의 projection dimension; 일반적으로 $d \ll D$ | §3.1 |

## B. Future-Attention Teacher (§3.2)

### B.1 Attention extraction (Eq.~\eqref{eq:future_attn})

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $q_t^{(l,m)}$ | $\mathbb{R}^{d_h}$ | 레이어 $l$, head $m$, decoding step $t$의 답변 토큰 query 벡터 | Eq.~\eqref{eq:future_attn} |
| $k_p^{(l,m)}$ | $\mathbb{R}^{d_h}$ | 레이어 $l$, head $m$, key 위치 $p$의 key 벡터 | Eq.~\eqref{eq:future_attn} |
| $\mathcal{K}_t$ | index set | Decoding step $t$에서 query가 attend 가능한 key 위치 (causal mask 하의 prompt + 그 이전 답변 토큰들) | Eq.~\eqref{eq:future_attn} 아래 |
| $a_{t,i}^{(l,m)}$ | scalar $\in [0, 1]$ | 답변 토큰 $t$ → $i$-번째 image token의 attention weight. **softmax는 $\mathcal{K}_t$ 전체에 걸쳐 정규화**하되, image-token 위치 $\pi_i$ 항만 추출 ⇒ 일반적으로 $\sum_i a_{t,i}^{(l,m)} \neq 1$ | Eq.~\eqref{eq:future_attn} |

### B.2 Teacher score (Eq.~\eqref{eq:future_teacher}, \eqref{eq:teacher_norm})

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $s_i^{(l), \mathrm{teacher}}$ | scalar $\geq 0$ | Head·step 평균된 layer-$l$ raw teacher score for image token $i$ | Eq.~\eqref{eq:future_teacher} |
| $s^{(l),\mathrm{teacher}}$ | $\mathbb{R}^{N_I}$ | Raw teacher score 벡터 | §3.1, §3.2 |
| $\epsilon$ | scalar $> 0$ | 정규화 분모의 수치적 안정화 상수 | Eq.~\eqref{eq:teacher_norm} |
| $y_i^{(l), \mathrm{teacher}}$ | scalar $\in [0, 1]$ | Image token에 대한 teacher distribution; $\epsilon$ 무시 시 $\sum_i y_i = 1$ | Eq.~\eqref{eq:teacher_norm} |
| $y^{(l),\mathrm{teacher}}$ | $\mathbb{R}^{N_I}$ | Teacher 정규화 분포 벡터 | §3.1, §3.2 |

## C. Question-Conditioned Visual-Utility Student (§3.3)

### C.1 Student의 함수 표현

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $f_\theta^{(l)}$ | function | 레이어 $l$의 학생 모듈; 입력 $(X_{\mathrm{img}}^{(l)}, X_q^{(l)})$로부터 $\hat{s}^{(l)}$ 산출 | §3.3 첫 단락 |
| $\theta$ | parameter set | Student의 모든 학습 가능 파라미터: $\theta = \{W_q, W_v, \phi, \psi\}$ | §3.3 (Eq.~\eqref{eq:student_score} 직후) |

### C.2 Projection branches (Eq.~\eqref{eq:student_q}, \eqref{eq:student_v})

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $W_q$ | $\mathbb{R}^{d \times D}$ | Question representation의 projection 행렬 (학습) | Eq.~\eqref{eq:student_q} |
| $\bar{q}^{(l)}$ | $\mathbb{R}^{d}$ | Question token mean-pool 후 $W_q$로 projection된 question representation | Eq.~\eqref{eq:student_q} |
| $W_v$ | $\mathbb{R}^{d \times D}$ | Image-token-level의 projection 행렬 (학습) | Eq.~\eqref{eq:student_v} |
| $\bar{x}_i^{v,(l)}$ | $\mathbb{R}^{d}$ | $i$-번째 image token의 projected visual feature | Eq.~\eqref{eq:student_v} |

### C.3 Visual context encoder (식 없이 prose만)

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $g_\phi$ | function $\mathbb{R}^{N_I \times D} \to \mathbb{R}^{N_I \times d}$ | Visual context encoder; backbone의 image-token geometry에 따라 instantiate (fixed-grid → 2D conv, AnyRes → layout-safe 1D conv) | §3.3 prose |
| $\phi$ | parameter set | $g_\phi$의 학습 파라미터 ($\theta$의 부분집합) | §3.3 prose |
| $\bar{C}^{(l)}$ | $\mathbb{R}^{N_I \times d}$ | $g_\phi$ 출력으로 얻은 context-aware visual feature 행렬 | §3.3 prose |
| $\bar{c}_i^{(l)}$ | $\mathbb{R}^{d}$ | $i$-번째 image token의 context-aware feature ($\bar{C}^{(l)}$의 $i$-번째 행) | §3.3 prose |

### C.4 Fusion (Eq.~\eqref{eq:student_fusion})

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $\odot$ | binary op | Element-wise (Hadamard) product | Eq.~\eqref{eq:student_fusion} 직후 |
| $z_i^{(l)}$ | $\mathbb{R}^{5d}$ | Image token $i$의 fusion vector: 5개 항 concat — projected visual, context, question, visual·question Hadamard, context·question Hadamard | Eq.~\eqref{eq:student_fusion} |

### C.5 MLP head (Eq.~\eqref{eq:student_score})

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $\mathrm{MLP}_\psi$ | function $\mathbb{R}^{5d} \to \mathbb{R}$ | 2-layer MLP head, GELU activation, hidden size $d_{\mathrm{mlp}}$ | Eq.~\eqref{eq:student_score} |
| $\psi$ | parameter set | $\mathrm{MLP}_\psi$의 학습 파라미터 ($\theta$의 부분집합) | §3.3 (Eq.~\eqref{eq:student_score} 직후) |
| $d_{\mathrm{mlp}}$ | scalar (int) | MLP의 hidden size | §3.3 prose |
| $\hat{s}_i^{(l)}$ | scalar | Image token $i$의 student raw utility score | Eq.~\eqref{eq:student_score} |
| $\hat{s}^{(l)}$ | $\mathbb{R}^{N_I}$ | Student raw utility score 벡터 | §3.1, §3.3 |

### C.6 학습 (식 없이 prose + Eq.~\eqref{eq:student_loss})

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $\hat{y}^{(l)}$ | $\mathbb{R}^{N_I}$ | $\hat{s}^{(l)}$를 image token 차원에서 softmax 정규화한 student-predicted distribution | §3.3 prose ("normalize the raw scores ... with a softmax") |
| $\mathcal{L}^{(l)}$ | scalar (loss) | 레이어 $l$의 distillation loss | Eq.~\eqref{eq:student_loss} |
| $\mathrm{MSE}(\cdot, \cdot)$ | scalar (loss) | Distribution regression loss term | Eq.~\eqref{eq:student_loss} |
| $\mathcal{L}_{\mathrm{rank}}$ | scalar (loss) | Ranking loss; teacher가 부여한 image-token 순서를 보존하도록 강제 (구체적 형태는 본문에 미명시) | Eq.~\eqref{eq:student_loss} |
| $\lambda_{\mathrm{rank}}$ | scalar $> 0$ | Ranking loss의 가중 hyperparameter | Eq.~\eqref{eq:student_loss} |

## D. Inference-time Pruning (§3.5)

| 기호 | 차원 / 타입 | 정의 | 출처 |
|---|---|---|---|
| $\mathcal{S}$ | index set $\subseteq \{1, \ldots, L\}$ | Eviction이 적용되는 layer 집합 (이외 레이어는 full cache 유지) | §3.5 |
| $r$ | $\in (0, 1]$ | Image-token keep ratio (사용자 지정 budget) | §3.5 |
| $k_I$ | scalar (int) | 레이어별 유지할 image token 수; $k_I = \lceil r N_I \rceil$ | §3.5 |
| $\mathrm{TopK}(s, k)$ | function | 벡터 $s$에서 값이 가장 큰 $k$개 원소의 인덱스 집합 | Eq.~\eqref{eq:retained_set} |
| $\mathcal{R}^{(l)}$ | index set $\subseteq \{1, \ldots, N_I\}$ | $\mathrm{TopK}(\hat{s}^{(l)}, k_I)$로 정의되는 retained image-token 인덱스 집합 | Eq.~\eqref{eq:retained_set} |
| $\{\pi_i : i \in \mathcal{R}^{(l)}\}$ | index set $\subseteq \mathcal{I}_{\mathrm{img}}$ | Retained image token들의 prompt 위치 — 이 위치의 KV entry만 유지 | §3.5 |

---

# 본문에 등장하는 모든 식의 빠른 reference

| 식 번호 | 무엇을 정의하나 | 위치 |
|---|---|---|
| Eq.~\eqref{eq:future_attn} | Answer-to-image attention weight $a_{t,i}^{(l,m)}$ — softmax over $\mathcal{K}_t$, then extract at $\pi_i$ | §3.2 |
| Eq.~\eqref{eq:future_teacher} | Raw teacher score $s_i^{(l),\mathrm{teacher}}$ — head·step 평균 | §3.2 |
| Eq.~\eqref{eq:teacher_norm} | Teacher distribution $y_i^{(l),\mathrm{teacher}}$ — $\epsilon$-smoothed L1 normalize | §3.2 |
| Eq.~\eqref{eq:student_q} | Question representation $\bar{q}^{(l)}$ — mean-pool + $W_q$ | §3.3 |
| Eq.~\eqref{eq:student_v} | Projected visual feature $\bar{x}_i^{v,(l)}$ — $W_v$ projection | §3.3 |
| Eq.~\eqref{eq:student_fusion} | Fusion vector $z_i^{(l)}$ — 5-way concat with Hadamard | §3.3 |
| Eq.~\eqref{eq:student_score} | Raw utility score $\hat{s}_i^{(l)}$ — MLP head | §3.3 |
| Eq.~\eqref{eq:student_loss} | Distillation loss $\mathcal{L}^{(l)}$ — MSE + ranking | §3.3 |
| Eq.~\eqref{eq:retained_set} | Retained image-token set $\mathcal{R}^{(l)}$ — TopK | §3.5 |


