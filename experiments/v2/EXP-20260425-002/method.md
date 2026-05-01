
# Future-supervised image-token eviction for LLaVA-1.5

**Branch**: `4090/cnn_image_scorer`
**Production ckpt**: `/workspace/zap/ckpts/student_v2_A_gqa_lr1e4/`
**Author**: ssoree912

---

## 1. 동기

LVLM (Large Vision-Language Model) 의 KV cache 는 image patch 토큰이 절대 다수를 차지한다.
LLaVA-1.5 의 경우 한 prompt 당 **576 개 image token + 30~60 text token** 으로,
image 만 약 90 % 의 cache 메모리를 점유한다. 따라서 image 토큰을 선택적으로
evict 하면 큰 메모리 절감이 가능하다.

문제는 **어느 image patch 가 답변 생성에 정말 중요한가** 를 prefill 시점에 알기
어렵다는 것이다. 본 연구는 이 결정을 학습된 lightweight scorer 에게 맡긴다:

[
\hat{s}_i^{(\ell)} = \text{Student}*\theta\big(H^{(\ell)}*{\text{img}}, H_q^{(\ell)}\big), \quad i=1,\dots,N_I
]

teacher 는 동일 모델의 **future decode attention** (정답 생성 단계에서 image
patch 들에 실제로 가는 attention) 으로 구한다.

---

## 2. 학습 데이터

총 **1,500 sample** (3 dataset × 500), 모두 single-image:

| Dataset | 샘플 수 | 출처 | 평균 text 길이 | 특성 |
|---|---|---|---|---|
| ScienceQA train | 500 | `/workspace/zap/data/scienceqa` | ~57 token | 과학 다이어그램 + 객관식 |
| TextVQA train | 500 | `/workspace/zap/data/textvqa/train` | ~24 token | 자연 사진 + OCR VQA |
| GQA val_balanced | 500 | `/workspace/zap/data/gqa` | ~16 token | 자연 사진 + scene reasoning |

**선정 기준**:
1. 평가 dataset (mm-vet / detail_1k / MileBench) 과 sample 단위로 **겹치지 않음**
2. Image domain 다양성 (다이어그램·OCR·자연사진 모두 포함)
3. Question style 다양성 (factual / multi-choice / reasoning)

DocVQA (MileBench 와 100 % 겹침) 는 학습 후보에서 제외.
LLaVA-Instruct, Mantis multi-image 는 cache 만 보존 (학습 미포함, future ablation).

Teacher cache 위치:
```
/workspace/zap/data/teacher_v2/
├── scienceqa/   (500 .pt)
├── textvqa/     (500 .pt)
└── gqa/         (500 .pt)
```

---

## 3. Notation

| 기호 | 의미 | 값 (LLaVA-1.5-7B) |
|---|---|---|
| (B) | batch size | 1 |
| (L) | LLM layer 수 | 32 |
| (H) | attention head 수 | 32 |
| (D) | hidden dim | 4096 |
| (N) | prefill 총 토큰 수 | 가변 |
| (N_I) | image token 수 | 576 |
| (N_Q) | question token 수 | 가변 |
| (H_I, W_I) | image patch grid | 24 × 24 |
| (T) | decode step 수 | 가변 (mean ~3-4) |
| (\mathcal{I}*{\text{img}}) | image 토큰의 prompt 위치 집합 | — |
| (\mathcal{I}*q) | question 토큰의 prompt 위치 집합 | — |

---

## 4. Teacher signal

LLaVA-1.5 의 prefill 직후 greedy decode 를 수행하면서 매 step 의 attention 을
캡쳐한다. layer (\ell), step (t) 의 attention tensor:

[
A_t^{(\ell)} \in \mathbb{R}^{B \times H \times Q_t \times K_t}
]

여기서 (Q_t) 는 t step 의 query length (=1 for decode), (K_t) 는 prefix + decoded
prefix 길이. 마지막 query 위치만 슬라이스:

[
\bar A_t^{(\ell)} = A_t^{(\ell)}[:,:,-1,:] \in \mathbb{R}^{B\times H\times K_t}
]

image 위치만 추출:

[
\bar A_{t,\text{img}}^{(\ell)} = \bar A_t^{(\ell)}[:,:,\mathcal{I}*{\text{img}}]
\in \mathbb{R}^{B\times H\times N_I}
]

모든 decode step 누적·평균 후 head 평균:

[
s_{b,i}^{*(\ell)}
==================

\frac{1}{HT}\sum_{h=1}^{H}\sum_{t=1}^{T}
\bar A_{t,\text{img}}^{(\ell)}[b,h,i]
]

per-layer 정규화 (sum = 1) 하여 **teacher 분포** 로 사용:

[
\tilde s_{b,i}^{*(\ell)}
========================

\frac{s_{b,i}^{*(\ell)}}{\sum_{j=1}^{N_I} s_{b,j}^{*(\ell)} + \varepsilon}
,\quad
\tilde s^{*(\ell)} \in \mathbb{R}^{B\times N_I},\
\sum_i \tilde s_{b,i}^{*(\ell)} = 1
]

저장 형태: `(L=32) × (N_I=576)` per sample, fp16 (~36 KB/sample).
1500 sample 총 ~54 MB 디스크.

---

## 5. Student architecture

per-layer 독립 모듈 32 개 (scope = "A"), 각 layer 의 hidden state
(H^{(\ell)}\in\mathbb{R}^{B\times N\times D}) 를 입력받아 **score 분포** 를 예측한다.
하이퍼파라미터: (D=4096), (C=d=256), (d_m=512), (k_{\text{ConvNeXt}}=7), 2 ConvNeXt blocks.

### 5.1 Index slicing

prefill hidden state 에서 image / question 위치만 슬라이스:

[
H_{\text{img}}^{(\ell)} = H^{(\ell)}[:,\mathcal{I}_{\text{img}},:] \in \mathbb{R}^{B\times N_I\times D}
]

[
H_q^{(\ell)} = H^{(\ell)}[:,\mathcal{I}_q,:] \in \mathbb{R}^{B\times N_Q\times D}
]

### 5.2 Image CNN branch

#### 5.2.1 Reshape to 2D grid

(N_I = H_I W_I = 576) (LLaVA-1.5 single-image). reshape + permute 로 channel-first 4D:

[
F_{\text{img}}^{(\ell)}
=

\text{Permute}*{(0,3,1,2)}\!\Big(\text{Reshape}*{B\times H_I\times W_I\times D}(H_{\text{img}}^{(\ell)})\Big)
\in \mathbb{R}^{B\times D\times H_I\times W_I}
]

Multi-image (k 장) 일반화는 (B\cdot k) 단위로 batched 처리, token 순서 보존.

#### 5.2.2 1×1 channel projection

[
X^{(\ell)} = \text{Conv}*{1\times1}^{D\to C}(F*{\text{img}}^{(\ell)}) \in \mathbb{R}^{B\times C\times H_I\times W_I}
]

파라미터: (D\cdot C + C = 4096\cdot 256 + 256 = 1{,}048{,}832).

#### 5.2.3 ConvNeXt-style block

block 1 개의 step:

[
Y_0 = X
]

[
Y_1 = \text{DWConv}*{7\times7}^{C\to C, \text{groups}=C}(Y_0) \in \mathbb{R}^{B\times C\times H_I\times W_I}
]

[
Y_2 = \text{GroupNorm}*{(\text{groups}=1, C)}(Y_1) \quad\text{(layer-norm equivalent)}
]

[
Y_3 = \sigma_{\text{GELU}}\!\big(\text{Conv}*{1\times1}^{C\to 4C}(Y_2)\big)
]

[
Y_4 = \text{Conv}*{1\times1}^{4C\to C}(Y_3)
]

[
\text{ConvBlock}(X) = X + Y_4 \in \mathbb{R}^{B\times C\times H_I\times W_I}
]

DW7×7: (49 \cdot C + C = 12{,}800) params. 1×1 expand: (C\cdot 4C + 4C = 263{,}168).
1×1 contract: (4C\cdot C + C = 262{,}400). GroupNorm: (2C = 512). per-block ~ **538 K**.
2 blocks → **1.08 M**.

2 block stacked:

[
C_{\text{img}}^{(\ell)} = \text{ConvBlock}\!\circ\!\text{ConvBlock}(X^{(\ell)}) \in \mathbb{R}^{B\times C\times H_I\times W_I}
]

#### 5.2.4 Flatten + project

spatial axis flatten → token-major 형태 (B, N_I, C):

[
C_{\text{flat}}^{(\ell)} = \text{Transpose}*{(1,2)}\!\Big(\text{Flatten}*{2}(C_{\text{img}}^{(\ell)})\Big) \in \mathbb{R}^{B\times N_I\times C}
]

(C=d) 이므로 (W_c) 는 identity:

[
\bar C_{\text{flat}}^{(\ell)} = W_c\,C_{\text{flat}}^{(\ell)} = C_{\text{flat}}^{(\ell)} \in \mathbb{R}^{B\times N_I\times d}
]

### 5.3 Question branch

(N_Q) 개 question token 을 mean-pool, projection, broadcast:

[
\bar q_{\text{pool}}^{(\ell)}
================================

\frac{1}{N_Q}\sum_{j=1}^{N_Q} H_q^{(\ell)}[:,j,:] \in \mathbb{R}^{B\times D}
]

[
\bar q^{(\ell)} = W_q\,\bar q_{\text{pool}}^{(\ell)} + b_q \in \mathbb{R}^{B\times d}
]

[
Q^{(\ell)} = \text{Unsqueeze}*1(\bar q^{(\ell)}).\text{expand}(-1, N_I, -1) \in \mathbb{R}^{B\times N_I\times d}
]

(W_q) 파라미터: (D\cdot d + d = 1{,}049{,}344).

> N_Q = 0 (edge case) 일 때 (\bar q_{\text{pool}}^{(\ell)} = \mathbf{0}) (zero fallback).

### 5.4 Raw projection branch

image hidden state 자체를 다른 공간으로 project (CNN 거치지 않은 view):

[
\bar H_{\text{img}}^{(\ell)} = W_h\,H_{\text{img}}^{(\ell)} + b_h \in \mathbb{R}^{B\times N_I\times d}
]

(W_h): (D\cdot d + d = 1{,}049{,}344).

### 5.5 Fusion: 5-way feature concatenation

세 branch + 두 question-conditional Hadamard interaction:

[
Z_i^{(\ell)} =
\Big[
\underbrace{\bar H_{\text{img},i}^{(\ell)}}*{\text{raw}};
\underbrace{\bar C*{\text{flat},i}^{(\ell)}}*{\text{CNN}};
\underbrace{Q_i^{(\ell)}}*{\text{question}};
\underbrace{\bar H_{\text{img},i}^{(\ell)}\odot Q_i^{(\ell)}}*{\text{raw}\times Q};
\underbrace{\bar C*{\text{flat},i}^{(\ell)}\odot Q_i^{(\ell)}}_{\text{CNN}\times Q}
\Big]
\in \mathbb{R}^{5d}
]

Hadamard 항: (\big[v_1\odot v_2\big]_j = v_{1,j}\cdot v_{2,j}) (element-wise, 같은 dim).

stacked:

[
Z^{(\ell)} \in \mathbb{R}^{B\times N_I\times 5d},\quad 5d = 1280
]

### 5.6 MLP scoring head

per-token (i.e. per image patch) 2-layer MLP, 입력 (\mathbb{R}^{5d}) → 스칼라:

[
u_i^{(\ell)} = \sigma_{\text{GELU}}\big(W_1 z_i^{(\ell)} + b_1\big),\quad
W_1 \in \mathbb{R}^{d_m\times 5d},; b_1 \in \mathbb{R}^{d_m}
]

[
\hat s_i^{(\ell)} = W_2 u_i^{(\ell)} + b_2,\quad
W_2 \in \mathbb{R}^{1\times d_m},; b_2 \in \mathbb{R}
]

vectorized 형태:

[
\hat s^{(\ell)} = \text{Squeeze}*{-1}\Big(W_2,\sigma*{\text{GELU}}(Z^{(\ell)} W_1^\top + b_1) + b_2\Big)
\in \mathbb{R}^{B\times N_I}
]

(W_1) 파라미터: (5d\cdot d_m + d_m = 656{,}384). (W_2): (d_m\cdot 1 + 1 = 513). MLP 합 ~**657 K**.

### 5.7 학습 시 distribution 형태

teacher (\tilde s^{*(\ell)}) 가 per-layer (\sum_i = 1) 인 분포이므로 student logits
도 softmax 정규화:

[
\hat p_i^{(\ell)} = \frac{\exp(\hat s_i^{(\ell)})}{\sum_{j=1}^{N_I}\exp(\hat s_j^{(\ell)})},\quad
\sum_i \hat p_i^{(\ell)} = 1
]

손실은 (\hat p^{(\ell)}) 와 (\tilde s^{*(\ell)}) 사이에서 계산.

### 5.8 추론 시 top-k selection

학습 시와 달리 추론에서는 정규화 불필요 — logits 의 **순위** 만 사용:

[
\mathcal{K}^{(\ell)} = \text{TopK}*i\!\big(\hat s_i^{(\ell)}, ; n*{\text{image\_keep}}\big) \subset \mathcal{I}_{\text{img}}
]

(n_{\text{image\_keep}}) 은 §8.2 의 keep-budget 식 참고. (\mathcal{K}^{(\ell)}) 에 속하는
token 만 layer-(\ell) 의 KV cache 에 남고 나머지는 evict.

### 5.9 Multi-image 일반화

`N_I = k · 576` (k images per sample) 인 경우 §5.2.1 의 reshape 을 (B,k,H_I,W_I,D)
→ (Bk,H_I,W_I,D) 로 batched 후 CNN. flatten 결과를 (B, N_I, d) 로 다시 묶어 token
순서 보존. Question branch / fusion / MLP 는 동일 (이미 (N_I) 일반화).

### 5.10 Parameter 합계

per-layer:

| Component | Formula | params |
|---|---|---|
| (Conv_{1×1}^{D\to C}) | (DC + C) | 1,048,832 |
| 2× ConvNeXt block | (2\cdot 538{,}880) | 1,077,760 |
| (W_h) | (Dd + d) | 1,049,344 |
| (W_q) | (Dd + d) | 1,049,344 |
| (W_c) | identity (C=d) | 0 |
| MLP head ((W_1, W_2)) | (5d\cdot d_m + d_m + d_m + 1) | 656,897 |
| **per-layer 합** | | **~ 4.88 M** |

scope A (32 layer) 총: **~ 156 M parameters** (LLaVA backbone 7 B 의 2.2 %).
checkpoint 크기 ≈ **625 MB** (fp32).

---

## 6. Loss

per-(sample, layer) 손실:

[
\mathcal{L}*{\text{MSE}}^{(\ell)} = \frac{1}{N_I}\sum*{i=1}^{N_I}
\big(\hat p_i^{(\ell)} - \tilde s_{b,i}^{*(\ell)}\big)^2
]

ranking term — top 20 % vs bottom 40 % image token (teacher 기준) 사이 hinge:

[
\mathcal{P}^{(\ell)} = \big{(i,j) ;\big|; i \in \text{Top}*{0.2}(\tilde s^{*(\ell)}),; j \in \text{Bot}*{0.4}(\tilde s^{*(\ell)})\big}
]

[
\mathcal{L}*{\text{rank}}^{(\ell)}
=
\frac{1}{|\mathcal{P}^{(\ell)}|}\sum*{(i,j)\in\mathcal{P}^{(\ell)}}
\max\!\big(0,\ m - (\hat p_i^{(\ell)} - \hat p_j^{(\ell)})\big),\quad m=0.05
]

per-sample 총 손실:

[
\mathcal{L}
=

\sum_{\ell=0}^{L-1}\!\Big(\mathcal{L}*{\text{MSE}}^{(\ell)} + \lambda*{\text{rank}}\,\mathcal{L}*{\text{rank}}^{(\ell)}\Big),\quad
\lambda*{\text{rank}}=0.1
]

---

## 7. Training 절차

| 항목 | 값 |
|---|---|
| Backbone (frozen) | LLaVA-1.5-7B (`/workspace/zap/ckpts/llava-1.5-7b-hf`) |
| Backbone dtype | bf16 (no_grad prefill) |
| Student dtype | fp32 (master), bf16 inference |
| Optimizer | AdamW, weight_decay 0 |
| Learning rate | **1e-4** (1e-3 은 epoch 0 부터 발산) |
| Batch | 1 sample × all 32 layer (gradient accumulation across layers) |
| Grad clip | max_norm 1.0 |
| Epoch | 20 (validation loss plateau ~ epoch 19) |
| Train / Val split | 90 % / 10 % (1350 / 150) |
| Seed | 0 (deterministic shuffle, pre-loaded teacher) |
| Wall clock | ~64 min on 1× RTX 4090 |

각 학습 step:

1. `.pt` cache 에서 teacher 분포 (\tilde s^{*(\ell)}) 와 image / question
   indices 로드
2. LLaVA prefill forward (no_grad, bf16) → 32-layer hidden state 확보
3. layer 마다 student forward → softmax → loss (\sum_\ell)
4. backward → step (LLaVA freeze)

학습 곡선 (lr = 1e-4, scope A):

```
epoch 0  → val_loss 0.13719  (best)
epoch 5  → 0.13624
epoch 10 → 0.13597
epoch 19 → 0.13594  (final best)
```

20 epoch 후 plateau. 40 epoch 시도해도 best 동일 (확장 무의미).

---

## 8. Inference / eviction protocol

### 8.1 Per-sample 흐름

```
prefill input → infer image_positions, question_positions
              → LLaVA prefill (with VisualUtilityStudentPress hook)
              → at each layer:
                  scores = student.forward_layer(H_l, image_idx, q_idx)
                  evict bottom-(N_I - n_image_keep) image tokens
              → cache 압축 완료 → decode
```

### 8.2 Keep-budget 계산 (total_keep_ratio mode)

[
n_{\text{total\_keep}} = \lceil k \cdot (N_I + N_Q + N_{\text{sys}})\rceil,\quad
n_{\text{image\_keep}} = \max(0,\ n_{\text{total\_keep}} - N_{\text{text}})
]

text 는 **항상 보존**, image budget = total budget − text. 즉 KV cache 총 크기는
methods 간 동일 (memory-fair), 분배 정책만 다름.

### 8.3 Layer-wise eviction

scope A 의 경우 layer 0..31 모두 student 점수로 top-k image token 만 보존.
나머지 N_I − n_image_keep 토큰은 KV cache 에서 제거.

### 8.4 Multi-image / multi-prompt 일반화

- multi-image: 위 5.5 의 batched CNN 로 student forward.
- 매우 긴 context: LOOK-M 표준 truncation (`--truncate`) 으로 max_context_len 캡 후
  image-우선 선택.

---

## 9. 평가 metric

| Metric | 정의 | 비교 대상 |
|---|---|---|
| **PPL** | Teacher-forcing perplexity over GT answer tokens | (\text{exp}(\frac{1}{N}\sum -\log p(t_i\,|\,t_{<i},I,q))) |
| **ROUGE-L vs full** | Free-generation output 의 ROUGE-L F1, **same-model full-cache prediction 을 reference 로 사용** | (\text{ROUGE-L}(\text{gen}*{\text{evict}}, \text{gen}*{\text{full-cache}})) |
| MileBench Acc / ROUGE-L | LookMileBench evaluator 의 task-specific 점수 | task 표준 정답 |

> **vs-GT ROUGE 는 paper 에서 사용하지 않는다** (LVLM 의 mm-vet 은 본질적으로
> GPT-4 채점 기준이고 ROUGE-L 은 vs full-cache 보존도 측정용 — 본 연구는 후자만
> 보고).

---

## 10. 데이터 / 코드 위치

| 항목 | 경로 |
|---|---|
| Teacher cache | `/workspace/zap/data/teacher_v2/{sci,text,gqa}qa/` |
| Full-cache ROUGE 기준 | `/workspace/zap/data/rouge_ref/our_full_{mm-vet,detail_1k}.json` |
| Production ckpt | `/workspace/zap/ckpts/student_v2_A_gqa_lr1e4/` |
| Stage 1 collector | `/workspace/zap/collect_future_teacher_v2.py` |
| Stage 2 trainer | `/workspace/zap/train_visual_utility_student.py` |
| Student class | `/workspace/zap/kvpress/presses/visual_utility_student.py` |
| Press class | `/workspace/zap/kvpress/presses/image_token_press.py::VisualUtilityStudentPress` |
| Reference generator | `/workspace/zap/scripts/generate_full_cache_reference.py` |
| Result CSV | `/workspace/zap/result/ppl_rouge_v2.csv`, `milebench_v2.csv` |
| Plan / 진행 로그 | `/workspace/zap/experiments/EXP-20260425-002/` |
