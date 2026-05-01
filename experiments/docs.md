# Method: 질문 조건부 Visual Utility Student

**VisualUtilityStudentPress**의 레이어별 scoring student에 대한 공식 표기.
구현 출처: `kvpress/presses/visual_utility_student.py`.

본 문서는 LLaVA-1.5용 `VisualUtilityStudentPress` / `mode=visual_utility_student`에 대한 표기이다.
기존 `mode=probe`의 `ProbeImageTeacherPress` / `KVzapModel` probe는 image hidden만 입력으로 쓰는 별도 경로이며, 이 문서의 CNN branch, question branch, fusion 구조를 사용하지 않는다.

핵심 규약:

| 의미 | 기호 |
|---|---|
| 레이어 수 | $L$ |
| Hidden dimension | $D$ |
| Attention head 수 | $N_h$ |
| Head dimension | $d_h = D / N_h$ |
| Prefill hidden state | $X^{(l)}$ |
| Image-token hidden matrix | $X_{\mathrm{img}}^{(l)}$ |
| Question-token hidden matrix | $X_q^{(l)}$ |
| $i$-번째 image token hidden | $x_i^{v,(l)}$ |
| Projected visual feature | $\bar{x}_i^{v,(l)} = W_v\, x_i^{v,(l)}$ |
| Projected question feature | $\bar{q}^{(l)} = W_q\,\mathrm{Pool}(X_q^{(l)})$ |
| Projected CNN context | $\bar{c}_i^{(l)}$ |
| Teacher raw / dist | $s_i^{(l),\star}$ / $y_i^{(l),\star}$ |
| Student raw / dist | $\hat{s}_i^{(l)}$ / $\hat{y}_i^{(l)}$ |

코드 변수명은 논문 표기와 반드시 일치하지 않는다. 예를 들어 구현의 `W_h`, `H_proj`는 논문 표기에서 각각 $W_v$, $\bar{X}_{\mathrm{img}}^{(l)}$에 대응한다.

---

## 1. Prefill hidden state - 토큰 분할

레이어 $l$에서 prefill hidden state 전체는

$$
X^{(l)} \in \mathbb{R}^{B \times N \times D}
$$

이며, $B$는 batch size, $N$은 prefill 시퀀스 길이, $D$는 hidden dimension이다.

$\mathcal{I}_{\mathrm{img}}$와 $\mathcal{I}_q$를 각각 image token, question token의 인덱스 집합이라 하면

$$
X_{\mathrm{img}}^{(l)} = X^{(l)}[:, \mathcal{I}_{\mathrm{img}}, :]
\in \mathbb{R}^{B \times N_I \times D}
\qquad
X_q^{(l)} = X^{(l)}[:, \mathcal{I}_q, :]
\in \mathbb{R}^{B \times N_Q \times D}
$$

이고, 각 image token의 hidden state는 $x_i^{v,(l)} \in \mathbb{R}^D$로 표기한다 ($i = 1, \ldots, N_I$).

> **Multi-image 주의.** 한 sample에 $K$개의 이미지가 있고 각 이미지가 $H_I \times W_I$ patch로 구성되면 $N_I = K \cdot H_I \cdot W_I$이다. 단일 이미지($K=1$)에서는 $N_I = H_I W_I$.

구현에서는 `image_indices`와 `question_indices`로 각각 `H_l.index_select(dim=1, ...)`를 수행한다.

---

## 2. Question representation

Question token을 mean-pooling한 뒤 차원 $d$로 projection한다:

$$
\bar{q}^{(l)}
= W_q \!\left( \frac{1}{N_Q}\sum_{j=1}^{N_Q} X_q^{(l)}[:,j,:] \right)
\in \mathbb{R}^{B \times d}
$$

이를 모든 image token 위치에 broadcast한다:

$$
Q^{(l)} = \mathrm{Broadcast}\!\left(\bar{q}^{(l)},\, N_I\right)
\in \mathbb{R}^{B \times N_I \times d}
$$

구현:

```python
q = H_q.mean(dim=1)
q_proj = self.W_q(q)
Q = q_proj.unsqueeze(1).expand(-1, N_I, -1)
```

---

## 3. Image-token raw projection

$$
\bar{X}_{\mathrm{img}}^{(l)} = W_v\, X_{\mathrm{img}}^{(l)}
\in \mathbb{R}^{B \times N_I \times d}
$$

토큰 단위:

$$
\bar{x}_i^{v,(l)} \in \mathbb{R}^d.
$$

구현에서는 `self.W_h`와 `H_proj`라는 이름을 사용한다:

```python
H_proj = self.W_h(H_img)
```

논문 표기에서는 head index $h$와의 충돌을 피하기 위해 $W_h$ 대신 $W_v$를 사용한다.

---

## 4. CNN branch - spatial image context

Image token을 patch grid로 reshape한다. Multi-image 입력의 경우 각 $K$개 이미지가 독립적으로 처리된다. 이때 배치 축은 $B \cdot K$가 된다:

$$
F_{\mathrm{img}}^{(l)}
= \mathrm{Grid}\!\left(X_{\mathrm{img}}^{(l)}\right)
\in \mathbb{R}^{BK \times D \times H_I \times W_I}.
$$

$1\times1$ convolution으로 채널을 projection한다:

$$
X_{\mathrm{conv}}^{(l)} = \mathrm{Conv}_{1\times1}\!\left(F_{\mathrm{img}}^{(l)}\right)
\in \mathbb{R}^{BK \times C \times H_I \times W_I}.
$$

$n_{\mathrm{blk}}$개의 **ConvNeXt-style** residual block을 적용한다. 기본값은 $n_{\mathrm{blk}}=2$이다:

$$
C_{\mathrm{img}}^{(l)} = \mathrm{ConvNeXt}^{n_{\mathrm{blk}}}\!\left(X_{\mathrm{conv}}^{(l)}\right)
\in \mathbb{R}^{BK \times C \times H_I \times W_I}.
$$

각 block 구성:

$$
\mathrm{DWConv}_{7\times7}
\to \mathrm{GroupNorm}
\to \mathrm{Conv}_{1\times1}^{4C}
\to \mathrm{GELU}
\to \mathrm{Conv}_{1\times1}^{C}
+ \mathrm{residual}
$$

Spatial dimension을 flatten하고 batch를 복원한 뒤 채널을 $d$로 projection한다:

$$
C_{\mathrm{flat}}^{(l)} = \mathrm{Flatten}\!\left(C_{\mathrm{img}}^{(l)}\right)
\in \mathbb{R}^{B \times N_I \times C}
$$

$$
\bar{C}_{\mathrm{flat}}^{(l)} = W_c\, C_{\mathrm{flat}}^{(l)}
\in \mathbb{R}^{B \times N_I \times d}.
$$

토큰 단위:

$$
\bar{c}_i^{(l)} \in \mathbb{R}^d.
$$

$C = d$이면 $W_c$는 항등사상이고, 그렇지 않으면 학습되는 linear layer이다.

구현:

```python
F_img = (
    H_img.reshape(B, n_images, grid_h, grid_w, D)
    .reshape(B * n_images, grid_h, grid_w, D)
    .permute(0, 3, 1, 2)
    .contiguous()
)
X_img = self.conv_1x1_proj(F_img)
C_img = self.conv_blocks(X_img)
C_flat = C_img.flatten(2).transpose(1, 2)
C_flat = C_flat.reshape(B, N_I, -1)
C_proj = self.W_c(C_flat)
```

---

## 5. Fusion

세 branch와 두 Hadamard interaction term을 concat한다:

$$
z_i^{(l)}
= \Bigl[
    \bar{x}_i^{v,(l)};\
    \bar{c}_i^{(l)};\
    \bar{q}^{(l)};\
    \bar{x}_i^{v,(l)} \odot \bar{q}^{(l)};\
    \bar{c}_i^{(l)} \odot \bar{q}^{(l)}
  \Bigr]
\in \mathbb{R}^{5d}.
$$

전체 tensor 형태:

$$
Z^{(l)} \in \mathbb{R}^{B \times N_I \times 5d}.
$$

구현:

```python
Z = torch.cat([H_proj, C_proj, Q, H_proj * Q, C_proj * Q], dim=-1)
```

---

## 6. MLP head - 토큰 score

Hidden size $d_{\mathrm{mlp}}$와 GELU activation을 갖는 2-layer MLP:

$$
u_i^{(l)} = \mathrm{GELU}\!\left(W_1 z_i^{(l)} + b_1\right)
\in \mathbb{R}^{d_{\mathrm{mlp}}}
$$

$$
\hat{s}_i^{(l)} = W_2 u_i^{(l)} + b_2
\in \mathbb{R}.
$$

전체 출력:

$$
\hat{s}^{(l)} \in \mathbb{R}^{B \times N_I}.
$$

이는 student raw score이다. 학습에서 image token dimension에 대해 softmax를 적용해 student distribution을 만든다:

$$
\hat{y}_i^{(l)}
=
\frac{\exp(\hat{s}_i^{(l)} / \tau)}{\sum_{j=1}^{N_I} \exp(\hat{s}_j^{(l)} / \tau)},
\qquad i = 1, \ldots, N_I.
$$

현재 구현에는 `softmax_temp` config가 따로 없으므로 실질적으로 $\tau=1$이다:

```python
pred_norm = F.softmax(s_pred, dim=-1)
```

실제 KV eviction 시점에서는 raw score $\hat{s}_i^{(l)}$를 그대로 ranking에 사용한다. Softmax는 monotonic이므로 top-$k_I$ 선택에는 영향을 주지 않는다.

---

## 7. Training objective

Teacher cache에는 image token별 teacher distribution이 저장된다:

$$
y_i^{(l),\star}
=
\frac{s_i^{(l),\star}}{\sum_{j=1}^{N_I} s_j^{(l),\star} + \epsilon}.
$$

현재 `train_visual_utility_student.py`의 구현 objective는 KL이 아니라 softmax-MSE와 pairwise ranking loss의 합이다:

$$
\mathcal{L}^{(l)}
=
\mathrm{MSE}\!\left(\hat{y}^{(l)}, y^{(l),\star}\right)
+
\lambda_{\mathrm{rank}}\,
\mathcal{L}_{\mathrm{rank}}\!\left(\hat{y}^{(l)}, y^{(l),\star}\right).
$$

기본 ranking weight는 `lambda_rank=0.1`이다. 논문에서 KL distillation을 쓰는 버전과 구분해야 하며, 현재 코드 기준 표현은 **softmax-normalized distribution matching + ranking loss**가 정확하다.

---

## 8. Compact 표기 (논문용)

$$
\hat{s}_i^{(l)}
= f_\theta\!\Bigl(
    \bar{x}_i^{v,(l)},\;
    \bar{c}_i^{(l)},\;
    \bar{q}^{(l)},\;
    \bar{x}_i^{v,(l)} \odot \bar{q}^{(l)},\;
    \bar{c}_i^{(l)} \odot \bar{q}^{(l)}
  \Bigr).
$$

표기 정리:

| 기호 | 정의 |
|---|---|
| $\bar{x}_i^{v,(l)}$ | 레이어 $l$에서 image token $i$의 $W_v$-projected hidden state |
| $\bar{c}_i^{(l)}$ | 레이어 $l$에서 token $i$의 ConvNeXt spatial context feature |
| $\bar{q}^{(l)}$ | 레이어 $l$에서 mean-pool 후 $W_q$-projected question representation |
| $\odot$ | element-wise product |
| $f_\theta$ | GELU를 갖는 2-layer MLP |

---

## 9. Block-equation 요약 (논문 섹션용)

$$
X_{\mathrm{img}}^{(l)} = X^{(l)}[:,\,\mathcal{I}_{\mathrm{img}},:]
\qquad
X_q^{(l)} = X^{(l)}[:,\,\mathcal{I}_q,:]
$$

$$
\bar{q}^{(l)} = W_q\!\left(\frac{1}{N_Q}\sum_j X_q^{(l)}[:,j,:]\right)
$$

$$
\bar{X}_{\mathrm{img}}^{(l)} = W_v\, X_{\mathrm{img}}^{(l)}
$$

$$
\bar{C}_{\mathrm{flat}}^{(l)} =
W_c\,\mathrm{Flatten}\!\Bigl(
  \mathrm{ConvNeXt}^{n_{\mathrm{blk}}}\!\bigl(
    \mathrm{Conv}_{1\times1}(\mathrm{Grid}(X_{\mathrm{img}}^{(l)}))
  \bigr)
\Bigr)
$$

$$
z_i^{(l)}
= \Bigl[
    \bar{x}_i^{v,(l)};\
    \bar{c}_i^{(l)};\
    \bar{q}^{(l)};\
    \bar{x}_i^{v,(l)}\odot\bar{q}^{(l)};\
    \bar{c}_i^{(l)}\odot\bar{q}^{(l)}
  \Bigr]
$$

$$
\hat{s}_i^{(l)} = \mathrm{MLP}(z_i^{(l)})
$$

$$
\hat{y}_i^{(l)} = \mathrm{softmax}_i\!\left(\hat{s}^{(l)} / \tau\right),
\qquad \tau=1\ \text{in the current implementation}.
$$

---

## 구현 파라미터 (기본 config - scope A)

| 파라미터 | 값 |
|---|---|
| `hidden_dim` $D$ | 4096 |
| `conv_dim` $C$ | 256 |
| `proj_dim` $d$ | 256 |
| `mlp_dim` $d_{\mathrm{mlp}}$ | 512 |
| `num_conv_blocks` $n_{\mathrm{blk}}$ | 2 |
| `kernel_size` | 7 |
| `grid_h`, `grid_w` | 24, 24 |
| effective $\tau$ | 1.0 |
| scope A 레이어 | 0-31 (전체 32개 레이어) |

---

## 구현 경로 구분

| 경로 | 구현 | 구조 |
|---|---|---|
| `mode=visual_utility_student` | `VisualUtilityStudentPress` | 이 문서의 3-branch CNN + question + raw visual student |
| `mode=probe` | `ProbeImageTeacherPress` + `KVzapModel` | image hidden만 입력으로 쓰는 per-layer MLP probe |
| `mode=future` / `mode=hybrid` | `KVzapModel` 기반 future / hybrid probe | 이 문서의 CNN/question fusion 구조가 아님 |

---

## 표기 변경 이력 (v1 -> v2)

| v1 (이전) | v2 (현재) | 사유 |
|---|---|---|
| $H^{(l)}$ | $X^{(l)}$ | $H$가 attention head 수와 충돌 |
| $H_{\mathrm{img}}^{(l)}$ | $X_{\mathrm{img}}^{(l)}$ | 위와 동일 |
| $H_q^{(l)}$ | $X_q^{(l)}$ | 위와 동일 |
| $W_h$ | $W_v$ | "visual"의 의미를 명확히 하고 head index $h$와 분리 |
| $\bar{h}_i^{v,(l)}$ | $\bar{x}_i^{v,(l)}$ | hidden-state 표기 충돌 완화 |
| head 수 미정의 | $N_h$ | Future-Attention Teacher 섹션과 통일 |
| head dim 미정의 | $d_h = D/N_h$ | Preliminaries에서 명시 |
| $\hat{s}^{(l)}$만 정의 | $\hat{s}^{(l)}$ / $\hat{y}^{(l)}$ 모두 정의 | Teacher 표기 $s^\star$ / $y^\star$와 대칭 |

코드 변수명(`q_proj`, `H_proj`, `W_h` 등)은 변경하지 않는다. 논문 표기와 코드 식별자는 일치할 필요가 없다.
