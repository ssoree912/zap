# Implementation Guide: Paper Method → Code Mapping

이 문서는 논문의 `\section{Method}` 각 절이 코드의 어느 파일/클래스/함수에 구현되어 있는지를 설명합니다.

---

## Overall Pipeline

논문 Figure (pipeline)에 해당하는 세 단계가 아래와 같이 코드에 분리되어 있습니다.

| 논문 단계 | 코드 역할 |
|---|---|
| Teacher signal 수집 (offline) | `foresight/teacher/collect_llava*.py` |
| Student 학습 | `foresight/train/llava_*.py` + `train.py` |
| Inference-time 추론 | `foresight/eval/vlmeval_onevision_student.py` 등 |

---

## Section 3.1 — Preliminaries (Notation)

논문에서 정의한 기호들이 코드에서 어떻게 대응되는지입니다.

| 논문 기호 | 의미 | 코드 변수/위치 |
|---|---|---|
| `L` | transformer layer 수 | LLaVA-1.5: 32, OneVision: 28 |
| `D` | hidden dimension | `hidden_dim` (4096 or 3584) |
| `N_h` | attention head 수 | model config에서 참조 |
| `X^{(l)}` | layer-l prefill hidden states | `H_all[li+1]` — `vlmeval_onevision_student.py:330` |
| `I_img` | image-token positions | `image_positions` — `infer_onevision_image_positions_no_forward()` |
| `I_q` | question-token positions | `q_idx` — teacher 수집 및 학습 코드 내 |
| `N_I` | image token 수 | `n_img` |
| `N_Q` | question token 수 | `len(q_idx)` |
| `ŝ^{(l)}` | student predicted scores | `all_scores[li]` — `vlmeval_onevision_student.py:333` |

**Image token position 복원 (no-forward):**

```
foresight/llava_onevision_extractor.py
  └── infer_onevision_image_positions_no_forward()   # OneVision

foresight/llava_15b_extractor.py
  └── infer_llava_image_positions_no_forward()       # LLaVA-1.5
```

input_ids에서 `image_token_id` 위치를 탐색하여 추가 forward pass 없이 image token 위치를 반환합니다.

---

## Section 3.2 — Future-Attention Teacher

> **논문 핵심:** full-cache decode 중 각 answer token이 image token에 얼마나 attend하는지를 집계해 teacher signal `y^{(l),teacher}` 를 구성합니다.

### Eq.(1) — `a^{(l,m)}_{t,i}` : decode-to-image attention weight

**구현 위치:** `foresight/teacher/collect_llava_onevision.py`

```python
# collect_llava_onevision.py (약 line 400-600)
# decode loop 안에서 attention weight 캡처
with torch.no_grad():
    out = model(
        input_ids=next_token,
        past_key_values=past_kv,
        use_cache=True,
        output_attentions=True,   # <-- 여기서 attention weight 추출
    )
attn_weights = out.attentions   # [L] 각 layer: [B, N_h, 1, T_prompt]
# image 위치만 슬라이싱
attn_to_img = attn_weights[l][:, :, 0, image_positions]  # [B, N_h, N_I]
```

### Eq.(2) — `s^{(l),teacher}_i` : answer token 및 head에 대한 평균 집계

```python
# T step의 decode attention을 누적
for t in range(T_decode):
    ...
    teacher_raw[l] += attn_to_img.mean(dim=1).squeeze(0)  # head 평균 후 누적
teacher_raw[l] /= T_decode   # token 평균
```

### Eq.(3) — `y^{(l),teacher}_i` : 정규화 (probability distribution)

```python
teacher_norm[l] = teacher_raw[l] / (teacher_raw[l].sum() + 1e-8)
```

**저장 포맷** (`.pt` shard 파일):

```python
{
    "image_token_indices":  torch.Tensor [N_I],
    "question_token_indices": torch.Tensor [N_Q],
    "teacher_raw":  torch.Tensor [L, N_I],   # Eq.(2) 결과
    "teacher_norm": torch.Tensor [L, N_I],   # Eq.(3) 결과 (학습 타겟)
    "prompt_len_mm": int,
    "T": int,
}
```

LLaVA-1.5 버전은 `foresight/teacher/collect_llava15.py`에 동일 로직이 구현되어 있습니다.

---

## Section 3.3 — Question-Conditioned Visual-Utility Student

> **논문 핵심:** prefill 단계의 hidden states만으로 `y^{(l),teacher}`를 예측하는 경량 per-layer scorer `f_θ^{(l)}`를 학습합니다.

### Student Architecture (`f_θ^{(l)}`)

| 모델 | 파일 | CNN 종류 | backbone |
|---|---|---|---|
| LLaVA-1.5 | `kvpress/presses/visual_utility_student.py` | 2D ConvNeXt | LLaMA-2-7B (grid 24×24) |
| LLaVA-OneVision | `kvpress/presses/visual_utility_student_onevision.py` | 1D Conv | Qwen2-7B (AnyRes) |

#### 논문 수식 → 코드 매핑

**논문:**
```
q̄^{(l)} = W_q · Mean(X_q^{(l)})         # pooled question feature
x̄_i^{v,(l)} = W_v · x_i^{v,(l)}         # token-level visual feature
C̄^{(l)} = g_φ(X_img^{(l)})              # context-aware visual (CNN)
```

**코드 (`VisualUtilityStudentLayer.forward`):**

```python
# visual_utility_student_onevision.py

# q̄^{(l)}: question pooling
q_feat = self.q_proj(hidden_states[:, q_idx, :].mean(dim=1))        # [B, d]
q_feat = q_feat.unsqueeze(1).expand(-1, n_img, -1)                  # [B, N_I, d]

# x̄_i^{v,(l)}: token-level visual projection
v_feat = self.v_proj(hidden_states[:, img_idx, :])                  # [B, N_I, d]

# C̄^{(l)}: CNN context encoder (g_φ)
img_h = hidden_states[:, img_idx, :]                                 # [B, N_I, D]
c_feat = self.cnn_encoder(img_h)                                     # [B, N_I, C]
```

**CNN encoder (g_φ) 구조:**

- **LLaVA-1.5 (2D):** `visual_utility_student.py` — image token을 `[B, H, W, D]` grid로 reshape 후 ConvNeXtStyleBlock (depthwise 2D conv, kernel=7)
- **OneVision (1D):** `visual_utility_student_onevision.py` — `[B, D, N_I]`에 Conv1d 적용 (AnyRes 가변 해상도 대응)

**논문 (fusion + MLP head):**
```
ŝ_i^{(l)} = MLP_ψ([x̄_i^v; c̄_i; q̄; x̄_i^v ⊙ q̄; c̄_i ⊙ q̄])
```

**코드:**
```python
# Hadamard interaction + concatenation
fused = torch.cat([
    v_feat,             # x̄_i^v
    c_feat,             # c̄_i
    q_feat,             # q̄
    v_feat * q_feat,    # x̄_i^v ⊙ q̄
    c_feat * q_feat,    # c̄_i ⊙ q̄
], dim=-1)              # [B, N_I, 5d]

logits = self.mlp_head(fused)   # [B, N_I, 1] → [B, N_I]
```

`mlp_head`: Linear(5d → mlp_dim=512) → GELU → Linear(512 → 1)

---

### Training Loss — Eq.(4)

**구현 위치:** `foresight/train/llava_onevision.py` (LLaVA-1.5는 `foresight/train/llava_15.py`)

```python
# ℒ^{(l)} = MSE(ŷ^{(l)}, y^{(l),teacher}) + λ_rank · ℒ_rank(ŷ^{(l)}, y^{(l),teacher})

pred_norm = torch.softmax(s_pred, dim=-1)            # ŷ^{(l)}
teacher_norm = record["teacher_norm"][li]            # y^{(l),teacher}

loss_mse = F.mse_loss(pred_norm, teacher_norm)       # MSE term

loss_rank = pairwise_ranking_loss(pred_norm, teacher_norm,
                                  top_ratio=0.2,
                                  bottom_ratio=0.4,
                                  margin=0.05)       # ranking term

loss = loss_mse + lambda_rank * loss_rank            # λ_rank=0.1 (default)
```

**Pairwise Ranking Loss 구현** (`visual_utility_student_onevision.py`):

```python
# teacher 기준 상위 20% / 하위 40% 선택
top_idx = teacher_norm.topk(k=n_top, largest=True).indices
bot_idx = teacher_norm.topk(k=n_bot, largest=False).indices

# hinge loss: margin - (pred_top - pred_bot) > 0 인 경우 페널티
loss_rank = clamp(margin - (pred[top_idx] - pred[bot_idx]), min=0).mean()
```

**학습 루프 요약:**

```
for each sample:
    H_all = frozen_llava_prefill(sample)              # [L+1, 1, T, D]
    for li in layer_indices:
        s_pred = student.layers[li](H_all[li+1], img_idx, q_idx)
        loss += (mse + λ_rank * rank) / n_layers
    optimizer.step()
```

Optimizer: AdamW, lr=1e-4, gradient clipping max_norm=1.0

---

## Section 3.4 — Inference-time Pruning

> **논문 핵심:** prefill 이후 student score로 TopK image token을 선택하고, 나머지 image KV entry를 제거한 뒤 decode합니다.

**구현 위치:** `foresight/eval/vlmeval_onevision_student.py` (`generate_inner_image` 메서드)

### Step 1. Prefill + hidden state 추출

```python
# vlmeval_onevision_student.py:228-
prefill = model(**inputs, use_cache=True, output_hidden_states=True)
H_all   = prefill.hidden_states    # [L+1, 1, T_prefill, D]
past_kv = prefill.past_key_values  # [L] (K, V) tensors
```

### Step 2. Student scoring (per layer)

논문 `ŝ^{(l)} = f_θ^{(l)}(X_img^{(l)}, X_q^{(l)})` 에 해당:

```python
# vlmeval_onevision_student.py:330-333
for li in student.layer_indices:
    H_l = H_all[li + 1]                              # [1, T, D]
    scores = student.layers[str(li)](H_l, img_idx, q_idx)  # [1, N_I]
    all_scores[li] = scores.squeeze(0)               # [N_I]
```

### Step 3. TopK selection — Eq.(5) `R^{(l)} = TopK(ŝ^{(l)}, k_I)`

```python
# vlmeval_onevision_student.py:372-382
for li in student.layer_indices:
    k = layer_n_keep[li]                              # k_I = ⌈r · N_I⌉
    top = torch.topk(all_scores[li], k=k, largest=True).indices
    
    image_keep = torch.zeros(n_img, dtype=torch.bool)
    image_keep[top.cpu()] = True
    keep_masks[li] = image_keep                       # R^{(l)}
```

### Step 4. KV cache pruning (text tokens 전부 보존)

```python
# _trim_kv_cache_per_layer() 함수
# text tokens: 항상 keep / image tokens: keep_masks[li] 적용
past_kv = _trim_kv_cache_per_layer(past_kv, keep_masks, image_positions)
# key/value: [1, H, T, D] → [1, H, T_pruned, D]
```

논문: "text tokens encode instruction and question context, so their KV entries are preserved at all layers"

### Step 5. Autoregressive decode (pruned KV cache)

```python
# RoPE alignment: absolute position_ids & cache_position 사용
answer_ids = _greedy_decode_with_kv(
    model, past_kv, next_token,
    prompt_len=prompt_len,           # absolute position offset
    eos_token_id=eos_token_id,
    max_new_tokens=max_new_tokens,
)
```

pruned cache를 사용하므로 각 decode step에서 attention computation이 감소합니다.

---

## Additional: Entropy-Based Budget Redistribution

논문에는 없지만 코드에서 layer별 budget을 entropy로 재분배하는 확장 기능이 구현되어 있습니다.

```python
# vlmeval_onevision_student.py:336-360
for li in student.layer_indices:
    p = torch.softmax(all_scores[li], dim=-1)
    entropy = -(p * p.log().clamp(min=-20)).sum()
    layer_entropy[li] = entropy.item()

# 엔트로피 비례 budget 할당
total = sum(layer_entropy.values())
for li in student.layer_indices:
    layer_n_keep[li] = int(n_keep_total * layer_entropy[li] / total)
```

---

## File Structure Summary

```
/workspace/zap/
├── kvpress/presses/
│   ├── visual_utility_student.py           # Student (LLaVA-1.5, 2D CNN)
│   └── visual_utility_student_onevision.py # Student (OneVision, 1D CNN)
│
├── foresight/
│   ├── llava_onevision_extractor.py        # Image position 복원
│   ├── llava_15b_extractor.py              # LLaVA-1.5 position 복원
│   ├── image_teacher_utils.py              # Dataset loading 유틸
│   ├── teacher/
│   │   ├── collect_llava_onevision.py      # Teacher 수집 (OneVision)
│   │   └── collect_llava15.py             # Teacher 수집 (LLaVA-1.5)
│   ├── train/
│   │   ├── llava_onevision.py              # 학습 루프 (OneVision)
│   │   └── llava_15.py                    # 학습 루프 (LLaVA-1.5)
│   └── eval/
│       ├── vlmeval_onevision_student.py    # Inference + VLMEvalKit
│       ├── lmms_onevision_student.py       # lmms-eval wrapper
│       └── milebench_onevision_student.py  # MileBench runner
│
├── train.py   # 학습 dispatcher (--model llava15 / onevision)
└── eval.py    # 평가 dispatcher
```

---

## Quick Reference: Paper Equation → Code

| 논문 수식 | 파일 | 핵심 코드 |
|---|---|---|
| Eq.(1) `a^{(l,m)}_{t,i}` | `teacher/collect_llava_onevision.py` | `output_attentions=True` + image 위치 슬라이싱 |
| Eq.(2) `s^{(l),teacher}` | `teacher/collect_llava_onevision.py` | head/token 평균 누적 |
| Eq.(3) `y^{(l),teacher}` | `teacher/collect_llava_onevision.py` | `/ (sum + 1e-8)` 정규화 |
| `q̄, x̄, C̄` | `visual_utility_student_onevision.py` | `q_proj`, `v_proj`, `cnn_encoder` |
| `ŝ_i^{(l)}` (fusion) | `visual_utility_student_onevision.py` | `torch.cat([...]) → mlp_head` |
| Eq.(4) `ℒ^{(l)}` | `train/llava_onevision.py` | `mse_loss + λ * pairwise_ranking_loss` |
| Eq.(5) `R^{(l)}` | `eval/vlmeval_onevision_student.py` | `torch.topk(scores, k=k_I)` |
| KV pruning | `eval/vlmeval_onevision_student.py` | `_trim_kv_cache_per_layer()` |
