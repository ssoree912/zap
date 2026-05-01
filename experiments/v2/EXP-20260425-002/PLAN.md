## Experiment Plan

**ID**: EXP-20260425-002
**Author**: ssoree912
**Date**: 2026-04-25
**Branch**: `4090/cnn_image_scorer`
**Status**: [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned
**Parent**: EXP-20260417-001 (image-only future probe, MLP scorer)

---

### 1. 동기 (Motivation)

기존 future probe (`KVzapModel`, `kvpress/presses/kvzap_press.py`) 는 image token 의 hidden
state 를 **token-by-token** 으로만 보는 2-layer MLP 다:

[
\hat{s}_i^{(l)} = \mathrm{MLP}^{(l)}\big(h_i^{(l)}\big), \quad h_i^{(l)} \in \mathbb{R}^{D}
]

세 가지 한계가 있다.

1. **Spatial blindness**: image token 은 본래 ViT 24×24 patch grid 에서 유래하지만, MLP 는
   주변 patch 를 모른다 (object boundary, OCR region, local texture, background repetition
   등을 무시).
2. **Question agnostic**: 동일한 image token 이라도 사용자 질문에 따라 중요도가 달라져야
   하는데, MLP 는 question token hidden state 를 보지 않는다.
3. **Single-stream**: hidden state 한 가지 view 만 사용. 같은 token 의 raw representation
   과 contextualized representation 의 상호작용을 활용하지 못한다.

본 실험에서는 위 세 가지를 동시에 해소하는 **3-branch student**

[
H^{(l)}*{\mathrm{img}},; H^{(l)}*{\mathrm{q}}
\xrightarrow{\text{CNN+pool+proj+fusion}}
\hat{s}^{(l)} \in \mathbb{R}^{B \times N_I}
]

를 도입한다. teacher 는 **future-decode attention** 의 head·step 평균 (image position
한정) 을 사용 (기존 future probe 의 teacher 와 동일 정의), student 는 prefill hidden state
한 번 forward 만으로 동일한 score 를 추론한다.

---

### 2. 가설 (Hypothesis)

**H1 (Student > MLP, image-only)**: 같은 future-decode teacher · 같은 layer 수에서
3-branch student 의 detail_1k PPL ≤ MLP scorer 의 PPL. 차이는 keep_ratio = 0.3 ~ 0.5 에서
가장 크다.

**H2 (Question branch ablation)**: question branch 제거 시 PPL 이 기존 MLP 와 student
사이로 회귀. 즉 question conditioning 이 필수 기여.

**H3 (CNN branch ablation)**: CNN branch 제거 시에도 PPL 이 MLP 보다는 낮지만, full student
보다는 높다. spatial context 와 question conditioning 이 **상보적**.

**H4 (Ranking loss 효과)**: `λ_rank > 0` 인 모델이 MSE-only 모델보다 top-k retention 정확도
와 PPL 모두에서 우수. eviction 은 본질적으로 **순위** 작업이므로 ranking 신호가 직접
도움이 된다.

**H5 (Teacher cache 의 일관성)**: pre-collected teacher 와 on-the-fly 재계산 teacher 의
샘플 평균 cosine similarity ≥ 0.99 (i.e. determinism 확보).

---

### 3. 독립변수 (What we change)

| 변수 | 값 |
|------|----|
| Scorer arch | (a) `MLP-2L` (baseline), (b) `student_full` (CNN + Q + raw, **default**), (c) `student_no_q` (Q branch 제거), (d) `student_no_cnn` (CNN branch 제거 → raw + Q) |
| Loss | `mse_only` vs `mse + 0.1·rank` (default) |
| Keep ratio | 0.3, 0.5, 0.7 (image token 기준) |
| Dataset | mm-vet (218), detail_1k (1000) |
| Eviction scope | image-only (사용자 명시 — `feedback_eviction_default.md` default 와 다름, §9 caveat) |

> Layer 선택은 본 실험 범위 밖 → 기존 32-layer 전부 학습. layer ablation 은 EXP-20260425-002B (조건부, §14.A).

---

### 4. 종속변수 (What we measure)

#### Primary
- **PPL** (`eval_ppl.py`)
- **ΔPPL_vs_MLP** = `PPL_student - PPL_MLP_2L` (같은 keep_ratio · dataset)

#### Secondary
- **ROUGE-L** (`eval_rouge.py`)
- **Top-k retention** = `|top-k(s_pred) ∩ top-k(s_teacher)| / k`, k = N_I·keep_ratio (per layer 평균)
- **Spearman r** = student-teacher 순위 상관 (image positions 한정, per-(sample, layer))
- **Scorer params** + **probe_score_total_ms** (B=1, detail_1k 평균)
- **학습 곡선**: train/val MSE, ranking loss, total loss

---

### 5. 고정 조건 (What stays the same)

| 항목 | 값 |
|------|-----|
| 모델 | `/workspace/zap/ckpts/llava-1.5-7b-hf` (D=4096, 32 layer, 24×24=576 image grid) |
| Teacher 정의 | `s*_i = (1/HT)·Σ_h Σ_t A^(l)[h, t, last_query, image_pos_i]` (head·step 평균, image position 한정) |
| Teacher 생성 | **Stage 1 사전 cache** — `output_attentions=True, return_dict_in_generate=True` 로 generate, step 별 마지막 query attention 만 stack |
| Student 입력 | **별도 prefill forward** (`output_hidden_states=True, use_cache=True`) — generate 출력 재사용 안 함 |
| 학습 데이터 | ScienceQA primary (~5k 샘플), unified shard 형식 확장 (§11.2) |
| Layer 선택 | 32 layer 전부 (per-layer 독립 student) |
| Optimizer | AdamW, lr=1e-3, wd=0.0, batch=1 sample × all layers |
| Epoch | 10 |
| dtype | bf16 (master fp32) |
| attn_implementation | sdpa (image-only 추론) / **eager (teacher 수집 시 attention 출력 필수)** |
| seed | `torch.manual_seed(0)` |
| eval_samples | mm-vet 218 / detail_1k 1000 (전체) |
| Question 식별 | LLaVA-1.5 prompt template 기준 — `<image>` placeholder 이후 user role 의 마지막 turn 텍스트 토큰 (assistant 응답 토큰 제외). 정확한 slice 는 §11.1 확정 |

---

### 6. 베이스라인

| 방법 | 설명 |
|------|------|
| `full` | KV eviction 없음 (PPL upper bound) |
| `random` | image-only random eviction, 동일 keep_ratio |
| `MLP-2L` | 기존 `KVzapModel hidden=512` (`ProbeImageTeacherPress`) |
| `student_no_q` (ablation) | CNN + raw 만, question 없음 |
| `student_no_cnn` (ablation) | raw + Q 만, CNN 없음 |

---

### 7. 핵심 수식 정리

#### 7.1 Notation
[
H^{(l)} \in \mathbb{R}^{B\times N\times D},\quad
H_{\mathrm{img}}^{(l)} \in \mathbb{R}^{B\times N_I\times D},\quad
H_q^{(l)} \in \mathbb{R}^{B\times N_Q\times D}
]
[
N_I = H_I W_I = 576,\ H_I=W_I=24,\ D=4096,\ C=d=256
]

#### 7.2 Teacher
[
A_t^{(l)} \in \mathbb{R}^{B\times H\times Q_t\times K_t}
,\quad
\bar{A}*t^{(l)} = A_t^{(l)}[:,:,-1,:]
,\quad
\bar{A}*{t,\mathrm{img}}^{(l)} = \bar{A}*t^{(l)}[:,:,\mathcal{I}*{\mathrm{img}}]
]
[
A_{\mathrm{img}}^{(l)} = \mathrm{Stack}*{t=1}^{T}\big(\bar{A}*{t,\mathrm{img}}^{(l)}\big)
\in\mathbb{R}^{B\times H\times T\times N_I}
]
[
s_{b,i}^{*(l)} = \frac{1}{HT}\sum_{h,t} A_{\mathrm{img}}^{(l)}[b,h,t,i]
,\quad
\tilde{s}_{b,i}^{*(l)} = \frac{s_{b,i}^{*(l)}}{\sum_j s_{b,j}^{*(l)}+\varepsilon}
]

#### 7.3 Student
**Image CNN branch**
[
F_{\mathrm{img}}^{(l)} = \mathrm{Grid}(H_{\mathrm{img}}^{(l)}) \in \mathbb{R}^{B\times D\times H_I\times W_I}
,\
X = \mathrm{Conv}*{1\times1}(F),\
C = \mathrm{ConvBlock}(X) \in \mathbb{R}^{B\times C\times H_I\times W_I}
]
[
\bar{C}*{\mathrm{flat}}^{(l)} = W_c \cdot \mathrm{Flatten}(C) \in \mathbb{R}^{B\times N_I\times d}
]

**Question branch** (mean-pool + project + broadcast)
[
\bar{q}^{(l)} = W_q \cdot \mathrm{Pool}(H_q^{(l)})\in\mathbb{R}^{B\times d}
,\quad
Q^{(l)} = \mathrm{Broadcast}(\bar{q}^{(l)}, N_I)\in\mathbb{R}^{B\times N_I\times d}
]

**Raw image token projection**
[
\bar{H}*{\mathrm{img}}^{(l)} = W_h H*{\mathrm{img}}^{(l)} \in \mathbb{R}^{B\times N_I\times d}
]

**Fusion** (concat with two Hadamard interactions)
[
Z^{(l)} = \big[ \bar{H}*{\mathrm{img}};\ \bar{C}*{\mathrm{flat}};\ Q;\ \bar{H}*{\mathrm{img}}\odot Q;\ \bar{C}*{\mathrm{flat}}\odot Q \big] \in \mathbb{R}^{B\times N_I\times 5d}
]

**MLP head** (per-token)
[
\hat{s}_i^{(l)} = W_2 \cdot \mathrm{GELU}(W_1 z_i^{(l)} + b_1) + b_2,\
W_1\in\mathbb{R}^{d_m\times 5d},\ W_2\in\mathbb{R}^{1\times d_m},\ d_m=512
]

**ConvBlock** (ConvNeXt-style residual)
[
Y = \mathrm{DWConv}*{7\times 7}(X) \to \mathrm{Norm}\to \mathrm{Conv}*{1\times1}^{C\to 4C}\to\mathrm{GELU}\to\mathrm{Conv}_{1\times1}^{4C\to C}
,\
\mathrm{ConvBlock}(X)=X+Y
]

#### 7.4 Loss
[
\mathcal{L}*{\mathrm{MSE}} = \tfrac{1}{BN_I}\sum*{b,i} (\hat{s}*{b,i}^{(l)} - \tilde{s}*{b,i}^{*(l)})^2
]
[
\mathcal{P} = {(i,j)\mid s_i^* \in \text{top-}r_t,\ s_j^* \in \text{bottom-}r_b}
]
[
\mathcal{L}*{\mathrm{rank}} = \tfrac{1}{|\mathcal{P}|}\sum*{(i,j)\in\mathcal{P}} \max\big(0,\ m - (\hat{s}_i - \hat{s}_j)\big)
]
[
\mathcal{L} = \mathcal{L}*{\mathrm{MSE}} + \lambda*{\mathrm{rank}}\mathcal{L}_{\mathrm{rank}},\
\lambda_{\mathrm{rank}}=0.1,\ m=0.05,\ r_t=0.2,\ r_b=0.4
]

---

### 8. 예상 결과

PPL on detail_1k (full ≈ 2.95) at keep_ratio = 0.5:

| Scorer | PPL (예상) | params/layer | 비고 |
|--------|-----------|--------------|------|
| full | 2.95 | — | upper bound |
| random | 3.40 | 0 | lower bound |
| MLP-2L (baseline) | 3.05 ~ 3.10 | ~2.10 M | 기존 |
| student_no_cnn | 3.02 ~ 3.07 | ~2.5 M | Q-cond. only — H3 partial credit |
| student_no_q | 3.00 ~ 3.05 | ~2.0 M | spatial only — H2 partial credit |
| **student_full** | **2.95 ~ 3.02** | **~3.0 M** | **타깃** — MLP 대비 -0.05 ~ -0.10 |
| student_full (mse_only) | 2.97 ~ 3.04 | ~3.0 M | rank loss ablation |

> mm-vet 은 OCR/text 비중이 높아 student_full 의 gap 이 detail_1k 대비 작을 것으로 예상.
> H4 의 retention/Spearman 은 student_full 이 모든 레이어에서 baseline 대비 상승해야 함.

---

### 9. 판단 기준 (Success Criteria)

**Primary (H1 지지)**:
`PPL(student_full) < PPL(MLP-2L) - 0.03` on detail_1k, k ∈ {0.3, 0.5}.

**Ablation (H2)**:
`PPL(student_no_q) > PPL(student_full)` AND `PPL(student_no_q) ≥ PPL(MLP-2L) - 0.02`
→ question branch 가 핵심 기여.

**Ablation (H3)**:
`PPL(student_no_cnn) > PPL(student_full)` AND `PPL(student_no_cnn) < PPL(MLP-2L) - 0.02`
→ CNN branch 도 유의미 기여.

**Loss (H4)**:
`top-k retention(rank+mse) > top-k retention(mse_only)` (≥ 2%p) on detail_1k.

**Determinism (H5)**:
mean cosine sim ≥ 0.99 between cached teacher and re-collected teacher (10 random samples).

**Failure**:
`PPL(student_full) ≥ PPL(MLP-2L)` on detail_1k → §14.B 진단.

---

### 10. 이 실험으로 증명할 수 없는 것

- **All-token eviction 일반화**: 본 실험은 image-only. text 토큰은 2D grid 가 없으므로 CNN
  branch 적용 불가. all-token scope 효과는 별도 실험 필요.
- **Multi-image / any-resolution**: LLaVA-1.5 single-image 24×24 가정. Q-Former / Perceiver
  query token, multi-crop, post-compression 토큰은 grid 복원이 자연스럽지 않음.
- **Layer 선택 최적성**: 32 layer 전부 학습. 어느 layer hidden state 가 가장 유효한지는
  EXP-...A에서 별도 ablation.
- **Question pooling 선택**: mean pooling 가정. last-token / attention-pool / multi-turn
  처리는 본 실험 범위 밖.
- **인과 분해**: student 가 잘 한다고 해서 "object boundary 를 읽었기 때문" 인지 단정 불가.
  §13.4 attention/score heatmap 으로 정성적 단서만 확보.

---

### 11. 구현 세부사항

#### 11.1 Question token 식별

LLaVA-1.5 chat template 은 다음과 같다:

```
USER: <image>\n{question} ASSISTANT:
```

prefill 시점에는 ASSISTANT 응답이 없으므로 prompt 끝까지가 input.

- `image_token_indices`: 기존 `infer_llava_image_positions_no_forward()` 재사용
  (`kvzap/llava_extractor.py:123`).
- `question_token_indices`: image block 의 마지막 위치 + 1 부터 prompt 끝 (assistant turn
  marker 제외) 까지의 텍스트 토큰 인덱스. 시스템 prompt / role tag 토큰 제외 여부는
  별도 ablation 가능하나 본 실험은 **포함** (가장 단순한 정의 우선).

→ `kvpress/presses/image_token_press.py` 또는 `kvzap/llava_extractor.py` 에
`infer_question_positions(input_ids, image_positions)` 헬퍼 추가.

#### 11.2 Teacher pre-collection (Stage 1)

신규 스크립트: `collect_future_teacher_v2.py` (기존 `collect_unified_teacher_shards.py` 와
별개로 시작 — schema 가 다름).

```python
gen_out = lvlm.generate(
    **inputs,
    max_new_tokens=T_max,
    do_sample=False,
    output_attentions=True,
    output_hidden_states=False,
    return_dict_in_generate=True,
    use_cache=True,
)

# decode_attentions: tuple(T) of tuple(L) of [B, H, Q_t, K_t]
A_img_layers = []  # per layer
for l in range(num_layers):
    per_t = []
    for step_attn in gen_out.attentions:
        a = step_attn[l][:, :, -1, image_token_indices]   # (B, H, N_I)
        per_t.append(a)
    A_img = torch.stack(per_t, dim=2)                     # (B, H, T, N_I)
    s_star = A_img.mean(dim=(1, 2))                       # (B, N_I)
    s_star = s_star / (s_star.sum(-1, keepdim=True) + 1e-8)
    A_img_layers.append(s_star)

teacher = torch.stack(A_img_layers, dim=1)                # (B, L, N_I)
```

각 sample 별로 다음을 저장:
- `sample_id` (str)
- `teacher` (L, N_I) fp16
- `image_token_indices` (LongTensor)
- `question_token_indices` (LongTensor)
- `grid_h, grid_w` (int)
- `T` (decoded steps), `T_max` (cap)

저장 경로: `/workspace/zap/data/teacher_v2_scienceqa/{sample_id}.pt`

Determinism check (H5): 임의 10 샘플에 대해 두 번 수집 → cosine sim ≥ 0.99.

#### 11.3 Student 모듈

신규 파일: `kvpress/presses/visual_utility_student.py`.

```python
class ConvNeXtStyleBlock(nn.Module):
    """[N, C, H, W] -> [N, C, H, W] residual block (DW7x7 + GN + 1x1×4 + GELU + 1x1)."""

class VisualUtilityStudent(nn.Module):
    """3-branch student.

    Args:
      hidden_dim=4096, conv_dim=256, proj_dim=256, mlp_dim=512, num_conv_blocks=2.

    forward(H_l, image_token_indices, question_token_indices, grid_h, grid_w) -> (B, N_I)
    """
```

저장은 HuggingFace `PreTrainedModel` 스타일로 통일 (`from_pretrained` 가능하도록 config +
state dict).

#### 11.4 Press: `VisualUtilityStudentPress`

신규 dataclass: `kvpress/presses/image_token_press.py` 에 추가, `ImageTokenTopKPress` 상속.

```python
@dataclass
class VisualUtilityStudentPress(ImageTokenTopKPress):
    student_model_name: str = ""
    grid_h: int = 24
    grid_w: int = 24
    _student: Optional[VisualUtilityStudent] = field(default=None, init=False, repr=False)
    _loaded_name: Optional[str] = field(default=None, init=False, repr=False)
    _question_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def set_question_positions(self, q_positions: torch.Tensor) -> None:
        self._question_positions = q_positions

    def post_init_from_model(self, model):
        if self.student_model_name != self._loaded_name:
            self._loaded_name = self.student_model_name
            self._student = VisualUtilityStudent.from_pretrained(self.student_model_name)

    def score_image_tokens(self, module, hidden_states, keys, values, attentions, kwargs, image_positions):
        if self._student is None:
            raise RuntimeError("Student model not loaded")
        if self._question_positions is None:
            raise RuntimeError("Call set_question_positions() before generate()")
        # student 는 모든 layer 의 score 를 동시에 계산하지 않고, layer 별 forward 호출.
        # 단순화: 각 layer 마다 독립 student instance — config.n_modules=32, layer_idx 로 라우팅.
        ...
```

> 기존 image-only press 와 동일하게 `set_image_positions()` 외에 `set_question_positions()`
> 가 generate 직전 호출되어야 함. evaluation 스크립트 (§11.6) 에서 inputs 처리 시 자동 설정.

#### 11.5 학습 스크립트

신규: `train_visual_utility_student.py` (기존 `train_unified_probe_onepass.py` fork).

Stage 2 학습 루프:

```python
for epoch in range(E):
    for batch in dl_train:
        teacher = load_teacher_batch(batch.sample_ids)        # (B, L, N_I)
        with torch.no_grad():
            out = lvlm(**batch.inputs, output_hidden_states=True, use_cache=False)
            H_all = out.hidden_states                          # tuple(L+1) of (B, N, D)

        for l in range(L):
            H_l = H_all[l + 1]
            s_pred = student_l(
                H_l, batch.image_token_indices, batch.question_token_indices,
                batch.grid_h, batch.grid_w,
            )                                                  # (B, N_I)
            l_mse = F.mse_loss(s_pred, teacher[:, l, :])
            l_rank = pairwise_ranking_loss(s_pred, teacher[:, l, :], top_ratio=0.2, bottom_ratio=0.4, margin=0.05)
            loss = l_mse + 0.1 * l_rank
            loss.backward()  # 32 layer 누적
        opt.step(); opt.zero_grad()
```

> hidden state 는 1회 forward 로 32 layer 모두 확보. teacher 는 cache 에서 load. 메모리는
> (B=1) × (32 layer × 4096) × N(prefill) bf16 ≈ 수백 MB 수준 → 4090 24GB 에 충분.

#### 11.6 평가 스크립트

`eval_ppl.py`, `eval_rouge.py` 에 분기 추가:

```
--method visual_utility_student
--student-model-name <ckpt path>
--grid-h 24 --grid-w 24
```

press 인스턴스 생성 직후, sample 마다:
1. `image_positions = infer_llava_image_positions_no_forward(...)`
2. `question_positions = infer_question_positions(input_ids, image_positions)`
3. `press.set_image_positions(image_positions)`
4. `press.set_question_positions(question_positions)`
5. `model.generate(..., past_key_values=press_wrapped)`

#### 11.7 Sweep 스크립트

```bash
# experiments/EXP-20260425-002/run_sweep.sh
ARCHES=("mlp" "student_full" "student_no_q" "student_no_cnn" "student_full_mse_only")
KEEP_RATIOS=("0.3" "0.5" "0.7")
DATASETS=("mm-vet" "detail_1k")

# Stage 1: teacher cache (한 번만)
python /workspace/zap/collect_future_teacher_v2.py \
  --dataset scienceqa --output-dir /workspace/zap/data/teacher_v2_scienceqa \
  --max-new-tokens 64

# Stage 2: train per-arch
for ARCH in "${ARCHES[@]}"; do
  python /workspace/zap/train_visual_utility_student.py \
    --arch ${ARCH} \
    --teacher-dir /workspace/zap/data/teacher_v2_scienceqa \
    --epochs 10 \
    --output /workspace/zap/ckpts/student_${ARCH}
done

# Stage 3: evaluate
for ARCH in "${ARCHES[@]}"; do
  for DS in "${DATASETS[@]}"; do
    for KR in "${KEEP_RATIOS[@]}"; do
      python /workspace/zap/eval_ppl.py \
        --method visual_utility_student \
        --student-model-name /workspace/zap/ckpts/student_${ARCH} \
        --keep-ratio ${KR} --data-path /workspace/data/${DS} \
        --output-dir /workspace/zap/artifacts/EXP-20260425-002/${DS}/${ARCH}_k${KR}/ppl
      python /workspace/zap/eval_rouge.py [...동일...]
    done
  done
done
```

---

### 12. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| `output_attentions=True` + `generate()` 가 `eager` attention 강제 → OOM (long context) | teacher 수집 실패 | scienceqa 짧은 prompt 우선; layer-by-layer attention consumption (EXP-20260420-003 패턴) |
| Cache implementation 차이 (DynamicCache vs legacy tuple) — KV pruning 시 간섭 | inference crash | press 측 `prune_legacy_tuple_kv_cache` 가 아닌 기존 `BasePress` hook 경로 그대로 사용 (already supports both via hf compress hook) |
| Question token slice 정의 모호 (system prompt 포함 여부) | student_no_cnn 가 의외로 잘 / 못 함 | 사전 단위 테스트로 `infer_question_positions` 출력을 5 샘플 디코딩해 sanity 확인 |
| Multi-image / any-resolution 샘플 | reshape assertion 실패 | `_find_image_blocks` 로 chunk 후 각 24×24 block 별 CNN. block 크기 ≠ 576 인 샘플은 학습/평가에서 skip + 카운트 |
| Pre-collected teacher 의 결정성 (greedy + temperature 0) | sample 간 inconsistency | H5 determinism check. 실패 시 `do_sample=False, num_beams=1, top_k=0, temperature=1.0` 명시 + seed 고정 |
| Ranking loss 의 batch=1 통계 불안정 | 학습 발산 | `top_ratio=0.2`, `bottom_ratio=0.4` 로 비율 기반 → N_I 변동에 안정. 손실 NaN 시 grad clip 1.0 적용 |
| Teacher cache 디스크 점유 | scienceqa 5k × 32 × 576 × fp16 ≈ 600 MB → 무난 | 그대로 진행 |
| MLP baseline 가 EXP-20260417-001 결과와 불일치 | 비교 불공정 | sweep 전 sanity 재현 (±0.02 PPL) |

---

### 13. 분석 계획

#### 13.1 Main Result Table

```
Table A. PPL on detail_1k (full = 2.95)
Scorer                  | k=0.3 | k=0.5 | k=0.7 | params/L | ms/sample | top-k retention | Spearman r
------------------------|-------|-------|-------|----------|-----------|-----------------|-----------
random                  |   ?   |   ?   |   ?   |    0     |    0      |       n/a       |    n/a
MLP-2L (baseline)       |   ?   |   ?   |   ?   |    ?     |    ?      |        ?        |     ?
student_no_cnn          |   ?   |   ?   |   ?   |    ?     |    ?      |        ?        |     ?
student_no_q            |   ?   |   ?   |   ?   |    ?     |    ?      |        ?        |     ?
student_full (default)  |   ?   |   ?   |   ?   |    ?     |    ?      |        ?        |     ?
student_full mse_only   |   ?   |   ?   |   ?   |    ?     |    ?      |        ?        |     ?
```

`Table B` = mm-vet 동일 포맷. `Table C` = ROUGE-L.

#### 13.2 시각화

- **Line plot**: x=keep_ratio, y=PPL — 6 라인 (full, random, MLP, no_q, no_cnn, full).
- **Bar chart**: scorer params / FLOPs / latency.
- **Score heatmap**: 임의 sample 1~3 개에서 24×24 score grid 를 image overlay (teacher
  vs student_full vs MLP). 정성적 단서.
- **Per-layer Spearman r**: x=layer index (0..31), y=Spearman — student_full 곡선이 다른
  것 위에 있는지.

#### 13.3 가설 검증 체크리스트
- [ ] H1: `PPL(student_full) < PPL(MLP) - 0.03` (detail_1k, k ∈ {0.3, 0.5})
- [ ] H2: `PPL(student_no_q) > PPL(student_full)` AND `PPL(student_no_q) ≈ PPL(MLP) ± 0.02`
- [ ] H3: `PPL(student_no_cnn) > PPL(student_full)` AND `PPL(student_no_cnn) < PPL(MLP) - 0.02`
- [ ] H4: top-k retention (rank+mse) > (mse_only) by ≥ 2%p
- [ ] H5: cosine sim teacher-recollect ≥ 0.99 (10 샘플 평균)

#### 13.4 실패 시 진단

- H1 실패: per-layer Spearman 으로 student 가 어느 layer 에서 약한지 식별 → §14.B 분기.
- H2/H3 동시 실패 (둘 다 풀-student 와 동등): branch redundancy → fusion 단순화 (concat 대신
  sum or single hadamard).
- H5 실패: teacher 결정성 문제 → `do_sample=False` 외 `temperature=1.0, top_p=1.0, top_k=0`
  명시, 시드 추가, KV-cache reset.

---

### 14. 다음 실험 (조건부)

#### A. H1 지지 시
- **A1 — Layer ablation**: 32 layer 중 가장 유효한 일부만 student 적용 (1L / L/4 / L/2 / L).
- **A2 — RF / depth sweep**: `num_conv_blocks` ∈ {1, 2, 3}, kernel ∈ {3, 5, 7}.
- **A3 — All-token 변형**: image part 만 CNN+Q+raw, text part 는 raw+Q (default scope 회귀).

#### B. H1 실패 시
- **B1 — Probe layer 진단**: per-layer 별 student vs MLP PPL → 어느 layer 만 student 이득.
- **B2 — Question encoder 강화**: mean-pool → attention-pool / 마지막 토큰 / 작은 transformer.
- **B3 — Teacher 재정의**: future-decode → "answer-token only" attention (질문/시스템 토큰
  생성분 제외).

---

### 15. 체크리스트

#### 사전작업
- [ ] LLaVA-1.5 prompt template 으로 5 샘플 input_ids 디코딩하여 `infer_question_positions`
      출력 sanity check
- [ ] MLP baseline ckpt 로 detail_1k PPL 재현 (EXP-20260417-001 ±0.02)
- [ ] `output_attentions=True, return_dict_in_generate=True` generate 가 OOM 없이 scienceqa
      샘플 1개에서 통과하는지 smoke test

#### Stage 1 (teacher cache)
- [ ] `collect_future_teacher_v2.py` 구현
- [ ] determinism check (10 샘플 cosine sim ≥ 0.99)
- [ ] scienceqa 전체 cache 생성 (~2~3h 예상)

#### Stage 2 (학습)
- [ ] `VisualUtilityStudent`, `ConvNeXtStyleBlock` 구현 + 단위 테스트
- [ ] `train_visual_utility_student.py` 구현
- [ ] 5 arch × 10 epoch 학습 (~30min × 5 = 2.5h)
- [ ] train/val loss 곡선 확인 (수렴 + 과적합 여부)

#### Stage 3 (평가)
- [ ] `VisualUtilityStudentPress` 구현 + `--eval-samples 3` 단위 테스트
- [ ] `eval_ppl.py`, `eval_rouge.py` 분기 추가
- [ ] 5 arch × 3 keep_ratio × 2 datasets = 30 runs 실행 (~8h)

#### 실험 후
- [ ] H1 ~ H5 verdict 기록
- [ ] Score heatmap 1~3 장 첨부
- [ ] `RESULT.md` 작성
- [ ] `experiments/README.md` 인덱스 업데이트

---

### 16. 예상 런타임 / 리소스

| 항목 | 값 |
|------|-----|
| GPU | RTX 4090 1장 |
| Stage 1 (teacher) | scienceqa 5k 샘플 × ~2s/샘플 ≈ 3h |
| Stage 2 (train) | 5 arch × 30min ≈ 2.5h |
| Stage 3 (eval) | 30 runs × 평균 16min ≈ 8h |
| **총** | **~14h** |
| 디스크 | teacher cache ~600 MB + ckpt ~250 MB + logs ~50 MB |
| 출력 경로 | `/workspace/zap/artifacts/EXP-20260425-002/...`, `/workspace/zap/ckpts/student_<arch>/` |
