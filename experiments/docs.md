# Future 지도학습 & Hybrid 점수 블렌딩 — All-Token 방법론 레퍼런스

**배경.** LLaVA-1.5-7B 의 KV cache pruning 을 **all-token 예산**(이미지·텍스트
구분 없이 프롬프트 전 위치를 후보로 하는 단일 budget) 위에서 수행하는 방법
문서. 본 문서는 *Future 지도학습 probe* 와, 추론 시 Future 신호에 H2O
(prefill 누적 어텐션) 를 블렌딩하는 *Hybrid* all-token scorer 를 재현 가능한
수준으로 정리한다. 수식, 텐서 shape, 코드 위치, 그리고 전체 파이프라인(수집
→ 학습 → 평가)을 한 곳에 모았다.

image-only 변형(텍스트 토큰을 예산 밖으로 두고 이미지 토큰만 top-k) 은 본
문서에서 다루지 않는다. 공정 비교를 위해 모든 비교군(LOOK-M, vanilla H2O,
Future, Hybrid) 은 동일한 `r_total` (= `total_keep_ratio`) 위에서 해석된다.

전체에서 사용하는 표기:

- `L`  = Transformer layer 수 (LLaVA-1.5-7B 의 경우 32).
- `D`  = hidden dimension (4096).
- `N_prompt` = 프롬프트 전체 토큰 수 (= `N_img + N_txt`).
- `H` = attention head 수; `H_kv` = key/value head 수.
- `T` = greedy-decoded 정답 토큰 수.
- `A^{(l)} ∈ R^{H × T × N_prompt}` = layer-l 의 *decode → prompt* 어텐션.
- `A_pre^{(l)} ∈ R^{H × N_prompt × N_prompt}` = layer-l prefill self-attention.
- `h^{(l)} ∈ R^{N_prompt × D}` = layer-l prefill hidden state.
- `r_total` = 보존할 프롬프트 토큰 비율 (예산 = `⌈r_total · N_prompt⌉`).

---

## 1. Future 지도학습 probe (all-token)

### 1.1 Teacher 신호 (증류 대상 oracle)

프롬프트 전 위치 `i ∈ {0, …, N_prompt−1}` 에 대해, Future teacher 점수는
*decode → 해당 위치* 어텐션을 모든 head 와 모든 decode step 에 걸쳐 평균한
값이다:

```
s_future^{(l)}(i) = (1 / (H · T)) · Σ_{h=1..H} Σ_{t=1..T}  A^{(l)}[h, t, i]        (1)
```

image-only 변형과 **수식은 동일**하되, `i` 가 이미지 인덱스 집합에 국한되지
않고 프롬프트 전 위치에 걸쳐 정의된다는 점이 핵심이다. 이 all-token teacher
는 "미래 decode 가 어느 텍스트 / 시스템 / 이미지 토큰에 실제로 접근하는가"
를 한 번에 말해준다.

**구현.** `collect_unified_teacher_shards.py::UnifiedCollector.collect_sample_unified`
(라인 274–281), `all_token_targets=True` 분기:

```python
if all_token_targets:
    h_out = h_l.to(storage_dtype)                         # [prompt_len, D]
    y_pv  = torch.zeros(prompt_len_mm, ...)              # PV 는 이미지 전용 — 0 채움
    fu_block = attn[:, decode_start:decode_end, :prompt_len_mm]  # [H, T, prompt_len]
    y_fu  = fu_block.mean(dim=0).mean(dim=0).to(storage_dtype)   # [prompt_len] — Eq. (1)
```

저장 shard schema 는 image-only 와 공유된다
(`{x, y_pv, y_future, layer, sample_id, token_idx}`). 단, `y_pv` 는 전 위치에
0 으로 채워지므로 all-token 학습에서는 `--teacher future` 만 의미 있다.

수집 CLI:

```bash
python collect_unified_teacher_shards.py \
  --dataset textvqa --data_dir /workspace/zap/data/textvqa/train \
  --out_dir /workspace/zap/artifacts/teacher/unified_alltoken/textvqa \
  --all_token_targets \
  --device cuda:0 --max_new_tokens 64
```

### 1.2 Probe 아키텍처

all-token 버전에서도 probe 는 layer 당 하나의 경량 MLP,
`KVzapModel` (`kvpress/presses/kvzap_press.py`, 라인 21–43) 로 구성된다:

```python
layers[l] = nn.Sequential(
    nn.Linear(D, H_mlp),
    nn.GELU(),
    nn.Linear(H_mlp, 1),
)                                    # D=4096, H_mlp=512 (default)
```

임의 프롬프트 위치 `i` 의 hidden state `h_i^{(l)} ∈ R^D` 에 대한 중요도
예측은

```
ŝ^{(l)}(i)  =  W2^{(l)} · GELU( W1^{(l)} · h_i^{(l)} + b1^{(l)} ) + b2^{(l)}        (2)
```

image-only 버전과 구조는 같고, **학습 대상 위치 집합이 `I` (이미지) 에서
프롬프트 전 위치로 확장**되었을 뿐이다.

### 1.3 학습 목적함수

샘플 하나에 프롬프트 토큰이 `N_prompt` 개, teacher 벡터가
`y ∈ R^{N_prompt}` 일 때, 라벨·예측 모두 *샘플 단위 softmax 정규화* 로 같은
simplex 위에 올린 뒤 MSE 로 매칭한다:

```
ỹ_i    = y_i / (Σ_j y_j + ε)                        (label 정규화)
p̂_i    = exp(ŝ_i) / Σ_j exp(ŝ_j)                    (샘플 단위 softmax)
L_l    = (1 / N_prompt) · Σ_i (p̂_i − ỹ_i)^2         (샘플 단위 MSE)          (3)
```

코드상 "method B" — `(sample_id, layer)` 로 그룹핑된 샘플 단위 softmax MSE.
참고 위치:

- `train_unified_probe_onepass.py::softmax_mse_loss` (라인 88–115) —
  `(sample_id, layer)` 기준 벡터화 그룹핑으로 **모든 layer 를 1-pass 학습**.
  all-token shard 를 그대로 소비할 수 있다 (`--teacher future`).

학습 CLI:

```bash
python train_unified_probe_onepass.py \
  --shard_dirs /workspace/zap/artifacts/teacher/unified_alltoken/textvqa \
               /workspace/zap/artifacts/teacher/unified_alltoken/scienceqa \
  --teacher future \
  --out_dir /workspace/zap/ckpts/future_probe_alltoken_32L \
  --n_layers_model 32 \
  --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
                    16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --mlp_max_epochs 20 \
  --device cuda:0
```

Validation: 샘플 단위 Spearman ρ, top-50% overlap, MSE
(`train_unified_probe_onepass.py::eval_all_layers`, 라인 118–170).

### 1.4 End-to-end 파이프라인

```
(1) 라벨 수집 (all-token)
    collect_unified_teacher_shards.py --all_token_targets   → shards/*.pt

(2) Probe 학습 (all-token)
    train_unified_probe_onepass.py --teacher future          → KVzapModel dir

(3) 추론 시 scoring press
    H2OAllTokenPress                          (baseline, Future 미사용)
    FutureAllTokenPress                       (Future 단독)
    HybridH2OFutureAllTokenPress              (H2O ⊕ Future 블렌드)

(4) 평가
    evaluate_image_teacher_pruning.py --mode {h2o_all_token,future_all_token,
                                              hybrid_h2o_future_all_token}
```

---

## 2. 추론 시 All-Token scorer

모든 scorer 는 `kvpress/presses/image_token_press.py` 의 `@dataclass` `Press`
객체다. 각 Press 는 `compress(module, hidden_states, keys, values, attentions,
kwargs) → (keys', values')` 를 제공하며, 이 훅은 *prefill 이후 layer 마다* 한
번씩 호출된다. `module.layer_idx` 는 절대 Transformer layer 인덱스다.

image-only press 와 달리 **이미지/텍스트 구분이 없다** — 예산은
`⌈r_total · N_prompt⌉` 이고, text 도 evict 대상이다.

### 2.1 H2O 단독 — `H2OAllTokenPress` (baseline)

라인 527–587. Zhang et al. 2023 의 heavy-hitter 기준을 layer 단위 prefill 에
그대로 적용한다. 쿼리 전 위치에 대한 어텐션을 합산하여 얻는 "받은 어텐션
총량" 으로 모든 KV 위치를 정렬한 뒤 상위 `K = ⌈r_total · N_prompt⌉` 를
유지한다:

```
s_h2o^{(l)}(i)    = (1 / H) · Σ_{h, q}  A_pre^{(l)}[h, q, i]        i ∈ {0,…,N_prompt−1}   (4)
K                 = ⌈r_total · N_prompt⌉
Keep^{(l)}        = top-K indices of s_h2o^{(l)}
```

참고 코드 (`compress`, 라인 551–587):

```python
h2o = attentions[0].sum(dim=1).float().unsqueeze(0)  # (1, H, kv_len)
h2o = _aggregate_scores_to_kv_heads(h2o, module, reduce=self.head_reduce)[0]
total_keep = int(math.ceil(self.total_keep_ratio * seq_len))
topk = torch.topk(h2o, k=total_keep, dim=-1).indices
```

Eager attention (`--attn_implementation eager`) 필수.

### 2.2 Future 단독 — `FutureAllTokenPress`

라인 590–659. all-token 으로 학습된 Future probe 를 prompt 전 hidden state 에
적용하여 점수를 낸다. H2O 어텐션은 사용하지 않는다:

```
s_future^{(l)}    = MLP_future^{(l)}( h^{(l)} )              # R^{N_prompt}         (5)
K                 = ⌈r_total · N_prompt⌉
Keep^{(l)}        = top-K indices of s_future^{(l)}
```

참고 코드 (`compress`, 라인 620–659):

```python
fu_layer = self._future_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()
with torch.no_grad():
    s_fu = fu_layer(hidden_states).transpose(1, 2)    # (1, 1, seq_len)
s_fu = _aggregate_scores_to_kv_heads(s_fu, module, reduce=self.head_reduce)[0]
```

어텐션 텐서를 읽지 않으므로 eager 가 **불필요** (SDPA 도 가능). PV probe 는
이미지 전용으로 정의되어 있어 all-token 시나리오에서는 쓰지 않는다.

### 2.3 Hybrid H2O ⊕ Future — `HybridH2OFutureAllTokenPress`

라인 662–755. **본 방법론의 핵심 변형**. H2O 와 Future 점수를 각각 프롬프트 전
위치에 걸친 softmax 로 정규화한 뒤 볼록결합한다:

```
s_h2o^{(l)}        = (1 / H) · Σ_{h, q}  A_pre^{(l)}[h, q, ·]            # R^{N_prompt}    (6)
s_fu^{(l)}         = MLP_future^{(l)}( h^{(l)} )                          # R^{N_prompt}
p_h2o^{(l)}        = softmax_all( s_h2o^{(l)} )
p_fu^{(l)}         = softmax_all( s_fu^{(l)} )
α_eff^{(l)}        = α        if (layer_set == ∅) or (l ∈ layer_set)
                   = 1        otherwise                                                   (7)
ŝ_blend^{(l)}      = α_eff^{(l)} · p_h2o^{(l)}  +  (1 − α_eff^{(l)}) · p_fu^{(l)}         (8)
K                  = ⌈r_total · N_prompt⌉
Keep^{(l)}         = top-K indices of ŝ_blend^{(l)}
```

`layer_set = future_blend_layers` 가 **per-layer gate**. 비어 있지 않으면 해당
Transformer layer 에서만 Future 가 블렌드되고, 그 외 layer 에서는 `α_eff = 1`
로 H2O-only 에 퇴화한다 (§3 참고). 비면 전역 α.

참고 코드 (`compress`, 라인 700–755):

```python
# H2O: (1, 1, seq_len)
h2o_per_tok = attentions[0].sum(dim=1).float().mean(dim=0)      # (seq_len,)
s_h2o = h2o_per_tok.view(1, 1, -1)

# Future: (1, 1, seq_len)
fu_layer = self._future_probe.layers[module.layer_idx].to(dev, dtype=dtype).eval()
with torch.no_grad():
    s_fu = fu_layer(hidden_states).transpose(1, 2)

s_h2o_norm = torch.softmax(s_h2o.float(), dim=-1)
s_fu_norm  = torch.softmax(s_fu.float(),  dim=-1)

if self.future_blend_layers and module.layer_idx not in self.future_blend_layers:
    effective_alpha = 1.0        # 이 layer 는 H2O-only
else:
    effective_alpha = self.alpha

s_blend = effective_alpha * s_h2o_norm + (1.0 - effective_alpha) * s_fu_norm
```

Eager attention 필수. 선택된 상위 `K` 인덱스는 정렬 후 `keys`/`values` 를
gather 한다 (라인 744–754).

---

## 3. Per-layer α 스케줄

all-token 블렌드에서도 동일하게 작동한다 — probe 의 **학습 범위** 와 **추론
적용 범위** 를 분리하는 용도. Future probe 가 일부 layer 집합
`L_train ⊆ {0,…,L−1}` 에서만 학습되었다면, 학습되지 않은 layer 에서 Future
신호는 노이즈로 기능한다.

식 (7) 의 gate 는 블렌딩을 `L_train` 에 국한한다:

- `future_blend_layers = L_train` → 해당 layer 에서만 Future 블렌드,
  그 외 layer 는 H2O-only.
- `future_blend_layers = ()` → 전역 α (모든 layer 에서 블렌드 — Future
  probe 가 전 layer 학습된 경우에만 안전).

KV cache 예산은 layer 간 불변이고, gate 는 *어느 토큰을 남기는지* 만 바꾼다.

CLI:

```bash
--mode hybrid_h2o_future_all_token \
  --alpha 0.25 \
  --future_blend_layers 24 25 26 27 28 29 30 31
```

Wiring: `evaluate_image_teacher_pruning.py::build_press` 의
all-token 분기 — `hybrid_h2o_future_all_token` (라인 368–378),
`future_all_token` (라인 360–367), `h2o_all_token` (라인 353–359).

---

## 4. KV 선택 & 예산 (all-token 공통)

all-token press 의 선택 단계는 image-only 계열의 `ImageTokenTopKPress` 와는
독립된 경량 로직이다. 절차는 세 press 모두 공통:

1. **예산 계산.** `K = ⌈r_total · N_prompt⌉`, 최소 1 이상으로 clamp.
2. **Layer 별 점수 조립.** `H2OAllTokenPress` 는 식 (4), `FutureAllTokenPress`
   는 식 (5), `HybridH2OFutureAllTokenPress` 는 식 (8). 점수는
   `_aggregate_scores_to_kv_heads` 로 KV head 축 (`H_kv`) 에 맞춰진다.
3. **Top-K gather.** `torch.topk(score, k=K, dim=-1).indices` 를 정렬한 뒤
   `keys`/`values` 를 gather.

image-only 와 달리 텍스트·이미지 분리, `forced-keep`, `_iterative_topk` 등
이미지 특화 옵션은 all-token 경로에 적용되지 않는다. 예산은 단일 축
`N_prompt` 위에서 결정된다.

---

## 5. 코드 ↔ 수식 매핑

| 수식 | 파일 | 라인 | 코드 심볼 |
|------|------|------|-----------|
| (1) Future teacher (all-token) | `collect_unified_teacher_shards.py` | 274–281 | `all_token_targets` 분기 |
| (2) Probe MLP | `kvpress/presses/kvzap_press.py` | 32–39 | `layers[l]` |
| (3) 학습 loss | `train_unified_probe_onepass.py` | 99–114 | `softmax_mse_loss` |
| (4) H2O all-token 점수 | `image_token_press.py` | 572–579 | `H2OAllTokenPress.compress` |
| (5) Future all-token 점수 | `image_token_press.py` | 641–646 | `FutureAllTokenPress.compress` |
| (6) all-token H2O (블렌드용) | `image_token_press.py` | 723–725 | `h2o_per_tok` |
| (7) Per-layer gate | `image_token_press.py` | 735–738 | `effective_alpha` 분기 |
| (8) H2O ⊕ Future 블렌드 | `image_token_press.py` | 722–740 | `HybridH2OFutureAllTokenPress.compress` |

---

## 6. 재현 명령 (예시)

**H2O all-token (baseline).**

```bash
python evaluate_image_teacher_pruning.py \
  --mode h2o_all_token \
  --dataset_path /workspace/zap/data/MileBench/DocVQA/DocVQA.json \
  --image_root   /workspace/zap/data/MileBench/DocVQA/images \
  --image_column images_path \
  --output_dir   /workspace/zap/artifacts/all_token/docvqa/h2o_k0p2 \
  --implementation_model_name /workspace/zap/ckpts/llava-1.5-7b-hf \
  --total_keep_ratio 0.2 \
  --prompt_style look_milebench \
  --attn_implementation eager \
  --truncate_like_lookm \
  --look_dataset_name DocVQA --look_model_name zap_docvqa \
  --look_result_root /workspace/zap/artifacts/combine_prob \
  --device cuda:0
```

**Future all-token.**

```bash
python evaluate_image_teacher_pruning.py \
  --mode future_all_token \
  ...
  --future_probe_name /workspace/zap/ckpts/future_probe_alltoken_32L \
  --total_keep_ratio 0.2 \
  ...
```

**Hybrid H2O ⊕ Future all-token (per-layer α).**

```bash
python evaluate_image_teacher_pruning.py \
  --mode hybrid_h2o_future_all_token \
  --dataset_path /workspace/zap/data/MileBench/DocVQA/DocVQA.json \
  --image_root   /workspace/zap/data/MileBench/DocVQA/images \
  --image_column images_path \
  --output_dir   /workspace/zap/artifacts/all_token/docvqa/hybrid_a025_L24-31_k0p2 \
  --implementation_model_name /workspace/zap/ckpts/llava-1.5-7b-hf \
  --future_probe_name /workspace/zap/ckpts/future_probe_alltoken_32L \
  --alpha 0.25 \
  --future_blend_layers 24 25 26 27 28 29 30 31 \
  --total_keep_ratio 0.2 \
  --prompt_style look_milebench \
  --attn_implementation eager \
  --truncate_like_lookm \
  --look_dataset_name DocVQA --look_model_name zap_docvqa \
  --look_result_root /workspace/zap/artifacts/combine_prob \
  --device cuda:0
```

---

## 7. 설계 메모

**왜 all-token 예산인가?** LOOK-M, vanilla H2O 등 기존 baseline 은 이미지·
텍스트를 구분하지 않고 전체 KV 위에 예산을 부여한다. 공정 비교를 위해 Future
/ Hybrid 계열도 동일한 `r_total` 위에서 평가해야 r_eff_prompt 가 맞는다.
image-only 변형은 text 를 강제 보존하므로 같은 nominal ratio 라도 실제로는
유리한 예산을 쥐고 있다 — 비교 대상이 될 수 없다.

**왜 블렌딩 전에 softmax 인가?** H2O 점수 (어텐션 합) 와 probe logit 은
스케일이 전혀 다르다. 전 위치 `N_prompt` 축에서 샘플 단위 softmax 정규화를
하지 않으면 α 는 스케일이 큰 쪽에 지배되고, 블렌드는 사실상 더 큰 채널로
퇴화한다. Softmax 로 두 쪽 모두 확률 simplex 위에 올리면 α 가 순수한 보간
가중치로 의미를 갖는다.

**왜 모든 layer 에 블렌딩하지 않는가?** 실험적으로 (EXP-20260418 계열), 전역
α 는 대부분의 dataset 에서 H2O-only 또는 Future-only 극단으로 쏠렸다 —
블렌드 자체가 희석되고 있다는 신호. Future 를 학습 범위로 제한하면 가장 큰
희석 원인(학습하지 않은 hidden-state 분포 위의 노이즈)이 제거된다.
`future_blend_layers` gate 는 이 가설을 검증하기 위해 도입되었다.

**PV probe 는 왜 all-token 에서 쓰지 않는가?** PostVision teacher 는 식
`y_pv(i) = max_q mean_h A_pre[h, q ∈ Q_pv, i ∈ I]` 로 *이미지 토큰에 대해서만*
정의된다 (post-vision 텍스트 query 가 이미지 key 를 얼마나 보는지). 텍스트 /
시스템 위치에 대한 자연스러운 라벨이 없으므로, all-token 시나리오에서는
Future 만 유효한 학습 가능 신호가 된다. 그래서 all-token 패밀리의 Hybrid 는
`PV ⊕ Future` 가 아니라 `H2O ⊕ Future` 이다.

**`FutureAllTokenPress` 에는 왜 `selected_layer_indices` 안전장치가 없나?**
all-token 변형은 설계상 **전 layer 학습된** Future probe 를 전제한다
(§1.3 학습 CLI 의 `--selected_layers 0..31`). 일부 layer 만 학습된 probe 로
all-token pruning 을 돌리는 것은 정의상 안전하지 않으므로, 해당 시나리오에서는
image-only 경로의 `HybridImageTeacherPress` + `selected_layer_indices` 를
사용하거나, 전 layer 학습 후 본 문서의 all-token 경로를 써야 한다.
