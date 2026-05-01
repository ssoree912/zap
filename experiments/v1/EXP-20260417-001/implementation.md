# Implementation Journal — EXP-20260417-001
# True Iterative Probe Pruning (A-option) + Phase 2 Auxiliary Metrics

Date: 2026-04-17

---

## 1. 구현 전 설계 메모

### Phase 2: 보조 메트릭 수집

```
Module: phase2_metrics.py

[목적]
probe press가 oracle과 얼마나 일치하는지 정량화.
probe가 맞는 token을 고르는지(overlap), 선택 분포가 spatially 다양한지(entropy)를 측정.

[왜 이 방식인가]
- task accuracy만으로는 probe가 "운 좋게" 맞는지 "구조를 학습했는지" 구분 불가
- oracle keep mask와 overlap ratio → probe가 teacher와 같은 판단을 내리는지 직접 측정
- spatial entropy → cluster에 몰려 선택하는지 vs. 전체 image에 분산 선택하는지 측정

[핵심 가정]
- VizCapture가 per-layer keep_mask를 정확히 기록함
- oracle teacher score는 att_only_postvision 또는 splus_postvision .pt 파일에서 로드됨
- LLaVA-1.5 기준 n_image_per_image=576, grid_side=24

[이 방식의 한계]
- 메트릭이 높다고 task accuracy가 높은 것은 아님 (diagnostic용)
- oracle score와의 비교이므로 oracle score 자체의 quality에 의존

[관련 코드]
- kvzap/phase2_metrics.py (신규)
- evaluate_image_teacher_pruning.py: --collect_phase2_metrics 플래그
```

### Phase 1: True Iterative Pruning (A-option)

```
Module: _iterative_probe_preselect() in evaluate_image_teacher_pruning.py

[목적]
probe score를 n_rounds에 걸쳐 반복적으로 적용해 image token pool을 점진적으로 축소.
매 라운드마다 LLaVA full forward를 재실행해 evict된 token이 더 이상
remaining token의 attention에 영향을 주지 않도록 함.

[왜 이 방식인가 — A-option 선택 이유]
처음에는 "같은 score로 n_rounds topk"를 구현했으나 수학적으로 1-shot topk와 동일함.
  이유: probe가 per-token MLP로 cross-token interaction이 없음.
       → 각 token의 score는 다른 token의 존재 여부와 무관.
       → 매 라운드 score가 바뀌지 않으면 sequential topk = 1-shot topk.

A-option: LLaVA forward를 각 라운드마다 재실행.
  evict된 token을 attention_mask=0으로 마스킹 → surviving token의 hidden state가 바뀜
  → probe가 genuinely 다른 hidden state를 보고 score를 재계산.

[핵심 가정]
- hidden_states[0] = merged multimodal embedding (vision feature가 이미 합쳐진 상태)
  → 라운드 2+에서 inputs_embeds로 재사용 가능, vision encoder 재실행 불필요
- probe_model은 per-layer Linear/MLP: probe_model.layers[layer_idx]
- 모든 layer에서 amax로 global consensus → 단일 선택 기준

[이 방식의 한계]
- 라운드당 LLaVA full forward 1회 추가 → 샘플당 n_rounds × prefill 비용
- position_ids가 명시적으로 전달되지 않음 (LLaVA default가 0부터 sequential 생성)
  → evict 후에도 position_ids가 변하지 않아 positional encoding이 원래 위치 기준
  → 이는 generate() 시점과 동일한 조건이므로 허용 가능

[관련 코드]
- evaluate_image_teacher_pruning.py:140 (_iterative_probe_preselect)
- kvpress/presses/image_token_press.py (PreselectedImagePress)
- kvpress/presses/kvzap_press.py (KVzapModel — probe architecture)
```

---

## 2. DECISION 태그 — 주요 결정

```python
# DECISION: A-option — full LLaVA forward re-execution per round
# B-option(cached KV states 업데이트)은 구조상 hook 기반 press와 충돌.
# A-option은 generate() 전 별도 forward이므로 press에 독립적.

# DECISION: hidden_states[0]을 inputs_embeds로 재사용
# LLaVA forward의 hidden_states[0] = embedding layer output = merged multimodal embeddings.
# vision encoder를 n_rounds 반복 실행하면 비용이 너무 큼.
# llava_extractor.py의 _collect_postvision_minimal_sample에서 동일 패턴 확인됨.

# DECISION: global consensus = amax across all probe layers
# per-layer 개별 선택이면 각 layer가 다른 token set을 보게 됨.
# PreselectedImagePress는 모든 layer에 동일한 mask를 적용 → global consensus 필요.
# amax = "어떤 layer든 중요하다고 하면 살린다" → 보수적, 정보 손실 최소화.

# DECISION: linear schedule n_image → n_image_keep over n_rounds
# 한번에 많이 자르면 정보 손실이 크고, 매 라운드 조금씩 자르면 forward 비용 대비 효과 작음.
# linear schedule은 symmetric한 pruning 강도. 다른 schedule(cosine 등)은 ablation 대상.

# DECISION: del out after per-layer scoring
# out.hidden_states = (n_layers+1) tensors, 각 [1, mm_len, D].
# LLaVA-13B 기준 ~32 layers × 700 tokens × 5120 dim ≈ 450MB per round.
# scoring 후 즉시 free하지 않으면 n_rounds × 450MB GPU 메모리 누수.

# DECISION: img_pos_dev[img_pos_dev < mm_len] range guard
# prompt_len_mm = merged embedding 길이. image_positions는 mm-space index.
# 이론상 모든 img_pos < mm_len이어야 하지만 truncation edge case에서
# img_pos가 mm_len 밖에 있을 수 있음 → IndexError 방지.
```

---

## 3. 구현 후 Codex 디버깅 리뷰

### A. Correctness 체크리스트

| 항목 | 결과 | 비고 |
|------|------|------|
| 핵심 알고리즘과 수식 일치 | ✅ | linear schedule 확인, schedule[-1]=n_image_keep 강제 |
| edge case (n_rounds=1) | ✅ | `n_rounds <= 1` 조기 반환 처리 |
| edge case (n_image_keep >= n_image) | ✅ | `n_image_keep >= n_image` 조기 반환 |
| edge case (pool이 k보다 작을 때) | ✅ | `k = min(round_k, pool_cpu.numel())` |
| fail-closed (silent wrong result 없음) | ✅ | RuntimeError 전파, None 반환 없이 예외 |
| tensor shape 가정 명시 | ✅ | docstring + 인라인 주석에 shape 기록 |
| random seed 고정 여부 | N/A | iterative은 deterministic (topk) |

### B. Efficiency 체크리스트

| 항목 | 결과 | 비고 |
|------|------|------|
| 불필요한 forward pass 없음 | ✅ | 라운드당 정확히 1 forward |
| `del out` 즉시 free | ✅ | Bug 1 수정: scoring loop 후 즉시 del |
| `torch.no_grad()` 적용 | ✅ | 모든 forward에 `torch.no_grad()` wrapping |
| probe_layer device 이동 | ⚠️ | `.to(device=hs.device)` 매 layer 호출 — probe가 항상 GPU에 있다면 no-op이지만 매 라운드 반복됨. 실용상 문제 없음 |
| CPU→GPU 텐서 이동 최소화 | ✅ | global_scores는 CPU float (agg = .detach().cpu().float()) |

### C. 발견된 버그 및 수정

#### Bug 1: OOM — `del out` 누락
```
증상: n_rounds > 1에서 GPU 메모리가 라운드마다 증가
원인: out.hidden_states (~450MB/round for LLaVA-13B)가 GC 전까지 GPU에 남음
수정: per_layer_scores 루프 완료 직후 `del out` 추가
```
```python
# BEFORE (buggy)
global_scores = torch.stack(per_layer_scores, dim=0).amax(dim=0)

# AFTER (fixed)
del out  # Free GPU memory: hidden states no longer needed after scoring
global_scores = torch.stack(per_layer_scores, dim=0).amax(dim=0)
```

#### Bug 2: IndexError — image position range guard 누락
```
증상: 간헐적 IndexError: index N is out of bounds for dimension 0 with size M
원인: is_image[img_pos_dev] = True 에서 img_pos_dev에 mm_len 이상의 값이 있을 경우
     (truncation edge case: image_positions가 mm_len보다 큰 mm-space index 포함)
수정: img_pos_dev < mm_len 조건으로 필터링
```
```python
# BEFORE (buggy)
is_image[img_pos_dev] = True

# AFTER (fixed)
is_image[img_pos_dev[img_pos_dev < mm_len]] = True  # guard against out-of-range
```

#### Bug 3: 코드 명확성 — inputs_embeds/pixel_values 의도 미기재
```
증상: 코드 리뷰 시 "pixel_values를 왜 안 넘기는가?" 혼동 가능
원인: LLaVA forward에서 pixel_values=None이면 vision encoder 생략되는데,
     이 동작이 의도적인지 실수인지 불명확
수정: 명시적 주석 추가
```
```python
# AFTER (fixed)
# pixel_values intentionally omitted: merged_embeds already contains vision features.
# input_ids intentionally omitted: inputs_embeds takes precedence in LLaVA forward.
out = model(
    inputs_embeds=merged_embeds,
    attention_mask=attn_mask,
    use_cache=False,
    output_hidden_states=True,
    return_dict=True,
)
```

### D. 알려진 위험 및 향후 확인 필요 사항

```
KNOWN RISK 1: position_ids 일관성
  round 2+에서 position_ids를 명시하지 않음.
  LLaVA는 attention_mask 기반으로 position_ids를 생성하므로
  evict된 위치가 attention_mask=0이면 position_ids 계산에서 제외됨.
  generate() 시점(PreselectedImagePress)에서는 evict된 token이 KV cache에 없으므로
  position encoding mismatch가 발생할 수 있음.
  → 실험 결과로 성능 확인 필요. 문제가 되면 explicit position_ids 주입 검토.

KNOWN RISK 2: 이 함수는 generate() 전 forward이므로 VizCapture와 무관
  iterative 모드에서는 VizCapture의 keep_masks가 _iterative_probe_preselect의
  round 결과를 반영하지 않음.
  Phase 2 메트릭(overlap ratio)은 generate() 시의 PreselectedImagePress 동작 기준으로
  기록되며, 이는 iterative 최종 선택 결과와 일치해야 함 — 추가 검증 필요.

KNOWN RISK 3: probe_layer.eval() 호출
  probe_layer.to(device=...).eval()을 매 라운드 호출.
  probe가 BatchNorm을 가지면 eval() 전환이 중요하지만, 현재 KVzapModel은
  Linear 또는 Linear-GELU-Linear만 사용하므로 eval()은 no-op.
  그러나 모델 변경 시를 대비해 유지하는 것이 안전.
```

---

## 4. Codex 협업 — 디버깅 요청 프롬프트 (기록용)

실제 Codex에게 보낸 디버깅 요청:

```
아래 코드를 리뷰해줘.

맥락:
- 목적: LLaVA full forward를 n_rounds 재실행해 image token pool을 점진적으로 축소
- 핵심 가정: hidden_states[0] = merged embeddings, probe = per-token MLP (no cross-token)
- 이미 알고 있는 한계: n_rounds × forward cost, position_ids 비명시

코드: _iterative_probe_preselect (evaluate_image_teacher_pruning.py:140-236)

리뷰 포인트:
1. GPU 메모리 누수 지점
2. IndexError 가능성 (image_positions out of range)
3. LLaVA forward의 inputs_embeds/pixel_values/input_ids 동작 정확성
4. 재현성 문제 (topk tie-breaking)
```

Codex 응답으로 발견된 이슈:
- Bug 1 (del out 누락) — 메모리 누수 경고
- Bug 2 (range guard 누락) — IndexError 가능성
- Bug 3 (의도 불명확 주석) — 명확성 개선 권고

---

## 5. 파일별 변경 요약

| 파일 | 변경 유형 | 핵심 내용 |
|------|----------|----------|
| `kvzap/phase2_metrics.py` | 신규 생성 | oracle overlap ratio, spatial entropy 계산 |
| `kvpress/presses/image_token_press.py` | 수정 | `n_iterative_rounds` 필드, `PreselectedImagePress` 추가 |
| `evaluate_image_teacher_pruning.py` | 수정 | `_iterative_probe_preselect`, `--n_iterative_rounds`, Phase 2 수집 |

### PreselectedImagePress 설계 요점

```
목적: _iterative_probe_preselect가 결정한 image position을 
      generate() 시점에 그대로 적용하는 pass-through press.

핵심: evict_mask 기반으로 KV cache에서 evict된 token 제거.
      ProbeImageTeacherPress와 달리 per-layer re-scoring 없음.
      모든 layer에 동일한 global mask 적용 (global consensus 결과).

인터페이스:
  set_selection(all_image_positions, keep_positions)
  compress(module, hidden_states, keys, values, attentions, kwargs)
  clear_sample_context()
```

---

## 6. 실험 실행 커맨드 (예시)

```bash
# n_iterative_rounds=4, Phase 2 메트릭 수집 포함
python evaluate_image_teacher_pruning.py \
  --mode probe \
  --teacher_type att_only_postvision \
  --total_keep_ratio 0.5 \
  --n_iterative_rounds 4 \
  --collect_phase2_metrics \
  --output_dir experiments/EXP-20260417-001/results \
  [기타 데이터셋/모델 인자]

# n_iterative_rounds=1 (one-shot 베이스라인)
python evaluate_image_teacher_pruning.py \
  --mode probe \
  --teacher_type att_only_postvision \
  --total_keep_ratio 0.5 \
  --n_iterative_rounds 1 \
  --output_dir experiments/EXP-20260417-001/results_baseline \
  [기타 데이터셋/모델 인자]
```

---

## 7. 관련 실험 및 참고

- EXP-20260412-003: `n_initial_keep`, `n_recent_keep`, `n_random_keep` 구현 (forced-keep positional)
- PLAN.md: 본 실험 전체 계획 및 가설
- `kvpress/presses/kvzap_press.py`: KVzapModel 구조 (no cross-token interaction 확인처)
- `kvzap/llava_extractor.py`: `hidden_states[0]` = merged embeddings 확인처 (`_collect_postvision_minimal_sample`)
