# Experiment Plan

**ID:** EXP-20260420-003
**Author:** ssoree912
**Date:** 2026-04-20
**Status:** [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned
**Parent:** EXP-20260420-002 (H2O-prefill vs Hybrid on MileBench)
**Purpose tag:** PPL cross-benchmark — Future probe on PrefixKV protocol (LLaVA-Description / MM-Vet)

---

## 1. Motivation

### 1.1 Gap in the current eval surface

우리는 지금까지 Future probe 성능을 **MileBench (task accuracy / ROUGE-L)** 로만 측정해 왔습니다.
PrefixKV 논문은 **LLaVA-Description (`detail_1k.json`, 1000 samples) + MM-Vet (218 samples)** 에서
**teacher-forcing PPL** 로 KV cache 압축 방법들을 비교합니다 (ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}).

→ 두 가지 이유로 PPL cross-eval 이 필요합니다:

1. **논문 비교 가능성**: PrefixKV 뿐 아니라 FastV, H2O, LOOK-M 등이 같은 프로토콜을 사용.
   Future probe 를 이 dataset 위에 올리면 **기존 KV eviction 문헌과 직접 비교** 가능.
2. **Downstream-agnostic 신호 품질**: task accuracy 는 generation/decoding 에 의존. PPL 은
   순수 "보존된 KV 가 다음 token 분포를 얼마나 잘 재현하는가" 를 측정 — image-token eviction
   의 근본적 손실을 가장 엄격하게 드러냄.

### 1.2 왜 detail_1k / mm-vet 인가

- **detail_1k**: COCO train2017 이미지에 대한 long-form caption (평균 ~150 token answer).
  긴 decode sequence → image token 재참조가 많아 **eviction 난이도 높음**.
- **mm-vet**: 218 questions, multi-capability (OCR/math/spatial). Short-form answer.
  PrefixKV figure 의 primary benchmark — 직접 수 대비 가능.

---

## 2. Hypothesis

**H1 (낮은 ratio 에서 Future 우위)**: ratio ∈ {0.1, 0.2, 0.3} 에서 Future probe 의 PPL 증가량
`ΔPPL = PPL_method - PPL_full` 이 H2O-prefill 보다 **작거나 같다** (detail_1k 기준).
- 이유: long-answer generation 은 decode 단계 image-token reuse 가 중요 → "미래 utility"
  신호가 prefill attention 보다 우위일 공산.

**H2 (mm-vet 에서 PrefixKV 수치와 근접)**: Future probe @ ratio=0.5 의 MM-Vet PPL 이
PrefixKV 논문 reported 수치와 **동일 order (Δ ≤ 0.5)** 내에 위치.
- 이유: Sanity — eviction 이 제대로 동작하면 full cache 대비 현저히 나빠지지 않아야 함.

**H3 (full-cache sanity)**: `--method full` 의 PPL 이 각 dataset 에서 PrefixKV 논문 reported
full-cache 수치와 ≤ 0.1 차이.
- 이유: 구현 정합성 검증 (prompt template, tokenization, teacher-forcing 알고리즘).

---

## 3. Independent Variables

| Variable | Values | Notes |
|---|---|---|
| Method | `full`, `future`, `h2o_image_only` | h2o 는 prefill-observable baseline (Paper Table 1 과 동일 축) |
| Image keep ratio (image-only scope) | 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 | PrefixKV `--ratio` grid |
| Dataset | `detail_1k.json` (1000), `mm-vet.json` (218) | `/workspace/data/` |

- Probe: `/workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31` (last-8 layers, indices 24..31).
- Model: `/workspace/zap/ckpts/llava-1.5-7b-hf` (HF LLaVA-1.5-7B).
- Baseline `full`: press 없음, dense KV.

**Note on keep-ratio semantics**: PrefixKV `--ratio r` = 전체 KV의 r keep (all-token).
우리 press 들은 image-only scope (text/system token 은 전부 keep). 기존 zap 실험과 일관성
유지 차원에서 `--image-keep-ratio` 로 default. 공정 비교를 위한 total-scope 재실행은 §14.

**Note on H2O single-shot vs recurrent**: PrefixKV `eval_ppl.py` 는 decode 의 매 step 마다
`kv_cache(past_key_values, num_of_token, attentions)` 를 재호출 → H2O score 가 decode 전반에
걸쳐 **재갱신**. 우리 kvpress `forward_hook` 은 `cache_position[-1] > q_len` guard 때문에
**prefill 에서 1 회만** fire. 본 실험의 `h2o_image_only` 는 따라서 **"prefill-only H2O"**
변형이며 PrefixKV 의 H2O 수치와 직접 비교되지 않음 (EXP-20260420-002 와 동일 setting).
Future probe 는 설계상 prefill-only 이므로 영향 없음. PrefixKV-recurrent H2O 재현이 필요하면
§14 에 별도 실험으로 기재.

---

## 4. Dependent Variables

### Primary
- **PPL** = `exp(mean_over_tokens( NLL(answer_tok_i | prompt, answer_tok_<i>) ))`
- teacher-forcing, PrefixKV `eval_ppl.py` 와 동일 공식 (`CrossEntropyLoss(reduction="none")`
  per token → exp(mean)).

### Secondary
- Δ vs full: `PPL_method - PPL_full` per (dataset, ratio).
- Wall-clock time / sample (효율성 참고용).
- Answer length distribution (length-bias 점검용).

---

## 5. Fixed Conditions

- **Prompt template**: `USER: <image>\n{question}\nASSISTANT:`
  (PrefixKV `conv_templates["llava_v1"]` 결과와 일치.)
- **Attn implementation**:
  - `future`, `full` → SDPA (attention weights 불필요).
  - `h2o_image_only` → `eager` (press 가 raw attention 을 읽음).
- **Dtype**: bfloat16 (model), float32 (probe softmax 내부).
- **Device**: `cuda:0`, dataset 별 순차.
- **Seed**: `torch.manual_seed(0)`, 데이터 순서대로 (PrefixKV 의 `random.shuffle` 은 재현성
  해침 → 제거).
- **Eval samples**:
  - `detail_1k.json`: 1000 (전체)
  - `mm-vet.json`: 218 (전체)

---

## 6. Baselines & Comparisons

### 우리 신규 실행 (본 실험)

| Method | Dataset | Ratios | Runs |
|---|---|---|---|
| full | detail_1k, mm-vet | (N/A) | 2 |
| future | detail_1k, mm-vet | 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 | 12 |
| h2o_image_only | detail_1k, mm-vet | 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 | 12 |

**Total: 26 runs**.

### 외부 reference (PrefixKV 논문 수치)

- PrefixKV, FastV, H2O, StreamingLLM 수치는 논문 본문/figure 에서 옮겨 Table A 에 주석으로
  포함. 재실행 아님.

---

## 7. Expected Results

### 정량 예측 (PPL, 낮을수록 좋음)

**MM-Vet (short-answer, easy)**

| Method \ ratio | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | full |
|---|---|---|---|---|---|---|---|
| full | — | — | — | — | — | — | ~4.5 |
| future | ~5.5 | ~5.0 | ~4.8 | ~4.6 | ~4.5 | ~4.5 | — |
| h2o | ~6.0 | ~5.3 | ~4.9 | ~4.6 | ~4.5 | ~4.5 | — |

**LLaVA-Description (long-answer, hard)**

| Method \ ratio | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | full |
|---|---|---|---|---|---|---|---|
| full | — | — | — | — | — | — | ~3.5 |
| future | ~5.5 | ~4.5 | ~4.0 | ~3.7 | ~3.5 | ~3.5 | — |
| h2o | ~7.0 | ~5.2 | ~4.3 | ~3.8 | ~3.5 | ~3.5 | — |

→ long-answer 에서 future vs h2o 격차가 더 크게 벌어질 것으로 기대 (H1 근거).

### 정성 예측
- ratio ≥ 0.7: 차이 거의 없음 (image KV 대부분 보존).
- ratio ≤ 0.3: 방법 간 차이 가장 명확. future > h2o 우위 기대.
- ratio = 0.1: 양 방법 모두 붕괴 가능성. "아예 eviction 이 불가능한 극단" 체크.

---

## 8. Success Criteria

### Primary
- **H1 지지**: ratio ∈ {0.1, 0.2, 0.3} 중 **최소 2 개** 에서 `ΔPPL_future ≤ ΔPPL_h2o - 0.1`
  (detail_1k 기준).
- **H3 지지**: `|PPL_full_ours − PPL_full_paper| ≤ 0.1` (두 dataset 모두).

### Secondary
- future @ ratio=0.5 ΔPPL ≤ 0.3 (mm-vet), ≤ 0.5 (detail_1k).
- ratio=0.9 에서 모든 method ≈ full (sanity).

### Failure Mode
- `full` 수치가 PrefixKV reported 값과 > 0.5 차이: prompt template / tokenization 버그 의심.
  mm-vet 에서 PrefixKV `eval_ppl.py` 를 그대로 돌려 재현 → 우리 코드와 diff 비교.
- future = h2o (모든 ratio Δ < 0.05): image-only eviction 에서 scoring method 효과가 작음.
  total_keep_ratio 재실행 or 13B 로 확장 검토.

---

## 9. What This Experiment Cannot Prove

- **"Future probe 가 모든 multimodal PPL benchmark 에서 우수하다"**: 2 dataset, 단일 모델
  (LLaVA-1.5-7B), image-only scope 전제.
- **PrefixKV 자체와의 직접 재실행 비교**: PrefixKV layer-wise prefix budget 자체 로직은
  재구현 하지 않음 — 논문 수치만 reference.
- **Generation quality**: PPL 은 teacher-forced. ROUGE / accuracy 는 별도 측정 필요.
- **Text-KV 도 evict 하는 setting**: 현재는 image-only. total_keep_ratio 재실행은 §14.

---

## 10. Runtime / Resources

- **Runs**: 26 (2 ds × (1 full + 6 future + 6 h2o))
- **Sample throughput** (bf16, 4090): ~2–3 sample/s (mm-vet), ~0.5–1 sample/s (detail_1k).
- **Run 당 시간**: mm-vet ~2 min, detail_1k ~30 min.
- **총 시간**: `(13 × 2) + (13 × 30) ≈ 7 hours` (단일 GPU), 2 GPU 병렬 시 ~3.5h.
- **디스크**: ~5 MB.
- **출력 경로**: `/workspace/zap/artifacts/EXP-20260420-003/<dataset>/<method>_k{ratio}/result.json`

---

## 11. Implementation Details

### 11.1 신규 스크립트

**`/workspace/zap/eval_ppl.py`** — PrefixKV `eval_ppl.py` 를 zap stack
(HF `LlavaForConditionalGeneration` + kvpress) 으로 포팅.

핵심 변경사항:
- HF `LlavaForConditionalGeneration` + `AutoProcessor` 로드 (기존
  `evaluate_image_teacher_pruning.py` 와 동일 스타일).
- Prompt: `build_prompt(question, "USER: <image>\n{question}\nASSISTANT:", image_count=1)`.
- Image positions: `infer_llava_image_positions_no_forward(prompt_inputs, model.config, 1)`.
- Press:
  - `future` → `FutureSupervisedImagePress(image_keep_ratio=r, probe_model_name=...,
    selected_layer_indices=(24..31))`
  - `h2o_image_only` → `H2OImageOnlyPress(image_keep_ratio=r)` + eager attention
  - `full` → press 없음, context manager bypass
- Prefill 1 step → evict (press hook) → pruned cache 로 decode loop.
- Decode: answer token 1 개씩 forward, `cache_position=torch.tensor([prompt_len + i])`
  명시 전달 (pruned cache 와 RoPE position 정합성 보장).
- NLL: `F.cross_entropy(logits, label, reduction="none")` per answer token, 모든 sample 의
  token-level NLL concat → `PPL = exp(mean)`.

### 11.2 데이터 핸들링

- **detail_1k.json**: list of `{id, image, conversations: [{human, gpt}]}`. `image` 는
  `coco/train2017/<file>.jpg`. `--image-path /workspace/data` → full path
  `/workspace/data/coco/train2017/<file>.jpg`. `question` 에 `<image>` marker 있음 → strip
  후 template 재삽입.
- **mm-vet.json**: list of `{id, image, question, answer, capability}`. `image` 는
  `images/<file>.png` (혹은 `v1_*.png`). `--image-path /workspace/data/mm-vet/images`
  → full path `/workspace/data/mm-vet/images/<file>.png`. question 에 `<image>` 없음 →
  template 에서 추가.

### 11.3 재현 커맨드 예시

```bash
# Full cache sanity (mm-vet)
python /workspace/zap/eval_ppl.py \
  --method full \
  --data-path /workspace/data/mm-vet/mm-vet.json \
  --image-path /workspace/data/mm-vet/images \
  --eval-samples 218 \
  --output-dir /workspace/zap/artifacts/EXP-20260420-003/mmvet/full

# Future probe k=0.2 (detail_1k)
python /workspace/zap/eval_ppl.py \
  --method future \
  --image-keep-ratio 0.2 \
  --future-probe-name /workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31 \
  --selected-layer-indices 24 25 26 27 28 29 30 31 \
  --data-path /workspace/data/detail_1k.json \
  --image-path /workspace/data \
  --eval-samples 1000 \
  --output-dir /workspace/zap/artifacts/EXP-20260420-003/detail1k/future_k0p2

# H2O-prefill k=0.2 (mm-vet)
python /workspace/zap/eval_ppl.py \
  --method h2o_image_only \
  --image-keep-ratio 0.2 \
  --attn-implementation eager \
  --data-path /workspace/data/mm-vet/mm-vet.json \
  --image-path /workspace/data/mm-vet/images \
  --eval-samples 218 \
  --output-dir /workspace/zap/artifacts/EXP-20260420-003/mmvet/h2o_k0p2
```

### 11.4 Sweep script

**`experiments/EXP-20260420-003/run_sweep.sh`**
- 26 runs 를 dataset × method × ratio 로 순회.
- 각 run 의 결과 JSON 을 `artifacts/EXP-20260420-003/<ds>/<method_k{r}>/result.json` 저장.
- 실패 run 은 `failures.log` 에 기록하고 이어서 진행.

### 11.5 분석 스크립트

**`analyze_results.py`** (실험 후 작성)
- 26 run 의 result.json 을 aggregate → Table A (markdown).
- ratio × method 축 line plot PNG.
- PrefixKV 논문 수치 hardcode reference overlay.

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| HF LLaVA cache_position handling (pruned cache + teacher-forcing) 버그 | 모든 run 잘못됨 | 먼저 `full` mode 로 PrefixKV PPL 재현 확인. mm-vet 218, 논문 수치와 ±0.1 이내일 때 계속. |
| `<image>` token 중복/누락 | eval 망가짐 | `_inject_image_tokens` 규칙 로깅, 처음 3 sample prompt 출력해 검수. |
| Future probe `selected_layer_indices` 미지정 | PPL 왜곡 | `--strict-selected` 옵션 (기본 켬), 미지정 시 abort. 기본 24..31 hardcode. |
| detail_1k 긴 answer → OOM | 일부 sample 실패 | `--max-answer-tokens 256` 옵션, default 는 truncate 없이. |
| h2o + eager → OOM on long context | 일부 run 실패 | attention weights layer-by-layer 소비 (EXP-002 패턴). detail_1k 는 이미지 1 장이라 OK 예상. |
| Per-sample press 재생성 overhead | 속도 저하 | 동일 press 재사용, `set_image_positions` 로 sample 별 위치만 갱신. |

---

## 13. Analysis Plan

### 13.1 Main PPL Table

**Table A. PPL on MM-Vet (218) and LLaVA-Description (1000)**

| Dataset | Method | r=0.1 | r=0.2 | r=0.3 | r=0.5 | r=0.7 | r=0.9 | full |
|---|---|---|---|---|---|---|---|---|
| mm-vet | full | — | — | — | — | — | — | ? |
| mm-vet | h2o | | | | | | | |
| mm-vet | future | | | | | | | |
| detail_1k | full | — | — | — | — | — | — | ? |
| detail_1k | h2o | | | | | | | |
| detail_1k | future | | | | | | | |

+ optional row: PrefixKV reported values (논문 figure digitize).

### 13.2 Claim 검증

1. **H1**: ΔPPL(future, r) vs ΔPPL(h2o, r) for r ∈ {0.1,0.2,0.3}.
2. **H2**: mm-vet @ 0.5 절댓값을 논문 수치와 비교.
3. **H3**: full-cache PPL 직접 비교.

### 13.3 Plot

- `ppl_vs_ratio_mmvet.png` / `ppl_vs_ratio_detail1k.png`: x=ratio, y=PPL, lines=method.
  `full` 은 horizontal dashed line. `ylim (3, 8)` 기본.

### 13.4 Secondary (optional)

- Answer length vs ΔPPL correlation.
- Per-sample PPL distribution (outlier).

---

## 14. Next Experiments (조건부)

### 14.1 Phase 7a (H1 지지 시): PrefixKV 재실행 직접 비교
- PrefixKV `eval_ppl.py` 를 동일 환경에서 돌려 wall-clock + PPL 둘 다 비교.

### 14.2 Phase 7b (H1 실패 시): total_keep_ratio 재실행
- `--total_keep_ratio` 로 전체 KV 예산 통일.

### 14.3 Phase 7c: mm-vet capability-별 분석
- `capability` 필드 (ocr, math, spatial) 별 ΔPPL breakdown.

### 14.4 Phase 7d: 13B 모델 재실행
- `llava-v1.5-13b` 체크포인트 추가. scale 일반화 검증.

---

## 15. Checklist

### 실험 시작 전
- [x] `FutureSupervisedImagePress` 구현 확인 (`kvpress/presses/image_token_press.py:1050`)
- [x] `H2OImageOnlyPress` 구현 확인 (`kvpress/presses/image_token_press.py`)
- [x] Future probe ckpt 존재 (`/workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31`)
- [x] detail_1k.json, mm-vet.json 경로 확인 (`/workspace/data/`)
- [ ] `eval_ppl.py` 작성 및 smoke test (`--eval-samples 3`)
- [ ] `run_sweep.sh` 실행 권한 부여
- [ ] `full` mode 로 mm-vet PPL 재현 — PrefixKV 논문 수치와 ±0.1 이내 확인

### 실험 후
- [ ] 26 runs 전부 result.json 생성
- [ ] `analyze_results.py` 로 main table / plot 생성
- [ ] H1 / H2 / H3 verdict 기록
- [ ] RESULT.md 작성
- [ ] `experiments/README.md` 인덱스 업데이트
