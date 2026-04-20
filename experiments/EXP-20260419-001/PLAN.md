# Experiment Plan

**ID:** EXP-20260419-001
**Author:** ssoree912
**Date:** 2026-04-19
**Status:** [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned

---

## 1. 동기 (Motivation)

zap 코드베이스에는 현재 **PPL eval 이 존재하지 않는다** (MileBench exact-match / LOOK-M metric 만 있음). PrefixKV 논문이 주 metric 으로 쓰는 PPL 을 zap 에서도 계산해서 PrefixKV 표와 **같은 축에서 나란히 비교**할 수 있게 한다.

베이스 모델은 둘 다 LLaVA-1.5-7B 로 같고 (PrefixKV: liuhaotian weight, zap: HF re-wrap된 같은 weight), tokenizer 도 같은 Llama tokenizer 이므로, **PrefixKV eval_ppl.py 의 세팅 (conv template, answer tokenization, per-token loop) 을 그대로 재현하면 full-cache PPL 수치가 PrefixKV 수치와 일치해야 한다**. 그 위에서 zap 의 image-only press (oracle / probe / h2o_image_only) 를 붙여 PPL 을 측정하고 PrefixKV 표와 나란히 제시한다.

---

## 2. 가설 (Hypothesis)

1. **Full-cache PPL 일치**: PrefixKV 세팅 (conv template, per-token teacher-forcing loop, answer tokenization) 을 그대로 재현하면 zap 의 full-cache PPL 은 PrefixKV 논문의 full-cache PPL 과 **소수점 오차 수준** 으로 일치한다.
2. **Hybrid ≥ PostVision, Future**: Future decode 신호와 PostVision 신호는 서로 보완적이라 같은 keep ratio 에서 hybrid 의 PPL 이 둘 각각보다 낮거나 같다 (EXP-20260418-001 의 학습단 결과와 일관).
3. **낮은 keep ratio 에서 Future 가 PostVision 보다 유리**: keep ratio 0.2 · 0.4 구간에서는 decode 에서 실제 재사용되는 토큰만 살아남는 게 유리해 Future distill 이 PostVision distill 보다 PPL 이 낮을 가능성이 높다.

### Success target
- Full-cache PPL 이 PrefixKV 논문의 full-cache 값과 `±5%` 이내 (세팅을 제대로 재현했다는 증거).
- 세 방법 (probe / future / hybrid) 모두 keep ratio 0.2 · 0.4 · 0.6 · 0.8 sweep 에서 full-cache 대비 PPL 상승이 monotonic.
- hybrid 가 두 데이터셋 × 4 ratio = 8 조합 중 6 이상에서 probe, future 각각보다 PPL 이 낮거나 같다.

### Minimum viability
- Full-cache run 이 두 데이터셋 모두 성공하고 PrefixKV 수치와 ±10% 이내.
- 세 방법 모두 keep_ratio 0.4 이상에서 full-cache 대비 PPL 상승폭이 20% 이내.

---

## 3. 독립변수 (What we change)

| 변수 | 값 |
|---|---|
| `dataset` | `llava_description` (detail_1k), `mm_vet` |
| `method` | `full_cache` (sanity), `probe` (PostVision distill), `future` (Future distill), `hybrid` |
| `total_keep_ratio` | `1.0` (full), `0.8`, `0.6`, `0.4`, `0.2` |
| `probe_model_name` | `ckpts/postvision_probe_v4_20ep` (PostVision distill; probe/hybrid) |
| `future_probe_name` | `ckpts/future_probe_v4_last8_20ep_bcast31` (Future distill; future/hybrid) |
| `selected_layer_indices` | `24 25 26 27 28 29 30 31` (Future distill 가 학습된 last-8 레이어; future/hybrid) |
| `alpha` | `0.5` (hybrid 의 PostVision / Future 가중; 1 차는 기본값 고정, 필요 시 sweep) |

---

## 4. 종속변수 (What we measure)

- **주요 metric**: `ppl = exp(mean(NLL over answer tokens))` — 샘플별 answer 토큰 NLL 을 모두 모아 평균한 뒤 지수.
- **보조 metric**:
  - sample-wise `ppl_i`의 중앙값, p95 (outlier 확인용)
  - per-sample `answer_len_text`, `n_image_tokens`
  - run-time (초/샘플)
  - n_failures

로그 파일 경로: `experiments/EXP-20260419-001/outputs/{dataset}/{method}_r{ratio}.json`

---

## 5. 고정 조건 (What stays the same)

- **Model**: `/workspace/zap/ckpts/llava-1.5-7b-hf` (HF LlavaForConditionalGeneration).
- **Processor**: `AutoProcessor.from_pretrained(..., use_fast=False)` + `configure_llava_processor()`.
- **attn_implementation**: `eager` (press 가 dense attention 을 요구).
- **dtype**: `torch.float16` (GPU) — PrefixKV 도 bfloat16, 근사적으로 같은 자릿수.
- **Prompt template**: PrefixKV eval_ppl.py 와 **동일하게** `llava_v1` conv template 을 재현한다. 즉 system prefix `"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."` + `"USER: <image>\n{question} ASSISTANT:"` 형태를 만들고 HF tokenizer 로 그대로 인코딩 (zap 의 default `USER:/ASSISTANT:` 축약형은 쓰지 않음).
- **Answer tokenization**: `tokenizer(answer, add_special_tokens=False)` 로 BOS 제거 (PrefixKV 도 `[:, 1:]` 로 BOS 스트립).
- **Forward loop**: PrefixKV eval_ppl.py 와 동일한 **per-token teacher-forcing loop** (prefill 1회 + decode step 별 loss 누적). 이게 PrefixKV 의 decode-time KV 압축을 붙이기 위한 원래 이유였지만, 우리는 prefill-only press 라 single teacher-forced forward 와 수학적으로 동치 — 그래도 숫자 재현성 확인용으로 같은 loop 을 쓴다.
- **Seed**: PrefixKV 와 동일하게 `set_seed(0)` + `random.shuffle(data)` 후 `data[:eval_samples]` (PrefixKV 는 shuffle 이후에 slice 하는 것으로 보이지만 실제로는 `data[:eval_samples]` 먼저 슬라이스 후 shuffle). eval_ppl.py 코드의 순서를 그대로 따른다.
- **Eval samples**: LLaVA-Description 1000, MM-Vet 218 (PrefixKV 값 그대로).
- **Hardware**: 단일 GPU (A6000 / A100 급).

---

## 6. 베이스라인

- **PrefixKV eval_ppl.py** 수치 (논문 figure) — 논문 보고치를 기준점으로 놓지만 직접 대입은 caveat.
- **zap full cache** — 본 실험의 internal upper bound.
- **zap h2o_image_only** — probe 와 oracle 대비 naive image-only baseline.
- **zap oracle** (att_only_postvision teacher) — image-only pruning 의 실질 상한.

---

## 7. 예상 결과

- Full cache:
  - LLaVA-Description PPL ≈ 2.x ~ 3.x
  - MM-Vet PPL ≈ 4.x ~ 5.x (PrefixKV 논문 그림과 같은 스케일)
- keep_ratio ↓ 에 따라 PPL ↑ 이 monotonic. oracle / probe 는 h2o 대비 상승폭이 작다.
- keep_ratio 0.2 구간에서 PrefixKV 와 zap 간 상대적 ranking 이 바뀔 수 있음 (global KV 압축이 image-only 보다 유리한 구간).

---

## 8. 판단 기준 (Success Criteria)

- (a) 두 데이터셋 × 4 method × 5 ratio 조합이 모두 성공적으로 저장됨.
- (b) full-cache PPL 이 PrefixKV 논문 full-cache 값과 `±30%` 이내.
- (c) 같은 ratio 에서 oracle PPL ≤ probe PPL ≤ h2o PPL 이 8/10 조합 이상에서 성립.

---

## 9. 이 실험으로 증명할 수 없는 것 (Caveats)

- **압축 범위 차이는 “방법 비교”이지 측정 오류가 아님**: PrefixKV 는 layer-adaptive global prefix 압축 (text + image) 을 decode step 마다 재적용, zap press 는 image-only prefill pruning 을 한 번만 적용. 같은 total keep ratio 라도 실제 제거되는 토큰 집합과 시점이 다르다. PPL 차이가 나면 그건 **알고리즘 차이 그 자체**이고, 본 실험의 비교 대상이다.
- **Probe 학습 데이터 도메인 차이**: ScienceQA / DocVQA 로 학습된 probe 가 LLaVA-Description / MM-Vet 에서 generalize 하는지는 별개 이슈. 일부 downgrade 가 관찰되면 probe 의 domain gap 탓일 수 있다 (EXP-20260419-003 으로 후속).
- **HF ↔ 원본 LLaVA 수치 오차**: 같은 weight 라도 HF LlavaForConditionalGeneration 의 forward 경로와 원본 `llava.model` 경로는 fused kernel / image-token expansion 순서 등에서 미세 차이가 있을 수 있다. full-cache 가 ±5% 이내면 통과로 간주.

---

## 10. 예상 런타임 / 리소스

- GPU: A6000 1장 (48GB) 기준.
- 예상 시간:
  - LLaVA-Description (1000 샘플) full cache: ~15 분
  - MM-Vet (218 샘플) full cache: ~4 분
  - 4 method × 5 ratio × 2 dataset ≈ 40 run → 약 6~8 시간.
- 디스크: JSON 로그 + per-sample NLL tensor → `<500 MB`.

---

## 11. 데이터

### 11.1 LLaVA-Description (detail_1k)

- **다운로드**: [Google Drive (PrefixKV README 링크)](https://drive.google.com/file/d/1_I2sokdpv8hLzLUe8UUmFvihbh6Kmytv/view)
- **설치 경로**: `/workspace/zap/data/prefixkv/llava_description/detail_1k.json`
- **이미지**: COCO `train2017`, 경로 `/workspace/zap/data/coco/train2017/` (이미 없는 경우 `cocodataset.org` 에서 받아 untar).
- **스키마** (PrefixKV eval_ppl.py 기준):
  ```json
  {
    "id": "...",
    "image": "coco/000000XXXXXX.jpg",  // 또는 그냥 "000000XXXXXX.jpg"
    "conversations": [
      {"from": "human", "value": "<image>\n<question>"},
      {"from": "gpt",   "value": "<reference caption>"}
    ]
  }
  ```
- **Q / A 추출 규칙**:
  - question = `item["conversations"][0]["value"]` (반드시 `<image>` 포함, 우리가 템플릿에서 중복 삽입하지 않도록 `build_prompt` 의 placeholder normalize 에 맡김).
  - answer   = `item["conversations"][1]["value"]` (teacher forcing label).
- **이미지 해상도**: `image_aspect_ratio="pad"` (PrefixKV 설정) — HF processor 는 기본 center-crop/resize. 본 실험은 HF processor 기본값을 따른다 (caveat 9 에 이미 반영).

### 11.2 MM-Vet

- **다운로드**: [Google Drive (PrefixKV README 링크)](https://drive.google.com/file/d/1MLB7Pr_zo2Nu5iihuXRXE38nHzY-TnRN/view)
- **설치 경로**:
  - JSON: `/workspace/zap/data/prefixkv/mm-vet/mm-vet.json`
  - 이미지: `/workspace/zap/data/prefixkv/mm-vet/images/`
- **스키마** (PrefixKV eval_ppl.py 의 `"mm-vet" in args.data_path` 분기 기준):
  ```json
  {
    "question_id": "...",
    "image": "v1_0.png",
    "question": "<question text>",
    "answer": "<ground truth>"
  }
  ```
- **Q / A 추출 규칙**:
  - question = `item["question"] + "\n<image>"` → HF 쪽은 `build_prompt` 가 `<image>` 를 문두에 주입하므로 우리는 **뒤에 붙이지 않고** 그냥 `item["question"]` 만 question 으로 쓴다.
  - answer   = `item["answer"]`.

### 11.3 사이즈

- LLaVA-Description: 총 1000 샘플, 한 샘플당 긴 description (~수백 토큰). Answer 가 길어 PPL 분산 큼.
- MM-Vet: 218 샘플, 단답 위주. PPL 이 더 뾰족 (outlier 민감).

---

## 12. 코드 구현 계획

### 12.1 새 파일

**`/workspace/zap/eval_ppl_prefixkv.py`** — PrefixKV 데이터로 PPL 을 재는 신규 엔트리포인트.

핵심 설계 원칙:
- zap 의 기존 helper 재사용: `AutoProcessor`, `LlavaForConditionalGeneration`, `configure_llava_processor`, `infer_llava_image_positions_no_forward`, `build_press`.
- **PrefixKV eval_ppl.py 의 per-token loop 을 그대로 포팅** — prefill forward 1회, 그 다음 answer token 개수만큼 1-token forward 루프. 각 step 의 logits 에 대해 CE loss (reduction='none') 를 answer token 기준으로 쌓아 평균 후 exp.
- Conv template 은 PrefixKV 와 동일한 `llava_v1` 포맷 (system prefix 포함) 을 문자열로 직접 생성 — HF processor 는 pixel_values 용도로만 쓰고, input_ids 는 `tokenizer(prompt_string)` 로 만든다.
- dataset loader 는 스크립트 내부에 `_load_llava_description`, `_load_mmvet` 두 함수로 둔다.

핵심 루프 (PrefixKV eval_ppl.py 의 per-token loop 을 HF LLaVA 로 포팅):

```python
LLAVA_V1_SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

def build_llava_v1_prompt(question: str) -> str:
    # PrefixKV eval_ppl.py 의 conv_templates["llava_v1"] 와 동일한 문자열.
    # 질문에 <image> 가 없으면 앞에 붙인다.
    if "<image>" not in question:
        question = "<image>\n" + question
    return f"{LLAVA_V1_SYSTEM} USER: {question} ASSISTANT:"


def compute_sample_ppl(model, processor, press, sample, device, float_dtype):
    prompt_text = build_llava_v1_prompt(sample["question"])
    images      = open_images(sample["image_paths"])

    # pixel_values 만 processor 로 얻고, input_ids 는 tokenizer 로 직접
    vis = processor.image_processor(images=images, return_tensors="pt")
    prompt_ids = processor.tokenizer(prompt_text, return_tensors="pt").input_ids
    answer_ids = processor.tokenizer(
        sample["answer"], return_tensors="pt", add_special_tokens=False,
    ).input_ids

    prompt_inputs = {
        "input_ids": prompt_ids.to(device),
        "attention_mask": torch.ones_like(prompt_ids).to(device),
        "pixel_values": vis["pixel_values"].to(device=device, dtype=float_dtype),
    }
    answer_ids = answer_ids.to(device)

    # press 구성 (evaluate_image_teacher_pruning.build_press 재사용)
    if press is not None:
        image_positions, _ = infer_llava_image_positions_no_forward(
            prompt_inputs=prompt_inputs, model_config=model.config, num_images=1,
        )
        if hasattr(press, "set_sample_teacher"):
            # oracle_onthefly: teacher score on-the-fly
            t = extract_att_only_postvision_on_the_fly(
                model=model, prompt_inputs=prompt_inputs,
                image_positions=image_positions, device=device,
            )
            press.set_sample_teacher(image_positions, t)
        else:
            press.set_image_positions(image_positions)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    nlls = []
    past_key_values = None

    ctx = press(model) if press is not None else contextlib.nullcontext()
    with ctx, torch.no_grad():
        # prefill
        out = model(**prompt_inputs, use_cache=True, return_dict=True)
        past_key_values = out.past_key_values
        prefill_last_logits = out.logits[0, -1, :]  # (vocab,)

        # first answer token NLL from prefill_last_logits
        nlls.append(loss_fn(
            prefill_last_logits.unsqueeze(0), answer_ids[0, 0:1],
        ).cpu())

        # decode step (teacher forcing)
        for idx in range(1, answer_ids.shape[1]):
            step_input = answer_ids[:, idx - 1 : idx]
            step_out = model(
                input_ids=step_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = step_out.past_key_values
            step_logits = step_out.logits[0, -1, :]
            nlls.append(loss_fn(
                step_logits.unsqueeze(0), answer_ids[0, idx : idx + 1],
            ).cpu())

    return torch.cat(nlls), int(answer_ids.shape[1])


def main(...):
    per_token_nlls = []
    for s in samples:
        nll, n = compute_sample_ppl(model, processor, press, s, device, dtype)
        per_token_nlls.append(nll)
    all_nll = torch.cat(per_token_nlls)
    ppl = torch.exp(all_nll.mean())
```

주의:
- zap press 는 `press(model)` context manager 진입 시 attention hook 을 건다. prefill 단계에서 image-only token 이 물리적으로 제거되는 게 아니라 kv cache 에서 지워진 형태로 저장되므로, `past_key_values` 를 이어받는 decode step 에서도 이미 제거된 상태가 유지됨 → per-token loop 에서도 정상 동작.
- HF LLaVA 에서 `pixel_values` + `input_ids` 를 함께 넣으면 내부에서 `<image>` placeholder 가 merged-space 로 확장된다. `prompt_len_mm` 은 `out.logits.shape[1]` 로 직접 읽으면 됨 (별도 util 필요 없음).
- Press 구성은 `evaluate_image_teacher_pruning.build_press(args)` 를 그대로 재사용. `full_cache` 모드는 `press=None` 으로 분기.

### 12.2 CLI args (최소)

```
--dataset {llava_description,mm_vet}
--data_path
--image_root
--mode {full_cache,h2o_image_only,oracle,probe}
--total_keep_ratio (default None → full_cache 는 무시)
--teacher_dir     (oracle 전용)
--probe_model_name (probe 전용)
--implementation_model_name /workspace/zap/ckpts/llava-1.5-7b-hf
--attn_implementation eager
--torch_dtype float16
--device cuda:0
--max_samples (default 1000 / 218 per dataset)
--output_dir experiments/EXP-20260419-001/outputs/{dataset}/{tag}
--seed 0
```

### 12.3 출력 포맷

`outputs/{dataset}/{method}_r{ratio}.json`:

```json
{
  "dataset": "llava_description",
  "method": "probe",
  "total_keep_ratio": 0.4,
  "n_samples": 1000,
  "n_failures": 0,
  "ppl": 3.142,
  "ppl_median_sample": 2.95,
  "ppl_p95_sample": 6.11,
  "total_answer_tokens": 83241,
  "runtime_sec": 812.3,
  "probe_model_name": ".../postvision_probe_v4_20ep",
  "prompt_template": "USER: <image>\n{question}\nASSISTANT:",
  "seed": 0
}
```

또한 `per_sample.jsonl` 에 샘플별 `{sample_id, answer_len, mean_nll, ppl}` 기록.

### 12.4 Sweep 스크립트

**`/workspace/zap/experiments/EXP-20260419-001/run.sh`**:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/workspace/zap/data/prefixkv
CKPT=/workspace/zap/ckpts/llava-1.5-7b-hf
OUT=/workspace/zap/experiments/EXP-20260419-001/outputs
PROBE=/workspace/zap/ckpts/postvision_probe_v4_20ep
TEACHER_DIR=/workspace/zap/artifacts/prefixkv_teacher  # teacher를 별도로 구축해야 함 (§12.5)

for DATASET in llava_description mm_vet; do
  if [[ "$DATASET" == "llava_description" ]]; then
    DATA="$DATA_ROOT/llava_description/detail_1k.json"
    IMG="/workspace/zap/data/coco"
    N=1000
  else
    DATA="$DATA_ROOT/mm-vet/mm-vet.json"
    IMG="$DATA_ROOT/mm-vet/images"
    N=218
  fi

  # Full cache
  python eval_ppl_prefixkv.py --dataset "$DATASET" --data_path "$DATA" --image_root "$IMG" \
    --mode full_cache --max_samples "$N" \
    --output_dir "$OUT/$DATASET" --exp_tag "full_cache"

  for R in 0.8 0.6 0.4 0.2; do
    python eval_ppl_prefixkv.py --dataset "$DATASET" --data_path "$DATA" --image_root "$IMG" \
      --mode h2o_image_only --total_keep_ratio "$R" --max_samples "$N" \
      --output_dir "$OUT/$DATASET" --exp_tag "h2o_r${R}"

    python eval_ppl_prefixkv.py --dataset "$DATASET" --data_path "$DATA" --image_root "$IMG" \
      --mode probe --probe_model_name "$PROBE" --total_keep_ratio "$R" --max_samples "$N" \
      --output_dir "$OUT/$DATASET" --exp_tag "probe_r${R}"

    python eval_ppl_prefixkv.py --dataset "$DATASET" --data_path "$DATA" --image_root "$IMG" \
      --mode oracle --teacher_dir "$TEACHER_DIR/$DATASET" --total_keep_ratio "$R" --max_samples "$N" \
      --output_dir "$OUT/$DATASET" --exp_tag "oracle_r${R}"
  done
done
```

### 12.5 Oracle 모드의 teacher record 생성

Oracle 을 돌리려면 각 샘플에 대해 `att_only_postvision` teacher score 가 필요하다. 두 가지 옵션:
- (A) `oracle_onthefly` mode 를 쓰면 teacher forward 가 실시간으로 돌아 사전 생성 불필요. 하지만 forward 2번 (teacher + PPL) → 2배 비용.
- (B) `build_docvqa_teacher4.py` 를 PrefixKV 데이터셋 로더에 맞게 소규모 확장해 `artifacts/prefixkv_teacher/{dataset}/{sample_id}.pt` 를 사전 생성.

**결정**: 1차 실험은 (A) `oracle_onthefly` 로 진행해 코드 변경을 최소화한다. MM-Vet 218 + LLaVA-Desc 1000 = 1218 샘플이므로 on-the-fly 2-pass 비용 허용 가능.

### 12.6 최소 변경 / 기존 코드 재사용 요약

| 새로 추가 | 재사용 |
|---|---|
| `eval_ppl_prefixkv.py` (엔트리포인트) | `kvzap/image_teacher_utils.build_prompt` |
| `experiments/EXP-20260419-001/run.sh` | `kvzap/llava_extractor.configure_llava_processor`, `_move_batch_to_device`, `_get_model_device`, `_get_model_float_dtype`, `infer_llava_image_positions_no_forward` |
| `experiments/EXP-20260419-001/summarize.py` (PPL CSV merge) | `evaluate_image_teacher_pruning.build_press`, `extract_att_only_postvision_on_the_fly` |
| (optional) PrefixKV dataset loader in script-local scope | `kvpress.presses.image_token_press.*` press classes |

### 12.7 수치 검증 체크리스트 (개발 중)

- [ ] `answer_ids` 앞에 BOS 가 붙지 않았는지 확인 (`add_special_tokens=False`).
- [ ] merged-space 의 `prompt_len_mm` 이 실제 model forward 의 `hidden_states.shape[1] - answer_len` 과 일치하는지 1 샘플에서 assert.
- [ ] Full cache 모드에서 NLL 이 HF 공식 `model(labels=...)` 결과와 같은 자릿수인지 1 샘플 cross-check.
- [ ] LLaVA-Description 의 `conversations[0]["value"]` 에 `<image>` 가 이미 있을 때 `build_prompt` 가 중복 삽입하지 않는지 확인 (`_inject_image_tokens` normalize).
- [ ] MM-Vet 의 image 경로가 잘못 join 되지 않는지 (`image_root` + `item["image"]`).

---

## 13. 이후 실험 제안 (Next Experiments)

- EXP-20260419-002: ROUGE 도 함께 (PrefixKV 와 같은 generate + rouge_score).
- EXP-20260419-003: LLaVA-Description 에 probe v4 가 generalize 하는가 (domain gap sanity).
- EXP-20260419-004: 같은 total_keep_ratio 에서 zap image-only press vs LOOK-M 전역 압축 PPL 비교.

---

## 14. 재현 커맨드 (Preview)

```bash
cd /workspace/zap
bash experiments/EXP-20260419-001/run.sh
python experiments/EXP-20260419-001/summarize.py \
    --input_dir experiments/EXP-20260419-001/outputs \
    --out_csv  experiments/EXP-20260419-001/outputs/summary.csv
```
