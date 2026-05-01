## Pilot Plan — Decode-to-Image Attention Layer-wise Distribution

**ID**: EXP-20260425-002 / Pilot
**Author**: ssoree912
**Date**: 2026-04-25
**Branch**: `4090/cnn_image_scorer`
**Status**: [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned
**Parent PLAN**: `PLAN.md` (3-branch student, image-only)

---

### 1. 동기 (Motivation)

본 실험 (PLAN.md) 의 student 는 단순 MLP 가 아니라 **3-branch (CNN + Question + Raw)
+ ConvNeXt block + fusion** 구조이므로, 32 layer 전부에 대해 별도 student instance 를
학습하면:
- 학습 시간: scorer 파라미터 32× → 단순 MLP 대비 학습 부담 큼
- 메모리: per-layer student state + per-layer hidden state 동시 보관
- 평가 latency: per-layer student forward 32회 누적

따라서 **decode 시점에 image token 이 실제로 attention 받는 layer 만 골라 학습** 하는
것이 합리적이다. 그러나 어느 layer 가 의미 있게 image 를 보는지는 측정해야 안다.

**선행 보고 (FastV, ECCV 2024)**: prefill self-attention 기준 deep layer image attention
은 ~0.21%. 그러나 본 연구의 teacher 는 **decode → image attention** (FastV 의 prefill
attention 과 다른 분포). 두 attention 의 layer-wise 패턴이 일치하는지는 검증되지 않았다.

핵심 질문: **decode 시점 image attention 의 layer-wise 분포는 어떤 모양인가?**
이 답에 따라 PLAN.md 의 `n_modules` 와 학습 layer scope 를 결정한다.

---

### 2. 가설 (Hypothesis)

**P1 (Layer-wise sparsity 존재)**: 32 layer 가 image 에 거는 attention magnitude 는
균일하지 않다. 적어도 한 자릿수 (×10) 차이가 layer 간에 존재한다.

**P2 (Early-mid concentration)**: vision-language fusion 이 일어나는 early-mid layer
(0~15) 가 deep layer (16~31) 보다 image attention magnitude 가 크다. (FastV 의 prefill
관찰과 일관된 방향. 단, 정도는 다를 수 있음.)

**P3 (Selectivity ≠ magnitude)**: top-K concentration 이 높은 layer (선택적으로 일부
patch 만 보는 layer) 와 mean magnitude 가 큰 layer 가 반드시 일치하지 않는다.
"선택적으로 본 layer" 가 student 학습에 더 가치 있을 수 있다.

> 본 pilot 은 **측정만** 한다. 위 3개 가설은 reporting 시 verdict 만 기록.

---

### 3. 작업 범위

| 항목 | 값 |
|------|----|
| 측정 종류 | decode step 별 self_attn output `attentions` → image position slice |
| Sample 수 | 50 (default) — smoke 2 → 본 50 |
| 모델 학습/평가 | **없음** (순수 측정) |
| 예상 시간 | 코드 작성 15분 + smoke 5분 + 본 실행 30분 + 분석 10분 = **~1h** |

---

### 4. 사전 확인 사항

| 항목 | 경로 / 출처 | 확인 결과 |
|------|------------|----------|
| 모델 | `/workspace/zap/ckpts/llava-1.5-7b-hf` | ✅ 존재 (이전 실험 사용 중) |
| Image position helper | `kvzap/llava_extractor.py::infer_llava_image_positions_no_forward` | ✅ 존재 — 재사용 |
| 데이터 (옵션 1) | `/workspace/zap/data/scienceqa` | ✅ `problems.json + images/` |
| 데이터 (옵션 2) | `/workspace/zap/data/textvqa` | ✅ `train/, val/` |
| 데이터 (옵션 3) | `/workspace/zap/data/MileBench/DocVQA` | ✅ 존재 |
| GPU | RTX 4090 24GB 1장 | ✅ |
| 출력 디렉토리 | `/workspace/zap/artifacts/pilot_attention/` | 신규 생성 |

> **데이터셋 선택 confirm 필요**: scienceqa (decode 짧음, T 작음 가능) vs textvqa
> (OCR/text 비중 높음, image attention 강할 가능) vs DocVQA (긴 prompt, OOM 위험).
> 본 PLAN.md 의 학습용 ScienceQA 와 일치시키는 것이 자연스러우므로 **default = scienceqa**.

---

### 5. 측정 방법

#### 5.1 입력
- 50 (image, question) pair, 데이터셋 random sample (seed=0)

#### 5.2 처리 (per sample)

```python
inputs = processor(image, question, return_tensors="pt").to(device)
image_indices = infer_llava_image_positions_no_forward(input_ids, config)  # (576,)

gen_kwargs = dict(
    max_new_tokens=64, do_sample=False,
    temperature=1.0, top_p=1.0, top_k=0, num_beams=1,
    output_attentions=True, return_dict_in_generate=True, use_cache=True,
)
out = model.generate(**inputs, **gen_kwargs)  # eager attention 강제

# out.attentions: tuple(T) of tuple(L) of [B=1, H, Q_t, K_t]
# decode step 1+ 부터는 Q_t = 1
sample_attn = torch.zeros(L, N_I, dtype=torch.float32)  # accumulator
T = len(out.attentions)
for t, step in enumerate(out.attentions):
    for l in range(L):
        a = step[l][0, :, -1, image_indices]  # (H, N_I)
        sample_attn[l] += a.float().mean(dim=0).cpu()
sample_attn /= T
```

#### 5.3 메모리 절약
- 각 sample 별로 layer-wise reduce 즉시 → `[L, N_I] = [32, 576]` fp32 만 저장
- per-step / per-head tensor 는 즉시 폐기, GPU cache empty
- 50 샘플 누적 후 최종 `[N=50, L=32, N_I=576]` ≈ 3.5 MB → 디스크/메모리 문제 없음

#### 5.4 결정성
- greedy + temperature=1.0, top_p=1.0, top_k=0, num_beams=1
- `torch.manual_seed(0)`
- bf16 weight, fp32 attention accumulation (정밀도 보장)
- `attn_implementation="eager"` 명시 (sdpa 는 attention 출력 안 함)

---

### 6. 통계 (per layer, layer 0..31)

| 통계 | 정의 | 의미 |
|------|------|------|
| `mean_magnitude` | `attn[:, l, :].mean()` | layer 평균 image attention 강도 |
| `max_magnitude` | `attn[:, l, :].max()` | 가장 집중하는 patch 강도 |
| `top50_concentration` | top-50 patch attention 합 / 전체 합 | 선택성 (1.0 에 가까울수록 선택적) |
| `cross_sample_variance` | sample 간 attention vector variance 평균 | 0 에 가까울수록 sample-invariant (= question conditioning 약함) |
| `entropy` | `-Σ p log p` (image position 분포 정규화 후) | 추가 선택성 지표 |

---

### 7. 출력 파일

```
/workspace/zap/pilot_attention_analysis.py            # 측정 스크립트
/workspace/zap/artifacts/pilot_attention/
├── per_sample_attention.pt    # {"attention": [N, L, N_I] fp32, meta}
├── layer_stats.csv            # 32 row × {layer, mean, max, top50, var, entropy}
├── layer_attention_plot.png   # 2x2 subplot
└── decision.md                # case 분류 + scope 추천
```

#### 7.1 `per_sample_attention.pt` schema

```python
{
    "attention": torch.FloatTensor,        # [50, 32, 576]
    "sample_ids": list[str],               # length 50
    "model": "llava-1.5-7b-hf",
    "dataset": "scienceqa",                # confirm 후 채움
    "n_decode_steps": list[int],           # length 50, per-sample T
    "n_decode_steps_mean": float,
    "image_positions_example": list[int],  # 첫 sample image_indices.tolist()
    "seed": 0,
    "max_new_tokens": 64,
    "dtype": "bfloat16",
}
```

#### 7.2 `layer_stats.csv`

```
layer,mean_magnitude,max_magnitude,top50_concentration,cross_sample_variance,entropy
0,0.000891,0.012345,0.456,0.0023,5.78
...
31,0.000012,0.000089,0.234,0.0001,6.32
```

---

### 8. 시각화 (`layer_attention_plot.png`, 2×2 subplot)

| 위치 | 그래프 | 축 |
|------|--------|----|
| Top-L | `mean_magnitude` vs layer | x=layer 0..31, y=mean (log scale) |
| Top-R | `top50_concentration` vs layer | x=layer 0..31, y=concentration ∈ [0, 1] |
| Bot-L | Heatmap `[32, 576]` | x=image position 0..575, y=layer 0..31, color=magnitude |
| Bot-R | 24×24 spatial heatmap, 1~3 sample × 선정 layer (early/mid/late) | 9개 mini-heatmap |

---

### 9. 분석 + 의사결정 (`decision.md`)

#### 9.1 Case 분류

| Case | 정의 | 추천 scope |
|------|------|-----------|
| **A. Early-concentrated** | layer 0~3 mean ≥ 5× (다른 layer 평균) | A=all 32 / B=0-7 / C=0-3 |
| **B. Mid-layer peak** | argmax(mean) ∈ [8, 16] | A=all / B=4-15 / C=8-15 |
| **C. Bimodal** | early peak (0~3) AND late peak (16~24), 중간 골 | A=all / B=0-3 ∪ 16-23 (disjoint) / C=0-3 |
| **D. Uniform / 분산** | layer 간 magnitude std/mean ≤ 0.3 | A=all / B=stride-2 (0,2,...,30) / C=stride-4 (0,4,...,28) |

> 경계 케이스 (Case A vs B 동시 가능 등) 는 **mean magnitude 기준 우선**, top50_concentration
> 을 보조 지표로 사용. decision.md 에 근거 명시.

#### 9.2 decision.md 필수 항목

1. 측정 sample 수 N, mean T, dataset
2. layer-wise mean/max/top50/var 32-row 표 (markdown)
3. 어느 case 인지 + 근거 1~2 문장
4. **3가지 scope 추천** (A/B/C) + 각 motivation 1줄
5. (선택) Top-K concentration 분석 — magnitude 외 selectivity 관점
6. PLAN.md `n_modules` 후보 (e.g., scope B 채택 시 12 → student 메모리/시간 -62%)

---

### 10. 실행 절차

| Step | 작업 | 시간 |
|------|------|------|
| 1 | 사전 확인 (path 4종 + 데이터셋 confirm) | 5min |
| 2 | `pilot_attention_analysis.py` 작성 (argparse, eager attn, image_positions) | 15min |
| 3 | Smoke test `--n-samples 2` (OOM 체크 + layer 0/31 magnitude diff sanity) | 5min |
| 4 | 본 실행 `--n-samples 50` (wall-clock 측정) | 30min |
| 5 | 통계 계산 + plot + decision.md | 10min |
| 6 | 사용자에게 보고 + scope confirm | — |

총 **~1h**.

---

### 11. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| `output_attentions=True` + eager → OOM (long prefix) | 측정 실패 | scienceqa 짧은 prompt 우선; `max_new_tokens 64 → 32` 축소; per-sample `torch.cuda.empty_cache()` |
| Image placeholder 가 expand 안 된 input | image_indices 1개만 잡혀 attention ≈ 0 | helper 함수 (`infer_llava_image_positions_no_forward`) 강제 사용; 예시 1샘플의 image_indices 길이 = 576 검증 |
| Decode T 가 너무 짧음 (scienceqa 답변 1~2 token) | layer-wise variance 측정 신뢰도 ↓ | T_mean 을 decision.md 에 보고; T < 3 인 sample 별도 표시 |
| Layer indexing off-by-one | wrong layer 의 통계 출력 | smoke test 에서 `len(step_attns) == config.num_hidden_layers (32)` 검증 |
| Multi-image / any-resolution 샘플 혼입 | image_indices ≠ 576 → 통계 깨짐 | 길이 ≠ 576 인 샘플 skip + 카운트 (decision.md 에 보고) |
| 모든 layer attention ≈ 0 | placeholder 미확장 또는 image embedding 미주입 | smoke test 에서 layer 0 magnitude > 0 확인. 0 이면 input 처리 자체 점검 |

---

### 12. 완료 보고 형식

```
Pilot 완료:
- Dataset: scienceqa
- 측정 sample 수: N=50
- Mean decode steps: T=...
- Layer-wise pattern: Case [A/B/C/D]

Layer scope 추천:
- Scope A (full): layer 0-31  (학습 시간 baseline 1.0×)
- Scope B: layer X-Y          (근거: ...; 시간 ~M.M×)
- Scope C: layer X-Y          (근거: ...; 시간 ~K.K×)

Plot: /workspace/zap/artifacts/pilot_attention/layer_attention_plot.png
Detail: /workspace/zap/artifacts/pilot_attention/decision.md

다음 단계 후보:
- 사용자가 scope 1개 confirm → PLAN.md `n_modules` 확정 후 Stage 1 (teacher cache) 시작
- 또는 scope 3개 모두 → 3 GPU 병렬 학습 (A/B/C 각 1 GPU)
```

---

### 13. 체크리스트

#### 사전
- [ ] 데이터셋 선택 사용자 confirm (default: scienceqa)
- [ ] `infer_llava_image_positions_no_forward` import path 확인
- [ ] 출력 디렉토리 생성

#### 코드
- [ ] `pilot_attention_analysis.py` argparse: `--model`, `--dataset`, `--n-samples`, `--max-new-tokens`, `--output-dir`, `--seed`
- [ ] eager attention 강제 (`attn_implementation="eager"`)
- [ ] greedy decode 명시 (do_sample=False, temp=1.0, top_p=1.0, top_k=0, num_beams=1)
- [ ] per-sample reduce (메모리 절약)

#### 검증
- [ ] Smoke `--n-samples 2`: OOM 없음 + image_indices 길이=576 + layer 0/31 magnitude 출력
- [ ] `len(out.attentions[0]) == 32` 검증
- [ ] T_mean > 0 검증 (decode 정상)

#### 본 실행
- [ ] `--n-samples 50` 실행, wall-clock 기록
- [ ] `per_sample_attention.pt` 저장 확인 (shape `[50, 32, 576]`)
- [ ] `layer_stats.csv` 32 row 생성

#### 분석
- [ ] `layer_attention_plot.png` 4-subplot 생성
- [ ] Case 판정 + scope 3종 추천
- [ ] `decision.md` 작성

#### 보고
- [ ] 사용자에게 plot + decision.md 핵심 요약 전달
- [ ] scope confirm 받기 → PLAN.md `n_modules` 확정
