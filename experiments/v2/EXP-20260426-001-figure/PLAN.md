# EXP-20260426-001-figure — Figure 1 Teaser Design

## 0. Goal

Intro의 **Figure 1 teaser**를 만든다. Pipeline diagram이 아니라 *prefill saliency vs future visual utility* mismatch를 시각적으로 보여주는 figure다.

**Takeaway 한 줄**: KV에 남겨야 할 image token은 prefill에서 가장 눈에 띄는 token이 아니라, future answer가 실제로 다시 참조할 token이다.

---

## 1. Figure Layout (4-panel)

```
┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
│ (a)          │ (b)              │ (c)              │ (d)              │
│ Image +      │ Prefill saliency │ Future teacher   │ Student          │
│ Question +   │ heatmap          │ heatmap          │ predicted score  │
│ GT answer    │ (baseline focus) │ (decode-revealed)│ (our prediction) │
└──────────────┴──────────────────┴──────────────────┴──────────────────┘
```

각 heatmap은 **24×24 image token grid**를 원본 이미지 위에 overlay (LLaVA-1.5 7B는 patch grid 24×24).

---

## 2. 각 패널 정의

### (a) Image + Question + Answer
- 원본 이미지 + 질문 텍스트를 caption으로 표기.
- **Answer는 full-cache LLaVA가 생성한 답변**을 기준으로 함 (future attention도 이 generation 기준으로 계산됨).
- GT answer는 작은 글씨로 병기 가능하지만, attention은 model-generated answer 기준임을 명시.
- 작은 evidence가 정답에 결정적인 sample이어야 한다.

### (b) Prefill text saliency heatmap
- **Definition**: prefill의 **모든 text token** (role + instruction + question) 이 image token에 주는 attention.
- **Compute**:
  ```
  s_i^text2img = (1/|L|) * (1/(H*|T|)) * Σ_{l∈L} Σ_h Σ_{j∈T} A_prefill^(l)[h, j, i]
  ```
  여기서 T = 모든 non-image prefill position (role/template/question 포함).
- 의미: "decoding 전, prefill text 전체가 image token을 얼마나 중요하게 봤는가" — 가장 보수적인 static saliency baseline.
- **Layer 범위: L = {0, …, 31} (all 32 layers)**. student scope A와 동일하게 통일.

### (c) Future teacher heatmap (decode-revealed utility)
- **Definition**: model이 답변을 생성하는 동안 decode token들이 prefill image token에 준 attention.
- **Compute**:
  ```
  s_i^* = (1/|L|) * (1/(H*T)) * Σ_{l∈L} Σ_h Σ_{t=1..T} A_decode^(l)[h, t, i]
  ```
- **Generation 기준**: full-cache model의 greedy generation (GT force-decode 아님).
  - 실제 inference 시나리오와 일치하며, panel (a)의 answer caption과 동일한 generation.
- **Layer 범위: L = {0, …, 31}** (b)와 통일.

### (d) Student predicted heatmap
- **Definition**: Question-conditioned CNN-MLP student가 prefill hidden state로부터 예측한 utility score.
- **Compute**: `student_v2_A_gqa_lr1e4` checkpoint forward (scope A = all 32 layers).
  - **`future_probe_allL_limit100` (구 MLP probe)는 사용하지 않음** — 현재 main method는 CNN-MLP student.
- (c)와 시각적으로 가까울수록 **"우리가 미리 예측 가능"** 메시지가 강해진다.
- Layer-averaged score: 32개 layer score를 mean하여 single heatmap으로 표시.

---

## 3. Sample 선정 기준

### 우선순위 (작은/덜 salient한 region이 정답에 필요, 추론셋 한정)
1. **OCR / document text** — MileBench DocVQA, OCR-VQA, SlideVQA
2. **Slide / chart number** — MileBench SlideVQA, MultiModalQA
3. **Long-context needle** — MileBench TextNeedleInAHaystack
4. **Fine-grained attribute** — MileBench GPR1200, mm-vet `rec` tag
5. **OCR / math / spatial** — mm-vet `ocr`, `math`, `spat` tag

### 정량 selection criteria (자동 필터)
샘플별로 다음 두 점수를 계산해서 **차이가 가장 큰** 후보 풀을 만든다:
- `prefill_top10`: prefill saliency 상위 10% 토큰의 mass
- `future_top10`: future teacher attention 상위 10% 토큰의 mass
- `IoU(prefill_top10, future_top10)` 가 **낮은** 샘플 = mismatch가 큰 샘플

목표: IoU < 0.3 인 샘플 30~50개 추려서 시각화 → 그중 도메인 다양성 (OCR / counting / spatial / attribute) 고려하여 **3~4개 hero sample** 선택.

---

## 4. Required Runs

| # | Run | 입력 | 출력 | 구현 |
|---|-----|-----|------|------|
| R1 | 후보 데이터 30~50샘플 prefill+future attention dump | LLaVA full-cache (greedy gen) | `attn_dump/{dataset}/{id}.pt` per sample | `collect_attn_dump.py` (new) |
| R2 | CNN-MLP student score 포함 4-panel figure | R1 .pt + `student_v2_A_gqa_lr1e4` ckpt | `figure1_{id}.png` | `make_figure1.py --ckpt ckpts/student_v2_A_gqa_lr1e4` |
| R3 | mismatch metric (Spearman, TopKOverlap, IoU) 계산 | R1 .pt | `candidates_{dataset}.csv` (mismatch 큰 순) | `compute_mismatch.py` (new) |
| R4 | hero sample 선정 + figure 최종 PNG | R3 candidates + R2 | `figure1_{hero_id}.png` × 3~4 | manual review → `make_figure1.py` |

**훈련은 추가로 안 돌려도 된다.** Panel (d)는 반드시 **`student_v2_A_gqa_lr1e4` (question-conditioned CNN-MLP student, scope A)** 사용.
- `future_probe_allL_limit100` (구 MLP probe)는 Figure 1에 쓰지 않음 — main method와 구조가 다름.
- random image-only vs all-token 결과는 Figure 1 heatmap이 아닌 **image-only scope 정당화 motivation table**로 별도 배치.

---

## 5. 데이터 후보 (추론 데이터셋 한정)

우리가 평가에 쓰는 **MileBench / mm-vet / detail_1k** 안에서만 고른다.

### MileBench 서브셋 우선순위
| Subset | 적합도 | 이유 |
|--------|-------|------|
| **DocVQA** | ★★★ | document 안 small text → OCR mismatch 명확 |
| **OCR-VQA** | ★★★ | book cover 작은 글자 |
| **SlideVQA** | ★★★ | slide 안 차트/숫자/텍스트 |
| **MultiModalQA** | ★★ | text + image, 답 토큰이 image 작은 영역 참조 가능 |
| **TextNeedleInAHaystack** | ★★ | 긴 document 안 specific span |
| **GPR1200** | ★★ | fine-grained retrieval, 작은 attribute |
| **WebQA / TQA / WikiVQA** | ★ | 일반 VQA, mismatch 약할 수 있음 |
| **CharacterOrder / SceneTransition / ActionPrediction** | ✗ | 이건 multi-image temporal — single-image teaser에 안 맞음 |

→ Multi-image task는 제외 (T-1~T-4, S-3~S-5 일부). **single-image** subset 위주.

### mm-vet
- 218 sample 안에 capability tag 있음 (`rec`, `ocr`, `know`, `gen`, `math`, `spat`).
- **`ocr`, `math`, `spat`** 태그 우선 — 작은 evidence 비중 높음.

### detail_1k
- LLaVA description 태스크 = open-ended 캡션 생성, **specific question 없음**.
- "이 작은 영역이 답에 결정적" 메시지 만들기 어려움 → **Figure 1 후보에서 제외** (PPL/ROUGE evaluation용으로만).

### 최종 sampling 전략
- **MileBench single-image subsets** (DocVQA, OCR-VQA, SlideVQA, MultiModalQA, TextNeedleInAHaystack, GPR1200): 각 5~10샘플 → 약 30~50샘플
- **mm-vet ocr/math/spat tag**: 10~20샘플
- 합쳐서 50~70 후보 → mismatch IoU 낮은 순으로 hero 3~4개 선정.

---

## 6. 시각화 디자인

- **Heatmap**: 24×24 grid를 원본 이미지 해상도로 bilinear upsample, jet/turbo colormap, alpha=0.5로 overlay.
- **정규화**: 패널 (b)/(c)/(d) 각각 max-1 정규화 (절대값 비교 X, 분포 비교 O).
- 패널마다 overlay 위에 **top-k token (k=20~50)** 빨간 outline 표시 → "남기는 토큰"을 직관적으로.
- (d)에서는 student의 **keep mask (binary)** 도 옵션으로 추가 → "이 토큰만 KV에 남는다" 메시지.
- Figure caption: 1번에 적어둔 takeaway 그대로.

---

## 7. 구현 순서 (체크리스트)

- [ ] **S1**. 샘플 후보 리스트업: DocVQA/TextVQA/mm-vet에서 OCR·counting·spatial 50샘플 sampling script
- [ ] **S2**. `pilot_attention_*` 코드 포팅 → 후보 50샘플에 대해 prefill + decode attention dump (`run_dump_attn.sh`)
- [ ] **S3**. `future_probe_allL_limit100` student forward → per-sample score (`run_student_score.sh`)
- [ ] **S4**. mismatch metric 계산 + ranking (`compute_mismatch.py`) → `candidates.csv`
- [ ] **S5**. hero sample 3~4개 manual review (도메인 다양성 + GT가 명확한지)
- [ ] **S6**. 4-panel 시각화 스크립트 (`make_figure1.py`) → PNG/PDF 저장
- [ ] **S7**. Figure caption 다듬고 paper.md에 임베드

---

## 8. Risk / Open Questions

1. **Layer 평균 범위**: student (scope A)는 layer [0-31] **전체 32 layer**에서 eviction. (b) prefill saliency도 동일 layer 범위에서 평균 — all-layer 평균이 fair baseline. [24-31] subset 버전은 비교용으로만 참고.
   → all-layer 평균 (0-31) 기본, 필요시 두 버전 비교.
2. **Decode attention 추출**: teacher가 GT를 force-decode 해야 하나, 자유 generation으로 충분한가?
   → 자유 generation 우선 (실제 inference 시나리오). GT-force는 supplementary.
3. **Student vs Teacher mismatch**: (c)와 (d)가 너무 다르면 메시지 약화. probe 성능이 낮은 sample은 candidate에서 제외하는 필터 필요.
4. **24×24 grid mapping**: image preprocessing의 padding/resize 고려해서 patch index → image coordinate 매핑 정확히 할 것 (off-by-one 주의).

---

## 9. Deliverables

- `figure1_<sample_id>.png` × 3~4 (hero samples)
- `candidates.csv` (mismatch ranking, supplementary)
- `make_figure1.py` (재현용 스크립트)
- Figure caption (paper.md용)
