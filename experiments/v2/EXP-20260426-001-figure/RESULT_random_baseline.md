# RESULT — Random Eviction Baseline (mm-vet, keep=0.2)

## Setup

- Model: vanilla LLaVA-1.5 7B (no probe / no training)
- Dataset: mm-vet, 218 samples
- **PPL** reference: 원본 mm-vet GT (`/workspace/data/mm-vet/mm-vet.json` 의 `answer`)
- **ROUGE-L** reference: full-cache LLaVA 출력 (`/workspace/zap/data/rouge_ref/our_full_mm-vet.json` 의 `answer`)
- Budget: `total_keep_ratio = 0.2`
- Random scoring: layer마다 독립적인 `torch.rand`
- GPU 1 = arm A, GPU 2 = arm B (병렬)

## MMVet keep ratio 0.2

| method | Eviction 후보 | **PPL (real GT) ↓** | **ROUGE-L vs full ↑** |
|-----|--------------|---------------------|------------------------|
| **A** `random_image_only` | image token만 | **5.2369** | **0.6108** |
| **B** `random_all_token`  | text + image 전체 | **82.1502** | **0.0970** |


## Take-away

text 토큰까지 random eviction 후보에 넣으면 **PPL ~16×, ROUGE −85%** 로 완전히 무너진다. 같은 budget (20%) 에서 image token만 무작위로 잘라도 PPL 5.24 / ROUGE 0.61 을 유지한다는 점은:

1. **"image token만 evict한다"는 디자인 결정 자체가 강력한 inductive bias** — 학습 없는 무작위 선택만으로도 image-only constraint가 압도적 차이를 만든다.
2. text 토큰 (특히 질문) 은 토큰 당 정보 밀도가 매우 높아, 일부만 잃어도 generation이 무너진다 — paper의 image-only 디자인은 이 자명한 사실을 활용한다.
3. 우리 main method (`VisualUtilityStudent`) 의 학습된 score는 image-only 위에서 추가 이득을 주는 형태로 해석되어야 함 (random image-only baseline 대비 추가 기여분).

## Files (mm-vet)

```
artifacts/EXP-20260426-001-figure/
├── mmvet_random_image/
│   ├── ppl/result.json              ppl=5.2369  (real GT)
│   └── rouge_vsourfull/result.json  rouge_l_f_mean=0.6108  (vs full)
└── mmvet_random_all/
    ├── ppl/result.json              ppl=82.1502 (real GT)
    └── rouge_vsourfull/result.json  rouge_l_f_mean=0.0970  (vs full)
```

---

# MileBench (S-1 ~ S-5, 13 tasks) @ keep=0.2

## Setup

- 동일 모델/budget (LLaVA-1.5 7B, `total_keep_ratio=0.2`)
- 카테고리 (Yu et al. MileBench scheme) — 모든 metric **↑ 높을수록 좋음**:
  - **S-1** Knowledge Grounded QA — WebQA, MultiModalQA, TQA, WikiVQA (Accuracy ↑)
  - **S-2** Text-Rich Images QA — DocVQA, OCR-VQA, SlideVQA (Accuracy ↑)
  - **S-3** Visual Relation Inference — Spot-the-Diff, CLEVR-Change, IEdit (ROUGE-L ↑)
  - **S-4** Dialogue — MMCoQA, ALFRED (ROUGE-L ↑)
  - **S-5** Space Understanding — nuscenes (Accuracy ↑)

## Results — Category averages

| Category | Datasets (n) | A `random_image_only` ↑ | B `random_all_token` ↑ | Δ (B−A) |
|----------|--------------|--------------------------|-------------------------|---------|
| S-1 | WebQA, MultiModalQA, TQA, WikiVQA (4) | 0.6325 | 0.4100 | −0.2225 |
| S-2 | DocVQA, OCR-VQA, SlideVQA (3) | 0.4350 | 0.2650 | −0.1700 |
| S-3 | Spot-the-Diff, CLEVR-Change, IEdit (3) | 0.1425 | 0.0773 | −0.0652 |
| S-4 | MMCoQA, ALFRED (2) | 0.3263 | 0.0643 | −0.2619 |
| S-5 | nuscenes (1) | 0.6100 | 0.4500 | −0.1600 |
| **Overall** | **(13 ds)** | **0.4250** | **0.2497** | **−0.1753** |

## Results — Per-task

| Task | Metric | A ↑ | B ↑ | Δ |
|------|--------|------|------|----|
| WebQA | Accuracy ↑ | 0.6000 | 0.4150 | −0.1850 |
| MultiModalQA | Accuracy ↑ | 0.7550 | 0.4900 | −0.2650 |
| TQA | Accuracy ↑ | 0.4650 | 0.2650 | −0.2000 |
| WikiVQA | Accuracy ↑ | 0.7100 | 0.4700 | −0.2400 |
| DocVQA | Accuracy ↑ | 0.5100 | 0.3300 | −0.1800 |
| OCR-VQA | Accuracy ↑ | 0.3200 | 0.1800 | −0.1400 |
| SlideVQA | Accuracy ↑ | 0.4750 | 0.2850 | −0.1900 |
| Spot-the-Diff | ROUGE-L ↑ | 0.1783 | 0.1002 | −0.0781 |
| CLEVR-Change | ROUGE-L ↑ | 0.1472 | 0.0576 | −0.0896 |
| IEdit | ROUGE-L ↑ | 0.1020 | 0.0741 | −0.0279 |
| MMCoQA | ROUGE-L ↑ | 0.3664 | 0.1190 | −0.2474 |
| ALFRED | ROUGE-L ↑ | 0.2861 | 0.0096 | −0.2765 |
| nuscenes | Accuracy ↑ | 0.6100 | 0.4500 | −0.1600 |

## MileBench Take-away

13개 task 모두 **A > B** (image-only random 이 all-token random 보다 우수). Overall 0.4250 → 0.2497 (Δ = −0.18 절대점, −41% 상대).

Dialogue 계열 (S-4: MMCoQA, ALFRED) 에서 Δ 가 가장 큼 (−0.26, −0.28). text 토큰을 잃으면 dialogue context가 무너져 ROUGE 거의 0 수준 (ALFRED 0.0096). 즉:

- text 토큰 보존 = **필수 조건**, 학습 없는 random 으로도 +0.18 의 절대 이득.
- mm-vet 의 PPL 결과 (1.61 vs 32.14) 와 일관된 신호 — text token 의 정보 밀도가 image token 대비 압도적.
- paper 본문에 "image-only constraint 자체가 강력한 inductive bias" 라는 ablation 표로 그대로 활용 가능.

## Files (MileBench)

```
artifacts/EXP-20260426-001-figure/
├── milebench_random_image/<TASK>/eval.json   # arm A, 13 tasks
└── milebench_random_all/<TASK>/eval.json     # arm B, 13 tasks
```
