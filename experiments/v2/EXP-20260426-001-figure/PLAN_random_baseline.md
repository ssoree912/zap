# Random Eviction Baseline — figure 보강 실험

## 0. Goal

**"이미지 토큰만 evict 후보로 두기"** vs **"text+image 전체를 evict 후보로 두기"** 의 성능 차이를 무작위 eviction으로 측정한다.

이 실험은 paper의 핵심 디자인 결정을 정당화하는 데 쓰인다:
- 우리 방법(`VisualUtilityStudent`)은 **image token만** evict한다.
- "왜 image-only로 제한하느냐?" 라는 reviewer 질문에 답하기 위한 데이터.
- 직관: text 토큰 (특히 질문)은 토큰 당 중요도가 매우 높아, all-token random은 image-only random보다 훨씬 나쁠 것이다.

## 1. 두 arm 정의

| Arm | Mode | Eviction 후보 | 선택 방식 | text 보존? |
|-----|------|--------------|----------|-----------|
| **A** | `random_image_only` | image token만 | 무작위 | ✓ 항상 |
| **B** | `random_all_token` | text + image 전체 | 무작위 | ✗ 무작위 evict 가능 |

- 두 arm 모두 `total_keep_ratio = 0.2` (전체 prefill 토큰의 20%만 KV에 유지)
- system prompt / BOS 같은 special token도 random pool에 포함 (B만 해당, A는 자동 보존)
- random scoring: layer마다 독립적인 `torch.rand`. seed는 sample id + layer_idx 기반 (재현용)

## 2. Dataset / Metric

- **mm-vet** (218 samples)
- 두 metric:
  - **PPL** — teacher-forced NLL on the full-cache reference answer
  - **ROUGE-L f-mean** — generated output vs full-cache reference
- 두 metric 모두 reference (정답)는 **full-cache LLaVA의 출력**: `/workspace/zap/data/rouge_ref/our_full_mm-vet.json` (`answer` 필드)
- dataset의 GT label이 아니라 우리 full output이 ground truth라는 점이 핵심 — eviction이 full output을 얼마나 보존하는지 직접 측정.

## 3. Model

- `/workspace/zap/ckpts/llava-1.5-7b-hf` (vanilla LLaVA-1.5 7B, 추가 학습 X)
- `attn_implementation = sdpa`, `max_new_tokens = 32`, prompt style `look_milebench`

## 4. 실행 자원

- arm A → **GPU 1**, arm B → **GPU 2** (병렬)
- 각 arm은 PPL + ROUGE-L 두 단계 순차 실행
- 예상 소요: arm 당 ~30~40분 (PPL ~10분 + ROUGE 218샘플 generation ~25분)

## 5. 구현

세 파일로 구성:

### `random_press.py`
- `RandomImageOnlyPress(ImageTokenTopKPress)` — `score_image_tokens()` 가 `torch.rand` 반환
- `RandomAllTokenPress(BasePress)` — H2OAllTokenPress와 동일 구조, attention 대신 `torch.rand` 사용
- 두 클래스 모두 layer 별 독립적 random (각 layer가 자기만의 random subset 결정)
- Reproducibility: 스크립트 시작 시 `torch.manual_seed(42)` 고정

### Evaluator 패치
- `eval_ppl.py`, `eval_rouge.py` 에 method choices `random_image_only`, `random_all_token` 추가
- `eval_ppl.build_press()` 에 두 케이스 추가 (lazy import from this folder)
- 그 외 코드는 손대지 않음

### Shell scripts (각 GPU에서 독립 실행)
- `run_mmvet_random_image.sh` — GPU 1, arm A (PPL + ROUGE)
- `run_mmvet_random_all.sh`   — GPU 2, arm B (PPL + ROUGE)

## 6. Output

```
artifacts/EXP-20260426-001-figure/
├── mmvet_random_image/
│   ├── ppl/result.json              (ppl, n_samples, args, per_sample)
│   └── rouge_vsourfull/result.json  (rouge_l_f_mean, ...)
└── mmvet_random_all/
    ├── ppl/result.json
    └── rouge_vsourfull/result.json
```

비교 표 (RESULT_random_baseline.md):
| arm | PPL | ROUGE-L vs full |
|-----|-----|-----------------|
| random_image_only | x.xxxx | 0.xxxx |
| random_all_token  | x.xxxx | 0.xxxx |
| Δ (B − A)         | +Δ ppl | −Δ rouge |

## 7. Expected outcome (가설)

| 결과 패턴 | 해석 | paper 기여 |
|----------|------|-----------|
| A ≫ B (e.g., 0.30 vs 0.05) | text 토큰 보존이 결정적. image-only 제한이 강력한 inductive bias | 우리 design choice 정당화 |
| A ≈ B | text 토큰도 redundant. 우리 방법은 추가 supervision으로 이득 | 학습된 score의 가치 강조 |
| A < B | text 토큰이 오히려 noise. 거의 발생 안 할 것 | (예상 밖) |

가장 가능성 높은 시나리오: **A ≫ B**. paper의 image-only 디자인을 "토큰 타입 자체가 강한 prior"라는 식으로 motivate 가능.

## 8. Figure 활용

- 결과 표는 paper의 ablation section 또는 appendix
- Figure 1 teaser 와는 별개. 이건 "왜 image-only 인가" 정당화용.
- 만약 결과가 깔끔하면 본문에 한 줄 표 + bar chart로 요약.

## 9. 체크리스트

- [x] `random_press.py` 작성
- [x] eval_ppl/eval_rouge 에 두 method 등록
- [ ] GPU 1 — arm A 실행 (`bash run_mmvet_random_image.sh`)
- [ ] GPU 2 — arm B 실행 (`bash run_mmvet_random_all.sh`)
- [ ] `RESULT_random_baseline.md` 작성 (PPL/ROUGE 표 + take-away)
