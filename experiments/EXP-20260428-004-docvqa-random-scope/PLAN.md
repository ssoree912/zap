## Experiment Plan

**ID**: EXP-20260428-004-docvqa-random-scope  
**Author**: Codex  
**Date**: 2026-04-28  
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)

DocVQA에서 같은 random eviction이라도 text token을 보존하고 image token만 evict하는 설정이, text+image 전체를 무작위로 evict하는 설정보다 훨씬 안정적인지 확인한다. 이 실험은 본 방법의 image-only eviction 설계를 정당화하기 위한 보조 그래프용이다.

### 2. 가설 (Hypothesis)

같은 nominal keep ratio에서 `random_image_only`가 `random_all_token`보다 낮은 PPL과 높은 ROUGE-L을 보일 것이다. Text token은 질문과 지시문 정보를 담기 때문에 random all-token eviction에서 성능이 크게 무너질 것으로 예상한다.

### 3. 독립변수 (What we change)

- Eviction scope: `random_image_only` vs `random_all_token`
- Keep ratio: `0.1, 0.2, ..., 0.9`

### 4. 종속변수 (What we measure)

- PPL: DocVQA GT answer에 대한 teacher-forced PPL
- ROUGE-L: random eviction 생성 결과와 full-cache LLaVA 생성 결과 간 ROUGE-L F1

### 5. 고정 조건 (What stays the same)

- Dataset: MileBench DocVQA inference split, 200 samples
- Model: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Image root: `/workspace/zap/data/MileBench/DocVQA/combined_1_images`
- Generation: greedy decoding, `max_new_tokens=32`
- Random press implementation: `experiments/EXP-20260426-001-figure/random_press.py`
- Seed: no explicit global seed reset; random press uses torch RNG per run

### 6. 베이스라인

- Full-cache LLaVA PPL on GT answers
- Full-cache LLaVA generation used as ROUGE-L reference

### 7. 예상 결과

`random_image_only`는 keep ratio가 낮아져도 text token을 보존하므로 완만하게 저하되고, `random_all_token`은 낮은 keep ratio에서 PPL이 급격히 증가하고 ROUGE-L이 크게 감소할 것이다.

### 8. 판단 기준 (Success Criteria)

- 대부분 또는 모든 keep ratio에서 `random_image_only`가 `random_all_token`보다 낮은 PPL
- 대부분 또는 모든 keep ratio에서 `random_image_only`가 `random_all_token`보다 높은 ROUGE-L
- 결과 CSV와 그래프가 같은 실험 폴더에 저장됨

### 9. 이 실험으로 증명할 수 없는 것

- 학습된 student score가 random image-only보다 좋은지는 별도 실험이 필요하다.
- Same nominal keep ratio 비교이므로, image-only 설정은 text token 보존 floor 때문에 effective prompt keep ratio가 낮은 ratio에서 nominal ratio보다 커질 수 있다.
- Random eviction baseline이므로 learned or oracle eviction의 상한/하한을 직접 의미하지 않는다.

### 10. 예상 런타임 / 리소스

- GPU: 기본 `GPU=2`
- 예상 시간: full reference 생성 + 36 random eval runs로 수 시간 가능
- 디스크: result JSON과 per-sample generation 포함 수십 MB 수준

