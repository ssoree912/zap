# Implementation log — EXP-20260425-002

**Branch**: `4090/cnn_image_scorer`
**Author**: ssoree912
**Date**: 2026-04-25

---

## 1. 새로 추가된 파일

| 파일 | 역할 |
|------|------|
| `pilot_attention_analysis.py` | Pilot — decode-to-image attention layer-wise 측정 |
| `scripts/pilot_compare_datasets.py` | 3 dataset 비교 plot |
| `collect_future_teacher_v2.py` | Stage 1 — per-sample teacher cache |
| `kvpress/presses/visual_utility_student.py` | 3-branch student 모듈 + ConvNeXt block + ranking loss |
| `train_visual_utility_student.py` | Stage 2 학습 루프 |
| `experiments/EXP-20260425-002/PLAN.md` | 본 실험 계획서 |
| `experiments/EXP-20260425-002/PLAN_pilot.md` | pilot 계획서 |

---

## 2. 핵심 설계 결정과 PLAN.md 반영

### 2.1 Layer scope (3개) — pilot 결과 반영

`SCOPES` (`visual_utility_student.py:24`):

```python
SCOPES = {
    "A":      tuple(range(32)),                  # 32 layer (full)
    "Bprime": (2, 3, 4, 5, 6, 7, 21, 22, 23),    # 9 layer (cross-dataset robust)
    "B":      (2, 3, 4, 5, 6, 7),                # 6 layer (selectivity peak)
}
```

선정 근거 (`/workspace/zap/artifacts/pilot_attention/decision.md`):
- 3 dataset (scienceqa / textvqa / docvqa, 각 50 sample) 의 layer-wise 평균에서 **layer 2..7 의 top-50 concentration 0.77~0.85** — 정규화 teacher 와 가장 정렬되는 selective fusion 영역.
- TextVQA 만은 layer 21~23 도 top50_conc 0.78~0.80 — OCR-heavy 의 late re-attention. → scope B' 에 포함하여 cross-dataset robust.
- Pearson r ≥ 0.90 (3 dataset 쌍) — layer-wise 패턴은 dataset invariant.

### 2.2 Teacher cache (PLAN.md §11.2)

Per-sample `.pt` (`/workspace/zap/data/teacher_v2/{dataset}/{sample_id}.pt`):

| 필드 | 형태 |
|---|---|
| `teacher_raw` | `[L=32, N_I=576]` fp16 — head·step 평균된 raw |
| `teacher_norm` | `[L, N_I]` fp16 — per-layer sum=1 (학습 target) |
| `image_token_indices` | `[576]` long |
| `question_token_indices` | `[~50~110]` long — image block 이후 모든 텍스트 토큰 |
| `prompt_text`, `image_path` | Stage 2 prefill 재실행용 |
| `T`, `prompt_len_mm`, `grid_h=24`, `grid_w=24` | meta |

Per-sample ~80 KB. 600 sample (200 × 3 dataset) 총 **48 MB**.

> Hidden state 는 캐시 안 함 — Stage 2 마다 LLaVA prefill forward 재실행 (~50 ms/sample). 600 sample × 10 epoch = 5분 추가 학습 시간 vs 99 GB 디스크 절감.

### 2.3 Student 구조 (PLAN.md §7.3, §11.3)

`VisualUtilityStudentLayer` (`visual_utility_student.py:60`) — 3 branch + fusion:

1. **Image CNN**: `H_img [B, N_I, D] → reshape [B, D, 24, 24] → 1x1 conv (D→256) → ConvNeXt block ×2 (DW7×7 + GN + 1×1×4 + GELU + 1×1, 잔차) → flatten [B, N_I, 256]`
2. **Question**: `H_q [B, N_Q, D] → mean-pool [B, D] → linear [B, 256] → broadcast [B, N_I, 256]`
3. **Raw**: `H_img → linear (D→256) [B, N_I, 256]`

**Fusion**: `concat(raw, conv, q, raw⊙q, conv⊙q) [B, N_I, 5×256=1280] → MLP (1280→512→1) → score [B, N_I]`

Per-layer parameter: ~4.88 M. Scope 별:
- A (32 layer): 156 M
- Bprime (9 layer): 44 M
- B (6 layer): 29 M

### 2.4 Loss (PLAN.md §7.4)

학생 출력에 softmax 적용 후 정규화된 teacher 와 비교:

```
pred_norm = softmax(student_score, dim=-1)
loss_l    = MSE(pred_norm, teacher_norm[l])
          + 0.1 * pairwise_rank_loss(pred_norm, teacher_norm[l],
                                     top_ratio=0.2, bottom_ratio=0.4, margin=0.05)
loss      = sum_{l in scope} loss_l
```

Ranking loss 는 `visual_utility_student.py:217` — top 20% vs bottom 40% positions 를 hinge margin 으로 분리.

---

## 3. 코드 리뷰에서 반영한 항목

3개 reviewer agent (reuse / quality / efficiency) 가 보고한 우선순위 high·med 항목:

| # | Severity | Issue | Fix | 위치 |
|---|---|---|---|---|
| 1 | High (efficiency) | `Image.open` + `processor()` 매 step 반복 (600×10 = 6,000회) | TeacherCacheDataset 가 init 에서 모든 .pt + 전처리된 input ids 를 RAM 적재 | `train_visual_utility_student.py:38` |
| 2 | High (quality) | `train_ds.files = files` 직접 대입 (생성자 우회 → brittle) | Dataset 생성자가 `files: list[Path]` 직접 받도록 | `train_visual_utility_student.py:48` |
| 3 | High (quality) | `except Exception` 너무 넓음 → KeyboardInterrupt/OOM 까지 삼킴 | `(RuntimeError, ValueError, IOError)` 만 catch + `traceback.format_exc()` 출력 | 3 파일 모두 |
| 4 | Med (quality) | `from_pretrained` 의 `cfg.pop("layer_indices")` — scope 변경 시 silent 로딩 위험 | 저장된 `layer_indices` 와 현재 `SCOPES` 매핑 일치 검증, 불일치 시 `RuntimeError` | `visual_utility_student.py:208` |
| 5 | Med (quality) | DataLoader shuffle 결정성 | `torch.Generator().manual_seed(seed)` 명시 전달 | `train_visual_utility_student.py:140` |
| 6 | Low (quality) | PLAN.md §xx 참조 docstring (rot 위험) | 섹션 번호 제거, 의미만 남김 | 4 파일 docstring |
| 7 | Low (efficiency) | `_summary.json` 의 `skipped=skipped[:50]` truncation | 전부 저장 (600 미만이라 부담 없음) | `collect_future_teacher_v2.py:259` |

### 미반영 (skipped) 항목

| Issue | 사유 |
|---|---|
| 데이터셋 로더 3개 (`load_scienceqa/textvqa/docvqa_samples`) 통합 → `load_vlm_samples` 재사용 | 3 dataset 의 prompt 형식이 충분히 달라 (scienceqa=options, textvqa=plain, docvqa=choice_list+context) 통합 시 오히려 분기 복잡. 현 코드 유지 |
| Scope 이름 stringly-typed | 본 실험에서만 사용하는 3 scope 키. enum 도입 비용이 더 큼 |
| Magic number `4096`, `576`, `24×24` 하드코딩 | LLaVA-1.5 single-image 가정. 다른 모델 포팅 시 별도 작업 |
| `pairwise_ranking_loss` 의 `for b in range(B)` | B=1 고정 학습이라 영향 없음 |
| `output_attentions=True` GPU 메모리 peak | 4090 24GB 에서 수집 완료 (max 16GB 사용 확인됨) |

---

## 4. 변경 전후 smoke 비교 (scope B, 1 epoch, 480 train samples)

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 1 epoch wall clock | 64.8s | **56.2s** (-13%) |
| Train loss (epoch 0) | 0.02694 | 0.02769 |
| Val loss (epoch 0) | 0.02634 | 0.02655 |
| Best ckpt 저장 | ✓ | ✓ |
| Loss 감소 패턴 | 0.0269 → 0.0263 (얕음) | 0.0321 → 0.0277 (steeper) |

> 변경 후 첫 step 의 loss 가 더 큰 이유: `torch.Generator(seed=0)` 로 DataLoader shuffle 이 결정론화되어 첫 batch 가 달라짐. 동일 seed 에서 재현 가능.

---

## 5. 파이프라인 현황

| Stage | 상태 |
|---|---|
| Pilot (3 dataset × 50 sample) | ✅ 완료 — `decision.md` 와 `cross_dataset_compare.png` |
| Stage 1 teacher cache (3 dataset × 200) | ✅ 완료 — 600 sample, 48 MB |
| Stage 2 student 학습 코드 | ✅ 완료 + smoke 통과 |
| Stage 2 본 학습 (3 GPU 병렬 A/Bprime/B) | ⏸ 대기 — 사용자 confirm 후 시작 |
| Stage 3 평가 | ⏸ 미시작 |

---

## 6. 다음 단계

**Stage 2 본 학습** (10 epoch, 3 GPU 병렬):

```bash
# GPU 0: scope A
CUDA_VISIBLE_DEVICES=0 python train_visual_utility_student.py \
  --scope A --epochs 10 --output-dir /workspace/zap/ckpts/student_v2_A &

# GPU 1: scope Bprime
CUDA_VISIBLE_DEVICES=1 python train_visual_utility_student.py \
  --scope Bprime --epochs 10 --output-dir /workspace/zap/ckpts/student_v2_Bprime &

# GPU 2: scope B
CUDA_VISIBLE_DEVICES=2 python train_visual_utility_student.py \
  --scope B --epochs 10 --output-dir /workspace/zap/ckpts/student_v2_B &
```

예상 wall-clock (3 GPU 병렬, 10 epoch, 480 train + 120 val):
- A: 32 layer × ~110ms = 3.5s/step × 4800 step = ~4.7h
- Bprime: ~1.4h
- B: ~0.95h

> Scope A 가 가장 오래 걸림 — wall-clock 결정. Bprime/B 가 일찍 끝나면 idle GPU 활용 (eval / sweep).

학습 완료 후 Stage 3: `ProbeImageConvTeacherPress` 와 유사한 `VisualUtilityStudentPress` 작성 → `eval_ppl.py` / `eval_rouge.py` 분기 추가 → mm-vet × detail_1k × 3 keep_ratio sweep.
