## 실험 결과

**ID**: EXP-20260415-002  
**Date**: 2026-04-16  
**Status**: [x] Done (partial — OOM/format errors on several datasets)

---

### 실험 요약

`image_probe_combined_v1` (ScienceQA + TextVQA + NLVR2) probe를 **truncate 없이** 실행한 결과.
`truncate_like_lookm=False`, `look_max_context_len=None` 조건이었기 때문에 이미지 수가 많은 데이터셋에서
CUDA OOM이 다수 발생함.

probe 모델 경로: `/workspace/hd/artifacts/sq_teacher/image_probe_combined_v1/mlp`  
keep ratio: `total_keep_ratio=0.20`, teacher: `att_only_postvision`, method: MLP

---

### 실행 상태 분류

#### A. 완전 성공 (14개 데이터셋)

| dataset | metric | combined_probe | look_m (r=0.20) | Δ | 승자 |
|---|---|---|---|---|---|
| alfred | ROUGE-L | 0.2686 | 0.1649 | **+0.1037** | probe |
| clevr_change | ROUGE-L | 0.1410 | 0.1786 | -0.0376 | look_m |
| counterfactualinference | Accuracy | 0.3200 | 0.3000 | +0.0200 | probe |
| docvqa | Accuracy | 0.5150 | 0.4700 | **+0.0450** | probe |
| iedit | ROUGE-L | 0.1108 | 0.0391 | **+0.0717** | probe |
| movingattribute | Accuracy | 0.5100 | 0.4900 | +0.0200 | probe |
| movingdirection | Accuracy | 0.3350 | 0.3250 | +0.0100 | probe |
| nuscenes | Accuracy | 0.6100 | 0.6150 | -0.0050 | look_m |
| objectexistence | Accuracy | 0.4900 | 0.5100 | -0.0200 | look_m |
| ocr_vqa | Accuracy | 0.3200 | 0.0900 | **+0.2300** | probe |
| slidevqa | Accuracy | 0.4750 | 0.4600 | +0.0150 | probe |
| spot_the_diff | ROUGE-L | 0.1919 | 0.1612 | +0.0307 | probe |
| tqa | Accuracy | 0.3800 | 0.4100 | -0.0300 | look_m |
| webqa | Accuracy | 0.6100 | 0.5650 | +0.0450 | probe |

**W/L (valid 14개 기준)**: probe 10승 / look_m 4승

#### B. OOM으로 전체 실패 (5개 데이터셋, metrics.json 없음)

모든 샘플이 `CUDA out of memory`로 실패. metrics.json 미생성.

| dataset | n_samples | n_failures | 비고 |
|---|---|---|---|
| actionlocalization | 200 | 200 (100%) | OOM |
| actionprediction | 200 | 187 (93.5%) | OOM |
| actionsequence | 200 | 186 (93%) | OOM |
| characterorder | 200 | 177 (88.5%) | OOM |
| egocentricnavigation | 200 | 200 (100%) | OOM |

#### C. OOM으로 부분 실패 (7개 데이터셋, 메트릭 신뢰 불가)

성공 샘플이 일부 있으나 너무 적어 메트릭을 신뢰하기 어려움. look_eval 미산출.

| dataset | n_samples | n_preds | n_failures | 실패율 |
|---|---|---|---|---|
| gpr1200 | 600 | 403 | 197 | 33% |
| imageneedleinahaystack | 320 | 90 | 230 | 72% |
| objectinteraction | 200 | 8 | 192 | 96% |
| objectshuffle | 200 | 28 | 172 | 86% |
| scenetransition | 200 | 4 | 196 | 98% |
| statechange | 200 | 40 | 160 | 80% |
| textneedleinahaystack | 320 | 60 | 260 | 81% |
| wikivqa | 200 | 189 | 11 | 5.5% — look_eval 미산출 |

> wikivqa는 OOM 실패가 11건으로 적지만, look_eval (Accuracy) 미산출 상태.
> metrics.json의 `exact_match_accuracy=0.0053`은 MileBench 공식 평가 결과가 아님.

#### D. 데이터 포맷 오류 (2개 데이터셋)

OOM이 아닌 `ValueError: image placeholder mismatch` — 이미지 개수 vs 플레이스홀더 개수 불일치.

| dataset | n_samples | n_failures | 오류 유형 |
|---|---|---|---|
| mmcoqa | 200 | 159 (79.5%) | `Question contains N image placeholders but received M images` |
| multimodalqa | 200 | 200 (100%) | `Question contains 1 image placeholder but received 2 images` |

> multimodalqa는 전수 실패. 2-image 포맷을 combined probe 실행 스크립트가 처리하지 못하는 것으로 보임.
> 기존 truncated 평가(probe_vs_lookm_summary.csv)에서는 multimodalqa가 정상 동작했으므로,
> combined probe 실행 시 prompt_template이나 image 처리 경로에 차이가 있을 가능성.

---

### 이전 (truncated probe_mlp) vs 이번 (combined probe, no truncation) 비교

유효한 14개 데이터셋에 대해 비교:

| dataset | truncated_probe_mlp | combined_probe | look_m | 변화 |
|---|---|---|---|---|
| alfred | 0.2834 | 0.2686 | 0.1649 | -0.015 (둘 다 probe 승) |
| clevr_change | 0.1419 | 0.1410 | 0.1786 | ≈ (둘 다 look_m 승) |
| counterfactualinference | 0.3250 | 0.3200 | 0.3000 | ≈ (둘 다 probe 승) |
| docvqa | 0.4600 | 0.5150 | 0.4700 | **+0.055** (역전: truncated는 look_m 승) |
| iedit | 0.1102 | 0.1108 | 0.0391 | ≈ (둘 다 probe 승) |
| movingattribute | 0.4950 | 0.5100 | 0.4900 | +0.015 (둘 다 probe 승) |
| movingdirection | 0.2850 | 0.3350 | 0.3250 | **+0.050** (역전: truncated는 look_m 승) |
| nuscenes | 0.6500 | 0.6100 | 0.6150 | **-0.040** (역전: truncated는 probe 승) |
| objectexistence | 0.4800 | 0.4900 | 0.5100 | +0.010 (둘 다 look_m 승) |
| ocr_vqa | 0.1000 | 0.3200 | 0.0900 | **+0.220** (probe 승 유지, 대폭 향상) |
| slidevqa | 0.5200 | 0.4750 | 0.4600 | -0.045 (둘 다 probe 승) |
| spot_the_diff | 0.1953 | 0.1919 | 0.1612 | ≈ (둘 다 probe 승) |
| tqa | 0.3900 | 0.3800 | 0.4100 | ≈ (둘 다 look_m 승) |
| webqa | 0.6200 | 0.6100 | 0.5650 | ≈ (둘 다 probe 승) |

> `combined_probe`가 truncated 대비 눈에 띄게 향상된 데이터셋: **ocr_vqa (+0.22), movingdirection (+0.05), docvqa (+0.055)**  
> 하락한 데이터셋: nuscenes (-0.04), slidevqa (-0.045), alfred (-0.015)  
> ocr_vqa의 대폭 향상은 TextVQA 학습 데이터 추가 효과로 해석 가능.

---

### 결론

- 유효 14개 기준 **probe 10승 / look_m 4승** — 이전 (전체 29개, 17/10/2)과 유사한 경향
- `combined_v1` 모델은 ocr_vqa, docvqa 등 텍스트 dense 이미지에서 확실한 향상
- **truncation 없이 실행했기 때문에 이미지가 많은 데이터셋 (video-frame 계열) 에서 대부분 OOM**
- OOM 데이터셋들 (actionlocalization, actionsequence, egocentricnavigation 등)은 truncation을 걸고 재실행 필요
- multimodalqa, mmcoqa는 포맷 오류 — prompt template에서 multi-image 처리 로직 수정 필요

---

### 다음 단계

- [ ] OOM 데이터셋 12개: `truncate_like_lookm=True` 조건으로 재실행
- [ ] multimodalqa, mmcoqa: multi-image placeholder mismatch 원인 파악 및 수정
- [ ] wikivqa: look_eval 미산출 이유 확인 (look_result_dir 설정 문제일 수 있음)
- [ ] 전체 29개 유효 결과 확보 후 EXP-20260415-001 (ScienceQA-only) 와 직접 비교
