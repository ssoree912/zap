## 실험 결과

**ID**: EXP-20260415-002  
**Date**: 2026-04-16  
**Status**: Running (major reruns done, 일부 OOM 잔여)

---

### 1) 최신 요약

- 현재 기준 결론: **combine probe가 LOOK-M 대비 전반적으로 더 우세**.
- 비교 기준 CSV: `/workspace/zap/artifacts/results/probe_vs_lookm_r020_truncated_with_combine.csv`

요약 승패:

| 비교 | 비교 가능 데이터셋 수 | 승/패/무 |
|---|---:|---:|
| ScienceQA-only probe vs LOOK-M | 29 | **17 / 10 / 2** |
| Combine probe vs LOOK-M | 28 (wikivqa 제외) | **22 / 5 / 1** |
| Oracle(on-the-fly) vs LOOK-M | 14 (score 산출된 셋만) | **11 / 2 / 1** |

> wikivqa는 combine 쪽 `look_eval`(Accuracy)이 부분 추론으로 미산출되어, 해당 1개는 `exact_match_accuracy` fallback으로만 기록됨(LOOK-M Accuracy와 직접 비교 불가).

---

### 2) Combine 개선 포인트 (ScienceQA-only probe 대비)

동일 r=0.20 기준, combine이 특히 개선된 예시:

- `ocr_vqa`: **+0.22** (0.10 -> 0.32)
- `scenetransition`: **+0.15** (0.625 -> 0.775)
- `tqa`: **+0.08** (0.39 -> 0.47)
- `movingdirection`: **+0.05** (0.285 -> 0.335)
- `docvqa`: **+0.05** (0.46 -> 0.51)

하락 예시:

- `slidevqa`: -0.045
- `nuscenes`: -0.04
- `objectshuffle`: -0.03

전체(동일 metric 비교 가능 28개) 기준:

- 개선: 17
- 하락: 7
- 동일: 4

---

### 3) 실패 건 디버깅/패치 내용

#### 3.1 문제 현상

`wikivqa`에서 반복적으로 다음 에러가 발생:

- `IndexError('list index out of range')`

원인:

- LOOK-M 스타일 truncate 경로에서 일부 샘플이 `raw_img_list=[]`(이미지 0개)로 떨어짐.
- 기존 코드는 이미지가 0개여도 `processor(..., images=[])`를 호출하여 내부 전처리에서 `images[0]` 접근 시 `IndexError` 발생.

#### 3.2 적용한 코드 수정

수정 파일:

- `/workspace/zap/evaluate_image_teacher_pruning.py`

핵심 변경:

1. `image_paths`가 비어 있으면 text-only 토크나이즈 경로 사용
- 기존: 항상 `processor(text=..., images=images, ...)`
- 변경: `image_paths==[]`일 때 `processor(text=..., return_tensors="pt")`

2. `num_images=0`일 때 image position 추론 함수 호출 회피
- 기존: `infer_llava_image_positions_no_forward(..., num_images=0)`에서 `ValueError`
- 변경: 빈 `image_positions`를 설정하고 press는 no-op 압축으로 통과

#### 3.3 패치 영향 검증

재현 검증:

- `wikivqa` limit=6 재현에서 기존 오류 구간(sample_id 5 포함) `n_failures=0` 확인.
- 전체 `wikivqa` 재실행 결과:
  - `n_samples=200`
  - `n_predictions=194`
  - `n_failures=6`
  - 실패 원인 분해: **OOM 6, IndexError 0**

해석:

- 이번 패치는 **IndexError 제거용 안정화 패치**이며,
- 이미지가 정상(>=1)인 샘플의 score 계산/eviction 로직은 건드리지 않음.
- 따라서 성능 변화의 주원인은 패치가 아니라, 기존에 실패로 버려지던 샘플의 정상 처리 및 잔여 OOM 여부.

---

### 4) 정정: ScienceQA-only vs Combine 해석

앞선 설명의 일부는 부정확했고, 아래가 정확한 해석임.

- **둘 다 MileBench 전체 추론**을 수행함.
- ScienceQA-only vs Combine의 본질적 차이는 **probe 학습 데이터셋(체크포인트)**:
  - ScienceQA-only: ScienceQA 기반 probe
  - Combine: ScienceQA + TextVQA + NLVR2 기반 probe

이번 `IndexError`의 본질:

- probe 품질/학습데이터 차이 이슈가 아니라,
- truncate 경로에서 `raw_img_list=[]`가 된 샘플을 `processor(images=[])`로 넘긴 **코드 버그**.

근거:

- `evaluate_image_teacher_pruning.py`에서 오류 발생 지점은
  - 이미지 전처리(`processor(...)`) 단계이며
  - probe teacher score 계산/적용 이전.
- 즉, 같은 코드/같은 입력이면 ScienceQA-only probe에도 동일하게 재현 가능한 **모델 독립 버그**.

추가 확인 (WikiVQA, truncate=on, 4096/256):

- problem sample ids: `5, 9, 32, 83, 88, 89, 135, 136, 141, 145, 194` 등에서
- MileBench truncation 결과 `raw_img_list` 길이가 `0`으로 확인됨.

따라서 “ScienceQA-only라서 안 터졌다”가 아니라:

- 당시 실행 코드/설정/산출물 상태에서 해당 경로가 드러나지 않았던 것이고,
- 이번에 combine 실행 중 해당 샘플들이 실제로 통과되며 버그가 표면화된 것.

---

### 5) 현재 남은 리스크

- combine 결과에서 실질 실패가 남은 데이터셋: `wikivqa` (OOM 6건)
- oracle(on-the-fly)는 여전히 다수 데이터셋에서 OOM 비중이 높아, score 산출 데이터셋이 14개로 제한됨.

---

### 6) 현재 결론

- **주 결론 유지**: combine probe가 LOOK-M 대비 전반적으로 우세(22/5/1, comparable 28개 기준).
- 이번 패치는 실패 샘플 처리 안정화로서 타당하며, 성능 해석을 뒤집는 형태의 편향 패치가 아님.
- 잔여 이슈는 주로 OOM이며, 이는 메모리 예산/추론 설정 이슈로 분리 대응 필요.
