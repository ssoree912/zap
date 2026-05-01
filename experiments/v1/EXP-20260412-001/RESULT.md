## Experiment Result

**ID**: EXP-20260412-001
**Date**: 2026-04-13
**Status**: [x] Done

---

### 결론 요약

- **probe_mlp vs LOOK-M (r=0.20)**: 25개 유효 쌍 중 12승 13패 → 승률 48%
- **probe_mlp ≈ probe_linear**: 대부분 데이터셋에서 거의 동일, mlp가 미세하게 우위
- **3개 데이터셋 OOM 완전 실패**: ActionLocalization, EgocentricNavigation, MultiModalQA
  - GPU 당시 타 프로세스가 33 GiB 점유 → 모든 샘플 OOM
  - LOOK-M도 이 데이터셋들에서 성능 보유 (0.220, 0.285, 0.655)
- **ratio 변화에 무감각**: 대부분 데이터셋에서 r=0.05/0.10/0.20 결과가 동일 또는 차이 미미

---

### r=0.20 전체 결과 (probe_mlp / probe_linear / LOOK-M)

| Dataset | Metric | MLP | Linear | LOOK-M | Winner |
|---|---|---|---|---|---|
| actionlocalization | Accuracy | FAIL | FAIL | 0.220 | look_m |
| actionprediction | Accuracy | 0.125 | 0.231 | 0.530 | look_m |
| actionsequence | Accuracy | 0.200 | 0.143 | 0.440 | look_m |
| alfred | ROUGE-L | 0.265 | 0.277 | 0.165 | **linear** |
| characterorder | Accuracy | 0.294 | 0.238 | 0.310 | look_m |
| clevr_change | ROUGE-L | 0.150 | 0.127 | 0.179 | look_m |
| counterfactualinference | Accuracy | 0.320 | 0.320 | 0.300 | **mlp** |
| docvqa | Accuracy | 0.515 | 0.515 | 0.470 | **mlp** |
| egocentricnavigation | Accuracy | FAIL | FAIL | 0.285 | look_m |
| gpr1200 | Accuracy | 0.171 | 0.151 | 0.042 | **mlp** (+306%) |
| iedit | ROUGE-L | 0.110 | 0.112 | 0.039 | **linear** (+185%) |
| imageneedleinahaystack | ROUGE-L | 0.000 | 0.000 | 0.000 | tie |
| mmcoqa | ROUGE-L | 0.253 | 0.235 | 0.330 | look_m |
| movingattribute | Accuracy | 0.510 | 0.510 | 0.490 | **mlp** |
| movingdirection | Accuracy | 0.335 | 0.335 | 0.325 | **mlp** |
| multimodalqa | Accuracy | FAIL | FAIL | 0.655 | look_m |
| nuscenes | Accuracy | 0.610 | 0.610 | 0.615 | look_m (tiny) |
| objectexistence | Accuracy | 0.490 | 0.490 | 0.510 | look_m |
| objectinteraction | Accuracy | 0.125 | 0.333 | 0.485 | look_m |
| objectshuffle | Accuracy | 0.474 | 0.400 | 0.345 | **mlp** |
| ocr_vqa | Accuracy | 0.320 | 0.320 | 0.090 | **mlp** (+256%) |
| scenetransition | Accuracy | 0.000 | 0.000 | 0.665 | look_m (large gap) |
| slidevqa | Accuracy | 0.475 | 0.470 | 0.460 | **mlp** |
| spot_the_diff | ROUGE-L | 0.202 | 0.189 | 0.161 | **mlp** |
| statechange | Accuracy | 0.250 | 0.275 | 0.325 | look_m |
| textneedleinahaystack | ROUGE-L | 0.076 | 0.075 | 0.100 | look_m |
| tqa | Accuracy | 0.385 | 0.375 | 0.410 | look_m |
| webqa | Accuracy | 0.605 | 0.605 | 0.565 | **mlp** |
| wikivqa | Accuracy | 0.438 | 0.427 | 0.620 | look_m |

**probe_mlp 승: 12 / 패: 13** (유효 25쌍, OOM 3개 제외)

---

### ratio 감도 분석 (probe_mlp: r=0.05 / 0.10 / 0.20)

대부분 데이터셋이 ratio에 무감각:
- **완전 평탄** (r=0.05~0.20 동일): docvqa(0.515), counterfactualinference(0.320), movingattribute(0.510), movingdirection(0.335), nuscenes(0.610), objectexistence(0.490), objectinteraction(0.125), ocr_vqa(0.320), wikivqa(0.438)
- 이는 probe가 ratio 관계없이 동일한 예측을 생성함을 의미
  - → ScienceQA로만 학습된 probe의 domain shift로 인해 score 분포가 데이터셋마다 비슷하게 작동하거나 full cache 수준의 정보를 이미 확보한 경우

---

### 실패 분석

#### OOM 완전 실패 (3개)
- ActionLocalization, EgocentricNavigation, MultiModalQA
- 원인: 실험 당시 GPU에 타 프로세스(33 GiB) 점유 → 할당 불가
- 재실험 필요 시 GPU 단독 사용 조건에서 재시도

#### SceneTransition (0.000 vs LOOK-M 0.665)
- probe는 모든 ratio에서 0점 → 답변 형식 또는 태스크 구조 문제
- LOOK-M은 동일 데이터셋에서 0.665 달성 → probe 모델의 domain mismatch 가능성

#### ObjectInteraction (0.125 vs LOOK-M 0.485)
- probe 전 ratio 동일(0.125) → 답변이 항상 같은 패턴으로 고정됨

---

### probe_mlp vs probe_linear 비교

대체로 동일하지만 mlp 미세 우위:
- MLP 우세: gpr1200 (0.171 vs 0.151), objectshuffle (0.474 vs 0.400), characterorder (0.294 vs 0.238)
- Linear 우세: alfred (0.277 vs 0.265), iedit (0.112 vs 0.110), statechange (0.275 vs 0.250)
- 실험 논문에는 **mlp 기본 사용** 권장

---

### Oracle (att_only_postvision) 참조값 @ r=0.20

| Dataset | Oracle |
|---|---|
| clevr_change | 0.122 |
| iedit | 0.097 |
| spot_the_diff | 0.206 |
| docvqa | 0.515 |

probe_mlp와 oracle이 거의 동일 → probe가 oracle 수준 근접.

---

### 시사점

1. **probe 승률 48%는 공정 비교 기준(total_keep_ratio 통일)**에서의 실제 성능
2. 이전 image_keep_ratio 기준 대비 약간 낮아진 것 예상 (r_eff가 좁혀짐)
3. **probe가 크게 이기는 데이터셋**: GPR1200, IEdit, OCR-VQA, Spot-the-Diff — 시각적 content 이해 중심
4. **LOOK-M이 크게 이기는 데이터셋**: SceneTransition, ActionSequence, WikiVQA — 시간적/텍스트 reasoning 중심
5. **ratio 무감각성**은 probe 모델의 domain generalization 한계 시사 → 더 다양한 데이터로 학습된 probe 필요

---

### 아티팩트

- probe_mlp 결과: `/workspace/hd/artifacts/probe_global/*/probe_mlp/`
- probe_linear 결과: `/workspace/hd/artifacts/probe_global/*/probe_linear/`
- LOOK-M 결과: `/workspace/hd/artifacts/probe_global/*/look_m/keep_0p20/`
