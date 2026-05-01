# EXP-20260425-002 Results ver2 (A vs traj, 현재 완료 기준)

**Dataset**: mm-vet (218) / MileBench  
**Model**: LLaVA-1.5-7B  
**Student A**: `student_v2_A` (scienceqa+textvqa+docvqa 600샘플, lr=1e-3)  
**Student traj**: `student_v2_traj` (scienceqa+textvqa+docvqa+llava_instruct 800샘플, M=3 trajectory, lr=3e-4)  
**attn_implementation**: sdpa (양쪽 동일)

> ⚠️ ROUGE vs OurFull: k=0.2/0.3/0.5/0.7만 sdpa 기반 신뢰 값. k=0.1/0.4/0.6/0.8/0.9는 재실행 중.

---

## 1. mm-vet PPL (n=218)

| keep_ratio | full   | A      | traj   | traj-A  |
|:----------:|:------:|:------:|:------:|:-------:|
| 0.1        | —      | 5.1662 | 5.1696 | +0.0034 |
| 0.2        | —      | 5.1403 | 5.1360 | -0.0043 |
| 0.3        | —      | 5.1367 | 5.1285 | -0.0082 |
| 0.4        | —      | 5.1415 | 5.1348 | -0.0067 |
| 0.5        | —      | 5.1454 | 5.1392 | -0.0062 |
| 0.6        | —      | 5.1488 | 5.1405 | -0.0083 |
| 0.7        | —      | 5.1517 | 5.1428 | -0.0089 |
| 0.8        | —      | 5.1539 | 5.1460 | -0.0079 |
| 0.9        | —      | 5.1573 | 5.1459 | -0.0114 |
| 1.0 (full) | 5.1362 | —      | —      | —       |

- k=0.2 이상 전 구간에서 traj가 A보다 PPL 낮음 (개선)
- 개선 폭 Δ0.004~0.011, 작지만 일관적
- k=0.1에서만 traj가 미세하게 높음 (+0.003)

---

## 2. mm-vet ROUGE-L vs GT (n=218)

> mm-vet 공식 GT 답변 기준. 짧은 GT 대비 긴 생성문 → 구조적으로 낮은 스케일.

| keep_ratio | full   | A      | traj   |
|:----------:|:------:|:------:|:------:|
| 0.2        | —      | 0.0994 | 0.1023 |
| 0.3        | —      | 0.1001 | 0.1020 |
| 0.5        | —      | 0.1026 | 0.0995 |
| 0.7        | —      | 0.1008 | 0.0973 |
| 1.0 (full) | 0.0986 | —      | —      |

- k=0.2/0.3: traj가 A보다 소폭 우세 (+0.002~0.003)
- k=0.5/0.7: traj가 A보다 소폭 하락 (-0.003)
- 전반적으로 두 모델 간 차이 미미 (±0.003 수준)

---

## 3. mm-vet ROUGE-L vs OurFull (n=218, sdpa 기준)

> Full-cache LLaVA-1.5-7B 생성 결과를 reference로 사용. 압축 모델이 full 모델 출력에 얼마나 가까운지 측정.

| keep_ratio | traj   | A (재실행 예정) |
|:----------:|:------:|:--------------:|
| 0.2        | 0.6333 | —              |
| 0.3        | 0.6481 | —              |
| 0.5        | 0.6640 | —              |
| 0.7        | 0.6688 | —              |

- keep ratio 높을수록 full 모델 출력에 가까워지는 단조 증가 패턴
- k=0.5 기준 full 대비 66.4% 유사도

---

## 4. MileBench (keep_ratio=0.2)

> 현재 traj 22/29 tasks 완료. A는 17 tasks 기준. 공통 17개 task 기준으로 비교.

| Method     | Avg (공통 17 tasks) | 비고 |
|:----------:|:-------------------:|:----:|
| LOOK-M     | 0.3596              | baseline (29 tasks) |
| **A**      | **0.3429**          | 17 tasks |
| **traj**   | **0.3428**          | 17 tasks (공통) |

- A vs traj: 공통 17개 기준 차이 없음 (0.3429 vs 0.3428, Δ0.0001)
- traj 전체 22 tasks avg: 0.3812 (나머지 5개 task 포함 시 더 높음)

> ⚠️ LOOK-M은 29 tasks 기준이라 직접 비교 시 주의 필요. 나머지 7개 task 완료 후 재비교 예정.

---

## 5. 종합 요약

| 지표 | A vs traj | 방향 |
|------|-----------|------|
| mm-vet PPL | traj 우세 (Δ0.004~0.011) | ✅ traj |
| mm-vet ROUGE vs GT | 거의 동일 (±0.003) | ➡ 무차별 |
| mm-vet ROUGE vs Full | traj 0.63~0.67 (A 미완) | — |
| MileBench (17 tasks) | 동일 (Δ0.0001) | ➡ 무차별 |

- **PPL에서 traj가 일관적으로 우세**하나 절대 폭이 작음 (≤0.011)
- **MileBench/ROUGE에서 traj 추가 효과 미확인**
- Trajectory teacher 효과: teacher_var≈0로 M=3이 M=1과 실질적으로 동일한 teacher 생성 → 데이터 800개 증가 효과가 PPL 소폭 개선에 기여한 것으로 추정
- 다음 방향: A에 대한 ROUGE vs Full 비교, MileBench 전체 29 tasks 완료 후 재평가
