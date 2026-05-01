# EXP-20260425-002 Results (trajectory teacher, all-layer student)

**Dataset**: mm-vet (218) / MileBench (29 tasks)  
**Model**: LLaVA-1.5-7B  
**Student scope**: A (32 layers)  
**Teacher**: future-decode attention, M=3 trajectory average  
**Training data**: scienceqa + textvqa + docvqa + llava_instruct (800 samples, M=3)  
**Source**: `result/ppl_rouge_v2.csv`, `result/traj_all_layer_mmvet.csv`, `result/prefixkv_zuyan_ppl_rouge_results_20260424.csv`, `result/lookm_milebench.csv`

---

## 1. mm-vet PPL (n=218)

> PrefixKV는 eviction_ratio 기준 → keep_ratio = 1 - eviction_ratio 로 변환해서 비교.

| keep_ratio | full   | PrefixKV | A (ours) | traj (ours, M=3) |
|:----------:|:------:|:--------:|:--------:|:----------------:|
| 0.1        | —      | 7.3750   | 5.1662   | 5.1696           |
| 0.2        | —      | 5.9688   | 5.1403   | 5.1360           |
| 0.3        | —      | 5.7188   | 5.1367   | 5.1285           |
| 0.4        | —      | 5.5313   | 5.1415   | 5.1348           |
| 0.5        | —      | 5.5000   | 5.1454   | 5.1392           |
| 0.6        | —      | 5.4063   | 5.1489   | 5.1405           |
| 0.7        | —      | 5.3750   | 5.1517   | 5.1428           |
| 0.8        | —      | 5.2813   | 5.1539   | 5.1460           |
| 0.9        | —      | 5.2813   | 5.1573   | 5.1459           |
| 1.0 (full) | 5.1362 | —        | —        | —                |

- A / traj 모두 전 구간에서 PrefixKV보다 PPL 낮음 (낮을수록 좋음), 특히 low keep ratio에서 격차 큼
- traj vs A: 전 구간에서 traj가 0.003~0.008 낮은 PPL, 그러나 차이 미미

---

## 2. mm-vet ROUGE-L (n=218)

> PrefixKV eviction_ratio → keep_ratio = 1 - eviction_ratio 변환 적용.  
> mm-vet answer는 `<OR>/<AND>` 구분자로 복수 정답을 표기 → best match로 계산 (eval_rouge.py 수정 반영).  
> k=0.2 결과는 재실행 중.

| keep_ratio | full   | PrefixKV | A (ours) | traj (ours, M=3) |
|:----------:|:------:|:--------:|:--------:|:----------------:|
| 0.1        | —      | 0.383    | 0.0995   | —                |
| 0.2        | —      | 0.408    | 0.0994   | 0.1023           |
| 0.3        | —      | 0.462    | 0.1001   | 0.0988           |
| 0.4        | —      | 0.468    | 0.1017   | —                |
| 0.5        | —      | 0.483    | 0.1026   | 0.0995           |
| 0.6        | —      | 0.491    | 0.1021   | —                |
| 0.7        | —      | 0.597    | 0.1008   | 0.0973           |
| 0.8        | —      | 0.738    | 0.0995   | —                |
| 0.9        | —      | 0.767    | 0.0991   | —                |
| 1.0 (full) | 0.0986 | —        | —        | —                |

> ⚠️ 우리 ROUGE(~0.10)와 PrefixKV ROUGE(0.38~0.77) 스케일 차이가 크다. mm-vet answer가 `<OR>/<AND>` 처리 후에도 짧은 단어/숫자인 반면 모델이 긴 문장을 생성하면 ROUGE-L이 낮게 나오는 구조적 문제가 있다. 수정된 eval_rouge.py 재실행 결과 확인 필요.

- A 기준: full(0.0986) 대비 k=0.5에서 +0.004 개선
- traj: A 대비 전 구간 소폭 하락 (k=0.5 기준 -0.003)

---

## 3. MileBench (29 tasks, keep_ratio=0.2)

> LOOK-M 및 우리 방법 모두 LLaVA-1.5 기반. LLaVA-1.5는 멀티이미지 미지원으로 concat 방식 우회.  
> 우리 student는 단일이미지 기준 학습이므로 멀티이미지에서 구조적 불리함 있음.

| Method         | Avg (29 tasks) | LOOK-M 대비 |
|:--------------:|:--------------:|:-----------:|
| LOOK-M         | 0.3493         | baseline    |
| H2O-img        | 0.3947         | +0.0454     |
| Future-img     | 0.3944         | +0.0451     |
| Hybrid-img     | 0.3933         | +0.0440     |
| H2O-all        | 0.3208         | -0.0285     |
| **Future-all** | **0.3876**     | **+0.0383** |
| Hyb-flat       | 0.3875         | +0.0382     |
| Hyb-late       | 0.0801         | -0.2692     |
| Hyb-early      | 0.3522         | +0.0029     |

주요 관찰:
- Image-only 방식이 LOOK-M 대비 일관되게 우세 (+0.044~+0.045)
- Future-all / Hyb-flat도 LOOK-M 대비 +0.038 수준으로 유의미한 개선
- Hyb-late (L24-31만 적용)는 catastrophic failure → early layer에서 future signal 필수
- All-token H2O는 LOOK-M보다 낮음 → attention score 품질이 critical

---

## 4. Trajectory Teacher가 효과 없었던 원인 분석

### 4-1. Teacher variance ≈ 0 (실측)

M=3 trajectory 수집 후 `teacher_var` 확인:

```
teacher_var  max=1.07e-6   mean≈0
teacher_norm max=0.194     mean=0.00174
CV = var/mean ≈ 0
```

temperature=0.7 sampling에도 이미지 token에 대한 attention 패턴이 trajectory마다 사실상 동일하다.

### 4-2. 모델 구조적 이유

Student는 **prefill hidden state**를 입력으로 받는다. Prefill은 sampling 여부와 무관하게 항상 동일하다. Teacher인 decode attention도 "어떤 image token이 중요한가"를 prefill representation에 근거해 판단하므로, decode trajectory가 달라져도 image attention 패턴은 고정된다.

```
decode text는 trajectory마다 다름
→ 그러나 image token attention은 prefill에 의해 고정
∴ M > 1 은 M = 1과 실질적으로 동일한 teacher 생성
```

### 4-3. 데이터 추가 효과 미미

llava_instruct 200개 추가 (600 → 800):
- 가용 이미지가 COCO train2017 일부(1000장) 중 매칭된 2579개로 제한
- scienceqa / textvqa / docvqa와 질문 유형 유사 → 분포 다양성 기여 낮음
- ROUGE 소폭 하락 원인으로 추정 (데이터 domain mismatch 가능성)

### 4-4. 요약

| 요인 | 결과 |
|------|------|
| trajectory M=3 | teacher signal 개선 없음 (분산≈0) |
| llava_instruct 추가 | ROUGE 소폭 하락 |
| lr 1e-3 → 3e-4 | 학습 안정화 기여 |
| PPL vs A | 미미한 개선 (최대 Δ0.008) |
| ROUGE vs A | 소폭 하락 (Δ-0.003) |
| PPL vs PrefixKV | 현저한 개선 (k=0.5 기준 Δ-0.36) |

Trajectory teacher는 hidden state 기반 student + future-decode teacher 구조에서 효과 없다.  
다음 방향: teacher 정의 변경 (prefill attention 활용) 또는 데이터 다양성 확보.
