# 결과 — EXP-20260418-001

**ID:** EXP-20260418-001
**상태:** Phase 3 (전체 MileBench 평가) 진행 중
**업데이트:** 2026-04-19

---

## 타임라인 요약

| 버전 | 수집 방식 | 학습 방식 | Epoch | 비고 |
|---|---|---|---|---|
| v1 | per-sample `.pt`, 4 layer | 레이어별 독립 MLP, textvqa 단독 | 10 | 초기 sanity (기존) |
| v2 | per-sample `.pt`, 4/32 layer | 레이어별 lazy load | 10 | 3 데이터셋 확장 (실패·OOM 경험) |
| **v3** | **unified shard, 32 layer** | **one-pass multi-layer, per-sample softmax MSE** | **10** | **B 방법 통일 완료, MileBench 평가** |
| **v4** | 동일 (SSD 재수집) | 동일 | **20** | **추가 수렴 — 최종 채택** |

---

## Phase 1: Probe 수집·학습 파이프라인 (unified B 방법)

### 설계
- **공통 shard 포맷** (한 번의 LLaVA full run으로 두 label 동시 추출):
  ```
  x         [R, 4096] fp16    hidden at image position
  y_pv      [R]       fp16    post-vision text → image attention (max over Q, mean over H)
  y_future  [R]       fp16    decode → image attention (mean over T, H)
  layer     [R]       uint8
  sample_id [R]       int32
  token_idx [R]       int16
  ```
- **통일 학습 loss**: per (sample_id, layer) group
  ```
  pred  = softmax(MLP_l(x_group), dim=0)
  label = y_group / y_group.sum()
  loss  = MSE(pred, label)
  ```
- 두 방법의 차이는 **teacher label (y_pv vs y_future)** 뿐. Input, grouping, loss, batch 모두 동일.

### 데이터
- textvqa 500 + scienceqa 500 + nlvr2 500 = **1500 샘플 × 32 layer × 576(or 1152) token**
- shard 파일: 584개 (167 + 250 + 167), 총 36.8M 행, **~282 GB**
- 저장 위치: `/workspace/zap/artifacts/teacher_ssd/unified/{textvqa,scienceqa,nlvr2}/` (SSD)

### v4 학습 결과 (20 epoch)

| Probe | 입력 | 레이어 | Best Epoch | Best val Spearman |
|---|---|---|---|---|
| **PostVision v4** | 공통 shard (y_pv) | **32** (0-31) | 20 | **0.7512** |
| **Future v4** | 공통 shard (y_future) | **8** (24-31) | 13 | **0.7124** |

- PV는 20 epoch까지 꾸준히 상승 (0.67 → 0.75). 추가 epoch 여지 있음.
- Future는 epoch 13이 최고점, 이후 oscillation (0.62-0.71). 거의 수렴.

체크포인트:
- `/workspace/zap/ckpts/postvision_probe_v4_20ep`
- `/workspace/zap/ckpts/future_probe_v4_last8_20ep`
- `/workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31` — inference용 (layer 31 가중치를 0-23에 복사)

---

## Phase 2: 초기 평가 (v3 probes, 단일 데이터셋)

v4 전 단계 v3 probe로 한 1차 비교.

### CLEVR-Change (2 이미지, 200 샘플, ROUGE-L)

| k | PV | Future | Hybrid α=0.5 |
|---|---|---|---|
| 0.5 | 0.1227 | 0.1205 | **0.1242** ⭐ |
| 0.2 | 0.1293 | **0.1339** ⭐ | 0.1268 |
| 0.1 | 0.1398 | **0.1459** ⭐ | 0.1343 |

**관찰**: k가 작아질수록 Future 우위. Hybrid는 k=0.5에서만 최고.

### Spot-the-Diff (2 이미지, 200 샘플, ROUGE-L)

| k | PV | Future | Hybrid α=0.5 |
|---|---|---|---|
| 0.5 | **0.1985** ⭐ | 0.1970 | 0.1978 |
| 0.2 | **0.1925** ⭐ | 0.1819 | 0.1851 |
| 0.1 | **0.1834** ⭐ | 0.1727 | 0.1753 |

**관찰**: PV가 전 keep_ratio에서 일관되게 최고. Future 열세.

### ActionSequence (다중 이미지, truncated, Accuracy)

| k | PV | Future | Hybrid |
|---|---|---|---|
| 0.5 | 0.46 | 0.46 | 0.46 |
| 0.2 | 0.46 | 0.46 | 0.46 |
| 0.1 | 0.46 | 0.46 | 0.46 |

**포화** — multi-choice 지표로는 method 차이 안 드러남.

### DocVQA (단일 이미지, 200 샘플, Accuracy)

모든 조합에서 Accuracy=0.510 포화. 단, 실제 prediction은 서로 다름 (Future가 가장 다양).

### Phase 2 결론

- Future의 우월성이 **dataset-dependent**: CLEVR-Change(합성, 단순) vs Spot-the-Diff(자연, 복잡) 정반대
- PV는 Spot-the-Diff 등 semantic-heavy task에서 robust
- **전체 MileBench 평가 필요** — 어느 쪽이 전반적으로 우위인지, 두 signal 상호보완 여부 검증

---

## Phase 3: 전체 MileBench 평가 (v4 probes) — **완료 (29/29)**

### 설정
- **체크포인트**: `postvision_probe_v4_20ep`, `future_probe_v4_last8_20ep_bcast31`
- **데이터셋**: MileBench 29개
- **Keep ratio**: 0.5, 0.2, 0.1
- **Mode**: PV only / Future only / Hybrid α=0.5 (3 GPU 병렬)
- **Truncation**: `--truncate_like_lookm` (OOM 방지)

### Summary (mode별 승수, k별 row 기준 29×3 = 87 row)

| PV win | Future win | Hybrid win | tie |
|---:|---:|---:|---:|
| 16 | 7 | 3 | 61 |

### 데이터셋 분류

- **포화(mode 간 동일, 16개)**: actionlocalization, actionprediction, actionsequence, characterorder, counterfactualinference, docvqa, egocentricnavigation, gpr1200, imageneedleinahaystack, movingattribute, movingdirection, nuscenes, objectexistence, objectinteraction, objectshuffle, ocr_vqa — 다지선다 / 단답 accuracy 지표가 pruning에 무감. 방법 비교에는 의미 없음.
- **방법 간 차이 발생(13개)**: alfred, clevr_change, iedit, mmcoqa, multimodalqa, scenetransition, slidevqa, spot_the_diff, statechange, textneedleinahaystack, tqa, webqa, wikivqa.

### 방법 간 차이가 있는 13개 dataset 결과

| Dataset | Metric | k=0.5 | k=0.2 | k=0.1 | Winner/경향 |
|---|---|---|---|---|---|
| alfred | ROUGE-L | Future 0.2790 | PV 0.2903 | **Future 0.2924** | k↓ 시 Future |
| clevr_change | ROUGE-L | PV 0.1226 | Future 0.1318 | **Future 0.1442** | k↓ 시 Future 격차 ↑ |
| iedit | ROUGE-L | PV 0.1158 | PV 0.1145 | PV 0.1148 | PV 지배 |
| mmcoqa | ROUGE-L | Hybrid 0.3757 | Hybrid 0.3772 | PV 0.3741 | **Hybrid 유일 우위** |
| multimodalqa | Acc | PV/H 0.7650 | PV/H 0.7600 | PV/H 0.7600 | Future만 열세 |
| scenetransition | Acc | PV/H 0.7750 | PV/H 0.7750 | PV 0.7750 | Future 약세 |
| slidevqa | Acc | tie | PV 0.4750 | PV 0.4750 | PV 근소 우위 |
| spot_the_diff | ROUGE-L | **PV 0.2008** | **PV 0.1949** | **PV 0.1822** | PV 완전 지배 |
| statechange | Acc | tie 0.41 | PV 0.4100 | tie | 미묘 |
| textneedleinahaystack | ROUGE-L | tie | Future 0.0601 | Hybrid 0.0604 | Future/Hybrid 근소 |
| tqa | Acc | tie | Future 0.4750 | Future 0.4750 | Future 근소 |
| webqa | Acc | tie | PV 0.6150 | PV 0.6150 | PV 지배 |
| wikivqa | Acc | PV 0.7150 | tie | tie | PV 근소 |

### 주요 관찰

1. **Accuracy 기반 과제는 pruning 전반에 무감.** 지표 한계 — 비교는 ROUGE-L 기반 generative task에서만 의미 있음.
2. **"k↓ 시 성능↑"이 generative task에서 일관적으로 관찰됨** (clevr_change Future 0.1192→0.1442, alfred Future 0.2790→0.2924).
   - ROUGE-L 길이 편향 + token 중복성이 결합된 것으로 보임. dataset별 부호가 갈림 (spot_the_diff PV는 k↓ 시 ↓).
3. **Future vs PV는 dataset-dependent.**
   - **Future 우위**: 합성/변화 기술, 검색형 generation (clevr_change, alfred, tqa, textneedleinahaystack)
   - **PV 우위**: 자연 이미지 semantic task (spot_the_diff, iedit, webqa, multimodalqa, scenetransition)
4. **Hybrid α=0.5는 거의 중간값** — 3 row만 승리 (mmcoqa k=0.5/0.2, textneedleinahaystack k=0.1). 단일 α 고정으로는 이득 작음.

### 로그
- 체인: `/workspace/zap/artifacts/EXP-20260418-001/v4_eval_chain.log`
- 결과: `/workspace/zap/artifacts/EXP-20260418-001/v4_eval/<dataset>/<mode>_k<kr>/metrics.json`

---

## Phase 4: α Sweep (PV × Future 혼합 비율) — **완료**

### 설정
- α 정의: `s = α · softmax(s_PV) + (1-α) · softmax(s_Future)` (α=1.0 → PV-only, α=0.0 → Future-only)
- **Sweep 범위**: α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (5-point grid). 0.0/0.5/1.0은 Phase 3 결과 재사용, 0.25/0.75만 추가 실행.
- **Dataset 범위**: Phase 3에서 max-min spread ≥ 0.005인 6개 — spot_the_diff, clevr_change, webqa, alfred, iedit, mmcoqa
- 총 36 run (6 ds × 3 k × 2 α), 3-GPU 병렬 ~33분.

### 5-point α grid 결과

| Dataset | Metric | k | α=0.0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 | α* |
|---|---|---|---|---|---|---|---|---|
| spot_the_diff | ROUGE-L | 0.5 | 0.1967 | **0.2016** | 0.1984 | 0.1991 | 0.2008 | 0.25 |
| spot_the_diff | ROUGE-L | 0.2 | 0.1819 | 0.1845 | 0.1866 | 0.1929 | **0.1949** | 1.0 |
| spot_the_diff | ROUGE-L | 0.1 | 0.1730 | 0.1750 | 0.1753 | **0.1832** | 0.1822 | 0.75 |
| clevr_change | ROUGE-L | 0.5 | 0.1192 | **0.1242** | 0.1225 | 0.1231 | 0.1226 | 0.25 |
| clevr_change | ROUGE-L | 0.2 | 0.1318 | **0.1329** | 0.1259 | 0.1300 | 0.1302 | 0.25 |
| clevr_change | ROUGE-L | 0.1 | **0.1442** | 0.1398 | 0.1342 | 0.1352 | 0.1359 | 0.0 |
| webqa | Acc | 0.5 | 0.6100 | 0.6100 | 0.6150 | 0.6150 | 0.6150 | tie |
| webqa | Acc | 0.2 | 0.6000 | 0.6000 | 0.6100 | 0.6150 | 0.6150 | tie |
| webqa | Acc | 0.1 | 0.6000 | 0.6000 | 0.6050 | 0.6100 | **0.6150** | 1.0 |
| alfred | ROUGE-L | 0.5 | **0.2790** | 0.2777 | 0.2787 | 0.2790 | 0.2783 | 0.0 |
| alfred | ROUGE-L | 0.2 | 0.2878 | 0.2871 | 0.2825 | 0.2826 | **0.2903** | 1.0 |
| alfred | ROUGE-L | 0.1 | **0.2924** | 0.2870 | 0.2814 | 0.2824 | 0.2811 | 0.0 |
| iedit | ROUGE-L | 0.5 | 0.1144 | 0.1161 | 0.1154 | **0.1173** | 0.1158 | 0.75 |
| iedit | ROUGE-L | 0.2 | 0.1114 | 0.1109 | 0.1143 | 0.1122 | **0.1145** | 1.0 |
| iedit | ROUGE-L | 0.1 | 0.1107 | 0.1116 | 0.1093 | 0.1106 | **0.1148** | 1.0 |
| mmcoqa | ROUGE-L | 0.5 | 0.3738 | 0.3718 | 0.3757 | **0.3776** | 0.3750 | 0.75 |
| mmcoqa | ROUGE-L | 0.2 | 0.3733 | 0.3763 | 0.3772 | **0.3830** | 0.3723 | 0.75 |
| mmcoqa | ROUGE-L | 0.1 | 0.3712 | **0.3756** | 0.3735 | 0.3753 | 0.3741 | 0.25 |

### 핵심 발견

1. **중간 α* peak (진짜 상호보완) = mmcoqa 1개 dataset.**
   mmcoqa k=0.2에서 α=0.75 (0.3830)는 양 끝(0.3733/0.3723)보다 **+0.010 개선**. 두 signal이 독립 정보를 담는다는 유일한 강한 증거.
2. **나머지 5개는 한쪽 signal이 지배** — α* ∈ {0.0, 1.0}로 몰림.
   - Future 지배: clevr_change, alfred
   - PV 지배: spot_the_diff, iedit, webqa
3. **Keep ratio가 α*를 바꿈.**
   - spot_the_diff: k↓ 시 α* 0.25 → 0.75/1.0 (Future 기여 감소)
   - clevr_change: k↓ 시 α* 0.25 → 0.0 (Future 기여 증가)
   - → **단일 α 고정 불충분**; dataset/k-aware 또는 per-layer α 필요.
4. **Future-only 모드(α=0.0)는 probe 훈련 layer 범위에 제약됨.**
   Future probe는 last 8 layer(24-31)만 의미 있는 가중치; layer 0-23은 layer 31 broadcast. 따라서 Hybrid에서 Future의 "진짜 기여"는 후반부 layer에 집중되어 있을 가능성이 큼 → Phase 5(per-layer α)에서 검증 예정.

### 로그·결과
- 체인: `/workspace/zap/artifacts/EXP-20260418-001/v4_alpha_sweep_chain.log`
- 실행 스크립트: `/workspace/zap/artifacts/EXP-20260418-001/alpha_sweep_chain.sh`
- 결과: `/workspace/zap/artifacts/EXP-20260418-001/v4_alpha_sweep/<ds>/hybrid_a{025,075}_k{0p5,0p2,0p1}/metrics.json`

---

## 주요 설계 결정

### 1. 미학습 레이어 broadcast
Future probe는 last 8 (24-31)만 학습. 하지만 LLaVA KV 캐시는 모든 레이어에서 동일 길이 필요. `selected_layer_indices`로 untrained layer에서 compress 스킵하면 attention mask mismatch 발생.
**해결**: layer 31 가중치를 0-23에 복사 (`_bcast31` 변형). 모든 레이어에서 probe 동작 가능.

### 2. Per-sample softmax MSE loss (B 방법)
- Teacher label과 scale 매칭: label을 sum-to-1로 정규화, prediction도 softmax
- 샘플 경계 유지: 모든 batch 연산은 `(sample_id, layer)` 그룹 단위
- PV/Future가 **같은 loss family** 로 학습되어 비교 공정

### 3. 한 forward pass로 두 label 동시 추출
`collect_unified_teacher_shards.py`는 LLaVA full run (prefill + greedy decode)을 1회 돌려 `y_pv`, `y_future` 모두 GPU에서 바로 계산 후 fp16으로 shard에 저장. 파일 중복·GPU 중복 실행 없음.

### 4. One-pass multi-layer 학습
초기 구현(레이어별 32회 shard 스캔)은 SSD 기준 epoch당 1시간+. `train_unified_probe_onepass.py`에서 PostVision pattern 도입:
```python
for shard in shards:
  for layer in shard:
    pred = MLP_layer(x_layer)
    loss += MSE(...)
  backward()
```
Epoch당 disk scan 1회로 감소 → **~5-6분/epoch** (SSD), **~50분/epoch** (HDD).

### 5. SSD 우선 저장
`/workspace/zap/artifacts/teacher_ssd/` (nvme SSD 500MB/s+)에 shard 저장. HDD(117MB/s) 대비 epoch 시간 ~8배 단축.

---

## 관련 파일

**수집·학습 스크립트**:
- `collect_unified_teacher_shards.py` — 통일 수집
- `train_unified_probe_onepass.py` — one-pass multi-layer 학습 (per-sample softmax MSE, `--teacher pv|future`)

**평가 스크립트**:
- `evaluate_image_teacher_pruning.py` — `--mode probe|future|hybrid` 지원

**Press 클래스** (`kvpress/presses/image_token_press.py`):
- `FutureSupervisedImagePress`
- `HybridImageTeacherPress`
- `selected_layer_indices` 지원 (untrained layer 스킵, broadcast 체크포인트와 병용)

**로그·체크포인트**:
- 체크포인트: `/workspace/zap/ckpts/{postvision_probe_v4_20ep, future_probe_v4_last8_20ep[_bcast31]}`
- Shard: `/workspace/zap/artifacts/teacher_ssd/unified/{textvqa,scienceqa,nlvr2}/`
- 학습 로그: `/workspace/zap/artifacts/EXP-20260418-001/v4_{future,pv}_{train,progress}.log`
- 평가 로그: `/workspace/zap/artifacts/EXP-20260418-001/v4_eval/`
- 과거 로그: `/workspace/zap/artifacts/EXP-20260418-001/logs/archive/` (73개)
