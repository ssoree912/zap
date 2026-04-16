# EXP-20260415-001: 시각화 파이프라인 구현 + 전체 실험 결과 최종 정리

**ID**: EXP-20260415-001  
**Completed**: 2026-04-15  
**Status**: Done

---

## 1. 구현 완료 사항

### 1.1 VizCapture 시각화 파이프라인

| 파일 | 변경 내용 |
|---|---|
| `kvpress/presses/image_token_press.py` | `VizCapture` 데이터클래스 추가, `ImageTokenTopKPress`에 `_viz_capture` 필드 및 `attach_viz_capture()` / `detach_viz_capture()` 메서드 추가, `compress()` 내 zero-overhead 캡처 훅 삽입 |
| `evaluate_image_teacher_pruning.py` | `--save_viz_for_first_n N` 옵션 추가, `--viz_output_dir` 옵션 추가 |
| `scripts/render_eviction_visualizations.py` | 신규 생성 — per-sample 시각화 / dataset aggregate 렌더링 |

#### VizCapture 구조

```
VizCapture
├── layer_scores : List[(n_image,) float32]  — 레이어별 amax 집계 스코어
├── keep_masks   : List[(n_image,) bool]     — 레이어별 keep/evict 결정 (kv-head union)
└── n_image_keep : List[int]                 — 레이어별 유지 token 수
```

#### 캡처 훅 위치 (image_token_press.py, compress() 내부)

```
score_tensor = score_image_tokens(...)         # 메서드별 스코어 계산
score_tensor = _aggregate_scores_to_kv_heads(...)  # KV head 집계
...                                            # top-k 선택 완료 후
if self._viz_capture is not None:              # ← 여기서 캡처 (pruning에 영향 없음)
    agg_score = score_tensor[0].amax(dim=0)    # (n_image,) — head amax
    keep_mask  = [forced | top-k 선택된 indices]
    self._viz_capture.record(...)
gather → keys, values 반환
```

#### 렌더링 출력 구조

```
{out_dir}/
├── per_sample/
│   ├── 0000_{sample_id}.png   ← 이미지별 패널: Original | Score | KeepMask | [Diff]
│   └── ...
└── aggregate/
    └── mean_heatmaps.png       ← 데이터셋 평균 score heatmap + std + [correlation]
```

---

## 2. 전체 실험 결과 통합 정리

### 2.1 Phase별 요약

| Phase | 실험 ID | 핵심 질문 | 결론 |
|---|---|---|---|
| Phase 0 | EXP-20260410-001 | 어떤 teacher score가 최적인가? | **att_only_postvision** (배포 가능 중 최고) |
| Phase 1+2 | EXP-20260410-001, EXP-20260412-002 | Eviction scope vs Scoring signal — 무엇이 중요한가? | **Image-only scope +9.6%p** >> Scoring signal +0.7%p |
| Phase 3 | EXP-20260410-001 | Probe distillation이 oracle을 근사하는가? | ✓ 평균 -0.9%p, 3/4 데이터셋에서 probe ≥ oracle |
| Phase 4 | EXP-20260412-001 | MileBench 29개 전체에서 LOOK-M 대비 경쟁력? | **probe 17W / LOOK-M 10W / 2T** |
| Phase 5 | EXP-20260412-003 | Position forced-keep이 필요한가? | 불필요 — probe score가 이미 위치 중요도 내재 |
| Phase 6 | EXP-20260412-004 | Ratio 무감각성 확인 (r=0.05도 충분한가?) | ✓ r=0.05/0.10/0.20 모두 17W/10L/2T 동일 |
| Phase 7 | EXP-20260415-001 | 실측 효율성 (Prefill/TBT/메모리)? | probe: TBT **2.63×** ↑, Prefill **39%** ↓ vs LOOK-M |
| Phase 7C | EXP-20260415-001 | Oracle 효율성 (on-the-fly 2-pass)? | oracle prefill 856ms (Pass2만) / 공정 비교 별도 필요 |

---

### 2.2 핵심 성능 결과 — MileBench 전체 (29 datasets, r=0.20, n=200/dataset)

| Dataset | Metric | **Probe** | LOOK-M | Winner |
|---|---|---:|---:|---|
| actionlocalization | Acc | 0.265 | 0.220 | **probe** |
| actionprediction | Acc | 0.505 | 0.530 | look_m |
| actionsequence | Acc | 0.435 | 0.440 | look_m |
| alfred | ROUGE-L | **0.283** | 0.165 | **probe** |
| characterorder | Acc | **0.430** | 0.310 | **probe** |
| clevr_change | ROUGE-L | 0.142 | **0.179** | look_m |
| counterfactualinference | Acc | **0.325** | 0.300 | **probe** |
| docvqa | Acc | 0.460 | **0.470** | look_m |
| egocentricnavigation | Acc | **0.305** | 0.285 | **probe** |
| gpr1200 | Acc | **0.092** | 0.042 | **probe** |
| iedit | ROUGE-L | **0.110** | 0.039 | **probe** |
| imageneedleinahaystack | ROUGE-L | 0.000 | 0.000 | tie |
| mmcoqa | ROUGE-L | 0.325 | **0.330** | look_m |
| movingattribute | Acc | **0.495** | 0.490 | **probe** |
| movingdirection | Acc | 0.285 | **0.325** | look_m |
| multimodalqa | Acc | **0.745** | 0.655 | **probe** |
| nuscenes | Acc | **0.650** | 0.615 | **probe** |
| objectexistence | Acc | 0.480 | **0.510** | look_m |
| objectinteraction | Acc | **0.500** | 0.485 | **probe** |
| objectshuffle | Acc | 0.345 | 0.345 | tie |
| ocr_vqa | Acc | **0.100** | 0.090 | **probe** |
| scenetransition | Acc | 0.625 | **0.665** | look_m |
| slidevqa | Acc | **0.520** | 0.460 | **probe** |
| spot_the_diff | ROUGE-L | **0.195** | 0.161 | **probe** |
| statechange | Acc | **0.410** | 0.325 | **probe** |
| textneedleinahaystack | ROUGE-L | 0.061 | **0.100** | look_m |
| tqa | Acc | 0.390 | **0.410** | look_m |
| webqa | Acc | **0.620** | 0.565 | **probe** |
| wikivqa | Acc | **0.702** | 0.620 | **probe** |
| **집계** | | **mean=0.372** | mean=0.349 | **probe 17W / 10L / 2T** |

> **avg Δ = +0.023** (probe가 LOOK-M보다 평균 2.3%p 높음)

---

### 2.3 핵심 효율성 결과 — Common 69 샘플 (공정 비교, r_eff=0.200 통일)

| 지표 | full_cache | **Probe (Ours)** | LOOK-M | Oracle |
|---|---:|---:|---:|---:|
| **n** | 69 | 69 | 69 | 69 |
| **Prefill latency** | 496 ms | **673 ms** | 1,597 ms | 856 ms |
| **TBT (ms/token)** | 29.0 ms | **27.6 ms** | 74.0 ms | 27.4 ms |
| **TTFT** | — | — | — | — |
| **Peak GPU mem** | 15.04 GiB | 13.79 GiB | 14.60 GiB | 13.67 GiB |
| **KV cache** | 0.887 GiB | **0.129 GiB** | 0.182 GiB | 0.129 GiB |
| **r_eff_prompt** | 1.000 | **0.200** | 0.200 | 0.200 |

**핵심 수치 (probe vs LOOK-M, matched 77 샘플 기준)**:
- TBT: **2.63× 빠름** (27.9ms vs 73.5ms)
- Prefill: **39% 단축** (657ms vs 1,080ms)
- KV cache: LOOK-M 대비 유사 수준, full_cache 대비 **72% 절감**

> **Oracle 주의**: 현재 저장된 oracle prefill(856ms)은 Pass 2(압축 generate)만 측정한 값.  
> Pass 1(teacher 추출, eager attention) 비용 ~350ms 추가 시 실제 총 비용 ≈ 1,200ms.  
> oracle은 실제 배포 불가 (매 샘플마다 full forward 필요).

---

### 2.4 Common 69 성능 결과 (3-way, r=0.20)

| Dataset | Metric | LOOK-M | **Probe** | Oracle | Probe vs LOOK-M |
|---|---|---:|---:|---:|---|
| ALFRED | ROUGE-L | 0.165 | **0.283** | 0.271 | **+0.118** ↑ |
| CLEVR-Change | ROUGE-L | **0.179** | 0.142 | 0.138 | -0.037 ↓ |
| CounterfactualInference | Acc | 0.300 | 0.325 | **0.345** | **+0.025** ↑ |
| DocVQA | Acc | **0.470** | 0.460 | 0.455 | -0.010 ↓ |
| IEdit | ROUGE-L | 0.039 | **0.110** | 0.110 | **+0.071** ↑ |
| ImageNeedleInAHaystack | ROUGE-L | 0.000 | 0.000 | 0.000 | 0.000 |
| MMCoQA | ROUGE-L | **0.330** | 0.325 | 0.335 | -0.005 ↓ |
| MovingDirection | Acc | **0.325** | 0.285 | 0.295 | -0.040 ↓ |
| OCR-VQA | Acc | 0.090 | **0.100** | 0.100 | **+0.010** ↑ |
| ObjectExistence | Acc | **0.510** | 0.480 | 0.475 | -0.030 ↓ |
| SlideVQA | Acc | 0.460 | **0.520** | 0.520 | **+0.060** ↑ |
| Spot-the-Diff | ROUGE-L | 0.161 | **0.195** | 0.202 | **+0.034** ↑ |
| TQA | Acc | 0.410 | 0.390 | **0.420** | -0.020 ↓ |
| TextNeedleInAHaystack | ROUGE-L | **0.100** | 0.061 | 0.043 | -0.039 ↓ |
| WebQA | Acc | 0.565 | **0.620** | **0.620** | **+0.055** ↑ |
| WikiVQA | Acc | 0.620 | **0.702** | 0.699 | **+0.082** ↑ |
| nuscenes | Acc | 0.615 | **0.650** | 0.640 | **+0.035** ↑ |
| **집계 (17개)** | | 0.328 | **0.359** | **0.360** | **probe 9W / 7L / 1T** |

> weighted mean (공통 69 샘플 수 기준): LOOK-M 0.328 / Probe 0.359 / Oracle 0.360

---

### 2.5 2×2 Ablation 결과 (Scoring × Eviction Scope, r=0.20)

|  | image-only eviction | all-token eviction |
|---|---|---|
| **H2O score** | Cell A: 0.228 avg | LOOK-M: ~0.200 ref |
| **att_only_pv score** | **Ours: 0.235 avg** ← best | Cell B: 0.139 avg |

**Score 기여 (A vs Ours)**: +0.007 (+0.7%p 평균)  
**Scope 기여 (B vs Ours)**: +0.096 (+9.6%p 평균) ← **결정적**

결론: image-only eviction이 가장 중요한 설계 결정.

---

### 2.6 Ratio 무감각성 (r=0.05 vs r=0.20 vs LOOK-M r=0.20)

| 비교 | W/L/T | avg Δ |
|---|---|---|
| probe r=0.05 vs LOOK-M r=0.20 | **17/10/2** | **+0.021** |
| probe r=0.10 vs LOOK-M r=0.20 | **17/10/2** | **+0.022** |
| probe r=0.20 vs LOOK-M r=0.20 | 17/10/2 | +0.022 |

→ token budget을 1/4로 줄여도 성능 유지. 효율성 클레임의 핵심 근거.

---

## 3. 시각화 실행 가이드

### 3.1 VizCapture 데이터 생성

```bash
# Probe 시각화 데이터 (첫 20 샘플)
conda run -n kv --no-capture-output \
  python evaluate_image_teacher_pruning.py \
    --mode probe \
    --probe_model_name /workspace/hd/artifacts/zap/probe_model_scienceqa \
    --dataset_path /workspace/hd/data/MileBench/Spot-the-Diff/Spot-the-Diff.json \
    --image_root /workspace/hd/data/MileBench/Spot-the-Diff/images \
    --look_dataset_name Spot-the-Diff \
    --total_keep_ratio 0.20 \
    --truncate_like_lookm \
    --save_viz_for_first_n 20 \
    --viz_output_dir /workspace/hd/artifacts/viz/probe_spotdiff \
    --output_dir /workspace/hd/artifacts/viz/probe_spotdiff_out \
    --device cuda:0

# Oracle 시각화 데이터 (같은 20 샘플, on-the-fly)
conda run -n kv --no-capture-output \
  python evaluate_image_teacher_pruning.py \
    --mode oracle_onthefly \
    --dataset_path /workspace/hd/data/MileBench/Spot-the-Diff/Spot-the-Diff.json \
    --image_root /workspace/hd/data/MileBench/Spot-the-Diff/images \
    --look_dataset_name Spot-the-Diff \
    --total_keep_ratio 0.20 \
    --truncate_like_lookm \
    --save_viz_for_first_n 20 \
    --viz_output_dir /workspace/hd/artifacts/viz/oracle_spotdiff \
    --output_dir /workspace/hd/artifacts/viz/oracle_spotdiff_out \
    --device cuda:0
```

### 3.2 시각화 렌더링

```bash
# 단일 방법 (probe만)
python scripts/render_eviction_visualizations.py \
  --viz_dir /workspace/hd/artifacts/viz/probe_spotdiff \
  --out_dir /workspace/hd/artifacts/viz/figures_spotdiff \
  --label_a probe

# Probe vs Oracle 비교
python scripts/render_eviction_visualizations.py \
  --viz_dir /workspace/hd/artifacts/viz/probe_spotdiff \
  --compare_viz_dir /workspace/hd/artifacts/viz/oracle_spotdiff \
  --out_dir /workspace/hd/artifacts/viz/figures_spotdiff_compare \
  --label_a probe --label_b oracle
```

---

## 3.5 시각화 실행 결과 — Spot-the-Diff 3-way 비교 (probe vs oracle vs H2O)

### 실행 조건

- **데이터셋**: Spot-the-Diff (MileBench), 첫 10 샘플
- **비교 방법**: probe / oracle (on-the-fly att_only_postvision) / H2O image-only
- **설정**: `total_keep_ratio=0.20`, `truncate_like_lookm`
- **GPU**: cuda:1 (oracle/H2O), cuda:0 (probe)
- **아티팩트**:
  - 데이터: `/workspace/hd/artifacts/viz/{probe,oracle,h2o}_spotdiff/`
  - 그림: `/workspace/hd/artifacts/viz/figures_3way/`

### Score Correlation (Pearson r, 10 샘플 × 2 이미지 = 20 image block 기준)

| 비교 쌍 | mean Pearson r | 해석 |
|---|---|---|
| probe vs oracle | **0.284** | 낮음 — 서로 다른 region 선택 |
| probe vs H2O | **0.367** | 중간 — probe가 oracle보다 H2O에 더 가까움 |

### 시각적 관찰

**Per-sample 그림** (컬럼 순서: Original | Score-probe | Keep-probe | Score-oracle | Keep-oracle | Diff(probe−oracle) | Score-H2O | Keep-H2O | Diff(probe−H2O)):

| 관찰 | 내용 |
|---|---|
| Probe score | 공간적으로 뚜렷한 hot region 존재 (globally discriminative) |
| Oracle score | 전반적으로 어두움 — 샘플마다 중요 영역이 달라 평균에서 상쇄됨 |
| H2O score | Oracle보다 분명하지만 probe보다는 덜 뚜렷한 패턴 |
| Keep mask (probe) | 선택된 token이 클러스터 형태로 집중 |
| Keep mask (H2O) | 더 분산된 선택 패턴 |
| Diff(probe−oracle) | 대부분 빨간색 → probe가 대부분 영역에서 더 높은 score 부여 |
| Diff(probe−H2O) | 역시 빨간색 우세, 일부 파란색 patch 존재 → 일부 영역은 H2O가 높음 |

**Aggregate 그림** (10 샘플 평균):
- **Mean score (probe)**: 좌측/중앙에 일관된 bright 패턴 — 샘플 간 공통된 spatial preference 존재
- **Score Std (probe)**: 중앙 좌측 patch가 샘플마다 가장 많이 변동
- **Mean score (oracle/H2O)**: 대부분 어두움 — 샘플별로 중요 region이 달라서 평균에서 희미해짐

### 해석 및 논의

1. **Probe와 oracle의 낮은 correlation (r=0.28)**은 probe가 oracle의 attention을 그대로 복제하지 않고 자체적인 선택 전략을 학습했음을 의미. ScienceQA 학습이 task-generic한 중요도를 학습한 것으로 해석.

2. **Probe와 H2O의 더 높은 correlation (r=0.37)**은 probe score가 attention 누적 패턴(heavy-hitter)과 부분적으로 일치함을 시사. 즉, 자주 attended되는 patch를 probe도 중요하다고 판단하는 경향.

3. **성능이 유사함에도 다른 patch를 선택**한다는 것은, 다양한 patch selection 전략이 downstream task에서 유사한 정보를 제공할 수 있음을 보여줌 — KV cache pruning에서 equifinality(등종결성)의 증거.

4. **논문 활용**: probe score heatmap의 뚜렷한 spatial pattern이 oracle/H2O보다 interpretable → figure로서 설득력 있음.

### Caveat

- 10 샘플만 사용 → statistical significance 낮음
- H2O는 decode-time에 누적 attention을 사용하지만 여기서는 prefill attention만 사용 (생성 첫 step) → 실제 LOOK-M과는 다름
- Spot-the-Diff에 특화된 결과 — 다른 데이터셋에서는 패턴이 다를 수 있음

---

## 4. 남은 실험 및 TODO

| 우선순위 | 항목 | 상태 |
|---|---|---|
| 🔴 High | 시각화 실행 (Spot-the-Diff, IEdit 위주, probe vs oracle) | ⬜ 미실행 |
| 🔴 High | FastV 비교 실험 | ⬜ 미구현 |
| 🟡 Mid | SparseVLM 비교 실험 | ⬜ 미구현 |
| 🟡 Mid | 학습 데이터 다양화 (ScienceQA + NExT-QA + TextVQA + NLVR2) | ⬜ 미실행 |
| 🟢 Low | Oracle 2-pass 공정 비교 (Pass1+Pass2 합산 prefill) | ⬜ 재실행 필요 |
| 🟢 Low | H2O ablation (image-only) 효율성 측정 | ⬜ 미실행 |

---

## 5. 아티팩트 위치

| 항목 | 경로 |
|---|---|
| 시각화 코드 | `kvpress/presses/image_token_press.py` (VizCapture), `scripts/render_eviction_visualizations.py` |
| 효율성 CSV (공통 69) | `/workspace/hd/artifacts/probe_global/efficiency_all_datasets/per_sample_common69_4methods.csv` |
| 성능 CSV (공통 69, 17 datasets) | `/workspace/hd/artifacts/probe_global/efficiency_all_datasets/performance_common69_datasets_keep0p20.csv` |
| MileBench 전체 성능 CSV (29 datasets) | `/workspace/hd/artifacts/probe_global/probe_vs_lookm_r020_truncated.csv` |
| 전체 ablation 보고서 | `experiments/ablation_report.md` |
