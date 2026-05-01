# Experiment Plan

**ID:** EXP-20260420-001
**Author:** ssoree912
**Date:** 2026-04-20
**Status:** [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned
**Parent:** EXP-20260418-001 (Future-supervised probe + Hybrid α sweep)

---

## 1. 동기 (Motivation)

EXP-20260418-001에서 Hybrid probe (PV + Future)의 단일 α sweep을 완료했으나 다음 3가지 한계가 드러남:

1. **α=0.5 (Phase 3)는 대체로 suboptimal.** 29개 MileBench 중 Hybrid가 PV·Future 어느 한쪽보다 우월한 dataset은 3개(mmcoqa, textneedleinahaystack 일부)에 불과.
2. **5-point α sweep (Phase 4)**에서도 α* ∈ {0.0, 1.0} 양 극단으로 몰림. 중간 peak("진짜 상호보완")은 mmcoqa 단 1 dataset.
3. **Future probe는 last 8 layer(24-31)만 학습됨.** Layer 0-23은 layer 31의 MLP를 broadcast한 것 — 즉 **훈련되지 않은 representation space에 trained MLP를 적용**해서 노이즈를 만들고 있을 가능성.

본 실험은 **Future signal을 실제 훈련된 layer에만 블렌드**하고 나머지 layer에서는 PV만 사용하는 **per-layer α schedule**을 도입하여, Future distill이 성능 향상에 실제로 기여하는지 재검증한다.

Future probe의 학습 범위(layer 24-31)와 추론 시 적용 범위가 일치할 때, Hybrid가 PV-only를 유의미하게 상회한다면 "Future supervision은 유효하나 broadcast noise에 의해 지금까지 효과가 희석됐다"는 가설이 뒷받침된다.

---

## 2. 가설 (Hypothesis)

**H1 (Main)**: Layer 0-23에서 PV-only, Layer 24-31에서만 PV+Future 블렌드를 적용한 per-layer Hybrid는, 동일 α의 전역(global) Hybrid보다 ROUGE-L 기준 평균 성능이 높다.

**H2 (Mechanism)**: per-layer 적용에서 최적 α는 EXP-20260418-001 Phase 4의 전역 α*보다 **더 낮은 값**(즉 Future 비중 큼)에서 발생한다.
- 이유: 전역 적용 시 noise가 섞여 α를 보수적으로(=PV 쪽으로) 당겼을 것.

**H3 (Dataset categorization)**: Future 우세 dataset (clevr_change, alfred)에서는 per-layer α 적용이 큰 이득을 주고, PV 지배 dataset (spot_the_diff, iedit, webqa)에서는 변화가 미미하다.

### Success target
- 6 dataset 중 **3개 이상에서 per-layer 최적 α >= 전역 최적 α 성능** (즉 per-layer가 최소한 타이)
- **1개 이상에서 유의미한 개선** (ROUGE-L +0.005 이상)
- 특히 Future 우세 2 dataset(clevr_change, alfred)에서 per-layer Hybrid가 Future-only보다 **≥ 동등 성능** — Future 유용성의 "깨끗한 증거"

### Minimum viability
- 전역 Hybrid 대비 per-layer Hybrid가 어느 dataset에서든 명백한 열세가 **아닐 것**
- 최소 1개 dataset에서 중간 α* peak 재현

---

## 3. 독립변수 (What we change)

- **α (Late layer 블렌드 비율)**: {0.0, 0.25, 0.5, 0.75} — 4점
  - α=1.0은 전역 PV-only와 동일 → 스킵
- **Future blend layer 범위**: fixed = {24, 25, 26, 27, 28, 29, 30, 31} (Future probe 실제 훈련 범위)
  - Ablation으로 {28-31} (last 4)만 블렌드하는 variant 추가 검토 가능하나 일단 제외

---

## 4. 종속변수 (What we measure)

### 주요 metric
- **ROUGE-L** (generative task: clevr_change, alfred, iedit, mmcoqa, spot_the_diff)
- **Accuracy** (choice task: webqa)

### 보조 metric
- Best α* 위치 (per-layer vs 전역 최적 α와의 차이)
- 전역 Hybrid (EXP-20260418-001 Phase 4 결과) 대비 delta

### 비교 축
- per-layer Hybrid (본 실험) vs 전역 Hybrid (Phase 4) vs PV-only (α=1.0) vs Future-only (α=0.0)
- keep_ratio별 변화 추이

---

## 5. 고정 조건 (What stays the same)

- **모델**: `/workspace/zap/ckpts/llava-1.5-7b-hf` (LLaVA-1.5-7B)
- **PV probe**: `/workspace/zap/ckpts/postvision_probe_v4_20ep` (32 layer 전체 학습)
- **Future probe**: `/workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31` (last 8 학습 + broadcast)
- **Dataset**: EXP-20260418-001 Phase 4와 동일 6개 — spot_the_diff, clevr_change, webqa, alfred, iedit, mmcoqa
- **Keep ratio**: {0.5, 0.2, 0.1}
- **Prompt style**: `look_milebench`
- **Attention impl**: `eager` (output_attentions=True 필요)
- **Truncation**: `--truncate_like_lookm` (OOM 방지)
- **seed**: 42 (probe 학습), evaluation은 seed 없음
- **하드웨어**: RTX 4090 × 3, CUDA 12.1 (container)

---

## 6. 베이스라인

| 비교 대상 | 출처 | 값 |
|---|---|---|
| PV-only | Phase 3 v4_eval/<ds>/pv_k<k>/ | α=1.0 상등 |
| Future-only | Phase 3 v4_eval/<ds>/future_k<k>/ | α=0.0 상등 |
| 전역 Hybrid α=0.5 | Phase 3 v4_eval/<ds>/hybrid_k<k>/ | 기존 |
| 전역 Hybrid α=0.25 | Phase 4 v4_alpha_sweep/<ds>/hybrid_a025_k<k>/ | 기존 |
| 전역 Hybrid α=0.75 | Phase 4 v4_alpha_sweep/<ds>/hybrid_a075_k<k>/ | 기존 |

per-layer sweep 본 실험 결과와 위 5개 baseline을 5점 grid로 비교.

---

## 7. 예상 결과

### 정성적 예측

- **clevr_change (Future 우세)**: per-layer α=0.0~0.25에서 전역 α=0.0 (Future-only) 대비 **약간 개선 또는 동등**.
  - 이유: Future-only는 layer 0-23에서도 noisy future score로 pruning 결정. per-layer에서 그걸 PV로 교체하면 초반 layer 결정이 안정화.
- **alfred**: clevr_change와 유사 패턴. k=0.1에서 per-layer α=0.0이 Future-only(0.2924)보다 +0.003~0.010 이득 기대.
- **mmcoqa**: 이미 전역 α=0.75에서 peak(0.3830)가 있었음. per-layer에서 α=0.5~0.75가 추가 개선할 여지.
- **spot_the_diff / iedit / webqa**: PV 지배. per-layer α=0.75에서 미미한 상승 또는 동등. α=0.0~0.25에서는 Future noise 차단 효과로 기존 Future-only보다 개선.

### 정량적 예측

- 6 dataset × 3 k × 4 α = **72 runs**
- 최소 4개 dataset에서 per-layer의 best α가 **전역 best α와 동등 이상**
- 최소 1개 dataset에서 per-layer best가 **전역 best + 0.005 이상**

---

## 8. 판단 기준 (Success Criteria)

### Primary
- **H1 지지**: per-layer Hybrid best α가 6 dataset 평균 ROUGE-L/Accuracy 기준으로 전역 Hybrid best α와 **동등 이상**.

### Secondary
- **H2 지지**: per-layer 최적 α 분포가 전역 최적 α보다 **Future 쪽(낮은 α)으로 치우침**.
- **H3 지지**: Future 우세 2 dataset에서 per-layer Hybrid가 Future-only 대비 **≥ 동등 성능** (broadcast noise 제거 효과).

### Failure mode
- 6 dataset 모두에서 per-layer와 전역이 사실상 구별되지 않으면 (Δ < 0.002):
  - 결론: "Future의 early-layer broadcast는 dominant한 영향이 아님. Phase 6 (all-layer Future 재학습)은 보류하고 다른 방향 탐색 필요."
- 2개 이상 dataset에서 per-layer가 유의미하게 나쁘면:
  - 결론: "Future signal은 late layer 단독으로는 부족; PV와의 전역 혼합 자체가 유효한 regularization이었을 가능성."

---

## 9. 이 실험으로 증명할 수 없는 것

- **"Future probe를 all-32-layer로 재학습하면 더 좋아진다"** — 본 실험은 late-8 학습된 probe를 그대로 사용. all-layer 재학습은 Phase 6 별도 실험 필요.
- **"per-layer α가 최선의 설계"** — linear ramp, per-layer 개별 α 등 더 정교한 schedule은 범위 밖.
- **"Accuracy dataset에서 방법 차이"** — EXP-20260418-001 Phase 3에서 확인된 것처럼 accuracy 지표는 pruning에 무감. 본 실험도 동일 한계 지님.
- **"MileBench 전체 일반화"** — Phase 4 6 dataset 기준. 나머지 23개는 per-layer α가 유의미한 차이를 만들지 않을 가능성이 크지만 확정 불가.

---

## 10. 예상 런타임 / 리소스

- **GPU**: RTX 4090 × 3 (cuda:0, cuda:1, cuda:2), container 재시작 후 CUDA init 정상화 필요
- **Run 수**: 6 dataset × 3 keep_ratio × 4 α = **72 runs**
- **Run당 시간**: ~3 min (기존 v4_eval / alpha_sweep 동일 기준)
- **3-wide 병렬 총 시간**: 72 / 3 × 3 min ≈ **72-80 min**
- **디스크**: 각 run당 metrics.json + pred.jsonl ≈ 1 MB, 총 ~70 MB 수준 (기존 구조와 동일)
- **출력 경로**: `/workspace/zap/artifacts/EXP-20260420-001/v5_perlayer_sweep/<ds>/hybrid_a{000,025,050,075}_L24-31_k{0p5,0p2,0p1}/`

---

## 11. 구현 (Implementation Details)

### 11.1 Press 코드 수정 (`kvpress/presses/image_token_press.py`)

`HybridImageTeacherPress`에 새 필드 `future_blend_layers: tuple[int, ...]` 추가. `score_image_tokens()`에서:

```python
if self.future_blend_layers and module.layer_idx not in self.future_blend_layers:
    effective_alpha = 1.0  # PV-only at this layer
else:
    effective_alpha = self.alpha
s_hybrid = effective_alpha * s_pv_norm + (1 - effective_alpha) * s_fu_norm
```

KV cache length는 모든 layer에서 동일하게 `keep_ratio × N` 유지 (어느 토큰을 남기느냐만 layer별로 다름). 기존 broadcast 구조와 호환.

### 11.2 CLI 플래그 (`evaluate_image_teacher_pruning.py`)

```bash
--future_blend_layers 24 25 26 27 28 29 30 31
```

`hybrid` 모드 전용. 생략 시 모든 layer에 블렌드 (legacy).

### 11.3 Sweep 스크립트 (`artifacts/EXP-20260420-001/perlayer_sweep_chain.sh`)

- 3-wide 병렬
- 동일 chain_log 포맷 (기존 `alpha_sweep_chain.sh` 구조 재사용)
- Container 재시작 후 실행
- 각 run에 `run_in_background`-style 방어: rc 체크 후 다음 wave 진행

### 11.4 재현 커맨드 예시

```bash
python evaluate_image_teacher_pruning.py \
  --mode hybrid \
  --dataset_path /workspace/zap/data/MileBench/CLEVR-Change/CLEVR-Change.json \
  --image_root /workspace/zap/data/MileBench/CLEVR-Change/images \
  --image_column images_path \
  --output_dir /workspace/zap/artifacts/EXP-20260420-001/v5_perlayer_sweep/clevr_change/hybrid_a025_L24-31_k0p1 \
  --implementation_model_name /workspace/zap/ckpts/llava-1.5-7b-hf \
  --probe_model_name /workspace/zap/ckpts/postvision_probe_v4_20ep \
  --future_probe_name /workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31 \
  --alpha 0.25 \
  --future_blend_layers 24 25 26 27 28 29 30 31 \
  --image_keep_ratio 0.1 \
  --prompt_style look_milebench \
  --attn_implementation eager \
  --truncate_like_lookm \
  --look_dataset_name DocVQA --look_model_name zap_docvqa \
  --look_result_root /workspace/zap/artifacts/combine_prob \
  --device cuda:0
```

---

## 12. 리스크와 완화책

| Risk | 영향 | 완화책 |
|---|---|---|
| CUDA driver 상태 깨짐 (이전 alpha sweep 후 발생) | 실험 자체 불가 | 컨테이너 재시작 후 smoke test 먼저. sweep 전 `nvidia-smi` + torch init 확인. |
| 3-wide 병렬 context stress로 UVM fault 재발 | sweep 중단 | run 사이 3-5초 sleep 삽입, 또는 sequential fallback 준비. `CUDA_MODULE_LOADING=LAZY` 환경변수 설정. |
| per-layer도 전역과 차이 없음 | 가설 기각, 연구 방향 전환 필요 | H2 검증만으로도 가치 있음 (broadcast noise가 dominant가 아니라는 진단). 바로 Phase 6 all-layer Future 재학습으로 이동. |
| Future probe weight가 layer 24-31 내에서도 spatial 편향 있음 | 결과 해석 혼란 | 분석 단계에서 per-layer keep mask 분포 시각화 검토. |
| ROUGE-L 길이 편향이 결과를 왜곡 | 해석 오류 | 생성 길이 분포 동시 기록 (Phase 3와 비교). keep_ratio별 generated length avg 측정. |

---

## 13. 분석 계획

1. **Table**: 6 dataset × 3 k × (α=0.0/0.25/0.5/0.75 per-layer + α=0.0/0.25/0.5/0.75/1.0 전역) 결과 합친 grid.
2. **Best α*** per-layer vs 전역 위치 비교 (H2 검증).
3. **Per-dataset delta**: per-layer best – 전역 best (H1 검증).
4. **생성 길이 분포**: ROUGE-L 변화가 진짜 content 개선인지 length artifact인지 판별.
5. **Phase 6 go/no-go 판단**: per-layer에서 유의미한 개선 있으면 all-layer Future 재학습 계획 수립, 없으면 다른 방향.

---

## 14. 다음 실험 (미리 설계된 Phase 6 조건부)

**Phase 6 (조건부)**: all-32-layer Future probe 재학습
- 본 실험 성공(H1/H2 지지) 시에만 진행
- 기존 unified shard 재사용 (`/workspace/zap/artifacts/teacher_ssd/unified/`)
- `train_unified_probe_onepass.py --teacher future --selected_layers 0..31 --mlp_max_epochs 20`
- 예상 소요: 학습 2-3시간, 평가 72 run (또는 전 MileBench 29 dataset 재평가)
- 판단 기준: 재학습 Future probe의 val Spearman이 last-8 학습 대비 유의미한 개선이 있으면 채택.

---

## 15. Checklist

실험 시작 전 확인:
- [ ] 컨테이너 재시작 완료
- [ ] `torch.cuda.is_available() == True` smoke test 통과
- [ ] `HybridImageTeacherPress.future_blend_layers` 필드 추가 완료
- [ ] `evaluate_image_teacher_pruning.py --future_blend_layers` 플래그 작동 확인
- [ ] Smoke test (`--limit 5`) 1회 통과
- [ ] Sweep 스크립트 작성 및 실행권한 부여
- [ ] 6 dataset 경로/image_root 정확한지 확인 (run_config.json 참조)

실험 후 확인:
- [ ] 72 runs 전부 `rc=0`
- [ ] `metrics.json`에 `look_eval.ROUGE-L` or `Accuracy` 정상 기록
- [ ] 전역 sweep과 per-layer sweep 결과 통합 테이블 생성
- [ ] results.md 업데이트 (Phase 5 섹션 추가)
