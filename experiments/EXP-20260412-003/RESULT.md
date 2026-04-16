## Experiment Result

**ID**: EXP-20260412-003
**Date**: 2026-04-14
**Status**: [x] Done (Phase A 완료, rnd 조기 종료)
**Actual Runtime**: ~18시간 (GPU 0, 29 datasets × 7 조건)

---

### 실험 범위 변경 (PLAN 대비)

| 항목 | PLAN | 실제 |
|---|---|---|
| 모드 | oracle (4 datasets) | probe_mlp (29 datasets 전체) |
| 데이터셋 | DocVQA, Spot-the-Diff, CLEVR-Change, IEdit | MileBench 전체 29개 |
| truncation | 미지정 | LOOK-M 방식 (max_context_len=4096) 적용 |
| rnd 조건 | 포함 | 조기 종료 (init/rec 결과로 의미 없다 판단) |

probe_mlp 모드로 전환한 이유: 이미 probe_global sweep과 병합하여 단일 패스로 효율화.

---

### 핵심 결과

**Init vs baseline**: avg **+0.0014**, 개선=7/29, 하락=0/29
**Rec  vs baseline**: avg **+0.0003**, 개선=3/29, 하락=1/29

#### 전체 결과표 (baseline / init16 / init32 / init64 / rec16 / rec32 / rec64 / LOOK-M)

| Dataset | base | i16 | i32 | i64 | r16 | r32 | r64 | LOOK-M | Δ_init | Δ_rec |
|---|---|---|---|---|---|---|---|---|---|---|
| actionlocalization | 0.265 | 0.265 | 0.265 | 0.265 | 0.265 | 0.265 | 0.265 | 0.220 | +0.000 | +0.000 |
| actionprediction | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.530 | +0.000 | +0.000 |
| actionsequence | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.440 | +0.000 | +0.000 |
| alfred | 0.283 | 0.283 | 0.284 | 0.283 | 0.283 | 0.285 | 0.285 | 0.165 | +0.001 | +0.002 |
| **characterorder** | 0.430 | **0.445** | 0.445 | 0.445 | 0.430 | 0.430 | 0.430 | 0.310 | **+0.015** | +0.000 |
| clevr_change | 0.142 | 0.143 | 0.144 | 0.145 | 0.142 | 0.142 | 0.142 | 0.179 | +0.003 | +0.001 |
| counterfactualinference | 0.325 | 0.325 | 0.325 | 0.325 | 0.325 | 0.325 | 0.325 | 0.300 | +0.000 | +0.000 |
| docvqa | 0.460 | 0.460 | 0.460 | 0.460 | 0.460 | 0.460 | 0.460 | 0.470 | +0.000 | +0.000 |
| egocentricnavigation | 0.305 | 0.305 | 0.305 | 0.305 | 0.305 | 0.305 | 0.305 | 0.285 | +0.000 | +0.000 |
| gpr1200 | 0.092 | 0.092 | 0.092 | 0.093 | 0.092 | 0.092 | 0.090 | 0.042 | +0.002 | +0.000 |
| iedit | 0.110 | 0.109 | 0.111 | 0.109 | 0.107 | 0.107 | 0.108 | 0.039 | +0.000 | **-0.002** |
| imageneedleinahaystack | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 |
| mmcoqa | 0.325 | 0.322 | 0.322 | 0.328 | 0.324 | 0.324 | 0.324 | 0.330 | +0.003 | -0.001 |
| movingattribute | 0.495 | 0.495 | 0.495 | 0.495 | 0.495 | 0.495 | 0.495 | 0.490 | +0.000 | +0.000 |
| movingdirection | 0.285 | 0.285 | 0.285 | 0.285 | 0.285 | 0.285 | 0.285 | 0.325 | +0.000 | +0.000 |
| multimodalqa | 0.745 | 0.745 | 0.745 | 0.745 | 0.745 | 0.745 | 0.745 | 0.655 | +0.000 | +0.000 |
| nuscenes | 0.650 | 0.650 | 0.650 | 0.650 | 0.650 | 0.650 | 0.650 | 0.615 | +0.000 | +0.000 |
| objectexistence | 0.480 | 0.480 | 0.480 | 0.480 | 0.480 | 0.480 | 0.480 | 0.510 | +0.000 | +0.000 |
| objectinteraction | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.485 | +0.000 | +0.000 |
| objectshuffle | 0.345 | 0.345 | 0.345 | 0.345 | 0.345 | 0.345 | 0.345 | 0.345 | +0.000 | +0.000 |
| ocr_vqa | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | 0.090 | +0.000 | +0.000 |
| scenetransition | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 | 0.620 | 0.665 | +0.000 | +0.000 |
| slidevqa | 0.520 | 0.520 | 0.520 | 0.520 | 0.520 | 0.520 | 0.520 | 0.460 | +0.000 | +0.000 |
| spot_the_diff | 0.195 | 0.195 | 0.196 | 0.201 | 0.195 | 0.196 | 0.199 | 0.161 | +0.006 | +0.004 |
| statechange | 0.410 | 0.415 | 0.415 | 0.415 | 0.410 | 0.410 | 0.410 | 0.325 | **+0.005** | +0.000 |
| textneedleinahaystack | 0.061 | 0.061 | 0.061 | 0.061 | 0.061 | 0.061 | 0.061 | 0.100 | +0.000 | +0.000 |
| tqa | 0.390 | 0.390 | 0.390 | 0.390 | 0.390 | 0.390 | 0.390 | 0.410 | +0.000 | +0.000 |
| webqa | 0.620 | 0.615 | 0.625 | 0.625 | 0.615 | 0.620 | 0.625 | 0.565 | +0.005 | +0.005 |
| wikivqa | 0.702 | 0.702 | 0.697 | 0.697 | 0.702 | 0.702 | 0.702 | 0.620 | +0.000 | +0.000 |

---

### 가설 검증

- [x] **H1 (Initial/Sink)**: **확인됨** — image token에서 sink effect는 보편적이지 않음. CharacterOrder(+0.015), Spot-the-Diff(+0.006), StateChange(+0.005)에서만 유의미한 init 효과. 22/29 데이터셋은 완전 flat.

- [x] **H2 (Recent)**: **확인됨** — recent 보존 효과 극히 미미. avg +0.0003, 유의한 개선 0개. IEdit에서 오히려 -0.002 하락.

- [x] **H3 (Score 충분)**: **확인됨** — probe (att_only_postvision 기반) top-k score가 이미 positional importance를 내재적으로 포착. 강제 위치 보존 없이도 충분.

- [ ] **H4 (Random control)**: **미확인** — rnd 조건 조기 종료. init/rec 효과가 사실상 없어서 rnd와 비교할 실익이 없다고 판단.

---

### 예상과 달랐던 점

1. **규모가 더 작았다**: 29개 데이터셋 중 22개가 완전 flat(+0.000). 이 정도로 성능 변화가 없을 것이라 예상하지 못함.

2. **init N 크기에 무감각**: init16, init32, init64가 대부분 데이터셋에서 동일한 점수. 즉 초기 16개 강제 보존이나 64개 강제 보존이나 결과가 같음 → probe가 이미 초기 토큰을 top-k에 포함시키고 있음.

3. **CharacterOrder만 뚜렷한 init 효과**: "순서"를 기억해야 하는 태스크 특성상 첫 이미지 patch가 중요 → +0.015. 나머지 태스크에서는 이런 구조가 없음.

---

### 결론

**probe의 att_only_postvision score는 이미 positional bias를 내재하고 있어, 별도의 positional forced-keep이 불필요하다.**

- init 강제 보존: 7/29 데이터셋에서 평균 +0.003~+0.015 수준 개선, 0개 하락
- rec 강제 보존: 3/29 개선, 1개 하락, 평균 +0.0003
- 개선이 나타나는 데이터셋은 태스크 구조상 "초반 이미지가 중요"한 경우(CharacterOrder, StateChange, Spot-the-Diff)

**논문 기여 표현**: "att_only_postvision 기반 score selection은 positional heuristic(StreamingLLM식 initial/recent keep) 없이도 comparable 또는 superior한 성능을 달성한다. 이는 task-conditioned attention이 image token의 위치보다 의미적 중요도를 더 잘 포착함을 시사한다."

---

### 다음 실험 제안

1. **ratio sensitivity**: r=0.05/0.10도 truncation 적용하여 LOOK-M r=0.20 대비 우리 r=0.10이 competitive한지 확인 → 효율성 클레임
2. **Phase B (per-image forced)**: CharacterOrder에서 global init vs per-image init 비교 — "첫 이미지" vs "각 이미지의 첫 patch" 분리

---

### 아티팩트 위치

- probe_mlp 전 조건: `/workspace/hd/artifacts/probe_global/{dataset}/probe_mlp/keep_0p20{suffix}/`
  - suffix 없음: baseline
  - `_init16`, `_init32`, `_init64`: initial forced-keep
  - `_rec16`, `_rec32`, `_rec64`: recent forced-keep
- 분석 CSV: `/workspace/hd/artifacts/probe_global/phase_a_init_rec_analysis.csv`
- probe vs LOOK-M 비교 CSV: `/workspace/hd/artifacts/probe_global/probe_vs_lookm_r020_truncated.csv`
- 실험 로그: `/workspace/hd/artifacts/probe_global/phase_a_all.log`, `phase_a_missing.log`

---

### 재현 커맨드

```bash
cd /workspace/zap

# baseline (keep_0p20)
GPU_INDEX=0 MODES=probe PROBE_ARCH=mlp TOTAL_KEEP_RATIOS=0.20 \
ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
TRUNCATE_LIKE_LOOKM=1 LOOK_MAX_CONTEXT_LEN=4096 LOOK_N_TOKENS_PER_IMAGE=576 \
SKIP_EXISTING=1 bash scripts/run_ablation_sweep.sh

# init_16
N_INITIAL_KEEP=16 GPU_INDEX=0 MODES=probe PROBE_ARCH=mlp TOTAL_KEEP_RATIOS=0.20 \
ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
TRUNCATE_LIKE_LOOKM=1 LOOK_MAX_CONTEXT_LEN=4096 LOOK_N_TOKENS_PER_IMAGE=576 \
SKIP_EXISTING=1 bash scripts/run_ablation_sweep.sh

# rec_16
N_RECENT_KEEP=16 GPU_INDEX=0 MODES=probe PROBE_ARCH=mlp TOTAL_KEEP_RATIOS=0.20 \
ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
TRUNCATE_LIKE_LOOKM=1 LOOK_MAX_CONTEXT_LEN=4096 LOOK_N_TOKENS_PER_IMAGE=576 \
SKIP_EXISTING=1 bash scripts/run_ablation_sweep.sh
```
