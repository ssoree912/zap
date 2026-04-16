## Experiment Plan

**ID**: EXP-20260412-004
**Author**: 
**Date**: 2026-04-14
**Status**: [ ] Planned  [ ] Running  [x] Done  [ ] Abandoned

---

### 제목
Ratio Sensitivity — probe_mlp r=0.05/0.10 (truncation) vs LOOK-M r=0.20 효율성 비교

---

### 1. 동기

EXP-20260412-001 결과: probe_mlp r=0.20 (truncation) 기준 LOOK-M r=0.20 대비 **17승/10패/2타이 (29개 데이터셋)**.

우리 방법이 같은 token budget에서 LOOK-M을 이긴다면, 더 낮은 budget (r=0.10, r=0.05)에서도 LOOK-M r=0.20과 competitive하다면 **효율성 클레임** 가능:

> "우리 방법은 LOOK-M 대비 절반의 token만 유지하면서 동등하거나 더 나은 성능을 달성한다."

기존 r=0.05/0.10 결과는 두 가지 이유로 무효:
1. `image_keep_ratio` 기준 (구버전) → `total_keep_ratio`로 재정의 필요
2. truncation 미적용 → OOM 발생 데이터셋 결과 오염

---

### 2. 가설

- **H1**: probe_mlp r=0.10 (truncation) ≥ LOOK-M r=0.20 on 평균 성능
  - 근거: probe의 score 기반 선택이 LOOK-M의 H2O-like 선택보다 정보 밀도가 높을 것
- **H2**: r=0.10 → r=0.20 성능 향상폭이 r=0.05 → r=0.10보다 작을 것
  - 근거: 이미 EXP-001에서 r=0.05/0.10/0.20 flat 현상 관찰됨 (대부분 데이터셋)
- **H3**: 이미지 수가 많은 데이터셋(SceneTransition, ActionLocalization 등)은 낮은 ratio에서 degradation 더 클 것

---

### 3. 독립변수

- `total_keep_ratio`: 0.05 / 0.10 (비교 기준: r=0.20 은 이미 완료)
- 비교 대상: LOOK-M r=0.20 (기존 결과 활용)

### 4. 종속변수

- LOOK-M evaluation metric (Accuracy / ROUGE-L, per dataset)
- 승/패 카운트 vs LOOK-M r=0.20

### 5. 고정 조건

- 모드: probe_mlp (att_only_postvision probe)
- 모델: LLaVA-1.5-7B
- truncation: LOOK-M 방식 (max_context_len=4096, n_tokens_per_image=576)
- 데이터셋: MileBench 전체 (데이터 존재하는 것)
- GPU: 0번 단독

---

### 6. 베이스라인 (비교 대상)

| Method | Ratio | 결과 위치 |
|---|---|---|
| LOOK-M | r=0.20 | `probe_global/{ds}/look_m/keep_0p20/` |
| probe_mlp (ours) | r=0.20 | `probe_global/{ds}/probe_mlp/keep_0p20/` (완료) |

---

### 7. 예상 결과

- r=0.10: LOOK-M r=0.20 대비 15승 이상 (현재 r=0.20이 17승이므로 소폭 감소)
- r=0.05: LOOK-M r=0.20 대비 10승 전후
- ratio 무감각 데이터셋 (docvqa, movingattribute 등)은 r=0.05에서도 동일 성능 유지

### 8. 판단 기준 (Success Criteria)

- "r=0.10 probe ≥ LOOK-M r=0.20" 데이터셋 수 ≥ 12 (전체 29개의 40% 이상)
- 평균 성능 차이 |probe r=0.10 - LOOK-M r=0.20| < 0.03

### 9. 이 실험으로 증명할 수 없는 것

- 다른 VLM(Qwen2-VL, Gemma3 등)에서의 일반화
- LOOK-M을 r=0.10으로 낮췄을 때와의 비교 (LOOK-M r=0.10 미실행)
- Latency/throughput 실측 (효율성은 token 수 기준 간접 측정)

---

### 10. 실행 계획

#### 사전 정리
기존 r=0.05/0.10 결과 삭제 (image_keep_ratio 기준 + no truncation):
```bash
for ds in $(ls hd/artifacts/probe_global/); do
  rm -rf hd/artifacts/probe_global/$ds/probe_mlp/keep_0p05
  rm -rf hd/artifacts/probe_global/$ds/probe_mlp/keep_0p10
done
```

#### sweep 실행
```bash
GPU_INDEX=0 MODES=probe PROBE_ARCH=mlp \
TOTAL_KEEP_RATIOS="0.05 0.10" \
ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
TRUNCATE_LIKE_LOOKM=1 LOOK_MAX_CONTEXT_LEN=4096 LOOK_N_TOKENS_PER_IMAGE=576 \
SKIP_EXISTING=1 \
bash scripts/run_ablation_sweep.sh
```

#### 결과 분석
```bash
python3 experiments/EXP-20260412-004/analyze.py
```

---

### 아티팩트 위치

- probe_mlp r=0.05: `/workspace/hd/artifacts/probe_global/{ds}/probe_mlp/keep_0p05/`
- probe_mlp r=0.10: `/workspace/hd/artifacts/probe_global/{ds}/probe_mlp/keep_0p10/`
- 분석 결과: `/workspace/zap/experiments/EXP-20260412-004/RESULT.md`
- 로그: `/workspace/hd/artifacts/probe_global/ratio_sensitivity.log`
