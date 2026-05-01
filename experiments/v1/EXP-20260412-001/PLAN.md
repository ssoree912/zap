## Experiment Plan

**ID**: EXP-20260412-001
**Date**: 2026-04-12
**Status**: [ ] Planned  [ ] Running  [ ] Done  [ ] Abandoned

---

### 제목
Probe 성능 재측정 — image_keep_ratio → total_keep_ratio (global token basis, 전체 데이터셋)

---

### 1. 동기

기존 probe sweep (`*_scienceqa_probe_sweep`)은 `image_keep_ratio` 기준으로 돌림.
- `image_keep_ratio=0.20` → image token의 20%만 유지, text는 항상 전부 → r_eff_prompt ≈ 23.7%
- LOOK-M의 `hh+recent=0.20` → r_eff_prompt ≈ 20%
- **비교 기준이 다름 → 공정 비교 불가**

`total_keep_ratio`(전체 token 기준)로 통일해야 LOOK-M, ablation 결과와 동일한 기준에서 비교 가능.
기존 결과 디렉토리는 삭제하고 새로 돌림.

---

### 2. 가설

- total_keep_ratio 기준에서도 probe는 LOOK-M 대비 대부분 데이터셋에서 우위
- ratio 통일 후 성능 gap이 일부 좁혀질 수 있음 (우리 쪽이 약간 더 많이 보존했었으므로)
- 그래도 image-only eviction의 구조적 이점은 유지될 것

---

### 3. 독립변수

- total_keep_ratio: 0.05, 0.10, 0.20  (LOOK-M이 ≈20% 기준이므로 그 이하만)
- probe 아키텍처: linear, mlp (best 선택)
- 데이터셋: MileBench 전체 (29개)

---

### 4. 종속변수

- 주요: LOOK-M evaluation metric (ROUGE-L / Accuracy per dataset)
- 보조: n_failures, r_eff_prompt (= total_keep_ratio, 확인용)

---

### 5. 고정 조건

- Teacher: att_only_postvision (scienceqa로 학습된 probe)
- Probe 모델:
  - mlp: `/workspace/hd/artifacts/sq_teacher/image_probe_scienceqa_att_only_postvision/mlp`
  - linear: `/workspace/hd/artifacts/sq_teacher/image_probe_scienceqa_att_only_postvision/linear`
- 기반 모델: LLaVA-1.5-7B
- max_new_tokens: 32
- prompt_style: look_milebench

---

### 6. 베이스라인

- LOOK-M (hh=0.10, recent=0.10): 기존 결과 (`/workspace/LOOK-M/outputs/`)
- Oracle (att_only_postvision + image-only): ablation 결과 (`/workspace/hd/artifacts/ablation/*/oracle/`)

---

### 7. 예상 결과

- probe(total r=0.20) ≈ probe(image r=0.20) — 큰 차이 없을 것 (text token 수가 작아서 r_eff 차이가 작음)
- 여전히 LOOK-M 대비 다수 데이터셋에서 우위

---

### 8. 판단 기준

- 29개 데이터셋 기준 probe vs LOOK-M 승률 > 60%
- total_keep_ratio=0.20 기준 r_eff_prompt 실제 검증

---

### 9. 이 실험으로 증명할 수 없는 것

- probe 모델 자체의 품질 (scienceqa로만 학습됨, domain gap 있음)
- 다른 image_keep_ratio best를 쓴 기존 결과와의 공정 비교 (ratio 기준이 달랐음)

---

### 10. 실행 계획

**사전 작업: 기존 probe sweep 결과 삭제**
```bash
rm -rf /workspace/hd/artifacts/prob/*_scienceqa_probe_sweep
```

**스크립트 수정 필요:**
`run_ablation_sweep.sh`에 probe 모드 추가:
- `--mode probe`일 때 `--probe_model_name` 전달
- teacher dir 불필요 (probe는 runtime에 self-contained)

**실행:**
```bash
# GPU 0: probe mlp sweep
screen -dmS probe_mlp bash -c "
GPU_INDEX=0 MODES='probe' PROBE_ARCH='mlp' \
TOTAL_KEEP_RATIOS='0.10 0.20 0.30' \
ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
bash scripts/run_probe_sweep.sh 2>&1 | tee /workspace/hd/artifacts/probe_global/sweep_mlp.log
"

# GPU 1: probe linear sweep
screen -dmS probe_linear bash -c "
GPU_INDEX=1 MODES='probe' PROBE_ARCH='linear' \
TOTAL_KEEP_RATIOS='0.10 0.20 0.30' \
ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
bash scripts/run_probe_sweep.sh 2>&1 | tee /workspace/hd/artifacts/probe_global/sweep_linear.log
"
```

**아티팩트 위치:** `/workspace/hd/artifacts/probe_global/`
**예상 시간:** ~6시간 (29 datasets × 3 ratios × 2 archs)
