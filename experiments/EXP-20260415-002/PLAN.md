## Experiment Plan

**ID**: EXP-20260415-002  
**Author**: ZAP Team  
**Date**: 2026-04-15  
**Status**: [ ] Planned  [ ] Running  [x] Done (Phase 1 — truncation 없이 실행 완료, OOM 데이터셋 재실행 필요)  [ ] Abandoned

---

### 1. 동기 (Motivation)

현재 probe는 **ScienceQA만으로 학습**됨 (과학 MCQ, 단일~소수 이미지, 정적 레이아웃).
MileBench에서 probe가 LOOK-M에 뒤지는 데이터셋들은 공통적으로:
- **TextNeedleInAHaystack** (−0.039): 텍스트가 이미지 안에 있는 retrieval
- **MovingDirection** (−0.040): 다중 프레임의 동적 객체
- **CLEVR-Change** (−0.037): 두 장면 비교

→ ScienceQA의 도메인 한계로 probe가 "텍스트 dense 이미지"와 "두 이미지 비교" 태스크에서 약함.

**TextVQA** + **NLVR2**를 훈련 데이터에 추가하면 이 약점을 보완할 수 있을 것으로 기대:
- **TextVQA**: 자연 이미지 속 텍스트에 대한 질문 → TextNeedleInAHaystack 보완
- **NLVR2**: 두 이미지를 보고 statement T/F 판단 → Spot-the-Diff, CLEVR-Change 보완

---

### 2. 가설 (Hypothesis)

`ScienceQA + TextVQA + NLVR2`로 학습한 probe가 ScienceQA 단독 probe 대비:
- **TextNeedleInAHaystack**: +0.02 이상 개선
- **Spot-the-Diff / CLEVR-Change**: 유지 또는 소폭 개선
- **전체 MileBench 평균**: 동등 이상 유지 (하락 없음)

---

### 3. 독립변수 (What we change)

- 훈련 데이터: ScienceQA only → ScienceQA + TextVQA + NLVR2

---

### 4. 종속변수 (What we measure)

- **주요**: MileBench 29개 데이터셋 성능 (r=0.20, probe_mlp)
- **보조**: 약점 데이터셋 4개 (TextNeedleInAHaystack, MovingDirection, CLEVR-Change, SceneTransition) 성능

---

### 5. 고정 조건 (What stays the same)

- 모델: LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf)
- Teacher score: `att_only_postvision`
- Probe 구조: MLP (2-layer, GELU), per-layer 독립 학습
- keep ratio: `total_keep_ratio=0.20`
- 평가: MileBench 29 datasets, n=200/dataset
- seed: 42 (기존과 동일)
- 하드웨어: A100 80GB

---

### 6. 베이스라인

- **Baseline**: ScienceQA only probe (기존, 17W/10L/2T vs LOOK-M)

---

### 7. 데이터 스키마 통일

각 데이터셋의 원본 필드를 아래 통일 schema로 변환:

| 필드 | 설명 | ScienceQA | TextVQA | NLVR2 |
|---|---|---|---|---|
| `question` | 질문 텍스트 | `question` | `question` | `sentence` |
| `context` | 부가 맥락 | `hint` (optional) | — | — |
| `options` | 선택지 리스트 | `choices` | — | `["True", "False"]` |
| `image_paths` | 이미지 경로 리스트 | 1장 | 1장 | 2장 |
| `answer` | 정답 | `choices[answer]` | `answers[0]` | `label` |

**프롬프트 템플릿 (데이터셋별)**:

```
ScienceQA:
  USER: <image>
  Context: {hint}          ← hint가 있을 때만
  Question: {question}
  Options:
  A. {choice_0}
  ...
  Select the best answer based on the image and text.
  ASSISTANT:

TextVQA:
  USER: <image>
  Question: {question}
  ASSISTANT:

NLVR2:
  USER: <image>
  <image>
  Statement: {sentence}
  Is this statement True or False?
  Options:
  A. True
  B. False
  ASSISTANT:
```

---

### 8. 파이프라인 (3단계)

```
[Stage 1] 데이터 다운로드
  TextVQA → /workspace/hd/data/textvqa/
  NLVR2   → /workspace/hd/data/nlvr2/

[Stage 2] Teacher XY 수집 (각 데이터셋)
  스크립트: collect_vqa_teacher_xy.py (신규, 통합 schema 지원)
  출력: /workspace/hd/artifacts/sq_teacher/{textvqa,nlvr2}_xy/att_only_postvision/

[Stage 3] 통합 probe 훈련
  입력 shard: ScienceQA + TextVQA + NLVR2 합산
  스크립트: train_image_teacher_probe_shards.py (기존, shard dir 여러 개 지원)
  출력: /workspace/hd/artifacts/sq_teacher/image_probe_combined_v1/mlp/
```

---

### 9. 예상 결과

```
약점 데이터셋 개선:
  TextNeedleInAHaystack: 0.061 → 0.08+ (TextVQA 효과)
  Spot-the-Diff:         0.195 → 0.20+ (NLVR2 효과)
  CLEVR-Change:          0.142 → 0.15+ (NLVR2 효과)

전체:
  W/L/T = 17~18W / 9~10L / 2T (현재 17/10/2 유지 또는 개선)
```

---

### 10. 판단 기준 (Success Criteria)

- [ ] MileBench 전체 W/L/T ≥ 17/10/2 (하락 없음)
- [ ] TextNeedleInAHaystack ≥ 0.08 (현재 0.061)
- [ ] 전체 평균 Δ vs LOOK-M ≥ +0.020 (현재 +0.023)

---

### 11. 이 실험으로 증명할 수 없는 것

- 각 데이터셋의 개별 기여도 (TextVQA vs NLVR2 각각의 효과)
- 최적 데이터 비율 (ScienceQA:TextVQA:NLVR2)
- 학습 샘플 수의 영향

---

### 12. 예상 런타임 / 리소스

| 단계 | 예상 시간 | GPU |
|---|---|---|
| TextVQA XY 수집 (train ~34K) | ~6h | A100 ×1 |
| NLVR2 XY 수집 (train ~86K, 2img/sample) | ~18h | A100 ×1 |
| Probe 훈련 (합산 shard) | ~30min | A100 ×1 |
| MileBench 전체 평가 | ~4h | A100 ×1 |

> 수집 단계가 오래 걸리므로 screen 세션 + cuda:0 사용 권장.

---

### 13. 구현 필요 사항

| 항목 | 파일 | 상태 |
|---|---|---|
| 데이터 다운로드 | `scripts/download_textvqa_nlvr2.py` | 🔄 실행 중 (textvqa/train ✅, nlvr2 진행 중) |
| 통합 XY 수집 스크립트 | `collect_vqa_teacher_xy.py` | ✅ 구현 완료 |
| 파이프라인 쉘 스크립트 | `scripts/run_combined_probe_pipeline.sh` | ✅ 구현 완료 |
| 훈련 (기존 재사용) | `train_image_teacher_probe_shards.py` | ✅ 기존 |

---

### 14. 경로 정보 (새 환경 세팅 시 참고)

#### 데이터

| 항목 | 경로 |
|---|---|
| MileBench 전체 | `/workspace/hd/data/MileBench/` |
| ScienceQA | `/workspace/hd/data/scienceqa/` |
| TextVQA (train/val) | `/workspace/hd/data/textvqa/` |
| NLVR2 (train/val) | `/workspace/hd/data/nlvr2/` |

MileBench 각 데이터셋 레이아웃:
```
/workspace/hd/data/MileBench/{DatasetName}/
  {DatasetName}.json     ← 샘플 목록
  images/                ← 이미지 파일
```

#### Probe 모델

| 항목 | 경로 |
|---|---|
| **combined_v1 (ScienceQA+TextVQA+NLVR2)** | `/workspace/hd/artifacts/sq_teacher/image_probe_combined_v1/mlp/` |
| ScienceQA-only (baseline) | `/workspace/hd/artifacts/sq_teacher/image_probe_scienceqa_att_only_postvision/mlp/` |

Probe 모델 파일 구성:
```
{probe_dir}/
  config.json
  model.safetensors
```

#### XY Shard (probe 재훈련 시)

| 항목 | 경로 |
|---|---|
| ScienceQA XY (train) | `/workspace/hd/artifacts/sq_teacher/scienceqa_xy/att_only_postvision/train/` |
| TextVQA XY (train) | `/workspace/hd/artifacts/sq_teacher/textvqa_xy/att_only_postvision/train/` |
| NLVR2 XY (train) | `/workspace/hd/artifacts/sq_teacher/nlvr2_xy/att_only_postvision/train/` |

---

### 15. 재현 커맨드

#### Stage 0 (필요 시): 데이터 다운로드
```bash
conda run -n kv python zap/scripts/download_textvqa_nlvr2.py \
  --out_dir /workspace/hd/data
```

#### Stage 1 (필요 시): XY 수집 + Probe 재훈련
```bash
cd /workspace/zap
conda run -n kv \
  TEACHER_TYPE=att_only_postvision \
  DEVICE=cuda:0 \
  bash scripts/run_combined_probe_pipeline.sh
```

#### Stage 2: MileBench 전체 평가 (truncation 포함, OOM 방지)

**전체 29개 데이터셋 일괄 실행 (권장)**:
```bash
cd /workspace/zap
conda run -n kv env \
  GPU_INDEX=0 \
  DATA_ROOT=/workspace/hd/data/MileBench \
  PROB_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
  ATT_ONLY_PROBE_ROOT=/workspace/hd/artifacts/sq_teacher/image_probe_combined_v1 \
  PROBE_LABEL=combined \
  TEACHERS=att_only_postvision \
  METHODS=mlp \
  KEEP_RATIOS="0.20" \
  TRUNCATE_LIKE_LOOKM=1 \
  LOOK_MAX_CONTEXT_LEN=4096 \
  LOOK_N_TOKENS_PER_IMAGE=256 \
  SKIP_EXISTING=1 \
  bash scripts/run_milebench_probe_all.sh
```

> `SKIP_EXISTING=1`이므로 이미 완료된 데이터셋은 건너뜀.
> `ATT_ONLY_PROBE_ROOT`를 combined_v1 경로로 오버라이드하는 것이 핵심.

**OOM이 났던 데이터셋만 선택적 재실행**:
```bash
cd /workspace/zap
conda run -n kv env \
  GPU_INDEX=0 \
  DATA_ROOT=/workspace/hd/data/MileBench \
  PROB_ARTIFACT_ROOT=/workspace/hd/artifacts/probe_global \
  ATT_ONLY_PROBE_ROOT=/workspace/hd/artifacts/sq_teacher/image_probe_combined_v1 \
  PROBE_LABEL=combined \
  TEACHERS=att_only_postvision \
  METHODS=mlp \
  KEEP_RATIOS="0.20" \
  TRUNCATE_LIKE_LOOKM=1 \
  LOOK_MAX_CONTEXT_LEN=4096 \
  LOOK_N_TOKENS_PER_IMAGE=256 \
  SKIP_EXISTING=0 \
  DATASETS="ActionLocalization ActionPrediction ActionSequence CharacterOrder EgocentricNavigation GPR1200 ImageNeedleInAHaystack ObjectInteraction ObjectShuffle SceneTransition StateChange TextNeedleInAHaystack WikiVQA" \
  bash scripts/run_milebench_probe_all.sh
```

**screen 세션 권장 (장시간 실행)**:
```bash
screen -S exp002_eval
# 위 명령어 실행 후 Ctrl+A, D로 detach
# 재접속: screen -r exp002_eval
```

#### Stage 3: 결과 취합
```bash
conda run -n kv python zap/scripts/export_probe_lookm_performance_csv.py \
  --probe_artifact_root /workspace/hd/artifacts/probe_global \
  --probe_label combined \
  --out_csv /workspace/hd/artifacts/probe_global/probe_vs_lookm_combined_r020.csv
```

---

### 16. multimodalqa / mmcoqa 포맷 오류 메모

이번 실행에서 `multimodalqa`(200/200), `mmcoqa`(159/200)가 아래 오류로 실패:
```
ValueError: Question contains N image placeholders but received M images
```
이는 OOM이 아닌 prompt template 문제임. MileBench의 multi-image 샘플에서
`<image>` 플레이스홀더 개수와 실제 입력 이미지 수가 불일치.
→ `evaluate_image_teacher_pruning.py`의 multi-image 처리 경로 수정 필요 (별도 이슈).
