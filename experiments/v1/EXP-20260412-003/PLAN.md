## Experiment Plan

**ID**: EXP-20260412-003
**Date**: 2026-04-12
**Status**: [ ] Planned  [ ] Running  [ ] Done  [ ] Abandoned

---

### 제목
Image Token의 Position Bias 분석 — Initial / Recent / Random forced keep이 성능에 미치는 영향

---

### 1. 동기

현재 우리 방법은 image token을 `att_only_postvision` score 기반 top-k로만 선택한다.
text token KV cache 연구(StreamingLLM, SnapKV)에서는 다음이 알려져 있다:
- **Attention sink**: 첫 1~4개 token이 항상 높은 attention → 제거 시 성능 급락
- **Recent tokens**: 최근 K개 token은 현재 생성에 직접 영향

그러나 이는 text token 중심 연구다. **image token에서도 동일한 현상이 나타나는가?**
image token은 grid-based patch이므로 position의 의미가 text와 근본적으로 다르다.

핵심 위험: global image token 순서로 initial/recent를 정의하면,
multi-image 환경에서 "첫 번째 이미지의 앞쪽 patch", "마지막 이미지의 뒤쪽 patch"를
보존하는 실험이 되어버린다. 이 경우 성능 변화가 나타나더라도
- "initial token이 sink라서"인지
- "첫 번째 이미지가 답을 담고 있어서"인지

구분이 불가능하다. 따라서 **global**과 **per-image** 버전을 모두 실험해야 한다.

---

### 2. 가설

- **H1 (Initial/Sink)**: image token에서 text token과 같은 sink effect는 약하거나
  dataset-specific일 것이다. image token은 grid 구조라서 position별 sink가 뚜렷하지 않을 것.

- **H2 (Recent)**: recent image token 보존의 효과는 보편적이지 않고,
  multi-image ordering이나 task 구조에 의존할 것이다.

- **H3 (Score 충분)**: fixed budget 하에서 att_only_postvision top-k가
  별도 positional keep 없이도 충분할 가능성이 높다.

- **H4 (Random control)**: position-based forced keep의 효과가 있더라도
  random forced keep 대비 유의미한 차이가 없다면, 그것은 position bias가 아니라
  단순 budget 다양화 효과일 뿐이다.

---

### 3. 실험 설계

#### Phase A — 1D sweep (global 기준)

변수 하나씩만 변경, 나머지 고정:

| 조건 | n_initial_keep | n_recent_keep | n_random_keep |
|---|---|---|---|
| baseline | 0 | 0 | 0 |
| initial_16 | 16 | 0 | 0 |
| initial_32 | 32 | 0 | 0 |
| initial_64 | 64 | 0 | 0 |
| recent_16 | 0 | 16 | 0 |
| recent_32 | 0 | 32 | 0 |
| recent_64 | 0 | 64 | 0 |
| random_16 | 0 | 0 | 16 |
| random_32 | 0 | 0 | 32 |
| random_64 | 0 | 0 | 64 |

→ random_keep은 같은 N개를 무작위로 강제 보존하는 control.
  initial/recent 효과가 random 대비 유의하지 않으면 positional bias 없음.

#### Phase B — per-image vs global 비교 (Phase A에서 signal 나오면)

| 조건 | 방식 |
|---|---|
| initial_global_16 | 전체 image token 시퀀스에서 첫 16개 |
| initial_per_image_16 | 각 이미지 블록마다 첫 16개씩 |
| recent_global_16 | 전체 시퀀스에서 마지막 16개 |
| recent_per_image_16 | 각 이미지 블록마다 마지막 16개씩 |

→ "position bias냐, image-order bias냐" 분리

#### Phase C — 2D interaction (의미 있는 경우만, 데이터셋 1~2개)

initial + recent 동시 적용:
- (initial=16, recent=16), (initial=32, recent=32)

처음부터 전체 grid는 불필요. Phase A/B 결과 보고 결정.

---

### 4. 고정 조건

- **모드**: oracle (att_only_postvision, teacher record 사용) — probe 변수 제거
- **total_keep_ratio**: 0.20 고정
- **데이터셋**: DocVQA, Spot-the-Diff, CLEVR-Change, IEdit
- 모델: LLaVA-1.5-7B
- max_new_tokens: 32

---

### 5. 구현 계획

#### 5-1. `ImageTokenTopKPress` 파라미터 추가

```python
@dataclass
class ImageTokenTopKPress(BasePress):
    ...
    n_initial_keep: int = 0    # 무조건 보존할 첫 N개 image token (global or per-image)
    n_recent_keep: int = 0     # 무조건 보존할 마지막 N개 image token
    n_random_keep: int = 0     # 무조건 보존할 random N개 (control)
    per_image_forced: bool = False  # True면 각 이미지 블록별로 적용
```

#### 5-2. compress() 로직

```python
# 1. forced_keep indices 결정
forced = set()
if per_image_forced and image_block_boundaries available:
    for block in image_blocks:
        forced |= set(block[:n_initial_keep])
        forced |= set(block[-n_recent_keep:])
else:
    forced |= set(image_positions[:n_initial_keep].tolist())
    forced |= set(image_positions[-n_recent_keep:].tolist())
if n_random_keep > 0:
    random_candidates = [p for p in image_positions if p not in forced]
    forced |= set(random.sample(random_candidates, min(n_random_keep, len(random_candidates))))

# 2. 나머지 budget → score top-k
remaining_budget = max(0, n_image_keep - len(forced))
scored_candidates = [p for p in image_positions if p not in forced]
# top-k on scored_candidates

# 3. 합산
keep = forced ∪ top_scored
```

#### 5-3. 필수 로그 저장 (per sample)

```python
{
    "n_forced_kept": len(forced),
    "n_scored_kept": remaining_budget_used,
    "n_random_kept": n_random_actually_kept,
    "forced_overlap_with_teacher_topk": # forced 중 teacher top-k에도 들어갔을 비율
    "forced_mean_teacher_rank": # forced token들의 teacher score 기준 평균 순위
    "r_eff_prompt": actual_kept / seq_len,
}
```

→ "성능이 올랐다면 forced token이 진짜 중요해서인지, budget 구조 변화 때문인지" 분리 가능

#### 5-4. Special token 처리

LLaVA-1.5는 image token이 일반 patch token만으로 구성되어 있으나,
로그에 `forced token 중 position ≤ 4인 비율`은 기록하여 sink 가능성 모니터링.

---

### 6. 종속변수

**성능 지표:**
- LOOK-M evaluation metric (ROUGE-L / Accuracy per dataset)

**해석 보조 지표 (per run):**
- n_forced_kept, n_scored_kept
- forced_overlap_with_teacher_topk (%)
- forced_mean_teacher_rank (1 = best)

**효율 지표:**
- r_eff_prompt (= total_keep_ratio, 모든 조건에서 동일해야 함)

---

### 7. 시각화 계획

각 샘플에 대해 eviction map 생성:

```
[원본 이미지]
  ↓ 오버레이
[score heatmap]     : att_only_postvision score → 색상 강도
[evicted mask]      : 제거된 patch → 회색
[forced kept]       : initial/recent/random으로 강제 보존 → 초록/파랑/주황 테두리
[forced rank label] : forced token의 teacher score 순위 표시
```

추가: **forced token의 원래 teacher rank 분포 히스토그램**
- forced token 중 teacher top-k에 들어갔을 비율 → "forced가 실제로 좋은 token이었는가"

저장: `/workspace/hd/artifacts/viz/eviction_maps/{dataset}/{sample_id}/`

---

### 8. 데이터셋별 해석 기준

| Dataset | 특성 | 기대 해석 |
|---|---|---|
| DocVQA | text-heavy, multi-image, MCQ | text 보존이 dominant → positional effect 약할 것 |
| Spot-the-Diff | 두 이미지 비교, 차이 영역이 핵심 | 특정 spatial region이 중요 → position bias 아닐 것 |
| CLEVR-Change | 객체 변화 감지 | 유사 패턴 예상 |
| IEdit | 지시 기반 편집, 관련 영역 파악 | task-specific spatial attention이 dominant |

→ dataset마다 다른 결과가 나오면 "universal position bias 없음"이 결론이고,
그 자체가 논문에서 중요한 기여가 된다.

---

### 9. 이 실험으로 증명할 수 없는 것

- Text token에 대한 recent/initial 효과 (우리는 image-only eviction이므로 해당 없음)
- 다른 VLM 아키텍처 (Flamingo, InstructBLIP 등)에서의 일반화
- Greedy top-k 외의 선택 전략 (threshold-based, adaptive 등)과의 비교

---

### 10. 실행 순서

```bash
# Phase A: global 1D sweep (oracle, total_keep_ratio=0.20)
for N_INIT in 0 16 32 64; do
  N_INITIAL_KEEP=${N_INIT} GPU_INDEX=1 MODES=oracle \
  TOTAL_KEEP_RATIOS=0.20 DATASETS="DocVQA Spot-the-Diff CLEVR-Change IEdit" \
  ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/ablation_position/initial_${N_INIT} \
  TEACHER_ROOT=/workspace/hd/artifacts/oracle \
  bash scripts/run_ablation_sweep.sh
done

for N_REC in 0 16 32 64; do
  N_RECENT_KEEP=${N_REC} GPU_INDEX=1 MODES=oracle \
  TOTAL_KEEP_RATIOS=0.20 DATASETS="DocVQA Spot-the-Diff CLEVR-Change IEdit" \
  ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/ablation_position/recent_${N_REC} \
  TEACHER_ROOT=/workspace/hd/artifacts/oracle \
  bash scripts/run_ablation_sweep.sh
done

for N_RND in 16 32 64; do
  N_RANDOM_KEEP=${N_RND} GPU_INDEX=1 MODES=oracle \
  TOTAL_KEEP_RATIOS=0.20 DATASETS="DocVQA Spot-the-Diff CLEVR-Change IEdit" \
  ABLATION_ARTIFACT_ROOT=/workspace/hd/artifacts/ablation_position/random_${N_RND} \
  TEACHER_ROOT=/workspace/hd/artifacts/oracle \
  bash scripts/run_ablation_sweep.sh
done

# Phase B: per-image vs global (Phase A 결과 보고 결정)
# Phase C: 2D interaction (DocVQA, Spot-the-Diff 한정)
```

### 아티팩트 위치

- 결과: `/workspace/hd/artifacts/ablation_position/`
- 시각화: `/workspace/hd/artifacts/viz/eviction_maps/`
- 예상 런타임: Phase A ~4시간 (10 조건 × 4 datasets)
