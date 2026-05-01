## Experiment Plan

**ID**: EXP-20260417-001
**Author**: ssoree912
**Date**: 2026-04-17
**Status**: [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned

---

### 1. 동기 (Motivation)

현재 ZAP의 inference-time pruning은 **one-shot**: prefilling 중 각 레이어에서 probe score를 한 번 계산하고, 그 결과로 즉시 top-k 토큰을 선택한다. 이 방식의 잠재적 문제는 probe score에 포함된 노이즈가 한 번의 ranking에 그대로 반영된다는 점이다.

**Iterative pruning**은 동일한 probe score를 **4라운드**에 걸쳐 매 라운드 원본 대비 20%p씩 점진적으로 제거한다:
- Round 1: N개 이미지 토큰 → **80%** 유지 (20% 제거)
- Round 2: 80% pool → **60%** 유지 (추가 20% 제거)
- Round 3: 60% pool → **40%** 유지 (추가 20% 제거)
- Round 4: 40% pool → **20%** 유지 ← 최종 target (추가 20% 제거)

핵심 직관: "**잘 찾는 애들끼리 모이지 않을까**" — 진짜 중요한 토큰은 매 라운드에서 일관되게 높은 점수를 받는 반면, 노이즈성 토큰은 한 라운드에서는 상위권에 들어도 다음 라운드에서 탈락한다. 결과적으로 살아남는 토큰의 집합이 더 안정적(precision이 높음)이다.

---

### 2. 가설 (Hypothesis)

> **Iterative pruning은 one-shot pruning보다 높은 task accuracy를 달성한다** — 왜냐하면 매 라운드에서 명확한 losers를 먼저 제거함으로써 남은 pool의 신호 대 잡음비가 향상되고, 최종 top-k 선택의 정밀도가 높아지기 때문이다.

구체적 예측:
- MileBench 평균 점수: iterative (4-round, 20%p/round) ≥ one-shot, 같은 final keep ratio 조건
- 효과 크기: probe → oracle gap의 20~50%를 iterative가 메울 것
- Round별 제거 선택의 안정성: 각 라운드에서 제거되는 토큰의 oracle-overlap이 라운드가 진행될수록 감소할 것 (초반에 쉬운 losers가 제거됨)

---

### 3. 독립변수 (What we change)

- **변수 1: Pruning 방식** — one-shot(baseline) vs iterative-4round
- **변수 2: Score 종류** — probe (`att_only_postvision`), oracle (상한선 참조용)

**고정된 iterative schedule (20%p/round × 4 rounds)**:

| 방식 | Round 1 | Round 2 | Round 3 | Round 4 | Final |
|------|---------|---------|---------|---------|-------|
| One-shot | — | — | — | — | 20% |
| **Iterative-4** | **80%** | **60%** | **40%** | **20%** | **20%** |

각 라운드에서 원본 N 대비 20%p씩 추가 제거. pool은 이전 라운드 생존 토큰만으로 구성.

- **변수 3 (보조)**: head 집계 전략 — 라운드별 pool 결정 시 `amax` head 기준 vs head-별 union

---

### 4. 종속변수 (What we measure)

**주요 metric**:
- MileBench 29 datasets 평균 정확도 (keep=0.20, 기존 실험과 동일 조건)

**보조 metric**:
- per-dataset Δ vs one-shot probe (어느 dataset에서 이득/손해 있는지)
- 선택된 토큰의 **공간적 분산도** (eviction map의 entropy): iterative가 더 다양한 region을 커버하는지
- **Overlap ratio** (oracle이 선택한 토큰과의 겹침): precision @ k 측면에서 oracle에 얼마나 근접하는지
- 추가 prefill latency (per-round re-ranking 비용)

---

### 5. 고정 조건 (What stays the same)

- 모델: LLaVA-1.5-7B
- Probe 체크포인트: `/workspace/hd/artifacts/zap/probe_model_scienceqa` (재학습 없음)
- 평가 데이터셋: MileBench 29 datasets
- Final keep ratio: `total_keep_ratio = 0.20` (기존 실험과 동일)
- Seed: 42
- 하드웨어: 동일 GPU 환경
- Pruning scope: image-only eviction (text 토큰은 항상 유지)
- head_reduce: `amax`

---

### 6. 베이스라인

- **One-shot probe** (현재 방법, N=1): MileBench 29 avg = 0.235 (EXP-20260412-002 기준)
- **Oracle one-shot**: 성능 상한선
- **LOOK-M**: 외부 비교 대상 (avg = 0.212 기준)

---

### 7. 예상 결과

```
방향성 예측 (MileBench avg score @ keep=0.20):

oracle one-shot ≥ iterative-4round > one-shot probe > LOOK-M

수치 예측:
  one-shot probe:         0.235 (baseline)
  iterative-4 (probe):    0.238 ~ 0.243  (+0.3~0.8%p)
  oracle one-shot:        ~0.255 (상한)
  LOOK-M:                 0.212 (외부 비교)

부가 예측:
  - Round별 제거 토큰 중 oracle 제거 토큰과의 overlap:
      Round 1 제거분 > Round 4 제거분  (쉬운 losers가 먼저 탈락)
  - Spatial entropy: iterative ≥ one-shot (중복 지역 토큰이 정리됨)
```

---

### 8. 판단 기준 (Success Criteria)

- **성공**: iterative-4round (probe)가 one-shot probe를 MileBench avg 기준 **+0.3%p 이상** 개선
- **부분 성공**: 통계적으로 유의미하지 않은 개선이지만, spatial diversity metric에서 일관된 향상
- **실패**: one-shot과 차이 없음 (±0.1%p 이내)
- **역효과**: iterative가 one-shot보다 낮음 → round 간 score rank가 너무 안정적이어서 iterative ≡ one-shot임을 의미

---

### 9. 이 실험으로 증명할 수 없는 것

- **Score 재계산 없는 iterative의 한계**: 본 실험은 동일한 probe score를 반복 사용한다. 진정한 iterative score re-estimation (매 라운드에서 partial forward pass 재실행)은 계산 비용이 크기 때문에 이번 scope 밖이다.
- **Score 자체의 개선**: iterative는 pruning 전략을 바꾸는 것이지, probe 학습 방법을 바꾸는 게 아님
- **Multi-image generalization**: 현재 평가는 single-image 위주; iterative가 multi-image에 효과적인지는 별도 검증 필요
- **Latency 실용성**: 라운드마다 topk 연산이 추가되므로, MileBench 성능 개선이 overhead를 정당화하는지는 별개 판단

---

### 10. 구현 계획

#### Phase 1: 핵심 구현 (iterative top-k in `ImageTokenTopKPress.compress()`)

**현재 one-shot 로직** (`image_token_press.py`, `compress()` L307-404):
```python
# 현재: N개 이미지 토큰에서 한 번에 top-k 선택
topk_within_free = torch.topk(free_scores, k=k, dim=-1).indices
```

**변경: iterative top-k (4-round, 20%p/round 고정)**:
```python
def _iterative_topk_4round(scores: torch.Tensor, final_k: int) -> torch.Tensor:
    """
    4라운드, 매 라운드 원본 대비 20%p 제거.
    
    scores:  (num_kv_heads, n_free)
    final_k: 최종 유지 토큰 수 (= 0.2 * n_free 기준)
    returns: (num_kv_heads, final_k) — 원본 index 기준
    """
    n_free = scores.shape[-1]
    # [0.8N, 0.6N, 0.4N, 0.2N]
    schedule = [
        max(1, int(n_free * 0.8)),
        max(1, int(n_free * 0.6)),
        max(1, int(n_free * 0.4)),
        final_k,
    ]
    
    # pool: 현재 라운드에서 후보로 남은 토큰의 원본 인덱스
    pool = torch.arange(n_free, device=scores.device)  # (n_free,)
    
    for round_k in schedule:
        pool_scores = scores[:, pool]               # (H, |pool|)
        # amax head 기준으로 pool 결정 (per-head top-k는 마지막 라운드에서만)
        agg_scores = pool_scores.amax(dim=0)        # (|pool|,)
        topk_local = torch.topk(agg_scores, k=round_k).indices  # (round_k,)
        pool = pool[topk_local]                     # (round_k,) — 원본 인덱스
    
    # 마지막 라운드 이후 pool이 final_k개 남음 → per-head top-k 적용
    final_pool_scores = scores[:, pool]             # (H, final_k)
    final_local = torch.topk(final_pool_scores, k=final_k, dim=-1).indices  # (H, final_k)
    return pool[final_local]                        # (H, final_k) — 원본 인덱스
```

**Args 추가** (`evaluate_image_teacher_pruning.py`):
```bash
--iterative_pruning        # flag: 이 옵션 있으면 iterative-4round 사용, 없으면 one-shot
```

#### Phase 2: 보조 metric 수집

**Overlap ratio with oracle**:
```python
# 평가 루프 내에서 oracle press와 probe press의 keep_mask를 동시에 기록
overlap = (probe_keep_mask & oracle_keep_mask).sum() / oracle_keep_mask.sum()
```

**Spatial entropy** (토큰 분산도):
```python
# VizCapture에 기록된 keep_positions를 사용
# 이미지를 grid로 나누고 선택 토큰의 Shannon entropy 계산
from kvpress.presses.image_token_press import VizCapture
```

---

### 11. 예상 런타임 / 리소스

- GPU: 1× A100 (또는 기존 실험 동일 GPU)
- 예상 시간: 
  - per config: ~3~4시간 (MileBench 29 datasets full run)
  - 전체 실험 (iterative-4 probe + iterative-4 oracle + one-shot probe baseline): 3 runs × 3.5h ≈ 10시간
  - **실행 순서**: iterative-4 probe vs one-shot probe → 방향성 확인 → oracle 비교
- 디스크: config당 ~200MB CSV + optional eviction maps

---

## Implementation Log — Phase 1: Iterative Pruning

**날짜**: 2026-04-17
**구현자**: Claude (ssoree912 요청)
**상태**: 구현 완료, 미실행

---

### 구현 파일 목록

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `kvpress/presses/image_token_press.py` | **수정** | `_iterative_topk()` 함수 추가, `ImageTokenTopKPress`에 `n_iterative_rounds` 필드 추가, `compress()` top-k 호출 교체 |
| `evaluate_image_teacher_pruning.py` | **수정** | `--n_iterative_rounds` 인수 추가, `build_press()`의 `forced_kwargs`에 포함 |

---

### 1. 핵심 구현: `_iterative_topk()` (`image_token_press.py`)

#### 위치

`_find_image_blocks()` 직전 (~L156)에 standalone 함수로 추가.

#### 설계 결정

**DECISION: `n_rounds <= 1`이면 기존 `torch.topk` fallback**

`n_iterative_rounds=1`(기본값)일 때 완전히 one-shot과 동일한 코드 경로를 따른다. 기존 실험 재현성에 영향 없음.

**DECISION: 중간 라운드는 amax-head, 최종 라운드만 per-head top-k**

중간 라운드에서 per-head top-k를 하면 head마다 다른 pool이 생겨서 "공통 pool 좁히기"라는 iterative의 개념이 깨진다. pool은 모든 head에 공유되어야 하므로 amax를 써서 단일 pool을 유지한다. 최종 라운드에서만 per-head로 분리해서 KV cache 구조(head별로 다른 positions 허용)를 활용한다.

**DECISION: 선형 schedule (PLAN.md의 고정 20%p/round 대신)**

PLAN.md에서 제안한 `[0.8N, 0.6N, 0.4N, final_k]`는 `final_k ≈ 0.2N`일 때만 자연스럽다. `final_k`가 다를 때(e.g. `total_keep_ratio=0.5`) 라운드 수가 실질적으로 줄어들거나 역순이 된다. 대신 `n_free`에서 `final_k`까지 n_rounds 등분 선형 schedule을 채택:

```
round_i_keep = ceil(n_free - (n_free - final_k) * (i+1) / n_rounds)
```

이 방식은 final_k와 무관하게 n_rounds가 항상 고정 횟수의 실질적인 pruning을 수행한다.

**DECISION: `schedule[-1] = k_clamped` 강제**

부동소수점 반올림으로 마지막 round_k가 final_k ± 1이 되는 엣지 케이스를 방지하기 위해 마지막 값을 명시적으로 덮어쓴다.

#### compress()와의 연결

기존 코드:
```python
topk_within_free = torch.topk(free_scores, k=k, dim=-1).indices
```

변경 후:
```python
topk_within_free = _iterative_topk(free_scores, k, n_rounds=self.n_iterative_rounds)
```

반환 shape이 동일 `(num_kv_heads, k)`이므로 이후 코드(scored_image_positions, VizCapture 기록)는 무변경.

---

### 2. `ImageTokenTopKPress` 필드 추가

```python
n_iterative_rounds: int = 1
```

기본값 1 = one-shot. 모든 subclass(`OracleImageTeacherPress`, `ProbeImageTeacherPress`, `H2OImageOnlyPress`)가 상속받아 자동 지원.

`OracleAllTokenPress`는 `ImageTokenTopKPress`를 상속하지 않고 `BasePress`를 직접 상속하므로 무영향.

---

### 3. CLI 인수

```
--n_iterative_rounds INT   (default: 1)
```

`build_press()`의 `forced_kwargs`에 포함되어 `OracleImageTeacherPress`, `ProbeImageTeacherPress`, `H2OImageOnlyPress` 모두에 전달된다.

---

### 4. 사용 예시

```bash
# Iterative-4 probe (EXP-20260417-001 메인 실험)
python evaluate_image_teacher_pruning.py \
  --mode probe \
  --probe_model_name /workspace/hd/artifacts/zap/probe_model_scienceqa \
  --total_keep_ratio 0.20 \
  --n_iterative_rounds 4 \
  --dataset_path /workspace/zap/data/MileBench/DocVQA/DocVQA.json \
  --output_dir /workspace/zap/artifacts/exp_20260417_001/probe_iterative4

# One-shot baseline (비교용, n_iterative_rounds=1 기본값)
python evaluate_image_teacher_pruning.py \
  --mode probe \
  --probe_model_name /workspace/hd/artifacts/zap/probe_model_scienceqa \
  --total_keep_ratio 0.20 \
  --dataset_path /workspace/zap/data/MileBench/DocVQA/DocVQA.json \
  --output_dir /workspace/zap/artifacts/exp_20260417_001/probe_oneshot
```

---

### 5. 알려진 제약 / 향후 작업

- **Score 재계산 없음**: 모든 라운드에서 prefill 시 계산된 동일한 probe score를 재사용한다. 라운드 간 score rank가 매우 안정적이라면 iterative ≡ one-shot이 될 수 있음 (실패 조건으로 명시됨).
- **추가 prefill 오버헤드**: `torch.topk` 호출이 n_rounds번 실행되지만, n_free가 수백 토큰 수준이라 latency 영향은 무시 가능할 것으로 예상.
- **`n_rounds=2`나 `n_rounds=3`은 미실험**: 4라운드가 PLAN.md 기준이며, 다른 값은 ablation으로 남김.

---

## Implementation Log — Phase 2 보조 metric 수집

**날짜**: 2026-04-17
**구현자**: Claude (ssoree912 요청)
**상태**: 구현 완료, 미실행

---

### 구현 파일 목록

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `kvzap/phase2_metrics.py` | **신규 생성** | Overlap ratio, Spatial entropy 계산 함수 모음 |
| `evaluate_image_teacher_pruning.py` | **수정** | `--collect_phase2_metrics` 플래그 + 루프 내 수집 코드 추가 |

---

### 1. `kvzap/phase2_metrics.py` — 설계 의도

#### 왜 별도 파일인가

metric 계산 로직을 평가 루프에 직접 인라인하면 루프가 복잡해지고, 나중에 분석 스크립트에서 재사용할 때 import가 어려워진다. 함수 단위로 분리해두면 `--save_viz_for_first_n`으로 저장한 `.npz` 파일을 노트북에서 오프라인으로 재분석할 때도 동일한 코드를 쓸 수 있다.

#### 함수별 설계 결정

**`compute_oracle_keep_masks(teacher_scores, n_image_keep_per_layer, head_reduce)`**
- `n_image_keep_per_layer`는 probe의 `VizCapture.n_image_keep`을 그대로 받는다. 이렇게 하면 동일한 예산으로 oracle이 어떤 토큰을 선택했을지 재현하므로, budget 차이에 의한 편향 없이 순수하게 "같은 k개 중 얼마나 겹치는가"를 측정한다.
- `head_reduce`는 probe press의 설정과 반드시 일치시켜야 한다 (기본값 `amax`). 다르면 oracle mask 자체가 probe의 관점과 다른 기준으로 만들어진다.

**`compute_overlap_ratios(probe_masks, oracle_masks)`**
- 분모를 `oracle.sum()`으로 쓴다 (probe.sum() 아님). 이는 "oracle이 선택한 토큰 중 probe가 맞춘 비율(recall)"을 측정한다. oracle을 정답 집합으로 볼 때 자연스러운 정의.
- oracle이 아무것도 선택하지 않은 레이어(keep=0)는 `NaN`으로 처리해서 평균 계산 시 무시된다.

**`compute_spatial_entropy_per_layer(keep_masks, n_image_per_image=576, grid_side=24)`**
- LLaVA-1.5 336px / patch-14 기준 576 = 24×24. 다른 해상도/패치 크기 모델은 `grid_side`를 바꿔야 한다.
- 다중 이미지인 경우 각 576-토큰 블록을 독립적인 24×24 그리드로 처리하고, 레이어당 블록 entropy의 평균을 취한다.
- 선택 토큰이 0개인 블록은 entropy 0으로 처리 (undefined 아님).
- `n_image_total`이 `n_image_per_image`의 배수가 아닌 경우 나머지 토큰은 무시(안전한 fallback).

**`summarize_phase2_sample` / `aggregate_phase2_records`**
- 샘플별 요약과 전체 집계를 분리해서, 나중에 per-dataset/per-sample 분석을 별도로 할 수 있도록 했다.

---

### 2. `evaluate_image_teacher_pruning.py` 수정 사항

#### 새 CLI 인수

```
--collect_phase2_metrics     # probe mode + teacher_dir 필요. VizCapture 자동 활성화.
--phase2_grid_side      (24) # Spatial entropy 계산용 그리드 크기
--phase2_n_image_per_image (576) # 이미지 1장당 토큰 수
```

#### 흐름 (루프 내)

```
model.generate()  →  VizCapture에 probe keep_masks 기록
        ↓
collect_phase2==True이면:
  teacher_record 로드 (이미 oracle 모드에서 쓰는 것과 같은 .pt 파일)
        ↓
  compute_oracle_keep_masks()  →  oracle_masks (n_layers, n_image) bool
        ↓
  summarize_phase2_sample(probe_masks, oracle_masks)
        ↓
  phase2_records에 append
        ↓
  viz_capture.reset()  ← save_viz 블록이 돌지 않는 경우 여기서 리셋
```

루프 종료 후:
```
aggregate_phase2_records(phase2_records)
→ {output_dir}/phase2_metrics.json  (aggregate + per_sample)
```

#### DECISION: Phase 2 수집은 non-fatal

루프 내 Phase 2 코드 전체를 `try/except` 로 감쌌다. teacher .pt 파일이 없거나 shape 불일치가 있어도 평가 루프 자체는 계속 돌아야 한다. 실패한 샘플은 조용히 skip하고 phase2_records에 포함되지 않는다.

#### DECISION: VizCapture 자동 활성화

`--collect_phase2_metrics`를 쓰면서 `--save_viz_for_first_n 0`(기본값)이어도 VizCapture를 자동으로 attach한다. 사용자가 두 플래그를 조합해야 하는 번거로움 없이, `--collect_phase2_metrics`만으로 충분히 동작한다.

#### DECISION: reset 위치

VizCapture reset이 두 곳에서 발생할 수 있다:
1. `save_viz_for_first_n > 0` 범위 내 → save_viz 블록이 reset
2. 그 외(phase2 전용 모드, 또는 save_viz 범위를 벗어난 샘플) → phase2 블록 finally에서 reset

`viz_sample_idx >= args.save_viz_for_first_n` 조건으로 판별해서 이중 reset을 방지한다.

---

### 3. 사용 예시

```bash
python evaluate_image_teacher_pruning.py \
  --mode probe \
  --probe_model_name /workspace/hd/artifacts/zap/probe_model_scienceqa \
  --teacher_dir /workspace/hd/artifacts/splus \
  --total_keep_ratio 0.20 \
  --dataset_path /workspace/zap/data/MileBench/DocVQA/DocVQA.json \
  --output_dir /workspace/zap/artifacts/exp_20260417_001/probe_phase2 \
  --collect_phase2_metrics
```

출력:
```
Phase 2 metrics (500 samples): overlap=0.712 ± 0.089, entropy=2.841 ± 0.312
```

`{output_dir}/phase2_metrics.json`:
```json
{
  "aggregate": {
    "n_samples": 500,
    "overlap_mean": 0.712,
    "overlap_std": 0.089,
    "entropy_mean": 2.841,
    "entropy_std": 0.312
  },
  "per_sample": [
    {
      "sample_id": "...",
      "overlap_per_layer": [...],
      "entropy_per_layer": [...],
      "mean_overlap": 0.731,
      "mean_entropy": 2.93
    },
    ...
  ]
}
```

---

### 4. 알려진 제약 / 향후 작업

- **Overlap은 forced-keep 토큰을 포함**: probe VizCapture의 keep_mask에는 `n_initial_keep` / `n_recent_keep` 등 강제 유지 토큰도 포함된다. oracle_masks 계산에서는 이 forced 토큰을 별도로 처리하지 않는다. 실험에서 forced_keep을 사용하지 않는 경우(기본값 0)에는 영향 없음.
- **Spatial entropy는 square grid 가정**: `n_image`가 `n_image_per_image`의 배수가 아닐 때 나머지 토큰은 무시된다. 이는 단일 이미지 실험(LLaVA-1.5)에서는 항상 576이므로 문제없다.
- **oracle_onthefly mode 미지원**: `collect_phase2_metrics`는 teacher .pt 파일이 있는 경우만 동작한다. `oracle_onthefly` mode에서의 phase2 수집은 구현하지 않았다.
