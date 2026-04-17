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
