## Experiment Plan

**ID**: EXP-20260425-001
**Author**: ssoree912
**Date**: 2026-04-25
**Status**: [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned
**Parent**: EXP-20260422-001 (PPL + ROUGE ratio sweep, all-token future probe)

---

### 1. 동기 (Motivation)

우리는 Future probe 단독(FutureAllTokenPress)과 H2O 단독(H2OAllTokenPress)을 각각 평가해 왔고,
Hybrid (HybridH2OFutureAllTokenPress, `score = α·s_H2O + (1-α)·s_Future`)도 구현되어 있다.

그러나 **"왜 hybrid가 효과적인가"**에 대한 해석적 근거가 없다.

핵심 질문: **Future 점수와 Prefill(H2O) 점수의 조합이 각각 단독보다 더 나은 eviction 결정을
내리는가?** — 이를 검증하려면, 두 점수로 정의되는 4개 사분면(Quadrant) 중 어느 토큰을
제거했을 때 성능이 얼마나 달라지는지를 직접 측정해야 한다.

**실험 설계 직관:**

두 scoring 축 {Future score, Prefill(H2O) score} 에서 각 image token 을 High/Low 이진
분류하면 4 개 사분면이 생긴다:

| 사분면 | Future 점수 | Prefill 점수 | 해석 |
|--------|------------|-------------|------|
| **HH** (HighFuture-HighPrefill) | High (상위 50%) | High (상위 50%) | 두 신호 모두 "중요하다" → **가장 가치 있는 토큰** |
| **HL** (HighFuture-LowPrefill) | High | Low (하위 50%) | Future만 중요하다고 본 토큰 |
| **LH** (LowFuture-HighPrefill) | Low | High | Prefill만 중요하다고 본 토큰 |
| **LL** (LowFuture-LowPrefill) | Low | Low | 두 신호 모두 "불필요" → **가장 가치 없는 토큰** |

각 케이스에서 해당 사분면의 토큰을 evict → PPL/ROUGE 측정.

**Hybrid 정당화 논리**: Hybrid score 는 HH 토큰에 높은 종합 점수를 부여해 Keep 하고,
LL 토큰을 낮은 점수로 Evict 한다. 만약 HH eviction이 성능을 가장 많이 떨어뜨리고
LL eviction이 성능을 가장 조금 떨어뜨린다면 → Hybrid가 "옳은 토큰"을 Keep/Evict하고 있음을
사분면 분해(decomposition)로 직접 증명하는 셈이다.

---

### 2. 가설 (Hypothesis)

**H1 (HH eviction이 최악)**: HH 사분면 토큰을 evict할 때 PPL 증가량이 4 케이스 중 최대.
- 이유: 두 신호 모두 "높음" → 실제로 decode 단계에서 참조 빈도가 높은 이미지 토큰 집합.
  어느 하나의 신호만으로는 이 집합을 완전히 식별할 수 없다.

**H2 (LL eviction이 최소 손상)**: LL 사분면 eviction이 PPL 증가량 최소.
- 이유: 두 신호 모두 "낮음" → 실제로도 불필요한 토큰. Hybrid가 이를 우선 evict함이 정당화.

**H3 (HL/LH은 HH-LL 사이)**: HL, LH는 중간 성능 손상.
- 이유: 하나의 신호만으로는 중요성을 과소/과대평가하는 토큰 → 단일 신호 press의 한계.
- HL > LH 인지 LH > HL 인지는 추가 분석으로 신호 간 상대 기여도 판단 가능.

**보조 H4 (Quadrant 크기 상관관계)**: HH + LL의 합산 크기 > HL + LH 합산 크기 이면
두 신호의 **양의 상관관계**가 있음 (중요한 토큰은 두 신호 모두 높게 평가).
반대면 신호들이 **보완적(complementary)** 관계.

---

### 3. 독립변수 (What we change)

| 변수 | 값 |
|------|----|
| Eviction 사분면 | HH, HL, LH, LL (4 조건) |
| Dataset | mm-vet (218), detail_1k (1000) |

---

### 4. 종속변수 (What we measure)

#### Primary
- **PPL** (teacher-forcing, `eval_ppl.py`) — 낮을수록 좋음
- **ΔPPL** = `PPL_quadrant - PPL_full` — 절대 손실량 비교

#### Secondary
- **ROUGE-L** (free generation, `eval_rouge.py`)
- **Quadrant 크기** (image token 수 대비 사분면 비율, per-layer 평균)
- **Score 상관계수** (s_future vs s_h2o Pearson r, per-layer 평균)

---

### 5. 고정 조건 (What stays the same)

| 항목 | 값 |
|------|-----|
| 모델 | `/workspace/zap/ckpts/llava-1.5-7b-hf` |
| Future probe | `/workspace/zap/ckpts/future_probe_allL_limit100` (all-layer, all-token) |
| Eviction scope | all-token (image + text 포함, FutureAllTokenPress 설정과 동일) |
| Layer 범위 | 0..31 (모든 레이어) |
| Eviction 비율 | **사분면 크기 = 전체의 ~25%** → 각 케이스 evict 비율 동일하게 통제 |
| seed | `torch.manual_seed(0)` |
| attn_implementation | eager (H2O score 계산에 attention weights 필요) |
| dtype | bfloat16 |
| eval_samples | mm-vet: 218, detail_1k: 1000 (전체) |

> **Eviction 비율 통제 방법**: 각 레이어·샘플마다 quadrant 크기가 다를 수 있음.
> 공정 비교를 위해 모든 케이스에서 `K = floor(N_image * 0.25)` 토큰을 evict.
> 해당 사분면에 K개 이상의 토큰이 있으면 그 중 score 기준 상위 K개를 evict.
> 해당 사분면이 K개 미만이면 사분면 전체 evict (실제 비율 기록, caveat 명시).

---

### 6. 베이스라인

| 방법 | 설명 |
|------|------|
| `full` | KV eviction 없음 (PPL/ROUGE upper bound) |
| `future_all_token` (k=0.75) | Future probe 단독, 동일한 25% eviction — HH/HL/LH/LL 혼합 evict |
| `h2o_all_token` (k=0.75) | H2O 단독, 동일한 25% eviction |
| `hybrid_h2o_future_all_token` (k=0.75, α=0.5) | 현재 hybrid, 동일 25% eviction |

> 4개 사분면 방법 + 4개 베이스라인 + full = **총 9 조건 × 2 datasets**.

---

### 7. 예상 결과

PPL on detail_1k (full ≈ 2.95):

| Method | ΔPPL (예상) | 설명 |
|--------|------------|------|
| evict HH | +0.20 ~ +0.35 | 최대 손상 — 두 신호 모두 중요하다고 한 토큰 제거 |
| evict LH | +0.10 ~ +0.20 | H2O만 중요하게 본 토큰 제거 |
| evict HL | +0.08 ~ +0.18 | Future만 중요하게 본 토큰 제거 |
| evict LL | +0.02 ~ +0.08 | 최소 손상 — 두 신호 모두 불필요하다고 한 토큰 제거 |
| future_all_token k=0.75 | +0.05 ~ +0.12 | 단일 신호 기준 25% evict (혼합) |
| hybrid k=0.75 | +0.04 ~ +0.10 | 복합 신호 기준 25% evict |
| full | +0.00 | 기준선 |

> HL vs LH 상대 순서는 신호 간 상대 기여도에 달려 있어 사전 예측 불확실.

---

### 8. 판단 기준 (Success Criteria)

**Primary (H1 지지)**: `ΔPPL(evict_HH) > ΔPPL(evict_HL)` AND `ΔPPL(evict_HH) > ΔPPL(evict_LH)`
on detail_1k.

**Primary (H2 지지)**: `ΔPPL(evict_LL) < ΔPPL(evict_HL)` AND `ΔPPL(evict_LL) < ΔPPL(evict_LH)`
on detail_1k.

**Secondary**: `hybrid_all_token ΔPPL ≤ min(future_k0.75, h2o_k0.75) ΔPPL` — hybrid가 단일
신호보다 우수.

**Failure Condition**: `ΔPPL(evict_HH) ≤ ΔPPL(evict_LL)` — HH가 LL보다 덜 중요하다는
뜻 → 두 신호 중 하나가 실제 decode utility와 무상관하거나 반상관(anticorrelated).
이 경우 probe 재학습 또는 신호 정의 재검토 필요.

---

### 9. 이 실험으로 증명할 수 없는 것

- **인과관계**: "HH 토큰이 중요하기 때문에 두 신호 모두 높다"는 방향성 — 역인과 가능.
- **레이어별 분해**: 사분면은 레이어 평균으로 정의. 레이어별 quadrant 구성은 다를 수 있음.
- **모델 일반화**: LLaVA-1.5-7B 단일 모델, image-only 단일 데이터 도메인.
- **전체 keep_ratio 스윕**: 본 실험은 ~75% keep (25% evict) 한 점만 측정.
  다른 eviction budget에서 사분면 순위가 뒤집힐 수 있음 (추후 §14 확장).

---

### 10. 예상 런타임 / 리소스

| 항목 | 값 |
|------|-----|
| GPU | RTX 4090 1장 |
| 조건 수 | 9 conditions × 2 datasets = 18 runs |
| 예상 시간 | mm-vet: ~2min/run × 9 = 18min, detail_1k: ~30min/run × 9 = 270min |
| 총 예상 | ~5 hours (단일 GPU) |
| 디스크 | ~10 MB |
| 출력 경로 | `/workspace/zap/artifacts/EXP-20260425-001/<dataset>/<method>/result.json` |

---

### 11. 구현 세부사항

#### 11.1 신규 Press: QuadrantEvictionPress

`/workspace/zap/kvpress/presses/image_token_press.py` 에 추가.

```python
@dataclass
class QuadrantEvictionPress(BasePress):
    """Evict image tokens from a specific (future_score, h2o_score) quadrant.

    Quadrant key: "HH" | "HL" | "LH" | "LL"
      - H = top-50% of the respective score distribution (image tokens only)
      - L = bottom-50%

    Evicts exactly K = floor(N_image * evict_ratio) tokens from the target
    quadrant (by magnitude within the quadrant).  If the quadrant has fewer
    than K tokens, evicts the entire quadrant (actual ratio recorded in logs).

    Requires eager attention for H2O score computation.
    """

    future_probe_name: str = ""
    quadrant: str = "LL"          # one of "HH", "HL", "LH", "LL"
    evict_ratio: float = 0.25     # fraction of ALL image tokens to evict
    _future_probe: Optional[KVzapModel] = field(default=None, init=False, repr=False)
    _loaded_fu_name: Optional[str] = field(default=None, init=False, repr=False)
    # image token positions — set per sample via set_image_positions()
    _image_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        self._image_positions = image_positions

    def clear_sample_context(self) -> None:
        self._image_positions = None

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        ...
        # 1. compute s_h2o (sum over all prefill queries, mean over heads)
        # 2. compute s_fu  (future probe output at this layer)
        # 3. binarize: High = top-50% rank within image positions
        # 4. select quadrant tokens
        # 5. evict min(K, |quadrant|) from quadrant
```

#### 11.2 구체 알고리즘

```
N_image = len(image_positions)
K = floor(N_image * evict_ratio)        # target evict count

# 이미지 토큰에 대한 두 점수 추출
s_h2o_img = s_h2o[image_positions]      # (N_image,)
s_fu_img  = s_fu[image_positions]       # (N_image,)

# High/Low 이진 마스크 (median 기준)
h2o_median = s_h2o_img.median()
fu_median  = s_fu_img.median()
is_high_h2o = s_h2o_img >= h2o_median  # (N_image,) bool
is_high_fu  = s_fu_img  >= fu_median

# 사분면 마스크 선택
quadrant_masks = {
    "HH": is_high_fu & is_high_h2o,
    "HL": is_high_fu & ~is_high_h2o,
    "LH": ~is_high_fu & is_high_h2o,
    "LL": ~is_high_fu & ~is_high_h2o,
}
q_mask = quadrant_masks[self.quadrant]  # (N_image,)

# 사분면 내에서 combined score 기준 상위 K개 evict
# (evict "most representative" tokens of the quadrant)
combined = s_h2o_img + s_fu_img   # magnitude proxy (already normalized)
q_scores = combined.clone()
q_scores[~q_mask] = -inf          # 사분면 밖은 제외

n_evict = min(K, q_mask.sum().item())
evict_local_idx = topk(q_scores, n_evict).indices  # image-local index
evict_global_idx = image_positions[evict_local_idx]  # seq-level index

# keys/values에서 evict_global_idx 제거
```

> **Layer-wise 일관성**: 각 레이어에서 독립적으로 사분면 계산. 레이어마다 분포가 다를 수 있음.
> 단순화를 위해 레이어 간 통합 사분면은 사용하지 않음 (추후 분석 가능).

#### 11.3 eval_ppl.py / eval_rouge.py 연동

`--method quadrant_eviction --quadrant HH --evict-ratio 0.25` 옵션 추가.
기존 `FutureAllTokenPress`, `H2OAllTokenPress`, `HybridH2OFutureAllTokenPress` 에서
total_keep_ratio=0.75 로 4개 베이스라인 실행.

#### 11.4 Sweep 스크립트

```bash
# experiments/EXP-20260425-001/run_sweep.sh
DATASETS=("mm-vet" "detail_1k")
METHODS=("HH" "HL" "LH" "LL" "future_k075" "h2o_k075" "hybrid_k075" "full")

for DS in "${DATASETS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    python /workspace/zap/eval_ppl.py \
      --method quadrant_eviction \
      --quadrant $METHOD \
      --evict-ratio 0.25 \
      --future-probe-name /workspace/zap/ckpts/future_probe_allL_limit100 \
      --attn-implementation eager \
      --data-path /workspace/data/${DS} \
      --output-dir /workspace/zap/artifacts/EXP-20260425-001/${DS}/${METHOD}/ppl
    python /workspace/zap/eval_rouge.py \
      ... (동일 옵션)
  done
done
```

---

### 12. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 사분면 크기 불균형 (상관관계 높으면 HL/LH 거의 없음) | K개 eviction 불가 → 실제 비율 낮아짐 | 각 레이어 quadrant 크기 로깅; caveat 명시 |
| Median-split의 tie (짝수 N_image → median이 경계) | 마스크 불안정 | `strictly greater than median` = High로 통일 |
| eager attention → OOM (detail_1k, long context) | 일부 sample 실패 | 기존 EXP-20260420-003 패턴 재사용 (layer-by-layer consumption) |
| score normalization 차이 (s_h2o vs s_fu scale 다름) | combined score proxy 부정확 | quadrant 내부 eviction 순서에만 사용 → scale 영향 없음; 사분면 경계는 median으로 결정 |
| future probe 미load (레이어 0..31 전부 필요) | 속도 저하 | 기존 FutureAllTokenPress 와 동일 패턴, probe 재사용 |

---

### 13. 분석 계획

#### 13.1 Main Result Table

```
Table A. ΔPPL vs Full (detail_1k)
Method         | ΔPPL   | 비고
---------------|--------|------
evict HH       | ?      | 두 신호 모두 중요하다는 토큰
evict HL       | ?      | Future만 중요
evict LH       | ?      | H2O만 중요
evict LL       | ?      | 두 신호 모두 불필요
future k=0.75  | ?      | 단일 신호 혼합 evict (baseline)
h2o k=0.75     | ?      | 단일 신호 혼합 evict (baseline)
hybrid k=0.75  | ?      | 복합 신호 혼합 evict (baseline)
full           | 0.00   | 기준
```

#### 13.2 시각화

- **Bar chart**: 4 사분면 + 3 baseline의 ΔPPL, 2 datasets side-by-side.
  → HH > (HL, LH) > LL 패턴이 나타나는지 한눈에 확인.
- **Scatter plot**: per-sample s_future vs s_h2o (image token 기준) — 상관계수 & 4분면 분포.
- **Quadrant size bar**: 각 사분면이 전체 image token의 몇 %인지.

#### 13.3 가설 검증 체크리스트

- [ ] H1: `ΔPPL(HH) > max(ΔPPL(HL), ΔPPL(LH))` on detail_1k
- [ ] H2: `ΔPPL(LL) < min(ΔPPL(HL), ΔPPL(LH))` on detail_1k
- [ ] H3: `HL ≠ LH` — HL vs LH 비교로 두 신호의 상대 중요도 파악
- [ ] H4: 사분면 크기에서 HH+LL > HL+LH → 두 신호 양의 상관관계 확인

---

### 14. 다음 실험 (조건부)

#### H1/H2 지지 시 (예상 메인 케이스)
- **EXP-??-001**: evict_ratio sweep (0.10, 0.25, 0.50) — 어느 budget에서도 HH > LL?
- **EXP-??-002**: per-layer 분석 — 어느 레이어에서 사분면 효과가 가장 강한가?

#### H1/H2 실패 시
- **원인 분석**: future probe와 H2O score의 실제 Pearson r 측정.
  r < 0.1 이면 신호가 아예 무관 → 사분면 정의 자체를 재검토.
  r > 0.8 이면 HL/LH 사분면이 너무 작아 실험 설계 자체가 약점.
- **대안 실험**: median 대신 quartile (상위 25% = High, 하위 25% = Low) 로 사분면 재정의.

---

### 15. 체크리스트

#### 실험 시작 전
- [ ] `QuadrantEvictionPress` 구현 및 단위 테스트 (`--eval-samples 3`, 사분면 크기 로깅 확인)
- [ ] `eval_ppl.py`, `eval_rouge.py` 에 `quadrant_eviction` method 분기 추가
- [ ] Future probe ckpt 경로 확인 (`/workspace/zap/ckpts/future_probe_allL_limit100`)
- [ ] `run_sweep.sh` 작성 및 실행 권한 부여
- [ ] `full` baseline PPL이 EXP-20260422-001 결과와 일치하는지 smoke test

#### 실험 후
- [ ] 18 runs 전부 result.json 생성 (ppl + rouge)
- [ ] per-layer quadrant 크기 통계 로깅 확인
- [ ] H1 ~ H4 verdict 기록
- [ ] `RESULT.md` 작성
- [ ] `experiments/README.md` 인덱스 업데이트
