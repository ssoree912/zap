# Experiment Plan

**ID:** EXP-20260420-002
**Author:** ssoree912
**Date:** 2026-04-20
**Status:** [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned
**Parent:** EXP-20260418-001 (Hybrid α sweep), EXP-20260420-001 (Per-layer Hybrid)
**Purpose tag:** Paper main baseline — prefill observable vs unobservable-at-prefill signals

---

## 1. Motivation

### 1.1 새로운 논문 서사 (핵심 pivot)

**핵심 질문**:

> *Prefill 시점에 LLaVA가 **직접 볼 수 있는 attention heuristic (H2O류)** 만으로 충분한가, 아니면 prefill 시점에는 직접 볼 수 없는 **미래 decode usefulness (Future probe)** 를 예측하는 신호가 추가로 필요한가?*

이 프레이밍이 PostVision MLP와 Future MLP를 분리해서 보게 만듭니다:

| 신호 | LLaVA prefill에서 직접 접근 가능? | MLP가 필요한가? | 과학적 기여 |
|---|---|---|---|
| Raw prefill attention (H2O, raw PV-attn) | ✅ 예 (계산만 하면 됨) | ❌ 불필요 (heuristic 그대로 사용) | 기존 문헌의 baseline |
| **PostVision MLP** (distill post-vision attn) | ✅ 예 | ⚠️ **불필요 — 같은 모델의 같은 attn을 MLP로 근사한 것뿐** | 약함 |
| **Future MLP** (decode→image attention 예측) | ❌ 아니오 (decode가 일어나기 전) | ✅ **필수** (prefill 시점에 접근 불가한 신호) | 강함 |

→ PostVision MLP는 reviewer 입장에서 "왜 attention을 그냥 쓰지 않고 MLP로 다시 배웠냐?" 라는 질문을 피하기 어렵습니다.
→ Future MLP는 "현재는 없지만 미래에 중요해질 token 예측"이라는 명확한 기여를 가집니다.

### 1.2 EXP-001 결과 반영 (broadcast noise 가설 기각)

per-layer sweep 12 cells 중 significant win 0, H1 FAIL (평균 Δ = -0.0014 / -0.0040). Future signal의 early-layer broadcast noise가 주 성능 저하 원인이 아니라는 결과. 이에 따라 **"PV가 좋은 signal이냐"를 더 단순한 prefill baseline과 대조**하려 했으나, 위 1.1 재프레이밍으로 **"PV MLP 자체를 메인에서 빼고 H2O를 baseline으로 쓰자"** 쪽으로 pivot.

### 1.3 논문 서사 (pivot 후)

> **Claim 1** (prefill observable 충분성 검증)
> 현재 prefill 시점에서 관찰 가능한 generic heavy-hitter (H2O-prefill)만으로는 image-only KV pruning에 충분하지 않다.
>
> **Claim 2** (future prediction의 고유 기여)
> Future MLP는 decode-time reuse signal을 예측하며, 이는 prefill 시점에 직접 관찰 불가능한 정보다. 단순 attention heuristic을 MLP로 근사한 것과 본질적으로 다르다.
>
> **Claim 3** (현재 saliency + 미래 utility 결합)
> H2O(현재 saliency) + Future(미래 utility) 의 blend는 두 신호가 상보적임을 보여준다.

---

## 2. Hypothesis

**H1 (H2O-prefill 단독 부족)**: H2O-prefill은 6 dataset 평균 ROUGE-L/Accuracy에서 Future-only 및 Hybrid(H2O+Future) 각각보다 **낮다** (Δ ≥ +0.003 대 Future, Δ ≥ +0.005 대 Hybrid).
- 이유: H2O는 현재 prefill 관찰만으로 "heavy hitter"를 뽑는 것이라 미래 decode에서 재참조될 token을 예측 불가.

**H2 (Future의 고유 기여)**: 최소 2 dataset에서 Future-only > H2O-prefill (Δ ≥ +0.005).
- 이유: prefill에서 보이지 않던 token이 decode 단계에서 중요해지는 경우가 존재.

**H3 (Hybrid > max(단독))**: Hybrid(H2O+Future, α=0.5) 가 최소 1 dataset에서 Future-only와 H2O-only 모두보다 동등 이상 (Δ ≥ 0).
- 이유: 현재·미래 신호 상보성이 있다면 평균이 둘 중 하나보다 나아야 함.

### Success Target (main claim)

- **H1 지지**: 평균에서 Future ≥ H2O + 0.003 **AND** Hybrid ≥ H2O + 0.003.
- **H2 지지**: 2 dataset 이상에서 Future > H2O (Δ ≥ +0.005).
- **H3 지지**: 1 dataset 이상에서 Hybrid ≥ max(H2O, Future).

### Failure Mode

- H2O ≥ Future (평균): Claim 2 약화. 재프레이밍 필요 — "Future MLP의 학습 범위(last-8)가 너무 좁아 representation을 제대로 뽑지 못함" 가설로 선회, Phase 6 (all-32-layer 재학습) 복귀 검토.
- Hybrid < max(H2O, Future): 상보성 부재. α sweep 재필요 or per-layer 설계 재고.

---

## 3. Independent Variables

### 메인 비교군 (본 실험)

- **Scoring method**: {H2O-prefill, Hybrid(H2O+Future)}
- **Keep ratio**: **0.2** (단일, — MileBench 전 dataset에서 돌림)
- **α for Hybrid**: **0.5** (모든 layer 동일 blend, per-layer gating 없음)
- **Future-only**: EXP-20260418-001 결과 재사용 (신규 실행 불필요)

### 방법별 정의

#### H2O-prefill (현재 prefill saliency, raw attention)
```
s_i^{H2O}(l) = (1/H) * Σ_h Σ_{q ∈ Q_prefill} A^{(l,h)}[q, i]
```
- `--mode h2o_image_only`
- 코드 수정 불필요 (기존 `H2OImageOnlyPress`)

#### Future-only (미래 decode usefulness)
```
s_i^{Future}(l) = MLP_Future(h_l[i])   // 학습: decode→image attention
```
- `--mode future --selected_layer_indices 24..31`

#### Hybrid(H2O + Future) **[본 실험의 신규 구현]**
```
s_i^{Hybrid}(l) = 0.5 * softmax(s_i^{H2O}(l)) + 0.5 * softmax(s_i^{Future}(l))
```
- `--mode hybrid_h2o_future --alpha 0.5`
- `kvpress/presses/image_token_press.py::HybridH2OFuturePress` (신규 클래스)
- `evaluate_image_teacher_pruning.py` CLI mode 신규 추가
- Per-layer gating 없이 전 32 layer 동일 α 적용 (간단화).

---

## 4. Dependent Variables

### Primary metrics
- **ROUGE-L** (clevr_change, alfred, iedit, mmcoqa, spot_the_diff)
- **Accuracy** (webqa)

### Secondary metrics
- vs H2O-prefill Δ (H1 검증)
- vs Future-only Δ (H2·H3 검증)
- H2O·Future 의 top-k keep mask IoU (상보성 직관 확인용, optional)

---

## 5. Fixed Conditions

- **Model**: `/workspace/zap/ckpts/llava-1.5-7b-hf` (LLaVA-1.5-7B)
- **Future probe**: `/workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31`
- **Dataset**: 6개 MileBench — spot_the_diff, clevr_change, webqa, alfred, iedit, mmcoqa
- **Keep ratio**: 0.2
- **Prompt style**: `look_milebench`
- **Attention impl**: `eager` (`output_attentions=True` 필요, H2O 및 Hybrid 둘 다)
- **Truncation**: `--truncate_like_lookm` (OOM 방지)
- **Hardware**: RTX 4090 × 2 (GPU 0 = H2O, GPU 1 = Hybrid, 병렬)

---

## 6. Baselines & Comparisons

### 메인 (본 논문 Table 1)

| 방법 | 축 | 신호 가용성 | mode | 출처 |
|---|---|---|---|---|
| **H2O-prefill** | current saliency | prefill 시점 접근 가능 | `h2o_image_only` | **본 실험** |
| **Future-only** | future utility | prefill 시점 접근 불가 → MLP 예측 | `future` | EXP-20260418-001 v4_eval 재사용 |
| **Hybrid(H2O+Future, α=0.5)** | 둘의 blend | 혼합 | `hybrid_h2o_future` | **본 실험** |
| Random | 비교 하한 | — | (추후 추가) | — |

### 부록/ablation (PV MLP는 main에서 제외)

| 방법 | 위치 | 비고 |
|---|---|---|
| PostVision MLP | Appendix / Ablation only | "같은 모델의 prefill attention을 다시 MLP로 근사" — 메인 기여로는 약함 |
| Hybrid(PV + Future) 기존 결과 | Appendix (EXP-20260418-001, EXP-20260420-001 자료 참조) | per-layer/global sweep 결과 보존 |
| raw PostVision-attention heuristic (MLP 없이 직접 attn 사용) | (선택적 future experiment) | reviewer 방어용. 본 실험 범위 밖. |

### 재사용 가능 결과

| 방법 | keep_ratio | 경로 |
|---|---|---|
| Future-only k=0.2 | 0.2 | `artifacts/EXP-20260418-001/v4_eval/<ds>/future_k0p2/` |
| PostVision MLP k=0.2 (appendix용) | 0.2 | `artifacts/EXP-20260418-001/v4_eval/<ds>/pv_k0p2/` |

### 신규 실행 (본 실험)

- H2O-prefill: 6 ds × k=0.2 = **6 runs** (GPU 0)
- Hybrid(H2O+Future): 6 ds × k=0.2 × α=0.5 = **6 runs** (GPU 1)
- **Total 12 runs**, 2 GPU 병렬.

---

## 7. Expected Results

### 정성적 예측

- **H2O-prefill**: 단순 saliency라 sink/common-bg token을 과대평가. Change detection (clevr_change, spot_the_diff) 에서만 경쟁력.
- **Future**: decode-reuse 반영. alfred, mmcoqa 에서 H2O 보다 유의미하게 우위 기대.
- **Hybrid(H2O + Future, α=0.5)**: 평균적으로 H2O 와 Future 사이 또는 약간 상회. 상보성이 있는 dataset(iedit, webqa) 에서 두 단독보다 약간 나을 가능성.

### 정량적 예측 (ROUGE-L / Accuracy, k=0.2)

| Dataset | H2O | Future | Hybrid(H2O+F) |
|---|---|---|---|
| spot_the_diff | ~0.18 | ~0.18 | ~0.19 |
| clevr_change | ~0.13 | ~0.13 | ~0.13 |
| webqa (Acc) | ~0.60 | ~0.60 | ~0.61 |
| alfred | ~0.28 | ~0.29 | ~0.29 |
| iedit | ~0.11 | ~0.11 | ~0.12 |
| mmcoqa | ~0.37 | ~0.37 | ~0.38 |

---

## 8. Success Criteria

### Primary (main claim 지지)
- **H1**: 평균 Δ(Future − H2O) ≥ +0.003 **AND** Δ(Hybrid − H2O) ≥ +0.003.
- **H2**: 2 dataset 이상에서 Future > H2O (Δ ≥ +0.005).
- **H3**: 1 dataset 이상에서 Hybrid ≥ max(H2O, Future).

### Secondary
- H2O-prefill 이 random baseline(미포함) 보다 현저히 상회 (baseline sanity).
- Hybrid(α=0.5) 가 최소한 H2O와 Future 의 평균 정도는 됨 (blend가 해를 끼치지 않음).

### Failure Mode
- H2O ≈ Future ≈ Hybrid (Δ < 0.003 전반): image-only top-k setting에서 scoring method 효과가 본질적으로 작음. Keep ratio 축소(0.1) 또는 더 challenge한 dataset mix 로 확장 검토.
- H2O > Future: claim 2 기각. Future MLP 재학습 (Phase 6) 방향 복귀.

---

## 9. What This Experiment Cannot Prove

- **PostVision 계열 신호 전반의 우열**: raw PV attention heuristic (MLP 없이 직접 사용)은 appendix 선택사항으로만 검토. 별도 sweep 필요.
- **α ≠ 0.5 에서의 Hybrid 최적점**: 이번 실험은 α=0.5 단일. α sweep은 H1·H3 이 지지되면 follow-up.
- **다른 keep_ratio / 다른 모델 일반화**: k=0.2 단일, LLaVA-1.5-7B 단일.
- **MileBench 전체 일반화**: 6 dataset subset 기준.
- **Claim 2 를 전면적 "PV MLP는 unnecessary"로 확장**: 본 실험은 PV MLP를 메인에서 제외한 것이지 "열등함"을 입증한 것은 아님. 그 주장을 하려면 raw PV-attn heuristic 실험 필요.

---

## 10. Runtime / Resources

- **신규 runs**: 12 (6 ds × 2 methods, k=0.2)
- **Run당 시간**: ~2–3 min (eager attention, 작은 dataset)
- **총 시간**: ~15–20 min (2 GPU 병렬)
- **디스크**: ~12 MB
- **출력 경로**: `/workspace/zap/artifacts/EXP-20260420-002/h2o_prefill_and_hybrid/<ds>/{h2o_k0p2,hybrid_h2o_future_a050_k0p2}/`

---

## 11. Implementation Details

### 11.1 Press 코드 (신규 클래스)

**`kvpress/presses/image_token_press.py::HybridH2OFuturePress`** (✅ 추가 완료)

```python
@dataclass
class HybridH2OFuturePress(ImageTokenTopKPress):
    future_probe_name: str = ""
    alpha: float = 0.5                             # H2O weight
    future_blend_layers: tuple[int, ...] = ()     # optional per-layer gating (본 실험 미사용)

    def score_image_tokens(self, module, hidden_states, keys, values, attentions, ...):
        # H2O: (1, 1, n_image) — sum over prefill Q, mean over heads
        importance   = attentions[0].sum(dim=1).float()
        h2o_per_tok  = importance.mean(dim=0)
        s_h2o        = h2o_per_tok[image_positions].view(1, 1, -1)

        # Future probe MLP: (1, 1, n_image)
        fu_layer = self._future_probe.layers[module.layer_idx].to(...).eval()
        s_fu     = fu_layer(image_hidden_states).transpose(1, 2)

        # Softmax normalize → blend
        s_h2o_norm = torch.softmax(s_h2o.float(), dim=-1).to(dtype)
        s_fu_norm  = torch.softmax(s_fu.float(), dim=-1).to(dtype)
        return self.alpha * s_h2o_norm + (1 - self.alpha) * s_fu_norm
```

### 11.2 CLI (신규 mode)

**`evaluate_image_teacher_pruning.py`** (✅ 추가 완료)
- `--mode hybrid_h2o_future` 추가.
- `--future_probe_name` 만 필요 (PV probe 불필요).
- `needs_output_attentions` 자동 활성화.

### 11.3 Sweep 스크립트

**`artifacts/EXP-20260420-002/h2o_future_sweep.sh`** (✅ 작성 완료, 현재 실행 중)
- GPU 0 lane: H2O-prefill × 6 ds (sequential)
- GPU 1 lane: Hybrid(H2O+Future, α=0.5) × 6 ds (sequential)
- 두 lane 병렬 실행.

### 11.4 재현 커맨드 예시 (Hybrid)

```bash
python /workspace/zap/evaluate_image_teacher_pruning.py \
  --mode hybrid_h2o_future \
  --dataset_path /workspace/zap/data/MileBench/CLEVR-Change/CLEVR-Change.json \
  --image_root /workspace/zap/data/MileBench/CLEVR-Change/images \
  --image_column images_path \
  --output_dir /workspace/zap/artifacts/EXP-20260420-002/h2o_prefill_and_hybrid/clevr_change/hybrid_h2o_future_a050_k0p2 \
  --implementation_model_name /workspace/zap/ckpts/llava-1.5-7b-hf \
  --future_probe_name /workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31 \
  --alpha 0.5 \
  --image_keep_ratio 0.2 \
  --prompt_style look_milebench \
  --attn_implementation eager \
  --truncate_like_lookm \
  --look_dataset_name DocVQA \
  --look_model_name zap_docvqa \
  --look_result_root /workspace/zap/artifacts/combine_prob \
  --device cuda:1
```

### 11.5 결과 통합

EXP-20260418-001 (Future-only), 본 실험 (H2O, Hybrid(H2O+Future)) 결과를 `analyze_results_mainaxis.py` (신규) 로 합쳐 논문 Table 1 markdown export. PV MLP 는 appendix 로 분리.

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| H2O-prefill 이 예상보다 강해서 Future 와 차이 없음 | Claim 2 약화 | Keep mask overlap 분석. k=0.1 로 축소 sweep 검토. |
| H2O 모드 / Hybrid 모드 OOM (긴 sequence) | 일부 run 실패 | `--truncate_like_lookm` 활성 (기본), 실패 시 `--limit` 적용. |
| Future probe가 last-8 만 학습되어 blend 전체 layer에서 노이즈 | Hybrid 약세 | H1 검증 후 per-layer gating (EXP-001 방식) 복귀 검토. |
| 2-GPU 병렬 context stress → CUDA fault | sweep 중단 | `CUDA_MODULE_LOADING=LAZY` 세팅. `nvidia-smi` 모니터. |
| PV MLP 제거로 인한 reviewer 방어 약화 | 질문 수증 | 부록에 raw PV-attn heuristic 소규모 결과 (follow-up 실험) 추가 계획 명시. |

---

## 13. Analysis Plan

### 13.1 논문용 Main Table 1 (신규 구조)

| Dataset | k | H2O-prefill | Future | Hybrid(H2O+Future) | Δ(Hyb − H2O) | Δ(Hyb − Fut) |
|---|---|---|---|---|---|---|
| spot_the_diff | 0.2 | | | | | |
| clevr_change | 0.2 | | | | | |
| webqa (Acc) | 0.2 | | | | | |
| alfred | 0.2 | | | | | |
| iedit | 0.2 | | | | | |
| mmcoqa | 0.2 | | | | | |
| **Average** | | | | | | |

### 13.2 Claim 검증

1. **Claim 1**: 평균 Δ(Future − H2O), Δ(Hybrid − H2O).
2. **Claim 2**: per-dataset Future vs H2O 순위 — 2+ 에서 Future > H2O 면 지지.
3. **Claim 3**: Hybrid 가 max(H2O, Future) 이상인 dataset 수.

### 13.3 부록 (appendix) Table

| Dataset | k | H2O | PV MLP (legacy) | Hybrid(PV+Future, best α, per-layer) | Future | Hybrid(H2O+Future) |
|---|---|---|---|---|---|---|

- PV MLP는 **"same-model attention을 MLP로 근사한 것"** 이라는 caveat 동반.
- Hybrid(PV+Future) 는 EXP-20260418-001/001 결과 복원.

### 13.4 추가 분석 (optional)

- H2O vs Future top-k keep mask IoU: 두 신호가 실제로 다른 token을 선택하는지 시각화.
- 생성 길이 분포 (ROUGE-L length bias 점검).

---

## 14. Next Experiments (조건부)

### 14.1 Phase 6a (H1 실패 시): Future probe all-32-layer 재학습
- Future MLP가 last-8 only 라 representation 공간이 전체 layer에서 최적화되지 않았을 가능성.
- EXP-001 PLAN §14 Phase 6 재개.

### 14.2 Phase 6b (H1 지지 시): α sweep on Hybrid(H2O+Future)
- α ∈ {0.25, 0.5, 0.75}
- 본 실험 α=0.5 만 했으므로 최적점 재확인.

### 14.3 Phase 6c (reviewer 방어): raw PostVision-attn heuristic
- MLP 없이 `att_only_postvision` 을 eager attention으로 직접 계산.
- "PV MLP는 약하지만 PV-attention 자체는 약한가?" 질문에 답.
- 별도 press `RawPVAttnImagePress` 구현 필요.

### 14.4 Phase 6d: Random / Recency baseline 추가
- 메인 Table 1 에 lower-bound 확보용.

---

## 15. Checklist

### 실험 시작 전
- [x] EXP-20260420-001 sweep 완료 (V5_PERLAYER_SWEEP_DONE)
- [x] `HybridH2OFuturePress` 클래스 작성 및 import 검증
- [x] `--mode hybrid_h2o_future` CLI 추가
- [x] Smoke test (`--limit 2`) 통과 (CLEVR-Change)
- [x] Sweep 스크립트 작성 및 실행권한
- [x] GPU 0, GPU 1 가용 확인 (nvidia-smi)

### 실험 후
- [ ] 12 runs 전부 `rc=0` (H2O × 6, Hybrid × 6)
- [ ] `metrics.json` 에 ROUGE-L / Accuracy 정상 기록
- [ ] 3-axis 메인 테이블 (`analyze_results_mainaxis.py`) 생성
- [ ] 부록 PV MLP 비교 table 병렬 생성
- [ ] Paper Table 1 markdown export + LaTeX 변환 검토
- [ ] RESULT.md 작성 (H1/H2/H3 verdict 포함)
- [ ] `experiments/README.md` 인덱스 업데이트
