# Implementation Notes

**ID:** EXP-20260418-001  
**업데이트:** 2026-04-19  
**상태:** B 방법으로 재구현 완료, v4 학습 완료, 전체 MileBench 평가 진행 중

---

## 파이프라인 진화 요약

| 세대 | 수집 포맷 | 학습 구조 | 문제점 / 개선 |
|---|---|---|---|
| v1 | per-sample `.pt`, 4 layer | 레이어별 독립 학습, records 전체 메모리 로드 | 메모리 225GB 필요 → OOM |
| v2 | per-sample `.pt`, 32 layer | lazy per-layer load | Disk I/O 7TB → HDD에서 매우 느림 |
| **v3/v4** | **공통 shard (y_pv+y_future)** | **one-pass multi-layer, per-sample softmax MSE** | **PV·Future 학습 완전 통일, SSD 기준 epoch당 ~5분** |

---

## 현재 구현 파일 (v3/v4)

| 파일 | 역할 |
|------|------|
| `collect_unified_teacher_shards.py` | Phase A: **PV + Future teacher label 동시 수집** (공통 shard 포맷) |
| `train_unified_probe_onepass.py` | Phase B: **one-pass multi-layer** 학습 (`--teacher pv|future` 선택) |
| `kvpress/presses/image_token_press.py` | Phase C: 추론 press (`FutureSupervisedImagePress`, `HybridImageTeacherPress`, `selected_layer_indices` 지원) |
| `evaluate_image_teacher_pruning.py` | Phase C: `--mode probe|future|hybrid` 지원 |

레거시 (비사용):
- `collect_future_supervised_labels.py` — v1/v2용, per-sample `.pt` 포맷
- `train_future_supervised_probe.py` — v1/v2용, records 메모리 전체 로드
- `collect_vqa_teacher_xy.py`, `collect_scienceqa_teacher_xy.py` — 기존 PostVision-only 수집 (v3에서 unified로 대체)

---

## Phase A: Unified Teacher Shard Collection

### 핵심: 한 번의 LLaVA full pass로 두 teacher label 동시 추출

스크립트: `collect_unified_teacher_shards.py`

```python
# (요약) UnifiedCollector.collect_sample_unified
# 1. prefill forward → prompt_len_mm 확인
# 2. generate (greedy, max_new_tokens=64) → full_ids
# 3. analysis forward (output_attentions=True, output_hidden_states=True)
# 4. GPU에서 바로 label 두 종류 계산 (cpu copy 전에):
for l in range(n_layers):
    attn = full_out.attentions[l][0]  # [H, full_len, full_len]
    h_l = full_out.hidden_states[l+offset][0, :prompt_len_mm, :]

    # PV: post-vision text → image attention, max over Q, mean over H
    pv_block = attn[:, pv_text_idx, :].index_select(2, image_idx)
    y_pv = pv_block.max(dim=1).values.mean(dim=0)   # [N_image]

    # Future: decode → image attention, mean over T, H
    fu_block = attn[:, decode_start:decode_end, :].index_select(2, image_idx)
    y_future = fu_block.mean(dim=0).mean(dim=0)     # [N_image]
```

### Shard 포맷 (per-row)

```
x         [R, 4096] fp16    hidden state at image token
y_pv      [R]       fp16    post-vision → image max(Q) mean(H)
y_future  [R]       fp16    decode → image mean(T, H)
layer     [R]       uint8
sample_id [R]       int32   (per-dataset 0..N-1)
token_idx [R]       int16   (0..N_image-1 within sample)
```

### 데이터 규모 (500 샘플 × 3 데이터셋 × 32 레이어)

| 데이터셋 | 이미지 수 | 토큰/샘플 | Shard 수 | 총 용량 |
|---|---|---|---|---|
| textvqa | 1 | 18,432 | 167 | ~66 GB |
| scienceqa | 1 | 18,432 | 167 | ~66 GB |
| nlvr2 | 2 | 36,864 | 250 | ~150 GB |
| **합계** | — | — | **584** | **~282 GB** |

저장: `/workspace/zap/artifacts/teacher_ssd/unified/<dataset>/shards/shard_XXXX.pt` (SSD)

### 사용 예시

```bash
python collect_unified_teacher_shards.py \
    --dataset textvqa \
    --data_dir /workspace/zap/data/textvqa/train \
    --out_dir /workspace/zap/artifacts/teacher_ssd/unified/textvqa \
    --device cuda:0 --max_new_tokens 64 --limit 500 --shard_size 50000
```

---

## Phase B: One-pass Multi-Layer Training

스크립트: `train_unified_probe_onepass.py`

### 핵심 설계: 한 epoch = 1회 shard scan × 모든 layer 동시 업데이트

```python
# 단일 shard 안에 모든 레이어 행이 interleaved 되어 있음
for batch in iter_shard_batches(...):
    # (sample_id, layer) 그룹별 per-sample softmax MSE
    key = sample_id * 64 + layer
    for group in unique(key):
        pred = softmax(MLP_layer(x_group), dim=0)
        label = y_group / y_group.sum()
        loss += MSE(pred, label)
    loss.backward()
    optimizer.step()     # 32 MLP 동시 업데이트
```

이전(v1/v2)의 "레이어별 32 epoch loop" → 모든 레이어를 **한 번에** 학습. Disk scan이 레이어 수만큼 줄어듦.

### MLP 구조 (PV·Future 공통)

```python
nn.Sequential(
    nn.Linear(4096, 512),
    nn.GELU(),
    nn.Linear(512, 1),
)
# KVzapModel 포맷 호환 — LayerNorm 없음 (inference 코드 무수정)
```

### Loss (B 방법)

```python
pred = softmax(mlp(x), dim=0)    # per-sample normalize
label = y / (y.sum() + 1e-8)     # per-sample normalize
loss = MSE(pred, label)          # MSE in probability space
```

### Sample ID 글로벌 매핑

데이터셋별로 `sample_id`가 0..N-1로 겹치므로 글로벌 ID로 remap:
```python
global_sid = dir_idx * 100000 + local_sid
```

Train/val split은 글로벌 sid 단위로 무작위 shuffle (90% train / 10% val).

### v4 하이퍼파라미터

| 항목 | 값 |
|---|---|
| hidden_dim | 512 |
| max_epochs | 20 |
| lr | 1e-3 (AdamW) |
| batch | per-shard (행 전체를 (sample_id, layer)로 그룹화) |
| seed | 42 |
| train_fraction | 0.9 |

### 사용 예시

```bash
python train_unified_probe_onepass.py \
    --shard_dirs /workspace/zap/artifacts/teacher_ssd/unified/{textvqa,scienceqa,nlvr2} \
    --teacher pv \                                    # or "future"
    --out_dir /workspace/zap/ckpts/postvision_probe_v4_20ep \
    --n_layers_model 32 --selected_layers 0 1 ... 31 \
    --mlp_hidden_dim 512 --mlp_max_epochs 20 --mlp_lr 1e-3 \
    --device cuda:0 --seed 42 \
    --progress_file /workspace/zap/artifacts/EXP-20260418-001/v4_pv_progress.log
```

---

## Phase C: Inference Press 클래스

위치: `kvpress/presses/image_token_press.py`

### `FutureSupervisedImagePress(ProbeImageTeacherPress)`

```python
@dataclass
class FutureSupervisedImagePress(ProbeImageTeacherPress):
    selected_layer_indices: tuple[int, ...] = ()

    def compress(self, module, ...):
        if self.selected_layer_indices and module.layer_idx not in self.selected_layer_indices:
            return keys, values  # untrained layer 스킵
        return super().compress(module, ...)
```

### `HybridImageTeacherPress(ImageTokenTopKPress)`

```python
@dataclass
class HybridImageTeacherPress(ImageTokenTopKPress):
    postvision_probe_name: str = ""
    future_probe_name: str = ""
    alpha: float = 0.5
    selected_layer_indices: tuple[int, ...] = ()

    def score_image_tokens(...):
        s_pv = mlp_pv(image_hidden).transpose(1, 2)
        s_fu = mlp_fu(image_hidden).transpose(1, 2)
        return alpha * softmax(s_pv) + (1 - alpha) * softmax(s_fu)
```

### 주의: untrained layer 처리 2가지 옵션

**옵션 A — Broadcast (선호)**
- Future probe는 last 8 layer만 학습.
- 추론 전에 layer 31 가중치를 layer 0-23에 복사 (`future_probe_v4_last8_20ep_bcast31`).
- 모든 레이어에서 probe 동작 → KV length 일관성 유지.

**옵션 B — selected_layer_indices 스킵**
- 트리거: `press.selected_layer_indices = (24,25,26,27,28,29,30,31)` 전달.
- Untrained layer에서 compress 스킵 → KV length 불일치 → `attention mask mismatch` 에러.
- **현재 안정성상 사용 금지, broadcast 권장.**

---

## 데이터 흐름 요약

```
[수집: collect_unified_teacher_shards.py]
dataset (textvqa/scienceqa/nlvr2)
    │
    ▼
UnifiedCollector.collect_sample_unified()
    │  prefill + greedy decode + analysis forward (1회)
    │  GPU-side: for each layer extract
    │    y_pv     = max(Q) mean(H) of attn[pv_text, image]
    │    y_future = mean(T, H) of attn[decode, image]
    │    x        = hidden_prompt[image_idx]
    ▼
UnifiedShardWriter.add_sample()
    │  append per-token rows (x, y_pv, y_future, layer, sample_id, token_idx)
    │  50000 rows/shard
    ▼
/workspace/zap/artifacts/teacher_ssd/unified/<ds>/shards/shard_XXXX.pt


[학습: train_unified_probe_onepass.py]
shards (584 files)
    │
    ▼
iter_shard_batches() — 1 epoch = shard 1회 스캔
    │  filter rows by layer ∈ selected_layers, sample_id ∈ split
    │  group by (sample_id, layer)
    ▼
compute_batch_loss():
    for each (sample, layer) group:
        pred  = softmax(MLP_layer(x), dim=0)
        label = y / y.sum()
        loss += MSE(pred, label)
    ▼
KVzapModel checkpoint (PV or Future)


[추론: evaluate_image_teacher_pruning.py]
MileBench samples
    │
    ├─ --mode probe:   ProbeImageTeacherPress (PV)
    ├─ --mode future:  FutureSupervisedImagePress (Future bcast)
    └─ --mode hybrid:  HybridImageTeacherPress (PV + Future bcast, α=0.5)
         │
         ▼
     score per (image token) → top-k keep → KV prune → decode
         │
         ▼
     ROUGE-L / Accuracy / task metric
```

---

## 주요 설계 결정 (B 방법 채택 이후)

### 1. 공통 shard 포맷
PV와 Future가 **동일한 (x, layer, sample_id)** 쌍을 공유. 차이는 `y_pv` vs `y_future`뿐. 결과 해석이 "label 차이"로 환원됨 (확실한 ablation 설정).

### 2. Per-sample softmax MSE
Teacher label은 본질적으로 "한 샘플 안에서 어느 토큰이 중요한가" 분포. 샘플 단위 softmax로 pred를 같은 공간으로 매핑 후 MSE.
- 장점: 스케일 일치, 해석 일관성, PV·Future 공유
- 단점: 기존 log-transformed MSE 대비 top-k magnitude separation이 작아질 수 있음 (Spot-the-Diff 관찰)

### 3. One-pass multi-layer training
이전 구현은 레이어마다 shard 전체를 다시 스캔 (32 epoch × 584 shard). 개선: 1 epoch = 1 scan, 그 안에서 모든 레이어 MLP 업데이트. SSD 기준 epoch당 ~5-6분.

### 4. SSD-first storage
HDD(117 MB/s) 기준 epoch 50+분, SSD(500+ MB/s) 기준 ~5-6분. 초기에 `/workspace/hd` symlink로 HDD에 저장했다가 SSD로 재수집.

### 5. Layer 31 broadcast for Future
Future probe는 layer 24-31만 의미 있는 weight. 하지만 KV cache는 레이어별 동일 길이 요구 → 모든 레이어에서 pruning 수행 필요. 해결: 최종 레이어(31) 가중치를 0-23에 복사한 별도 체크포인트 (`_bcast31`).

### 6. `--progress_file` 옵션
`conda run`의 stdout 버퍼링 때문에 tqdm이 로그 파일에 반영 안 됨. 학습 스크립트에 `--progress_file` 추가 — 매 3초마다 진행바+ETA, epoch 완료 시 요약을 flush된 파일에 기록.

### 7. Truncation for long-context datasets
ActionSequence 등 다중 이미지 데이터셋은 prefill attention 크기가 24GB GPU 초과 → OOM. 평가 시 `--truncate_like_lookm` 적용하여 LOOK-M 방식 입력 truncation.

---

## 관련 출력 경로

| 항목 | 경로 |
|---|---|
| 수집 shard | `/workspace/zap/artifacts/teacher_ssd/unified/{textvqa,scienceqa,nlvr2}/shards/` |
| 수집 리포트 | `.../<ds>/validation_report.json` |
| 학습 ckpt (v4) | `/workspace/zap/ckpts/{postvision_probe_v4_20ep, future_probe_v4_last8_20ep}` |
| Inference ckpt (Future bcast) | `/workspace/zap/ckpts/future_probe_v4_last8_20ep_bcast31` |
| 학습 로그 | `/workspace/zap/artifacts/EXP-20260418-001/v4_{future,pv}_{train,progress}.log` |
| 평가 결과 | `/workspace/zap/artifacts/EXP-20260418-001/v4_eval/<dataset>/<mode>_k<kr>/` |
| 평가 체인 로그 | `/workspace/zap/artifacts/EXP-20260418-001/v4_eval_chain.log` |
| 과거 로그 보관 | `/workspace/zap/artifacts/EXP-20260418-001/logs/archive/` |

---

## 체크리스트

- [x] Unified shard 수집 파이프라인 (`collect_unified_teacher_shards.py`)
- [x] One-pass multi-layer 학습 파이프라인 (`train_unified_probe_onepass.py`)
- [x] PostVision / Future / Hybrid press 구현
- [x] `--selected_layer_indices` 지원 (press)
- [x] Broadcast 체크포인트 유틸
- [x] `--progress_file` 실시간 로그
- [x] v4 학습 완료 (PV Spearman 0.7512, Future 0.7124)
- [x] 평가 스크립트 `--mode future|hybrid` 지원
- [x] 초기 평가 (CLEVR-Change, Spot-the-Diff, ActionSequence) 완료
- [ ] 전체 MileBench 평가 완료 (진행 중)
- [ ] α sweep (0.25 / 0.75) 비교
- [ ] Oracle / Random baseline 비교
- [ ] 최종 report
