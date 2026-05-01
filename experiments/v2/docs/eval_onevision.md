어# OneVision KV-Pruning Evaluation Guide

## Overview

VLMEvalKit 기반으로 LLaVA-OneVision-7B (+ student KV-pruning) 성능을 평가하는 파이프라인입니다.

| 벤치마크 | Baseline (full KV) | Student @ 50% |
|---|---|---|
| ChartQA_TEST | **79.96%** | **79.84%** (Δ −0.12%p) |
| DocVQA_VAL | (실행 중) | (실행 중) |
| TextVQA_VAL | (실행 중) | (실행 중) |

---

## 파일 위치

```
/workspace/zap/
├── kvzap/
│   └── vlmeval_onevision_student.py   ← 핵심 평가 래퍼 (LLaVA_OneVision_HF_Student)
├── scripts/
│   └── run_vlmeval_student.py         ← 런처 (monkey-patch + VLMEvalKit 위임)
├── eval_configs/
│   ├── onevision_baseline_chartqa.json
│   ├── onevision_student50_chartqa.json
│   ├── onevision_student50_docvqa.json
│   └── onevision_student50_textvqa.json
└── eval_results/
    ├── baseline_chartqa/              ← full-KV 결과
    ├── student50_chartqa_v2/          ← student@50% 최신 ChartQA 결과 ✓
    ├── student50_textvqa/             ← TextVQA (진행 중)
    └── student50_docvqa/              ← DocVQA (진행 중)
```

### 핵심 파일 설명

| 파일 | 역할 |
|---|---|
| `kvzap/vlmeval_onevision_student.py` | `LLaVA_OneVision_HF_Student` 클래스. prefill → student 스코어링 → KV trim → decode 루프 |
| `scripts/run_vlmeval_student.py` | `LLaVA_OneVision_HF_Student`를 `vlmeval.vlm`에 monkey-patch하고 VLMEvalKit `run.py` 호출 |
| `kvpress/presses/visual_utility_student_onevision.py` | 학습된 1D ConvNeXt student probe 모델 정의 |
| `ckpts/student_onevision_A_lr1e4_20ep/` | 학습된 student 체크포인트 |
| `data/eval_LMU/` | VLMEvalKit이 사용하는 벤치마크 데이터 루트 (`LMUData` 환경변수) |

---

## 실행 방법

### 공통 형식

```bash
CUDA_VISIBLE_DEVICES=<GPU번호> python /workspace/zap/scripts/run_vlmeval_student.py -- \
    --config /workspace/zap/eval_configs/<config파일> \
    --work-dir /workspace/zap/eval_results/<출력폴더>
```

### Baseline (full KV, keep_ratio=1.0)

```bash
# ChartQA
CUDA_VISIBLE_DEVICES=0 python /workspace/zap/scripts/run_vlmeval_student.py -- \
    --config /workspace/zap/eval_configs/onevision_baseline_chartqa.json \
    --work-dir /workspace/zap/eval_results/baseline_chartqa \
    2>&1 | tee /workspace/zap/eval_results/student50_logs/baseline_chartqa.log
```

> **참고**: baseline config는 `LLaVA_OneVision_HF` 클래스를 사용하므로 student 가중치를 로드하지 않음.

### Student @ 50%

```bash
# ChartQA
CUDA_VISIBLE_DEVICES=0 python /workspace/zap/scripts/run_vlmeval_student.py -- \
    --config /workspace/zap/eval_configs/onevision_student50_chartqa.json \
    --work-dir /workspace/zap/eval_results/student50_chartqa_v2 \
    2>&1 | tee /workspace/zap/eval_results/student50_logs/chartqa_v2.log

# DocVQA
CUDA_VISIBLE_DEVICES=1 python /workspace/zap/scripts/run_vlmeval_student.py -- \
    --config /workspace/zap/eval_configs/onevision_student50_docvqa.json \
    --work-dir /workspace/zap/eval_results/student50_docvqa \
    2>&1 | tee /workspace/zap/eval_results/student50_logs/docvqa_v2.log

# TextVQA
CUDA_VISIBLE_DEVICES=2 python /workspace/zap/scripts/run_vlmeval_student.py -- \
    --config /workspace/zap/eval_configs/onevision_student50_textvqa.json \
    --work-dir /workspace/zap/eval_results/student50_textvqa \
    2>&1 | tee /workspace/zap/eval_results/student50_logs/textvqa_v2.log
```

### 새 벤치마크 추가

`eval_configs/` 아래에 JSON을 복사해서 `data` 섹션의 dataset 이름만 바꾸면 됩니다.

```json
{
  "model": {
    "ov7b_student50": {
      "class": "LLaVA_OneVision_HF_Student",
      "model_path": "/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf",
      "student_path": "/workspace/zap/ckpts/student_onevision_A_lr1e4_20ep",
      "keep_ratio": 0.5,
      "max_new_tokens": 32
    }
  },
  "data": {
    "NEW_DATASET": {"class": "ImageVQADataset", "dataset": "NEW_DATASET"}
  }
}
```

---

## 로그 / 결과 확인

### 실시간 로그

```bash
tail -f /workspace/zap/eval_results/student50_logs/chartqa_v2.log
```

VLMEvalKit은 `--work-dir` 안에 타임스탬프 폴더도 함께 생성합니다.

```bash
# 예: eval_results/student50_chartqa_v2/logs/T20260427-091050_20260427091050.log
ls /workspace/zap/eval_results/student50_chartqa_v2/logs/
```

### 결과 파일

평가 완료 시 `--work-dir/<model_name>/<timestamp>/` 아래에 생성됩니다.

```
<work-dir>/
└── ov7b_student50/
    └── T<timestamp>/
        ├── ov7b_student50_<DATASET>.xlsx          ← 샘플별 예측값
        ├── ov7b_student50_<DATASET>_results.xlsx  ← 세부 결과
        └── ov7b_student50_<DATASET>_acc.csv       ← 최종 정확도 ← 여기가 핵심
```

```bash
cat /workspace/zap/eval_results/student50_chartqa_v2/ov7b_student50/T20260427-091050/ov7b_student50_ChartQA_TEST_acc.csv
# "test_human","test_augmented","Overall"
# "67.12","92.56","79.84"
```

---

## keep_ratio 변경

`eval_configs/*.json`의 `keep_ratio` 값만 바꾸면 됩니다.

- `1.0` → pruning 없음 (baseline과 동일한 코드 경로)
- `0.5` → 이미지 토큰 50% 유지
- `0.3` → 이미지 토큰 30% 유지

---

## 내부 동작 요약

```
generate_inner_image()
  ├── processor로 tokenize + pixel encode
  ├── model() prefill (output_hidden_states=True)  → past_kv, H_all, next_token
  ├── infer_onevision_image_positions_no_forward()  → image_positions, prompt_len
  ├── [keep_ratio < 1.0] student.layers[li](H_l, image_idx, q_idx) → scores
  │     → topk → keep_mask → _trim_kv_cache_per_layer(past_kv, keep_masks)
  └── _greedy_decode_with_kv(model, past_kv, next_token, prompt_len, ...)
        → cache_position/position_ids를 절대 위치로 명시 (RoPE 정합 보장)
```

> **RoPE 주의사항**: KV trim 후 `past_kv.get_seq_length()`는 trim된 길이를 반환합니다.
> HF 기본값을 쓰면 새 query가 잘못된 위치로 RoPE 회전 → 성능 폭락.
> `_greedy_decode_with_kv`에서 `cache_position=torch.tensor([prompt_len+step])`을 명시적으로 전달하여 해결합니다.
