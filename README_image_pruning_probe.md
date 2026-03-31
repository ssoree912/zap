# Image-Only Oracle Pruning And Probe Training

이 단계는 두 부분으로 나뉜다.

1. `oracle pruning`
   - teacher score를 그대로 써서 image token만 layer별 top-k%로 keep
   - text token은 전부 keep
   - 실제 generation / exact match로 teacher 품질을 먼저 검증

2. `probe training`
   - best teacher를 target으로 image token hidden state -> score 회귀
   - layer별 linear / MLP를 따로 학습
   - 이후 learned pruning으로 oracle을 근사

## 1. Oracle Pruning

스크립트:
- `/workspace/zap/evaluate_image_teacher_pruning.py`

oracle 모드 예시:

```bash
cd /workspace/zap
/opt/conda/envs/kv/bin/python /workspace/zap/evaluate_image_teacher_pruning.py   --mode oracle   --dataset_path /workspace/data/MileBench/DocVQA/DocVQA.json   --image_root /workspace/data/MileBench/DocVQA/combined_1_images   --teacher_dir /workspace/zap/artifacts/llava_docvqa_image_teacher_analysis   --teacher_score_name splus_postvision   --image_keep_ratio 0.10   --output_dir /workspace/zap/artifacts/docvqa_oracle_splus_postvision_k10
```

teacher 후보:
- `att_only_answer`
- `splus_answer`
- `splus_postvision`

권장 비교:
- keep ratio `0.05 / 0.10 / 0.20 / 0.40`
- 같은 prompt / decoding 설정 유지
- metric은 우선 exact match

## 2. Probe Training

스크립트:
- `/workspace/zap/train_image_teacher_probe.py`

학습 예시:

```bash
cd /workspace/zap
/opt/conda/envs/kv/bin/python /workspace/zap/train_image_teacher_probe.py   --extractor_dir /workspace/zap/artifacts/llava_docvqa_pilot   --teacher_dir /workspace/zap/artifacts/llava_docvqa_image_teacher_analysis   --output_dir /workspace/zap/artifacts/image_probe_splus_postvision   --target_score_name splus_postvision   --methods linear mlp   --train_fraction 0.8   --max_image_tokens_per_sample 128
```

산출물:
- `linear/` 또는 `mlp/` probe model
- `metrics.csv`
- `split.json`
- `run_config.json`

## 3. Learned Pruning

같은 평가 스크립트를 `probe` 모드로 돌리면 된다.

```bash
cd /workspace/zap
/opt/conda/envs/kv/bin/python /workspace/zap/evaluate_image_teacher_pruning.py   --mode probe   --dataset_path /workspace/data/MileBench/DocVQA/DocVQA.json   --image_root /workspace/data/MileBench/DocVQA/combined_1_images   --probe_model_name /workspace/zap/artifacts/image_probe_splus_postvision/linear   --image_keep_ratio 0.10   --output_dir /workspace/zap/artifacts/docvqa_probe_linear_k10
```

## 현재 구현 제약

- pruning 대상은 image token만이다.
- non-image text token은 항상 keep한다.
- 현재 `probe`는 score target을 per-layer / per-head 회귀로 학습한다.
- `oracle`는 `teacher_records/*.pt`가 있어야 한다.
- `probe` 평가는 teacher record 없이도 image positions를 prompt-only forward로 다시 추정할 수 있다.

## 실험 순서

권장 순서는 이렇다.

1. `oracle`: `att_only_answer`, `splus_answer`, `splus_postvision` 비교
2. best teacher 하나 선택
3. 그 teacher로 `linear`, `mlp` probe 학습
4. `probe` mode로 learned pruning 성능 비교

## 0. HD 경로 Quickstart (DocVQA 전체 + LOOK 호환 평가)

`hd` 디스크 경로를 기본으로 바로 실행하려면 아래 스크립트를 사용하면 된다.

```bash
cd /workspace/zap
bash /workspace/zap/scripts/run_docvqa_hd.sh probe
```

`oracle` 모드 실행:

```bash
cd /workspace/zap
bash /workspace/zap/scripts/run_docvqa_hd.sh oracle
```

빠른 점검(부분 샘플) 실행:

```bash
cd /workspace/zap
LIMIT=5 bash /workspace/zap/scripts/run_docvqa_hd.sh probe
```

기본 저장 위치:
- 예측/실행 로그: `/workspace/hd/artifacts/zap/<LOOK_MODEL_NAME>/`
- LOOK 호환 결과: `/workspace/hd/artifacts/zap/<LOOK_MODEL_NAME>/DocVQA/`

LOOK 호환 결과 디렉터리에는 아래 파일이 생성된다.
- `pred.json`
- `eval.json`
- `eval_score.json`
- `pred_with_extracted.json`

LOOK-M 평가 코드로 바로 비교할 때:

```bash
cd /workspace/LOOK-M
python evaluate.py \
  --data-dir /workspace/hd/data/MileBench \
  --dataset DocVQA \
  --result-dir /workspace/hd/artifacts/zap/<LOOK_MODEL_NAME>
```

여러 모델/런을 한 번에 집계할 때:

```bash
cd /workspace/LOOK-M
python score.py \
  --result-dir /workspace/hd/artifacts/zap \
  --models <LOOK_MODEL_NAME>
```
