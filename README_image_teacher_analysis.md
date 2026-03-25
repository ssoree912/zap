# Image-Only / Post-Vision Teacher Analysis

`/workspace/zap/analyze_image_teachers.py` 는 global teacher 대신 image token만 대상으로 teacher를 다시 정의하고, answer-conditioned / post-vision-conditioned score를 비교하는 스크립트다.

이 버전에서 계산하는 score는 3개다.

- `att_only_answer = max_{j in answer} a_{ji}`
- `splus_answer = max_{j in answer} a_{ji} * ||W_O v_i|| / ||h_j||`
- `splus_postvision = max_{j in postvision} a_{ji} * ||W_O v_i|| / ||h_j||`

여기서 `i` 는 image token만 포함한다. text token은 ranking 대상에서 제외한다.

## 현재 v1 post-vision 정의

현재 v1은 `postvision` query set을 다음처럼 둔다.

- prompt 안에서 image span 뒤에 오는 모든 non-image text position

즉 single-image LLaVA prompt에서 image 뒤 question / instruction token들을 주로 잡는 쿼리 집합이다. 현재 구현은 짧은 `ASSISTANT:` suffix도 포함할 수 있다.

## 입력 요구사항

extractor record에 아래 키들이 있어야 한다.

- `attn_answer_to_prompt`
- `hidden_answer`
- `hidden_prompt`
- `wov_norm_prompt` 또는 `W_O + vproj_outputs`
- `is_image_pos_mm`
- `full_attentions`

주의:
- `splus_postvision` 계산에는 prompt query -> image key attention이 필요하므로 현재 v1은 `full_attentions`가 있어야 한다.
- 지금 pilot extractor output은 `save_full_attentions=True` 였기 때문에 그대로 사용 가능하다.

## 실행

현재 pilot extractor output으로 바로 돌릴 명령:

```bash
cd /workspace/zap
/opt/conda/envs/kv/bin/python /workspace/zap/analyze_image_teachers.py \
  --input_dir /workspace/zap/artifacts/llava_docvqa_pilot \
  --out_dir /workspace/zap/artifacts/llava_docvqa_image_teacher_analysis \
  --save_teacher_records
```

`records/`를 직접 넣어도 된다.

```bash
/opt/conda/envs/kv/bin/python /workspace/zap/analyze_image_teachers.py \
  --input_dir /workspace/zap/artifacts/llava_docvqa_pilot/records \
  --out_dir /workspace/zap/artifacts/llava_docvqa_image_teacher_analysis \
  --keep_ratios 0.05 0.10 0.20 0.40 \
  --save_teacher_records
```

## 산출물

- `summary_image_scores.csv`
- `layerwise_image_score_stats.csv`
- `topk_overlap.csv`
- `failures.csv`
- `image_only_log_score_hist.png`
- `layerwise_mean_log_score.png`
- `layerwise_median_log_score.png`
- `topk_overlap__*.png`
- `teacher_records/*.pt` (옵션)

## 먼저 볼 파일

우선순위는 이 순서가 맞다.

1. `summary_image_scores.csv`
2. `image_only_log_score_hist.png`
3. `layerwise_median_log_score.png`
4. `topk_overlap__att_only_answer__vs__splus_postvision.png`
5. `topk_overlap__att_only_answer__vs__splus_answer.png`

해석 포인트:

- `splus_postvision` 분포가 `att_only_answer` 와 많이 다르면, query-set-aware teacher가 ranking을 실제로 바꾸고 있다는 뜻이다.
- layerwise median/mean에서 `splus_postvision` 이 특정 layer에서 분리되면, 이후 pruning budget을 layerwise percentile로 둘 근거가 생긴다.
- `topk_overlap` 이 낮은 layer는 `att_only` 와 `splus` 가 실제로 다른 image token을 고른다는 뜻이다.

## 다음 단계

이 스크립트는 teacher validation 전단계 분석용이다. 여기서 image-only teacher 중 후보를 고른 뒤 다음 순서로 가면 된다.

1. image token만 대상으로 oracle pruning 비교
2. best teacher 1개 선택
3. 그 teacher에 대해 image hidden-state probe 학습

즉 현재 용도는 `global teacher` 를 버리고 `image-only / query-set-aware teacher` 후보를 정리하는 것이다.
