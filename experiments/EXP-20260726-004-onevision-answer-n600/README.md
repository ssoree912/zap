# OneVision answer-only teacher and student

This run collects 600 teacher shards from each of TextVQA, GQA, and
ScienceQA under `/workspace/nips/data/train`.

The teacher target is the mean image-token attention over generated answer
decode steps. It does not mix in question-token attention and it does not
filter samples by answer correctness.

Run:

```bash
./run_extract_and_train.sh
```

Outputs:

- Teacher: `/workspace/nips/data/train/teacher/zap_onevision_answer_n600_seed0`
- Student: `/workspace/nips/zap/artifacts/original_onevision_teacher/student_onevision_answer_n1800_e15_seed0`
- Logs: `logs/`
