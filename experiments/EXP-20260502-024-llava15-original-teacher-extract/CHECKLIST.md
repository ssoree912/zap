# LLaVA-1.5 Original Teacher Extraction Checklist

- [x] Branch created from `implementation-guide`: `feat/llava15-original-teacher-extract`
- [x] Use VFlowOpt conda env: `/workspace/VFlowOpt/.conda/VFlowOpt`
- [x] Use original LLaVA checkpoint: `/workspace/zap/ckpts/llava-v1.5-7b`
- [x] Run extraction on GPU 0
- [x] Use same selected 600-sample subsets for GQA, TextVQA, ScienceQA
- [x] Save final teacher records under `artifacts/original_llava_teacher/future_decode_llava15_7b`
- [x] Verify record counts: GQA 600, TextVQA 600, ScienceQA 600
- [x] Verify teacher tensor shape: `(32, 576)` for all datasets
- [x] Verify `teacher_norm` sums are valid for all records
- [x] Confirm generation length sanity: GQA/TextVQA not saturated at max length
- [x] Start original LLaVA student training on GPU 0
- [x] Training config: 1800 teacher records, lr `1e-4`, 15 epochs
