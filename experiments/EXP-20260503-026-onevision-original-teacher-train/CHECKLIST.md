# OneVision Original Teacher/Student Checklist

- [x] Preserve existing LLaVA-1.5 7B teacher/student artifacts and experiment outputs.
- [x] Extract original LLaVA-OneVision teacher scores for `textvqa`, `gqa`, `scienceqa` using 600 samples each.
- [x] Smoke-test OneVision teacher extraction on GPU0 with 3 TextVQA samples (`T_mean=1.33`, `hit16=0/3`).
- [x] Verify teacher `T` distributions and `teacher_norm` finite rows.
- [x] Skip zero-sum teacher rows during training (`bad_teacher_rows` only occurs on layer 27).
- [ ] Train `VisualUtilityStudentOneVision` with `lr=1e-4`, `epochs=15`.
- [ ] Run MMVet/detail_1k inference at keep ratios `0.2`, `0.5`, `0.7`.
- [ ] Summarize ROUGE-L and PPL outputs.
