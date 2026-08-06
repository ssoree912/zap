Reference copies of the lmms-eval model/task files used to generate the
CHAIR/AMBER/POPE/MME/Video-MME/SEED-Bench-video comparison numbers for
full-cache, student, VFlowOpt, and VisionZip. The working copies live in
the VFlowOpt_llava1.5 repo's `src/lmms_eval-0.2.4/lmms_eval/{models,tasks}/`
tree, where lmms-eval's model/task auto-discovery expects them -- kept here
too so this repo's hallucination-eval code isn't scattered across repos.

To actually run these, they need to be placed at the corresponding path
under an lmms-eval checkout, not run from here directly:

- `llava_onevision_student_delayed_replay.py` -> `lmms_eval/models/`
  (also register it in `lmms_eval/models/__init__.py`'s `AVAILABLE_MODELS`)
- `llava_onevision_visionzip.py` -> `lmms_eval/models/`
- `llava_onevision_training_free.py` -> `lmms_eval/models/` -- the VFlowOpt
  model class (`illava_vit_*`/`illava_llm_*` inline pruning). Used via the
  standard lmms-eval CLI (`--model llava_onevision_training_free --tasks
  videomme_local` / `seedbench_local` / `coco2017_cap_val_chair500`, no
  delayed-replay needed since its pruning happens inline in the forward
  pass). Depends on the `forward_illava` patch to `modeling_qwen2.py` in
  VFlowOpt_llava1.5's vendored `src/transformers-4.46.0/` -- not vanilla
  transformers, and not copied here (see `../../onevision/README.md`).
- `coco2017_cap_val_chair500.yaml` -> `lmms_eval/tasks/coco_cap/`
  (needs `utils.py`'s `process_docs_chair500`, see this repo's
  `foresight/eval/chair/` for the scorer that consumes its output)

The full-cache baseline and VisionZip's video runs use the same standard
lmms-eval CLI path (`--model llava_onevision` / `llava_onevision_visionzip`
--tasks videomme_local / seedbench_local), reusing the task defs in
`../../onevision/lmms_tasks/`.
