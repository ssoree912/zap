Reference copies of the lmms-eval model/task files used to generate the
CHAIR/AMBER/POPE/MME hallucination-eval numbers. The working copies live in
the VFlowOpt_llava1.5 repo's `src/lmms_eval-0.2.4/lmms_eval/{models,tasks}/`
tree, where lmms-eval's model/task auto-discovery expects them -- kept here
too so this repo's hallucination-eval code isn't scattered across repos.

To actually run these, they need to be placed at the corresponding path
under an lmms-eval checkout, not run from here directly:

- `llava_onevision_student_delayed_replay.py` -> `lmms_eval/models/`
  (also register it in `lmms_eval/models/__init__.py`'s `AVAILABLE_MODELS`)
- `llava_onevision_visionzip.py` -> `lmms_eval/models/`
- `coco2017_cap_val_chair500.yaml` -> `lmms_eval/tasks/coco_cap/`
  (needs `utils.py`'s `process_docs_chair500`, see this repo's
  `foresight/eval/chair/` for the scorer that consumes its output)
