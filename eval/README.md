# Standalone lmms-eval driver for zap

Everything under `models/` and `tasks/` here is enough, by itself, to drive
the lmms-eval side of this project's evaluation -- no need to reach into
VFlowOpt_llava1.5 or look_rebuttal for the model or task *definitions*.
**Coverage is not uniform across models, see the compatibility table
below** -- most notably, `llava_onevision_student_delayed_replay` is
image-only and raises `ValueError: Unsupported visual type for
delayed-replay: <class 'str'>` if pointed at a video task; student's video
path is a separate, non-lmms-eval script (`../onevision/generate_onevision_student.py`).
An actual lmms-eval installation (the
framework itself: `lmms_eval/__main__.py`, `evaluator.py`, `api/`, the
stock `llava_onevision.py` base class, etc.), the `llava` package, and a
transformers build with the VFlowOpt `forward_illava` patch are still
required as external dependencies -- this directory holds the
project-specific code, not a reimplementation of lmms-eval itself.

## Setup

```
eval/install.sh /path/to/lmms_eval_pkg_root
```

where the target is the directory containing `models/` and `tasks/` inside
an lmms-eval checkout (e.g. `.../lmms_eval-0.2.4/lmms_eval`). This copies
the model classes and task defs into place and registers the three custom
models in the target's `AVAILABLE_MODELS` dict (idempotent -- safe to
re-run). It does NOT install lmms-eval/llava/transformers themselves.

Then, whatever process runs `python -m lmms_eval`, make sure zap's repo
root is also on `PYTHONPATH` (the model classes import `kvpress` and
`foresight.eval.keep_budget` from there):

```
export PYTHONPATH=/path/to/lmms_eval_pkg_root/..:/path/to/zap
```

## Verified 2026-08-06

Installed into a *fresh* lmms-eval-0.2.4 copy with all zap-specific
customizations stripped out first (no `llava_onevision_student_delayed_replay.py`,
no `llava_onevision_visionzip.py`, no `llava_onevision_training_free.py`, no
`process_docs_chair500`, nothing in `AVAILABLE_MODELS`) -- i.e. simulating a
genuinely clean machine with only vanilla lmms-eval present. After running
`install.sh` against it:

```
--model llava_onevision_student_delayed_replay --tasks coco2017_cap_val_chair500 --limit 5
```

produced `coco_ROUGE_L = 0.4594` and captions identical word-for-word to
the original run (against the "production" VFlowOpt_llava1.5 checkout) on
all 5 samples. Model registration, task loading, and generation all work
end to end from zap's own files.

**Not covered by this verification**: any of the video tasks
(`videomme_local`/`seedbench_local`) against `llava_onevision_student_delayed_replay`
-- that combination does not work at all (see the compatibility table
below), so it was never part of what got tested here. Confirmed
2026-08-06 on a separate machine (GPU 3): pointing that model at a video
task raises `ValueError: Unsupported visual type for delayed-replay:
<class 'str'>` immediately, exactly as the code's own docstring says it
would ("video/multi-image handling is dropped").

## Layout

- `models/` -- `llava_onevision_student_delayed_replay.py` (delayed-replay,
  needs `delayed_replay.py`/`image_only_kv_cache.py`, both here as
  siblings), `llava_onevision_visionzip.py`, `llava_onevision_training_free.py`
  (standard `.generate()`, no delayed-replay needed for either)
- `tasks/coco_cap/` -- `coco2017_cap_val_chair500.yaml` +
  `coco_cap_utils.py` (stock lmms-eval coco_cap/utils.py plus
  `process_docs_chair500`, the 500-image fixed-seed subset; install as
  `utils.py`, not `coco_cap_utils.py` -- the yaml resolves `!function
  utils.X` relative to its own directory)
- `tasks/{pope,mme,mmbench,mmstar,vizwiz_vqa}/` -- the `*_local.yaml`
  variants used throughout, reading from local dataset paths rather than
  the HF hub (see each task's own README note in the repo root
  `image_tasks/`... consolidated here instead as of this commit)
- `tasks/{videomme_local,seedbench_local}/` -- video task defs, shared with
  the standalone workers in `../onevision/`

## Model x task compatibility

Not a full cross-product -- `llava_onevision_student_delayed_replay` is
image-only:

| `--model` | image tasks (pope_local / mme_local / mmbench_en_dev_local / mmstar_local / vizwiz_vqa_val_local / coco2017_cap_val_chair500) | video tasks (videomme_local / seedbench_local) |
|---|---|---|
| `llava_onevision` (full-cache, stock lmms-eval, not copied here) | yes | yes |
| `llava_onevision_training_free` (VFlowOpt) | yes | yes |
| `llava_onevision_visionzip` (VisionZip) | yes | yes |
| `llava_onevision_student_delayed_replay` (student) | yes | **no** -- raises `ValueError: Unsupported visual type for delayed-replay: <class 'str'>`. Use `../onevision/generate_onevision_student.py` instead (standalone, not lmms-eval, handles frame-path lists) |

For CHAIR scoring itself (turning generated captions into CHAIR_s/CHAIR_i),
see `../chair/`. For AMBER, see `../amber/`. For the standalone
(non-lmms-eval) video workers -- LOOK-M, student, PrefixKV, which need
delayed-replay but don't go through lmms-eval at all -- see `../onevision/`.
