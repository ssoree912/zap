#!/usr/bin/env python3
"""Run lmms-eval with custom student models and local dataset paths.

Generates `<task>_local` task YAMLs at runtime that inherit from upstream
lmms-eval task configs (via `include:`) and override `dataset_path` to point
at data/eval/. Results are written to a consistent layout:

    results/<model_tag>/<task>/
        <date>_results.json
        <date>_samples_<task>.jsonl

where model_tag is `llava15_lmms` or `onevision_lmms`. Each task in --tasks
is run separately so it gets its own folder.
"""
import contextlib
import copy
import os
import sys
import tempfile
from pathlib import Path

import yaml

ZAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ZAP_ROOT))


class _Tee:
    """Write to multiple streams at once (stdout + per-task log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def _tee_to_file(log_path: Path):
    """Mirror stdout/stderr to log_path for the duration of the block."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, f), _Tee(old_err, f)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.close()


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


# Map local task name → upstream YAML + dataset_path/name override.
# `_local` tasks inherit everything (doc_to_visual, metrics, etc.) from upstream
# via `include:` and only override the dataset location. Datasets that ship only
# the eval split locally use the parquet/arrow builder with explicit data_files
# so HF doesn't try to infer a (missing) train/test split layout.
DATA_ROOT = ZAP_ROOT / "data/eval"
LOCAL_TASKS = {
    "textvqa": {
        "base": "textvqa/textvqa_val.yaml",
        "dataset_path": "arrow",
        "data_files": {
            "validation": str(
                DATA_ROOT
                / "TextVQA/lmms-lab___textvqa/default/0.0.0/*/textvqa-validation-*.arrow"
            )
        },
    },
    "chartqa": {
        "base": "chartqa/chartqa.yaml",
        "dataset_path": str(DATA_ROOT / "ChartQA"),
    },
    "docvqa": {
        "base": "docvqa/docvqa_val.yaml",
        "dataset_path": str(DATA_ROOT / "DocVQA"),
        "dataset_name": "DocVQA",
    },
    "gqa": {
        "base": "gqa/gqa.yaml",
        # GQA keeps the upstream `lmms-lab/GQA` path; its parquet configs
        # (instructions + images) are routed locally by _patch_gqa_load_dataset.
    },
    "coco_cap": {
        "base": "coco_cap/coco2017_cap_val.yaml",
        "dataset_path": "parquet",
        "data_files": {"val": str(DATA_ROOT / "COCO-Caption2017/data/val-*.parquet")},
    },
    "nocaps": {
        "base": "nocaps/nocaps_val.yaml",
        "dataset_path": "parquet",
        "data_files": {"validation": str(DATA_ROOT / "NoCaps/data/validation-*.parquet")},
    },
    "textcaps": {
        "base": "textcaps/textcaps_val.yaml",
        "dataset_path": "parquet",
        "data_files": {"val": str(DATA_ROOT / "TextCaps/data/val-*.parquet")},
    },
}

# lmms-eval ships built-in tasks with these same names; an --include_path task
# can't override a built-in, so the generated YAML registers under a distinct
# `<task>_local` id. Users still pass the plain names on --tasks.
_LOCAL_SUFFIX = "_local"

# GQA stores instructions/images as separate parquet configs and its upstream
# utils.py hardcodes load_dataset("lmms-lab/GQA", ...); route both to local.
_GQA_ROUTES = {
    ("lmms-lab/GQA", "testdev_balanced_instructions"): DATA_ROOT / "GQA/testdev_balanced_instructions",
    ("lmms-lab/GQA", "testdev_balanced_images"): DATA_ROOT / "GQA/testdev_balanced_images",
}


def _patch_gqa_load_dataset() -> None:
    """Route GQA's hardcoded load_dataset calls to local parquet files."""
    import datasets

    orig = datasets.load_dataset

    def patched(path=None, name=None, *args, **kwargs):
        local_dir = _GQA_ROUTES.get((path, name))
        if local_dir is not None:
            ds = orig("parquet", data_files=str(local_dir / "*.parquet"), split="train")
            if kwargs.get("split") is not None:
                return ds
            return datasets.DatasetDict({"testdev": ds})
        return orig(path, name, *args, **kwargs)

    datasets.load_dataset = patched
    # gqa/utils.py imports load_dataset by name, so patch that binding too
    import lmms_eval.tasks.gqa.utils as _gqa_utils

    _gqa_utils.load_dataset = patched

# --model → results/<model_tag>/ subfolder
MODEL_TAGS = {
    "lmms_llava15_student": "llava15_lmms",
    "lmms_onevision_student": "onevision_lmms",
}


def _generate_local_task_yamls() -> Path:
    """Write one YAML per LOCAL_TASKS entry into a temp dir; return that dir.

    Each YAML registers under `<task>_local` so it doesn't collide with the
    upstream built-in task of the same name.
    """
    import lmms_eval

    upstream_root = Path(lmms_eval.__file__).resolve().parent / "tasks"
    out_dir = Path(tempfile.mkdtemp(prefix="qvik_lmms_tasks_"))
    for task_name, spec in LOCAL_TASKS.items():
        local_id = task_name + _LOCAL_SUFFIX
        cfg = {
            "include": str(upstream_root / spec["base"]),
            "task": local_id,
        }
        if "dataset_path" in spec:
            cfg["dataset_path"] = spec["dataset_path"]
        if "dataset_name" in spec:
            cfg["dataset_name"] = spec["dataset_name"]
        if "data_files" in spec:
            cfg["dataset_kwargs"] = {"data_files": spec["data_files"]}
        (out_dir / f"{local_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir


def _flatten_model_subdir(task_out: Path) -> None:
    """lmms-eval nests results under <output_path>/<model_sanitized>/; lift them up."""
    if not task_out.is_dir():
        return
    subdirs = [d for d in task_out.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        return
    model_dir = subdirs[0]
    for item in model_dir.iterdir():
        item.rename(task_out / item.name)
    model_dir.rmdir()


def main() -> None:
    # Inject custom student-model wrappers into lmms_eval's AVAILABLE_MODELS registry
    import qvik.eval.lmms_llava15_student as _llava15_mod
    import qvik.eval.lmms_onevision_student as _ov_mod

    sys.modules["lmms_eval.models.lmms_llava15_student"] = _llava15_mod
    sys.modules["lmms_eval.models.lmms_onevision_student"] = _ov_mod

    from lmms_eval.models import AVAILABLE_MODELS

    AVAILABLE_MODELS["lmms_llava15_student"] = "LmmsLlava15Student"
    AVAILABLE_MODELS["lmms_onevision_student"] = "LmmsOnevisionStudent"

    from lmms_eval.__main__ import cli_evaluate, parse_eval_args

    _patch_gqa_load_dataset()

    # Generate local task YAMLs and add them to --include_path
    local_tasks_dir = _generate_local_task_yamls()
    if "--include_path" not in sys.argv:
        sys.argv += ["--include_path", str(local_tasks_dir)]

    args = parse_eval_args()
    model_tag = MODEL_TAGS.get(args.model, args.model)
    base_output = Path(args.output_path) if args.output_path else ZAP_ROOT / "results"
    # Mirror the logs layout under base_output: results/X/... -> logs/X/...
    results_root = (ZAP_ROOT / "results").resolve()
    try:
        rel = base_output.resolve().relative_to(results_root)
        base_log = ZAP_ROOT / "logs" / rel if str(rel) != "." else ZAP_ROOT / "logs"
    except ValueError:
        base_log = base_output.parent / (base_output.name + "_logs")
    tasks = [t.strip() for t in (args.tasks or "").split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No --tasks specified.")
    unknown = [t for t in tasks if t not in LOCAL_TASKS]
    if unknown:
        raise SystemExit(
            f"Unknown task(s): {unknown}. Available: {', '.join(LOCAL_TASKS)}"
        )

    # Tag results/logs by keep_ratio so multiple ratios don't overwrite each other
    keep_ratio = "1.0"
    for kv in (args.model_args or "").split(","):
        if kv.startswith("keep_ratio="):
            keep_ratio = kv.split("=", 1)[1]
    run_tag = f"{model_tag}/keep{keep_ratio}"

    # Users pass plain task names (chartqa, gqa, ...); each routes to the
    # generated `<task>_local` config. Results/logs use the plain name.
    for task in tasks:
        local_id = task + _LOCAL_SUFFIX
        task_out = base_output / run_tag / task
        task_out.mkdir(parents=True, exist_ok=True)
        task_args = copy.deepcopy(args)
        task_args.tasks = local_id
        task_args.output_path = str(task_out)
        # Write KV-pruning stats next to the results instead of the cwd
        if task_args.model_args and "stats_output_dir=" not in task_args.model_args:
            task_args.model_args += f",stats_output_dir={task_out}"
        log_file = base_log / run_tag / f"{task}.log"
        print(f"\n===== [{run_tag}] task={task} -> {task_out} (log: {log_file}) =====", flush=True)
        with _tee_to_file(log_file):
            cli_evaluate(task_args)
        _flatten_model_subdir(task_out)


if __name__ == "__main__":
    main()
