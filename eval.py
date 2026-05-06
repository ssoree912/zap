#!/usr/bin/env python3
"""Foresight: evaluate visual-utility student KV pruning.

Dispatches to lmms-eval or VLMEvalKit based on --framework.
All arguments after '--' separator are forwarded to the underlying framework.

Examples:
    # lmms-eval with OneVision student (post-prefill eviction)
    python eval.py --model onevision --framework lmms -- \\
        --model llava_onevision_student \\
        --model_args pretrained=...,student_path=...,keep_ratio=0.5 \\
        --tasks chartqa_local --batch_size 1 --output_path results/

    # lmms-eval with progressive student (layer-by-layer eviction during prefill)
    python eval.py --model onevision --framework lmms --variant progressive -- \\
        --model llava_onevision_student_progressive \\
        --model_args pretrained=...,student_path=...,keep_ratio=0.5,progressive_schedule=immediate \\
        --tasks chartqa_local --batch_size 1 --output_path results/

    # VLMEvalKit with OneVision student
    python eval.py --model onevision --framework vlmeval -- \\
        --data ChartQA --model LLaVA_OneVision_Student
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ZAP_ROOT = Path(os.environ.get("ZAP_REPO_ROOT", Path(__file__).resolve().parent)).resolve()
LMMS_EVAL_ROOT = Path(
    os.environ.get(
        "LMMS_EVAL_ROOT",
        ZAP_ROOT.parent / "VFlowOpt_llava1.5/src/lmms_eval-0.2.4",
    )
).resolve()

# Insert immediately at import time so sub-imports from foresight/* work
for _p in [str(ZAP_ROOT), str(LMMS_EVAL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _patch_local_eval_datasets():
    """Route known lmms eval datasets to local `data/eval` Arrow folders."""
    try:
        import datasets
        from datasets import DatasetDict, load_from_disk
        from datasets.features import features as dataset_features
        import pyarrow as pa
        import pyarrow.ipc as pa_ipc
    except ImportError:
        return

    local_root = Path(os.environ.get("LMMS_LOCAL_EVAL_ROOT", ZAP_ROOT / "data/eval")).resolve()
    if not local_root.exists():
        return

    # Some saved Arrow datasets use the newer "List" feature tag, while this
    # environment's datasets version still expects Sequence.
    if hasattr(dataset_features, "_FEATURE_TYPES"):
        dataset_features._FEATURE_TYPES.setdefault("List", dataset_features.Sequence)

    original_load_dataset = datasets.load_dataset
    local_specs = {
        ("lmms-lab/ChartQA", None): ("chartqa", "test"),
        ("lmms-lab/DocVQA", "DocVQA"): ("docvqa_val", "validation"),
        ("lmms-lab/textvqa", None): ("textvqa_val", "validation"),
        ("lmms-lab/COCO-Caption2017", None): ("coco2017_cap_val", "val"),
        ("lmms-lab/NoCaps", None): ("nocaps_val", "validation"),
        ("lmms-lab/TextCaps", None): ("textcaps_val", "val"),
        ("lmms-lab/GQA", "testdev_balanced_instructions"): ("gqa/instructions", "testdev"),
        ("lmms-lab/GQA", "testdev_balanced_images"): ("gqa/images", "testdev"),
    }

    def load_local_dataset(local_dir: Path):
        try:
            return load_from_disk(str(local_dir))
        except (TypeError, ValueError):
            arrow_files = sorted(local_dir.glob("*.arrow"))
            if not arrow_files:
                raise
            tables = []
            for arrow_file in arrow_files:
                with pa.memory_map(str(arrow_file), "r") as source:
                    tables.append(pa_ipc.open_stream(source).read_all())
            table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
            # Some local GQA shards were saved with newer `List` feature
            # metadata. datasets==2.21 misreads that metadata, while the Arrow
            # schema itself is valid, so strip metadata and infer features.
            return datasets.Dataset(table.replace_schema_metadata(None))

    def load_dataset(path, name=None, *args, split=None, **kwargs):
        key = (path, name)
        if key in local_specs:
            rel_path, split_name = local_specs[key]
            dataset = load_local_dataset(local_root / rel_path)
            if split is not None:
                return dataset
            return DatasetDict({split_name: dataset})
        return original_load_dataset(path, name=name, *args, split=split, **kwargs)

    datasets.load_dataset = load_dataset


def _ensure_paths():
    _patch_local_eval_datasets()


def main():
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True, choices=["llava15", "onevision"],
                   help="Which VLM backbone")
    p.add_argument("--framework", required=True, choices=["lmms", "vlmeval"],
                   help="Evaluation framework to use")
    p.add_argument("--variant", default="student", choices=["student", "progressive"],
                   help="student: post-prefill eviction / progressive: layer-by-layer during prefill")
    args, remaining = p.parse_known_args()

    if remaining and remaining[0] == "--":
        remaining = remaining[1:]

    _ensure_paths()

    if args.model == "onevision" and args.framework == "lmms":
        import lmms_eval.models as _models_pkg
        forwarded_model = None
        if "--model" in remaining:
            model_idx = remaining.index("--model")
            if model_idx + 1 < len(remaining):
                forwarded_model = remaining[model_idx + 1]

        if forwarded_model == "llava_onevision_original_student":
            import foresight.eval.lmms_onevision_original_student as _orig_mod
            sys.modules["lmms_eval.models.llava_onevision_original_student"] = _orig_mod
            _models_pkg.AVAILABLE_MODELS["llava_onevision_original_student"] = (
                "LlavaOnevisionOriginalStudent"
            )
        elif args.variant == "progressive":
            import foresight.eval.lmms_onevision_student_progressive as _mod
            sys.modules["lmms_eval.models.llava_onevision_student_progressive"] = _mod
            _models_pkg.AVAILABLE_MODELS["llava_onevision_student_progressive"] = (
                "LlavaOnevisionStudentProgressive"
            )
        else:
            import foresight.eval.lmms_onevision_student as _mod
            sys.modules["lmms_eval.models.llava_onevision_student"] = _mod
            _models_pkg.AVAILABLE_MODELS["llava_onevision_student"] = "LlavaOnevisionStudent"
        sys.argv = [str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py")] + remaining
        runpy.run_path(str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py"), run_name="__main__")

    elif args.model == "onevision" and args.framework == "vlmeval":
        from foresight.eval.vlmeval_onevision_student import LLaVA_OneVision_HF_Student
        import vlmeval.vlm as _vlm_pkg
        _vlm_pkg.LLaVA_OneVision_HF_Student = LLaVA_OneVision_HF_Student
        sys.argv = ["run.py"] + remaining
        from vlmeval.run import main as vlmeval_main
        vlmeval_main()

    elif args.model == "llava15" and args.framework == "lmms":
        import lmms_eval.models as _models_pkg
        forwarded_model = None
        if "--model" in remaining:
            model_idx = remaining.index("--model")
            if model_idx + 1 < len(remaining):
                forwarded_model = remaining[model_idx + 1]
        if forwarded_model == "llava15_original_oracle":
            import foresight.eval.lmms_llava15_original_oracle as _mod
            sys.modules["lmms_eval.models.llava15_original_oracle"] = _mod
            _models_pkg.AVAILABLE_MODELS["llava15_original_oracle"] = "Llava15OriginalOracle"
        else:
            import foresight.eval.lmms_llava15_original_student as _mod
            sys.modules["lmms_eval.models.llava15_original_student"] = _mod
            _models_pkg.AVAILABLE_MODELS["llava15_original_student"] = "Llava15OriginalStudent"
        sys.argv = [str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py")] + remaining
        runpy.run_path(str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py"), run_name="__main__")

    elif args.model == "llava15":
        raise NotImplementedError(f"llava15 + {args.framework} not yet implemented")


if __name__ == "__main__":
    main()
