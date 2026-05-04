# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run lmms-eval with local /workspace/zap/data/eval datasets and original LLaVA student.

This combines the local dataset routing used by VFlowOpt with registration of
`llava15_original_student`, which loads the original LLaVA checkpoint format.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import datasets as _ds
from datasets import DatasetDict, load_from_disk

ZAP_ROOT = Path(os.environ.get("ZAP_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
VFLOWOPT_ROOT = Path(os.environ.get("VFLOWOPT_ROOT", ZAP_ROOT.parent / "VFlowOpt_llava1.5/src")).resolve()
LMMS_EVAL_ROOT = Path(os.environ.get("LMMS_EVAL_ROOT", VFLOWOPT_ROOT / "lmms_eval-0.2.4")).resolve()
VFLOWOPT_LLAVA_ROOT = Path(
    os.environ.get("VFLOWOPT_LLAVA_ROOT", VFLOWOPT_ROOT / "LLaVA-OneVision")
).resolve()
VFLOWOPT_TRANSFORMERS_ROOT = Path(
    os.environ.get("VFLOWOPT_TRANSFORMERS_ROOT", VFLOWOPT_ROOT / "transformers-4.46.0/src")
).resolve()
LOCAL_ROOT = Path(os.environ.get("LMMS_LOCAL_EVAL_ROOT", ZAP_ROOT / "data/eval")).resolve()
ALT_LOCAL_ROOT = Path(
    os.environ.get("LMMS_ALT_LOCAL_EVAL_ROOT", ZAP_ROOT.parent / "VFlowOpt_llava1.5/zap/data/eval")
).resolve()

for _path in (
    str(ZAP_ROOT),
    str(VFLOWOPT_TRANSFORMERS_ROOT),
    str(VFLOWOPT_LLAVA_ROOT),
    str(LMMS_EVAL_ROOT),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _local_eval_path(*parts: str) -> Path:
    rel = Path(*parts)
    primary = LOCAL_ROOT / rel
    if primary.exists():
        return primary
    return ALT_LOCAL_ROOT / rel


ROUTES: dict[tuple[str, str | None], tuple[Path, str]] = {
    ("lmms-lab/textvqa", None): (_local_eval_path("textvqa_val"), "validation"),
    ("lmms-lab/GQA", "testdev_balanced_instructions"): (
        _local_eval_path("gqa/instructions"),
        "testdev",
    ),
    ("lmms-lab/GQA", "testdev_balanced_images"): (
        _local_eval_path("gqa/images"),
        "testdev",
    ),
    ("lmms-lab/DocVQA", "DocVQA"): (_local_eval_path("docvqa_val"), "validation"),
    ("lmms-lab/ChartQA", None): (_local_eval_path("chartqa"), "test"),
    ("lmms-lab/MME", None): (_local_eval_path("mme"), "test"),
    ("lmms-lab/ScienceQA", "ScienceQA-FULL"): (_local_eval_path("scienceqa"), "test"),
    ("lmms-lab/COCO-Caption2017", None): (_local_eval_path("coco2017_cap_val"), "val"),
    ("lmms-lab/NoCaps", None): (_local_eval_path("nocaps_val"), "validation"),
    ("lmms-lab/TextCaps", None): (_local_eval_path("textcaps_val"), "val"),
}

_orig_load_dataset = _ds.load_dataset


def _patch_datasets_list_feature_alias() -> None:
    """Keep newer saved HF dataset caches readable with datasets==2.21."""
    from dataclasses import is_dataclass

    import datasets.features.features as _features

    if not is_dataclass(getattr(_features, "List", None)):
        _features.List = _features.Sequence


def _patched_load_dataset(path=None, name=None, *args, **kwargs):
    key = (path, name)
    if key in ROUTES:
        local_dir, split_name = ROUTES[key]
        ds = load_from_disk(str(local_dir))
        requested_split = kwargs.get("split", None)
        if requested_split is None:
            print(
                f"[local-route] {path} (name={name}) -> {local_dir} "
                f"as DatasetDict[{split_name}]",
                file=sys.stderr,
                flush=True,
            )
            return DatasetDict({split_name: ds})
        print(
            f"[local-route] {path} (name={name}, split={requested_split}) -> {local_dir}",
            file=sys.stderr,
            flush=True,
        )
        return ds
    return _orig_load_dataset(path, name, *args, **kwargs)


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ZAP_ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ZAP_ROOT / ".cache/huggingface/datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _patch_datasets_list_feature_alias()
    _ds.load_dataset = _patched_load_dataset
    import datasets.load as _ds_load  # noqa: WPS433

    _ds_load.load_dataset = _patched_load_dataset

    import foresight.eval.lmms_llava15_original_student as _model_mod  # noqa: WPS433
    import lmms_eval.models as _models_pkg  # noqa: WPS433
    from lmms_eval.__main__ import cli_evaluate  # noqa: WPS433

    sys.modules["lmms_eval.models.llava15_original_student"] = _model_mod
    _models_pkg.AVAILABLE_MODELS["llava15_original_student"] = "Llava15OriginalStudent"

    cli_evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
