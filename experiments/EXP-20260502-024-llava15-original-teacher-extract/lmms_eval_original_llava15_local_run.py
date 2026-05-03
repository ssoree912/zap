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

ZAP_ROOT = Path("/workspace/zap")
LMMS_EVAL_ROOT = Path("/workspace/VFlowOpt/src/lmms_eval-0.2.4")
VFLOWOPT_LLAVA_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
LOCAL_ROOT = ZAP_ROOT / "data/eval"

for _path in (str(ZAP_ROOT), str(VFLOWOPT_LLAVA_ROOT), str(LMMS_EVAL_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

ROUTES: dict[tuple[str, str | None], tuple[Path, str]] = {
    ("lmms-lab/textvqa", None): (LOCAL_ROOT / "textvqa_val", "validation"),
    ("lmms-lab/GQA", "testdev_balanced_instructions"): (
        LOCAL_ROOT / "gqa/instructions",
        "testdev",
    ),
    ("lmms-lab/GQA", "testdev_balanced_images"): (
        LOCAL_ROOT / "gqa/images",
        "testdev",
    ),
    ("lmms-lab/DocVQA", "DocVQA"): (LOCAL_ROOT / "docvqa_val", "validation"),
    ("lmms-lab/ChartQA", None): (LOCAL_ROOT / "chartqa", "test"),
    ("lmms-lab/MME", None): (LOCAL_ROOT / "mme", "test"),
    ("lmms-lab/ScienceQA", "ScienceQA-FULL"): (LOCAL_ROOT / "scienceqa", "test"),
    ("lmms-lab/COCO-Caption2017", None): (LOCAL_ROOT / "coco2017_cap_val", "val"),
    ("lmms-lab/NoCaps", None): (LOCAL_ROOT / "nocaps_val", "validation"),
    ("lmms-lab/TextCaps", None): (LOCAL_ROOT / "textcaps_val", "val"),
}

_orig_load_dataset = _ds.load_dataset


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
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/.cache/huggingface/datasets")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
