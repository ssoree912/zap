# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score MileBench ``pred.json`` files with the local LOOK-M-compatible scorer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, "/workspace/zap")

from qvik.eval.look_milebench_metrics import LookMileBenchEvaluator

DATA_ROOT = "/workspace/zap/data/MileBench"

MILEBENCH_DATASETS = [
    "ALFRED",
    "ActionLocalization",
    "ActionPrediction",
    "ActionSequence",
    "CLEVR-Change",
    "CharacterOrder",
    "CounterfactualInference",
    "DocVQA",
    "EgocentricNavigation",
    "GPR1200",
    "IEdit",
    "ImageNeedleInAHaystack",
    "MMCoQA",
    "MovingAttribute",
    "MovingDirection",
    "MultiModalQA",
    "OCR-VQA",
    "ObjectExistence",
    "ObjectInteraction",
    "ObjectShuffle",
    "SceneTransition",
    "SlideVQA",
    "Spot-the-Diff",
    "StateChange",
    "TQA",
    "TextNeedleInAHaystack",
    "WebQA",
    "WikiVQA",
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _prediction_id_set(predictions: list[dict[str, Any]]) -> set[Any]:
    ids: set[Any] = set()
    for item in predictions:
        sample_id = item["sample_id"]
        ids.add(sample_id)
        ids.add(str(sample_id))
        try:
            ids.add(int(sample_id))
        except Exception:  # noqa: BLE001
            pass
    return ids


def _subset_core(core_json: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    pred_ids = _prediction_id_set(predictions)
    data = [sample for sample in core_json["data"] if sample["sample_id"] in pred_ids or str(sample["sample_id"]) in pred_ids]
    return {"meta_data": core_json["meta_data"], "data": data}


def score_dataset(
    *,
    data_root: str | Path,
    result_dir: str | Path,
    dataset: str,
    adv: bool = False,
    allow_partial: bool = False,
    overwrite: bool = False,
) -> dict[str, Any] | None:
    data_root = Path(data_root)
    result_dir = Path(result_dir)
    dataset_dir = result_dir / dataset
    pred_path = dataset_dir / "pred.json"
    eval_path = dataset_dir / "eval.json"

    if not pred_path.exists():
        print(f"[skip] {dataset}: missing {pred_path}")
        return None
    if eval_path.exists() and not overwrite:
        print(f"[skip] {dataset}: {eval_path} exists")
        return _load_json(eval_path)

    core_name = f"{dataset}-adv.json" if adv else f"{dataset}.json"
    core_path = data_root / dataset / core_name
    if not core_path.exists():
        raise FileNotFoundError(f"Missing MileBench annotation: {core_path}")

    predictions = _load_json(pred_path)
    if not predictions:
        raise ValueError(f"{pred_path} is empty")
    core_json = _load_json(core_path)

    if len(predictions) != len(core_json["data"]):
        if not allow_partial:
            raise ValueError(
                f"{dataset}: prediction count mismatch {len(predictions)}!={len(core_json['data'])}; "
                "use --allow-partial to score subsets"
            )
        core_json = _subset_core(core_json, predictions)

    evaluator = LookMileBenchEvaluator()
    predictions_for_eval = deepcopy(predictions)
    predictions_with_extracted, metrics, eval_list = evaluator.evaluate(
        predictions_for_eval,
        core_json,
        dataset_name=dataset,
    )

    _write_json(eval_path, metrics)
    _write_json(dataset_dir / "eval_score.json", eval_list)
    if predictions_with_extracted is not None:
        _write_json(dataset_dir / "pred_with_extracted.json", predictions_with_extracted)

    metric_preview = {
        key: value
        for key, value in metrics.items()
        if key not in {"image_quantity_level-Accuracy", "image_quantity_level-Result"}
    }
    print(f"[score] {dataset}: {metric_preview}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--result-dir", required=True, help="Directory containing {dataset}/pred.json")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'")
    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = MILEBENCH_DATASETS if args.dataset == "all" else [args.dataset]
    scored: dict[str, Any] = {}
    failed: dict[str, str] = {}

    for dataset in datasets:
        try:
            metrics = score_dataset(
                data_root=args.data_root,
                result_dir=args.result_dir,
                dataset=dataset,
                adv=args.adv,
                allow_partial=args.allow_partial,
                overwrite=args.overwrite,
            )
            if metrics is not None:
                scored[dataset] = metrics
        except Exception as exc:  # noqa: BLE001
            failed[dataset] = repr(exc)
            print(f"[error] {dataset}: {exc}", file=sys.stderr)

    summary = {"n_scored": len(scored), "n_failed": len(failed), "failed": failed}
    _write_json(Path(args.result_dir) / "_internal_eval_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
