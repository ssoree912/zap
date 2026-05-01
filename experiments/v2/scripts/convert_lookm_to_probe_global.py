#!/usr/bin/env python3
"""Convert LOOK-M outputs/*/dataset/eval.json → probe_global/{dataset}/look_m/keep_{ratio}/metrics.json"""
import argparse
import json
import math
from pathlib import Path


METRIC_KEY_MAP = {
    "Rouge-L f": "ROUGE-L",
    "Rouge-L": "ROUGE-L",
    "Accuracy": "Accuracy",
    "accuracy": "Accuracy",
}

DATASET_METRIC = {
    # ROUGE-L datasets
    "ALFRED": "ROUGE-L", "IEdit": "ROUGE-L", "CLEVR-Change": "ROUGE-L",
    "Spot-the-Diff": "ROUGE-L", "MMCoQA": "ROUGE-L",
    # Accuracy datasets (everything else)
}


def load_lookm_eval(eval_json: Path) -> dict:
    d = json.loads(eval_json.read_text())
    # Normalize metric key
    metric_name = None
    metric_value = None
    for raw_key, norm_key in METRIC_KEY_MAP.items():
        if raw_key in d:
            metric_name = norm_key
            metric_value = d[raw_key]
            break
    if metric_name is None:
        raise ValueError(f"Unknown metric keys in {eval_json}: {list(d.keys())}")

    qty = d.get("image_quantity_level-Result", {})
    return {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "image_quantity_level_result": qty,
    }


def hh_recent_to_ratio_tag(hh: float, recent: float) -> str:
    total = hh + recent
    # round to 2 decimal places
    total = round(total, 2)
    tag = f"{total:.2f}".replace("0.", "0p").replace(".", "p")
    return tag, total


def convert(lookm_output_dir: Path, probe_global_dir: Path, hh: float, recent: float):
    ratio_tag, total_ratio = hh_recent_to_ratio_tag(hh, recent)
    keep_tag = f"keep_{ratio_tag}"

    converted = 0
    skipped = 0
    for dataset_dir in sorted(lookm_output_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        eval_json = dataset_dir / "eval.json"
        pred_json = dataset_dir / "pred.json"
        if not eval_json.exists():
            print(f"  [skip] {dataset_name}: no eval.json")
            skipped += 1
            continue

        try:
            info = load_lookm_eval(eval_json)
        except Exception as e:
            print(f"  [skip] {dataset_name}: {e}")
            skipped += 1
            continue

        n_samples = 0
        if pred_json.exists():
            preds = json.loads(pred_json.read_text())
            n_samples = len(preds) if isinstance(preds, list) else len(preds)

        out_dir = probe_global_dir / dataset_name.lower().replace("-", "_").replace(" ", "_") / "look_m" / keep_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "mode": "look_m",
            "n_samples": n_samples,
            "n_predictions": n_samples,
            "n_failures": 0,
            "total_keep_ratio": total_ratio,
            "hh_ratio": hh,
            "recent_ratio": recent,
            "look_model_name": f"look_m_{lookm_output_dir.name}",
            "look_dataset_name": dataset_name,
            "look_eval": {
                info["metric_name"]: info["metric_value"],
                "image_quantity_level-Result": info["image_quantity_level_result"],
            },
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"  [ok] {dataset_name} → {out_dir}")
        converted += 1

    print(f"\n완료: {converted}개 변환, {skipped}개 skip  (ratio={total_ratio}, tag={keep_tag})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookm_output_dir", required=True, help="e.g. /workspace/LOOK-M/outputs/text_prior_pivot_merge_0.1_0.1_speed")
    parser.add_argument("--probe_global_dir", required=True, help="e.g. /workspace/hd/artifacts/probe_global")
    parser.add_argument("--hh_ratio", type=float, required=True)
    parser.add_argument("--recent_ratio", type=float, required=True)
    args = parser.parse_args()

    convert(
        Path(args.lookm_output_dir),
        Path(args.probe_global_dir),
        hh=args.hh_ratio,
        recent=args.recent_ratio,
    )


if __name__ == "__main__":
    main()
