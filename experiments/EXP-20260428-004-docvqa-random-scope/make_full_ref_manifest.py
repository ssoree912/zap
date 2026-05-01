#!/usr/bin/env python3
"""Replace manifest answers with full-cache generated predictions for ROUGE reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = json.loads(Path(args.manifest).read_text())
    result = json.loads(Path(args.full_result).read_text())
    pred_by_id = {str(row["id"]): row.get("pred", "") for row in result["per_sample"]}

    out_samples = []
    missing = []
    for sample in samples:
        sid = str(sample["id"])
        if sid not in pred_by_id:
            missing.append(sid)
            continue
        copied = dict(sample)
        copied["gt_answer"] = copied.get("answer", "")
        copied["answer"] = pred_by_id[sid]
        out_samples.append(copied)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_samples, indent=2, ensure_ascii=False))
    print(f"[full-ref] wrote {len(out_samples)} samples -> {out}")
    if missing:
        print(f"[full-ref] missing {len(missing)} ids: {missing[:10]}")


if __name__ == "__main__":
    main()

