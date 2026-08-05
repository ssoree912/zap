"""Sanity check for chair_metric.py: score COCO's own human reference
captions against ground truth built partly from those same captions.
A real (non-hallucinating) caption should score far below typical VLM
hallucination rates -- CHAIR is not expected to hit exactly 0 on human
captions (annotator perceptual slips, synonym-matching edge cases), but a
result in the same range as model output (tens of percent) would mean the
synonym matching or the instances/captions join is broken, not that models
hallucinate as much as humans.
"""

import glob
import re
import sys

import pyarrow.parquet as pq

from chair_metric import CHAIR

PARQUET_GLOB = "/workspace/hd/data/eval/coco_cap/data/val-*.parquet"
INSTANCES_PATH = "/workspace/zap/data/coco_annotations/instances_val2014.json"
CAPTIONS_PATH = "/workspace/zap/data/coco_annotations/captions_val2014.json"
N_SAMPLE = 500
SEED = 42


def load_subset():
    rows = []
    for f in sorted(glob.glob(PARQUET_GLOB)):
        tbl = pq.ParquetFile(f).read(columns=["question_id", "answer"]).to_pylist()
        rows.extend(tbl)
    rng_rows = sorted(rows, key=lambda r: r["question_id"])
    import random
    random.Random(SEED).shuffle(rng_rows)
    subset = rng_rows[:N_SAMPLE]
    out = []
    for row in subset:
        m = re.match(r"COCO_val2014_(\d+)\.jpg", row["question_id"])
        image_id = int(m.group(1))
        out.append({"image_id": image_id, "caption": row["answer"][0]})
    return out


def main():
    samples = load_subset()
    imids = {s["image_id"] for s in samples}
    print(f"Scoring {len(samples)} human reference captions ({len(imids)} unique images)...")

    evaluator = CHAIR(imids, [INSTANCES_PATH], [CAPTIONS_PATH])
    evaluator.get_annotations()
    result = evaluator.compute_chair(samples)

    print(f"CHAIR_s = {result['chair_s']*100:.2f}%  "
          f"({result['num_hallucinated_captions']}/{result['num_captions']} captions)")
    print(f"CHAIR_i = {result['chair_i']*100:.2f}%  "
          f"({result['num_hallucinated_words']}/{result['num_mscoco_words']} mscoco words)")

    flagged = [s for s in result["sentences"] if s["chair_s"] == 1]
    print(f"\n{len(flagged)} flagged captions (first 5):")
    for s in flagged[:5]:
        print(f"  image_id={s['image_id']} caption={s['caption']!r}")
        print(f"    hallucinated: {s['mscoco_hallucinated_words']}  gt: {sorted(s['mscoco_gt_words'])}")

    if result["chair_s"] > 0.15:
        print("\nWARNING: CHAIR_s on human captions is unexpectedly high (>15%) -- "
              "check synonym matching / instances-captions join before trusting model scores.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
