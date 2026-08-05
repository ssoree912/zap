"""Stronger variant of selftest.py: score each image's caption #0 against
ground truth built from segmentation + the OTHER 4 reference captions only
(caption #0's own annotation id excluded). Unlike selftest.py, a caption's
objects are no longer tautologically part of the ground truth that scores
it, so this actually exercises whether synonym matching + singularization
generalize correctly across independently-written captions of the same
image, rather than just checking the pipeline runs end-to-end.
"""

import glob
import json
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
    caption_text_by_imid = {s["image_id"]: s["caption"] for s in samples}
    print(f"Scoring {len(samples)} human reference captions ({len(imids)} unique images), leave-one-out ground truth...")

    evaluator = CHAIR(imids, [INSTANCES_PATH], [CAPTIONS_PATH])
    evaluator.get_annotations_from_segments()

    # Replicate get_annotations_from_captions but skip each image's own
    # tested caption (matched by exact text -- COCO ref captions are unique
    # per annotation id, and a text match is sufficient here).
    with open(CAPTIONS_PATH) as f:
        data = json.load(f)
    for annotation in data["annotations"]:
        imid = annotation["image_id"]
        if imid in evaluator.imid_to_objects and annotation["caption"] != caption_text_by_imid.get(imid):
            _, node_words, _, _ = evaluator.caption_to_words(annotation["caption"])
            evaluator.imid_to_objects[imid].update(node_words)

    result = evaluator.compute_chair(samples)

    print(f"CHAIR_s = {result['chair_s']*100:.2f}%  "
          f"({result['num_hallucinated_captions']}/{result['num_captions']} captions)")
    print(f"CHAIR_i = {result['chair_i']*100:.2f}%  "
          f"({result['num_hallucinated_words']}/{result['num_mscoco_words']} mscoco words)")

    flagged = [s for s in result["sentences"] if s["chair_s"] == 1]
    print(f"\n{len(flagged)} flagged captions (first 10):")
    for s in flagged[:10]:
        print(f"  image_id={s['image_id']} caption={s['caption']!r}")
        print(f"    hallucinated: {s['mscoco_hallucinated_words']}  gt: {sorted(s['mscoco_gt_words'])}")

    if result["chair_s"] > 0.15:
        print("\nWARNING: CHAIR_s on human captions is unexpectedly high (>15%) -- "
              "check synonym matching / instances-captions join before trusting model scores.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
