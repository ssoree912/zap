"""Score CHAIR_s / CHAIR_i on an lmms-eval coco2017_cap_val_chair500 output.

Reads the per-sample jsonl lmms-eval writes with --log_samples (each line
has doc.question_id -> COCO image_id, and filtered_resps[0] -> generated
caption), scores it with the vendored CHAIR implementation, and writes a
detailed json alongside a one-line summary.
"""

import argparse
import glob
import json
import os
import re

from chair_metric import CHAIR

INSTANCES_PATH = "/workspace/zap/data/coco_annotations/instances_val2014.json"
CAPTIONS_PATH = "/workspace/zap/data/coco_annotations/captions_val2014.json"


def load_captions(jsonl_path):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            question_id = d["doc"]["question_id"]
            m = re.match(r"COCO_val2014_(\d+)\.jpg", question_id)
            image_id = int(m.group(1))
            caption = d["filtered_resps"][0]
            samples.append({"image_id": image_id, "caption": caption})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_glob", required=True,
                         help="glob for the lmms-eval *_samples_coco2017_cap_val_chair500.jsonl file(s)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.samples_glob))
    if not paths:
        raise FileNotFoundError(f"no files matched {args.samples_glob}")

    samples = []
    for p in paths:
        samples.extend(load_captions(p))
    print(f"Loaded {len(samples)} generated captions from {len(paths)} file(s)")

    imids = {s["image_id"] for s in samples}
    evaluator = CHAIR(imids, [INSTANCES_PATH], [CAPTIONS_PATH])
    evaluator.get_annotations()
    result = evaluator.compute_chair(samples)

    print(f"CHAIR_s = {result['chair_s']*100:.2f}%  "
          f"({result['num_hallucinated_captions']}/{result['num_captions']} captions)")
    print(f"CHAIR_i = {result['chair_i']*100:.2f}%  "
          f"({result['num_hallucinated_words']}/{result['num_mscoco_words']} mscoco words)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote details to {args.output}")


if __name__ == "__main__":
    main()
