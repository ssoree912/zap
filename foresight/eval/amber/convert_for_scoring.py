"""Convert generate_responses.py's line-delimited output into the JSON-array
format inference.py expects (json.load, not jsonl), filtered to the id range
for a given --evaluation_type -- inference.py's dimension gates divide by
metrics.txt's 0.001 sentinel (rather than erroring) when a counter never got
incremented, so feeding it ids outside the requested dimension produces
garbage numbers silently rather than a clean failure. One dimension's worth
of ids per output file, matching the README's json-file-to-eval-arg table.
"""

import argparse
import json

QUERY_FILES = {
    "g": "/workspace/AMBER/data/query/query_generative.json",
    "d": "/workspace/AMBER/data/query/query_discriminative.json",
    "de": "/workspace/AMBER/data/query/query_discriminative-existence.json",
    "da": "/workspace/AMBER/data/query/query_discriminative-attribute.json",
    "dr": "/workspace/AMBER/data/query/query_discriminative-relation.json",
    "a": "/workspace/AMBER/data/query/query_all.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", default="/workspace/zap/look_rebuttal/outputs/amber_student_dr_0.1/responses.jsonl")
    parser.add_argument("--evaluation_type", required=True, choices=list(QUERY_FILES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wanted_ids = {q["id"] for q in json.load(open(QUERY_FILES[args.evaluation_type]))}

    responses = {}
    with open(args.responses) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                responses[d["id"]] = d["response"]

    missing = wanted_ids - responses.keys()
    have = sorted(wanted_ids & responses.keys())
    print(f"evaluation_type={args.evaluation_type}: {len(have)}/{len(wanted_ids)} ids present"
          + (f", {len(missing)} missing" if missing else ""))

    out = [{"id": i, "response": responses[i]} for i in have]
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} entries to {args.output}")


if __name__ == "__main__":
    main()
