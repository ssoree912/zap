#!/usr/bin/env python3
"""Foresight: extract future-token teacher signals.

Dispatches to the appropriate teacher collection script based on --model.
All arguments after '--' are forwarded to the underlying script.

Examples:
    # LLaVA-1.5 7B original checkpoint
    python experiments/EXP-20260502-024-llava15-original-teacher-extract/collect_original_llava15_teacher.py \\
        --model-path /workspace/zap/ckpts/llava-v1.5-7b

    # LLaVA-OneVision 7B
    python extractor.py --model onevision -- \\
        --dataset llava_instruct --n-samples 500 --output-root data/teacher_v2_onevision
"""

from __future__ import annotations

import sys


def main():
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True, choices=["llava15", "onevision"],
                   help="Which VLM backbone to extract teacher signals from")
    args, remaining = p.parse_known_args()

    if remaining and remaining[0] == "--":
        remaining = remaining[1:]

    sys.argv = [sys.argv[0]] + remaining

    if args.model == "llava15":
        raise SystemExit(
            "HF LLaVA-1.5 collector was removed. Use "
            "experiments/EXP-20260502-024-llava15-original-teacher-extract/"
            "collect_original_llava15_teacher.py with "
            "/workspace/zap/ckpts/llava-v1.5-7b."
        )
    else:  # onevision
        from foresight.teacher.collect_llava_onevision import main as _main
        sys.exit(_main())


if __name__ == "__main__":
    main()
