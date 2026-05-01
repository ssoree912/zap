#!/usr/bin/env python3
"""Foresight: train visual-utility student.

Dispatches to the appropriate training script based on --model.

Examples:
    # LLaVA-1.5 7B
    python train.py --model llava15 --scope all_token --output-dir ckpts/student_llava15

    # LLaVA-OneVision 7B
    python train.py --model onevision --scope all_token --output-dir ckpts/student_onevision
"""

from __future__ import annotations

import sys


def main():
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True, choices=["llava15", "onevision"],
                   help="Which VLM backbone to train the student for")
    args, remaining = p.parse_known_args()

    if args.model == "llava15":
        sys.argv = [sys.argv[0]] + remaining
        from foresight.train.llava_15 import main as _main
        sys.exit(_main())
    else:  # onevision
        sys.argv = [sys.argv[0]] + remaining
        from foresight.train.onevision import main as _main
        sys.exit(_main())


if __name__ == "__main__":
    main()
