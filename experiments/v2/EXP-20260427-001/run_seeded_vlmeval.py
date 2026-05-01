# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Launch VLMEvalKit without applying an experiment-local seed.

Kept for backward compatibility with old commands that referenced this
experiment-local launcher. It now only registers the local OneVision wrapper
and delegates to `run.py`.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path("/workspace/zap")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LMUData", "/workspace/zap/data/eval_LMU")

import vlmeval.vlm  # noqa: E402
from foresight.eval.vlmeval_onevision_student import LLaVA_OneVision_HF_Student  # noqa: E402

vlmeval.vlm.LLaVA_OneVision_HF_Student = LLaVA_OneVision_HF_Student
sys.modules["vlmeval.vlm"].__dict__["LLaVA_OneVision_HF_Student"] = LLaVA_OneVision_HF_Student


def main() -> int:
    forwarded = sys.argv[1:]
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    sys.argv = ["run.py"] + forwarded
    runpy.run_path("/workspace/VLMEvalKit/run.py", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
