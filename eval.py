#!/usr/bin/env python3
"""Foresight: evaluate visual-utility student KV pruning.

Dispatches to lmms-eval or VLMEvalKit based on --framework.
All arguments after '--' separator are forwarded to the underlying framework.

Examples:
    # lmms-eval with OneVision student (post-prefill eviction)
    python eval.py --model onevision --framework lmms -- \\
        --model llava_onevision_student \\
        --model_args pretrained=...,student_path=...,keep_ratio=0.5 \\
        --tasks chartqa_local --batch_size 1 --output_path results/

    # lmms-eval with progressive student (layer-by-layer eviction during prefill)
    python eval.py --model onevision --framework lmms --variant progressive -- \\
        --model llava_onevision_student_progressive \\
        --model_args pretrained=...,student_path=...,keep_ratio=0.5,progressive_schedule=immediate \\
        --tasks chartqa_local --batch_size 1 --output_path results/

    # VLMEvalKit with OneVision student
    python eval.py --model onevision --framework vlmeval -- \\
        --data ChartQA --model LLaVA_OneVision_Student
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

LMMS_EVAL_ROOT = Path("/workspace/VFlowOpt/src/lmms_eval-0.2.4")
ZAP_ROOT = Path("/workspace/zap")

# Insert immediately at import time so sub-imports from foresight/* work
for _p in [str(ZAP_ROOT), str(LMMS_EVAL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_paths():
    pass  # already done at module level


def main():
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True, choices=["llava15", "onevision"],
                   help="Which VLM backbone")
    p.add_argument("--framework", required=True, choices=["lmms", "vlmeval"],
                   help="Evaluation framework to use")
    p.add_argument("--variant", default="student", choices=["student", "progressive"],
                   help="student: post-prefill eviction / progressive: layer-by-layer during prefill")
    args, remaining = p.parse_known_args()

    if remaining and remaining[0] == "--":
        remaining = remaining[1:]

    _ensure_paths()

    if args.model == "onevision" and args.framework == "lmms":
        import lmms_eval.models as _models_pkg
        if args.variant == "progressive":
            import foresight.eval.lmms_onevision_student_progressive as _mod
            sys.modules["lmms_eval.models.llava_onevision_student_progressive"] = _mod
            _models_pkg.AVAILABLE_MODELS["llava_onevision_student_progressive"] = (
                "LlavaOnevisionStudentProgressive"
            )
        else:
            import foresight.eval.lmms_onevision_student as _mod
            sys.modules["lmms_eval.models.llava_onevision_student"] = _mod
            _models_pkg.AVAILABLE_MODELS["llava_onevision_student"] = "LlavaOnevisionStudent"
        sys.argv = [str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py")] + remaining
        runpy.run_path(str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py"), run_name="__main__")

    elif args.model == "onevision" and args.framework == "vlmeval":
        from foresight.eval.vlmeval_onevision_student import LLaVA_OneVision_HF_Student
        import vlmeval.vlm as _vlm_pkg
        _vlm_pkg.LLaVA_OneVision_HF_Student = LLaVA_OneVision_HF_Student
        sys.argv = ["run.py"] + remaining
        from vlmeval.run import main as vlmeval_main
        vlmeval_main()

    elif args.model == "llava15" and args.framework == "lmms":
        import lmms_eval.models as _models_pkg
        import foresight.eval.lmms_llava15_student as _mod
        sys.modules["lmms_eval.models.llava15_student"] = _mod
        _models_pkg.AVAILABLE_MODELS["llava15_student"] = "Llava15Student"
        sys.argv = [str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py")] + remaining
        runpy.run_path(str(LMMS_EVAL_ROOT / "lmms_eval" / "__main__.py"), run_name="__main__")

    elif args.model == "llava15":
        raise NotImplementedError(f"llava15 + {args.framework} not yet implemented")


if __name__ == "__main__":
    main()
