# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

# Establish the CUDA context before importing anything that pulls in decord or
# PyAV (av). Those libraries initialize their own multimedia/CUDA state at import
# time, and if they load before torch creates its CUDA context, the deferred
# torch.cuda._lazy_init() invoked by accelerate's device_map="auto" balanced
# memory probe segfaults. Touching the device here forces the context first.
import torch

torch.cuda.init()
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

from lmms_eval.models import AVAILABLE_MODELS

from onevision_zap import lmms_model


def main() -> None:
    sys.modules["lmms_eval.models.zap_onevision_student"] = lmms_model
    AVAILABLE_MODELS["zap_onevision_student"] = "ZapOneVisionStudent"
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
