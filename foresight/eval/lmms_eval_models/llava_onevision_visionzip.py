"""lmms-eval adapter: LLaVA-OneVision with VisionZip-style token eviction.

Subclass `Llava_OneVision` and apply `visionzip_onevision(model, keep_ratio=...)`
right after the standard model load. Everything else (loglikelihood, generate,
video handling, ...) is inherited unchanged.
"""

from typing import Optional

import torch  # noqa: F401  — pulled in by parent

from lmms_eval.api.registry import register_model
from lmms_eval.models.llava_onevision import Llava_OneVision


@register_model("llava_onevision_visionzip")
class Llava_OneVision_VisionZip(Llava_OneVision):
    def __init__(
        self,
        *args,
        visionzip_keep_ratio: Optional[float] = 0.5,
        visionzip_contextual_ratio: Optional[float] = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        keep_ratio = float(visionzip_keep_ratio)
        contextual_ratio = float(visionzip_contextual_ratio)
        if not (0 < keep_ratio <= 1):
            raise ValueError(f"visionzip_keep_ratio must be in (0, 1]; got {keep_ratio}")

        from visionzip import visionzip_onevision

        visionzip_onevision(self._model, keep_ratio=keep_ratio, contextual_ratio=contextual_ratio)
        self._visionzip_keep_ratio = keep_ratio
        self._visionzip_contextual_ratio = contextual_ratio
