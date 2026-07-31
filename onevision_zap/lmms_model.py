# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from PIL import Image
from tqdm import tqdm

from onevision_zap.model import OneVisionAdapterError, PreparedVisuals, ZapOneVisionModel


@register_model("zap_onevision_student")
class ZapOneVisionStudent(lmms):
    def __init__(
        self,
        pretrained: str,
        student_path: str,
        image_keep_ratio: float = 0.1,
        max_frames_num: int = 32,
        conv_template: str = "qwen_1_5",
        batch_size: str | int = 1,
        device: str = "cuda:0",
        **kwargs: str,
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise OneVisionAdapterError("ZAP OneVision requires batch size one")
        if device != "cuda:0":
            raise OneVisionAdapterError("ZAP OneVision requires logical device cuda:0")
        if kwargs:
            raise OneVisionAdapterError(f"unexpected model arguments: {sorted(kwargs)}")
        self._model = ZapOneVisionModel(
            Path(pretrained),
            Path(student_path),
            image_keep_ratio=float(image_keep_ratio),
            conv_template=conv_template,
        )
        self._max_frames_num = int(max_frames_num)
        self._prepared_key: str | None = None
        self._prepared_visuals: PreparedVisuals | None = None
        self._reported_protocol = False

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("ZAP OneVision supports generation tasks only")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        responses: list[str] = []
        for request in tqdm(requests, desc="ZAP OneVision"):
            arguments = request.args
            if len(arguments) != 6:
                raise OneVisionAdapterError("generate request must contain six arguments")
            context = arguments[0]
            generation_kwargs = arguments[1]
            visual_loader = arguments[2]
            document_id = arguments[3]
            task_name = arguments[4]
            split = arguments[5]
            document = self.task_dict[task_name][split][document_id]
            visuals = visual_loader(document)
            visual_key = self._visual_key(visuals)
            if visual_key != self._prepared_key:
                self._prepared_visuals = self._model.prepare_visuals(
                    self._frames(visuals),
                )
                self._prepared_key = visual_key
            if self._prepared_visuals is None:
                raise OneVisionAdapterError("visual preparation did not produce an input")
            generated = self._model.generate_prepared(
                context,
                self._prepared_visuals,
                max_new_tokens=int(generation_kwargs.get("max_new_tokens", 16)),
                is_video=self._is_video_document(document),
            )
            if not self._reported_protocol:
                stats = generated.eviction
                print(
                    "[zap-onevision] "
                    f"first_token_source=post_eviction_prompt_replay "
                    f"image_keep_ratio={stats.image_tokens_kept / stats.image_tokens:.6f} "
                    f"image_tokens={stats.image_tokens} "
                    f"image_tokens_kept={stats.image_tokens_kept} "
                    f"text_tokens_kept={stats.text_tokens}",
                    flush=True,
                )
                self._reported_protocol = True
            responses.append(generated.generated_text)
            self.cache_hook.add_partial(
                "generate_until",
                (context, generation_kwargs),
                generated.generated_text,
            )
        return responses

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("ZAP OneVision does not support multi-round tasks")

    def _frames(self, visuals: Sequence[Image.Image | str]) -> list[Image.Image]:
        if not visuals:
            raise OneVisionAdapterError("visual loader returned no frames")
        first = visuals[0]
        if isinstance(first, str):
            reader = VideoReader(first, ctx=cpu(0), num_threads=1)
            if len(reader) < 1:
                raise OneVisionAdapterError(f"video contains no frames: {first}")
            indices = np.linspace(
                0,
                len(reader) - 1,
                self._max_frames_num,
                dtype=np.int64,
            )
            return [Image.fromarray(frame).convert("RGB") for frame in reader.get_batch(indices).asnumpy()]
        return [frame for frame in visuals if isinstance(frame, Image.Image)]

    @staticmethod
    def _visual_key(visuals: Sequence[Image.Image | str]) -> str:
        if visuals and isinstance(visuals[0], str):
            return visuals[0]
        return f"frames:{id(visuals)}"

    @staticmethod
    def _is_video_document(document: Mapping[str, str]) -> bool:
        return document.get("data_type") == "video"
