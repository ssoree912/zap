# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless attention-output compaction for the LLaVA-1.5 Q2 runner.

The eval-700 ``run_one`` implementation requests the full prompt attention
matrix twice:

1. its explicit prefill pass uses the sum over every query row for H2O and the
   mean over semantic-question rows for the question-only selector;
2. ``generate`` uses only the final prompt-query row as the first element of
   the future trajectory.

Keeping all ``[B, H, Q, K]`` matrices alive is unnecessary.  This context
manager patches only the values returned to ``run_one``:

* on the first full-prefill call, each layer returns two query rows,
  ``[question_mean, h2o_sum - question_mean]``;
* on the second full-prefill call, each layer returns its final query row;
* one-query autoregressive calls are returned unchanged.

Consequently, the existing indexing and reductions in ``run_one`` retain
their original meaning without copying or modifying that function.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class _LayerCallState:
    """Per-layer count of calls whose attention has more than one query."""

    full_prefill_calls: int = 0


class Llava15AttentionCompaction:
    """Per-``run_one`` context that compacts returned self-attention weights.

    Args:
        model: LLaVA-1.5 model (or any module whose decoder attention modules
            are named ``self_attn``).
        agreement_module: Module/object that owns
            ``infer_user_question_positions``.

    A fresh context must be used for every sample because its first and second
    full-prefill calls have different meanings.
    """

    def __init__(self, model: nn.Module, agreement_module: Any) -> None:
        self._model = model
        self._agreement_module = agreement_module
        self._original_infer: Callable[..., torch.Tensor] | None = None
        self._wrapped_infer: Callable[..., torch.Tensor] | None = None
        self._hook_handles: list[Any] = []
        self._layer_states: dict[str, _LayerCallState] = {}
        self._actual_question_positions: torch.Tensor | None = None
        self._installed = False

    @property
    def actual_question_count(self) -> int:
        """Number of real semantic-question tokens found by the original helper."""

        if self._actual_question_positions is None:
            raise RuntimeError(
                "Question positions have not been inferred in this compaction context"
            )
        return int(self._actual_question_positions.numel())

    @property
    def actual_question_positions(self) -> torch.Tensor:
        """A defensive copy of the uncompressed semantic-question positions."""

        if self._actual_question_positions is None:
            raise RuntimeError(
                "Question positions have not been inferred in this compaction context"
            )
        return self._actual_question_positions.clone()

    @property
    def full_prefill_call_counts(self) -> Mapping[str, int]:
        """Full-prefill attention call count for each hooked decoder layer."""

        return {
            name: state.full_prefill_calls
            for name, state in self._layer_states.items()
        }

    def _decoder_self_attentions(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield only language-decoder attention modules, never vision modules."""

        candidates: list[tuple[str, Any]] = []
        language_model = getattr(self._model, "model", None)
        if language_model is not None:
            candidates.append(("model.layers", getattr(language_model, "layers", None)))
        candidates.append(("layers", getattr(self._model, "layers", None)))

        layers_name: str | None = None
        layers: Any = None
        for candidate_name, candidate_layers in candidates:
            if candidate_layers is not None:
                layers_name = candidate_name
                layers = candidate_layers
                break
        if layers is None or layers_name is None:
            raise ValueError(
                "Could not locate LLaVA language decoder layers at model.model.layers"
            )

        try:
            layer_count = len(layers)
        except TypeError as error:
            raise TypeError("Language decoder layers must be a sized sequence") from error

        config = getattr(self._model, "config", None)
        expected_count = getattr(config, "num_hidden_layers", layer_count)
        if int(expected_count) != layer_count:
            raise ValueError(
                "Language decoder layer-count mismatch: "
                f"config.num_hidden_layers={expected_count}, actual={layer_count}"
            )
        if layer_count < 1:
            raise ValueError("Language decoder has no layers")

        for index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if not isinstance(attention, nn.Module):
                raise TypeError(
                    f"{layers_name}.{index}.self_attn is not a torch module"
                )
            yield f"{layers_name}.{index}.self_attn", attention

    @staticmethod
    def _replace_attention(
        output: Any,
        attention: torch.Tensor,
    ) -> Any:
        if isinstance(output, tuple):
            values = list(output)
            values[1] = attention
            return tuple(values)
        if isinstance(output, list):
            values = list(output)
            values[1] = attention
            return values
        raise TypeError(
            "Expected self_attn output to be a tuple or list when attention "
            f"weights are present, got {type(output).__name__}"
        )

    def _make_attention_hook(
        self,
        layer_name: str,
        state: _LayerCallState,
    ) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                return None
            attention = output[1]
            if attention is None:
                return None
            if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
                raise ValueError(
                    f"{layer_name} returned an attention object with shape/type "
                    f"{getattr(attention, 'shape', type(attention).__name__)}; "
                    "expected [B,H,Q,K]"
                )

            query_length = int(attention.shape[-2])
            if query_length == 1:
                # Autoregressive decoding already has the minimal representation.
                return None
            if query_length < 1:
                raise ValueError(f"{layer_name} returned an empty query dimension")

            call_index = state.full_prefill_calls
            if call_index == 0:
                positions = self._actual_question_positions
                if positions is None:
                    raise RuntimeError(
                        "infer_user_question_positions must run before the first "
                        "full-prefill model call"
                    )
                question_index = positions.to(
                    device=attention.device,
                    dtype=torch.long,
                )
                if question_index.ndim != 1 or question_index.numel() == 0:
                    raise ValueError("Semantic-question position tensor is empty")
                minimum = int(question_index.min().item())
                maximum = int(question_index.max().item())
                if minimum < 0 or maximum >= query_length:
                    raise IndexError(
                        f"Question rows [{minimum}, {maximum}] are outside "
                        f"{layer_name}'s query length {query_length}"
                    )

                # Match eval-700's exact reduction order:
                # selected.float().mean(...) for question saliency, versus
                # attention.sum(...).float() for cumulative H2O.
                question_mean = attention.index_select(
                    -2,
                    question_index,
                ).float().mean(dim=-2, keepdim=True)
                h2o_sum = attention.sum(dim=-2, keepdim=True).float()
                compact = torch.cat(
                    (question_mean, h2o_sum - question_mean),
                    dim=-2,
                )
            elif call_index == 1:
                # ``generate`` reads only attention[..., -1, :] from its prompt
                # prefill when constructing the future trajectory.  clone() is
                # essential: a view would retain the full quadratic storage.
                compact = attention[..., -1:, :].clone()
            else:
                raise RuntimeError(
                    f"{layer_name} received an unexpected third full-prefill "
                    "attention call in one per-sample compaction context"
                )

            state.full_prefill_calls += 1
            return self._replace_attention(output, compact)

        return hook

    def install(self) -> "Llava15AttentionCompaction":
        """Install the temporary inference wrapper and forward hooks."""

        if self._installed:
            raise RuntimeError("Attention compaction context is already installed")

        original = getattr(
            self._agreement_module,
            "infer_user_question_positions",
            None,
        )
        if not callable(original):
            raise TypeError(
                "agreement_module.infer_user_question_positions must be callable"
            )

        decoder_attentions = list(self._decoder_self_attentions())
        if not decoder_attentions:
            raise ValueError("No decoder modules named self_attn were found")

        self._original_infer = original
        self._actual_question_positions = None
        self._layer_states = {
            name: _LayerCallState() for name, _module in decoder_attentions
        }

        def wrapped_infer(*args: Any, **kwargs: Any) -> torch.Tensor:
            positions = original(*args, **kwargs)
            if not isinstance(positions, torch.Tensor):
                raise TypeError(
                    "infer_user_question_positions must return a torch.Tensor"
                )
            if positions.ndim != 1 or positions.numel() == 0:
                raise ValueError(
                    "infer_user_question_positions returned an empty or non-vector "
                    "position tensor"
                )
            self._actual_question_positions = positions.detach().clone()
            # Existing run_one now selects compact row zero for question saliency.
            return positions.new_zeros((1,))

        self._wrapped_infer = wrapped_infer
        setattr(
            self._agreement_module,
            "infer_user_question_positions",
            wrapped_infer,
        )
        try:
            for name, module in decoder_attentions:
                handle = module.register_forward_hook(
                    self._make_attention_hook(name, self._layer_states[name])
                )
                self._hook_handles.append(handle)
        except BaseException:
            self.close()
            raise

        self._installed = True
        return self

    def close(self) -> None:
        """Remove all hooks and restore the exact original inference callable."""

        for handle in reversed(self._hook_handles):
            handle.remove()
        self._hook_handles.clear()

        if (
            self._wrapped_infer is not None
            and getattr(
                self._agreement_module,
                "infer_user_question_positions",
                None,
            )
            is self._wrapped_infer
            and self._original_infer is not None
        ):
            setattr(
                self._agreement_module,
                "infer_user_question_positions",
                self._original_infer,
            )
        self._installed = False

    def __enter__(self) -> "Llava15AttentionCompaction":
        return self.install()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: Any,
    ) -> None:
        self.close()


@contextmanager
def compact_llava15_attentions(
    model: nn.Module,
    agreement_module: Any,
) -> Iterator[Llava15AttentionCompaction]:
    """Yield a safely restored per-sample LLaVA-1.5 compaction context.

    Typical integration without copying the established runner is::

        with compact_llava15_attentions(model, q2.agreement) as state:
            result, masks = eval700.run_one(model=model, ...)
        result["question_token_count"] = state.actual_question_count
    """

    with Llava15AttentionCompaction(model, agreement_module) as state:
        yield state
