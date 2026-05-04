# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn
from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    PreTrainedModel,
    Qwen2ForCausalLM,
)

try:
    from transformers import QuantizedCache
except ImportError:
    class QuantizedCache:  # type: ignore[override]
        pass

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None  # type: ignore[assignment]

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    Gemma3ForConditionalGeneration = None  # type: ignore[assignment]

from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = tuple(
    model_cls
    for model_cls in (
        LlamaForCausalLM,
        MistralForCausalLM,
        Phi3ForCausalLM,
        Qwen2ForCausalLM,
        Qwen3ForCausalLM,
        Gemma3ForConditionalGeneration,
    )
    if model_cls is not None
)


def _resolve_decoder_model(model: PreTrainedModel):
    """Return the decoder module that owns `.layers` for LM or VLM wrappers."""
    candidates = []
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model"):
            candidates.append(inner.language_model)
        candidates.append(inner)
    candidates.append(model)

    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "layers"):
            return candidate
        nested = getattr(candidate, "model", None)
        if nested is not None and hasattr(nested, "layers"):
            return nested

    raise AttributeError(f"Could not resolve decoder layers for model {type(model)}")


@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.

    This class provides the foundation for implementing various key-value cache compression
    techniques. Subclasses must implement the `compress` method to define their specific
    compression logic.

    The compression is applied only during pre-filling (not during generation).
    """

    def post_init_from_model(self, model: PreTrainedModel):
        """
        Optional method to initialize press parameters from the model
        """
        pass

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where compression is applied.
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
            These represent the input to the attention layer.
        keys : torch.Tensor
            Key tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are keys ready for compression.
        values : torch.Tensor
            Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are values ready for compression.
        attentions : torch.Tensor
            Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if attention weights are not computed or needed.
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the compressed keys and values tensors. The returned tensors
            should have reduced sequence length dimension compared to the input tensors.
        """

        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression during the pre-filling phase by:
        1. Checking if we're still in pre-filling (not generation) phase
        2. Extracting keys and values from the cache (handling quantization)
        3. Calling the compress method to reduce the cache size
        4. Updating the cache with compressed keys and values

        The hook ensures compression is only applied during pre-filling and correctly
        handles both quantized and unquantized caches.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        input : list[torch.Tensor]
            Input tensors to the forward pass of the attention layer. This parameter
            is provided by PyTorch's hook mechanism but not used in the default implementation.
        kwargs : dict
            Keyword arguments passed to the attention layer's forward method, including:
            - hidden_states: Input embeddings to the attention layer
            - past_key_values: The KV cache object being modified
            - cache_position: Position indices indicating where we are in the sequence
            - position_embeddings: RoPE embeddings if applicable
        output : list
            Output from the attention layer's forward pass. Contains:
            - [0]: Hidden states output
            - [1]: Attention weights (may be None)

        Returns
        -------
        list
            The potentially modified output from the forward pass. This
            is the same as the input output, but the underlying cache has been compressed in-place.
        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = kwargs.get("past_key_value")
        if cache is None:
            return output
        cache_layer = cache.layers[module.layer_idx] if hasattr(cache, "layers") else None
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if kwargs["cache_position"][-1] > q_len:
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)

        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            if cache_layer is None:
                raise AttributeError(f"Quantized cache for {type(cache)} does not expose layers")
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.cumulative_length = keys.shape[2]
        elif cache_layer is not None:
            cache_layer.keys = keys
            cache_layer.values = values
        elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values
        else:
            raise AttributeError(f"Unsupported cache layout for {type(cache)}")

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.

        This method registers forward hooks on all attention layers of the model to enable
        automatic KV cache compression during the pre-filling phase. The hooks are automatically
        removed when exiting the context manager.

        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            The transformer model to apply compression to.

        Examples
        --------
        >>> press = my_press
        >>> with press(model):
        ...     # Forward pass with compression applied
        ...     outputs = model(input_ids, past_key_values=cache)
        """
        if not isinstance(model, SUPPORTED_MODELS):
            if not getattr(self, "_warned_unsupported_model", False):
                logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")
                self._warned_unsupported_model = True

        is_gemma3 = Gemma3ForConditionalGeneration is not None and isinstance(model, Gemma3ForConditionalGeneration)
        if is_gemma3:
            logger.warning_once("Compression in Gemma3 is only applied to layer without sliding window attention")

        self.post_init_from_model(model)
        hooks = []
        try:
            language_model = _resolve_decoder_model(model)
            for layer in language_model.layers:
                if is_gemma3 and layer.self_attn.is_sliding:
                    # Skip layers with sliding window attention, only for Gemma3
                    continue
                if hasattr(language_model, "rotary_emb"):
                    layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
