"""LOOK-M's image-only KV cache eviction/merge policy, extracted verbatim.

Copied from LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py's
``ImageOnlyKVCache_LayerWise`` and ``infer_image_token_spans`` (commit
d5cfe0f, "Add image-only KV cache compression and local-run fixes"). The
class body is pure tensor math with no Llama-specific dependency, so it is
reproduced here unmodified to avoid importing look_rebuttal's vendored
``llava`` package (transformers==4.33.1, Llama-only) into the OneVision
(transformers==4.46.0, Qwen2) process.

Only change from the original: none. Callers are responsible for reducing
attention-head-space scores down to KV-head-space before calling this (GQA
models have fewer KV heads than attention heads; LLaVA-1.5 was MHA so this
never came up there).
"""

import torch
import torch.nn.functional as F


def infer_image_token_spans(input_ids, expanded_seq_len, image_token_index=-200):
    """Return [start, end) image-token spans after LLaVA expands placeholders.

    Derives the expansion from the actual prompt instead of hard-coding a
    per-image token count, so it works for any tokens-per-image scheme
    (LLaVA-1.5's fixed 576, or OneVision's pooled/anyres variable counts).

    Requires a single prompt per generation call (batch size 1).
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "Image-only LOOK-M cache compression requires a single prompt per "
            f"generation call, got input_ids shape {tuple(input_ids.shape)}."
        )

    image_positions = torch.where(input_ids[0] == image_token_index)[0]
    num_images = image_positions.numel()
    if num_images == 0:
        return torch.empty((0, 2), dtype=torch.long, device=input_ids.device)

    text_tokens = input_ids.shape[1] - num_images
    total_image_tokens = int(expanded_seq_len) - text_tokens
    if total_image_tokens <= 0 or total_image_tokens % num_images != 0:
        raise ValueError(
            "Could not infer equal-sized image-token spans: "
            f"{num_images} placeholders expanded to {total_image_tokens} tokens."
        )

    tokens_per_image = total_image_tokens // num_images
    spans = []
    expansion_offset = 0
    for position in image_positions.tolist():
        start = position + expansion_offset
        end = start + tokens_per_image
        spans.append((start, end))
        expansion_offset += tokens_per_image - 1

    return torch.tensor(spans, dtype=torch.long, device=input_ids.device)


class ImageOnlyKVCache_LayerWise:
    """Compress only visual KV pairs while preserving every textual KV pair.

    Attention scores are still used to rank visual tokens, retaining LOOK-M's
    text/vision relevance signal. ``image_keep_ratio`` is relative to the number
    of visual tokens. ``total_keep_ratio`` instead budgets the complete prompt
    KV cache: every text token is protected first and visual tokens fill the
    remaining slots. The legacy heavy-hitter/recent ratios remain available as
    a fallback. Compression happens once at prompt prefill; subsequent decoded
    tokens are text and are appended without pruning.
    """

    _MERGE_STRATEGIES = {"none", "avg", "weighted", "pivot"}

    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        image_keep_ratio=None,
        total_keep_ratio=None,
        merge_strategy="none",
    ):
        if merge_strategy not in self._MERGE_STRATEGIES:
            raise ValueError(f"Unknown image-only merge strategy: {merge_strategy}")
        if image_keep_ratio is not None and not 0 <= image_keep_ratio <= 1:
            raise ValueError("image_keep_ratio must be between 0 and 1.")
        if total_keep_ratio is not None and not 0 <= total_keep_ratio <= 1:
            raise ValueError("total_keep_ratio must be between 0 and 1.")
        if image_keep_ratio is not None and total_keep_ratio is not None:
            raise ValueError(
                "image_keep_ratio and total_keep_ratio are mutually exclusive."
            )

        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.image_keep_ratio = image_keep_ratio
        self.total_keep_ratio = total_keep_ratio
        self.merge_strategy = merge_strategy
        self.hh_score = None
        self._current_image_mask = None

        # Set by the caller before prompt prefill.
        self.image_token_spans = torch.empty((0, 2), dtype=torch.long)
        self.image_position = torch.empty(0, dtype=torch.long)

        # Exposed for diagnostics and unit tests.
        self.last_keep_indices = None
        self.last_evicted_indices = None
        self.last_image_token_count = None
        self.last_image_keep_count = None
        self.last_text_token_count = None

    @property
    def has_image_tokens(self):
        return self.image_token_spans.numel() > 0

    def _update_hh_score(self, attn_score_cache):
        num_new_tokens = attn_score_cache.shape[2]
        new_score = attn_score_cache.sum(0).sum(1)
        if self.hh_score is not None:
            new_score[..., :-num_new_tokens] += self.hh_score
        self.hh_score = new_score

    def _image_mask(self, seq_len, device):
        image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for start, end in self.image_token_spans.tolist():
            start = max(0, min(int(start), seq_len))
            end = max(start, min(int(end), seq_len))
            image_mask[start:end] = True
        return image_mask

    @staticmethod
    def _gather_tokens(states, indices):
        bsz, num_heads, _, head_dim = states.shape
        gather_index = indices[None, :, :, None].expand(
            bsz, num_heads, indices.shape[-1], head_dim
        )
        return torch.gather(states, dim=2, index=gather_index)

    def _merge_evicted_images(
        self,
        key_cache,
        value_cache,
        original_key,
        original_value,
        keep_indices,
        evicted_indices,
        kept_image_mask,
    ):
        if self.merge_strategy == "none" or evicted_indices.shape[-1] == 0:
            return key_cache, value_cache

        num_heads = keep_indices.shape[0]
        target_cache_indices = torch.nonzero(
            kept_image_mask, as_tuple=False
        )[:, 1].view(num_heads, -1)
        if target_cache_indices.shape[-1] == 0:
            # Strict image-only mode never merges an image token into text.
            return key_cache, value_cache

        evicted_key = self._gather_tokens(original_key, evicted_indices)
        evicted_value = self._gather_tokens(original_value, evicted_indices)
        target_key = self._gather_tokens(key_cache, target_cache_indices)

        normalized_evicted = F.normalize(evicted_key, dim=-1, eps=1e-6)
        normalized_target = F.normalize(target_key, dim=-1, eps=1e-6)
        similarity = normalized_evicted @ normalized_target.transpose(-1, -2)
        max_values, target_local_indices = similarity.max(dim=-1)

        destination = torch.gather(
            target_cache_indices[None].expand(original_key.shape[0], -1, -1),
            dim=2,
            index=target_local_indices,
        )
        destination_expanded = destination.unsqueeze(-1).expand_as(evicted_key)

        if self.merge_strategy == "pivot":
            selected_key = torch.gather(key_cache, dim=2, index=destination_expanded)
            selected_value = torch.gather(value_cache, dim=2, index=destination_expanded)
            key_source = (evicted_key + selected_key) / 2
            value_source = (evicted_value + selected_value) / 2
        elif self.merge_strategy == "weighted":
            weights = max_values.unsqueeze(-1)
            key_source = weights * evicted_key
            value_source = weights * evicted_value
        else:
            key_source = evicted_key
            value_source = evicted_value

        key_cache = torch.scatter_reduce(
            key_cache,
            dim=2,
            index=destination_expanded,
            src=key_source,
            reduce="mean",
            include_self=True,
        )
        value_cache = torch.scatter_reduce(
            value_cache,
            dim=2,
            index=destination_expanded,
            src=value_source,
            reduce="mean",
            include_self=True,
        )
        return key_cache, value_cache

    def __call__(self, past_key_values, attn_score_cache):
        self._update_hh_score(attn_score_cache)
        if past_key_values is None or not self.has_image_tokens:
            return past_key_values

        key_states, value_states = past_key_values
        seq_len = key_states.size(self.k_seq_dim)
        num_heads = key_states.shape[1]
        if self.hh_score.shape[0] != num_heads:
            raise ValueError(
                "Image-only LOOK-M currently requires matching attention and "
                f"KV heads, got {self.hh_score.shape[0]} and {num_heads}."
            )

        # Visual tokens only occur in the prompt. Once prefill has been
        # compressed, every newly appended cache entry is a decoded text token.
        # Reusing the original prompt spans here would shift their coordinates
        # and eventually prune text by mistake.
        if self._current_image_mask is not None:
            cached_len = self._current_image_mask.shape[-1]
            if seq_len < cached_len:
                raise ValueError(
                    f"KV cache shrank unexpectedly from {cached_len} to {seq_len}."
                )
            if seq_len > cached_len:
                decoded_text_mask = torch.zeros(
                    (num_heads, seq_len - cached_len),
                    dtype=torch.bool,
                    device=key_states.device,
                )
                self._current_image_mask = torch.cat(
                    (self._current_image_mask.to(key_states.device), decoded_text_mask),
                    dim=-1,
                )
            return past_key_values

        hh_size = self.hh_size
        recent_size = self.recent_size
        if self.hh_ratio is not None:
            hh_size = int(seq_len * self.hh_ratio)
        if self.recent_ratio is not None:
            recent_size = int(seq_len * self.recent_ratio)
        recent_size = max(0, min(recent_size, seq_len))
        self.cache_size = hh_size + recent_size

        image_mask = self._image_mask(seq_len, key_states.device)
        if not image_mask.any():
            return past_key_values

        if self.total_keep_ratio is not None:
            # Protect every text token, then use the rest of the requested
            # whole-prompt budget for the highest-scoring visual tokens.
            candidate_indices = torch.where(image_mask)[0]
            text_token_count = int((~image_mask).sum().item())
            target_total_count = int(round(seq_len * self.total_keep_ratio))
            image_keep_count = target_total_count - text_token_count
            image_keep_count = max(
                0, min(image_keep_count, candidate_indices.numel())
            )
            keep_mask = (~image_mask)[None].expand(num_heads, -1).clone()
        elif self.image_keep_ratio is not None:
            # Exact image-only budget: the ratio is relative to visual tokens,
            # while all prompt and subsequently decoded text tokens are kept.
            candidate_indices = torch.where(image_mask)[0]
            image_keep_count = int(candidate_indices.numel() * self.image_keep_ratio)
            keep_mask = (~image_mask)[None].expand(num_heads, -1).clone()
        else:
            nonrecent_end = seq_len - recent_size
            recent_mask = (
                torch.arange(seq_len, device=key_states.device) >= nonrecent_end
            )
            nonrecent_image_mask = image_mask & ~recent_mask
            candidate_indices = torch.where(nonrecent_image_mask)[0]

            # Legacy behavior: all text and the recent window are protected;
            # images use whatever remains of the heavy-hitter budget.
            protected_nonrecent_text = int((~image_mask & ~recent_mask).sum().item())
            image_keep_count = max(0, hh_size - protected_nonrecent_text)
            image_keep_count = min(image_keep_count, candidate_indices.numel())
            keep_mask = (
                ((~image_mask) | recent_mask)[None].expand(num_heads, -1).clone()
            )

        if image_keep_count:
            candidate_scores = self.hh_score[:, candidate_indices]
            selected_local = torch.topk(
                candidate_scores, image_keep_count, dim=-1
            ).indices
            selected_images = candidate_indices[selected_local]
            keep_mask.scatter_(1, selected_images, True)

        keep_indices = torch.nonzero(keep_mask, as_tuple=False)[:, 1].view(num_heads, -1)
        evicted_mask = image_mask[None].expand(num_heads, -1) & ~keep_mask
        evicted_indices = torch.nonzero(
            evicted_mask, as_tuple=False
        )[:, 1].view(num_heads, -1)

        self.last_keep_indices = keep_indices.detach().cpu()
        self.last_evicted_indices = evicted_indices.detach().cpu()
        self.last_image_token_count = int(image_mask.sum().item())
        self.last_image_keep_count = int(
            (image_mask[None] & keep_mask).sum(dim=-1)[0].item()
        )
        self.last_text_token_count = int((~image_mask).sum().item())

        key_cache = self._gather_tokens(key_states, keep_indices)
        value_cache = self._gather_tokens(value_states, keep_indices)
        kept_image_mask = torch.gather(
            image_mask[None].expand(num_heads, -1), dim=1, index=keep_indices
        )
        key_cache, value_cache = self._merge_evicted_images(
            key_cache,
            value_cache,
            key_states,
            value_states,
            keep_indices,
            evicted_indices,
            kept_image_mask,
        )
        self._current_image_mask = kept_image_mask.detach()
        self.hh_score = torch.gather(self.hh_score, dim=1, index=keep_indices)
        return key_cache, value_cache

    def _clean_scores(self):
        self.hh_score = None
        self.image_token_spans = torch.empty((0, 2), dtype=torch.long)
        self.image_position = torch.empty(0, dtype=torch.long)
        self._current_image_mask = None
        self.last_keep_indices = None
        self.last_evicted_indices = None
        self.last_image_token_count = None
        self.last_image_keep_count = None
        self.last_text_token_count = None
