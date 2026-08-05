"""Delayed-replay decode utilities, adapted from zap/foresight/eval/one_token_replay.py
and kv_decode_utils.py (same repo family, OneVision/DynamicCache-compatible).

The pattern: hold out the last prompt token, prefill the rest (this is where
LOOK-M's image-only eviction fires, once), then "replay" the held-out token
through the now-pruned cache before generating anything -- so the first real
answer token is produced under the pruned cache, not the full one. Without
this, LlavaQwenForCausalLM's single generate() call produces the first token
from the full-attention prefill forward pass, before the cache shrinks, which
makes multiple-choice accuracy insensitive to the keep ratio.

Bypasses LlavaQwenForCausalLM.forward()/.generate() for these steps because
that wrapper does not forward `cache_position` to Qwen2Model.forward() (see
llava_qwen.py forward(), lines 101-113) -- letting it default to None would
make Qwen2Model re-derive cache_position from the *pruned* physical cache
length instead of the true absolute position, corrupting RoPE and the
DynamicCache append point. Calling model.get_model() directly and passing
`cache_position`/`position_ids` explicitly sidesteps that.
"""

import torch


@torch.no_grad()
def forward_one_token_with_kv(model, input_ids, past_kv, absolute_position):
    language_model = model.get_model()
    inputs_embeds = language_model.embed_tokens(input_ids)
    cache_position = torch.tensor(
        [int(absolute_position)], dtype=torch.long, device=input_ids.device
    )
    outputs = language_model(
        input_ids=None,
        attention_mask=None,
        position_ids=cache_position.unsqueeze(0),
        past_key_values=past_kv,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=cache_position,
    )
    return model.lm_head(outputs.last_hidden_state), outputs.past_key_values


@torch.no_grad()
def greedy_decode_after_prompt_replay(
    model,
    past_kv,
    first_next_token,
    prompt_len,
    eos_token_id,
    max_new_tokens,
):
    out_tokens = [int(first_next_token.item())]
    if out_tokens[0] == eos_token_id:
        return out_tokens

    next_token = first_next_token
    pos = int(prompt_len)
    for _ in range(max_new_tokens - 1):
        logits, past_kv = forward_one_token_with_kv(model, next_token, past_kv, pos)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = int(next_token.item())
        out_tokens.append(token_id)
        pos += 1
        if token_id == eos_token_id:
            break
    return out_tokens
