"""Escalation serving: hand a prompt from a small model to a large one without re-prefilling.

A request is served by the small model first. If it has to escalate -- low
confidence, a harder task, a premium tier -- the large model would normally prefill
the whole prompt again. Instead, the small model's KV cache is mapped into the large
model's layout (dispec/kv/transfer.py) and the large model is admitted with it, so
it forwards only the last prompt token before decoding.

Only n-1 prompt tokens are transferred. The logits that choose the first output
token are produced by forwarding the last prompt token, so that token has to go
through the target model itself.

These helpers are in-process, like `prefill_export`/`decode_from_kv` in
`disaggregated.py`; the source and target may equally live in separate workers with a
transport between them.
"""

from __future__ import annotations

import torch

from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.kv.transfer import CrossModelKVMap, rope_tables
from dispec.sampling import SamplingParams


def map_prefix(source_runner: ModelRunner, source_cache: PagedKVCache, block_ids: list[int],
               n_tokens: int, kv_map: CrossModelKVMap, target_runner: ModelRunner,
               use_residual: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the first n_tokens-1 tokens of a source sequence's KV into the target's layout.

    Args:
        source_runner, source_cache, block_ids: where the source sequence's KV lives.
        n_tokens: the prompt length (tokens stored for the sequence).
        kv_map: a map from the source model to the target model.
        target_runner: the target, whose rotary tables the keys are re-rotated with.

    Returns:
        (k, v) of shape (target_layers, n_tokens - 1, H, D), ready for
        `ContinuousBatchingEngine.add_request(prefix_kv=...)`.
    """
    n = n_tokens - 1
    k, v = source_cache.export_contiguous(block_ids, n)
    positions = torch.arange(n, device=k.device)
    return kv_map.map(k, v, rope_tables(source_runner.rotary_emb, positions),
                      rope_tables(target_runner.rotary_emb, positions), use_residual)


@torch.inference_mode()
def prefill_and_map(source_runner: ModelRunner, source_cache: PagedKVCache, sid: int,
                    prompt_ids: list[int], kv_map: CrossModelKVMap,
                    target_runner: ModelRunner, use_residual: bool = True):
    """Prefill a prompt on the source model and return its KV mapped for the target.

    For when the source has not already served the request. Frees the source sequence.
    """
    n = len(prompt_ids)
    table = source_cache.manager.allocate(sid, n)
    dev = source_runner.device
    slots = source_cache.context_slots(table.block_ids, n)
    table.num_tokens = n
    source_runner.forward(torch.tensor(prompt_ids, device=dev), torch.arange(n, device=dev),
                          slots, [SeqMeta(table.block_ids, n, n)], source_cache)
    try:
        return map_prefix(source_runner, source_cache, table.block_ids, n, kv_map,
                          target_runner, use_residual)
    finally:
        source_cache.manager.free(sid)


def escalate(target_engine, prompt_ids: list[int], mapped_kv, max_new_tokens: int,
             params: SamplingParams | None = None, eos_id: int | None = None,
             priority: int = 0) -> int:
    """Admit a request to the target engine with transferred prefix KV. Returns its id."""
    return target_engine.add_request(prompt_ids, max_new_tokens, params, eos_id,
                                     priority=priority, prefix_kv=mapped_kv)
