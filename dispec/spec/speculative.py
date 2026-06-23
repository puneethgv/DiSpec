"""Sequential speculative decoding: small draft proposes, big target verifies.

Per iteration:
  1. draft autoregressively proposes K tokens (cheap), recording its distributions q;
  2. the target verifies all K in a single forward, giving distributions p;
  3. rejection sampling (dispec.spec.rejection) accepts a lossless prefix + 1 token.

The target forward processes K tokens to confirm up to K+1 output tokens, cutting the
number of expensive target calls. We keep separate paged caches for draft and target
and roll back the KV of rejected tokens with BlockManager.truncate.

Design note: we carry each model's "next-token distribution" across iterations. After
rejection we keep the target KV of accepted tokens and forward only the single
correction token; the cheap draft re-forwards the (n+1) confirmed tokens to resync.
Batch size 1 (single sequence) — batched speculative decoding is a later optimization.
"""

from __future__ import annotations

import itertools

import torch

from dispec.config import KVConfig, SpecConfig
from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.sampling import SamplingParams, logits_to_probs, sample_from_probs
from dispec.spec.rejection import rejection_sample


class SpecStats:
    def __init__(self):
        self.iters = 0
        self.proposed = 0
        self.accepted = 0
        self.emitted = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_iter(self) -> float:
        return self.emitted / self.iters if self.iters else 0.0


class SpeculativeEngine:
    _ids = itertools.count(1)

    def __init__(self, target_model, draft_model, spec: SpecConfig | None = None,
                 kv: KVConfig | None = None, num_blocks: int = 4096):
        self.tgt = ModelRunner(target_model)
        self.drf = ModelRunner(draft_model)
        self.spec = spec or SpecConfig()
        kv = kv or KVConfig()
        self.block_size = kv.block_size
        self.tgt_cache = PagedKVCache.for_model(
            target_model.config, num_blocks, kv.block_size, self.tgt.dtype, str(self.tgt.device))
        self.drf_cache = PagedKVCache.for_model(
            draft_model.config, num_blocks, kv.block_size, self.drf.dtype, str(self.drf.device))
        self.dev = self.tgt.device

    def _forward(self, runner: ModelRunner, cache: PagedKVCache, sid: int,
                 tokens: list[int], start_pos: int) -> torch.Tensor:
        """Forward `tokens` at positions [start_pos, ...] into the cache; return logits (n, V)."""
        table = cache.manager.block_table(sid)
        n = len(tokens)
        cache.manager.reserve(sid, start_pos + n - table.num_tokens)
        positions = torch.arange(start_pos, start_pos + n, device=self.dev)
        slots = cache.slots_for_positions(table.block_ids, positions)
        table.num_tokens = start_pos + n
        hidden = runner.forward(
            torch.tensor(tokens, device=self.dev, dtype=torch.long),
            positions, slots, [SeqMeta(table.block_ids, n, start_pos + n)], cache)
        return runner.logits(hidden)

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int,
                 params: SamplingParams | None = None, eos_token_id: int | None = None,
                 generator: torch.Generator | None = None) -> tuple[list[int], SpecStats]:
        params = params or SamplingParams()
        stats = SpecStats()
        sid = next(self._ids)
        K = self.spec.num_speculative_tokens
        m = len(prompt_ids)

        self.tgt_cache.manager.allocate(sid, m)
        self.drf_cache.manager.allocate(sid, m)
        try:
            # Prefill prompt into both; carry each model's next-token distribution.
            tgt_logits = self._forward(self.tgt, self.tgt_cache, sid, prompt_ids, 0)
            drf_logits = self._forward(self.drf, self.drf_cache, sid, prompt_ids, 0)
            p_next = logits_to_probs(tgt_logits[-1], params)
            q_next = logits_to_probs(drf_logits[-1], params)

            out: list[int] = []
            while len(out) < max_new_tokens:
                # 1) Draft K tokens from the draft model.
                draft_tokens: list[int] = []
                q_stack = [q_next]
                tok = int(sample_from_probs(q_next, generator))
                draft_tokens.append(tok)
                for _ in range(K - 1):
                    dl = self._forward(self.drf, self.drf_cache, sid, [tok],
                                       self.drf_cache.manager.block_table(sid).num_tokens)
                    qd = logits_to_probs(dl[-1], params)
                    q_stack.append(qd)
                    tok = int(sample_from_probs(qd, generator))
                    draft_tokens.append(tok)

                # 2) Target verifies all K drafted tokens in one forward.
                tgt_start = self.tgt_cache.manager.block_table(sid).num_tokens
                tl = self._forward(self.tgt, self.tgt_cache, sid, draft_tokens, tgt_start)
                p_stack = [p_next] + [logits_to_probs(tl[j], params) for j in range(K - 1)]
                p_bonus = logits_to_probs(tl[K - 1], params)

                # 3) Rejection sampling -> lossless prefix + 1 token.
                emitted, n_acc = rejection_sample(
                    torch.tensor(draft_tokens, device=self.dev),
                    torch.stack(q_stack), torch.stack(p_stack), p_bonus, generator)
                stats.iters += 1
                stats.proposed += K
                stats.accepted += n_acc

                # 4) Roll back rejected KV; resync caches and carried distributions.
                #    Target: keep accepted KV, forward only the correction token.
                self.tgt_cache.manager.truncate(sid, tgt_start + n_acc)
                tl2 = self._forward(self.tgt, self.tgt_cache, sid, [emitted[-1]], tgt_start + n_acc)
                p_next = logits_to_probs(tl2[-1], params)
                #    Draft: accepted draft tokens are already cached (drafting forwarded
                #    t_1..t_{K-1}); only forward what's missing + the correction token.
                drf_base = m + len(out)
                if n_acc <= K - 1:
                    self.drf_cache.manager.truncate(sid, drf_base + n_acc)
                    to_fwd = [emitted[-1]]
                else:  # all K accepted: t_K was never forwarded into the draft cache
                    self.drf_cache.manager.truncate(sid, drf_base + K - 1)
                    to_fwd = [draft_tokens[K - 1], emitted[-1]]
                dl2 = self._forward(self.drf, self.drf_cache, sid, to_fwd,
                                    self.drf_cache.manager.block_table(sid).num_tokens)
                q_next = logits_to_probs(dl2[-1], params)

                # 5) Emit, honoring max_new_tokens and EOS.
                for t in emitted:
                    out.append(t)
                    stats.emitted += 1
                    if len(out) >= max_new_tokens or t == eos_token_id:
                        return out[:max_new_tokens], stats
            return out[:max_new_tokens], stats
        finally:
            self.tgt_cache.manager.free(sid)
            self.drf_cache.manager.free(sid)
