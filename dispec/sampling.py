"""Token sampling utilities (from scratch).

Kept deterministic and explicit because speculative decoding requires us to reason
about exact probabilities: rejection sampling compares draft vs target token
probabilities, so we need the same sampling transform applied consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0 => greedy (argmax)
    top_p: float = 1.0
    top_k: int = 0  # 0 => disabled

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask logits in-place style (returns new tensor) per top-k / top-p.

    logits: (..., vocab). Returns logits with filtered positions set to -inf.
    """
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        # Keep tokens up to and including the one that crosses top_p.
        remove = cum - probs > top_p
        remove[..., 0] = False
        remove_scattered = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove_scattered, float("-inf"))
    return logits


def logits_to_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Convert logits to the sampling distribution actually used to draw tokens.

    For greedy this returns a one-hot at argmax (so the same code path handles
    both greedy and sampling in rejection-sampling math).
    """
    if params.greedy:
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out
    logits = apply_top_k_top_p(logits / params.temperature, params.top_k, params.top_p)
    return logits.softmax(dim=-1)


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample token ids from a probability tensor of shape (..., vocab)."""
    flat = probs.reshape(-1, probs.size(-1))
    # Greedy one-hot rows sample deterministically; multinomial handles both.
    idx = torch.multinomial(flat, num_samples=1, generator=generator)
    return idx.reshape(probs.shape[:-1])


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None) -> torch.Tensor:
    """Logits -> sampled token ids of shape (...)."""
    if params.greedy:
        return logits.argmax(dim=-1)
    probs = logits_to_probs(logits, params)
    return sample_from_probs(probs, generator)
