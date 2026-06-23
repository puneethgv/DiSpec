"""Speculative-decoding verification via rejection sampling.

Implements the lossless acceptance test from Leviathan et al. (2023) / Chen et al.
(2023): given draft tokens proposed from distributions q and the target's own
distributions p at the same positions, accepting/correcting this way yields samples
*exactly* from the target distribution, regardless of draft quality.

For each speculative position i (token t_i drawn from q_i):
    accept t_i with probability min(1, p_i[t_i] / q_i[t_i])
    on the first rejection, emit one token sampled from normalize((p_i - q_i)+)
    if all K are accepted, emit one bonus token sampled from p_{K+1}

Greedy decoding is the one-hot special case and uses the same code path.
"""

from __future__ import annotations

import torch


def _sample_probs(probs: torch.Tensor, generator: torch.Generator | None) -> int:
    return int(torch.multinomial(probs, 1, generator=generator).item())


def rejection_sample(
    draft_tokens: torch.Tensor,  # (K,) long
    q: torch.Tensor,             # (K, V) draft probabilities at each position
    p: torch.Tensor,             # (K, V) target probabilities at each position
    p_bonus: torch.Tensor,       # (V,) target probabilities for the bonus token
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """Return (output_tokens, num_draft_accepted).

    output_tokens has length num_accepted + 1: the accepted draft tokens followed by
    one corrected token (on rejection) or the bonus token (if all accepted).
    """
    K = int(draft_tokens.shape[0])
    out: list[int] = []
    for i in range(K):
        t = int(draft_tokens[i].item())
        qi = q[i, t].item()
        pi = p[i, t].item()
        ratio = 1.0 if qi <= 0 else min(1.0, pi / qi)
        r = torch.rand((), generator=generator, device=q.device).item()
        if r < ratio:
            out.append(t)  # accept
        else:
            resid = torch.clamp(p[i] - q[i], min=0)
            s = resid.sum()
            resid = resid / s if s > 0 else p[i]
            out.append(_sample_probs(resid, generator))
            return out, i  # i tokens accepted before the correction
    out.append(_sample_probs(p_bonus, generator))
    return out, K
