"""Rejection sampling must be lossless: outputs follow the TARGET distribution.

These run on CPU with synthetic distributions — the core correctness guarantee of
speculative decoding, independent of any model.
"""

import torch

from dispec.spec.rejection import rejection_sample


def _rand_dist(v, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(v, generator=g), dim=-1)


def test_first_token_follows_target_regardless_of_draft():
    """Empirical distribution of the emitted token must match p, not q."""
    V = 6
    p1 = _rand_dist(V, 1)
    q1 = _rand_dist(V, 2)  # deliberately different draft distribution
    p_bonus = _rand_dist(V, 3)

    gen = torch.Generator().manual_seed(0)
    N = 60_000
    counts = torch.zeros(V)
    for _ in range(N):
        t = torch.multinomial(q1, 1, generator=gen)  # draft token ~ q
        out, _ = rejection_sample(t, q1[None, :], p1[None, :], p_bonus, gen)
        counts[out[0]] += 1
    emp = counts / N
    assert torch.allclose(emp, p1, atol=0.01), f"\nempirical={emp}\ntarget={p1}"


def test_all_accept_when_draft_equals_target():
    """If q == p, every draft token is accepted and a bonus is appended."""
    V = 10
    p = _rand_dist(V, 5).repeat(4, 1)  # K=4 identical positions
    gen = torch.Generator().manual_seed(7)
    accepts = 0
    trials = 2000
    for _ in range(trials):
        draft = torch.multinomial(p, 1, generator=gen).squeeze(1)  # (4,)
        out, n_acc = rejection_sample(draft, p, p, p[0], gen)
        accepts += n_acc
        assert len(out) == n_acc + 1
    # With q==p the acceptance probability is 1 everywhere => all 4 accepted.
    assert accepts == 4 * trials


def test_greedy_one_hot_path():
    """Greedy (one-hot) distributions: accept iff draft == target argmax."""
    V = 5
    p = torch.zeros(1, V); p[0, 3] = 1.0  # target greedy token = 3
    # draft proposes its own argmax 3 (matches) -> accepted, bonus appended
    q = torch.zeros(1, V); q[0, 3] = 1.0
    out, n = rejection_sample(torch.tensor([3]), q, p, p[0], torch.Generator().manual_seed(0))
    assert n == 1 and out[0] == 3
    # draft argmax is 1 (mismatch) -> rejected, corrected to target argmax 3
    q = torch.zeros(1, V); q[0, 1] = 1.0
    out, n = rejection_sample(torch.tensor([1]), q, p, p[0], torch.Generator().manual_seed(0))
    assert n == 0 and out[0] == 3
