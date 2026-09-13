"""Cross-model KV transfer: the map files, the apply math, and admitting transferred KV.

The failure modes worth guarding are silent. A map applied with its input features in
the wrong order, or keys re-rotated with the wrong tables, yields a correctly shaped
cache that is merely worse -- nothing raises. So:

  * CPU tests pin the apply math by hand and read a synthetic map directory written
    the way kvxfer writes one.
  * Parity tests (when kvxfer is importable) push identical tensors through DiSpec and
    kvxfer and require identical output.
  * Identity-injection gates: a model's *own* KV admitted through `prefix_kv` -- raw,
    and through an identity map -- must reproduce a plain prefill. If this fails, every
    transferred-cache number is off by an unknown amount.
  * With real weights and the trained Qwen3-1.7B -> 4B maps ($DISPEC_KVXFER_MAPS), the
    mapped cache must beat a zeroed one: the transfer carries signal.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.kv.transfer import (CrossModelKVMap, LayerMap, apply_rope, head_dim_of,
                                rope_tables, strip_rope)
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import ContinuousBatchingEngine
from dispec.workers.escalation import escalate, map_prefix, prefill_and_map
from tests.tiny_models import tiny_qwen3

H, D = 2, 4
KV_DIM = H * D


def _config(n_layers, vocab=100, heads=H, head_dim=D, name="org/model"):
    return SimpleNamespace(num_hidden_layers=n_layers, num_key_value_heads=heads,
                           head_dim=head_dim, hidden_size=16, num_attention_heads=4,
                           vocab_size=vocab, _name_or_path=name)


def _no_rope(n):
    return torch.ones(n, D), torch.zeros(n, D)


def _write_maps(path: Path, n_src, n_tgt, k=2, seed=0, source="org/src", target="org/tgt"):
    """A map directory in kvxfer's `save_maps` format."""
    g = torch.Generator().manual_seed(seed)
    for kind in ("keys", "values"):
        state = {}
        for t in range(n_tgt):
            layers = sorted(torch.randperm(n_src, generator=g)[:k].tolist())
            state[t] = {"weight": torch.randn(k * KV_DIM, KV_DIM, generator=g),
                        "bias": torch.randn(KV_DIM, generator=g), "source_layers": layers,
                        "target_layer": t, "r2": 0.5, "rank": None, "factors": None}
        torch.save(state, path / f"ridge__{kind}.pt")
    (path / "maps.json").write_text(json.dumps(
        {"source": source, "target": target, "variants": ["ridge"]}))
    return path


def _load(path, n_src, n_tgt, **kw):
    return CrossModelKVMap.load(path, _config(n_src, name="org/src"),
                                _config(n_tgt, name="org/tgt"), device="cpu",
                                dtype=torch.float32, **kw)


def _identity_map(config, device, dtype):
    kv_dim = config.num_key_value_heads * head_dim_of(config)
    maps = lambda: [LayerMap((i,), torch.zeros(kv_dim), weight=torch.eye(kv_dim))  # noqa: E731
                    for i in range(config.num_hidden_layers)]
    return CrossModelKVMap(maps(), maps(), config, config).to(device, dtype)


# -- file format and apply math ---------------------------------------------------
def test_load_reads_kvxfer_map_format(tmp_path):
    _write_maps(tmp_path, n_src=3, n_tgt=4)
    kv_map = _load(tmp_path, 3, 4)
    saved = torch.load(tmp_path / "ridge__keys.pt", weights_only=True)
    assert len(kv_map.keys) == len(kv_map.values) == 4
    for t, m in enumerate(kv_map.keys):
        assert m.source_layers == tuple(saved[t]["source_layers"])
        torch.testing.assert_close(m.weight, saved[t]["weight"])


def test_apply_uses_layer_head_dim_feature_order():
    """Row per token = concat over source layers (in map order) of heads then dims."""
    g = torch.Generator().manual_seed(3)
    src = torch.randn(3, 5, H, D, generator=g)
    m = LayerMap((2, 0), torch.randn(KV_DIM, generator=g),
                 weight=torch.randn(2 * KV_DIM, KV_DIM, generator=g))
    kv_map = CrossModelKVMap([m], [m], _config(3), _config(1))
    keys, _ = kv_map.map(src, src, _no_rope(5), _no_rope(5))
    for t in range(5):
        x = torch.cat([src[2, t].reshape(-1), src[0, t].reshape(-1)])
        expected = (x @ m.weight + m.bias).view(H, D)
        torch.testing.assert_close(keys[0, t], expected)


def test_factored_map_equals_dense():
    g = torch.Generator().manual_seed(4)
    left, right = torch.randn(2 * KV_DIM, 3, generator=g), torch.randn(3, KV_DIM, generator=g)
    bias, x = torch.randn(KV_DIM, generator=g), torch.randn(6, 2 * KV_DIM, generator=g)
    dense = LayerMap((0, 1), bias, weight=left @ right)
    factored = LayerMap((0, 1), bias, factors=(left, right))
    torch.testing.assert_close(factored.apply(x), dense.apply(x))


def test_residual_is_added_as_up_gelu_down():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(6, 2 * KV_DIM, generator=g)
    w, b = torch.randn(2 * KV_DIM, KV_DIM, generator=g), torch.randn(KV_DIM, generator=g)
    down_w, down_b = torch.randn(3, 2 * KV_DIM, generator=g), torch.randn(3, generator=g)
    up_w = torch.randn(KV_DIM, 3, generator=g)
    m = LayerMap((0, 1), b, weight=w, residual=(down_w, down_b, up_w))
    hidden = x @ down_w.T + down_b
    expected = x @ w + b + torch.nn.functional.gelu(hidden) @ up_w.T
    torch.testing.assert_close(m.apply(x), expected)
    torch.testing.assert_close(m.apply(x, use_residual=False), x @ w + b)

    # kvxfer initialises `up` to zeros: an untrained residual must change nothing.
    m.residual = (down_w, down_b, torch.zeros_like(up_w))
    torch.testing.assert_close(m.apply(x), x @ w + b)


def test_rope_strip_inverts_apply():
    model = tiny_qwen3()
    pos = torch.arange(37)
    cos, sin = rope_tables(model.model.rotary_emb, pos)
    content = torch.randn(2, 37, 2, 16)
    torch.testing.assert_close(strip_rope(apply_rope(content, cos, sin), cos, sin), content,
                               atol=1e-5, rtol=1e-5)


def test_residual_loads_from_kvxfer_format(tmp_path):
    _write_maps(tmp_path, n_src=3, n_tgt=2)
    hidden, design = 5, 2 * KV_DIM
    sd = lambda: {f"{i}.{name}": torch.randn(shape)  # noqa: E731
                  for i in range(2) for name, shape in
                  (("down.weight", (hidden, design)), ("down.bias", (hidden,)),
                   ("up.weight", (KV_DIM, hidden)))}
    res = tmp_path / "residual"
    res.mkdir()
    state = {"key_residual": sd(), "value_residual": sd(), "hidden": hidden, "design_dim": design}
    torch.save(state, res / "residual.pt")
    kv_map = _load(tmp_path, 3, 2, residual_dir=res)
    assert kv_map.has_residual
    torch.testing.assert_close(kv_map.values[1].residual[2], state["value_residual"]["1.up.weight"])


# -- validation -----------------------------------------------------------------------
def test_rejects_models_without_a_shared_tokenizer(tmp_path):
    _write_maps(tmp_path, 3, 2)
    with pytest.raises(ValueError, match="tokenizer"):
        CrossModelKVMap.load(tmp_path, _config(3), _config(2, vocab=101), device="cpu")


def test_rejects_mismatched_kv_geometry(tmp_path):
    _write_maps(tmp_path, 3, 2)
    with pytest.raises(ValueError, match="head_dim"):
        CrossModelKVMap.load(tmp_path, _config(3, head_dim=8), _config(2), device="cpu")
    with pytest.raises(ValueError, match="num_key_value_heads"):
        CrossModelKVMap.load(tmp_path, _config(3, heads=4), _config(2), device="cpu")


def test_rejects_maps_that_do_not_fit_the_models(tmp_path):
    _write_maps(tmp_path, n_src=3, n_tgt=2)
    with pytest.raises(ValueError, match="2 layer maps for a 3-layer target"):
        CrossModelKVMap.load(tmp_path, _config(3), _config(3), device="cpu")
    with pytest.raises(ValueError, match="source has 1"):
        CrossModelKVMap.load(tmp_path, _config(1), _config(2), device="cpu")


def test_warns_when_maps_were_fitted_for_other_models(tmp_path):
    _write_maps(tmp_path, 3, 2, source="org/other")
    with pytest.warns(UserWarning, match="fitted with source"):
        CrossModelKVMap.load(tmp_path, _config(3, name="org/src"), _config(2, name="org/tgt"),
                             device="cpu")


# -- admitting transferred KV -----------------------------------------------------------
def test_add_request_validates_prefix_kv():
    model = tiny_qwen3()
    eng = ContinuousBatchingEngine(model, num_blocks=32)
    n = 20
    shape = (model.config.num_hidden_layers, n, 2, 16)
    with pytest.raises(ValueError, match="fewer than all"):
        eng.add_request(list(range(n)), 4, prefix_kv=(torch.zeros(shape), torch.zeros(shape)))
    bad = (model.config.num_hidden_layers, n - 1, 2, 8)
    with pytest.raises(ValueError, match="cache layout"):
        eng.add_request(list(range(n)), 4, prefix_kv=(torch.zeros(bad), torch.zeros(bad)))


def _last_logits_with_prefix(runner, cache, ids, prefix_kv, sid=99):
    """Import KV for ids[:-1], forward only the last token, return its logits."""
    n = len(ids)
    dev = runner.device
    table = cache.manager.allocate(sid, n)
    cache.import_contiguous(table.block_ids, *prefix_kv)
    table.num_tokens = n
    slot = cache.context_slots(table.block_ids, n)[-1:]
    hidden = runner.forward(torch.tensor(ids[-1:], device=dev), torch.tensor([n - 1], device=dev),
                            slot, [SeqMeta(table.block_ids, 1, n)], cache)
    cache.manager.free(sid)
    return runner.logits(hidden[-1:]).float()[0]


def _plain_last_logits(runner, cache, ids, sid=98):
    n = len(ids)
    dev = runner.device
    table = cache.manager.allocate(sid, n)
    table.num_tokens = n
    hidden = runner.forward(torch.tensor(ids, device=dev), torch.arange(n, device=dev),
                            cache.context_slots(table.block_ids, n),
                            [SeqMeta(table.block_ids, n, n)], cache)
    cache.manager.free(sid)
    return runner.logits(hidden[-1:]).float()[0]


def _own_prefix(runner, cache, ids, sid=97):
    n = len(ids)
    dev = runner.device
    table = cache.manager.allocate(sid, n)
    table.num_tokens = n
    runner.forward(torch.tensor(ids, device=dev), torch.arange(n, device=dev),
                   cache.context_slots(table.block_ids, n), [SeqMeta(table.block_ids, n, n)],
                   cache)
    return table


def test_identity_injection_gate_tiny():
    """Own KV for n-1 tokens, raw and through an identity map, reproduces a prefill."""
    model = tiny_qwen3()
    runner = ModelRunner(model)
    cache = PagedKVCache.for_model(model.config, 32, 16, torch.float32, "cpu")
    ids = torch.randint(0, 128, (37,), generator=torch.Generator().manual_seed(6)).tolist()
    plain = _plain_last_logits(runner, cache, ids)

    table = _own_prefix(runner, cache, ids)
    raw = cache.export_contiguous(table.block_ids, len(ids) - 1)
    via_map = map_prefix(runner, cache, table.block_ids, len(ids),
                         _identity_map(model.config, "cpu", torch.float32), runner)
    cache.manager.free(97)

    torch.testing.assert_close(via_map[0], raw[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(via_map[1], raw[1])
    for prefix in (raw, via_map):
        torch.testing.assert_close(_last_logits_with_prefix(runner, cache, ids, prefix), plain,
                                   atol=1e-4, rtol=1e-4)


def test_scheduler_with_prefix_kv_reproduces_plain_generation_tiny():
    model = tiny_qwen3()
    ids = torch.randint(0, 128, (41,), generator=torch.Generator().manual_seed(7)).tolist()
    params = SamplingParams(0.0)

    plain = ContinuousBatchingEngine(model, num_blocks=32)
    rid = plain.add_request(ids, 12, params)
    plain.run_until_done()

    escalated = ContinuousBatchingEngine(model, num_blocks=32)
    prefix = prefill_and_map(escalated.runner, escalated.cache, 500, ids,
                             _identity_map(model.config, "cpu", torch.float32),
                             escalated.runner)
    rid2 = escalate(escalated, ids, prefix, 12, params)
    escalated.run_until_done()

    assert escalated.num_prefill_tokens == 1, "transferred tokens must not be prefilled"
    assert escalated.collect_outputs()[rid2] == plain.collect_outputs()[rid]


# -- parity with kvxfer ---------------------------------------------------------------
def _kvxfer_geometry(n_layers):
    from kvxfer.geometry import KVGeometry
    return KVGeometry(model_id="org/tgt", n_layers=n_layers, n_q_heads=4, n_kv_heads=H,
                      head_dim=D, hidden_size=16, rope_theta=1e6, vocab_size=100)


def test_parity_with_kvxfer_fitted_mapper(tmp_path):
    pytest.importorskip("kvxfer")
    from kvxfer.cache import ContentKV
    from kvxfer.mapstore import load_maps

    _write_maps(tmp_path, n_src=4, n_tgt=3, seed=8)
    theirs, _ = load_maps(tmp_path, _kvxfer_geometry(3))
    ours = _load(tmp_path, 4, 3)

    g = torch.Generator().manual_seed(9)
    keys, values = torch.randn(4, 1, H, 6, D, generator=g), torch.randn(4, 1, H, 6, D, generator=g)
    expected = theirs["ridge"].map(ContentKV(keys, values, torch.arange(6)[None]))
    to_ours = lambda t: t[:, 0].permute(0, 2, 1, 3)  # noqa: E731  (L,1,H,T,D) -> (L,T,H,D)
    got_k, got_v = ours.map(to_ours(keys), to_ours(values), _no_rope(6), _no_rope(6))
    torch.testing.assert_close(got_k, to_ours(expected.keys), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_v, to_ours(expected.values), atol=1e-5, rtol=1e-5)


def test_parity_with_kvxfer_trained_residual(tmp_path):
    pytest.importorskip("kvxfer")
    from kvxfer.cache import ContentKV
    from kvxfer.mapstore import load_maps
    from kvxfer.solvers.neural import ResidualMapper, load_residual

    _write_maps(tmp_path, n_src=4, n_tgt=3, seed=10)
    geom = _kvxfer_geometry(3)
    base = load_maps(tmp_path, geom)[0]["ridge"]
    trained = ResidualMapper(base.key_maps, base.value_maps, geom, 2 * KV_DIM, hidden=6)
    with torch.no_grad():
        for p in trained.parameters():
            p.normal_()  # kvxfer zero-inits `up`; make the residual do something
    res = tmp_path / "residual"
    res.mkdir()
    torch.save({"key_residual": trained.key_residual.state_dict(),
                "value_residual": trained.value_residual.state_dict(),
                "hidden": 6, "design_dim": 2 * KV_DIM}, res / "residual.pt")

    theirs = load_residual(res, base.key_maps, base.value_maps, geom)
    ours = _load(tmp_path, 4, 3, residual_dir=res)
    g = torch.Generator().manual_seed(11)
    keys, values = torch.randn(4, 1, H, 5, D, generator=g), torch.randn(4, 1, H, 5, D, generator=g)
    with torch.no_grad():
        expected = theirs.map(ContentKV(keys, values, torch.arange(5)[None]))
    to_ours = lambda t: t[:, 0].permute(0, 2, 1, 3)  # noqa: E731
    got_k, got_v = ours.map(to_ours(keys), to_ours(values), _no_rope(5), _no_rope(5))
    torch.testing.assert_close(got_k, to_ours(expected.keys), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got_v, to_ours(expected.values), atol=1e-4, rtol=1e-4)


def test_parity_with_kvxfer_rope_strip():
    pytest.importorskip("kvxfer")
    from kvxfer.rope import strip_keys

    model = tiny_qwen3()
    cos, sin = rope_tables(model.model.rotary_emb, torch.arange(9))
    keys = torch.randn(2, 9, 2, 16)  # (L, T, H, D)
    theirs = strip_keys(keys.permute(0, 2, 1, 3)[:, None], cos[None], sin[None])
    torch.testing.assert_close(strip_rope(keys, cos, sin), theirs[:, 0].permute(0, 2, 1, 3))


# -- CUDA: real weights ---------------------------------------------------------------
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
MAPS = os.environ.get("DISPEC_KVXFER_MAPS")
LONG_PROMPTS = [
    "The Antikythera mechanism is an ancient Greek hand-powered device that has been "
    "identified as the oldest known analogue computer. It was used to predict "
    "astronomical positions and eclipses decades in advance. It was recovered in 1901 "
    "from a shipwreck off the coast of the Greek island of",
    "Photosynthesis converts light energy into chemical energy. In the light-dependent "
    "reactions, water is split and oxygen is released, while ATP and NADPH are produced. "
    "Those carriers then power the Calvin cycle, which fixes carbon dioxide into",
]


@needs_cuda
def test_identity_injection_gate_qwen3_cuda():
    from dispec.models.loader import load_model, load_tokenizer

    model, tok = load_model("Qwen/Qwen3-0.6B"), load_tokenizer("Qwen/Qwen3-0.6B")
    runner = ModelRunner(model)
    cache = PagedKVCache.for_model(model.config, 64, 16, runner.dtype, "cuda")
    kv_map = _identity_map(model.config, "cuda", runner.dtype)
    for prompt in LONG_PROMPTS:
        ids = tok(prompt).input_ids
        plain = _plain_last_logits(runner, cache, ids)
        table = _own_prefix(runner, cache, ids)
        raw = cache.export_contiguous(table.block_ids, len(ids) - 1)
        mapped = map_prefix(runner, cache, table.block_ids, len(ids), kv_map, runner)
        cache.manager.free(97)
        for name, prefix in (("raw", raw), ("identity map", mapped)):
            got = _last_logits_with_prefix(runner, cache, ids, prefix)
            cos = torch.nn.functional.cosine_similarity(got, plain, dim=0).item()
            assert cos >= 0.999, f"{name}: cosine {cos:.5f}"
            assert int(got.argmax()) == int(plain.argmax()), name


@pytest.mark.skipif(not (torch.cuda.is_available() and MAPS),
                    reason="needs CUDA and $DISPEC_KVXFER_MAPS (kvxfer Qwen3-1.7B -> 4B maps)")
def test_trained_qwen3_map_carries_signal():
    """The mapped cache must predict the target's next token better than a zeroed cache."""
    from dispec.config import KVXFER_SOURCE_MODEL, KVXFER_TARGET_MODEL
    from dispec.models.loader import load_model, load_tokenizer

    src, tgt = load_model(KVXFER_SOURCE_MODEL), load_model(KVXFER_TARGET_MODEL)
    tok = load_tokenizer(KVXFER_TARGET_MODEL)
    src_runner, tgt_runner = ModelRunner(src), ModelRunner(tgt)
    src_cache = PagedKVCache.for_model(src.config, 64, 16, src_runner.dtype, "cuda")
    tgt_cache = PagedKVCache.for_model(tgt.config, 64, 16, tgt_runner.dtype, "cuda")
    kv_map = CrossModelKVMap.load(Path(MAPS) / "maps", src.config, tgt.config,
                                  residual_dir=Path(MAPS) / "residual", device="cuda",
                                  dtype=tgt_runner.dtype)

    def kl(target_logits, logits):
        p = torch.log_softmax(target_logits, -1)
        return torch.sum(p.exp() * (p - torch.log_softmax(logits, -1))).item()

    for prompt in LONG_PROMPTS:
        ids = tok(prompt).input_ids
        reference = _plain_last_logits(tgt_runner, tgt_cache, ids)
        for use_residual in (False, True):
            k, v = prefill_and_map(src_runner, src_cache, 600, ids, kv_map, tgt_runner,
                                   use_residual)
            mapped = _last_logits_with_prefix(tgt_runner, tgt_cache, ids, (k, v))
            zeroed = _last_logits_with_prefix(tgt_runner, tgt_cache, ids,
                                              (torch.zeros_like(k), torch.zeros_like(v)))
            assert kl(reference, mapped) < kl(reference, zeroed), (
                f"residual={use_residual}: mapped KL {kl(reference, mapped):.3f} not below "
                f"zeroed KL {kl(reference, zeroed):.3f}")

    engine = ContinuousBatchingEngine(tgt, num_blocks=64)
    ids = tok(LONG_PROMPTS[0]).input_ids
    prefix = prefill_and_map(src_runner, src_cache, 601, ids, kv_map, engine.runner)
    rid = escalate(engine, ids, prefix, 8, SamplingParams(0.0))
    engine.run_until_done()
    assert len(engine.collect_outputs()[rid]) == 8
    assert engine.num_prefill_tokens == 1
