"""Cross-model KV transfer: turn one model's KV cache into another model's.

A small and a large model from the same family can share a prompt's prefill. The
small model prefills; a per-layer linear map, fitted offline, converts its KV cache
into the large model's layout; the large model imports it and decodes without ever
prefilling the prompt itself. The map costs a matmul per layer, which is far less
than a prefill once the prompt is more than a few hundred tokens.

The maps are fitted by kvxfer (github.com/puneethgv/kvxfer) from one calibration
pass. This module reads its files and applies them; it does not depend on kvxfer.
The apply step is small enough to own, and keeping calibration code (datasets,
sufficient statistics, solvers) out of the engine matches how the rest of DiSpec is
built.

How a map is applied, per target layer:

  1. Strip RoPE from the source keys. Cached keys entangle content with absolute
     position; the map was fitted on position-free *content-space* keys, which is
     what lets a map fitted on 1k-token contexts serve 8k-token ones. Values carry
     no rotation and pass through.
  2. Concatenate the k source layers the map selected into one design row per token,
     ordered layer, then KV head, then head dim -- the order kvxfer fitted against.
     Getting this order wrong does not raise; it produces a quietly worse cache.
  3. y = x @ W + b, plus an optional trained residual up(gelu(down(x))).
  4. Re-apply RoPE at the target's positions, in the target's own rotary tables.

A pair is only mappable if both models have identical KV heads and head dim (so a
source head lands on a target head without reshaping) and share a tokenizer. Layer
counts may differ -- the map chooses which source layers feed each target layer.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

KINDS = ("keys", "values")


def head_dim_of(config) -> int:
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


# -- RoPE in and out of content space -----------------------------------------
def rope_tables(rotary_emb, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) of shape (T, head_dim), in float32, from a model's own rotary module.

    Float32 even for a bf16 model: a strip-and-restore round trip in bf16 loses about
    three significant digits, which lands directly in the mapped keys.
    """
    probe = torch.zeros(1, 1, dtype=torch.float32, device=positions.device)
    cos, sin = rotary_emb(probe, positions[None, :])
    return cos[0].float(), sin[0].float()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def strip_rope(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotated keys (L, T, H, D) -> content-space keys, float32.

    The rotation is orthogonal, so its inverse is the same rotation with sin negated.
    """
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    x = keys.float()
    return x * cos - _rotate_half(x) * sin


def apply_rope(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Content-space keys (L, T, H, D) -> rotated keys, float32."""
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    x = keys.float()
    return x * cos + _rotate_half(x) * sin


# -- the map ------------------------------------------------------------------
@dataclass
class LayerMap:
    """The affine map producing one target layer's keys or values.

    Attributes:
        source_layers: source layers concatenated into the design, in order.
        bias: (kv_dim,).
        weight: (k * kv_dim, kv_dim), row-vector convention (x @ weight). None when
            the map is stored as low-rank factors instead.
        factors: (left, right) with x @ left @ right == x @ weight.
        residual: (down.weight, down.bias, up.weight) of a trained correction
            up(gelu(down(x))), or None.
    """

    source_layers: tuple[int, ...]
    bias: torch.Tensor
    weight: torch.Tensor | None = None
    factors: tuple[torch.Tensor, torch.Tensor] | None = None
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = field(default=None)

    def apply(self, x: torch.Tensor, use_residual: bool = True) -> torch.Tensor:
        if self.factors is not None:
            y = (x @ self.factors[0]) @ self.factors[1] + self.bias
        else:
            y = x @ self.weight + self.bias
        if use_residual and self.residual is not None:
            down_w, down_b, up_w = self.residual
            y = y + F.linear(F.gelu(F.linear(x, down_w, down_b)), up_w)
        return y

    def to(self, device, dtype) -> "LayerMap":
        move = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
        self.bias = move(self.bias)
        if self.weight is not None:
            self.weight = move(self.weight)
        if self.factors is not None:
            self.factors = tuple(move(t) for t in self.factors)
        if self.residual is not None:
            self.residual = tuple(move(t) for t in self.residual)
        return self


class CrossModelKVMap:
    """Maps a source model's contiguous KV into a target model's, in DiSpec's layout.

    KV tensors are (num_layers, seq_len, num_kv_heads, head_dim), the shape
    `PagedKVCache.export_contiguous` produces and `import_contiguous` consumes.
    """

    def __init__(self, keys: list[LayerMap], values: list[LayerMap], source_config,
                 target_config, metadata: dict | None = None):
        self.keys = keys
        self.values = values
        self.metadata = metadata or {}
        self.num_kv_heads = target_config.num_key_value_heads
        self.head_dim = head_dim_of(target_config)
        self.dtype = keys[0].bias.dtype if keys else torch.float32
        self._check(source_config, target_config)

    # -- validation ----------------------------------------------------------
    def _check(self, source_config, target_config) -> None:
        src, tgt = source_config, target_config
        for attr, a, b in (
            ("num_key_value_heads", src.num_key_value_heads, tgt.num_key_value_heads),
            ("head_dim", head_dim_of(src), head_dim_of(tgt)),
            ("vocab_size", src.vocab_size, tgt.vocab_size),
        ):
            if a != b:
                reason = ("the models do not share a tokenizer" if attr == "vocab_size"
                          else "a source head cannot map onto a target head")
                raise ValueError(f"cannot transfer KV: {attr} differs ({a} vs {b}); {reason}")

        n_src, n_tgt = src.num_hidden_layers, tgt.num_hidden_layers
        kv_dim = self.num_kv_heads * self.head_dim
        for kind, maps in (("keys", self.keys), ("values", self.values)):
            if len(maps) != n_tgt:
                raise ValueError(f"{kind}: {len(maps)} layer maps for a {n_tgt}-layer target")
            for layer, m in enumerate(maps):
                bad = [s for s in m.source_layers if not 0 <= s < n_src]
                if bad:
                    raise ValueError(f"{kind} map for target layer {layer} reads source "
                                     f"layers {bad}, but the source has {n_src}")
                design_dim = len(m.source_layers) * kv_dim
                shape = (m.weight.shape if m.weight is not None
                         else (m.factors[0].shape[0], m.factors[1].shape[1]))
                if tuple(shape) != (design_dim, kv_dim) or tuple(m.bias.shape) != (kv_dim,):
                    raise ValueError(f"{kind} map for target layer {layer} has shape "
                                     f"{tuple(shape)}, expected {(design_dim, kv_dim)}")
                if m.residual is not None:
                    down_w, _, up_w = m.residual
                    if down_w.shape[1] != design_dim or up_w.shape[0] != kv_dim:
                        raise ValueError(f"{kind} residual for target layer {layer} does "
                                         f"not match a {design_dim}->{kv_dim} map")

        for side, config in (("source", src), ("target", tgt)):
            recorded = self.metadata.get(side)
            loaded = getattr(config, "_name_or_path", "") or getattr(config, "name_or_path", "")
            if recorded and loaded and recorded.rstrip("/").split("/")[-1] != loaded.rstrip("/").split("/")[-1]:
                warnings.warn(f"maps were fitted with {side} {recorded!r} but {loaded!r} is "
                              "loaded; geometry matches, quality is not guaranteed", stacklevel=3)

    # -- loading ---------------------------------------------------------------
    @classmethod
    def load(cls, maps_dir, source_config, target_config, residual_dir=None,
             variant: str = "ridge", device="cuda",
             dtype: torch.dtype = torch.bfloat16) -> "CrossModelKVMap":
        """Load maps written by kvxfer's `save_maps`, optionally with a trained residual.

        Everything is placed on `device` in `dtype` once, here. Leaving maps on the host
        and moving them per call costs more than the prefill being skipped.
        """
        maps_dir = Path(maps_dir)
        manifest = maps_dir / "maps.json"
        if not manifest.exists():
            raise FileNotFoundError(f"no maps.json in {maps_dir}")
        metadata = json.loads(manifest.read_text())
        if variant not in metadata.get("variants", []):
            raise ValueError(f"variant {variant!r} not in {metadata.get('variants')}")

        per_kind = {}
        for kind in KINDS:
            state = torch.load(maps_dir / f"{variant}__{kind}.pt", map_location="cpu",
                               weights_only=True)
            layers = []
            for layer in range(len(state)):
                s = state[layer]
                factors = s.get("factors")
                layers.append(LayerMap(
                    source_layers=tuple(int(i) for i in s["source_layers"]),
                    bias=s["bias"],
                    weight=s.get("weight"),
                    factors=tuple(factors) if factors is not None else None,
                ))
            per_kind[kind] = layers

        if residual_dir is not None:
            res = torch.load(Path(residual_dir) / "residual.pt", map_location="cpu",
                             weights_only=True)
            for kind, key in (("keys", "key_residual"), ("values", "value_residual")):
                sd = res[key]
                for layer, m in enumerate(per_kind[kind]):
                    m.residual = (sd[f"{layer}.down.weight"], sd[f"{layer}.down.bias"],
                                  sd[f"{layer}.up.weight"])

        kv_map = cls(per_kind["keys"], per_kind["values"], source_config, target_config,
                     metadata)
        return kv_map.to(device, dtype)

    def to(self, device, dtype: torch.dtype) -> "CrossModelKVMap":
        for m in (*self.keys, *self.values):
            m.to(device, dtype)
        self.dtype = dtype
        return self

    @property
    def has_residual(self) -> bool:
        return self.keys[0].residual is not None

    # -- apply -----------------------------------------------------------------
    def _apply(self, source: torch.Tensor, maps: list[LayerMap], use_residual: bool,
               rope: tuple[torch.Tensor, torch.Tensor] | None = None):
        seq = source.shape[1]
        out = torch.empty(len(maps), seq, self.num_kv_heads, self.head_dim,
                          dtype=self.dtype, device=source.device)
        for layer, m in enumerate(maps):
            x = source[list(m.source_layers)].permute(1, 0, 2, 3).reshape(seq, -1)
            y = m.apply(x, use_residual).view(1, seq, self.num_kv_heads, self.head_dim)
            if rope is not None:
                y = apply_rope(y, *rope)  # float32, one layer at a time
            out[layer] = y[0]
        return out

    @torch.inference_mode()
    def map(self, k: torch.Tensor, v: torch.Tensor,
            source_rope: tuple[torch.Tensor, torch.Tensor],
            target_rope: tuple[torch.Tensor, torch.Tensor],
            use_residual: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Map contiguous source KV to target KV.

        Args:
            k, v: source KV, (source_layers, T, H, D), keys rotated as cached.
            source_rope: (cos, sin) for the T positions, from the source model.
            target_rope: (cos, sin) for the same positions, from the target model.
            use_residual: apply the trained residual if one was loaded.

        Returns:
            (k, v) for the target, (target_layers, T, H, D), keys rotated, in the
            input dtype.
        """
        # RoPE is stripped and re-applied in float32, but one layer at a time. Doing it
        # on the whole cache holds a float32 copy plus two same-sized intermediates --
        # about 3 GB for 8k tokens of a 28-layer model -- and ran an L4 out of memory
        # at 8k tokens with both models resident.
        content = torch.empty(k.shape, dtype=self.dtype, device=k.device)
        for layer in range(k.shape[0]):
            content[layer] = strip_rope(k[layer:layer + 1], *source_rope)[0]
        keys = self._apply(content, self.keys, use_residual, rope=target_rope)
        del content
        values = self._apply(v.to(self.dtype), self.values, use_residual)
        return keys.to(k.dtype), values.to(v.dtype)
