"""Paged KV cache tensor pool (GPU storage backend).

Physical layout per cache (separate K and V tensors):
    [num_layers, num_blocks * block_size, num_kv_heads, head_dim]

A "slot" is a flat index `block_id * block_size + offset` into the second dim.
The `BlockManager` hands out block ids and slots; this class just reads/writes
the underlying tensors. Keeping storage and allocation separate means the same
pool can later be exported (CUDA IPC handle) to a different process for KV
transfer in the disaggregated setting.
"""

from __future__ import annotations

import torch

from dispec.kv.block_manager import BlockManager


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self.key = torch.zeros(shape, dtype=dtype, device=device)
        self.value = torch.zeros(shape, dtype=dtype, device=device)
        self.manager = BlockManager(num_blocks, block_size)

    @classmethod
    def for_model(cls, config, num_blocks: int, block_size: int, dtype, device="cuda") -> "PagedKVCache":
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return cls(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def memory_bytes(self) -> int:
        return self.key.numel() * self.key.element_size() * 2

    # -- per-step IO ---------------------------------------------------------
    def write(self, layer_idx: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter token KV into the pool.

        slots: (num_tokens,) long, flat slot indices.
        k, v : (num_tokens, num_kv_heads, head_dim).
        """
        self.key[layer_idx].index_copy_(0, slots, k)
        self.value[layer_idx].index_copy_(0, slots, v)

    def gather(self, layer_idx: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather KV for the given flat slots: returns (k, v) of (len(slots), H, D)."""
        return self.key[layer_idx][slots], self.value[layer_idx][slots]

    # -- slot helpers --------------------------------------------------------
    def slots_for_positions(self, block_ids: list[int], positions: torch.Tensor) -> torch.Tensor:
        """Flat slots for token `positions` of a sequence with `block_ids`."""
        bt = torch.tensor(block_ids, device=positions.device, dtype=torch.long)
        block_idx = bt[positions // self.block_size]
        return block_idx * self.block_size + (positions % self.block_size)

    def context_slots(self, block_ids: list[int], num_tokens: int) -> torch.Tensor:
        """Flat slots for the first `num_tokens` tokens of a sequence (its context)."""
        pos = torch.arange(num_tokens, device=self.device)
        return self.slots_for_positions(block_ids, pos)

    # -- disaggregation: export/import a sequence's KV as portable contiguous tensors --
    def export_contiguous(self, block_ids: list[int], seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a sequence's paged KV into contiguous (num_layers, seq_len, H, D) tensors."""
        slots = self.context_slots(block_ids, seq_len)
        return self.key[:, slots].contiguous(), self.value[:, slots].contiguous()

    def import_contiguous(self, block_ids: list[int], k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter contiguous KV (num_layers, seq_len, H, D) into the paged blocks."""
        slots = self.context_slots(block_ids, k.shape[1])
        self.key[:, slots] = k.to(self.key.dtype)
        self.value[:, slots] = v.to(self.value.dtype)
