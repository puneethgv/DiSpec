"""Paged KV block allocator (logic only, no tensors).

Mirrors the vLLM PagedAttention idea: the KV cache is a pool of fixed-size blocks;
each sequence owns a *block table* (ordered list of physical block ids). This
decouples logical sequence length from physical layout, which is what enables
continuous batching of variable-length sequences without padding waste, and
copy-on-write forking for tree speculation.

This module is deliberately tensor-free so it can be unit-tested on CPU and reused
by every worker (prefill / decode / draft) regardless of the storage backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OutOfBlocksError(RuntimeError):
    """Raised when the block pool is exhausted (the scheduler must preempt)."""


@dataclass
class BlockTable:
    """Physical blocks owned by one sequence, plus its logical length."""

    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0  # logical tokens currently stored

    def slot(self, block_size: int, token_pos: int) -> tuple[int, int]:
        """Map a logical token position to (physical_block_id, offset_in_block)."""
        return self.block_ids[token_pos // block_size], token_pos % block_size


class BlockManager:
    """Allocates/frees physical blocks and maintains per-sequence block tables."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._ref_count: dict[int, int] = {}  # for copy-on-write sharing
        self._tables: dict[int, BlockTable] = {}

    # -- introspection -------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def has_table(self, seq_id: int) -> bool:
        return seq_id in self._tables

    def block_table(self, seq_id: int) -> BlockTable:
        return self._tables[seq_id]

    # -- allocation ----------------------------------------------------------
    def _alloc_block(self) -> int:
        if not self._free:
            raise OutOfBlocksError("block pool exhausted")
        b = self._free.pop()
        self._ref_count[b] = 1
        return b

    def allocate(self, seq_id: int, num_tokens: int) -> BlockTable:
        """Create a sequence and reserve enough blocks for `num_tokens` (prefill)."""
        if seq_id in self._tables:
            raise ValueError(f"seq {seq_id} already allocated")
        n = self.blocks_needed(num_tokens)
        if n > self.num_free_blocks:
            raise OutOfBlocksError(f"need {n} blocks, have {self.num_free_blocks}")
        table = BlockTable(block_ids=[self._alloc_block() for _ in range(n)], num_tokens=num_tokens)
        self._tables[seq_id] = table
        return table

    def append_token(self, seq_id: int) -> tuple[int, int]:
        """Reserve space for one more token (decode step). Returns its slot."""
        table = self._tables[seq_id]
        # Need a fresh block when the last one is full (or none exist yet).
        if table.num_tokens % self.block_size == 0:
            table.block_ids.append(self._alloc_block())
        slot = table.slot(self.block_size, table.num_tokens)
        table.num_tokens += 1
        return slot

    def reserve(self, seq_id: int, num_new_tokens: int) -> None:
        """Ensure capacity for `num_new_tokens` future tokens (e.g. a spec tree)."""
        table = self._tables[seq_id]
        have = len(table.block_ids) * self.block_size - table.num_tokens
        if num_new_tokens > have:
            extra = self.blocks_needed(num_new_tokens - have)
            if extra > self.num_free_blocks:
                raise OutOfBlocksError(f"reserve needs {extra}, have {self.num_free_blocks}")
            table.block_ids.extend(self._alloc_block() for _ in range(extra))

    # -- copy-on-write fork (tree speculation / beam) ------------------------
    def fork(self, parent_id: int, child_id: int) -> BlockTable:
        """Share parent's blocks with a child (COW). Shared blocks bump refcount."""
        parent = self._tables[parent_id]
        for b in parent.block_ids:
            self._ref_count[b] += 1
        child = BlockTable(block_ids=list(parent.block_ids), num_tokens=parent.num_tokens)
        self._tables[child_id] = child
        return child

    def truncate(self, seq_id: int, num_tokens: int) -> None:
        """Roll a sequence back to `num_tokens` tokens, freeing trailing blocks.

        Used by speculative decoding to discard the KV of rejected draft tokens.
        """
        table = self._tables[seq_id]
        keep = self.blocks_needed(num_tokens)
        for b in table.block_ids[keep:]:
            self._ref_count[b] -= 1
            if self._ref_count[b] == 0:
                del self._ref_count[b]
                self._free.append(b)
        table.block_ids = table.block_ids[:keep]
        table.num_tokens = num_tokens

    def free(self, seq_id: int) -> None:
        """Release a sequence; physical blocks return to the pool at refcount 0."""
        table = self._tables.pop(seq_id, None)
        if table is None:
            return
        for b in table.block_ids:
            self._ref_count[b] -= 1
            if self._ref_count[b] == 0:
                del self._ref_count[b]
                self._free.append(b)
