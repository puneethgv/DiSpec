"""Unit tests for the paged KV block allocator (CPU-only, no model needed)."""

import pytest

from dispec.kv.block_manager import BlockManager, OutOfBlocksError


def test_allocate_and_slots():
    bm = BlockManager(num_blocks=10, block_size=4)
    table = bm.allocate(seq_id=1, num_tokens=6)  # needs ceil(6/4)=2 blocks
    assert len(table.block_ids) == 2
    assert bm.num_free_blocks == 8
    # token 0 -> (block0, 0); token 5 -> (block1, 1)
    assert table.slot(4, 0) == (table.block_ids[0], 0)
    assert table.slot(4, 5) == (table.block_ids[1], 1)


def test_append_grows_blocks_only_when_full():
    bm = BlockManager(num_blocks=10, block_size=2)
    bm.allocate(seq_id=1, num_tokens=2)  # exactly 1 block, full
    assert len(bm.block_table(1).block_ids) == 1
    blk, off = bm.append_token(1)  # token idx 2 -> needs new block
    assert off == 0
    assert len(bm.block_table(1).block_ids) == 2
    bm.append_token(1)  # token idx 3 -> same block, offset 1
    assert len(bm.block_table(1).block_ids) == 2
    assert bm.block_table(1).num_tokens == 4


def test_free_returns_blocks():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate(seq_id=1, num_tokens=16)  # all 4 blocks
    assert bm.num_free_blocks == 0
    with pytest.raises(OutOfBlocksError):
        bm.allocate(seq_id=2, num_tokens=1)
    bm.free(1)
    assert bm.num_free_blocks == 4
    bm.allocate(seq_id=2, num_tokens=1)  # now fits


def test_fork_is_copy_on_write_refcounted():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate(seq_id=1, num_tokens=8)  # 2 blocks
    assert bm.num_free_blocks == 2
    bm.fork(parent_id=1, child_id=2)  # shares the same 2 blocks, no new alloc
    assert bm.num_free_blocks == 2
    bm.free(1)  # parent gone but blocks still referenced by child
    assert bm.num_free_blocks == 2
    bm.free(2)  # last reference released
    assert bm.num_free_blocks == 4


def test_reserve_for_tree():
    bm = BlockManager(num_blocks=10, block_size=4)
    bm.allocate(seq_id=1, num_tokens=4)  # 1 full block
    bm.reserve(seq_id=1, num_new_tokens=5)  # need 5 more => 2 extra blocks
    assert len(bm.block_table(1).block_ids) == 3
