"""Automatic prefix caching: reuse KV blocks across requests with shared prefixes.

Many requests share a leading prefix (a system prompt, a few-shot preamble, a chat
history). Re-running prefill over that shared text every time is wasted compute. This
cache keys each *full* KV block by a rolling hash of all tokens up to and including that
block, so two prompts with the same prefix resolve to the same physical blocks. A new
request adopts the longest matching run of cached blocks (refcount bumped) and only
prefills the remaining tokens.

Cached blocks are pinned (the cache holds one reference each) so they survive after the
producing sequence finishes; under memory pressure the least-recently-used entries are
evicted, releasing their reference back to the block pool.
"""

from __future__ import annotations

from dispec.kv.block_manager import BlockManager


class PrefixCache:
    def __init__(self, manager: BlockManager, block_size: int):
        self.mgr = manager
        self.bs = block_size
        self._nodes: dict[int, list] = {}  # rolling_hash -> [block_id, last_used]
        self._clock = 0
        self.hits = 0  # blocks served from cache (telemetry)
        self.misses = 0

    def _block_hashes(self, token_ids: list[int]) -> list[int]:
        """Rolling hash for each full block: h_i = hash((h_{i-1}, block_i_tokens))."""
        h = 0
        out = []
        for i in range(len(token_ids) // self.bs):
            chunk = tuple(token_ids[i * self.bs:(i + 1) * self.bs])
            h = hash((h, chunk))
            out.append(h)
        return out

    def match(self, token_ids: list[int]) -> list[int]:
        """Longest run of cached physical blocks for this prompt's leading full blocks."""
        self._clock += 1
        matched: list[int] = []
        for key in self._block_hashes(token_ids):
            node = self._nodes.get(key)
            if node is None:
                break
            node[1] = self._clock  # touch (LRU)
            matched.append(node[0])
        self.hits += len(matched)
        return matched

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None:
        """Register a sequence's full blocks so later requests can reuse them."""
        self._clock += 1
        for key, blk in zip(self._block_hashes(token_ids), block_ids):
            node = self._nodes.get(key)
            if node is None:
                self.mgr.add_ref(blk)  # pin
                self._nodes[key] = [blk, self._clock]
                self.misses += 1
            else:
                node[1] = self._clock

    def evict(self, num_blocks: int) -> int:
        """Release up to `num_blocks` least-recently-used cached blocks. Returns count."""
        victims = sorted(self._nodes.items(), key=lambda kv: kv[1][1])[:num_blocks]
        for key, (blk, _) in victims:
            del self._nodes[key]
            self.mgr.release(blk)
        return len(victims)

    def __len__(self) -> int:
        return len(self._nodes)
