"""Central configuration for DiSpec.

Model choices are sized for a single 8 GB GPU (RTX 3070): a 1.5B target leaves
~4-5 GB for the draft model, paged KV cache, and tree-verify activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Modern, strong, same-tokenizer-family draft/target pair (lossless spec decode
# requires identical vocab/tokenizer between draft and target).
TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DTYPE = "bfloat16"


@dataclass
class GenConfig:
    """Sampling / generation parameters."""

    max_new_tokens: int = 256
    temperature: float = 0.0  # 0.0 => greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 => disabled
    seed: int = 0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass
class KVConfig:
    """Paged KV cache parameters."""

    block_size: int = 16  # tokens per block
    max_blocks: int = 4096  # total blocks in the pool (sized at runtime to fit VRAM)


@dataclass
class SpecConfig:
    """Speculative decoding parameters."""

    num_speculative_tokens: int = 5  # K for sequential drafting
    # Tree drafting: branching factor per depth (len => tree depth).
    tree_widths: list[int] = field(default_factory=lambda: [4, 2, 2, 1, 1])


# Canonical prompt set for reproducible benchmarks.
BENCH_PROMPTS: list[str] = [
    "Explain how a transformer attention mechanism works, step by step.",
    "Write a short story about a lighthouse keeper who discovers a hidden door.",
    "What are the trade-offs between prefill/decode disaggregation and colocation?",
    "Summarize the causes of the French Revolution in three paragraphs.",
    "Implement binary search in Python and explain its time complexity.",
    "Describe how speculative decoding preserves the target output distribution.",
]
