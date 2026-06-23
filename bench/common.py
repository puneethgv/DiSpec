"""Shared benchmarking utilities: latency metrics and pretty reporting."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field


@dataclass
class RequestMetrics:
    """Per-request latency record."""

    ttft_s: float  # time to first token
    total_s: float  # end-to-end wall time
    num_prompt_tokens: int
    num_output_tokens: int

    @property
    def tpot_s(self) -> float:
        """Time per output token (decode-only), excluding the first token."""
        gen = max(self.num_output_tokens - 1, 1)
        return (self.total_s - self.ttft_s) / gen


@dataclass
class BenchResult:
    name: str
    requests: list[RequestMetrics] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def add(self, m: RequestMetrics) -> None:
        self.requests.append(m)

    def summary(self) -> dict:
        s = {"name": self.name, "n_requests": len(self.requests)}
        if self.requests:
            ttfts = [r.ttft_s for r in self.requests]
            tpots = [r.tpot_s for r in self.requests]
            out_toks = sum(r.num_output_tokens for r in self.requests)
            total_time = sum(r.total_s for r in self.requests)
            s.update(
                ttft_p50_s=round(statistics.median(ttfts), 4),
                tpot_p50_s=round(statistics.median(tpots), 4),
                tpot_p50_tok_per_s=round(1.0 / statistics.median(tpots), 1),
                out_tokens=out_toks,
                aggregate_throughput_tok_per_s=round(out_toks / total_time, 1) if total_time else 0.0,
            )
        s.update(self.extra)
        return s

    def print_summary(self) -> None:
        print(json.dumps(self.summary(), indent=2))

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "requests": [asdict(r) for r in self.requests]}
