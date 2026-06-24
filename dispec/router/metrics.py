"""Prometheus metrics for the DiSpec serving layer.

Exposes the numbers an inference SRE actually watches: TTFT, TPOT, throughput, queue
depth, and per-step batch size. Scraped at /metrics; visualize with the bundled
Grafana dashboard (dashboards/).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("dispec_requests_total", "Requests completed")
TOKENS = Counter("dispec_tokens_generated_total", "Output tokens generated")

TTFT = Histogram(
    "dispec_ttft_seconds", "Time to first token",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
)
TPOT = Histogram(
    "dispec_tpot_seconds", "Time per output token (decode)",
    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32),
)
E2E = Histogram(
    "dispec_request_seconds", "End-to-end request latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)
STEP = Histogram(
    "dispec_step_seconds", "Scheduler step latency",
    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16),
)

RUNNING = Gauge("dispec_running_requests", "Requests currently decoding")
WAITING = Gauge("dispec_waiting_requests", "Requests waiting for admission")
BATCH = Histogram("dispec_batch_tokens", "Tokens per forward step",
                  buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
