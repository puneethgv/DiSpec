"""Draft-pool autoscaling controller for disaggregated speculative decoding.

In a disaggregated deployment the draft model runs as its own pool of workers feeding
the target verifier. As decode load rises, more draft replicas are needed to keep the
verifier fed; as it falls, replicas should be released. This is the control policy that
decides the replica count from observed load, with hysteresis (two thresholds) so it
doesn't flap. It's intentionally a pure, deterministic function of load so it can be
unit-tested and driven by either the live scheduler or a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoscaleConfig:
    min_replicas: int = 1
    max_replicas: int = 8
    seqs_per_replica: float = 4.0   # decode seqs one draft replica can keep fed
    scale_up_util: float = 0.8      # grow when utilization exceeds this
    scale_down_util: float = 0.4    # shrink when utilization drops below this


class DraftPoolAutoscaler:
    def __init__(self, cfg: AutoscaleConfig | None = None):
        self.cfg = cfg or AutoscaleConfig()
        self.replicas = self.cfg.min_replicas

    def observe(self, active_seqs: int) -> int:
        """Update and return the desired replica count given the current decode load."""
        c = self.cfg
        capacity = self.replicas * c.seqs_per_replica
        util = active_seqs / capacity if capacity > 0 else float("inf")
        if util > c.scale_up_util and self.replicas < c.max_replicas:
            self.replicas += 1
        elif util < c.scale_down_util and self.replicas > c.min_replicas:
            self.replicas -= 1
        return self.replicas
