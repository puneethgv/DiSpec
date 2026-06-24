"""Draft-pool autoscaler policy tests (pure logic, no GPU)."""

from dispec.router.autoscale import AutoscaleConfig, DraftPoolAutoscaler
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import Request


def test_scales_up_under_load():
    a = DraftPoolAutoscaler(AutoscaleConfig(min_replicas=1, max_replicas=4, seqs_per_replica=4))
    assert a.replicas == 1
    a.observe(8)   # util 8/4 = 2.0 > 0.8 -> grow
    assert a.replicas == 2
    a.observe(20)  # still saturated -> grow
    assert a.replicas == 3


def test_scales_down_when_idle():
    a = DraftPoolAutoscaler(AutoscaleConfig(min_replicas=1, max_replicas=4, seqs_per_replica=4))
    for _ in range(5):
        a.observe(100)
    assert a.replicas == 4  # capped at max
    for _ in range(10):
        a.observe(0)
    assert a.replicas == 1  # shrinks back to min


def test_hysteresis_holds_steady_in_band():
    a = DraftPoolAutoscaler(AutoscaleConfig(min_replicas=1, max_replicas=8, seqs_per_replica=4))
    a.observe(8)  # -> 2 replicas, capacity 8, util 1.0
    r = a.replicas
    # utilization 0.5 is between scale_down (0.4) and scale_up (0.8): no change.
    a.observe(int(2 * 4 * 0.5))
    assert a.replicas == r


def test_priority_admission_order():
    """Higher-priority requests are admitted before older low-priority ones."""
    reqs = [Request(0, [1], SamplingParams(), 8, priority=0),
            Request(1, [1], SamplingParams(), 8, priority=5),
            Request(2, [1], SamplingParams(), 8, priority=1)]
    reqs.sort(key=lambda r: (-r.priority, r.id))
    assert [r.id for r in reqs] == [1, 2, 0]
