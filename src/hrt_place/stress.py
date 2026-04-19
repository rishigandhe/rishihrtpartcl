"""Benchmark difficulty tiers for adaptive budgets."""

from __future__ import annotations

from macro_place.benchmark import Benchmark


def benchmark_stress_tier(benchmark: Benchmark) -> int:
    """
    Rough difficulty tier from benchmark shape (0 = default, 1 = heavy, 2 = very heavy).

    Tuned so small-proxy designs like ibm01 stay tier 0, while large / net-dense
    cases (ibm17, ibm12, ibm18, ibm06, …) get extra Adam / soft-refine / density.
    """
    nh = max(int(benchmark.num_hard_macros), 1)
    nn = int(benchmark.num_nets)
    nm = int(benchmark.num_macros)
    ratio = nn / nh
    if nh >= 620 or nn >= 30000 or nm >= 2500 or (ratio >= 78 and nh >= 220):
        return 2
    if nh >= 380 or nn >= 15000 or nm >= 1950 or ratio >= 50:
        return 1
    return 0
