"""
HRT / Partcl macro placement code (modular).

Competition default: ``src/analytical_placer.py`` (shim to ``placer_v6_pilot.py``).
Legacy VNext harness entry: ``src/placer_vnext.py``.
"""

from hrt_place.analytical import AnalyticalPlacer
from hrt_place.constants import PROXY_REFINE_HARD, proxy_refine_budget_for_tier
from hrt_place.stress import benchmark_stress_tier

__all__ = [
    "AnalyticalPlacer",
    "benchmark_stress_tier",
    "PROXY_REFINE_HARD",
    "proxy_refine_budget_for_tier",
]
