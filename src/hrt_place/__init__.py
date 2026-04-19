"""
HRT / Partcl macro placement code (modular).

Entry points for the competition harness live at repo ``src/placer_vnext.py`` and
``src/analytical_placer.py`` (shim).
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
