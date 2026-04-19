"""
Backward-compatible entry — core logic lives in `hrt_place/`.

`macro_place.evaluate` requires a subclass defined *in this module* (see ``__module__`` check).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hrt_place.analytical import AnalyticalPlacer as _AnalyticalPlacerBase
from hrt_place.constants import PROXY_REFINE_HARD, proxy_refine_budget_for_tier
from hrt_place.stress import benchmark_stress_tier


class AnalyticalPlacer(_AnalyticalPlacerBase):
    """Same as `hrt_place.analytical.AnalyticalPlacer`; defined here for the evaluator loader."""

    pass


__all__ = [
    "AnalyticalPlacer",
    "benchmark_stress_tier",
    "PROXY_REFINE_HARD",
    "proxy_refine_budget_for_tier",
]
