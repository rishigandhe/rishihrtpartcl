"""
Default competition entry — GPU v6 pilot (top-k vs eDensity race, halos, L-BFGS).

Implementation lives in ``placer_v6_pilot.py``. A subclass is defined here so
``macro_place.evaluate`` accepts the loader's ``__module__`` check
(``__module__ == "analytical_placer"``).

Benchmarks and evaluator ship in-repo under ``external/`` and ``macro_place/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from placer_v6_pilot import AnalyticalPlacer as _AnalyticalPlacerV6
from placer_v6_pilot import benchmark_stress_tier


class AnalyticalPlacer(_AnalyticalPlacerV6):
    """Same as ``placer_v6_pilot.AnalyticalPlacer``; defined here for the evaluator loader."""

    pass


__all__ = [
    "AnalyticalPlacer",
    "benchmark_stress_tier",
]
