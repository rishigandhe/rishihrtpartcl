"""
Optional DREAMPlace-class global placement hook.

Why a PyTorch smooth solver usually scores below "regular" DreamPlace on proxy:

- **Density**: DreamPlace / ePlace use a **Poisson–electrostatic** potential over a
  fine grid (global coupling). We use **local rectangle overlap → bins** + top-10%
  bin penalty — weaker global spreading, easier to get stuck in congested pockets.
- **Optimizer**: DreamPlace uses **Nesterov** + carefully scheduled **γ / λ**;
  we use Adam + short L-BFGS on a **different** smooth objective than PlacementCost.
- **Wirelength**: weighted-average (log-sum-exp) HPWL surrogate vs true half-perimeter
  wirelength in the judge — small γ helps but never identical.
- **Legalization**: we run a **simple push + grid**; academic/commercial flows chain
  stronger legal/detail after global placement.

Until this hook returns positions, we rely on `phase_config.py` + `losses.py` tuning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark


def dreamplace_global_positions(
    benchmark: "Benchmark",
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Optional[torch.Tensor]:
    """
    If implemented, return ``(num_macros, 2)`` centers on *device*/*dtype* for **all** macros.

    Returning ``None`` skips this phase; ``AnalyticalPlacer`` runs Adam/L-BFGS as usual.
    """
    return None
