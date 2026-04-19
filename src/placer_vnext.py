"""
Legacy VNext placer entry for the challenge harness (modular ``hrt_place`` pipeline).

Architecture: WA wirelength + density, Adam + L-BFGS before legalization.

The class **must** live in this module so ``macro_place.evaluate`` finds it
(``__module__ == "placer_vnext"``).

**Default submission** is ``analytical_placer.py`` (GPU v6 pilot). Run it with
``bash scripts/run_eval.sh`` or evaluate ``src/analytical_placer.py`` directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hrt_place.analytical import AnalyticalPlacer


class VNextPlacer(AnalyticalPlacer):
    """
    Adam global solve + L-BFGS steps on fixed late-γ / mid–late density weighting,
    then inherited legalization and soft-macro refine.

    Environment (optional):
      RISHI_PROXY_REFINE_BUDGET — oracle refinement base budget on PROXY_REFINE_HARD
        benchmarks. Default 0 (off). Example: 6 enables Phase 4 with tier scaling.
      RISHI_SOFT_GRAD_SCALE — L-BFGS (and optional Adam) soft-macro grad scale (default 0.12).
      RISHI_JOINT_SOFT_ADAM — set to 1 for late-Adam joint soft (default 0).
      RISHI_JOINT_SOFT_LBFGS — set to 0 to freeze soft during L-BFGS (default 1).
    """

    def __init__(self, seed=42, verbose=True):
        _pr = os.environ.get("RISHI_PROXY_REFINE_BUDGET")
        proxy_ref_base = 0 if _pr is None else int(_pr.strip())
        _sg = os.environ.get("RISHI_SOFT_GRAD_SCALE")
        soft_scale = 0.12 if _sg is None else float(_sg.strip())
        _ja = os.environ.get("RISHI_JOINT_SOFT_ADAM", "0").strip().lower()
        joint_adam = _ja in ("1", "true", "yes", "on")
        _jl = os.environ.get("RISHI_JOINT_SOFT_LBFGS", "1").strip().lower()
        joint_lbfgs = _jl not in ("0", "false", "no", "off")
        super().__init__(
            global_iters=700,
            soft_refine_iters=340,
            lr=0.275,
            lbfgs_steps=16,
            lbfgs_max_iter=14,
            adaptive_hard=True,
            seed=seed,
            verbose=verbose,
            soft_grad_scale=soft_scale,
            joint_soft_adam=joint_adam,
            joint_soft_lbfgs=joint_lbfgs,
            proxy_refine_base_budget=proxy_ref_base,
        )
