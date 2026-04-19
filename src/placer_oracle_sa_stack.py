"""
Stack B — Analytical warm start + simulated annealing on true proxy cost.

Uses the same `compute_proxy_cost` as the evaluator (oracle-aligned search).
SA can accept occasional worse moves to escape local minima vs pure hill climb.

The harness instantiates the placer with no arguments; budgets are fixed here
and scale with `benchmark_stress_tier` (cheap on ibm01-like cases, more oracle
evals on heavy designs).

Usage (from repo root, after setup_benchmark_env):

  bash scripts/run_eval_oracle_sa_stack.sh -b ibm01
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_SRC = Path(__file__).resolve().parent
_CHALLENGE = _SRC.parent / ".deps" / "macro-place-challenge-2026"
_ICCAD = _CHALLENGE / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def _import_analytical():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from analytical_placer import AnalyticalPlacer, benchmark_stress_tier

    return AnalyticalPlacer, benchmark_stress_tier


def _movable_indices(benchmark: Benchmark) -> torch.Tensor:
    return (~benchmark.macro_fixed).nonzero(as_tuple=False).view(-1)


def _clamp_centers(pos: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hw = benchmark.macro_sizes[:, 0] * 0.5
    hh = benchmark.macro_sizes[:, 1] * 0.5
    out = pos.clone()
    out[:, 0] = out[:, 0].clamp(hw, cw - hw)
    out[:, 1] = out[:, 1].clamp(hh, ch - hh)
    return out


def _sa_budget(tier: int) -> int:
    """Extra oracle evaluations (neighbor trials), after initial score."""
    if tier <= 0:
        return 40
    if tier == 1:
        return 72
    return 110


def _load_plc_for_benchmark(benchmark: Benchmark):
    from macro_place.loader import load_benchmark_from_dir

    d = _ICCAD / benchmark.name
    if not d.is_dir():
        return None, None
    return load_benchmark_from_dir(str(d))


class OracleSAStackPlacer:
    """
    1) Run `AnalyticalPlacer` (GPU WL + density + legalize + soft refine).
    2) SA on `compute_proxy_cost` with overlap checks (same spirit as proxy_refine).
    """

    def __init__(self):
        self._seed = 42
        self._step_frac = 0.004
        self._verbose = False

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        AnalyticalPlacer, benchmark_stress_tier = _import_analytical()
        device = benchmark.macro_positions.device

        tier = benchmark_stress_tier(benchmark)
        sa_steps = _sa_budget(tier)

        placer = AnalyticalPlacer(seed=self._seed, verbose=self._verbose, adaptive_hard=True)
        pos0 = placer.place(benchmark)

        loaded = _load_plc_for_benchmark(benchmark)
        if loaded[0] is None:
            return pos0

        _, plc = loaded
        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement

        pos = pos0.detach().float().cpu()
        pos = _clamp_centers(pos, benchmark)

        movable = _movable_indices(benchmark)
        if movable.numel() == 0:
            return pos.to(device)

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        diag = math.hypot(cw, ch)
        step = max(diag * self._step_frac, 0.05)

        g = torch.Generator()
        g.manual_seed(self._seed)

        def energy(p):
            return compute_proxy_cost(p, benchmark, plc)

        cur = pos.clone()
        c = energy(cur)
        if int(c.get("overlap_count", 0)) != 0:
            return pos0

        cur_e = float(c["proxy_cost"])
        best_p = cur.clone()
        best_e = cur_e

        T = max(cur_e * 0.018, 0.012)
        decay = 0.985

        for _ in range(sa_steps):
            idx = int(movable[torch.randint(0, movable.numel(), (1,), generator=g).item()].item())
            axis = int(torch.randint(0, 2, (1,), generator=g).item())
            delta = step if torch.rand((), generator=g).item() < 0.5 else -step
            trial = cur.clone()
            trial[idx, axis] += delta
            trial = _clamp_centers(trial, benchmark)
            ct = energy(trial)
            ok, _ = validate_placement(trial, benchmark, check_overlaps=False)
            if not ok or int(ct.get("overlap_count", 0)) != 0:
                T *= decay
                continue
            ne = float(ct["proxy_cost"])
            d = ne - cur_e
            if d < 0 or torch.rand((), generator=g).item() < math.exp(-d / max(T, 1e-9)):
                cur = trial
                cur_e = ne
                if ne < best_e - 1e-9:
                    best_e = ne
                    best_p = cur.clone()
            T *= decay

        return best_p.to(device)
