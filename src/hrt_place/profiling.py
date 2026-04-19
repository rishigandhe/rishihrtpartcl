"""Optional judge-proxy logging around legalization (debug)."""

from __future__ import annotations

import os

import torch
from macro_place.benchmark import Benchmark

from hrt_place.paths import DEPS_ICCAD


def legalize_proxy_profiling_enabled() -> bool:
    v = os.environ.get("RISHI_PROFILE_LEGALIZE_PROXY", "").strip().lower()
    return v in ("1", "true", "yes")


def load_plc_for_proxy_profile(benchmark: Benchmark):
    d = DEPS_ICCAD / benchmark.name
    if not d.is_dir():
        return None
    from macro_place.loader import load_benchmark_from_dir

    _, plc = load_benchmark_from_dir(str(d))
    return plc


def full_placement_cpu(pos: torch.Tensor, init: torch.Tensor) -> torch.Tensor:
    out = init.clone()
    out[:] = pos
    return out.detach().cpu().float()


def proxy_cost_dict(benchmark: Benchmark, plc, pos: torch.Tensor, init: torch.Tensor) -> dict:
    from macro_place.objective import compute_proxy_cost

    return compute_proxy_cost(full_placement_cpu(pos, init), benchmark, plc)


def log_proxy_legalization_profile(
    label: str,
    benchmark: Benchmark,
    plc,
    pos: torch.Tensor,
    init: torch.Tensor,
) -> dict:
    c = proxy_cost_dict(benchmark, plc, pos, init)
    print(
        f"  [{benchmark.name}] [PROFILE] {label}: proxy={c['proxy_cost']:.4f}  "
        f"wl={c['wirelength_cost']:.4f} den={c['density_cost']:.4f} "
        f"cong={c['congestion_cost']:.4f} overlaps={c['overlap_count']}",
        flush=True,
    )
    return c
