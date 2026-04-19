"""Initial hard-macro spread and net tensors."""

from __future__ import annotations

import math

import torch

from macro_place.benchmark import Benchmark


def uniform_spread(benchmark: Benchmark, dev, dt) -> torch.Tensor:
    nh = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    init = benchmark.macro_positions.to(dev, dt).clone()
    fix = benchmark.macro_fixed.to(dev)

    movable = (~fix[:nh]).nonzero(as_tuple=False).squeeze(1)
    n_mov = movable.shape[0]
    if n_mov == 0:
        return init

    cols = max(1, math.ceil(math.sqrt(n_mov * cw / ch)))
    rows = max(1, math.ceil(n_mov / cols))
    xs = torch.linspace(cw * 0.05, cw * 0.95, cols, device=dev, dtype=dt)
    ys = torch.linspace(ch * 0.05, ch * 0.95, rows, device=dev, dtype=dt)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)[:n_mov]

    hw = benchmark.macro_sizes[:nh, 0].to(dev, dt) / 2
    hh = benchmark.macro_sizes[:nh, 1].to(dev, dt) / 2

    for k, i in enumerate(movable.tolist()):
        init[i, 0] = grid_pts[k, 0].clamp(hw[i], cw - hw[i])
        init[i, 1] = grid_pts[k, 1].clamp(hh[i], ch - hh[i])
    return init


def build_nets(bm: Benchmark, dev, dt):
    port_pos = bm.port_positions.to(dev, dt)
    valid = [n for n in bm.net_nodes if len(n) >= 2]
    if not valid:
        return (
            torch.zeros(0, 1, dtype=torch.long, device=dev),
            torch.zeros(0, 1, dtype=torch.bool, device=dev),
            port_pos,
        )
    k_max = max(len(n) for n in valid)
    idx = torch.zeros(len(valid), k_max, dtype=torch.long, device=dev)
    msk = torch.zeros(len(valid), k_max, dtype=torch.bool, device=dev)
    for i, n in enumerate(valid):
        l = len(n)
        idx[i, :l] = n.to(dev)
        msk[i, :l] = True
    return idx, msk, port_pos
