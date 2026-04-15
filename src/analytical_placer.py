"""
GPU Analytical Macro Placer - v2
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


def _uniform_spread(benchmark: Benchmark, dev, dt) -> torch.Tensor:
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


def _build_nets(bm: Benchmark, dev, dt):
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


def _wa_wirelength(pos_all, net_idx, net_mask, gamma):
    neg_inf = float("-inf")
    px = pos_all[net_idx, 0]
    py = pos_all[net_idx, 1]
    px_pos = px.masked_fill(~net_mask, neg_inf)
    py_pos = py.masked_fill(~net_mask, neg_inf)
    px_neg = (-px).masked_fill(~net_mask, neg_inf)
    py_neg = (-py).masked_fill(~net_mask, neg_inf)

    sm_xp = F.softmax(px_pos / gamma, dim=1)
    sm_xn = F.softmax(px_neg / gamma, dim=1)
    sm_yp = F.softmax(py_pos / gamma, dim=1)
    sm_yn = F.softmax(py_neg / gamma, dim=1)

    return (
        (px * sm_xp).sum(1)
        - (px * sm_xn).sum(1)
        + (py * sm_yp).sum(1)
        - (py * sm_yn).sum(1)
    ).sum()


def _density_loss(pos_all, sizes, nm, gx, gy, bw, bh):
    cx = pos_all[:nm, 0]
    cy = pos_all[:nm, 1]
    mw = sizes[:nm, 0]
    mh = sizes[:nm, 1]

    ox = torch.clamp(
        torch.min(cx[:, None] + mw[:, None] / 2, gx[None, :] + bw / 2)
        - torch.max(cx[:, None] - mw[:, None] / 2, gx[None, :] - bw / 2),
        min=0.0,
    )
    oy = torch.clamp(
        torch.min(cy[:, None] + mh[:, None] / 2, gy[None, :] + bh / 2)
        - torch.max(cy[:, None] - mh[:, None] / 2, gy[None, :] - bh / 2),
        min=0.0,
    )

    dens = (oy.T @ ox) / (bw * bh)
    dens_flat = dens.reshape(-1)
    k = max(1, int(dens_flat.shape[0] * 0.1))
    top_k, _ = torch.topk(dens_flat, k, sorted=False)
    return 0.5 * top_k.mean()


def _legalize(pos_t, sizes, fixed_t, nh, cw, ch, gap=0.05):
    pos = pos_t[:nh].detach().cpu().numpy().copy().astype(np.float64)
    sz = sizes[:nh].cpu().numpy().astype(np.float64)
    fix = fixed_t[:nh].cpu().numpy()
    hw = sz[:, 0] / 2
    hh = sz[:, 1] / 2
    mov = ~fix

    sep_x = hw[:, None] + hw[None, :]
    sep_y = hh[:, None] + hh[None, :]

    for _ in range(150):
        dx = np.abs(pos[:, 0:1] - pos[:, 0])
        dy = np.abs(pos[:, 1:2] - pos[:, 1])
        ov = (dx < sep_x + gap) & (dy < sep_y + gap)
        np.fill_diagonal(ov, False)
        if not ov.any():
            break
        oi, oj = np.where(np.triu(ov, k=1))
        for i, j in zip(oi, oj):
            adx = abs(pos[i, 0] - pos[j, 0])
            ady = abs(pos[i, 1] - pos[j, 1])
            ovx = sep_x[i, j] + gap - adx
            ovy = sep_y[i, j] + gap - ady
            if ovx <= 0 or ovy <= 0:
                continue
            if ovx < ovy:
                sgn = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                d = ovx / 2 + 0.01
                if mov[i]:
                    pos[i, 0] += sgn * d
                if mov[j]:
                    pos[j, 0] -= sgn * d
            else:
                sgn = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                d = ovy / 2 + 0.01
                if mov[i]:
                    pos[i, 1] += sgn * d
                if mov[j]:
                    pos[j, 1] -= sgn * d
        pos[:, 0] = np.clip(pos[:, 0], hw, cw - hw)
        pos[:, 1] = np.clip(pos[:, 1], hh, ch - hh)

    dx = np.abs(pos[:, 0:1] - pos[:, 0])
    dy = np.abs(pos[:, 1:2] - pos[:, 1])
    ov = (dx < sep_x + gap) & (dy < sep_y + gap)
    np.fill_diagonal(ov, False)
    if not ov.any():
        result = pos_t.clone()
        result[:nh] = torch.tensor(pos, device=pos_t.device, dtype=pos_t.dtype)
        return result

    areas = sz[:, 0] * sz[:, 1]
    order = np.argsort(-areas)
    placed = fix.copy()
    legal = pos.copy()

    for idx in order:
        if fix[idx]:
            placed[idx] = True
            continue

        if placed.any():
            ddx = np.abs(legal[idx, 0] - legal[:, 0])
            ddy = np.abs(legal[idx, 1] - legal[:, 1])
            col = (ddx < sep_x[idx] + gap) & (ddy < sep_y[idx] + gap) & placed
            col[idx] = False
            if not col.any():
                placed[idx] = True
                continue

        step = max(sz[idx, 0], sz[idx, 1]) * 0.25
        orig = pos[idx].copy()
        best_p = legal[idx].copy()
        best_d = float("inf")

        for r in range(1, 300):
            found = False
            for dxi in range(-r, r + 1):
                ys = ([-r, r] if abs(dxi) != r else range(-r, r + 1))
                for dyi in ys:
                    cx_ = np.clip(orig[0] + dxi * step, hw[idx], cw - hw[idx])
                    cy_ = np.clip(orig[1] + dyi * step, hh[idx], ch - hh[idx])
                    if placed.any():
                        ddx = np.abs(cx_ - legal[:, 0])
                        ddy = np.abs(cy_ - legal[:, 1])
                        col = ((ddx < sep_x[idx] + gap) & (ddy < sep_y[idx] + gap) & placed)
                        col[idx] = False
                        if col.any():
                            continue
                    d = (cx_ - orig[0]) ** 2 + (cy_ - orig[1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx_, cy_])
                        found = True
            if found:
                break

        legal[idx] = best_p
        placed[idx] = True

    result = pos_t.clone()
    result[:nh] = torch.tensor(legal, device=pos_t.device, dtype=pos_t.dtype)
    return result


class AnalyticalPlacer:
    def __init__(
        self,
        global_iters=1000,
        soft_refine_iters=300,
        lr=0.3,
        gamma_start_frac=0.08,
        gamma_end_frac=0.003,
        dw_start=0.005,
        dw_end=5.0,
        dw_phase3=None,
        seed=42,
    ):
        self.global_iters = global_iters
        self.soft_refine_iters = soft_refine_iters
        self.lr = lr
        self.gamma_s = gamma_start_frac
        self.gamma_e = gamma_end_frac
        self.dw_s = dw_start
        self.dw_e = dw_end
        self.dw_p3 = dw_phase3 if dw_phase3 is not None else dw_end
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float64

        nh = benchmark.num_hard_macros
        nm = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        diag = math.hypot(cw, ch)
        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        bw, bh = cw / gc, ch / gr

        sizes = benchmark.macro_sizes.to(dev, dt)
        fixed = benchmark.macro_fixed.to(dev)
        init = benchmark.macro_positions.to(dev, dt)

        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        lb = torch.stack([hw, hh], dim=1)
        ub = torch.stack([cw - hw, ch - hh], dim=1)

        net_idx, net_mask, port_pos = _build_nets(benchmark, dev, dt)
        n_nets = net_idx.shape[0]
        if n_nets == 0:
            return init.cpu().float()

        gx = (torch.arange(gc, device=dev, dtype=dt) + 0.5) * bw
        gy = (torch.arange(gr, device=dev, dtype=dt) + 0.5) * bh

        pos = _uniform_spread(benchmark, dev, dt)
        p = pos.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=self.lr)

        gamma_init = diag * self.gamma_s
        with torch.no_grad():
            all_pos_init = torch.cat([p.detach(), port_pos], dim=0)
            wl_0 = max(abs(_wa_wirelength(all_pos_init, net_idx, net_mask, gamma_init).item()), 1.0)
            dens_0 = max(abs(_density_loss(p.detach(), sizes, nm, gx, gy, bw, bh).item()), 1e-6)
        print(f"  [{benchmark.name}] Phase 1: Adam ({self.global_iters} iters, {dev})  WL0={wl_0:.0f}  Dens0={dens_0:.4f}")

        for k in range(self.global_iters):
            frac = k / max(self.global_iters - 1, 1)
            gamma = diag * self.gamma_s * (self.gamma_e / self.gamma_s) ** frac
            dw = self.dw_s * (self.dw_e / self.dw_s) ** frac

            opt.zero_grad()
            all_pos = torch.cat([p, port_pos], dim=0)
            wl = _wa_wirelength(all_pos, net_idx, net_mask, gamma)
            wl_n = wl / wl_0
            dl = _density_loss(p, sizes, nm, gx, gy, bw, bh)
            dl_n = dl / dens_0
            loss = wl_n + dw * dl_n
            loss.backward()

            with torch.no_grad():
                p.grad[fixed] = 0.0
                p.grad[nh:] = 0.0
                torch.nn.utils.clip_grad_norm_([p], max_norm=diag * 0.5)

            opt.step()
            with torch.no_grad():
                p.data = torch.max(torch.min(p.data, ub), lb)
                p.data[fixed] = init[fixed]

            if k % 200 == 0 or k == self.global_iters - 1:
                print(f"    iter {k:4d}  WL={wl_n.item():.3f}  Dens={dl_n.item():.3f}  dw={dw:.4f}  g={gamma:.3f}")

        pos = p.detach()
        t1 = time.time()
        print(f"  Phase 1: {t1 - t0:.1f}s")

        with torch.no_grad():
            dens_pre_legal = _density_loss(pos, sizes, nm, gx, gy, bw, bh).item()
        print(f"  [{benchmark.name}] Phase 2: Legalization  (density before: {dens_pre_legal:.4f})")
        pos = _legalize(pos, sizes, fixed, nh, cw, ch)
        t2 = time.time()
        with torch.no_grad():
            dens_post_legal = _density_loss(pos, sizes, nm, gx, gy, bw, bh).item()
        print(f"  Phase 2: {t2 - t1:.1f}s  (density after: {dens_post_legal:.4f})")

        if nm > nh and self.soft_refine_iters > 0:
            n_soft = nm - nh
            print(f"  [{benchmark.name}] Phase 3: Soft refine ({self.soft_refine_iters} iters, {n_soft} soft macros)")
            gamma_f = max(diag * self.gamma_e * 2, 0.1)
            dw3 = self.dw_p3

            q = pos.clone().requires_grad_(True)
            opt2 = torch.optim.Adam([q], lr=self.lr * 0.3)

            for _ in range(self.soft_refine_iters):
                opt2.zero_grad()
                all_pos = torch.cat([q, port_pos], dim=0)
                wl_s = _wa_wirelength(all_pos, net_idx, net_mask, gamma_f)
                dl_s = _density_loss(q, sizes, nm, gx, gy, bw, bh)
                loss_s = wl_s / wl_0 + dw3 * dl_s / dens_0
                loss_s.backward()

                with torch.no_grad():
                    q.grad[:nh] = 0.0
                    q.grad[fixed] = 0.0
                    torch.nn.utils.clip_grad_norm_([q], max_norm=diag * 0.3)

                opt2.step()
                with torch.no_grad():
                    q.data = torch.max(torch.min(q.data, ub), lb)
                    q.data[:nh] = pos[:nh]
                    q.data[fixed] = init[fixed]

            pos = q.detach()
            t3 = time.time()
            with torch.no_grad():
                dens_post3 = _density_loss(pos, sizes, nm, gx, gy, bw, bh).item()
            print(f"  Phase 3: {t3 - t2:.1f}s  (density after: {dens_post3:.4f})")

        result = init.clone()
        result[:] = pos
        print(f"  [{benchmark.name}] Total: {time.time() - t0:.1f}s")
        return result.cpu().float()
