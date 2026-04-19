"""
GPU analytical macro placer: Adam (+ optional L-BFGS) → legalize → soft refine → optional oracle refine.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark

from hrt_place.constants import PROXY_REFINE_HARD, proxy_refine_budget_for_tier
from hrt_place.init_grid import build_nets, uniform_spread
from hrt_place.legalize import legalize
from hrt_place.losses import congestion_loss, density_loss, wa_wirelength
from hrt_place.paths import DEPS_ICCAD, HRT_SRC, ensure_src_on_path
from hrt_place.profiling import (
    legalize_proxy_profiling_enabled,
    load_plc_for_proxy_profile,
    log_proxy_legalization_profile,
)
from hrt_place.backends.dreamplace import dreamplace_global_positions
from hrt_place.phase_config import (
    CONGESTION_RAMP_START_FRAC,
    tier_cong_w_phase1,
    tier_cong_w_soft_refine,
)
from hrt_place.stress import benchmark_stress_tier


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
        verbose=True,
        adaptive_hard: bool = True,
        lbfgs_steps: int = 0,
        lbfgs_max_iter: int = 12,
        soft_grad_scale: float = 0.12,
        joint_soft_adam: bool = False,
        joint_soft_lbfgs: bool = True,
        proxy_refine_base_budget: int = 0,
        proxy_refine_verbose: bool = False,
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
        self.verbose = verbose
        self.adaptive_hard = adaptive_hard
        self.lbfgs_steps = lbfgs_steps
        self.lbfgs_max_iter = lbfgs_max_iter
        self.soft_grad_scale = float(soft_grad_scale)
        self.joint_soft_adam = bool(joint_soft_adam)
        self.joint_soft_lbfgs = bool(joint_soft_lbfgs)
        self.proxy_refine_base_budget = int(proxy_refine_base_budget)
        self.proxy_refine_verbose = bool(proxy_refine_verbose)

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

        tier = benchmark_stress_tier(benchmark) if self.adaptive_hard else 0
        global_iters = self.global_iters
        soft_refine_iters = self.soft_refine_iters
        lr_adapt = self.lr
        dw_s_adapt = self.dw_s
        dw_e_adapt = self.dw_e
        dw_p3_adapt = self.dw_p3
        gamma_e_adapt = self.gamma_e
        if tier == 1:
            global_iters = int(round(global_iters * 1.14))
            soft_refine_iters = int(round(soft_refine_iters * 1.22))
            dw_e_adapt = min(dw_e_adapt * 1.08, 6.0)
            dw_p3_adapt = min(max(self.dw_p3, dw_e_adapt * 0.95), dw_e_adapt)
            lr_adapt *= 0.95
        elif tier == 2:
            global_iters = int(round(global_iters * 1.36))
            soft_refine_iters = int(round(soft_refine_iters * 1.48))
            dw_e_adapt = min(dw_e_adapt * 1.16, 6.85)
            dw_p3_adapt = min(max(self.dw_p3, dw_e_adapt * 0.97), dw_e_adapt)
            lr_adapt *= 0.9
            gamma_e_adapt *= 0.92

        sizes = benchmark.macro_sizes.to(dev, dt)
        fixed = benchmark.macro_fixed.to(dev)
        init = benchmark.macro_positions.to(dev, dt)

        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        lb = torch.stack([hw, hh], dim=1)
        ub = torch.stack([cw - hw, ch - hh], dim=1)

        net_idx, net_mask, port_pos = build_nets(benchmark, dev, dt)
        n_nets = net_idx.shape[0]
        if n_nets == 0:
            return init.cpu().float()

        gx = (torch.arange(gc, device=dev, dtype=dt) + 0.5) * bw
        gy = (torch.arange(gr, device=dev, dtype=dt) + 0.5) * bh

        alt = dreamplace_global_positions(benchmark, device=dev, dtype=dt)
        if alt is not None and alt.shape[0] == nm:
            pos = torch.max(torch.min(alt.to(dev, dt), ub), lb)
            with torch.no_grad():
                pos[fixed] = init[fixed]
        else:
            pos = uniform_spread(benchmark, dev, dt)
        p = pos.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr_adapt)

        gamma_init = diag * self.gamma_s
        with torch.no_grad():
            all_pos_init = torch.cat([p.detach(), port_pos], dim=0)
            wl_0 = max(abs(wa_wirelength(all_pos_init, net_idx, net_mask, gamma_init).item()), 1.0)
            dens_0 = max(abs(density_loss(p.detach(), sizes, nm, gx, gy, bw, bh).item()), 1e-6)
            cl_0 = max(
                abs(
                    congestion_loss(
                        all_pos_init, net_idx, net_mask, gx, gy, bw, bh, gamma_init
                    ).item()
                ),
                1e-6,
            )
        if self.verbose:
            adapt_note = f"  stress_tier={tier}" if self.adaptive_hard else ""
            arch = ""
            if self.lbfgs_steps > 0:
                arch = f"  +L-BFGS×{self.lbfgs_steps}"
            print(
                f"  [{benchmark.name}] Phase 1: Adam ({global_iters} iters, {dev}){arch}  "
                f"WL0={wl_0:.0f}  Dens0={dens_0:.4f}{adapt_note}",
                flush=True,
            )

        for k in range(global_iters):
            frac = k / max(global_iters - 1, 1)
            gamma = diag * self.gamma_s * (gamma_e_adapt / self.gamma_s) ** frac
            dw = dw_s_adapt * (dw_e_adapt / dw_s_adapt) ** frac

            opt.zero_grad()
            all_pos = torch.cat([p, port_pos], dim=0)
            wl = wa_wirelength(all_pos, net_idx, net_mask, gamma)
            wl_n = wl / wl_0
            dl = density_loss(p, sizes, nm, gx, gy, bw, bh)
            dl_n = dl / dens_0

            ramp_denom = global_iters * max(1e-9, 1.0 - CONGESTION_RAMP_START_FRAC)
            ramp = max(0.0, (k - global_iters * CONGESTION_RAMP_START_FRAC) / ramp_denom)
            tier_weight = tier_cong_w_phase1(tier)
            cong_w = tier_weight * ramp
            if cong_w > 0:
                cl = congestion_loss(all_pos, net_idx, net_mask, gx, gy, bw, bh, gamma)
                cl_n = cl / cl_0
            else:
                cl_n = torch.tensor(0.0, device=p.device, dtype=p.dtype)

            loss = wl_n + dw * dl_n + cong_w * cl_n
            loss.backward()

            with torch.no_grad():
                p.grad[fixed] = 0.0
                frac_k = k / max(global_iters - 1, 1)
                allow_soft = self.joint_soft_adam and frac_k >= 0.78 and nh < nm
                if nh < nm:
                    mov_soft = (~fixed[nh:nm]).nonzero(as_tuple=False).view(-1) + nh
                    if mov_soft.numel() > 0:
                        if not allow_soft:
                            p.grad[mov_soft] = 0.0
                        elif self.soft_grad_scale != 1.0:
                            p.grad[mov_soft] *= self.soft_grad_scale
                torch.nn.utils.clip_grad_norm_([p], max_norm=diag * 0.5)

            opt.step()
            with torch.no_grad():
                p.data = torch.max(torch.min(p.data, ub), lb)
                p.data[fixed] = init[fixed]

            if self.verbose and (k % 200 == 0 or k == global_iters - 1):
                print(
                    f"    iter {k:4d}  WL={wl_n.item():.3f}  Dens={dl_n.item():.3f}  "
                    f"Cong={cl_n.item():.3f}  dw={dw:.4f}  g={gamma:.3f}",
                    flush=True,
                )

        if self.lbfgs_steps > 0:
            if self.verbose:
                print(
                    f"  [{benchmark.name}] Phase 1b: L-BFGS ({self.lbfgs_steps} steps, "
                    f"max_iter={self.lbfgs_max_iter})",
                    flush=True,
                )
            p_l = p.detach().clone().requires_grad_(True)
            opt_lbfgs = torch.optim.LBFGS(
                [p_l],
                lr=0.9,
                max_iter=self.lbfgs_max_iter,
                history_size=min(100, 20 + self.lbfgs_steps * 3),
                line_search_fn="strong_wolfe",
            )
            gamma_l = max(diag * gamma_e_adapt * 1.15, 0.06)
            dw_l = dw_e_adapt * 0.82
            tier_w = tier_cong_w_phase1(tier)

            def _lbfgs_closure():
                opt_lbfgs.zero_grad()
                all_pos = torch.cat([p_l, port_pos], dim=0)
                wl = wa_wirelength(all_pos, net_idx, net_mask, gamma_l)
                wl_n = wl / wl_0
                dl = density_loss(p_l, sizes, nm, gx, gy, bw, bh)
                dl_n = dl / dens_0
                cl = congestion_loss(all_pos, net_idx, net_mask, gx, gy, bw, bh, gamma_l)
                cl_n = cl / cl_0
                loss = wl_n + dw_l * dl_n + tier_w * cl_n
                loss.backward()
                with torch.no_grad():
                    p_l.grad[fixed] = 0.0
                    if nh < nm:
                        mov_soft = (~fixed[nh:nm]).nonzero(as_tuple=False).view(-1) + nh
                        if mov_soft.numel() > 0:
                            if self.joint_soft_lbfgs and self.soft_grad_scale != 1.0:
                                p_l.grad[mov_soft] *= self.soft_grad_scale
                            elif not self.joint_soft_lbfgs:
                                p_l.grad[mov_soft] = 0.0
                return loss

            for _s in range(self.lbfgs_steps):
                opt_lbfgs.step(_lbfgs_closure)
                with torch.no_grad():
                    p_l.data = torch.max(torch.min(p_l.data, ub), lb)
                    p_l.data[fixed] = init[fixed]
            p = p_l

        pos = p.detach()
        t1 = time.time()
        if self.verbose:
            print(f"  Phase 1: {t1 - t0:.1f}s", flush=True)

        plc_prof = None
        c_pre_legal = None
        if legalize_proxy_profiling_enabled():
            plc_prof = load_plc_for_proxy_profile(benchmark)
            if plc_prof is None:
                print(
                    f"  [{benchmark.name}] [PROFILE] skip (missing {DEPS_ICCAD / benchmark.name})",
                    flush=True,
                )
            else:
                c_pre_legal = log_proxy_legalization_profile(
                    "after Phase 1 (pre-legal)",
                    benchmark,
                    plc_prof,
                    pos,
                    init,
                )

        with torch.no_grad():
            dens_pre_legal = density_loss(pos, sizes, nm, gx, gy, bw, bh).item()
        if self.verbose:
            print(
                f"  [{benchmark.name}] Phase 2: Legalization  (density before: {dens_pre_legal:.4f})",
                flush=True,
            )
        pos = legalize(pos, sizes, fixed, nh, cw, ch)
        t2 = time.time()
        with torch.no_grad():
            dens_post_legal = density_loss(pos, sizes, nm, gx, gy, bw, bh).item()
        if self.verbose:
            print(
                f"  Phase 2: {t2 - t1:.1f}s  (density after: {dens_post_legal:.4f})",
                flush=True,
            )

        if plc_prof is not None and c_pre_legal is not None:
            c_post_legal = log_proxy_legalization_profile(
                "after Phase 2 (post-legal)",
                benchmark,
                plc_prof,
                pos,
                init,
            )
            d = float(c_post_legal["proxy_cost"]) - float(c_pre_legal["proxy_cost"])
            print(
                f"  [{benchmark.name}] [PROFILE] legalize Δproxy (post − pre) = {d:+.4f}",
                flush=True,
            )

        if nm > nh and soft_refine_iters > 0:
            n_soft = nm - nh
            if self.verbose:
                print(
                    f"  [{benchmark.name}] Phase 3: Soft refine "
                    f"({soft_refine_iters} iters, {n_soft} soft macros)",
                    flush=True,
                )
            gamma_f = max(diag * gamma_e_adapt * 2, 0.1)
            dw3 = dw_p3_adapt
            cong_w3 = tier_cong_w_soft_refine(tier)

            q = pos.clone().requires_grad_(True)
            opt2 = torch.optim.Adam([q], lr=lr_adapt * 0.3)

            for _ in range(soft_refine_iters):
                opt2.zero_grad()
                all_pos = torch.cat([q, port_pos], dim=0)
                wl_s = wa_wirelength(all_pos, net_idx, net_mask, gamma_f)
                dl_s = density_loss(q, sizes, nm, gx, gy, bw, bh)
                cl_s = congestion_loss(all_pos, net_idx, net_mask, gx, gy, bw, bh, gamma_f)
                cl_s_n = cl_s / cl_0
                loss_s = wl_s / wl_0 + dw3 * dl_s / dens_0 + cong_w3 * cl_s_n
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
                dens_post3 = density_loss(pos, sizes, nm, gx, gy, bw, bh).item()
            if self.verbose:
                print(f"  Phase 3: {t3 - t2:.1f}s  (density after: {dens_post3:.4f})", flush=True)

        result = init.clone()
        result[:] = pos

        if self.proxy_refine_base_budget > 0 and benchmark.name in PROXY_REFINE_HARD:
            case_dir = DEPS_ICCAD / benchmark.name
            if case_dir.is_dir():
                ensure_src_on_path()
                from proxy_refine import refine_placement
                from macro_place.loader import load_benchmark_from_dir

                _, plc_oracle = load_benchmark_from_dir(str(case_dir))
                budget = proxy_refine_budget_for_tier(self.proxy_refine_base_budget, tier)
                if self.verbose:
                    print(
                        f"  [{benchmark.name}] Phase 4: Proxy refine (oracle, budget<={budget})",
                        flush=True,
                    )
                t_pr0 = time.time()
                refined, st = refine_placement(
                    result,
                    benchmark,
                    plc_oracle,
                    max_evals=budget,
                    seed=(self.seed + 41) % (2**31),
                    verbose=self.proxy_refine_verbose and self.verbose,
                )
                if not st.get("skipped") and self.verbose:
                    print(
                        f"  [{benchmark.name}] Phase 4: {time.time() - t_pr0:.1f}s  "
                        f"proxy {st['proxy_before']:.4f} -> {st['proxy_after']:.4f}  "
                        f"evals={st['evals']}",
                        flush=True,
                    )
                result = refined
                pos = result.to(device=pos.device, dtype=pos.dtype)

        if plc_prof is not None and c_pre_legal is not None:
            c_final = log_proxy_legalization_profile(
                "after Phase 3 (final)" if (nm > nh and soft_refine_iters > 0) else "final (no soft refine)",
                benchmark,
                plc_prof,
                pos,
                init,
            )
            d_tot = float(c_final["proxy_cost"]) - float(c_pre_legal["proxy_cost"])
            print(
                f"  [{benchmark.name}] [PROFILE] total Δproxy (final − pre-legal) = {d_tot:+.4f}",
                flush=True,
            )

        if self.verbose:
            print(f"  [{benchmark.name}] Total: {time.time() - t0:.1f}s", flush=True)
        return result.cpu().float()
