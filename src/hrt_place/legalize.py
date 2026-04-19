"""Hard-macro overlap push + grid fallback (CPU numpy)."""

from __future__ import annotations

import numpy as np
import torch


def legalize(pos_t, sizes, fixed_t, nh, cw, ch, gap=0.05):
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
