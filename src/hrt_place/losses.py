"""Smooth surrogate losses (WA wirelength, density, RUDY-style congestion)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def wa_wirelength(pos_all, net_idx, net_mask, gamma):
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


def density_loss(pos_all, sizes, nm, gx, gy, bw, bh):
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


def congestion_loss(pos_all, net_idx, net_mask, gx, gy, bw, bh, gamma):
    """RUDY-style smooth congestion: per-net bbox demand deposited into bins."""
    neg_inf = float("-inf")
    px = pos_all[net_idx, 0]
    py = pos_all[net_idx, 1]
    xmax = (px * F.softmax(px.masked_fill(~net_mask, neg_inf) / gamma, dim=1)).sum(1)
    xmin = -((-px) * F.softmax((-px).masked_fill(~net_mask, neg_inf) / gamma, dim=1)).sum(1)
    ymax = (py * F.softmax(py.masked_fill(~net_mask, neg_inf) / gamma, dim=1)).sum(1)
    ymin = -((-py) * F.softmax((-py).masked_fill(~net_mask, neg_inf) / gamma, dim=1)).sum(1)
    n_pins = net_mask.sum(1).to(px.dtype)
    w = (xmax - xmin).clamp(min=bw)
    h = (ymax - ymin).clamp(min=bh)
    demand = n_pins / (w * h + 1e-9)
    tau = max(bw, bh) * 0.3
    gate_x = torch.sigmoid((gx[None, :] - xmin[:, None]) / tau) - torch.sigmoid(
        (gx[None, :] - xmax[:, None]) / tau
    )
    gate_y = torch.sigmoid((gy[None, :] - ymin[:, None]) / tau) - torch.sigmoid(
        (gy[None, :] - ymax[:, None]) / tau
    )
    cong = (demand[:, None, None] * gate_y[:, :, None] * gate_x[:, None, :]).sum(0)
    flat = cong.reshape(-1)
    k = max(1, int(flat.shape[0] * 0.1))
    top, _ = torch.topk(flat, k)
    return top.mean()
