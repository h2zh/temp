"""The part's own roof-surface fit on the SfM points (docs/STRUCTURE_FIRST.md §2.4.8 U1; §2.4.3, §6 D15 / D27).

One fit, two rules: D25-H (crowd3d.correct, crowd3d.correct_measure) lets the SfM points join an agreeing set only
through this fit, and D15 (crowd3d.judge, decision D27) changes a height only when the agreeing evidence reads the
part's own roof -- the 3-D lines' apex / ridge, or this fit (usable by fit_usable() and agreeing with the value). It
lives below both modules (crowd3d.correct imports crowd3d.judge) so that both call the same code.

sfm_fit(): per 1 m cell of the part's exposed area (minus the parts on it and 1.5 m around taller parts) the
2nd-highest point (cells with >= 3 points), scored against the part's own surface for trial eaves x roof heights
(pitched), tops (flat) or dome heights: inliers within 0.75 m minus cells above; the 90 % interval a 200-sample
bootstrap over the cells; and the share of interior cells whose highest point lies more than TOL under the fitted
roof (a lower roof seen under what the points' top reads).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import structure as S
from . import stylize as st

# D25's thresholds (§2.4.8; chosen, not tuned on data) that the fit and its usability need (crowd3d.correct too)
TOL_M, TOL_REL = 2.0, 0.10                   # off the map by more than max(2 m, 10 % of the map value)
PREC_M, PREC_REL = 3.0, 0.25                 # a channel's 90 % interval <= max(3 m, 25 % of its change)
SFM_FIT_INLIERS, SFM_FIT_CELLS, SFM_FIT_QCELLS = 0.4, 30, 3   # roof-surface fit: inlier share, cells, cells/quadrant
SFM_FIT_BELOW_MAX = 0.25                     # ... and at most this share of interior cells a TOL under the fitted roof
ROOF_MIN_M = 0.5                             # a roof (or a wall band) under this is no roof / no wall
N_BOOT = 200
FIT_TAU = 0.75


def tol_of(v: float) -> float:
    return max(TOL_M, TOL_REL * abs(float(v)))


def prec_of(delta: float) -> float:
    return max(PREC_M, PREC_REL * abs(float(delta)))


def part_kind(b: S.BuiltPart) -> str:
    if b.spec.params["height"].source == st.PROV_DEFAULT:
        return "stub"
    if b.roof.shape not in ("flat", "dome", "onion") and b.roof.height_m > ROOF_MIN_M:
        return "pitched"
    if b.roof.shape in ("dome", "onion") and b.roof.height_m > 0:
        return "dome"
    return "flat"


def exposed_cells(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray, cell: float = 1.0, min_pts: int = 3,
                  margin: float = 1.5):
    """Cells of the part's exposed area (minus the parts on it and margin around taller parts), per cell the
    2nd-highest point when >= min_pts points; and every point over the exposed area above the part's base."""
    lo, hi = b.outer.min(0) - cell, b.outer.max(0) + cell
    g = S._Grid(lo, hi, cell)
    m = g.fill(b.outer, b.holes) > 0
    for o in model.built:
        if o is b:
            continue
        on_me = o.spec.parent == b.id
        taller = o.top > b.top + 1.0
        if not (on_me or taller):
            continue
        om = g.fill(o.outer, o.holes) > 0
        if taller and not on_me:
            r = max(1, int(round(margin / cell)))
            om = cv2.dilate(om.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        m &= ~om
    sel = (P[:, 0] >= g.lo[0]) & (P[:, 1] >= g.lo[1]) & (P[:, 0] < g.lo[0] + g.nx * cell) & (P[:, 1] < g.lo[1] + g.ny * cell)
    q = P[sel]
    ix = np.clip(((q[:, 0] - g.lo[0]) / cell).astype(int), 0, g.nx - 1)
    iy = np.clip(((q[:, 1] - g.lo[1]) / cell).astype(int), 0, g.ny - 1)
    ok = m[iy, ix] & (q[:, 2] > b.z0)
    q, ix, iy = q[ok], ix[ok], iy[ok]
    key = iy * g.nx + ix
    order = np.lexsort((q[:, 2], key))
    key_s, z = key[order], q[order, 2]
    uk, start, cnt = np.unique(key_s, return_index=True, return_counts=True)
    good = cnt >= min_pts
    z2 = z[start[good] + cnt[good] - 2] if good.any() else np.zeros(0)
    cen = g.centres().reshape(g.ny, g.nx, 2)
    c = np.array([cen[k // g.nx, k % g.nx] for k in uk[good]]) if good.any() else np.zeros((0, 2))
    return c, z2, q, m, g


FIT_GRIDS = {"pitched": ((0.70, 2.00, 0.02), (0.3, 4.0, 0.1)),     # eave x roof multipliers (2-D)
             "flat": ((0.70, 2.50, 0.02), (1.0, 1.0, 1.0)),         # the top (eave multiplier), the map's roof kept
             "dome": ((1.0, 1.0, 1.0), (0.3, 4.0, 0.05))}           # the dome's height (roof multiplier), as the skyline
INTERIOR_M = 2.0              # a cell >= 2 m inside the outline sees nothing but the part's roof (or nothing at all)
INTERIOR_MIN_CELLS = 10       # fewer interior cells: the below share is taken over every exposed cell (stricter)


def _mgrid(lo: float, hi: float, step: float) -> list[float]:
    return [round(float(x), 3) for x in np.arange(lo, hi + 1e-9, step)] if hi > lo else [round(float(lo), 3)]


def cell_highest(q: np.ndarray, g: Any, cell: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per cell holding >= 1 of the points q (exposed_cells' points over the exposed area), its centre and highest
    point: a sparsely seen roof (one point per cell) still shows where it is."""
    if not len(q):
        return np.zeros((0, 2)), np.zeros(0)
    ix = np.clip(((q[:, 0] - g.lo[0]) / cell).astype(int), 0, g.nx - 1)
    iy = np.clip(((q[:, 1] - g.lo[1]) / cell).astype(int), 0, g.ny - 1)
    key = iy * g.nx + ix
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], q[order, 2]
    uk, start = np.unique(ks, return_index=True)
    zmax = np.maximum.reduceat(zs, start)
    cen = g.centres().reshape(g.ny, g.nx, 2)
    return np.array([cen[k // g.nx, k % g.nx] for k in uk]), zmax


def below_share(b: S.BuiltPart, c: np.ndarray, zmax: np.ndarray, s: np.ndarray, tol: float) -> dict[str, Any]:
    """The share of the part's interior cells (>= INTERIOR_M inside its outline, each holding >= 1 point) whose
    highest point lies more than TOL below the fitted roof surface: a surface UNDER the photo value, seen where a
    solid block at that value leaves nothing to see (trees or a crane over a roof that really is at the map value).
    With fewer than INTERIOR_MIN_CELLS interior cells every cell counts (edge cells hold wall points too: stricter,
    never looser)."""
    inside = st.points_in_polygon(c, b.outer) if len(c) else np.zeros(0, bool)
    d = st.distance_to_rings(c, [b.outer, *b.holes])[0] if len(c) else np.zeros(0)
    interior = inside & (d >= INTERIOR_M)
    use = interior if int(interior.sum()) >= INTERIOR_MIN_CELLS else np.ones(len(c), bool)
    res = zmax[use] - s[use]
    fin = np.isfinite(res)
    n = int(fin.sum())
    return {"below_share": round(float(np.mean(res[fin] < -tol)), 3) if n else None, "below_cells": n,
            "below_over": "interior" if int(interior.sum()) >= INTERIOR_MIN_CELLS else "all exposed cells",
            "below_tol_m": round(float(tol), 3)}


def sfm_fit(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray, start: tuple[float, float] | None = None,
            n_boot: int = N_BOOT, window: int = 5, kind: str | None = None) -> dict[str, Any]:
    """The roof-surface fit of a part on its exposed cells' 2nd-highest points (score: inliers within 0.75 m minus
    cells above): pitched roofs by eave 0.70-2.00 x 0.02 and roof 0.3-4.0 x 0.1 multipliers; flat roofs by the top
    (0.70-2.50 x 0.02 of the wall extent, the map's roof shape kept); domes by the dome's height (0.3-4.0 x 0.05). A
    box sample searches a window around its start, moved while the best sits on the window's edge. Also the share of
    interior cells under the fitted surface by more than TOL (below_share)."""
    kind = kind or part_kind(b)
    c, z2, q, _, g = exposed_cells(model, b, P)
    out: dict[str, Any] = {"cells": int(len(c)), "kind": kind}
    if len(c) < 15:
        out["why"] = f"{len(c)} cells with >= 3 points (< 15)"
        return out
    cc = c.mean(0)
    out["quadrant_cells"] = [int(np.sum((np.sign(c[:, 0] - cc[0]) == sx) & (np.sign(c[:, 1] - cc[1]) == sy)))
                             for sx in (-1, 1) for sy in (-1, 1)]
    (e0, e1, es), (r0, r1, rs) = FIT_GRIDS["pitched" if kind == "pitched" else kind if kind in FIT_GRIDS else "flat"]
    EMg, RMg = _mgrid(e0, e1, es), _mgrid(r0, r1, rs)
    if start is None:
        em, rm = EMg, RMg
    else:
        i0 = int(np.argmin(np.abs(np.array(EMg) - start[0])))
        j0 = int(np.argmin(np.abs(np.array(RMg) - start[1])))
        em, rm = EMg[max(0, i0 - window):i0 + window + 1], RMg[max(0, j0 - window):j0 + window + 1]
    while True:
        Sin, Sab, tops, eaves, combos = [], [], [], [], []
        for e in em:
            for r in rm:
                bb = {x.id: x for x in model.rebuild({b.id: {"eave": e, "roof": r}}, ids=[b.id], closed=False)}[b.id]
                s = bb.surface_z(c)
                res = z2 - s
                fin = np.isfinite(res)
                Sin.append((np.abs(res) <= FIT_TAU) & fin)
                Sab.append((res > FIT_TAU) & fin)
                tops.append(bb.top)
                eaves.append(bb.eave)
                combos.append((e, r))
        Sin, Sab = np.array(Sin), np.array(Sab)
        score = Sin.sum(1) - Sab.sum(1)
        k = int(np.argmax(score))
        e, r = combos[k]
        if start is not None:
            grow = False
            if e == em[0] and em[0] > EMg[0]:
                i = EMg.index(em[0]); em = EMg[max(0, i - window):EMg.index(em[-1]) + 1]; grow = True
            if e == em[-1] and em[-1] < EMg[-1]:
                i = EMg.index(em[-1]); em = EMg[EMg.index(em[0]):i + window + 1]; grow = True
            if r == rm[0] and rm[0] > RMg[0]:
                i = RMg.index(rm[0]); rm = RMg[max(0, i - window):RMg.index(rm[-1]) + 1]; grow = True
            if r == rm[-1] and rm[-1] < RMg[-1]:
                i = RMg.index(rm[-1]); rm = RMg[RMg.index(rm[0]):i + window + 1]; grow = True
            if grow:
                continue
        break
    rng = np.random.default_rng(0)
    n = len(c)
    bt, be = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        kk = int(np.argmax(Sin[:, idx].sum(1) - Sab[:, idx].sum(1)))
        bt.append(tops[kk])
        be.append(eaves[kk])
    at_bound = (len(EMg) > 1 and e in (EMg[0], EMg[-1])) or (len(RMg) > 1 and r in (RMg[0], RMg[-1]))
    bb = {x.id: x for x in model.rebuild({b.id: {"eave": e, "roof": r}}, ids=[b.id], closed=False)}[b.id]
    out.update({"top": round(tops[k], 3), "eave": round(eaves[k], 3), "mult": [e, r], "inliers": int(Sin[k].sum()),
                "above": int(Sab[k].sum()), "inlier_share": round(float(Sin[k].mean()), 3),
                "top_p05": round(float(np.quantile(bt, 0.05)), 3), "top_p95": round(float(np.quantile(bt, 0.95)), 3),
                "eave_p05": round(float(np.quantile(be, 0.05)), 3), "eave_p95": round(float(np.quantile(be, 0.95)), 3),
                "at_bound": bool(at_bound)})
    ch, zh = cell_highest(q, g)
    out.update(below_share(b, ch, zh, bb.surface_z(ch), tol_of(b.top)))
    return out


def fit_usable(fit: dict[str, Any], map_top: float) -> str | None:
    """Why a roof-surface fit (sfm_fit) cannot be read as the part's own roof, or None when it can: inlier share
    >= 0.4, >= 30 cells, >= 3 quadrants with >= 3 cells each, off its grid bounds, 90 % interval <= PREC (map_top: the
    value judged against) and at most 25 % of the interior cells a TOL under the fitted roof."""
    qc = fit.get("quadrant_cells") or []
    if fit.get("top") is None:
        return "no roof-surface fit"
    if float(fit.get("inlier_share") or 0) < SFM_FIT_INLIERS:
        return f"inlier share {fit.get('inlier_share')} < {SFM_FIT_INLIERS}"
    if int(fit.get("cells") or 0) < SFM_FIT_CELLS:
        return f"{fit.get('cells')} cells < {SFM_FIT_CELLS}"
    if sum(1 for n in qc if n >= SFM_FIT_QCELLS) < 3:
        return f"cells in < 3 quadrants with >= {SFM_FIT_QCELLS} each ({qc})"
    if fit.get("at_bound"):
        return "at its grid bound"
    w = float(fit["top_p95"]) - float(fit["top_p05"])
    if w > prec_of(float(fit["top"]) - float(map_top)) + 1e-9:
        return f"interval {w:.2f} m > PREC"
    if fit.get("below_share") is None:
        return "no share of cells under the fitted roof measured"
    if float(fit["below_share"]) > SFM_FIT_BELOW_MAX + 1e-9:
        return (f"{float(fit['below_share']):.0%} of its {fit.get('below_over', 'interior')} cells lie more than TOL under "
                f"the fitted roof (> {SFM_FIT_BELOW_MAX:.0%}): a lower roof is seen under what the points' top reads")
    return None
