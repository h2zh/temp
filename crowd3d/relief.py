"""Facade relief measured from the photos (docs/STRUCTURE_FIRST.md: photo-measured structure on the OSM frame).

What it adds to the toy model: on a PHOTOGRAPHED wall of the OSM frame, the bands, setbacks, pilasters and openings
(arches, doors, windows) the photos measure, as low-poly geometry with one provenance record per element
("photo-measured"). Nothing is drawn on a wall nobody photographed, nothing is copied to another wall (a measured
rhythm stays on the wall it was measured on), and nothing comes from a single image, an orthophoto alone or a
generative model.

Measurement, per photographed wall, in the wall's own frame (u along the wall from its start a, d outward from the
model wall plane, z up), from two channels:
  P  SfM points (track >= 3) near the wall plane -> a relief map rho(u, z) on 0.2 m cells (median d per cell, relative
     to the wall's photographed front surface, fitted with a yaw term), snapped to the wall's own relief levels
     (histogram peaks) and cleaned;
  L  3-D lines (crowd3d.lines3d: multi-view triangulated, verified, SfM-supported) near the plane. Horizontal lines
     confirm edge heights (their depth is poorly determined), vertical lines positions along the wall.
Decomposition (points): the wall's FRAME level F = the mode over rows of each row's front surface (q90 of the
cells); every cell is proud (>= F + lmin), frame, or recessed (<= F - lmin):
  bands      rows >= 60 % proud: cornices, string courses, plinths; rows >= 97 % recessed: setbacks (attics) and
             grooves; edges refined on a 0.1 m profile;
  registers  the z-ranges between them, each with its own frame level F_r;
  verticals  per register, column strips >= 60 % proud (pilasters, buttresses) or frame strips between recesses;
  openings   per register, connected recesses (not full width): windows, doors, arches, blind arcades;
             rectangle or round-arched top; depth = F_r - median inside; nested deeper recesses inside them.
Element edges are refined below the cell size from the raw points.

Guards (measure_relief): an element becomes geometry only when
  G-R1  its wall has >= 2000 points, >= 15 per m2, data on >= 20 % of its cells, the photographed front surface
        within 0.6 m of the frame wall, and >= 6 photos seeing the wall in each photo half;
  G-R2  it is found again (bbox IoU >= 0.3, same type and sign) in EACH of two disjoint photo halves (cameras sorted
        by bearing around the building, alternately), from points RE-TRIANGULATED from that half's observations
        only, with the two halves' depths within max(0.1 m, 50 %): the two-channel test;
  G-R3  >= 1 of its edges lies at a 3-D line cluster (<= 0.2 m);
  G-R4  its size is inside the detector's limits and its depth / projection <= 1.5 m;
  G-R5  >= 50 % of its box lies on observed texels of the wall's facade;
  and a nested opening only inside an accepted parent.
The QA (check_honesty, audit_model, audit_glb in crowd3d.stylize) re-checks the records, the geometry and, from the
raw points, every built element's depth."""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import cv2


@dataclass(frozen=True)
class ReliefParams:
    window_m: float = 2.0            # points within this of the model wall plane (outward / inward)
    min_track: int = 3
    ref_bin_m: float = 0.05
    ref_tol_m: float = 0.15
    max_ref_offset_m: float = 0.6    # the photographed frame must lie within this of the model wall
    min_points: int = 2000           # points on the wall (below: refused, too sparse for relief)
    min_density: float = 15.0        # points per m2 of wall
    top_extra_m: float = 1.5         # look this far above the wall top (crowning cornice)
    # relief map
    cell_m: float = 0.2
    cell_min_pts: int = 3
    level_min_m: float = 0.08        # relief step that counts as structure
    level_sigma_k: float = 2.5       # ... and at least this x the cell noise
    front_q: float = 90.0            # a row's front surface
    quantize: bool = True            # snap cells to the wall's relief levels (histogram peaks) and clean the labels
    level_sep_m: float = 0.12        # relief levels closer than this merge
    level_min_mass: float = 0.02     # a level holds at least this share of the wall's cells
    max_levels: int = 5
    min_region_m2: float = 0.5       # label regions smaller than this join their surroundings
    # bands
    row_min_cover: float = 0.35      # share of the wall's columns with data for a row to count
    band_proud_share: float = 0.6
    band_rec_share: float = 0.97     # a setback is recessed (nearly) full width: no frame columns left
    min_data_share: float = 0.2      # share of the wall's cells with data (below: too sparse for relief)
    frame_share: float = 0.7         # a frame strip between recesses: >= this share of its cells not recessed
    recess_min_share: float = 0.25   # ... in a register with at least this share of recessed cells
    band_h_max: float = 2.0          # taller projecting zones are 'projecting-zone'
    setback_h_min: float = 1.0
    register_min_h: float = 1.0
    # verticals
    col_min_cover: float = 0.5
    vert_share: float = 0.6
    vert_w: tuple = (0.2, 2.5)
    vert_min_h: float = 1.0
    # openings
    open_min_area: float = 0.4
    open_w: tuple = (0.4, 8.0)
    open_h: tuple = (0.6, 30.0)
    open_min_fill: float = 0.5       # area / bounding box
    open_min_data: float = 0.6       # share of the bounding box's cells with data
    open_max_width_share: float = 0.9
    arch_min_rise: float = 0.3
    # lines
    line_deg: float = 5.0
    line_d_max: float = 1.2
    line_bw_m: float = 0.05
    line_min_photos: int = 3
    strong_cover: float = 0.5        # band edges: horizontal clusters over >= half the wall
    strong_photos: int = 5
    local_cover: float = 0.25        # opening / pilaster edges: clusters over >= 25 % of the edge
    edge_tol_m: float = 0.2


DEFAULT_RELIEF = ReliefParams()


# ---------------------------------------------------------------- frames, reference


def to_wall(a2, t2, n2, X):
    a = np.r_[a2, 0.0]
    t = np.r_[t2, 0.0]
    n = np.r_[n2, 0.0]
    r = np.asarray(X, float) - a
    return np.stack([r @ t, r @ n, np.asarray(X, float)[:, 2]], 1)


def wall_points(wall, P, track, prm: ReliefParams = DEFAULT_RELIEF, others=None):
    """Points near the wall plane; with `others` (the other walls of the massing), only the points nearer to this
    wall's segment than to any other wall that spans their height (a convex corner's neighbouring face would read
    as a recess)."""
    U = to_wall(wall.a, wall.tangent, wall.normal, P)
    m = ((U[:, 0] > -0.2) & (U[:, 0] < wall.length + 0.2) & (np.abs(U[:, 1]) < prm.window_m)
         & (U[:, 2] > wall.z0 - 0.3) & (U[:, 2] < wall.z1 + prm.top_extra_m) & (track >= prm.min_track))
    idx = np.nonzero(m)[0]
    if others and len(idx):
        xy = np.asarray(P, float)[idx, :2]
        zz = np.asarray(P, float)[idx, 2]

        def segd(a, b):
            ab = b - a
            t = np.clip(((xy - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            return np.linalg.norm(xy - (a + t[:, None] * ab), axis=1)

        own = segd(wall.a, wall.b)
        keep = np.ones(len(idx), bool)
        for o in others:
            if o.index == wall.index:
                continue
            if np.linalg.norm(0.5 * (o.a + o.b) - 0.5 * (wall.a + wall.b)) > wall.length + o.length + 4:
                continue
            span = (zz >= o.z0 - 0.3) & (zz <= o.z1 + prm.top_extra_m)
            keep &= ~(span & (segd(o.a, o.b) < own - 1e-6))
        idx = idx[keep]
    return U[idx], idx


def reference_plane(U, prm: ReliefParams = DEFAULT_RELIEF):
    """The densest layer of points, refined by a robust fit d = c0 + c1 u (yaw)."""
    if len(U) < 50:
        return None
    h, e = np.histogram(U[:, 1], bins=np.arange(-prm.window_m, prm.window_m + 1e-9, prm.ref_bin_m))
    hs = np.convolve(h, [1, 2, 3, 2, 1], 'same')
    c0, c1 = 0.5 * (e[hs.argmax()] + e[hs.argmax() + 1]), 0.0
    for _ in range(4):
        r = U[:, 1] - (c0 + c1 * U[:, 0])
        inl = np.abs(r) < prm.ref_tol_m
        if inl.sum() < 30:
            break
        A = np.c_[np.ones(inl.sum()), U[inl, 0]]
        c0, c1 = np.linalg.lstsq(A, U[inl, 1], rcond=None)[0]
    r = U[:, 1] - (c0 + c1 * U[:, 0])
    return {"c0": float(c0), "c1": float(c1), "yaw_deg": float(np.degrees(np.arctan(c1))),
            "share": float((np.abs(r) < prm.ref_tol_m).mean()), "points": int(len(U))}


def _group_median(keys, vals, nkeys):
    out = np.full(nkeys, np.nan)
    cnt = np.zeros(nkeys, int)
    if not len(keys):
        return out, cnt
    o = np.argsort(keys, kind='stable')
    k, v = keys[o], vals[o]
    uk, st = np.unique(k, return_index=True)
    ends = np.r_[st[1:], len(k)]
    for kk, s0, s1 in zip(uk, st, ends):
        out[kk] = np.median(v[s0:s1])
        cnt[kk] = s1 - s0
    return out, cnt


def relief_map(u, rho, z, L, zlo, zhi, cell, min_pts):
    """(nz, nu) median relief per cell, row 0 at zhi (top); counts."""
    nu = max(1, int(math.ceil(L / cell)))
    nz = max(1, int(math.ceil((zhi - zlo) / cell)))
    m = (u >= 0) & (u < L) & (z >= zlo) & (z < zhi)
    iu = np.clip((u[m] / cell).astype(int), 0, nu - 1)
    iz = np.clip(((zhi - z[m]) / cell).astype(int), 0, nz - 1)
    med, cnt = _group_median(iz * nu + iu, rho[m], nz * nu)
    med = med.reshape(nz, nu)
    cnt = cnt.reshape(nz, nu)
    med[cnt < min_pts] = np.nan
    return med, cnt


def nan_median3(M):
    """3 x 3 nan-aware median (>= 3 finite values); an empty cell is filled only when >= 5 neighbours have data."""
    P = np.pad(M, 1, constant_values=np.nan)
    S = np.stack([P[i:i + M.shape[0], j:j + M.shape[1]] for i in range(3) for j in range(3)], 0)
    n = np.isfinite(S).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.nanmedian(S, 0)
    out[n < 3] = np.nan
    out[~np.isfinite(M) & (n < 5)] = np.nan
    return out


def cell_sigma(M):
    d = np.diff(M, axis=1).ravel()
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return 0.08
    return float(1.4826 * np.median(np.abs(d)) / math.sqrt(2))


def weighted_mode(vals, wts, bin_m=0.02):
    ok = np.isfinite(vals) & (wts > 0)
    if not ok.any():
        return float('nan')
    v, w = vals[ok], wts[ok]
    lo, hi = v.min() - 2 * bin_m, v.max() + 2 * bin_m
    h, e = np.histogram(v, bins=np.arange(lo, hi + bin_m, bin_m), weights=w)
    hs = np.convolve(h, [1, 2, 3, 2, 1], 'same')
    k = int(hs.argmax())
    near = np.abs(v - 0.5 * (e[k] + e[k + 1])) <= 2.5 * bin_m
    return float(np.average(v[near], weights=w[near]))


def row_front(M, q):
    out = np.full(M.shape[0], np.nan)
    cov = np.isfinite(M).mean(1)
    for i in range(M.shape[0]):
        f = M[i][np.isfinite(M[i])]
        if len(f) >= 2:
            out[i] = np.percentile(f, q)
    return out, cov


def fine_front(u, rho, z, L, zlo, zhi, q, row=0.1, col=0.4, min_pts=2):
    """Front-surface profile on 0.1 m rows (for sub-cell edge positions): q-th percentile of 0.4 m column medians."""
    nz = max(1, int(math.ceil((zhi - zlo) / row)))
    nc = max(1, int(math.ceil(L / col)))
    m = (u >= 0) & (u < L) & (z >= zlo) & (z < zhi)
    iz = np.clip(((z[m] - zlo) / row).astype(int), 0, nz - 1)
    ic = np.clip((u[m] / col).astype(int), 0, nc - 1)
    med, cnt = _group_median(iz * nc + ic, rho[m], nz * nc)
    med = med.reshape(nz, nc)
    med[cnt.reshape(nz, nc) < min_pts] = np.nan
    prof = np.full(nz, np.nan)
    for i in range(nz):
        f = med[i][np.isfinite(med[i])]
        if len(f) >= 2:
            prof[i] = np.percentile(f, q)
    return zlo + (np.arange(nz) + 0.5) * row, prof


def refine_edge(zf, pf, z_guess, half, win=0.35):
    """Where the fine profile crosses `half` nearest to z_guess (None when it does not within win)."""
    best = None
    for a in range(len(zf) - 1):
        if abs(zf[a] - z_guess) > win and abs(zf[a + 1] - z_guess) > win:
            continue
        y0, y1 = pf[a], pf[a + 1]
        if np.isfinite(y0) and np.isfinite(y1) and (y0 - half) * (y1 - half) <= 0 and y0 != y1:
            zz = float(zf[a] + (half - y0) / (y1 - y0) * (zf[a + 1] - zf[a]))
            if best is None or abs(zz - z_guess) < abs(best - z_guess):
                best = zz
    return best


# ---------------------------------------------------------------- 3-D line clusters


def wall_lines(wall, lines, ref, prm: ReliefParams = DEFAULT_RELIEF):
    if lines is None or not len(lines):
        return None
    A = to_wall(wall.a, wall.tangent, wall.normal, lines.A)
    B = to_wall(wall.a, wall.tangent, wall.normal, lines.B)
    for X in (A, B):
        X[:, 1] -= ref["c0"] + ref["c1"] * X[:, 0]
    D = B - A
    ln = np.linalg.norm(D, axis=1)
    mid = 0.5 * (A + B)
    near = ((np.abs(mid[:, 1]) < prm.line_d_max) & (np.maximum(A[:, 0], B[:, 0]) > 0) & (np.minimum(A[:, 0], B[:, 0]) < wall.length)
            & (np.maximum(A[:, 2], B[:, 2]) > wall.z0 - 0.3) & (np.minimum(A[:, 2], B[:, 2]) < wall.z1 + prm.top_extra_m))
    c = math.cos(math.radians(prm.line_deg))
    hor = near & (np.abs(D[:, 0]) / np.maximum(ln, 1e-9) > c)
    ver = near & (np.abs(D[:, 2]) / np.maximum(ln, 1e-9) > c)
    return {"A": A, "B": B, "mid": mid, "len": ln, "hor": hor, "ver": ver, "photos": lines.photos,
            "n_images": lines.n_images, "near": near}


def _union_len(iv):
    if not iv:
        return 0.0
    iv = sorted(iv)
    tot, (s, e) = 0.0, iv[0]
    for a, b in iv[1:]:
        if a > e:
            tot += e - s
            s, e = a, b
        else:
            e = max(e, b)
    return tot + e - s


def line_clusters(WL, kind, span_lo, span_hi, pos_lo, pos_hi, min_cover, min_photos, prm: ReliefParams = DEFAULT_RELIEF):
    """Clusters of horizontal (kind 'h': position z, extent along u) or vertical (kind 'v': position u, extent
    along z) lines; cover = union of the lines' extents inside [span_lo, span_hi] / its length."""
    if WL is None:
        return []
    m = WL["hor"] if kind == "h" else WL["ver"]
    A, B, mid = WL["A"], WL["B"], WL["mid"]
    if kind == "h":
        pos, e0, e1 = mid[:, 2], np.minimum(A[:, 0], B[:, 0]), np.maximum(A[:, 0], B[:, 0])
    else:
        pos, e0, e1 = mid[:, 0], np.minimum(A[:, 2], B[:, 2]), np.maximum(A[:, 2], B[:, 2])
    e0, e1 = np.clip(e0, span_lo, span_hi), np.clip(e1, span_lo, span_hi)
    m = m & (pos >= pos_lo) & (pos <= pos_hi) & (e1 - e0 > 0.05)
    idx = np.nonzero(m)[0]
    if not len(idx):
        return []
    grid = np.arange(pos_lo, pos_hi + 1e-9, 0.01)
    wts = e1[idx] - e0[idx]
    dens = (wts[None, :] * np.exp(-0.5 * ((grid[:, None] - pos[idx][None, :]) / prm.line_bw_m) ** 2)).sum(1)
    peaks = [i for i in range(1, len(grid) - 1) if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1] and dens[i] > 0]
    out = []
    span = max(span_hi - span_lo, 1e-9)
    for i in sorted(peaks, key=lambda i: -dens[i]):
        g = grid[i]
        if any(abs(c["pos"] - g) < 2.5 * prm.line_bw_m for c in out):
            continue
        mem = idx[np.abs(pos[idx] - g) <= 2.5 * prm.line_bw_m]
        if not len(mem):
            continue
        cover = _union_len([(e0[k], e1[k]) for k in mem]) / span
        photos = sorted({p for k in mem for p in WL["photos"][k]})
        w_ = e1[mem] - e0[mem]
        pm = float(np.average(pos[mem], weights=w_))
        out.append({"pos": pm, "rho": float(np.median(mid[mem, 1])), "cover": float(cover), "photos": photos,
                    "n_photos": len(photos), "lines": [int(k) for k in mem], "n_lines": int(len(mem)),
                    "spread": float(np.sqrt(np.average((pos[mem] - pm) ** 2, weights=w_)))})
    out = [c for c in out if c["n_photos"] >= min_photos and c["cover"] >= min_cover]
    return sorted(out, key=lambda c: c["pos"])


def nearest_cluster(clusters, pos, tol):
    best = None
    for c in clusters:
        dd = abs(c["pos"] - pos)
        if dd <= tol and (best is None or dd < abs(best["pos"] - pos)):
            best = c
    return best


def chance_rate(clusters, lo, hi, tol):
    """Share of positions in [lo, hi) within tol of a cluster: how often a random edge would be 'confirmed'."""
    if not clusters or hi <= lo:
        return 0.0
    zz = np.arange(lo, hi, 0.01)
    P = np.array([c["pos"] for c in clusters])
    return float(np.mean(np.min(np.abs(zz[:, None] - P[None, :]), axis=1) <= tol))


def _edge_rec(c, pos):
    if c is None:
        return None
    return {"pos": round(c["pos"], 3), "delta": round(c["pos"] - pos, 3), "cover": round(c["cover"], 3),
            "photos": c["n_photos"], "lines": c["n_lines"], "photo_ids": c["photos"], "line_ids": c["lines"]}


def _confirm(el, edges, rule):
    el["edges"] = edges
    real = [e for e in edges.values() if not e["limit"]]
    el["edges_real"] = len(real)
    el["edges_confirmed"] = sum(1 for e in real if e["line"])
    el["two_channel"] = bool(rule(edges, real))
    el["photos"] = sorted({p for e in real if e["line"] for p in e["line"]["photo_ids"]})
    el["line_ids"] = sorted({k for e in real if e["line"] for k in e["line"]["line_ids"]})


# ---------------------------------------------------------------- the wall


@dataclass
class WallRelief:
    wall: int
    part: str | None
    length: float
    z0: float
    z1: float
    ref: dict | None
    refused: str = ""
    frame: float = 0.0               # frame level relative to the point reference (m, outward +)
    bands: list = field(default_factory=list)
    registers: list = field(default_factory=list)
    verticals: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    maps: dict = field(default_factory=dict)

    def elements(self):
        return [*self.bands, *self.verticals, *self.openings]

    def to_json(self):
        return {"wall": self.wall, "part": self.part, "length": round(self.length, 3), "z0": self.z0, "z1": self.z1,
                "ref": self.ref, "refused": self.refused, "frame": round(self.frame, 3), "bands": self.bands,
                "registers": self.registers, "verticals": self.verticals, "openings": self.openings, "stats": self.stats}


def _runs(labels, value):
    out, k = [], 0
    while k < len(labels):
        if labels[k] != value:
            k += 1
            continue
        j = k
        while j + 1 < len(labels) and labels[j + 1] == value:
            j += 1
        out.append((k, j + 1))
        k = j + 1
    return out


def measure_wall(wall, points, track, lines=None, prm: ReliefParams = DEFAULT_RELIEF, others=None) -> WallRelief:
    t0 = time.perf_counter()
    WR = WallRelief(wall.index, getattr(wall, 'part', None), float(wall.length), float(wall.z0), float(wall.z1), None)
    U, idx = wall_points(wall, points, track, prm, others)
    WR.stats["points"] = int(len(U))
    ref = reference_plane(U, prm)
    WR.ref = ref
    area = wall.length * (wall.z1 - wall.z0)
    if ref is None or len(U) < prm.min_points or len(U) < prm.min_density * area:
        WR.refused = f"too few SfM points for relief ({len(U)}, {len(U) / max(area, 1e-9):.1f} per m2)"
        WR.stats["seconds"] = round(time.perf_counter() - t0, 3)
        return WR
    u, z = U[:, 0], U[:, 2]
    L = wall.length
    cell = prm.cell_m
    zlo, zhi = wall.z0, wall.z1 + prm.top_extra_m
    # yaw from the FRONT surface: per column the q90 of its cells (frame columns span the wall; a yaw fitted to
    # the densest layer tilts a two-level facade), robust line over the columns
    rho0 = U[:, 1] - ref["c0"]
    raw0, _ = relief_map(u, rho0, z, L, zlo, wall.z1, cell, prm.cell_min_pts)
    colq = np.array([np.percentile(c[np.isfinite(c)], prm.front_q) if np.isfinite(c).sum() >= 5 else np.nan for c in raw0.T])
    uc = (np.arange(len(colq)) + 0.5) * cell
    ok = np.isfinite(colq)
    if ok.sum() >= 6:
        rng = np.random.default_rng(0)
        best, bi = None, -1
        cu, cv_ = uc[ok], colq[ok]
        for _ in range(200):
            i, j = rng.choice(len(cu), 2, replace=False)
            if abs(cu[j] - cu[i]) < 0.25 * L:
                continue
            sl = (cv_[j] - cv_[i]) / (cu[j] - cu[i])
            inl = np.abs(cv_ - (cv_[i] + sl * (cu - cu[i]))) < 0.05
            if inl.sum() > bi:
                bi, best = int(inl.sum()), inl
        if best is not None and best.sum() >= 4:
            A = np.c_[np.ones(best.sum()), cu[best]]
            k0, k1 = np.linalg.lstsq(A, cv_[best], rcond=None)[0]
            if abs(np.degrees(np.arctan(k1))) <= 5.0:
                ref["c0_front"], ref["c1"] = float(ref["c0"] + k0), float(k1)
                ref["c0"] = float(ref["c0"] + k0 - (k0 - np.median(cv_[best] - k1 * cu[best])))
                ref["yaw_deg"] = float(np.degrees(np.arctan(k1)))
                ref["yaw_from"] = f"front surface, {int(best.sum())}/{int(ok.sum())} columns"
    rho = U[:, 1] - (ref["c0"] + ref["c1"] * u)
    raw, cnt = relief_map(u, rho, z, L, zlo, zhi, cell, prm.cell_min_pts)
    P3 = (u, rho, z)
    sig = cell_sigma(raw)
    M = nan_median3(raw)
    levels = None
    if prm.quantize:
        M, levels = quantize_levels(M, prm, cell)
    nz, nu = M.shape
    zrow = zhi - (np.arange(nz) + 0.5) * cell             # cell row centres (row 0 at the top)
    lmin = max(prm.level_min_m, prm.level_sigma_k * sig)
    front, cov = row_front(M, prm.front_q)
    inwall = (zrow < wall.z1) & (cov >= prm.row_min_cover)
    F = weighted_mode(front[inwall], cov[inwall])
    if not np.isfinite(F):
        WR.refused = "no rows with enough data"
        WR.stats["seconds"] = round(time.perf_counter() - t0, 3)
        return WR
    WR.frame = F
    off = ref["c0"] + ref["c1"] * L / 2 + F
    WR.stats["levels"] = None if levels is None else [round(float(x), 3) for x in levels]
    WR.stats.update({"sigma_cell": round(sig, 4), "level_min": round(lmin, 3), "frame_offset_m": round(off, 3),
                     "rows": nz, "cols": nu, "cells_with_data": int(np.isfinite(M).sum()),
                     "data_share_in_wall": round(float(np.isfinite(M[zrow < wall.z1]).mean()), 3)})
    if WR.stats["data_share_in_wall"] < prm.min_data_share:
        WR.refused = f"relief data on {WR.stats['data_share_in_wall']:.0%} of the wall's cells (< {prm.min_data_share:.0%})"
        WR.stats["seconds"] = round(time.perf_counter() - t0, 3)
        return WR
    if abs(off) > prm.max_ref_offset_m:
        WR.refused = f"the photographed frame stands {off:+.2f} m off the frame wall"
        WR.stats["seconds"] = round(time.perf_counter() - t0, 3)
        return WR
    WL = wall_lines(wall, lines, ref, prm)
    zf, pf = fine_front(u, rho, z, L, zlo, zhi, prm.front_q)
    # ---------------- bands (rows)
    R = M - F
    lab = np.where(np.isfinite(R), np.where(R >= lmin, 1, np.where(R <= -lmin, -1, 0)), -9)
    nd = (lab != -9).sum(1)
    share_p = (lab == 1).sum(1) / np.maximum(nd, 1)
    share_r = (lab == -1).sum(1) / np.maximum(nd, 1)
    rowlab = np.where(cov < prm.row_min_cover, 9, np.where(share_p >= prm.band_proud_share, 1,
                                                            np.where(share_r >= prm.band_rec_share, -1, 0)))
    strong = line_clusters(WL, "h", 0.0, L, zlo - 0.2, zhi, prm.strong_cover, prm.strong_photos, prm)
    WR.stats["strong_hclusters"] = [round(c["pos"], 3) for c in strong]
    WR.stats["chance_hedge"] = round(chance_rate(strong, wall.z0, wall.z1, prm.edge_tol_m), 3)
    data_rows = np.nonzero(cov >= prm.row_min_cover)[0]
    ztop_data = zrow[data_rows[0]] + cell / 2 if len(data_rows) else zhi
    zbot_data = zrow[data_rows[-1]] - cell / 2 if len(data_rows) else zlo
    band_rows = np.zeros(nz, bool)

    def neighbour_level(rr):
        if 0 <= rr < nz and rowlab[rr] in (0, 1, -1):
            f = front[rr] - F
            return float(f) if np.isfinite(f) else 0.0
        return 0.0

    for sgn in (1, -1):
        for r0, r1 in _runs(list(rowlab), sgn):            # rows r0..r1-1, r0 at the top
            zt, zb = float(zrow[r0] + cell / 2), float(zrow[r1 - 1] - cell / 2)
            h = zt - zb
            cells = R[r0:r1][lab[r0:r1] == sgn]
            level = float(np.median(cells)) if len(cells) else 0.0
            if sgn > 0:
                kind = "band" if h <= prm.band_h_max else "projecting-zone"
                if zb <= wall.z0 + 0.15 and h <= 1.2:
                    kind = "plinth"
            else:
                kind = "setback" if h >= prm.setback_h_min else "groove"
            zt_r = refine_edge(zf, pf - F, zt, 0.5 * (level + neighbour_level(r0 - 1)))
            zb_r = refine_edge(zf, pf - F, zb, 0.5 * (level + neighbour_level(r1)))
            el = {"type": "band", "kind": kind, "sign": sgn, "u0": 0.0, "u1": round(L, 3),
                  "z_bottom": round(zb if zb_r is None else zb_r, 3), "z_top": round(zt if zt_r is None else zt_r, 3),
                  "z_bottom_cell": round(zb, 3), "z_top_cell": round(zt, 3), "level": round(level, 3),
                  "share": round(float((share_p if sgn > 0 else share_r)[r0:r1].mean()), 3),
                  "cover": round(float(cov[r0:r1].mean()), 3), "rows": int(r1 - r0),
                  "above_wall_top": zb >= wall.z1 - 0.05}
            el["height"] = round(el["z_top"] - el["z_bottom"], 3)
            if el["height"] < 0.1:                       # refined edges crossed: not a band
                continue
            edges = {}
            for name, ez, limit in (("bottom", el["z_bottom"], zb <= max(wall.z0 + 0.15, zbot_data + 0.05)),
                                    ("top", el["z_top"], zt >= ztop_data - 0.05)):
                c = None if limit else nearest_cluster(strong, ez, prm.edge_tol_m)
                edges[name] = {"pos": round(ez, 3), "limit": bool(limit), "line": _edge_rec(c, ez)}
            _confirm(el, edges, lambda e, real: bool(real) and all(x["line"] for x in real))
            WR.bands.append(el)
            if sgn > 0 or kind == "groove":
                band_rows[r0:r1] = True
    WR.bands.sort(key=lambda b: b["z_bottom"])
    # ---------------- registers: between projecting bands / grooves, inside the wall; a setback is its own register
    zr = []
    cur = None
    for i in range(nz - 1, -1, -1):                      # bottom to top
        inside = zrow[i] < wall.z1 and not band_rows[i]
        if inside:
            cur = [i, i] if cur is None else [i, cur[1]]
        if (not inside or i == 0) and cur is not None:
            zr.append((cur[0], cur[1] + 1))              # (top row, bottom row + 1)
            cur = None
    for r0, r1 in zr:
        sub, s0 = [], r0
        for r in range(r0 + 1, r1):
            if (rowlab[r] == -1) != (rowlab[r - 1] == -1):
                sub.append((s0, r))
                s0 = r
        sub.append((s0, r1))
        for a, b in sub:
            zt_, zb_ = float(zrow[a] + cell / 2), float(zrow[b - 1] - cell / 2)
            if zt_ - zb_ < prm.register_min_h:
                continue
            fr, cv = front[a:b], cov[a:b]
            ok = cv >= prm.row_min_cover
            if ok.sum() < 3:
                continue
            Fr = weighted_mode(fr[ok] - F, cv[ok])               # relative to the wall frame F
            WR.registers.append({"z_bottom": round(zb_, 3), "z_top": round(zt_, 3), "rows": [int(a), int(b)],
                                 "frame": round(Fr, 3), "setback": bool((rowlab[a:b] == -1).mean() > 0.5)})
    WR.registers.sort(key=lambda r: r["z_bottom"])
    # ---------------- verticals and openings per register
    for reg in WR.registers:
        a, b = reg["rows"]
        Rr = M[a:b] - (F + reg["frame"])
        labr = np.where(np.isfinite(Rr), np.where(Rr >= lmin, 1, np.where(Rr <= -lmin, -1, 0)), -9)
        H = reg["z_top"] - reg["z_bottom"]
        vcl = line_clusters(WL, "v", reg["z_bottom"], reg["z_top"], -0.3, L + 0.3, prm.local_cover * 0.5, prm.line_min_photos, prm)
        reg["vclusters"] = [round(c["pos"], 3) for c in vcl]
        reg["chance_vedge"] = round(chance_rate(vcl, 0.0, L, prm.edge_tol_m), 3)
        ndc = (labr != -9).sum(0)
        colcov = ndc / max(b - a, 1)
        colp = (labr == 1).sum(0) / np.maximum(ndc, 1)
        colf = (labr >= 0).sum(0) / np.maximum(ndc, 1)              # frame or proud
        rec_share = float((labr == -1).sum() / max((labr != -9).sum(), 1))
        reg["recess_share"] = round(rec_share, 3)
        isv = (colcov >= prm.col_min_cover) & (colp >= prm.vert_share)
        isf = (colcov >= prm.col_min_cover) & (colf >= prm.frame_share) & ~isv if rec_share >= prm.recess_min_share else np.zeros(len(colp), bool)
        strips = sorted([(c0, c1, "proud") for c0, c1 in _runs(list(isv.astype(int)), 1)] +
                        [(c0, c1, "frame") for c0, c1 in _runs(list(isf.astype(int)), 1)])
        merged = []
        for c0, c1, how in strips:                    # a proud core beside a frame flank is one element
            if merged and c0 <= merged[-1][1]:
                p0, p1, ph = merged[-1]
                merged[-1] = (p0, max(p1, c1), "proud" if "proud" in (ph, how) else "frame")
            else:
                merged.append((c0, c1, how))
        strips = merged
        if H >= prm.vert_min_h:
            for c0, c1, how in strips:
                w_ = (c1 - c0) * cell
                if not (prm.vert_w[0] - 1e-9 <= w_ <= prm.vert_w[1] + 1e-9):
                    continue
                u0, u1 = c0 * cell, c1 * cell
                if how == "proud":
                    cells = Rr[:, c0:c1][labr[:, c0:c1] == 1]
                    proj = float(np.percentile(cells, 75)) if len(cells) else 0.0
                else:                                                   # relief over the recesses beside it
                    side = np.r_[max(0, c0 - 3):c0, c1:min(Rr.shape[1], c1 + 3)]
                    sv = Rr[:, side][labr[:, side] == -1] if len(side) else np.zeros(0)
                    if not len(sv):
                        continue
                    proj = float(-np.median(sv))
                kind = "corner-pier" if (u0 <= 0.25 or u1 >= L - 0.25) else ("buttress" if proj >= 0.5 else "pilaster")
                el = {"type": "vertical", "kind": kind, "how": how, "u0": round(u0, 3), "u1": round(u1, 3),
                      "z_bottom": reg["z_bottom"], "z_top": reg["z_top"], "width": round(w_, 3),
                      "projection": round(proj, 3), "share": round(float((colp if how == "proud" else colf)[c0:c1].mean()), 3),
                      "register_frame": reg["frame"]}
                for key_, lim_ in (("u0", u0 <= 0.25), ("u1", u1 >= L - 0.25)):
                    if lim_:
                        continue
                    r_ = refine_side(P3, "u", el[key_], reg["z_bottom"] + 0.2, reg["z_top"] - 0.2, lmin)
                    if r_ is not None:
                        el[key_ + "_cell"], el[key_] = el[key_], round(r_, 3)
                u0, u1 = el["u0"], el["u1"]
                el["width"] = round(u1 - u0, 3)
                vloc = line_clusters(WL, "v", reg["z_bottom"], reg["z_top"], u0 - 0.5, u1 + 0.5, prm.local_cover, prm.line_min_photos, prm)
                edges = {}
                for name, eu, limit in (("left", u0, u0 <= 0.25), ("right", u1, u1 >= L - 0.25)):
                    c = None if limit else nearest_cluster(vloc, eu, prm.edge_tol_m)
                    edges[name] = {"pos": round(eu, 3), "limit": bool(limit), "line": _edge_rec(c, eu)}
                _confirm(el, edges, lambda e, real: bool(real) and all(x["line"] for x in real))
                el["chance_edge"] = round(chance_rate(vloc, u0 - 0.5, u1 + 0.5, prm.edge_tol_m), 3)
                WR.verticals.append(el)
        # openings: recesses, cut into bays at the vertical strips (a thin pilaster must not join two bays)
        rec = (labr == -1).astype(np.uint8)
        for v in WR.verticals:
            if v["z_bottom"] == reg["z_bottom"] and v["z_top"] == reg["z_top"]:
                rec[:, int(round(v["u0"] / cell)):int(round(v["u1"] / cell))] = 0
        fields = []
        found = _find_openings(rec, Rr, labr, reg, wall, L, cell, lmin, WL, prm, level=1, fields=fields, P3=P3)
        found_base = len(WR.openings)
        WR.openings += found
        reg["fields"] = [[round(x * cell, 2), round((x + ww) * cell, 2), round(reg["z_top"] - (y + hh) * cell, 2),
                          round(reg["z_top"] - y * cell, 2)] for x, y, ww, hh, _ in fields]
        # nested: deeper recesses inside an opening (a door or window inside a blind arch / panel) or a field
        containers = [(int(round(op["u0"] / cell)), int(round(op["u1"] / cell)),
                       int(round((reg["z_top"] - op["z_top"]) / cell)), int(round((reg["z_top"] - op["z_bottom"]) / cell)),
                       -op["depth"], (op["u0"], op["u1"], op["z_bottom"], op["z_top"]), found.index(op), None) for op in found]
        for x, y, ww, hh, fm in fields:
            fv = Rr[y:y + hh, x:x + ww][fm]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fb = float(np.nanmedian(fv))
            containers.append((x, x + ww, y, y + hh, fb, (x * cell, (x + ww) * cell, reg["z_top"] - (y + hh) * cell, reg["z_top"] - y * cell), None, fm))
        for x0, x1, y0, y1, base, pbox, pidx, fm in containers:
            sub = Rr[y0:y1, x0:x1]
            deep = np.zeros(Rr.shape, np.uint8)
            dm = np.isfinite(sub) & (sub <= base - lmin)
            if fm is not None:
                dm &= fm
            deep[y0:y1, x0:x1] = dm.astype(np.uint8)
            Rn = Rr - base                      # relative to the outer recess
            labn = np.where(np.isfinite(Rn), np.where(Rn <= -lmin, -1, 0), -9)
            inner = _find_openings(deep, Rn, labn, reg, wall, L, cell, lmin, WL, prm, level=2, parent=pbox, P3=P3)
            for el in inner:
                el["parent"] = None if pidx is None else found_base + pidx
                el["in_field"] = pidx is None
                el["depth_total"] = round(el["depth"] - base, 3)
            WR.openings += inner
    WR.stats["seconds"] = round(time.perf_counter() - t0, 3)
    WR.maps = {"M": M, "raw": raw, "cnt": cnt, "zrow": zrow, "cell": cell, "F": F, "lmin": lmin, "zhi": zhi,
               "rowlab": rowlab, "front": front, "cov": cov, "WL": WL, "zf": zf, "pf": pf, "idx": idx}
    return WR


def _split_recesses(rec, L, cell, prm, max_w):
    """Connected recesses; a component too wide (or too ragged) for one opening is split by a watershed on its
    distance transform (arches joined by a recessed band below them); what cannot be split is a 'field'."""
    n, cc = cv2.connectedComponents(rec, connectivity=4)
    out = np.zeros(cc.shape, np.int32)
    nxt = 1
    for k in range(1, n):
        m = cc == k
        ys, xs = np.nonzero(m)
        W = (xs.max() - xs.min() + 1) * cell
        fill = m.sum() / float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        if W < max_w and fill >= prm.open_min_fill:
            out[m] = nxt
            nxt += 1
            continue
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        best = None
        for frac in (0.5, 0.4, 0.6, 0.3, 0.7):
            nm, ml = cv2.connectedComponents((dt >= frac * dt.max()).astype(np.uint8), connectivity=4)
            if nm - 1 >= 2:
                best = (nm, ml)
                break
        if best is None:
            out[m] = nxt
            nxt += 1
            continue
        nm, ml = best
        markers = ml.astype(np.int32)
        markers[~m] = nm                                     # background: its own basin
        img = np.zeros(m.shape + (3,), np.uint8)
        img[..., :] = (255 - np.clip(dt / max(dt.max(), 1e-9) * 255, 0, 255)).astype(np.uint8)[..., None]
        cv2.watershed(img, markers)
        for j in range(1, nm):
            sub = (markers == j) & m
            if sub.any():
                out[sub] = nxt
                nxt += 1
    return out


def _find_openings(rec, Rr, labr, reg, wall, L, cell, lmin, WL, prm, level=1, parent=None, fields=None, P3=None):
    out = []
    pw = None if parent is None else (parent[1] - parent[0])
    max_w = prm.open_max_width_share * L if pw is None else 0.95 * pw
    lab_img = _split_recesses(rec, L, cell, prm, max_w)
    for k in range(1, int(lab_img.max()) + 1):
        msk = lab_img == k
        if not msk.any():
            continue
        ys_, xs_ = np.nonzero(msk)
        x, y = int(xs_.min()), int(ys_.min())
        ww, hh = int(xs_.max() - x + 1), int(ys_.max() - y + 1)
        ar = int(msk.sum())
        area = ar * cell * cell
        W_, Hh = ww * cell, hh * cell
        if W_ >= max_w and fields is not None and area >= prm.open_min_area:
            fields.append((x, y, ww, hh, msk[y:y + hh, x:x + ww]))   # a recessed field: search inside it only
            continue
        if area < prm.open_min_area or not (prm.open_w[0] <= W_ <= prm.open_w[1]) or not (prm.open_h[0] <= Hh <= prm.open_h[1]):
            continue
        if W_ >= max_w:
            continue
        comp = msk[y:y + hh, x:x + ww]
        fill = float(comp.mean())
        datash = float((labr[y:y + hh, x:x + ww] != -9).mean())
        if fill < prm.open_min_fill or datash < prm.open_min_data:
            continue
        u0, u1 = x * cell, (x + ww) * cell
        zt_o = reg["z_top"] - y * cell
        zb_o = reg["z_top"] - (y + hh) * cell
        inside = Rr[y:y + hh, x:x + ww][comp]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            depth = float(-np.nanmedian(inside))
        tops = np.array([zt_o - (np.argmax(comp[:, j]) if comp[:, j].any() else hh) * cell for j in range(ww)])
        uc = 0.5 * (u0 + u1)
        uu = u0 + (np.arange(ww) + 0.5) * cell
        Rad = 0.5 * W_
        arch_shape = np.sqrt(np.maximum(Rad ** 2 - (uu - uc) ** 2, 0.0))
        zs = float(np.median(tops - arch_shape))
        res_arch = float(np.median(np.abs(tops - (zs + arch_shape))))
        res_flat = float(np.median(np.abs(tops - np.median(tops))))
        arched = bool(res_arch + 0.05 < res_flat and Rad >= prm.arch_min_rise and zs > zb_o + 0.2)
        to_ground = zb_o <= wall.z0 + 0.3
        if level == 1:
            kind = ("arch" if arched else "panel") if not to_ground else ("arched-bay" if arched else "bay")
        else:
            kind = ("arched-door" if arched else "door") if to_ground else ("arched-window" if arched else "window")
        el = {"type": "opening", "kind": kind, "level": level, "u0": round(u0, 3), "u1": round(u1, 3),
              "z_bottom": round(zb_o, 3), "z_top": round(zt_o, 3), "width": round(W_, 3), "height": round(Hh, 3),
              "area": round(area, 3), "fill": round(fill, 3), "data_share": round(datash, 3), "depth": round(depth, 3),
              "arched": arched, "spring_z": round(zs, 3) if arched else None,
              "arch_resid": round(res_arch, 3), "flat_resid": round(res_flat, 3), "register_frame": reg["frame"],
              "register": [reg["z_bottom"], reg["z_top"]]}
        if P3 is not None:
            ztop_ref = zs if arched else zt_o
            for key_, axis_, lo_, hi_, ok_ in (("u0", "u", zb_o + 0.2, ztop_ref - 0.2, True),
                                               ("u1", "u", zb_o + 0.2, ztop_ref - 0.2, True),
                                               ("z_bottom", "z", u0 + 0.2, u1 - 0.2, zb_o > max(wall.z0 + 0.3, reg["z_bottom"] + 0.05)),
                                               ("z_top", "z", u0 + 0.2, u1 - 0.2, not arched and zt_o < reg["z_top"] - 0.05)):
                if not ok_ or hi_ - lo_ < 0.3:
                    continue
                r_ = refine_side(P3, axis_, el[key_], lo_, hi_, lmin)
                if r_ is not None:
                    el[key_ + "_cell"], el[key_] = el[key_], round(r_, 3)
            u0, u1, zb_o, zt_o = el["u0"], el["u1"], el["z_bottom"], el["z_top"]
            el["width"], el["height"] = round(u1 - u0, 3), round(zt_o - zb_o, 3)
        vloc = line_clusters(WL, "v", zb_o, (zs if arched else zt_o), u0 - 0.5, u1 + 0.5, prm.local_cover, prm.line_min_photos, prm)
        hloc = line_clusters(WL, "h", u0, u1, zb_o - 0.5, zt_o + 0.5, prm.local_cover, prm.line_min_photos, prm)
        edges = {}
        for name, ep, cl, limit in (("left", u0, vloc, False), ("right", u1, vloc, False),
                                    ("bottom", zb_o, hloc, zb_o <= max(wall.z0 + 0.3, reg["z_bottom"] + 0.05)),
                                    ("top", zt_o, hloc, arched or zt_o >= reg["z_top"] - 0.05)):
            c = None if limit else nearest_cluster(cl, ep, prm.edge_tol_m)
            edges[name] = {"pos": round(ep, 3), "limit": bool(limit), "line": _edge_rec(c, ep)}
        _confirm(el, edges, lambda e, real: sum(1 for x in real if x["line"]) >= 2
                 and (e["left"]["line"] is not None or e["right"]["line"] is not None))
        el["chance_jamb"] = round(chance_rate(vloc, u0 - 0.5, u1 + 0.5, prm.edge_tol_m), 3)
        out.append(el)
    return out


def quantize_levels(M, prm, cell):
    """The wall's relief levels = peaks of the histogram of cell values (smoothed 0.03 m), >= level_min_mass of the
    cells each, >= level_sep_m apart; every cell snaps to its nearest level; then label regions under min_region_m2
    join the label most of their border has (repeated 3 x 3 majority on those cells). Returns the quantized map
    (cell values = level values) and the levels."""
    v = M[np.isfinite(M)]
    if len(v) < 50:
        return M, None
    lo, hi = np.percentile(v, 0.5) - 0.1, np.percentile(v, 99.5) + 0.1
    bins = np.arange(lo, hi + 0.01, 0.01)
    h, e = np.histogram(v, bins=bins)
    k = np.exp(-0.5 * (np.arange(-9, 10) * 0.01 / 0.03) ** 2)
    hs = np.convolve(h, k / k.sum(), 'same')
    centres = 0.5 * (e[:-1] + e[1:])
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]]
    peaks.sort(key=lambda i: -hs[i])
    lv = []
    for i in peaks:
        c = centres[i]
        if any(abs(c - x) < prm.level_sep_m for x in lv):
            continue
        lv.append(c)
        if len(lv) >= prm.max_levels:
            break
    lv = np.array(sorted(lv))
    # mass per level (nearest assignment); drop levels under the minimum mass (re-assign)
    while len(lv) > 1:
        lab = np.argmin(np.abs(v[:, None] - lv[None]), axis=1)
        mass = np.bincount(lab, minlength=len(lv)) / len(v)
        if mass.min() >= prm.level_min_mass:
            break
        lv = np.delete(lv, int(mass.argmin()))
    # refine each level to the median of its cells
    lab = np.argmin(np.abs(v[:, None] - lv[None]), axis=1)
    lv = np.array([np.median(v[lab == j]) if (lab == j).any() else lv[j] for j in range(len(lv))])
    fin = np.isfinite(M)
    L = np.full(M.shape, -1, int)
    L[fin] = np.argmin(np.abs(M[fin][:, None] - lv[None]), axis=1)
    # clean small regions
    min_cells = max(1, int(round(prm.min_region_m2 / (cell * cell))))
    for _ in range(4):
        changed = False
        for j in range(len(lv)):
            n, cc, st_, _ = cv2.connectedComponentsWithStats((L == j).astype(np.uint8), connectivity=4)
            for r in range(1, n):
                if st_[r, 4] >= min_cells:
                    continue
                m = cc == r
                ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m & (L >= 0)
                if not ring.any():
                    continue
                vals = L[ring]
                vals = vals[vals != j]
                if not len(vals):
                    continue
                L[m] = np.bincount(vals).argmax()
                changed = True
        if not changed:
            break
    Q = np.where(L >= 0, lv[np.clip(L, 0, None)], np.nan)
    return Q, lv


def refine_side(P3, axis, e, lo, hi, lmin, win=0.5, step=0.05):
    """Sub-cell position of an element edge from the raw points: along `axis` ('u' or 'z') around e, over the other
    coordinate in [lo, hi]: the median relief per 5 cm bin; the crossing of the half-way level between the two
    sides (each side: the bins 0.15-0.5 m from e) nearest e (within 0.25 m). None when the sides differ < lmin."""
    u, rho, z = P3
    a, b = (u, z) if axis == "u" else (z, u)
    m = (b >= lo) & (b <= hi) & (np.abs(a - e) <= win)
    if m.sum() < 30:
        return None
    edges = np.arange(e - win, e + win + 1e-9, step)
    k = np.clip(((a[m] - (e - win)) / step).astype(int), 0, len(edges) - 2)
    med, cnt = _group_median(k, rho[m], len(edges) - 1)
    med[cnt < 3] = np.nan
    c = 0.5 * (edges[:-1] + edges[1:])
    left = med[(c <= e - 0.15) & np.isfinite(med)]
    right = med[(c >= e + 0.15) & np.isfinite(med)]
    if len(left) < 2 or len(right) < 2:
        return None
    A, B = float(np.median(left)), float(np.median(right))
    if abs(A - B) < lmin:
        return None
    half = 0.5 * (A + B)
    best = None
    for i in range(len(c) - 1):
        y0, y1 = med[i], med[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)) or (y0 - half) * (y1 - half) > 0 or y0 == y1:
            continue
        x = c[i] + (half - y0) / (y1 - y0) * (c[i + 1] - c[i])
        if abs(x - e) <= 0.25 and (best is None or abs(x - e) < abs(best - e)):
            best = float(x)
    return best




# ================================================================ split-half photo check (the second channel)


@dataclass(frozen=True)
class ReliefGuards:
    min_photos_per_half: int = 6        # G-R1: photos seeing the wall (>= photo_wall_frac of it) in EACH photo half
    photo_wall_frac: float = 0.3
    split_iou: float = 0.3              # G-R2: found again in each half (bbox IoU, same type and sign) ...
    split_depth_abs_m: float = 0.1      # ... with the two halves' depths within max(this, split_depth_rel x depth)
    split_depth_rel: float = 0.5
    min_line_edges: int = 1             # G-R3: edges at a 3-D line cluster (within ReliefParams.edge_tol_m)
    max_depth_m: float = 1.5            # G-R4: depth / projection (sizes: the detector's limits)
    min_observed_share: float = 0.5     # G-R5: share of the element's box on observed facade texels
    tri_max_err_px: float = 2.0         # split-half re-triangulation: every used observation within this ...
    tri_min_angle_deg: float = 2.0      # ... and a triangulation angle of at least this
    tri_max_obs: int = 12               # observations per point used (the first by camera index)


DEFAULT_GUARDS = ReliefGuards()
DEPTH_KEY = {"band": "level", "vertical": "projection", "opening": "depth"}
GUARD_NAMES = ("G-R1_photos_per_half", "G-R2_split_half", "G-R3_line_edge", "G-R4_depth", "G-R5_observed_texels",
               "parent_accepted")


@dataclass
class PointTracks:
    """The 2-D observations behind the SfM points: observation k is point `point[k]` (index into the scene's points)
    seen by camera `camera[k]` (index into the scene's cameras) at pixel `xy[k]` (the stored image's pixels, as the
    camera model projects them). The split-half check re-triangulates points from one half of the photos."""

    point: np.ndarray
    camera: np.ndarray
    xy: np.ndarray

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, np.int64).reshape(-1)
        self.camera = np.asarray(self.camera, np.int64).reshape(-1)
        self.xy = np.asarray(self.xy, float).reshape(-1, 2)
        if not (len(self.point) == len(self.camera) == len(self.xy)):
            raise ValueError("point, camera and xy must have the same length")

    def __len__(self) -> int:
        return len(self.point)

    @classmethod
    def from_colmap(cls, rec, camera_of_image: dict[int, int]) -> PointTracks:
        """From a pycolmap Reconstruction, points in rec.points3D.values() order (the order the scene's points were
        read in); camera_of_image maps an image id to the scene's camera index (images without one are skipped)."""
        xy_of = {}
        for iid in camera_of_image:
            im = rec.images[iid]
            xy_of[iid] = np.array([p.xy for p in im.points2D], float).reshape(-1, 2)
        pt, cam, idx = [], [], []
        for k, p in enumerate(rec.points3D.values()):
            for el in p.track.elements:
                c = camera_of_image.get(el.image_id)
                if c is None:
                    continue
                pt.append(k)
                cam.append(c)
                idx.append((el.image_id, el.point2D_idx))
        xy = np.array([xy_of[i][j] for i, j in idx], float).reshape(-1, 2)
        return cls(np.array(pt, np.int64), np.array(cam, np.int64), xy)


def split_halves(cameras, centre) -> np.ndarray:
    """Photo half (0 / 1) per camera: sorted by bearing around the building centre, alternately, so that both halves
    see every side and neither shares a single photo with the other."""
    if not len(cameras):
        return np.zeros(0, int)
    c = np.array([np.asarray(cam.center, float)[:2] for cam in cameras]).reshape(-1, 2)
    b = np.arctan2(c[:, 1] - centre[1], c[:, 0] - centre[0])
    order = np.lexsort((np.arange(len(b)), b))
    half = np.empty(len(cameras), int)
    half[order] = np.arange(len(cameras)) % 2
    return half


def retriangulate(ids, tracks: PointTracks, cameras, use_camera: np.ndarray,
                  g: ReliefGuards = DEFAULT_GUARDS) -> tuple[np.ndarray, np.ndarray]:
    """Points `ids` re-triangulated (linear DLT on undistorted normalized coordinates, poses of the joint model) from
    the observations of the cameras with use_camera[c] only: (len(ids), 3) ENU (nan where refused: fewer than 2
    views, a view behind the camera, a reprojection error over tri_max_err_px or a triangulation angle under
    tri_min_angle_deg) and the number of observations used."""
    ids = np.asarray(ids, np.int64).reshape(-1)
    X = np.full((len(ids), 3), np.nan)
    nobs = np.zeros(len(ids), int)
    if not len(ids) or tracks is None or not len(tracks) or not len(cameras):
        return X, nobs
    use_camera = np.asarray(use_camera, bool)
    o_ids = np.argsort(ids, kind="stable")
    sid = ids[o_ids]
    pos = np.clip(np.searchsorted(sid, tracks.point), 0, len(sid) - 1)
    cam_ok = (tracks.camera >= 0) & (tracks.camera < len(cameras))
    m = (sid[pos] == tracks.point) & cam_ok
    m[m] &= use_camera[tracks.camera[m]]
    loc, cam, xy = o_ids[pos[m]], tracks.camera[m], tracks.xy[m]
    o = np.lexsort((cam, loc))
    loc, cam, xy = loc[o], cam[o], xy[o]
    if not len(loc):
        return X, nobs
    start = np.r_[0, np.flatnonzero(np.diff(loc)) + 1]
    counts = np.diff(np.r_[start, len(loc)])
    rank = np.arange(len(loc)) - np.repeat(start, counts)
    keep = rank < g.tri_max_obs
    loc, cam, xy = loc[keep], cam[keep], xy[keep]
    start = np.r_[0, np.flatnonzero(np.diff(loc)) + 1]
    counts = np.diff(np.r_[start, len(loc)])
    K = np.array([[c.fx, c.fy, c.cx, c.cy, c.k1, c.k2] for c in cameras], float)
    Rs = np.stack([np.asarray(c.rot, float) for c in cameras])
    Cs = np.stack([np.asarray(c.center, float) for c in cameras])
    fx, fy, cx, cy, k1, k2 = K[cam].T
    x, y = (xy[:, 0] - cx) / fx, (xy[:, 1] - cy) / fy
    xu, yu = x.copy(), y.copy()
    for _ in range(10):
        r2 = xu * xu + yu * yu
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        xu, yu = x / f, y / f
    Pm = np.concatenate([Rs, -(Rs @ Cs[:, :, None])], axis=2)          # (ncam, 3, 4), normalized coordinates
    Po = Pm[cam]
    rx = xu[:, None] * Po[:, 2, :] - Po[:, 0, :]
    ry = yu[:, None] * Po[:, 2, :] - Po[:, 1, :]
    for k in np.unique(counts):
        if k < 2:
            continue
        sel = np.flatnonzero(counts == k)
        oi = start[sel][:, None] + np.arange(k)[None, :]              # (n, k) observation indices
        A = np.empty((len(sel), 2 * k, 4))
        A[:, 0::2], A[:, 1::2] = rx[oi], ry[oi]
        A /= np.maximum(np.linalg.norm(A, axis=2, keepdims=True), 1e-12)
        try:
            Vt = np.linalg.svd(A)[2]
        except np.linalg.LinAlgError:
            continue
        Xh = Vt[:, -1, :]
        ok = np.abs(Xh[:, 3]) > 1e-12
        Xi = Xh[:, :3] / np.where(ok, Xh[:, 3], 1.0)[:, None]
        c_ = cam[oi]
        pc = np.einsum("nkij,nkj->nki", Rs[c_], Xi[:, None, :] - Cs[c_])
        z = pc[..., 2]
        ok &= (z > 1e-6).all(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xn, yn = pc[..., 0] / z, pc[..., 1] / z
        r2 = xn * xn + yn * yn
        f = 1.0 + K[c_, 4] * r2 + K[c_, 5] * r2 * r2
        err = np.hypot(K[c_, 0] * xn * f + K[c_, 2] - xy[oi, 0], K[c_, 1] * yn * f + K[c_, 3] - xy[oi, 1])
        ok &= np.nan_to_num(err, nan=np.inf).max(axis=1) <= g.tri_max_err_px
        ray = Xi[:, None, :] - Cs[c_]
        ray /= np.maximum(np.linalg.norm(ray, axis=2, keepdims=True), 1e-12)
        cmin = np.einsum("nki,nli->nkl", ray, ray).min(axis=(1, 2))
        ok &= np.degrees(np.arccos(np.clip(cmin, -1.0, 1.0))) >= g.tri_min_angle_deg
        pts = loc[start[sel]]
        X[pts[ok]] = Xi[ok]
        nobs[pts[ok]] = k
    return X, nobs


def _box(e: dict) -> tuple[float, float, float, float]:
    return float(e["u0"]), float(e["u1"]), float(e["z_bottom"]), float(e["z_top"])


def box_iou(a, b) -> float:
    iu = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iz = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = iu * iz
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / max(union, 1e-9)


def match_element(e: dict, others: list[dict], min_iou: float) -> tuple[float, dict | None]:
    """The best match of element e among others (same type and sign): (IoU, the match or None under min_iou)."""
    best, f = 0.0, None
    for o in others:
        if o["type"] != e["type"] or (o.get("sign") or 0) != (e.get("sign") or 0):
            continue
        s = box_iou(_box(e), _box(o))
        if s > best:
            best, f = s, o
    return best, (f if best >= min_iou else None)


def observed_share(e: dict, wall, mask: np.ndarray | None) -> float:
    """Share of the element's box (clipped to the wall) on observed texels of the facade mask (row 0 at the wall's
    top, columns along u)."""
    if mask is None or not mask.size:
        return 0.0
    H, W = mask.shape
    span = max(float(wall.z1 - wall.z0), 1e-9)
    c0 = int(np.clip(math.floor(e["u0"] / wall.length * W), 0, W - 1))
    c1 = int(np.clip(math.ceil(e["u1"] / wall.length * W), c0 + 1, W))
    r0 = int(np.clip(math.floor((wall.z1 - min(e["z_top"], wall.z1)) / span * H), 0, H - 1))
    r1 = int(np.clip(math.ceil((wall.z1 - max(e["z_bottom"], wall.z0)) / span * H), r0 + 1, H))
    return float(mask[r0:r1, c0:c1].mean())


def element_offset(e: dict, parent: dict | None) -> float:
    """The element's surface relative to the wall's photographed frame surface (m, outward +): a recess's back, a
    projection's front, a band's level."""
    t = e["type"]
    if t == "band":
        return float(e["level"])
    if t == "vertical":
        rf = float(e.get("register_frame") or 0.0)
        return rf + float(e["projection"]) if e.get("how") == "proud" else rf
    rf = float(e.get("register_frame") or 0.0)
    if (e.get("level") or 1) == 2:                     # nested: below its parent's back, or its field's level
        if parent is not None:
            return float(parent["offset_m"]) - float(e["depth"])
        return rf - float(e.get("depth_total", e["depth"]))
    return rf - float(e["depth"])


def guard_elements(WR: WallRelief, halves: list, per_half: list[int], mask: np.ndarray | None, wall,
                   g: ReliefGuards = DEFAULT_GUARDS) -> list[dict]:
    """Every element the full photo set detected on the wall, with its split-half match, guards and the verdict
    ('accepted': all guards pass; 'builds_geometry': accepted and not a frame strip, which is the wall itself)."""
    els = WR.elements()
    nb, nv = len(WR.bands), len(WR.verticals)
    out: list[dict] = []
    frame = {"c0": round(float(WR.ref["c0"] + WR.frame), 4), "c1": round(float(WR.ref["c1"]), 5)} if WR.ref else None
    for i, e in enumerate(els):
        key = DEPTH_KEY[e["type"]]
        depth = abs(float(e[key]))
        found = []
        split = {}
        for h, name in ((0, "a"), (1, "b")):
            hr = halves[h] if h < len(halves) else None
            if hr is None or hr.refused:
                split[f"iou_{name}"] = 0.0
                found.append(None)
                continue
            s, f = match_element(e, hr.elements(), g.split_iou)
            split[f"iou_{name}"] = round(s, 3)
            found.append(f)
        fa, fb = found
        dd = abs(float(fa[key]) - float(fb[key])) if fa is not None and fb is not None else None
        split["depth_diff"] = None if dd is None else round(dd, 3)
        if fa is not None and fb is not None:
            ks = ("z_bottom", "z_top") if e["type"] == "band" else ("u0", "u1", "z_bottom", "z_top")
            split["edge_diff"] = [round(abs(float(fa[k]) - float(fb[k])), 3) for k in ks]
        parent_abs = None
        if e["type"] == "opening" and (e.get("level") or 1) == 2 and e.get("parent") is not None:
            parent_abs = nb + nv + int(e["parent"])
        parent = out[parent_abs] if parent_abs is not None and parent_abs < len(out) else None
        guards = {
            "G-R1_photos_per_half": bool(min(per_half) >= g.min_photos_per_half),
            "G-R2_split_half": bool(dd is not None and dd <= max(g.split_depth_abs_m, g.split_depth_rel * depth)),
            "G-R3_line_edge": bool(e.get("edges_confirmed", 0) >= g.min_line_edges),
            "G-R4_depth": bool(depth <= g.max_depth_m),
            "G-R5_observed_texels": bool(observed_share(e, wall, mask) >= g.min_observed_share),
            "parent_accepted": bool(e["type"] != "opening" or (e.get("level") or 1) == 1
                                    or (parent is not None and parent["accepted"])),
        }
        acc = all(guards.values())
        r = {"i": i, "id": f"w{wall.index}-{e['type'][0]}{i}", "type": e["type"], "kind": e["kind"],
             "sign": e.get("sign"), "how": e.get("how"),
             "u0": float(e["u0"]), "u1": float(e["u1"]), "z_bottom": float(e["z_bottom"]), "z_top": float(e["z_top"]),
             "depth_key": key, "depth": float(e[key]), "register_frame": float(e.get("register_frame") or 0.0),
             "arched": bool(e.get("arched", False)), "spring_z": e.get("spring_z"), "level": e.get("level"),
             "parent_el": parent_abs, "frame": frame,
             "edges_real": int(e.get("edges_real", 0)), "edges_confirmed": int(e.get("edges_confirmed", 0)),
             "line_photo_ids": list(e.get("photos") or []),
             "split": split, "photos_per_half": [int(v) for v in per_half], "guards": guards,
             "accepted": acc, "builds_geometry": bool(acc and not (e["type"] == "vertical" and e.get("how") == "frame"))}
        r["offset_m"] = round(element_offset(e, parent), 3)
        out.append(r)
    return out


@dataclass
class ReliefResult:
    """Facade relief on the photographed walls: per wall its report (refused or measured: points, photos per half,
    element counts) and its elements with their guards."""

    walls: dict[int, dict] = field(default_factory=dict)
    elements: dict[int, list[dict]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    lines: Any = None                     # the 3-D lines used (Lines3D), for other checks of the same model

    def accepted(self, wall: int) -> list[dict]:
        return [e for e in self.elements.get(wall, []) if e["accepted"]]

    def to_meta(self) -> dict[str, Any]:
        n = lambda f: sum(1 for els in self.elements.values() for e in els if f(e))
        return {"walls": {str(k): v for k, v in sorted(self.walls.items())}, "stats": self.stats,
                "detected": n(lambda e: True), "accepted": n(lambda e: e["accepted"]),
                "geometry": n(lambda e: e["builds_geometry"])}


def _count_check(wall, points, track, prm: ReliefParams, others) -> bool:
    """The cheap part of G-R1 (enough points near the wall), to decide whether 3-D lines are worth computing."""
    U, _ = wall_points(wall, points, track, prm, others)
    area = wall.length * (wall.z1 - wall.z0)
    return len(U) >= prm.min_points and len(U) >= prm.min_density * area


def measure_relief(massing, observations, facades: dict, points: np.ndarray | None, track: np.ndarray | None,
                   tracks: PointTracks | None, cameras: list, lines=None, stub_parts: set | None = None,
                   prm: ReliefParams = DEFAULT_RELIEF, g: ReliefGuards = DEFAULT_GUARDS, log=None) -> ReliefResult:
    """Measure and guard facade relief on every photographed wall (the keys of `facades`; observations indexed by
    wall). points / cameras in the model's final ENU frame (after the snap); track: per point track length;
    lines: a crowd3d.lines3d Lines3D (or LinesResult), or a callable returning one (called only when a wall has
    enough points), or None (then no edge is line-confirmed: G-R3 refuses every element)."""
    t0 = time.perf_counter()
    res = ReliefResult()
    stubs = stub_parts or set()
    if points is None or not len(points) or not facades:
        res.stats = {"seconds": 0.0, "note": "no registered points or no photographed wall"}
        return res
    P = np.asarray(points, float)
    trk = np.full(len(P), prm.min_track) if track is None else np.asarray(track, float)
    walls = massing.walls
    centre = np.mean(np.concatenate([np.stack([w.a, w.b]) for w in walls]), axis=0)
    half = split_halves(cameras, centre)
    cand = []
    for wi in sorted(facades):
        wall = walls[wi]
        rep: dict[str, Any] = {"wall": wi, "part": getattr(wall, "part", None)}
        res.walls[wi] = rep
        ob = observations[wi] if observations is not None and wi < len(observations) else None
        seen = [ci for ci, frac, _ in (ob.cameras if ob is not None else []) if frac >= g.photo_wall_frac]
        rep["photos_per_half"] = [int(sum(1 for c in seen if c < len(half) and half[c] == h)) for h in (0, 1)]
        if getattr(wall, "profile", None) is not None:
            rep["refused"] = "a gable / skillion wall (top profile): relief is measured on rectangular walls only"
        elif rep["part"] is not None and rep["part"] in stubs:
            rep["refused"] = "a 'height pending' stub part"
        elif min(rep["photos_per_half"]) < g.min_photos_per_half:
            rep["refused"] = (f"G-R1: photos seeing the wall per photo half {rep['photos_per_half']} "
                              f"(< {g.min_photos_per_half})")
        elif tracks is None or not len(tracks):
            rep["refused"] = "no point tracks: the split-half check (G-R2) cannot run"
        elif not _count_check(wall, P, trk, prm, walls):
            rep["refused"] = "G-R1: too few SfM points for relief"
        else:
            cand.append(wi)
    if callable(lines):
        if cand:                                            # computed only when a wall can carry relief
            tl = time.perf_counter()
            lines = lines()
            res.stats["lines_s"] = round(time.perf_counter() - tl, 2)
        else:
            lines = None
    if lines is not None and hasattr(lines, "lines") and not hasattr(lines, "A"):
        lines = lines.lines                                 # a LinesResult
    res.stats["lines"] = 0 if lines is None else int(len(lines))
    res.lines = lines
    if lines is None or not len(lines):
        for wi in cand:                                     # nothing could pass G-R3: do not measure for nothing
            res.walls[wi]["refused"] = "no 3-D lines: no edge can be line-confirmed (G-R3)"
        cand = []
    full: dict[int, WallRelief] = {}
    for wi in cand:
        WR = measure_wall(walls[wi], P, trk, lines, prm, others=walls)
        rep = res.walls[wi]
        rep["points"] = WR.stats.get("points")
        if WR.refused:
            rep["refused"] = "G-R1: " + WR.refused
            continue
        rep["frame_offset_m"] = WR.stats.get("frame_offset_m")
        rep["data_share"] = WR.stats.get("data_share_in_wall")
        full[wi] = WR
    if full:
        ts = time.perf_counter()
        ids = np.unique(np.concatenate([WR.maps["idx"] for WR in full.values()]))
        tri = [retriangulate(ids, tracks, cameras, half == h, g) for h in (0, 1)]
        res.stats["split_points"] = [int(np.isfinite(X[:, 0]).sum()) for X, _ in tri]
        res.stats["split_of"] = int(len(ids))
        res.stats["split_s"] = round(time.perf_counter() - ts, 2)
        for wi, WR in full.items():
            wall = walls[wi]
            halves = []
            for X, c in tri:
                ok = np.isfinite(X[:, 0])
                halves.append(measure_wall(wall, X[ok], np.maximum(c[ok], prm.min_track), None, prm, others=walls))
            rep = res.walls[wi]
            rep["split_refused"] = [h.refused for h in halves]
            fac = facades.get(wi)
            els = guard_elements(WR, halves, rep["photos_per_half"], None if fac is None else fac.mask, wall, g)
            for e in els:
                e["line_photos"] = sorted({(cameras[c].photo_id or cameras[c].name) for c in e["line_photo_ids"]
                                           if 0 <= c < len(cameras)})
            res.elements[wi] = els
            rep["detected"] = len(els)
            rep["accepted"] = sum(e["accepted"] for e in els)
    res.stats["seconds"] = round(time.perf_counter() - t0, 2)
    res.stats["walls_measured"] = len(full)
    if log:
        log(f"relief: {len(full)} of {len(facades)} photographed walls measured, "
            f"{sum(len(v) for v in res.elements.values())} elements, "
            f"{sum(e['accepted'] for v in res.elements.values() for e in v)} accepted ({res.stats['seconds']} s)")
    return res


# ================================================================ geometry: the accepted elements on their wall
#
# Accepted elements are built as low-poly geometry ON the OSM-frame wall they were measured on, only inside the
# wall's photographed piece:
#   - recesses (openings, setbacks, grooves): a hole in the wall face, reveal faces and a back face at the measured
#     depth; a nested opening is a hole in its parent's back face;
#   - projections (bands, proud pilasters / buttresses): a box in front of the wall face.
# Depths are relative to the wall's photographed frame surface, which the model wall stands for. Every vertex keeps
# its (u, z) on the wall, so the facade texture maps onto the new faces as on the flat wall (reveals take the texels
# of their edge).

PROV_MEASURED = "photo-measured"
RELIEF_MAX_DEPTH = 1.5
RELIEF_MARGIN = 0.05


def _arch_polygon(u0, u1, z0, spring, ztop_max, segs=8):
    """CCW (u, z) polygon of a round-arched opening: rectangle up to the springing line, half circle above."""
    r = 0.5 * (u1 - u0)
    uc = 0.5 * (u0 + u1)
    pts = [(u0, z0), (u1, z0), (u1, spring)]
    for k in range(1, segs):
        a = math.pi * k / segs
        pts.append((uc + r * math.cos(a), min(spring + r * math.sin(a), ztop_max)))
    pts.append((u0, spring))
    return np.array(pts, float)


def _rect(u0, u1, z0, z1):
    return np.array([(u0, z0), (u1, z0), (u1, z1), (u0, z1)], float)


def signed_area(poly) -> float:
    p = np.asarray(poly, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _in_polygon(pts, poly) -> np.ndarray:
    from .stylize import points_in_polygon
    pts, poly = np.asarray(pts, float), np.asarray(poly, float)
    inside = points_in_polygon(pts, poly)
    return inside | (np.min(np.linalg.norm(pts[:, None] - poly[None], axis=2), axis=1) < 1e-3)


def relief_plan(wall, u0, u1, elements, margin=RELIEF_MARGIN):
    """Which elements become which geometry on the piece u0..u1 of the wall: (recesses, boxes, skipped).
    recess: {"poly": (k, 2) CCW (u, z), "d": back offset, "parent": index into recesses or None, "el": element
    index}; box: {"u0", "u1", "z0", "z1", "p": projection, "el"}; skipped: (element index, why)."""
    z0w, z1w = float(wall.z0), float(wall.z1)
    rec, boxes, skipped = [], [], []
    lo_u, hi_u, lo_z, hi_z = u0 + margin, u1 - margin, z0w + margin, z1w - margin

    def clip(a, b, c, d):
        return max(a, lo_u), min(b, hi_u), max(c, lo_z), min(d, hi_z)

    level1 = []
    for i, e in enumerate(elements):
        if not e.get("builds_geometry") or e["u1"] <= u0 or e["u0"] >= u1:
            continue
        if e["type"] == "band" and (e.get("sign") or 0) < 0:
            a, b, c, d = clip(e["u0"], e["u1"], e["z_bottom"], e["z_top"])
            if b - a >= 0.2 and d - c >= 0.15:
                level1.append({"poly": _rect(a, b, c, d), "d": float(e["depth"]), "parent": None, "el": i})
            else:
                skipped.append((i, "outside the photographed piece after clipping"))
        elif e["type"] == "opening" and (e.get("level") or 1) == 1:
            a, b, c, d = clip(e["u0"], e["u1"], e["z_bottom"], e["z_top"])
            if b - a < 0.2 or d - c < 0.2:
                skipped.append((i, "outside the photographed piece after clipping"))
                continue
            back = float(e.get("register_frame") or 0.0) - float(e["depth"])
            if e.get("arched") and e.get("spring_z") is not None and e["spring_z"] > c + 0.1:
                poly = _arch_polygon(a, b, c, min(float(e["spring_z"]), d), d)
            else:
                poly = _rect(a, b, c, d)
            level1.append({"poly": poly, "d": back, "parent": None, "el": i})
        elif e["type"] in ("band", "vertical"):
            p = float(e["depth"])
            if p <= 0:
                continue
            a, b = max(e["u0"], u0), min(e["u1"], u1)
            c, d = max(e["z_bottom"], z0w), min(e["z_top"], z1w)
            if b - a >= 0.1 and d - c >= 0.1:
                boxes.append({"u0": a, "u1": b, "z0": c, "z1": d, "p": min(p, RELIEF_MAX_DEPTH), "el": i})
            else:
                skipped.append((i, "outside the photographed piece after clipping"))

    def bbox(poly):
        return poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max()

    level1.sort(key=lambda r: -abs(signed_area(r["poly"])))           # level-1 recesses must not overlap: larger first
    for r in level1:
        bu0, bu1, bz0, bz1 = bbox(r["poly"])
        clash = any(not (bu1 <= q0 - 0.02 or bu0 >= q1 + 0.02 or bz1 <= s0 - 0.02 or bz0 >= s1 + 0.02)
                    for q0, q1, s0, s1 in (bbox(x["poly"]) for x in rec))
        if clash:
            skipped.append((r["el"], "overlaps a larger recess"))
            continue
        if r["d"] >= -0.02:
            skipped.append((r["el"], "not behind the wall face"))
            continue
        rec.append(r)
    built_el = {r["el"]: k for k, r in enumerate(rec)}
    for i, e in enumerate(elements):
        if not e.get("builds_geometry") or e["type"] != "opening" or (e.get("level") or 1) != 2:
            continue
        if e["u1"] <= u0 or e["u0"] >= u1:
            continue
        k = built_el.get(e.get("parent_el"))
        if k is None:
            skipped.append((i, "parent opening not built"))
            continue
        P = rec[k]["poly"]
        pu0, pu1, pz0, pz1 = bbox(P)
        a, b = max(e["u0"], pu0 + margin), min(e["u1"], pu1 - margin)
        c, d = max(e["z_bottom"], pz0 + margin), min(e["z_top"], pz1 - margin)
        if b - a < 0.15 or d - c < 0.15:
            skipped.append((i, "too small inside its parent"))
            continue
        poly = _rect(a, b, c, d)
        if not _in_polygon(poly, P).all():
            skipped.append((i, "not inside its parent's outline"))
            continue
        if any(r.get("parent") == k and not (b <= bbox(r["poly"])[0] or a >= bbox(r["poly"])[1]
                                             or d <= bbox(r["poly"])[2] or c >= bbox(r["poly"])[3]) for r in rec):
            skipped.append((i, "overlaps a sibling"))
            continue
        rec.append({"poly": poly, "d": rec[k]["d"] - float(e["depth"]), "parent": k, "el": i})
    return rec, boxes, skipped


def relief_mesh(wall, u0, u1, elements):
    """The wall piece u0..u1 with its relief: (pos ENU, normals, uz, triangles, report), or None when nothing is
    built. The first four vertices are the piece's frame corners (bottom-left, bottom-right, top-right, top-left, as
    the flat wall's quad: the audits and the viewer read them); they belong to no triangle."""
    from .stylize import triangulate_with_holes
    if getattr(wall, "profile", None) is not None:
        return None
    rec, boxes, skipped = relief_plan(wall, u0, u1, elements)
    if not rec and not boxes:
        return None
    a = np.r_[wall.a, 0.0]
    t = np.r_[wall.tangent, 0.0]
    n = np.r_[wall.normal, 0.0]
    up = np.array([0.0, 0.0, 1.0])
    P, N, UZ, T = [], [], [], []

    def add_face(uz, d, normal, tris):
        base = len(P)
        for (u, z), dd in zip(uz, d):
            P.append(a + u * t + dd * n + z * up)
            UZ.append((u, z))
            N.append(normal)
        for i, j, k in tris:
            T.append((base + i, base + j, base + k))

    z0, z1 = float(wall.z0), float(wall.z1)
    add_face(_rect(u0, u1, z0, z1), [0.0] * 4, n, [])
    holes = [r["poly"] for r in rec if r["parent"] is None]
    ring, tris = triangulate_with_holes(_rect(u0, u1, z0, z1), holes)
    add_face(ring, np.zeros(len(ring)), n, tris)
    for k, r in enumerate(rec):
        poly = r["poly"]
        dp = 0.0 if r["parent"] is None else rec[r["parent"]]["d"]
        d = r["d"]
        m = len(poly)
        for i in range(m):
            p0, p1 = poly[i], poly[(i + 1) % m]
            du, dz = p1 - p0
            ln = math.hypot(du, dz)
            if ln < 1e-6:
                continue
            nin = np.array([-dz, du]) / ln                     # left of a CCW edge: into the opening
            nrm = nin[0] * t + nin[1] * up
            add_face([tuple(p0), tuple(p1), tuple(p1), tuple(p0)], [dp, dp, d, d], nrm, [(0, 1, 2), (0, 2, 3)])
        kids = [x["poly"] for x in rec if x["parent"] == k]
        ring, tris = triangulate_with_holes(poly, kids)
        add_face(ring, np.full(len(ring), d), n, tris)
    for b in boxes:
        ua, ub, za, zb, p = b["u0"], b["u1"], b["z0"], b["z1"], b["p"]
        add_face([(ua, za), (ub, za), (ub, zb), (ua, zb)], [p] * 4, n, [(0, 1, 2), (0, 2, 3)])          # front
        add_face([(ua, zb), (ub, zb), (ub, zb), (ua, zb)], [0, 0, p, p], up, [(0, 1, 2), (0, 2, 3)])     # top
        add_face([(ua, za), (ub, za), (ub, za), (ua, za)], [0, 0, p, p], -up, [(0, 1, 2), (0, 2, 3)])   # bottom
        add_face([(ua, za), (ua, zb), (ua, zb), (ua, za)], [0, 0, p, p], -t, [(0, 1, 2), (0, 2, 3)])     # start end
        add_face([(ub, za), (ub, zb), (ub, zb), (ub, za)], [0, 0, p, p], t, [(0, 1, 2), (0, 2, 3)])      # far end
    P, N, UZ, T = np.array(P), np.array(N), np.array(UZ), np.array(T, int).reshape(-1, 3)
    g = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])     # every triangle faces its own normal
    flip = np.einsum("ij,ij->i", g, N[T[:, 0]]) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    report = {"recesses": [{"el": r["el"], "d": round(r["d"], 3), "parent": r["parent"]} for r in rec],
              "boxes": [{"el": b["el"], "p": round(b["p"], 3)} for b in boxes],
              "skipped": [{"el": i, "why": w} for i, w in skipped]}
    return P, N, UZ, T, report


def element_record(e: dict, built: bool) -> dict:
    """The provenance record of one facade element, as it travels in its wall node's extras (GLB)."""
    sp = e.get("split") or {}
    out = {"id": e.get("id"), "type": e["type"], "kind": e["kind"], "u0": round(e["u0"], 3), "u1": round(e["u1"], 3),
           "z_bottom": round(e["z_bottom"], 3), "z_top": round(e["z_top"], 3),
           e["depth_key"]: round(float(e["depth"]), 3), "offset_m": e.get("offset_m"),
           "provenance": PROV_MEASURED,
           "channels": {"sfm_points": True,
                        "split_half": {"iou_a": sp.get("iou_a"), "iou_b": sp.get("iou_b"),
                                       "depth_diff_m": sp.get("depth_diff")},
                        "lines3d": {"edges_confirmed": e["edges_confirmed"], "edges": e["edges_real"],
                                    "photos": len(e.get("line_photos") or [])}},
           "photos_per_half": list(e.get("photos_per_half") or []), "line_photos": list(e.get("line_photos") or []),
           "guards": dict(e.get("guards") or {}), "built": bool(built)}
    if e.get("frame"):
        out["frame"] = e["frame"]
    if e["type"] == "opening":
        out["nesting"] = int(e.get("level") or 1)
        if e.get("arched") and e.get("spring_z") is not None:
            out["arched"], out["spring_z"] = True, round(float(e["spring_z"]), 3)
    if e.get("sign") is not None:
        out["sign"] = int(e["sign"])
    if e.get("how"):
        out["how"] = e["how"]
    return out


# ---------------------------------------------------------------- QA (called by crowd3d.stylize's checks)


RELIEF_LIMITS = {"band": {"h": (0.1, 6.0)}, "vertical": {"w": (0.2, 2.5)}, "opening": {"w": (0.4, 8.0), "h": (0.6, 30.0)}}


def record_violations(r: dict, g: ReliefGuards = DEFAULT_GUARDS) -> list[str]:
    """What is wrong with one element record (empty: backed): photo-measured, both photo halves found it, >= 1 line-
    confirmed edge, enough photos per half, every guard passed, inside the size and depth limits."""
    v = []
    ch = r.get("channels") or {}
    sh = ch.get("split_half") or {}
    if r.get("provenance") != PROV_MEASURED:
        v.append(f"element provenance {r.get('provenance')!r}")
    if not (sh.get("iou_a") or 0) >= g.split_iou or not (sh.get("iou_b") or 0) >= g.split_iou:
        v.append("element not measured from both photo halves")
    if not ((ch.get("lines3d") or {}).get("edges_confirmed") or 0) >= g.min_line_edges:
        v.append("no line-confirmed edge")
    pph = r.get("photos_per_half") or [0, 0]
    if len(pph) < 2 or min(pph) < g.min_photos_per_half:
        v.append(f"photos per half {pph} < {g.min_photos_per_half}")
    gd = r.get("guards")
    if not isinstance(gd, dict) or not gd or not all(bool(x) for x in gd.values()):
        v.append("a guard did not pass")
    depth = max((abs(float(r[k])) for k in ("depth", "level", "projection") if r.get(k) is not None), default=None)
    if depth is None or depth > g.max_depth_m + 1e-9:
        v.append(f"depth {depth} m missing or over the limit")
    try:
        w, h = float(r["u1"]) - float(r["u0"]), float(r["z_top"]) - float(r["z_bottom"])
    except (KeyError, TypeError, ValueError):
        return v + ["element without its box"]
    lim = RELIEF_LIMITS.get(r.get("type"), None)
    if lim is None:
        v.append(f"element type {r.get('type')!r}")
    else:
        if "w" in lim and not (lim["w"][0] - 1e-6 <= w <= lim["w"][1] + 1e-6) and r.get("kind") != "corner-pier":
            v.append(f"{r.get('kind')} width {w:.2f} m outside {lim['w']}")
        if "h" in lim and not (lim["h"][0] - 1e-6 <= h <= lim["h"][1] + 1e-6):
            v.append(f"{r.get('kind')} height {h:.2f} m outside {lim['h']}")
    return v


def node_violations(extras: dict, textured: bool, stub_parts: set | None = None) -> list[str]:
    """A node carrying relief must be a photographed wall: kind wall, observed, textured, appearance photo-observed,
    not on a stub part."""
    if extras.get("kind") != "wall" or not extras.get("observed") or not textured \
            or (extras.get("provenance") or {}).get("appearance") != "photo-observed":
        return ["relief on a node that is not a photographed wall"]
    if stub_parts and extras.get("part") in stub_parts:
        return ["relief on a stub part"]
    return []


def audit_points(r: dict, wall, points: np.ndarray, track: np.ndarray | None, others=None, min_points: int = 12,
                 prm: ReliefParams = DEFAULT_RELIEF) -> dict:
    """Independent re-measurement of one built element from the raw SfM points (no relief map, no levels, no
    cells): the median offset of the points inside its box (shrunk by up to 0.15 m; below the springing line of an
    arch) from its record's photographed frame surface must agree with its claimed offset within
    max(0.15 m, 50 %). Returns {"points", "measured_m", "claimed_m", "ok"} ("ok": None when too few points)."""
    claimed = r.get("offset_m")
    fr = r.get("frame") or {}
    if claimed is None or "c0" not in fr:
        return {"points": 0, "measured_m": None, "claimed_m": claimed, "ok": False, "why": "no frame / offset"}
    U, _ = wall_points(wall, points, np.full(len(points), prm.min_track) if track is None else track, prm, others)
    w, h = r["u1"] - r["u0"], r["z_top"] - r["z_bottom"]
    su, sz = min(0.15, 0.25 * w), min(0.15, 0.25 * h)
    ztop = r["z_top"] - sz
    if r.get("arched") and r.get("spring_z") is not None:
        ztop = min(ztop, float(r["spring_z"]))
    m = (U[:, 0] >= r["u0"] + su) & (U[:, 0] <= r["u1"] - su) & (U[:, 2] >= r["z_bottom"] + sz) & (U[:, 2] <= ztop)
    off = U[m, 1] - (fr["c0"] + fr["c1"] * U[m, 0])
    if m.sum() < min_points:
        return {"points": int(m.sum()), "measured_m": None, "claimed_m": claimed, "ok": None}
    meas = float(np.median(off))
    ok = abs(meas - claimed) <= max(0.15, 0.5 * abs(claimed))
    return {"points": int(m.sum()), "measured_m": round(meas, 3), "claimed_m": claimed, "ok": bool(ok)}
