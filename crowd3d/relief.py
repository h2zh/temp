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

Objects in front of the wall (poles, trunks, lamp posts) are not structure: standing_objects (raw points) and
free_standing_strips (the relief map's strips) find what stands off the wall with the wall seen behind it and an empty
depth gap between them; its points are dropped and the wall re-measured from what is seen behind it.

Guards (measure_relief): an element becomes geometry only when
  G-R1  its wall has >= 2000 points, >= 15 per m2, data on >= 20 % of its cells, the photographed front surface
        within 0.6 m of the frame wall, and >= 6 photos seeing the wall in each photo half;
  G-R2  it is found again (bbox IoU >= 0.3, same type and sign) in BOTH halves of BOTH photo splits -- the
        interleaved one (cameras sorted by bearing around the building, alternately) and the viewpoint-disjoint one
        (the photos seeing the wall split at their median bearing) -- from points RE-TRIANGULATED from that half's
        observations only, with the halves' depths within max(0.1 m, 50 %), every measured edge within 0.3 m, and in
        each half as the same element: its sides (a band: bottom and top) within 0.6 m of its own (split_box_m: a box
        merged over two bays matches one of them by IoU, not by its sides);
  G-R3  >= 1 of its edges lies at a 3-D line cluster (<= 0.2 m) AND that many confirmed edges are unlikely by chance
        (line_null: P <= 0.1 from each edge's chance rate -- the share of positions within 2 m of the element, away
        from its own edges, that some cluster would 'confirm');
  G-R4  its size is inside the detector's limits and its depth / projection <= 1.5 m;
  G-R5  >= 50 % of its box lies on observed texels of the wall's facade and >= 50 % of its surface is seen
        unobstructed by most photos seeing the wall (dense points >= 0.3 m in front of it, at any distance);
  G-R6  the raw points in its outline as built (an arch's head included) re-measure its offset, the points beside it
        show its step, and a recess holds no structure inside it, measured there (an interior the points cannot
        measure is no pass) (audit_points; interior_structure: a pier between two bays or a mullion at the frame plane
        across >= 2/3 of it; a horizontal member across it -- an entablature between two orders, standing only part
        of the way out on real points -- whose pooled rows rise above the recess on both sides; the detector itself
        cuts a recess at such a vertical strip, which becomes a vertical element guarded like the others, and at such
        a horizontal member, into two openings);
  G-R7  no measured edge within 0.3 m of a free-standing object;
  and a nested opening only inside an accepted parent. Recesses are cut into bays only at verticals the points
  confirm (G-R2, G-R6, G-R7: halves_cutters, which the publication audit derives again itself).
D24 (decided by the product owner 2026-10-24, on the condition that the merged-bays gap -- interior_structure -- was
closed first): an element that passes every POINT-based guard (G-R1, G-R2 in both photo splits, G-R4, G-R5, G-R6
with its ring and interior checks, G-R7, an accepted parent) while its 3-D line evidence is only at chance (>= 1 edge
at a line cluster, line_null > line_null_p: a striped or densely lined facade puts a line near any edge) is built on
the points alone (apply_d24), but only on a model whose common scale is established on its own data
(d24_scale_condition; d24_allowed), only when EVERY edge of it, limit edges included, is found again in both halves of
both photo splits (d24_halves_check: the photo split is its only second channel), and only where a bottom / top that is
its register's boundary comes from a band accepted itself (register_bounds: a rejected band -- people in front of the
ground floor read as one -- is no measured sill). Its record says so: provenance PROV_D24 ("photo-measured (points, D24)") and a
"d24" block with the G-R3 statistics, both photo splits' measurements of it, the ring beside it and the model's scale
condition. An element with no edge at any line cluster is not D24 (the lines are not at chance there, they are
absent or elsewhere). Elements failing only G-R3 on a model where D24 is not allowed are listed as a possible
decision (line_chance_decisions), never built.
The QA (check_honesty, audit_model, audit_glb in crowd3d.stylize) re-checks the records, the geometry against the
records (geometry_violations; every wall node without records planar; no record copied between walls) and, from the
raw points, every record's offset and interior; audit_model also re-runs every record's photo splits from the 2-D
observations (cut where its own first pass says), and a D24 record's every edge in them."""
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
    frame_min_span: float = 0.75     # ... and its cells form one connected piece over this share of the wall's width
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
    # free-standing objects in front of the wall (poles, trunks, lamp posts): a vertical strip or band whose interior
    # holds >= this share of its points more than max(gap, 1.5 x lmin) BEHIND its own surface is not a solid part of
    # the wall (nothing can be seen behind a pilaster); its points in front of the wall behind it are dropped and the
    # wall re-measured from what is seen behind it
    occluder_behind_share: float = 0.3
    occluder_gap_m: float = 0.15
    occluder_min_points: int = 100
    occluder_standoff_m: float = 0.45   # standing_objects: an object this far in front of the wall on both sides of it
    occluder_wall_ratio: float = 0.3    # ... with the wall seen behind it at >= this x the wall's point density beside it
    occluder_object_points: int = 100   # ... and at least this many points inside it
    occluder_depth_gap_m: float = 0.25  # ... and an empty depth gap this long between the wall and its front (both tests)


DEFAULT_RELIEF = ReliefParams()


# ---------------------------------------------------------------- frames, reference


def to_wall(a2, t2, n2, X):
    a = np.r_[a2, 0.0]
    t = np.r_[t2, 0.0]
    n = np.r_[n2, 0.0]
    r = np.asarray(X, float) - a
    return np.stack([r @ t, r @ n, np.asarray(X, float)[:, 2]], 1)


def wall_points(wall, P, track, prm: ReliefParams = DEFAULT_RELIEF, others=None, drop=None):
    """Points near the wall plane; with `others` (the other walls of the massing), only the points nearer to this
    wall's segment than to any other wall that spans their height (a convex corner's neighbouring face would read
    as a recess). drop: bool per point, points to leave out (a free-standing object's)."""
    U = to_wall(wall.a, wall.tangent, wall.normal, P)
    m = ((U[:, 0] > -0.2) & (U[:, 0] < wall.length + 0.2) & (np.abs(U[:, 1]) < prm.window_m)
         & (U[:, 2] > wall.z0 - 0.3) & (U[:, 2] < wall.z1 + prm.top_extra_m) & (track >= prm.min_track))
    if drop is not None:
        m &= ~np.asarray(drop, bool)
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


def chance_rate(clusters, lo, hi, tol, exclude=(), excl_m=None):
    """Share of positions in [lo, hi) within tol of a cluster: how often a random edge would be 'confirmed'. exclude:
    positions (the element's own edges) whose zones (+- excl_m, default 2 x tol) are left out -- an element's own
    edge clusters are its evidence, not the background rate a chance edge would meet. 1.0 (no evidence of a low rate)
    when no position is left."""
    if hi <= lo:
        return 1.0 if len(exclude) else 0.0
    zz = np.arange(lo, hi, 0.01)
    if len(exclude):
        ex = 2.0 * tol if excl_m is None else excl_m
        zz = zz[np.min(np.abs(zz[:, None] - np.asarray(exclude, float)[None, :]), axis=1) > ex]
        if not len(zz):
            return 1.0
    if not clusters:
        return 0.0
    P = np.array([c["pos"] for c in clusters])
    return float(np.mean(np.min(np.abs(zz[:, None] - P[None, :]), axis=1) <= tol))


CHANCE_WINDOW_M = 2.0      # G-R3's background rate: line clusters within this of an element's edges ...


def _background_chance(WL, kind, span_lo, span_hi, e0, e1, lim_lo, lim_hi, prm: ReliefParams = DEFAULT_RELIEF) -> float:
    """The chance that an edge of an element spanning e0..e1 (along u for kind 'v', z for 'h'; its extent across:
    span_lo..span_hi) meets a 3-D line cluster by accident: the share of positions within CHANCE_WINDOW_M of the
    element (clipped to lim_lo..lim_hi, the wall) and OUTSIDE +- 2 x edge_tol_m of its own two edges that lie within
    edge_tol_m of a cluster (the clusters found with the edges' own cover and photo minimums)."""
    lo, hi = max(e0 - CHANCE_WINDOW_M, lim_lo), min(e1 + CHANCE_WINDOW_M, lim_hi)
    cl = line_clusters(WL, kind, span_lo, span_hi, lo - prm.edge_tol_m, hi + prm.edge_tol_m, prm.local_cover,
                       prm.line_min_photos, prm)
    return chance_rate(cl, lo, hi, prm.edge_tol_m, exclude=(e0, e1))


def _edge_rec(c, pos):
    if c is None:
        return None
    return {"pos": round(c["pos"], 3), "delta": round(c["pos"] - pos, 3), "cover": round(c["cover"], 3),
            "photos": c["n_photos"], "lines": c["n_lines"], "photo_ids": c["photos"], "line_ids": c["lines"]}


def tail_prob(ps, k: int) -> float:
    """P(at least k of independent events with probabilities ps happen) (Poisson-binomial tail)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        p = float(min(max(p, 0.0), 1.0))
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[max(int(k), 0):].sum())


def line_null(el: dict) -> float:
    """How likely the element's line-confirmed edges are by chance: each real edge's chance rate (the share of
    positions near it, along its axis, that some 3-D line cluster would 'confirm') -> P(>= edges_confirmed of them).
    Recorded with every element; G-R3 needs it under ReliefGuards.line_null_p."""
    real = [e for e in (el.get("edges") or {}).values() if not e["limit"]]
    ps = [float(e.get("chance", 1.0)) for e in real]
    el["edge_chance"] = [round(p, 3) for p in ps]
    el["line_p_chance"] = round(tail_prob(ps, el.get("edges_confirmed", 0)), 4) if real else 1.0
    return el["line_p_chance"]


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


def measure_wall(wall, points, track, lines=None, prm: ReliefParams = DEFAULT_RELIEF, others=None,
                 cut_boxes=None) -> WallRelief:
    """The wall's relief from its points (see the module docstring). A strip that is a free-standing object in front
    of the wall (free_standing_strips) is not structure: its points are dropped and the wall re-measured once from
    what the photos see behind it; the objects are listed in stats["occluders"] (guard G-R7 refuses elements with an
    edge at one). cut_boxes: (u0, u1, z_bottom, z_top) of the verticals that may cut recesses into separate openings
    (None: every vertical strip the measurement finds; measure_relief passes the accepted ones)."""
    occ, drop = _object_drop(wall, points, track, prm, others)
    WR = _measure_wall_once(wall, points, track, lines, prm, others, cut_boxes, drop=drop if occ else None)
    if not WR.refused:
        occ2, drop2 = free_standing_strips(WR, len(points), prm)    # a second net, on the strips the relief map found
        if occ2:
            occ, drop = occ + occ2, drop | drop2
            WR = _measure_wall_once(wall, points, track, lines, prm, others, cut_boxes, drop=drop)
    WR.stats["occluders"] = occ
    if occ:
        WR.stats["occluder_points_dropped"] = int(drop.sum())
        if WR.refused:
            WR.refused = "after dropping a free-standing object's points: " + WR.refused
    return WR


def _object_drop(wall, points, track, prm: ReliefParams = DEFAULT_RELIEF, others=None):
    """standing_objects on the wall's raw points (against their densest layer): (objects, drop over all points)."""
    U, idx = wall_points(wall, points, track, prm, others)
    ref = reference_plane(U, prm)
    if ref is None:
        return [], np.zeros(len(points), bool)
    return standing_objects(U[:, 0], U[:, 1] - (ref["c0"] + ref["c1"] * U[:, 0]), U[:, 2], idx, len(points), wall, prm)


def independent_frame(wall, points, track, prm: ReliefParams = DEFAULT_RELIEF, others=None) -> dict:
    """The wall's G-R1 verdict and frame re-derived from the raw points alone, nothing from any record (the
    publication audit compares every record's frame with it and re-measures every element against it):
    free-standing objects dropped (standing_objects), then wall_frame. Returns {"refused", "frame" ({"c0", "c1"}:
    d = c0 + c1 u of the photographed frame surface from the model wall plane; None when refused), "ref" (the points'
    reference plane the frame level is measured from -- the plane the measurement selects its 3-D lines by), "stats"}."""
    occ, drop = _object_drop(wall, points, track, prm, others)
    fit = wall_frame(wall, points, track, prm, others, drop if occ else None)
    ref = fit.get("ref")
    return {"refused": fit["refused"], "frame": fit["frame"], "stats": fit["stats"],
            "ref": None if not ref else {"c0": float(ref["c0"]), "c1": float(ref["c1"])}}


def frame_offset_diff(fa: dict | None, fb: dict | None, u0: float, u1: float) -> float:
    """Largest difference (m) between two frames d = c0 + c1 u over u0..u1 (inf when one is missing)."""
    try:
        da = [float(fa["c0"]) + float(fa["c1"]) * u for u in (u0, u1)]
        db = [float(fb["c0"]) + float(fb["c1"]) * u for u in (u0, u1)]
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return max(abs(a - b) for a, b in zip(da, db))


def standing_objects(u, rho, z, idx, n_points: int, wall, prm: ReliefParams = DEFAULT_RELIEF, debug=None):
    """Free-standing objects in front of the wall, from the raw points (before any relief map, so that they cannot
    move the wall's frame level): on 0.1 m columns x 0.5 m rows, a cell whose front surface (q90 of its points)
    stands more than max(occluder_gap_m, 0.15) in front of the front surfaces on BOTH sides of it (0.3-0.9 m away;
    one side at the wall's end) marks something proud; such cells joined into an object at most vert_w[1] wide and
    >= 1 m tall are a free-standing object (a pole, a trunk, a lamp post) when it stands >= occluder_standoff_m in
    front of the wall beside it, holds >= occluder_object_points points, and the wall is SEEN BEHIND it: the
    wall-level points per m2 inside its footprint (5-10 cm in from its sides, 0.3 m from its ends) reach
    occluder_wall_ratio x their density beside it, with an empty depth gap of >= occluder_depth_gap_m between the
    wall and its front (depth_gap). Oblique photos see past a free-standing object; nothing is seen behind a buttress
    or a pier, and its side faces fill the depths between the wall and its front, so it stays. Returns (objects, drop) as free_standing_strips; drop marks
    the object's points (in front of the wall behind it)."""
    drop = np.zeros(n_points, bool)
    du, dz = 0.1, 0.5
    L = float(wall.length)
    zlo, zhi = float(wall.z0), float(wall.z1) + prm.top_extra_m
    nu, nz = max(1, int(math.ceil(L / du))), max(1, int(math.ceil((zhi - zlo) / dz)))
    m = (u >= 0) & (u < L) & (z >= zlo) & (z < zhi)
    if m.sum() < 100:
        return [], drop
    iu = np.clip((u[m] / du).astype(int), 0, nu - 1)
    iz = np.clip(((z[m] - zlo) / dz).astype(int), 0, nz - 1)
    key = iz * nu + iu
    o = np.lexsort((rho[m], key))
    ks, rs = key[o], rho[m][o]
    uk, st_, cnt = np.unique(ks, return_index=True, return_counts=True)
    front = np.full(nz * nu, np.nan)
    ok = cnt >= 4
    front[uk[ok]] = rs[st_[ok] + (0.9 * (cnt[ok] - 1)).astype(int)]
    front = front.reshape(nz, nu)
    gap = max(prm.occluder_gap_m, 0.15)

    def side(lo, hi):
        S = np.full((nz, nu), np.nan)
        stack = []
        for k in range(lo, hi + 1):
            sh = np.full((nz, nu), np.nan)
            if k > 0:
                sh[:, :-k] = front[:, k:]
            else:
                sh[:, -k:] = front[:, :k]
            stack.append(sh)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            S = np.nanmedian(np.stack(stack), axis=0)
        return S
    left, right = side(-9, -3), side(3, 9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        both = np.fmax(left, right)                       # one side missing (the wall's end): the other side
    proud = np.isfinite(front) & np.isfinite(both) & (front > both + gap)
    n, cc = cv2.connectedComponents(proud.astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        rr, cl = np.nonzero(cc == k)
        w_ = (cl.max() - cl.min() + 1) * du
        h_ = (rr.max() - rr.min() + 1) * dz
        if w_ > prm.vert_w[1] or h_ < 1.0 - 1e-9:
            continue
        u0, u1 = cl.min() * du, (cl.max() + 1) * du
        z0, z1 = zlo + rr.min() * dz, zlo + (rr.max() + 1) * dz
        fr_lv = float(np.median(front[rr, cl]))
        sd_lv = float(np.nanmedian(both[rr, cl]))
        if fr_lv - sd_lv < prm.occluder_standoff_m:
            continue                                     # relief of the wall's own scale (a pilaster, a pier)
        mu = min(0.1, 0.25 * (u1 - u0))
        zi0, zi1 = z0 + 0.3, z1 - 0.3
        inner = (u >= u0 + mu) & (u <= u1 - mu) & (z >= zi0) & (z <= zi1)
        nin = int(inner.sum())
        if nin < prm.occluder_object_points or zi1 <= zi0:
            continue
        # the wall seen BEHIND it: wall-level points per m2 inside its footprint against beside it (0.3-0.9 m away)
        wl = rho <= sd_lv + 0.1
        a_in = max((u1 - mu) - (u0 + mu), 1e-6) * (zi1 - zi0)
        sides = (((u >= u0 - 0.9) & (u <= u0 - 0.3)) | ((u >= u1 + 0.3) & (u <= u1 + 0.9))) & (z >= zi0) & (z <= zi1)
        a_sd = (min(u0 - 0.3, L) - max(u0 - 0.9, 0.0) if u0 - 0.3 > 0 else 0.0) \
            + (min(u1 + 0.9, L) - max(u1 + 0.3, 0.0) if u1 + 0.3 < L else 0.0)
        if a_sd <= 0.05:
            continue
        d_in = float((inner & wl).sum()) / a_in
        d_sd = float((sides & wl).sum()) / (a_sd * (zi1 - zi0))
        ratio = d_in / max(d_sd, 1e-9)
        if debug is not None:
            debug.append({"u0": round(float(u0), 2), "u1": round(float(u1), 2), "z0": round(float(z0), 2),
                          "z1": round(float(z1), 2), "front": round(fr_lv, 3), "side": round(sd_lv, 3),
                          "sep": round(fr_lv - sd_lv, 3), "wall_ratio": round(ratio, 3), "n": nin})
        if ratio < prm.occluder_wall_ratio:
            continue                                     # nothing seen behind it: solid (a buttress), or unknowable
        gap_m = depth_gap(rho[inner], sd_lv + 0.1, fr_lv - 0.1)
        if debug is not None:
            debug[-1]["depth_gap"] = round(gap_m, 3)
        if gap_m < prm.occluder_depth_gap_m:
            continue                                     # side faces run from the wall to its front: attached (a pier)
        share = ratio
        sel = (u >= u0 - 0.05) & (u <= u1 + 0.05) & (z >= z0 - dz) & (z <= z1 + dz) & (rho > sd_lv + gap)
        drop[idx[sel]] = True
        out.append({"type": "object", "kind": "free-standing", "u0": round(float(u0), 3), "u1": round(float(u1), 3),
                    "z_bottom": round(float(z0), 3), "z_top": round(float(z1), 3), "front_m": round(fr_lv, 3),
                    "wall_seen_behind": round(share, 3), "wall_behind_m": round(sd_lv, 3), "depth_gap_m": round(gap_m, 3),
                    "points": nin,
                    "dropped": int(sel.sum()), "levels_from": "point reference"})
    return out, drop


def depth_gap(rho: np.ndarray, lo: float, hi: float, empty_share: float = 0.005, step: float = 0.05) -> float:
    """The longest run of empty depth (m) between lo and hi: 5 cm bins each holding < empty_share of the points. A
    pole standing off the wall leaves such a gap between the wall seen behind it and its own surface; a buttress or
    pier, whose side faces run from the wall to its front, does not."""
    if hi - lo < step or not len(rho):
        return 0.0
    edges = np.arange(lo, hi + 1e-9, step)
    if len(edges) < 2:
        return 0.0
    h, _ = np.histogram(rho, bins=edges)
    empty = h < empty_share * len(rho)
    best = run = 0
    for e in empty:
        run = run + 1 if e else 0
        best = max(best, run)
    return float(best * step)


def free_standing_strips(WR: WallRelief, n_points: int, prm: ReliefParams = DEFAULT_RELIEF):
    """Vertical strips (proud or frame) and projecting bands of a measured wall that are NOT solid parts of it:
    >= occluder_behind_share of the points inside the strip (edges and register ends left out) lie more than
    max(occluder_gap_m, 1.5 x lmin) behind the strip's own surface. A pilaster, buttress or cornice hides the wall
    behind it from every photo; a pole, trunk or lamp post in front of the wall does not (oblique photos see past
    it), so the wall shows up behind it as a second, deeper layer. Returns (objects, drop): per object its box,
    surface, the share behind and the level of the wall behind it; drop marks (over all n_points) its points in front
    of that wall behind it, over the strip's width (+- 5 cm) and height."""
    drop = np.zeros(n_points, bool)
    if not WR.maps or WR.refused:
        return [], drop
    mp = WR.maps
    u, rho, z = mp["P3"]
    idx = mp["idx"]
    F, lmin = float(mp["F"]), float(mp["lmin"])
    gap = max(prm.occluder_gap_m, 1.5 * lmin)
    cands = []
    for v in WR.verticals:
        surf = F + float(v.get("register_frame") or 0.0) + (float(v["projection"]) if v.get("how") == "proud" else 0.0)
        mu = min(0.1, 0.25 * (v["u1"] - v["u0"]))
        cands.append((v, "u", v["u0"] + mu, v["u1"] - mu, v["z_bottom"] + 0.3, v["z_top"] - 0.3, surf,
                      v["u0"] - 0.05, v["u1"] + 0.05, v["z_bottom"], v["z_top"]))
    for b in WR.bands:
        if b["sign"] <= 0:
            continue
        mz = min(0.1, 0.25 * (b["z_top"] - b["z_bottom"]))
        cands.append((b, "z", 0.3, WR.length - 0.3, b["z_bottom"] + mz, b["z_top"] - mz, F + float(b["level"]),
                      -0.3, WR.length + 0.3, b["z_bottom"] - 0.05, b["z_top"] + 0.05))
    out = []
    for e, axis, a0, a1, z0, z1, surf, du0, du1, dz0, dz1 in cands:
        m = (u >= a0) & (u <= a1) & (z >= z0) & (z <= z1)
        n = int(m.sum())
        if n < prm.occluder_min_points:
            continue
        behind = rho[m] < surf - gap
        share = float(behind.mean())
        if share < prm.occluder_behind_share:
            continue
        back = float(np.median(rho[m][behind]))
        if depth_gap(rho[m], back + 0.1, surf - 0.1) < prm.occluder_depth_gap_m:
            continue                                     # side faces join it to the wall behind: attached
        sel = (u >= du0) & (u <= du1) & (z >= dz0) & (z <= dz1) & (rho > back + gap)
        drop[idx[sel]] = True
        out.append({"type": e["type"], "kind": e["kind"], "u0": round(float(e["u0"]), 3), "u1": round(float(e["u1"]), 3),
                    "z_bottom": round(float(e["z_bottom"]), 3), "z_top": round(float(e["z_top"]), 3),
                    "surface_m": round(surf - F, 3), "behind_share": round(share, 3), "behind_level_m": round(back - F, 3),
                    "points": n, "dropped": int(sel.sum())})
    return out, drop


def wall_frame(wall, points, track, prm: ReliefParams = DEFAULT_RELIEF, others=None, drop=None) -> dict:
    """G-R1 and the wall's frame level from its points (the first stage of the measurement; independent_frame
    repeats it for the publication audit): the points near the wall (wall_points, `drop` left out), their reference
    plane with the yaw of the front surface, the relief map (quantized to the wall's levels), the frame level F (the
    mode over rows of each row's front surface) -- refused with too few points, no rows with enough data, data on
    too few cells, the frame standing more than max_ref_offset_m off the model wall, or a frame that is not the
    wall's front surface: the cells at the frame level or prouder must hold one connected piece spanning
    frame_min_span of the wall's photographed width (a sheet, scaffold or neighbour's facade in front of part of the
    wall would otherwise become the frame, and a wall with two front planes has no single one). Returns {"refused":
    "" or why, "frame": {"c0", "c1"} (d = c0 + c1 u of the frame surface from the model wall plane) or None, "ref",
    "F", "stats"} and, when measured, the maps the measurement continues with."""
    out: dict[str, Any] = {"refused": "", "frame": None, "ref": None, "F": None, "stats": {}}
    st_ = out["stats"]
    U, idx = wall_points(wall, points, track, prm, others, drop=drop)
    st_["points"] = int(len(U))
    out["U"], out["idx"] = U, idx
    ref = reference_plane(U, prm)
    out["ref"] = ref
    area = wall.length * (wall.z1 - wall.z0)
    if ref is None or len(U) < prm.min_points or len(U) < prm.min_density * area:
        out["refused"] = f"too few SfM points for relief ({len(U)}, {len(U) / max(area, 1e-9):.1f} per m2)"
        return out
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
    out.update(u=u, z=z, rho=rho, raw=raw, cnt=cnt, sig=sig, M=M, levels=levels, zrow=zrow, lmin=lmin, front=front,
               cov=cov, zlo=zlo, zhi=zhi, cell=cell)
    if not np.isfinite(F):
        out["refused"] = "no rows with enough data"
        return out
    out["F"] = F
    off = ref["c0"] + ref["c1"] * L / 2 + F
    st_["levels"] = None if levels is None else [round(float(x), 3) for x in levels]
    st_.update({"sigma_cell": round(sig, 4), "level_min": round(lmin, 3), "frame_offset_m": round(off, 3),
                "rows": nz, "cols": nu, "cells_with_data": int(np.isfinite(M).sum()),
                "data_share_in_wall": round(float(np.isfinite(M[zrow < wall.z1]).mean()), 3)})
    if st_["data_share_in_wall"] < prm.min_data_share:
        out["refused"] = f"relief data on {st_['data_share_in_wall']:.0%} of the wall's cells (< {prm.min_data_share:.0%})"
        return out
    if abs(off) > prm.max_ref_offset_m:
        out["refused"] = f"the photographed frame stands {off:+.2f} m off the frame wall"
        return out
    # one front surface: the frame (and prouder) cells of the wall's rows form one connected piece across the wall
    A_ = M[zrow < wall.z1] - F
    fr = (np.isfinite(A_) & (A_ > -lmin)).astype(np.uint8)
    cols = np.flatnonzero(np.isfinite(A_).any(axis=0))
    n_, _, cs_, _ = cv2.connectedComponentsWithStats(fr, connectivity=8)
    span = 0.0
    if n_ > 1 and len(cols):
        k_ = 1 + int(np.argmax(cs_[1:, cv2.CC_STAT_AREA]))
        span = float(cs_[k_, cv2.CC_STAT_WIDTH]) / float(cols.max() - cols.min() + 1)
    st_["frame_span"] = round(span, 3)
    if span < prm.frame_min_span:
        out["refused"] = (f"the frame level is not the wall's front surface: its largest connected piece spans "
                          f"{span:.0%} of the photographed width (< {prm.frame_min_span:.0%}; something in front of "
                          f"part of the wall, or two front planes)")
        return out
    out["frame"] = {"c0": round(float(ref["c0"] + F), 4), "c1": round(float(ref["c1"]), 5)}
    return out


def _measure_wall_once(wall, points, track, lines=None, prm: ReliefParams = DEFAULT_RELIEF, others=None,
                       cut_boxes=None, drop=None) -> WallRelief:
    t0 = time.perf_counter()
    WR = WallRelief(wall.index, getattr(wall, 'part', None), float(wall.length), float(wall.z0), float(wall.z1), None)
    fit = wall_frame(wall, points, track, prm, others, drop)
    WR.ref = fit["ref"]
    WR.stats.update(fit["stats"])
    if fit["refused"]:
        WR.refused = fit["refused"]
        WR.stats["seconds"] = round(time.perf_counter() - t0, 3)
        return WR
    ref, F = fit["ref"], fit["F"]
    WR.frame = F
    u, z, rho, raw, cnt, M = fit["u"], fit["z"], fit["rho"], fit["raw"], fit["cnt"], fit["M"]
    zrow, lmin, front, cov, cell = fit["zrow"], fit["lmin"], fit["front"], fit["cov"], fit["cell"]
    zlo, zhi, idx = fit["zlo"], fit["zhi"], fit["idx"]
    nz, nu = M.shape
    L = wall.length
    P3 = (u, rho, z)
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
            # the background rate over the wall's height, away from this band's own edges
            ch_h = chance_rate(strong, wall.z0, wall.z1 + prm.top_extra_m, prm.edge_tol_m,
                               exclude=(el["z_bottom"], el["z_top"]))
            # a band's limit edges: the ground and the top of the measurement window -- by position, the same rule
            # line_evidence recomputes in the audit (verify-r1 V5a: an edge at the top of the DATA counted as a limit
            # here and as a real edge there); an edge where the data ends is an edge to measure, not a limit
            for name, ez, limit in (("bottom", el["z_bottom"], el["z_bottom"] <= wall.z0 + 0.15),
                                    ("top", el["z_top"], el["z_top"] >= zhi - 0.05)):
                c = None if limit else nearest_cluster(strong, ez, prm.edge_tol_m)
                edges[name] = {"pos": round(ez, 3), "limit": bool(limit), "line": _edge_rec(c, ez), "chance": round(ch_h, 3)}
            _confirm(el, edges, lambda e, real: bool(real) and all(x["line"] for x in real))
            line_null(el)
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
        def add_vertical(u0, u1, zb, zt, how, proj, share, origin=None):
            """A vertical element u0..u1 over zb..zt of this register (sides refined from the raw points, its 3-D
            line evidence), appended to WR.verticals."""
            kind = "corner-pier" if (u0 <= 0.25 or u1 >= L - 0.25) else ("buttress" if proj >= 0.5 else "pilaster")
            el = {"type": "vertical", "kind": kind, "how": how, "u0": round(u0, 3), "u1": round(u1, 3),
                  "z_bottom": zb, "z_top": zt, "width": round(u1 - u0, 3), "projection": round(proj, 3),
                  "share": None if share is None else round(share, 3), "register_frame": reg["frame"]}
            if origin is not None:          # found inside an opening (interior_structure), over that opening's height
                el["found_inside"] = origin
                el["register_z"] = [reg["z_bottom"], reg["z_top"]]
            for key_, lim_ in (("u0", u0 <= 0.25), ("u1", u1 >= L - 0.25)):
                if lim_:
                    continue
                r_ = refine_side(P3, "u", el[key_], zb + 0.2, zt - 0.2, lmin)
                if r_ is not None:
                    el[key_ + "_cell"], el[key_] = el[key_], round(r_, 3)
            if el["u1"] - el["u0"] < prm.vert_w[0] - 1e-9 and origin is not None:
                el["u0"], el["u1"] = round(u0, 3), round(u1, 3)     # a refinement that collapses the strip: keep it
            u0, u1 = el["u0"], el["u1"]
            el["width"] = round(u1 - u0, 3)
            vloc = line_clusters(WL, "v", zb, zt, u0 - 0.5, u1 + 0.5, prm.local_cover, prm.line_min_photos, prm)
            edges = {}
            el["chance_edge"] = round(_background_chance(WL, "v", zb, zt, u0, u1, 0.0, L, prm), 3)
            for name, eu, limit in (("left", u0, u0 <= 0.25), ("right", u1, u1 >= L - 0.25)):
                c = None if limit else nearest_cluster(vloc, eu, prm.edge_tol_m)
                edges[name] = {"pos": round(eu, 3), "limit": bool(limit), "line": _edge_rec(c, eu),
                               "chance": el["chance_edge"]}
            _confirm(el, edges, lambda e, real: bool(real) and all(x["line"] for x in real))
            line_null(el)
            WR.verticals.append(el)
            return el

        if H >= prm.vert_min_h:
            for c0, c1, how in strips:
                w_ = (c1 - c0) * cell
                if not (prm.vert_w[0] - 1e-9 <= w_ <= prm.vert_w[1] + 1e-9):
                    continue
                if how == "proud":
                    cells = Rr[:, c0:c1][labr[:, c0:c1] == 1]
                    proj = float(np.percentile(cells, 75)) if len(cells) else 0.0
                else:                                                   # relief over the recesses beside it
                    side = np.r_[max(0, c0 - 3):c0, c1:min(Rr.shape[1], c1 + 3)]
                    sv = Rr[:, side][labr[:, side] == -1] if len(side) else np.zeros(0)
                    if not len(sv):
                        continue
                    proj = float(-np.median(sv))
                add_vertical(c0 * cell, c1 * cell, reg["z_bottom"], reg["z_top"], how, proj,
                             float((colp if how == "proud" else colf)[c0:c1].mean()))
        # openings: recesses, cut into bays at the vertical strips (a thin pilaster must not join two bays)
        nrow = Rr.shape[0]

        def cut_rows(b):
            """The register rows a cutter box (u0, u1, z_bottom, z_top) cuts: all of them when it spans most of the
            register, its own rows when it is a strip found inside one opening (>= vert_min_h tall), else None."""
            ov = min(b[3], reg["z_top"]) - max(b[2], reg["z_bottom"])
            if ov >= 0.5 * (reg["z_top"] - reg["z_bottom"]) - 1e-9:
                return 0, nrow
            if ov >= prm.vert_min_h - 1e-9:
                return (int(np.clip(math.floor((reg["z_top"] - min(b[3], reg["z_top"])) / cell + 1e-6), 0, nrow)),
                        int(np.clip(math.ceil((reg["z_top"] - max(b[2], reg["z_bottom"])) / cell - 1e-6), 0, nrow)))
            return None

        def find(boxes, hcuts=()):
            rec = (labr == -1).astype(np.uint8)
            used = []
            for b in boxes:
                rows = cut_rows(b)
                if rows is None:
                    continue
                c0_ = int(round(b[0] / cell))
                rec[rows[0]:rows[1], c0_:max(int(round(b[1] / cell)), c0_ + 1)] = 0
                used.append(b)
            for hb in hcuts:            # a horizontal member inside an opening: its rows over the opening's columns
                r0_ = int(np.clip(math.floor((reg["z_top"] - hb[3]) / cell + 1e-6), 0, nrow))
                r1_ = int(np.clip(math.ceil((reg["z_top"] - hb[2]) / cell - 1e-6), 0, nrow))
                c0_ = int(np.clip(math.floor(hb[0] / cell + 1e-6), 0, rec.shape[1]))
                c1_ = int(np.clip(math.ceil(hb[1] / cell - 1e-6), c0_ + 1, rec.shape[1]))
                rec[r0_:r1_, c0_:c1_] = 0
            flds = []
            fnd = _find_openings(rec, Rr, labr, reg, wall, L, cell, lmin, WL, prm, level=1, fields=flds, P3=P3)
            return fnd, flds, used

        def members(fnd, hcuts):
            """The horizontal members (interior_structure's horizontal strips, strict) inside the openings `fnd`
            that are not at an opening's own bottom or top (there its edge is off: it fails G-R6 instead) and not cut
            yet: (u0, u1, z0, z1) cuts over the opening's cell columns -- a lower order and an upper one, or a bay
            and the window above it, become two openings, each guarded on its own."""
            out_ = []
            for op in fnd:
                if op["z_top"] - op["z_bottom"] < prm.vert_min_h:
                    continue
                iv = _opening_interior(op, P3, F)
                bz0, bz1 = interior_box(op)[2:]
                cu0, cu1 = float(op.get("u0_cell", op["u0"])), float(op.get("u1_cell", op["u1"]))
                for a, b in iv["h"]:
                    if a <= bz0 + INTERIOR_BIN_M + 1e-6 or b >= bz1 - INTERIOR_BIN_M - 1e-6:
                        continue
                    if any(min(cu1, x[1]) - max(cu0, x[0]) > 0 and min(b, x[3]) - max(a, x[2]) > 0
                           for x in list(hcuts) + out_):
                        continue
                    out_.append((round(cu0, 3), round(cu1, 3), round(a, 3), round(b, 3)))
            return out_

        def hsplit(boxes, hcuts):
            """find() with the horizontal members found inside the openings as cuts (up to two passes)."""
            hcuts = list(hcuts)
            fnd, flds, used_ = find(boxes, hcuts)
            for _ in range(2):
                nh = members(fnd, hcuts)
                if not nh:
                    break
                hcuts += nh
                fnd, flds, used_ = find(boxes, hcuts)
            return fnd, flds, used_, hcuts

        in_reg = lambda v: tuple(v.get("register_z") or (v["z_bottom"], v["z_top"])) == (reg["z_bottom"], reg["z_top"])
        boxes = [(v["u0"], v["u1"], v["z_bottom"], v["z_top"]) for v in WR.verticals if in_reg(v)]
        found, fields, used, hcuts = hsplit(boxes, [])
        # structure at the frame plane INSIDE an opening (interior_structure on the raw points: a pier between two
        # bays the strip detection missed) becomes a vertical element of its own, guarded like every other vertical,
        # and the opening is split there -- with every detected vertical as a cutter; when only the given (accepted)
        # verticals cut (cut_boxes), a strip found again here is a vertical again but cuts only if it was accepted,
        # and an opening still spanning it fails its re-measurement (audit_points (4)): never one recess over a pier
        seen_strips: list[tuple[float, float]] = []
        for _ in range(2):
            new = []
            for k_, op in enumerate(found):
                if op["z_top"] - op["z_bottom"] < prm.vert_min_h:
                    continue
                iv = _opening_interior(op, P3, F)
                bu0, bu1 = interior_box(op)[:2]
                for a, b in iv["v"]:
                    if b - a > prm.vert_w[1]:
                        continue                    # wider than any pier: the opening is mostly wall (it fails G-R6)
                    if a <= bu0 + INTERIOR_BIN_M + 1e-6 or b >= bu1 - INTERIOR_BIN_M - 1e-6:
                        continue                    # at its own side: the opening's edge is off, no pier between two
                        #                             recesses (it fails G-R6's interior check instead)
                    taken = [(v["u0"], v["u1"]) for v in WR.verticals if in_reg(v)] + seen_strips + [(x[0], x[1]) for x in new]
                    if any(min(b, x1) - max(a, x0) > 0 for x0, x1 in taken):
                        continue
                    new.append((a, b, op, k_))
            if not new:
                break
            for a, b, op, k_ in new:
                seen_strips.append((a, b))
                v = add_vertical(a, b, op["z_bottom"], op["z_top"], "frame", float(op["depth"]), None,
                                 origin=[op["u0"], op["u1"], op["z_bottom"], op["z_top"]])
                boxes.append((v["u0"], v["u1"], v["z_bottom"], v["z_top"]))
            found, fields, used, hcuts = hsplit(boxes, hcuts)
        if cut_boxes is not None:                   # only the given (accepted) verticals cut (members still do)
            found, fields, used, hcuts = hsplit(list(cut_boxes), hcuts)
        reg["cutters"] = [[round(b[0], 3), round(b[1], 3)] for b in used]
        reg["member_cuts"] = [list(h) for h in hcuts]
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
               "rowlab": rowlab, "front": front, "cov": cov, "WL": WL, "zf": zf, "pf": pf, "idx": idx, "P3": P3}
    return WR


def _opening_interior(el: dict, P3, F: float, g: "ReliefGuards | None" = None) -> dict:
    """interior_structure of a level-1 opening the detector just found, on the measurement's own points (P3: u, rho
    relative to the point reference, z; F its frame level), with the pipeline's strict limits."""
    g = g or DEFAULT_GUARDS
    u, rho, z = P3
    base = float(el.get("register_frame") or 0.0)
    claimed = base - float(el["depth"])
    if (el.get("level") or 1) != 1 or claimed >= base:
        return {"v": [], "h": [], "ok": True}
    wnd = max(0.3, abs(claimed - base))
    off = rho - F
    bu0, bu1, bz0, bz1 = interior_box(el)
    m = (off >= claimed - wnd) & (off <= base + wnd) & (u >= bu0) & (u <= bu1) & (z >= bz0) & (z <= bz1)
    return interior_structure(u[m], off[m], z[m], dict(el, type="opening"), claimed, base, g, strict=True)


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
        el["chance_jamb"] = round(_background_chance(WL, "v", zb_o, (zs if arched else zt_o), u0, u1, 0.0, L, prm), 3)
        el["chance_sill"] = round(_background_chance(WL, "h", u0, u1, zb_o, zt_o, wall.z0, wall.z1 + prm.top_extra_m, prm), 3)
        for name, ep, cl, limit in (("left", u0, vloc, False), ("right", u1, vloc, False),
                                    ("bottom", zb_o, hloc, zb_o <= max(wall.z0 + 0.3, reg["z_bottom"] + 0.05)),
                                    ("top", zt_o, hloc, arched or zt_o >= reg["z_top"] - 0.05)):
            c = None if limit else nearest_cluster(cl, ep, prm.edge_tol_m)
            edges[name] = {"pos": round(ep, 3), "limit": bool(limit), "line": _edge_rec(c, ep),
                           "chance": el["chance_jamb"] if name in ("left", "right") else el["chance_sill"]}
        _confirm(el, edges, lambda e, real: sum(1 for x in real if x["line"]) >= 2
                 and (e["left"]["line"] is not None or e["right"]["line"] is not None))
        line_null(el)
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
    split_edge_m: float = 0.3           # ... and every measured (not limit) edge within this between the two halves
    split_box_m: float = 0.6            # ... and found again AS THE SAME element: in each half the edges a merge or a
                                        #     split moves (an opening's / a vertical's sides, a band's bottom and top)
                                        #     within this of the element's own (verify-r2 N10x: IoU >= 0.3 also
                                        #     matches one bay of a record merged over two bays and their pier)
    min_line_edges: int = 1             # G-R3: edges at a 3-D line cluster (within ReliefParams.edge_tol_m) ...
    line_null_p: float = 0.1            # ... and that many confirmed edges unlikely by chance (line_null <= this)
    max_depth_m: float = 1.5            # G-R4: depth / projection (sizes: the detector's limits)
    min_observed_share: float = 0.5     # G-R5: share of the element's box on observed facade texels
    occluder_clear_m: float = 0.3       # G-R7: no measured u-edge within this of a free-standing object in front
    min_clear_share: float = 0.5        # G-R5: share of the element's surface samples that the photos seeing the wall
    clear_margin_m: float = 0.3         # ... mostly see unobstructed by SfM points this far or more in front of the
                                        #     element's surface, at ANY distance (a pole 1 m in front counts) ...
    clear_voxel_points: int = 5         # ... that form a surface (>= this many points per 0.2 m voxel; not the sparse
                                        #     SfM noise around a facade)
    tri_max_err_px: float = 2.0         # split-half re-triangulation: every used observation within this ...
    tri_min_angle_deg: float = 2.0      # ... and a triangulation angle of at least this
    tri_max_obs: int = 12               # observations per point used (the first by camera index)
    recheck_abs_m: float = 0.05         # G-R6 (audit_points): the points' median offset in the element's box agrees with
    recheck_rel: float = 0.3            # ... the claimed offset within max(abs, rel x |relief|) ...
    recheck_k: float = 3.0              # ... and differs from the element's base by > max(k x standard error, min)
    recheck_min_m: float = 0.04
    ring_step_share: float = 0.5        # ... and the points right beside it show at least this share of its relief
    # ... and (a recess: an opening, a setback, a groove) no structure at the frame plane INSIDE it (interior_structure):
    # a strip whose 0.1 m bins are, over >= interior_share of the element's extent across them, prouder than the claimed
    # back by half its relief (to the record's base or to the frame, whichever is nearer) -- a pier or pilaster between
    # two bays, a mullion, the entablature between two orders -- this wide (vertical) or this tall (horizontal) means
    # the claim spans real structure (two bays merged into one, the wall between them deleted)
    interior_share: float = 2.0 / 3.0
    interior_w_m: float = 0.3
    interior_h_m: float = 0.25
    # ... or (horizontal) a MEMBER across it: rows whose pooled profile rises >= interior_member x the relief above
    # the recess's back both below and above it (the mean of the two flanks, never below the claimed back), over
    # >= interior_share of the width -- an entablature between two orders, a transom: on real points such a band
    # stands only part of the way out (verify-r1: the Baptistery's entablature 0.3-0.7 of the bays' relief, and on
    # the weakest bay a 0.3 m run at 0.2; inside the honest lower and upper parts no row reaches even the strict limits)
    interior_member: float = 0.2
    # the pipeline's own G-R6 refuses at these (stricter) limits -- share, width, height, member -- so that an element
    # it accepts passes the audit's check on a frame re-derived up to FRAME_TOL_M away; the detector splits a recess at
    # such a vertical strip, and at such a horizontal strip between two parts of it
    interior_guard: tuple = (0.6, 0.2, 0.2, 0.175)
    # D24 (an element built on the points alone, its 3-D lines at chance): the photo split is its only second channel,
    # so EVERY edge it builds -- a sill at the ground or at a register boundary too, not only the edges G-R2 compares
    # -- must be found again: in both halves of both splits, each edge within d24_edge_m between the two halves and
    # within d24_box_m of the element's own (verify-r1: a half that saw only the upper arch of a merged bay, or a sill
    # at the top of a crowd standing in front of one half's photos)
    d24_edge_m: float = 0.3
    d24_box_m: float = 0.6


DEFAULT_GUARDS = ReliefGuards()
DEPTH_KEY = {"band": "level", "vertical": "projection", "opening": "depth"}
SAME_KEYS = {"band": ("z_bottom", "z_top"), "vertical": ("u0", "u1"), "opening": ("u0", "u1")}   # split_match: split_box_m
GUARD_NAMES = ("G-R1_photos_per_half", "G-R2_split_half", "G-R3_line_edge", "G-R4_depth", "G-R5_observed_texels",
               "parent_accepted", "G-R6_points_recheck", "G-R7_clear_of_objects")


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


def side_halves(cameras, wall, seen: list[int]) -> tuple[np.ndarray, list[int]]:
    """Viewpoint-disjoint photo halves for one wall: the photos that see it (`seen`, in front of it) sorted by their
    bearing from the wall's middle (relative to its normal) and split at the median -- half 0 looks from the side of
    the wall's start, half 1 from its end; every other camera in neither (-1). Returns the half per camera and the
    number of photos per half. Unlike split_halves (alternating), neighbouring viewpoints land in the same half, so
    an element both halves find is seen from two different directions."""
    if not len(cameras):
        return np.zeros(0, int), [0, 0]
    m = 0.5 * (np.asarray(wall.a, float) + np.asarray(wall.b, float))
    C = np.array([np.asarray(c.center, float)[:2] for c in cameras]).reshape(-1, 2) - m
    ang = np.arctan2(C @ wall.tangent, C @ wall.normal)
    front = C @ wall.normal > 0
    sv = [c for c in seen if 0 <= c < len(cameras) and front[c]]
    half = np.full(len(cameras), -1, int)
    if not sv:
        return half, [0, 0]
    order = sorted(sv, key=lambda c: (ang[c], c))
    k = len(order) // 2
    for c in order[:k]:
        half[c] = 0
    for c in order[k:]:
        half[c] = 1
    return half, [k, len(order) - k]


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


def clear_view_share(r: dict, wall, front: np.ndarray, cams: list, g=None) -> float:
    """Share of an element's surface (samples on a <= 0.25 m grid over its box, at its offset from its frame
    surface) that the photos seeing the wall mostly see unobstructed: a sample is clear when, of the cameras that
    face it and have it in frame, fewer than half find SfM points of `front` (dense points >= clear_margin_m in front
    of the element's surface, at any distance) nearer along its ray (crowd3d.stylize.occluded_by_points, the facade
    orthophoto's own depth-buffer test, here without its 1.5 m margin)."""
    from .stylize import occluded_by_points, DEFAULT_PARAMS
    g = g or DEFAULT_GUARDS
    fr = r.get("frame") or {}
    if "c0" not in fr or r.get("offset_m") is None:
        return 0.0
    w, h = r["u1"] - r["u0"], r["z_top"] - r["z_bottom"]
    nu, nz = int(np.clip(math.ceil(w / 0.25), 2, 15)), int(np.clip(math.ceil(h / 0.25), 2, 15))
    UU, ZZ = np.meshgrid(np.linspace(r["u0"] + min(0.1, 0.25 * w), r["u1"] - min(0.1, 0.25 * w), nu),
                         np.linspace(r["z_bottom"] + min(0.1, 0.25 * h), r["z_top"] - min(0.1, 0.25 * h), nz))
    UU, ZZ = UU.ravel(), ZZ.ravel()
    D = fr["c0"] + fr["c1"] * UU + float(r["offset_m"])
    t, n = np.r_[wall.tangent, 0.0], np.r_[wall.normal, 0.0]
    X = np.r_[wall.a, 0.0][None] + UU[:, None] * t[None] + D[:, None] * n[None] + ZZ[:, None] * np.array([0.0, 0.0, 1.0])
    nvis = np.zeros(len(X), int)
    nblk = np.zeros(len(X), int)
    for cam in cams:
        uv, dep = cam.project(X)
        ray = np.asarray(cam.center, float)[None] - X
        vis = ((dep > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
               & (ray[:, :2] @ wall.normal > 0))
        if not vis.any():
            continue
        nvis += vis
        nblk += vis & occluded_by_points(cam, front, uv, dep, DEFAULT_PARAMS)
    return float(((nvis >= 1) & (2 * nblk < nvis)).mean())


def clear_of_objects(e: dict, occluders: list, g=None) -> bool:
    """G-R7: an element with a measured (not limit) edge along the wall within occluder_clear_m of a free-standing
    object in front of the wall (free_standing_strips), or lying within its strip, may be that object's silhouette
    or cut by it: refused. Bands (full width) and elements clear of every object pass."""
    g = g or DEFAULT_GUARDS
    if e["type"] == "band":
        return True
    edges = e.get("edges") or {}
    for o in occluders or []:
        if min(e["z_top"], o["z_top"]) - max(e["z_bottom"], o["z_bottom"]) <= 0:
            continue
        lo, hi = o["u0"] - g.occluder_clear_m, o["u1"] + g.occluder_clear_m
        if e["u0"] >= lo and e["u1"] <= hi:
            return False
        for name, key in (("left", "u0"), ("right", "u1")):
            ed = edges.get(name)
            if ed is not None and ed.get("limit"):
                continue
            if lo <= e[key] <= hi:
                return False
    return True


EDGE_KEYS = {"left": "u0", "right": "u1", "bottom": "z_bottom", "top": "z_top"}


def split_match(e: dict, halves: list | None, g: ReliefGuards = DEFAULT_GUARDS) -> tuple[bool, dict]:
    """The element found again in BOTH halves of one photo split (bbox IoU >= split_iou, same type and sign), the two
    halves' depths within max(split_depth_abs_m, split_depth_rel x depth), every measured (not limit) edge of the
    element within split_edge_m between the two halves, and in each half as the same element: its sides (an opening,
    a vertical) or its bottom and top (a band) within split_box_m of the element's own -- a box merged over two bays
    matches one of them by IoU, not by its sides. Returns (ok, the numbers)."""
    key = DEPTH_KEY[e["type"]]
    depth = abs(float(e[key]))
    found, info = [], {}
    for h, name in ((0, "a"), (1, "b")):
        hr = halves[h] if halves is not None and h < len(halves) else None
        if hr is None or hr.refused:
            info[f"iou_{name}"] = 0.0
            found.append(None)
            continue
        sc, f = match_element(e, hr.elements(), g.split_iou)
        info[f"iou_{name}"] = round(sc, 3)
        found.append(f)
    fa, fb = found
    for f, name in ((fa, "a"), (fb, "b")):          # what each half measured (kept in a D24 element's record)
        if f is not None:
            info[name] = {**{k: round(float(f[k]), 3) for k in ("u0", "u1", "z_bottom", "z_top")},
                          "depth": round(float(f[key]), 3)}
    dd = abs(float(fa[key]) - float(fb[key])) if fa is not None and fb is not None else None
    info["depth_diff"] = None if dd is None else round(dd, 3)
    ed_max = None
    if fa is not None and fb is not None:
        ks = ("z_bottom", "z_top") if e["type"] == "band" else ("u0", "u1", "z_bottom", "z_top")
        info["edge_diff"] = {k: round(abs(float(fa[k]) - float(fb[k])), 3) for k in ks}
        real = [EDGE_KEYS[n] for n, ed in (e.get("edges") or {}).items() if not ed.get("limit") and EDGE_KEYS[n] in ks]
        ed_max = max((info["edge_diff"][k] for k in real), default=0.0)
        info["edge_diff_max"] = round(ed_max, 3)
        info["box_diff"] = round(max(abs(float(f[k]) - float(e[k])) for f in (fa, fb) for k in SAME_KEYS[e["type"]]), 3)
    ok = bool(dd is not None and dd <= max(g.split_depth_abs_m, g.split_depth_rel * depth)
              and ed_max is not None and ed_max <= g.split_edge_m + 1e-9
              and info.get("box_diff") is not None and info["box_diff"] <= g.split_box_m + 1e-9)
    return ok, info


def register_bounds(WR: WallRelief, box: dict, wall) -> dict[str, int]:
    """Where an opening's or a vertical's bottom / top is its register's boundary (a LIMIT there, not an edge the
    points measured) above the ground / below the wall top: the band that makes that boundary, as its index in
    WR.bands (WR.elements() order), or -1 when no band does. {"bottom": i, "top": j} for the bounded sides only.
    verify-r1 (crowd): people standing in front of the ground floor read as a proud band; its top became the
    register boundary, and so the sill of the arches above -- an edge nothing measured. D24 builds such an element
    only where the band bounding it is accepted itself (apply_d24)."""
    out: dict[str, int] = {}
    cell = float((WR.maps or {}).get("cell", 0.2))
    reg = box.get("register") or box.get("register_z")
    if reg is None and box.get("type") == "vertical":
        reg = (box.get("z_bottom"), box.get("z_top"))
    if reg is None:
        return out
    try:
        zr0, zr1 = float(reg[0]), float(reg[1])
        zb, zt = float(box["z_bottom"]), float(box["z_top"])
    except (TypeError, ValueError, KeyError, IndexError):
        return out

    def band_at(z, key):
        for i, b in enumerate(WR.bands):
            if abs(float(b.get(key, b[key.replace("_cell", "")])) - z) <= 0.5 * cell + 1e-6:
                return i
        return -1
    if zb <= zr0 + 0.05 + 1e-6 and zr0 > float(wall.z0) + 0.3:
        out["bottom"] = band_at(zr0, "z_top_cell")
    if not box.get("arched") and zt >= zr1 - 0.05 - 1e-6 and zr1 < float(wall.z1) - 0.05:
        out["top"] = band_at(zr1, "z_bottom_cell")
    return out


def guard_elements(WR: WallRelief, halves: list, per_half: list[int], mask: np.ndarray | None, wall,
                   g: ReliefGuards = DEFAULT_GUARDS, recheck=None, clear=None, sides: list | None = None,
                   per_side: list[int] | None = None) -> list[dict]:
    """Every element the full photo set detected on the wall, with its split-half matches, guards and the verdict
    ('accepted': all guards pass; 'builds_geometry': accepted and not a frame strip, which is the wall itself).
    halves: the wall measured from the interleaved photo halves (split_halves); sides: from the viewpoint-disjoint
    halves (side_halves: the photos left / right of the median bearing) -- G-R2 needs both (when given; measure_relief
    always gives them). recheck: callable(element) -> audit_points() result, the independent re-measurement from the
    raw points that the publication audit repeats (G-R6; measure_relief always passes it, unit tests of the other
    guards may not). clear: callable(element) -> clear_view_share() (G-R5's second half; measure_relief passes it)."""
    els = WR.elements()
    nb, nv = len(WR.bands), len(WR.verticals)
    out: list[dict] = []
    frame = {"c0": round(float(WR.ref["c0"] + WR.frame), 4), "c1": round(float(WR.ref["c1"]), 5)} if WR.ref else None
    for i, e in enumerate(els):
        key = DEPTH_KEY[e["type"]]
        depth = abs(float(e[key]))
        ok1, split = split_match(e, halves, g)
        ok2, split2 = (True, None) if sides is None else split_match(e, sides, g)
        if sides is not None and (per_side is None or min(per_side) < g.min_photos_per_half):
            ok2 = False
            split2["refused"] = f"viewpoints too one-sided for the left / right split ({per_side})"
        parent_abs = None
        if e["type"] == "opening" and (e.get("level") or 1) == 2 and e.get("parent") is not None:
            parent_abs = nb + nv + int(e["parent"])
        parent = out[parent_abs] if parent_abs is not None and parent_abs < len(out) else None
        guards = {
            "G-R1_photos_per_half": bool(min(per_half) >= g.min_photos_per_half),
            "G-R2_split_half": bool(ok1 and ok2),
            "G-R3_line_edge": bool(e.get("edges_confirmed", 0) >= g.min_line_edges
                                   and float(e.get("line_p_chance", 1.0)) <= g.line_null_p),
            "G-R4_depth": bool(depth <= g.max_depth_m),
            "G-R5_observed_texels": bool(observed_share(e, wall, mask) >= g.min_observed_share),   # (and below)
            "parent_accepted": bool(e["type"] != "opening" or (e.get("level") or 1) == 1
                                    or (parent is not None and parent["accepted"])),
            "G-R7_clear_of_objects": clear_of_objects(e, WR.stats.get("occluders") or [], g),
        }
        r = {"i": i, "id": f"w{wall.index}-{e['type'][0]}{i}", "type": e["type"], "kind": e["kind"],
             "sign": e.get("sign"), "how": e.get("how"),
             "u0": float(e["u0"]), "u1": float(e["u1"]), "z_bottom": float(e["z_bottom"]), "z_top": float(e["z_top"]),
             "depth_key": key, "depth": float(e[key]), "register_frame": float(e.get("register_frame") or 0.0),
             "arched": bool(e.get("arched", False)), "spring_z": e.get("spring_z"), "level": e.get("level"),
             "parent_el": parent_abs, "frame": frame,
             "edges_real": int(e.get("edges_real", 0)), "edges_confirmed": int(e.get("edges_confirmed", 0)),
             "line_photo_ids": list(e.get("photos") or []),
             "split": split, "photos_per_half": [int(v) for v in per_half], "guards": guards,
             "parent_id": None if parent is None else parent["id"],
             "edge_chance": list(e.get("edge_chance") or []), "line_p_chance": float(e.get("line_p_chance", 1.0))}
        if sides is not None:
            r["split_sides"] = split2
            r["photos_per_side"] = [int(v) for v in (per_side or [0, 0])]
        r["offset_m"] = round(element_offset(e, parent), 3)
        if e["type"] in ("opening", "vertical"):
            r["bounds"] = register_bounds(WR, e, wall)        # band indices = their indices in els
        r["observed_share"] = round(observed_share(e, wall, mask), 3)
        if clear is not None:
            r["clear_share"] = round(clear(r), 3)
            guards["G-R5_observed_texels"] = bool(guards["G-R5_observed_texels"] and r["clear_share"] >= g.min_clear_share)
        if recheck is not None:
            chk = recheck(r)
            r["recheck"] = chk
            guards["G-R6_points_recheck"] = bool(chk.get("ok") is True)
        acc = all(guards.values())
        r["accepted"] = acc
        r["builds_geometry"] = bool(acc and not (e["type"] == "vertical" and e.get("how") == "frame"))
        out.append(r)
    return out


CUT_GUARDS = ("G-R2_split_half", "G-R6_points_recheck", "G-R7_clear_of_objects")


def halves_cutters(els: list[dict], WR: WallRelief) -> list[list[float]] | None:
    """The verticals the photo halves (and the full measurement) are cut at, from a first pass of guard_elements
    (halves measured with every vertical they find as a cutter): the verticals the points confirm -- both photo splits
    find them, the raw points re-measure them, no free-standing object stands at them (CUT_GUARDS) -- as (u0, u1,
    z_bottom, z_top) rounded to the mm, or None when every vertical is confirmed (no re-cut). measure_relief cuts with
    exactly this list and records it (halves_cut_at); the publication audit derives it again from the points the same
    way and never takes it from a record (verify-r2: a record's halves_cut_at emptied to [] made the audit's re-run
    halves merge bays)."""
    acc = [[round(float(e[k]), 3) for k in ("u0", "u1", "z_bottom", "z_top")] for e in els
           if e["type"] == "vertical" and all(e["guards"].get(k, True) for k in CUT_GUARDS)]
    return acc if len(acc) < len(WR.verticals) else None


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


def rhythm_decisions(res: ReliefResult, massing) -> list[dict]:
    """Possible decisions, never applied: a wall whose accepted openings show a measured rhythm (>= 2 openings of
    widths within 15 % of each other) and the walls of the same part that carry no measured relief (nobody
    photographed them, or their measurement was refused). Copying the rhythm there would draw structure nobody
    measured on them: recorded for the user to decide (docs/STRUCTURE_FIRST.md D16), not done."""
    out = []
    for wi, els in sorted(res.elements.items()):
        ops = [e for e in els if e["accepted"] and e["type"] == "opening" and (e.get("level") or 1) == 1]
        widths = sorted(round(e["u1"] - e["u0"], 2) for e in ops)
        if len(widths) < 2:
            continue
        w0 = float(np.median(widths))
        same = [w for w in widths if abs(w - w0) <= 0.15 * w0]
        if len(same) < 2:
            continue
        part = getattr(massing.walls[wi], "part", None)
        measured = {k for k, v in res.elements.items() if any(e["accepted"] for e in v)}
        cand = [w.index for w in massing.walls if getattr(w, "part", None) == part and w.index not in measured
                and getattr(w, "profile", None) is None]
        if not cand:
            continue
        out.append({"part": part, "measured_on": wi, "bay_widths_m": same,
                    "spring_z_m": sorted({round(float(e["spring_z"]), 2) for e in ops if e.get("spring_z") is not None}),
                    "depth_m": round(float(np.median([e["depth"] for e in ops])), 3), "candidates": cand,
                    "status": "not done: copying a measured rhythm onto walls nobody measured needs the user's decision"})
    return out


def d24_halves_check(box: dict, halves: dict | None, g: ReliefGuards = DEFAULT_GUARDS) -> tuple[bool, dict]:
    """D24's photo-split check on EVERY edge of an element (a band: its bottom and top; anything else: all four,
    limit edges included -- ReliefGuards.d24_edge_m): in both halves of both photo splits, each edge within d24_edge_m
    between the two halves and within d24_box_m of the element's own. box: the element (u0, u1, z_bottom, z_top,
    type); halves: {"interleaved": {"a": box, "b": box}, "viewpoint": {...}} -- each half's measurement of it
    (split_match's "a" / "b"). Returns (ok, {split: {"edge_diff_m": max |a - b|, "element_diff_m": max |half -
    element|, "worst": the edge}}; a split without both halves: {"missing": True})."""
    keys = ("z_bottom", "z_top") if box.get("type") == "band" else ("u0", "u1", "z_bottom", "z_top")
    info: dict[str, Any] = {}
    ok = True
    for name in ("interleaved", "viewpoint"):
        h = (halves or {}).get(name)
        a, b = (h.get("a"), h.get("b")) if isinstance(h, dict) else (None, None)
        try:
            dh = {k: abs(float(a[k]) - float(b[k])) for k in keys}
            de = {k: max(abs(float(a[k]) - float(box[k])), abs(float(b[k]) - float(box[k]))) for k in keys}
        except (KeyError, TypeError, ValueError):
            info[name] = {"missing": True}
            ok = False
            continue
        worst = max(keys, key=lambda k: max(dh[k] / g.d24_edge_m, de[k] / g.d24_box_m))
        info[name] = {"edge_diff_m": round(max(dh.values()), 3), "element_diff_m": round(max(de.values()), 3),
                      "worst": worst}
        ok = ok and max(dh.values()) <= g.d24_edge_m + 1e-9 and max(de.values()) <= g.d24_box_m + 1e-9
    return bool(ok), info


def d24_candidate(e: dict, g: ReliefGuards = DEFAULT_GUARDS) -> bool:
    """An element D24 may build on the points alone: not accepted, measured in both photo splits and re-measured from
    the raw points (G-R6 ran), every guard passed but G-R3 (and its parent's acceptance, which D24 itself may settle),
    every one of its edges found again in both halves of both photo splits (d24_halves_check: the photo split is its
    only second channel), and G-R3 failed only because its line evidence is AT CHANCE: >= min_line_edges edges at a
    3-D line cluster whose chance probability is above line_null_p. An element no line cluster confirms at any edge
    is not at chance (the lines are absent there, or elsewhere): not a candidate."""
    gd = e.get("guards") or {}
    if e.get("accepted") or gd.get("G-R3_line_edge", True) or "G-R6_points_recheck" not in gd \
            or e.get("split_sides") is None:
        return False
    if not all(v for k, v in gd.items() if k not in ("G-R3_line_edge", "parent_accepted")):
        return False
    if not d24_halves_check(e, {"interleaved": e.get("split"), "viewpoint": e.get("split_sides")}, g)[0]:
        return False
    return int(e.get("edges_confirmed", 0)) >= g.min_line_edges and float(e.get("line_p_chance", 1.0)) > g.line_null_p


def line_chance_decisions(res: ReliefResult) -> list[dict]:
    """Possible decision, never applied here: elements that pass every guard except G-R3 because the wall's 3-D lines
    are no better than chance there (d24_candidate) and that D24 did not build on this model -- D24 is switched off,
    or this model's scale condition is not met (d24_scale_condition(), res.stats['d24']: where it is not met D24 is
    off for the model), or their parent opening is not accepted. Each list carries the scale condition and why its
    elements are not built (docs/STRUCTURE_FIRST.md D24)."""
    out = []
    d24 = (res.stats or {}).get("d24") or {}
    allowed = d24_allowed(res)
    if not D24_ENABLED:
        status = ("not done: the 3-D lines on this wall are at chance; building these on the points alone (both photo "
                  "splits + the raw-point re-measurement) is D24, which is switched off")
    elif not allowed:
        status = ("not built: D24 is on, but this model's scale condition is not met (" + str(d24.get("why") or
                  "not evaluated (no structure / scale record)") + ")")
    else:
        status = ("not built: its parent opening is not accepted, or its bottom / top is a register boundary made by a "
                  "band that is not accepted, or its held-out photos contradict its sides (per element: refused)")
    for wi, els in sorted(res.elements.items()):
        cand = [e for e in els if d24_candidate(e)]
        if not cand:
            continue
        out.append({"wall": wi, "elements": [{"id": e["id"], "kind": e["kind"], "line_p_chance": e.get("line_p_chance"),
                                              "edge_chance": e.get("edge_chance"), "refused": e.get("d24_refused"),
                                              "builds_geometry": not (e["type"] == "vertical" and e.get("how") == "frame")}
                                             for e in cand],
                    "status": status,
                    # D24's scale condition on this model: nothing is built on the points alone where it is not met
                    "scale_condition": {"allowed": bool(d24.get("allowed")), "by": d24.get("by"),
                                        "why": d24.get("why") or "not evaluated (no structure / scale record)"}})
    return out


# D24 (docs/STRUCTURE_FIRST.md §6): elements that pass every POINT-based guard while their 3-D line evidence is at
# chance are built on the points alone -- decided by the product owner on 2026-10-24 ("同意 D24，但先补上'合并开间'
# 这个漏洞再打开"), switched on once the merged-bays check (interior_structure, G-R6) was in. Whatever the switch,
# d24_scale_condition() decides per model (d24_allowed).
D24_ENABLED = True
D24_DECISION = "D24: product owner, 2026-10-24, on the condition that the merged-bays gap is closed first"
D24_BASIS = ("SfM points only: found in both halves of both photo splits (interleaved, viewpoint-disjoint) and "
             "re-measured from the raw points (offset, ring beside it, nothing at the frame plane inside a recess); "
             "the wall's 3-D lines at chance")


def d24_scale_condition(structure, points: np.ndarray | None, quality: np.ndarray | None = None, lines=None,
                        snap_scale: float = 1.0) -> dict:
    """D24's scale condition (verify-r3: under a ~6 % common scale error the Baptistery's wall 3 still yields
    candidates, its frame within the 0.6 m of G-R1): D24 elements may be built on a model only when its common scale
    is ESTABLISHED on the data the relief measures -- crowd3d.judge.scale_status() of the whole-building similarities
    re-measured here from these points (SfM wall planes) and 3-D lines on the structure's outline (no snap rescale
    > 3 %, a similarity measured all round within 3 % of k = 1) -- or its frame was adopted under D15 WITH the eave
    check (an active frame adoption whose record passes check_frame_record() and carries a passing scale_check: the
    main body's eave measured by >= 2 channels on the adopted outline agrees with the map's); and in either case no
    similarity measured here contradicts it (a channel measured all round off by > 3 % or shifted > 1.5 m: the frame
    is in question on this data, as when the reconstruction itself is mis-scaled). Otherwise D24 is off for this
    model and `why` says why. structure: the crowd3d.structure.StructureModel (None: the OSM-footprint massing, no
    structure frame). Returns {"allowed", "by", "why", "scale_status", "channels", "frame_adoption"}."""
    from . import judge as J
    out: dict[str, Any] = {"allowed": False, "by": None, "why": "", "scale_status": None, "channels": {},
                           "frame_adoption": None}
    rings = getattr(structure, "main_rings", None)
    if structure is None or not rings:
        out["why"] = "no structure frame (OSM parts) to establish the common scale against"
        return out
    frame: dict[str, Any] = {"passed": True}
    if points is not None and len(points):
        frame["similarity"] = J.frame_similarity(np.asarray(points, float), rings, quality)
    L = getattr(lines, "lines", lines)
    if L is not None and hasattr(L, "A") and len(L):
        frame["similarity_lines"] = J.frame_similarity_lines(L, rings)
    for key in ("similarity", "similarity_lines"):
        sm = frame.get(key)
        if sm:
            out["channels"][sm.get("channel", key)] = {k: sm.get(k) for k in ("measured", "at_bound", "k", "t", "ticket",
                                                                                "reason", "bound_why")}
    sst = J.scale_status(frame, snap_scale)
    out["scale_status"] = sst
    contra = {n: c for n, c in out["channels"].items() if c.get("measured") and not c.get("at_bound") and c.get("ticket")}
    if contra:
        out["why"] = ("a whole-building similarity re-measured on this model's data puts the frame in question ("
                      + "; ".join(f"{n}: k {c.get('k')}, shift {c.get('t')}" for n, c in contra.items())
                      + "): the reconstruction's common scale may be off; D24 is off for this model")
        return out
    if sst.get("established"):
        out.update(allowed=True, by="scale_status", why=sst.get("why", ""))
        return out
    # the D15 route needs a similarity measured on THIS model's data that agrees (verify-r1 L: on a one-viewpoint photo
    # set nothing measures the scale, and an adoption recorded on other data would otherwise allow D24 unopposed): a
    # channel measured all round here with k within FRAME_K_TOL of 1, though at a fit bound (off its bounds it has
    # established the scale above already)
    backing = [n for n, c in out["channels"].items()
               if c.get("measured") and c.get("k") is not None and abs(float(c["k"]) - 1.0) <= J.FRAME_K_TOL]
    for a in getattr(structure, "adoptions", None) or []:
        rec = a.get("record") or {}
        if a.get("kind") != "frame" or not a.get("active", True) or not rec:
            continue
        sc = rec.get("scale_check") or {}
        if sc.get("passed") and J.check_frame_record(rec) is None and not backing:
            out["why"] = (f"{sst.get('why', 'the common scale is not established')}; frame {a.get('id')} was adopted "
                          f"under D15 with the eave check, but no similarity measured on this model's own data backs it "
                          f"(no channel measured all round): D24 is off for this model")
            return out
        if sc.get("passed") and J.check_frame_record(rec) is None:
            out["frame_adoption"] = {"id": a.get("id"), "k": rec.get("k"), "eaves": sc.get("eaves"),
                                     "map_eave": sc.get("map_eave")}
            out.update(allowed=True, by="D15 frame adoption with the eave check",
                       why=f"frame {a.get('id')} adopted under D15 (k {rec.get('k')}) with the main body's eave "
                           f"{sc.get('eaves')} m confirming the map's {sc.get('map_eave')} m, backed by "
                           f"{', '.join(backing)} measured here (at a fit bound) within 3 % of k = 1")
            return out
    out["why"] = (f"{sst.get('why', 'the common scale is not established')}; and no frame adopted under D15 with the "
                  f"eave check: D24 is off for this model")
    return out


def d24_scale_numbers(cond: dict | None) -> dict:
    """The numbers D24's scale condition was decided on (d24_scale_condition), compact, for the GLB's scene extras
    (relief.d24.scale): each whole-building similarity measured on the model's data (measured, at a fit bound, k,
    ticket: the frame in question), the snap's scale, and a D15 frame adoption with its eave check."""
    c = cond or {}
    sst = c.get("scale_status") or {}
    fa = c.get("frame_adoption")
    return {"channels": {str(n): {k: ch.get(k) for k in ("measured", "at_bound", "k", "ticket")}
                         for n, ch in (c.get("channels") or {}).items() if isinstance(ch, dict)},
            "snap_scale": sst.get("snap_scale"),
            "frame_adoption": None if not isinstance(fa, dict) else {"k": fa.get("k"), "eaves": fa.get("eaves"),
                                                                     "map_eave": fa.get("map_eave")}}


def d24_scale_inconsistency(nums: dict | None) -> str | None:
    """Do the scale numbers a GLB carries (d24_scale_numbers) allow D24 by d24_scale_condition's own rules? None when
    they do; else why not. audit_glb's check: the GLB alone cannot re-derive the scale (that needs the points and the
    3-D lines: audit_model does it), only that the numbers it carries are the ones that allow D24 -- a scene that says
    "allowed" over numbers that do not allow it was edited."""
    from . import judge as J
    if not isinstance(nums, dict) or not isinstance(nums.get("channels"), dict):
        return "no scale numbers in the scene extras"
    chs = [c for c in nums["channels"].values() if isinstance(c, dict)]
    if any(c.get("measured") and not c.get("at_bound") and c.get("ticket") for c in chs):
        return "a whole-building similarity it carries puts the frame in question"

    def within(c):
        try:
            return c.get("measured") and c.get("k") is not None and abs(float(c["k"]) - 1.0) <= J.FRAME_K_TOL
        except (TypeError, ValueError):
            return False
    try:
        snap_ok = abs(float(nums.get("snap_scale")) - 1.0) <= J.SNAP_SCALE_TOL
    except (TypeError, ValueError):
        snap_ok = False
    if snap_ok and any(within(c) and not c.get("at_bound") for c in chs):
        return None
    fa = nums.get("frame_adoption")
    if isinstance(fa, dict) and fa.get("eaves") and any(within(c) for c in chs):
        return None
    return ("its scale numbers do not establish the scale (no similarity measured off its bounds within 3 % of k = 1 "
            "with the snap unscaled, and no D15 frame adoption with the eave check backed by one)")


def d24_allowed(res: "ReliefResult") -> bool:
    """May D24 elements be built on this model? Only with the switch on (D24_ENABLED) and the model's scale condition
    met (d24_scale_condition(), recorded in res.stats['d24'] by crowd3d.stylize.measure_facade_relief)."""
    return bool(D24_ENABLED and ((res.stats or {}).get("d24") or {}).get("allowed"))


def d24_record(e: dict, cond: dict, g: ReliefGuards = DEFAULT_GUARDS) -> dict:
    """The "d24" block of a D24 element's record: the decision and basis; its G-R3 statistics (edges, edges at a line
    cluster, each edge's chance rate, the chance probability and the null it failed); both photo splits' measurements
    of it (each half's box and depth, IoU, depth and edge differences, photos per half); the ring beside it and the
    interior check from its raw-point re-measurement (G-R6); and the model's scale condition."""
    def split_rec(sp, photos):
        sp = sp or {}
        return {"a": sp.get("a"), "b": sp.get("b"), "iou_a": sp.get("iou_a"), "iou_b": sp.get("iou_b"),
                "depth_diff_m": sp.get("depth_diff"), "edge_diff_max_m": sp.get("edge_diff_max"),
                "photos": [int(v) for v in photos or []]}
    ck = e.get("recheck") or {}
    iv = ck.get("interior") or {}
    sr = signed_relief(e)
    _, edges_all = d24_halves_check(e, {"interleaved": e.get("split"), "viewpoint": e.get("split_sides")}, g)
    return {"decision": D24_DECISION, "basis": D24_BASIS,
            "lines3d": {"edges": int(e.get("edges_real", 0)), "edges_confirmed": int(e.get("edges_confirmed", 0)),
                        "chance": [round(float(x), 3) for x in e.get("edge_chance") or []],
                        "p_chance": round(float(e.get("line_p_chance", 1.0)), 4), "null_p": g.line_null_p},
            "halves": {"interleaved": split_rec(e.get("split"), e.get("photos_per_half")),
                       "viewpoint": split_rec(e.get("split_sides"), e.get("photos_per_side"))},
            # every edge in both splits (d24_halves_check), and the verticals the halves were cut at (halves_cutters;
            # None: every vertical the halves detected) -- the audit derives them again from the points and re-runs the
            # splits with its own; a record naming other cutters fails
            "halves_all_edges": edges_all, "halves_cut_at": e.get("halves_cut_at"),
            "ring": {"ring_m": ck.get("ring_m"), "measured_m": ck.get("measured_m"), "claimed_m": ck.get("claimed_m"),
                     "base_m": ck.get("base_m"),
                     "step_min_m": None if sr is None else round(max(g.ring_step_share * abs(sr), g.recheck_min_m), 3),
                     "interior_ok": iv.get("ok") if iv else None},
            "scale_condition": {"allowed": bool(cond.get("allowed")), "by": cond.get("by"), "why": cond.get("why")}}


# ---------------------------------------------------------------- held-out photos (D24: the image edges at its sides)
#
# verify-r2: in photos that gave an element no 3-D line, the outlines of built D24 elements were not clearly above
# shifted-copy chance, and the sides of two of them (a ground-floor bay, a window in an arch) below it. D24 builds an
# element only where its held-out photos do not contradict its sides: the image gradient ACROSS its projected sides
# beats the same sides moved along the wall. Sills and heads are scored and recorded, not required: on a striped /
# moulded facade a horizontal edge moved up or down lands on the next course as often as not (verify-r2: at chance for
# the line-confirmed elements too), so they stay "not confirmed by an image channel" in the record.
HELDOUT_FACING_DEG = 60.0      # held-out photos: seeing the wall (>= photo_wall_frac of it), facing it within this,
HELDOUT_MIN_PHOTOS = 3         # ... with its whole outline in frame; fewer than this: unchecked (recorded as such)
HELDOUT_STEP_M = 0.04          # outline samples along a part
HELDOUT_SHIFT_U = (-0.7, -0.4, 0.4, 0.7)     # the shifted copies of the sides (along the wall)
HELDOUT_SHIFT_Z = (-0.6, -0.3, 0.3, 0.6)     # ... of a sill / head (up and down)
HELDOUT_GRAD_Q = 85.0          # a sample hits where the gradient across the outline (max within +-HELDOUT_BAND_PX)
HELDOUT_BAND_PX = 2.0          # ... exceeds this percentile of the photo's gradient magnitude


def outline_parts(r: dict, du: float = 0.0, dz: float = 0.0) -> dict[str, np.ndarray]:
    """(u, z) samples every HELDOUT_STEP_M of an element's outline as built, by part: "sides" (an opening's or a
    vertical's left and right, an arch's up to its springing line), "sill" (its bottom) and "head" (its top; an arch:
    the half circle above its springing line, clipped at its top); a band: its bottom and top ("sill", "head") 0.3 m in
    from its ends. Moved du along the wall, dz up."""
    st_ = HELDOUT_STEP_M
    u0, u1, z0, z1 = float(r["u0"]) + du, float(r["u1"]) + du, float(r["z_bottom"]) + dz, float(r["z_top"]) + dz
    if r.get("type") == "band":
        us = np.arange(u0 + 0.3, u1 - 0.3, st_)
        return {"sill": np.c_[us, np.full(len(us), z0)], "head": np.c_[us, np.full(len(us), z1)]}
    a = outline_arch(r)
    top = z1 if a is None else a[0] + dz
    zs = np.arange(z0 + 0.15, top - 0.05, st_)
    us = np.arange(u0 + 0.1, u1 - 0.1, st_)
    out = {"sides": np.r_[np.c_[np.full(len(zs), u0), zs], np.c_[np.full(len(zs), u1), zs]],
           "sill": np.c_[us, np.full(len(us), z0)]}
    if a is None:
        out["head"] = np.c_[us, np.full(len(us), z1)]
    else:
        rad, uc = a[2], a[1] + du
        ang = np.linspace(0.08, math.pi - 0.08, max(10, int(math.pi * rad / st_)))
        out["head"] = np.c_[uc + rad * np.cos(ang), np.minimum(top + rad * np.sin(ang), z1)]
    return out


class PhotoEdges:
    """The photos' image gradients, loaded lazily per camera (PosedCamera.load: the pixels the SfM saw, scaled to the
    camera's size): a sigma-1 blur, Sobel x / y, and the photo's HELDOUT_GRAD_Q-th percentile of the magnitude."""

    def __init__(self, cameras: list):
        self.cameras = cameras
        self.cache: dict[int, Any] = {}

    def grad(self, ci: int):
        if ci not in self.cache:
            c = self.cameras[ci]
            try:
                img = np.asarray(c.load())
            except Exception:                        # no pixels for this photo: it cannot be a held-out photo
                img = None
            if img is None or img.ndim < 2:
                self.cache[ci] = None
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
                if gray.shape[1] != c.width or gray.shape[0] != c.height:
                    gray = cv2.resize(gray, (int(c.width), int(c.height)))
                f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
                gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
                self.cache[ci] = (gx, gy, float(np.percentile(np.hypot(gx, gy), HELDOUT_GRAD_Q)))
        return self.cache[ci]

    def hit_rate(self, ci: int, X: np.ndarray) -> float | None:
        """Share of the 3-D outline samples X (ordered along the outline) whose projection in photo ci has an image
        gradient across the outline over the photo's threshold; None when >= 5 % of them fall outside the photo."""
        g = self.grad(ci)
        if g is None or len(X) < 4:
            return None
        gx, gy, thr = g
        c = self.cameras[ci]
        xy, dep = c.project(X)
        ok = (dep > 0.5) & (xy[:, 0] >= 3) & (xy[:, 0] < c.width - 3) & (xy[:, 1] >= 3) & (xy[:, 1] < c.height - 3)
        if ok.mean() < 0.95:
            return None
        t = np.gradient(xy, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nrm = np.c_[-t[:, 1], t[:, 0]]
        best = np.zeros(len(xy))
        for sft in np.arange(-HELDOUT_BAND_PX, HELDOUT_BAND_PX + 0.01, 0.5):
            ij = np.clip(np.round(xy + sft * nrm).astype(int), [0, 0], [c.width - 1, c.height - 1])
            best = np.maximum(best, np.abs(gx[ij[:, 1], ij[:, 0]] * nrm[:, 0] + gy[ij[:, 1], ij[:, 0]] * nrm[:, 1]))
        return float((best[ok] > thr).mean())


def _on_frame(wall, frame: dict, uz: np.ndarray) -> np.ndarray:
    off = float(frame["c0"]) + float(frame["c1"]) * uz[:, 0]
    a, t, n = np.r_[wall.a, 0.0], np.r_[wall.tangent, 0.0], np.r_[wall.normal, 0.0]
    return a[None] + uz[:, :1] * t[None] + off[:, None] * n[None] + uz[:, 1:2] * np.array([0.0, 0.0, 1.0])[None]


def heldout_photos(wall, cameras: list, seen: list[int], exclude) -> list[int]:
    """The photos seeing the wall (`seen`) that face it within HELDOUT_FACING_DEG (from the wall's middle) and are
    not in `exclude` (the photos of the 3-D line clusters at the element's own edges: camera indices, or photo names /
    ids)."""
    mid = np.r_[0.5 * (np.asarray(wall.a, float) + np.asarray(wall.b, float)), 0.5 * (float(wall.z0) + float(wall.z1))]
    cmin = math.cos(math.radians(HELDOUT_FACING_DEG))
    ex = set()
    for x in exclude or []:                   # camera indices (Lines3D.photos), or photo names / ids
        if isinstance(x, (int, np.integer)):
            ex.add(int(x))
        else:
            ex |= {k for k, c in enumerate(cameras) if str(x) in (c.name, getattr(c, "photo_id", None))}
    out = []
    for ci in seen:
        if ci in ex or not (0 <= ci < len(cameras)):
            continue
        v = np.asarray(cameras[ci].center, float) - mid
        if float(v[:2] @ np.asarray(wall.normal, float)) / max(float(np.linalg.norm(v)), 1e-9) >= cmin:
            out.append(int(ci))
    return out


def heldout_edges(r: dict, wall, frame: dict, edges: PhotoEdges, photos: list[int]) -> dict:
    """The element's outline parts (outline_parts) on the wall's frame surface `frame`, scored in each held-out photo
    (PhotoEdges.hit_rate) against the same part's shifted copies (sides along the wall, a sill / head up and down;
    copies off the wall's ends skipped): per part {"photos", "own", "copies" (means over the photos), "ratio", "beats"
    (share of the photos where own > copies)}; "photos" (held-out photos scored), "above_chance" (the parts with
    >= HELDOUT_MIN_PHOTOS photos whose own mean beats the copies' and higher in >= half the photos: the others are not
    confirmed by an image channel), "sides_ok" (heldout_verdict)."""
    own = outline_parts(r)
    L = float(wall.length)
    rows: dict[str, list] = {k: [] for k in own}
    used = set()
    for ci in photos:
        for part, uz in own.items():
            ro = edges.hit_rate(ci, _on_frame(wall, frame, uz))
            if ro is None:
                continue
            cps = []
            for sh in (HELDOUT_SHIFT_U if part == "sides" else HELDOUT_SHIFT_Z):
                if part == "sides" and (float(r["u0"]) + sh < 0.0 or float(r["u1"]) + sh > L):
                    continue
                q = outline_parts(r, du=sh) if part == "sides" else outline_parts(r, dz=sh)
                rc = edges.hit_rate(ci, _on_frame(wall, frame, q[part]))
                if rc is not None:
                    cps.append(rc)
            if cps:
                rows[part].append((ro, float(np.mean(cps))))
                used.add(ci)
    out: dict[str, Any] = {"photos": len(used)}
    for part, v in rows.items():
        if not v:
            out[part] = {"photos": 0}
            continue
        o = np.array(v, float)
        om, cm = float(o[:, 0].mean()), float(o[:, 1].mean())
        out[part] = {"photos": len(v), "own": round(om, 3), "copies": round(cm, 3),
                     "ratio": round(om / cm, 3) if cm > 0 else None, "beats": round(float(np.mean(o[:, 0] > o[:, 1])), 3)}
    # the parts the held-out photos put above chance (own above its copies, and higher in at least half the photos);
    # the others -- typically a sill or head on a striped facade -- are not confirmed by an image channel
    out["above_chance"] = sorted(k for k in own if isinstance(out.get(k), dict) and (out[k].get("photos") or 0)
                                 >= HELDOUT_MIN_PHOTOS and (out[k].get("own") or 0) > (out[k].get("copies") or 0)
                                 and (out[k].get("beats") or 0) >= 0.5)
    out["sides_ok"], out["why"] = heldout_verdict(out)
    return out


def heldout_verdict(h: dict | None) -> tuple[bool | None, str]:
    """D24's image-channel verdict on an element's held-out scores (heldout_edges): False when >= HELDOUT_MIN_PHOTOS
    held-out photos score its sides and they do not beat their shifted copies (the mean hit rate above the copies'
    and higher in at least half the photos); None (unchecked) with fewer photos, or for a band (no sides); True
    otherwise. Sills and heads are recorded only (see HELDOUT_*)."""
    if not isinstance(h, dict):
        return None, "no held-out photos scored"
    sd = h.get("sides")
    if not isinstance(sd, dict):
        return None, "no sides (a band): its sill and head are recorded only"
    n = int(sd.get("photos") or 0)
    if n < HELDOUT_MIN_PHOTOS:
        return None, f"{n} held-out photo(s) score its sides (< {HELDOUT_MIN_PHOTOS}): unchecked"
    try:
        ok = float(sd["own"]) > float(sd["copies"]) and float(sd["beats"]) >= 0.5
    except (KeyError, TypeError, ValueError):
        return False, "malformed held-out scores"
    return bool(ok), (f"its sides in {n} held-out photos: hit rate {sd.get('own')} vs {sd.get('copies')} for copies moved "
                      f"along the wall, higher in {sd.get('beats')} of the photos")


def record_box(e: dict) -> dict:
    """The part of an element that its record carries and its outline is built from (element_record's rounding)."""
    out = {"type": e["type"], **{k: round(float(e[k]), 3) for k in ("u0", "u1", "z_bottom", "z_top")}}
    if e["type"] == "opening":
        out["nesting"] = int(e.get("level") or e.get("nesting") or 1)
        if e.get("arched") and e.get("spring_z") is not None:
            out["arched"], out["spring_z"] = True, round(float(e["spring_z"]), 3)
    return out


def heldout_check(r: dict, wall, ind: dict, lines, cameras: list, seen: list[int], edges: PhotoEdges) -> dict:
    """D24's image channel for one element, as the pipeline and the publication audit both run it from the inputs:
    its held-out photos are the photos seeing the wall that face it (heldout_photos) and gave none of the 3-D line
    clusters at its own edges (line_evidence on these lines, selected near the points' reference plane -- nothing from
    the record's own line photos); scored (heldout_edges) on the wall's frame re-derived from the points (`ind`:
    independent_frame). r: record-shaped (record_box)."""
    if ind.get("refused") or not ind.get("frame"):
        return {"photos": 0, "sides_ok": None, "why": "the wall's frame cannot be re-derived from the points"}
    ev = line_evidence(r, wall, lines, ind["frame"], ref=ind.get("ref")) if lines is not None and len(lines) else {}
    excl = ev.get("photo_ids") or []
    out = heldout_edges(r, wall, ind["frame"], edges, heldout_photos(wall, cameras, seen, excl))
    out["line_photos_excluded"] = len(excl)
    return out


def apply_d24(res: "ReliefResult", g: ReliefGuards = DEFAULT_GUARDS, heldout=None) -> dict:
    """D24 on a measured model (crowd3d.stylize.measure_facade_relief calls it once res.stats['d24'] holds the model's
    scale condition): where d24_allowed(res), every d24_candidate is accepted ('accepted_by': 'D24', its d24_record()
    in 'd24'), in element order, so that a nested opening is judged with its parent's acceptance as it now stands
    (accepted by its own guards, or by D24 when it is a candidate itself; never inside a parent that is not
    accepted). heldout: callable(element, wall index) -> heldout_edges() (crowd3d.stylize.measure_facade_relief passes
    it): a candidate whose held-out photos contradict its sides (heldout_verdict False) is not built; the scores go
    into its D24 record either way ("heldout"; without the callable: unchecked). Elsewhere nothing changes. Returns
    what it did, also recorded in res.stats['d24']: {"applied" (D24 elements accepted), "built" (of them building
    geometry), "children" (nested openings accepted by their own guards once their parent was), "bounded",
    "heldout_refused"}."""
    out = {"applied": 0, "built": 0, "children": 0, "bounded": 0, "heldout_refused": 0}
    d24 = (res.stats or {}).get("d24")
    if not d24_allowed(res):
        if isinstance(d24, dict):
            d24.update(out)
        return out
    for wi, els in res.elements.items():
        changed = False
        for e in els:
            if e["type"] == "opening" and (e.get("level") or 1) == 2 and e.get("parent_el") is not None:
                pa = int(e["parent_el"])
                e["guards"]["parent_accepted"] = bool(0 <= pa < len(els) and els[pa]["accepted"])
            if e["accepted"]:
                continue
            frame_strip = e["type"] == "vertical" and e.get("how") == "frame"
            bounds = e.get("bounds") or {}
            bounded_ok = all(0 <= int(b) < len(els) and els[int(b)]["accepted"] for b in bounds.values())
            if all(e["guards"].values()):
                out["children"] += 1
            elif d24_candidate(e, g) and e["guards"].get("parent_accepted", True) and not bounded_ok:
                e["d24_refused"] = ("its " + " and ".join(sorted(bounds)) + " is its register's boundary, made by a band "
                                    "that is not accepted (or by no band): an edge nothing measured")
                out["bounded"] = out.get("bounded", 0) + 1
                continue
            elif d24_candidate(e, g) and e["guards"].get("parent_accepted", True):
                ho = heldout(e, wi) if heldout is not None else None
                ok_h, why_h = heldout_verdict(ho)
                if ok_h is False:
                    e["heldout"] = ho
                    e["d24_refused"] = "held-out photos contradict its sides: " + why_h
                    out["heldout_refused"] += 1
                    continue
                e["accepted_by"] = "D24"
                e["d24"] = d24_record(e, d24, g)
                e["d24"]["bounded_by"] = [els[int(b)]["id"] for b in sorted(bounds.values())]
                e["d24"]["heldout"] = dict(ho or {"photos": 0}, sides_ok=ok_h, why=why_h)
                out["applied"] += 1
                out["built"] += int(not frame_strip)
            else:
                continue
            e["accepted"] = True
            e["builds_geometry"] = bool(not frame_strip)
            changed = True
        if changed and wi in res.walls:
            res.walls[wi]["accepted"] = sum(bool(e["accepted"]) for e in els)
            res.walls[wi]["accepted_d24"] = sum(e.get("accepted_by") == "D24" for e in els)
    d24.update(out)
    return out


def json_safe(x) -> str:
    """A short text form of a small dict / list for a violation message."""
    try:
        import json
        return json.dumps(x, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(x)


def d24_violations(r: dict, g: ReliefGuards = DEFAULT_GUARDS) -> list[str]:
    """What is wrong with a D24 record's own evidence (record_violations adds it for provenance PROV_D24): D24 on; its
    "d24" block (d24_record) present, with the model's scale condition met; its line evidence AT CHANCE -- >=
    min_line_edges edges at a line cluster (the same number as its lines3d channel), each edge's chance rate and the
    chance probability kept; both photo splits' measurements of it, each half's box overlapping the record's (IoU >=
    split_iou); for an opening or a vertical the ring beside it showing its step (toward the claimed side by >=
    ring_step_share of its relief) on every side with points; every guard but G-R3 passed. The independent checks
    of the same claims are audit_model's (the points, the 3-D lines, the scale condition re-derived)."""
    v = []
    if not D24_ENABLED:
        v.append("a D24 record while D24 is switched off")
    d = r.get("d24")
    if not isinstance(d, dict):
        return v + ["a D24 record without its D24 evidence"]
    sc = d.get("scale_condition") if isinstance(d.get("scale_condition"), dict) else {}
    if sc.get("allowed") is not True:
        v.append(f"a D24 record on a model whose scale condition is not met ({sc.get('why')})")
    l3 = d.get("lines3d") if isinstance(d.get("lines3d"), dict) else {}
    ch3 = ((r.get("channels") or {}).get("lines3d") or {}) if isinstance(r.get("channels"), dict) else {}
    try:
        nconf = int(l3["edges_confirmed"])
        ok_l = (nconf >= g.min_line_edges and nconf == int(ch3.get("edges_confirmed", -1))
                and l3.get("p_chance") is not None and float(l3["p_chance"]) >= 0.0
                and isinstance(l3.get("chance"), list) and len(l3["chance"]) == int(l3["edges"]))
    except (KeyError, TypeError, ValueError):
        ok_l = False
    if not ok_l:
        v.append("a D24 record needs its line evidence at chance: >= 1 edge at a 3-D line cluster, the chance "
                 "statistics kept")
    else:
        # one set of line numbers: the record's lines3d channel says what its D24 block says, and that is AT chance
        # (an edit of either copy to read as line-confirmed is caught here; audit_model recomputes them from the lines)
        try:
            same = (int(ch3.get("edges", -1)) == int(l3["edges"])
                    and [round(float(x), 3) for x in ch3.get("chance") or []] == [round(float(x), 3) for x in l3["chance"]]
                    and ch3.get("p_chance") is not None and abs(float(ch3["p_chance"]) - float(l3["p_chance"])) <= 1e-4)
            at_chance = float(l3["p_chance"]) > float(l3.get("null_p", g.line_null_p)) \
                and float(l3.get("null_p", g.line_null_p)) >= g.line_null_p - 1e-9
            agree = abs(tail_prob([float(x) for x in l3["chance"]], nconf) - float(l3["p_chance"])) <= 1e-3
        except (TypeError, ValueError):
            same = at_chance = agree = False
        if not same:
            v.append("the D24 block's line numbers differ from the record's lines3d channel")
        if not at_chance:
            v.append(f"a D24 record whose line evidence is not at chance (p {l3.get('p_chance')} <= null "
                     f"{max(float(l3.get('null_p') or 0), g.line_null_p)}): it is a line-confirmed element, not D24")
        if not agree:
            v.append("the D24 block's chance probability does not follow from its own edge chance rates")
    halves = d.get("halves") if isinstance(d.get("halves"), dict) else {}
    for name in ("interleaved", "viewpoint"):
        h = halves.get(name) if isinstance(halves.get(name), dict) else {}
        boxes = [h.get("a"), h.get("b")]
        if not all(isinstance(b, dict) and all(k in b for k in ("u0", "u1", "z_bottom", "z_top", "depth")) for b in boxes):
            v.append(f"a D24 record without both {name} photo halves' measurements")
            continue
        try:
            iou = min(box_iou(_box(r), _box(b)) for b in boxes)
        except (KeyError, TypeError, ValueError):
            iou = 0.0
        if not iou >= g.split_iou - 1e-9:
            v.append(f"the {name} photo halves' measurements do not overlap the element (IoU {iou:.2f} < {g.split_iou})")
    ok_e, info_e = d24_halves_check(r, halves, g)
    if not ok_e:
        v.append("an edge of the D24 element is not found again in both photo halves of both splits (every edge within "
                 f"{g.d24_edge_m} m between the halves and {g.d24_box_m} m of the element): " + json_safe(info_e))
    ho = d.get("heldout")
    if not isinstance(ho, dict) or "sides_ok" not in ho:
        v.append("a D24 record without its held-out photo scores")
    elif ho.get("sides_ok") is False or heldout_verdict(ho)[0] is False:
        v.append(f"a D24 record whose held-out photos contradict its sides ({ho.get('why')})")
    if r.get("type") in ("opening", "vertical"):
        ring = d.get("ring") if isinstance(d.get("ring"), dict) else {}
        rm = ring.get("ring_m")
        sr = signed_relief(r)
        try:
            got = [float(x) for x in rm if x is not None] if isinstance(rm, list) else []
        except (TypeError, ValueError):
            got = []
        if not got or sr is None:
            v.append("a D24 record without the ring beside it")
        else:
            need = max(g.ring_step_share * abs(sr), g.recheck_min_m)
            sgn = 1.0 if sr >= 0 else -1.0
            if not all(x * sgn >= need - 1e-9 for x in got):
                v.append(f"the ring beside it does not show its step ({got} m, needs {sgn * need:+.3f} m)")
    gd = r.get("guards")
    if not isinstance(gd, dict) or not all(bool(x) for k, x in gd.items() if k != "G-R3_line_edge"):
        v.append("a point-based guard did not pass")
    return v


def line_evidence(r: dict, wall, lines, frame: dict | None, prm: ReliefParams = DEFAULT_RELIEF,
                  ref: dict | None = None) -> dict:
    """G-R3 recomputed for one element record from the 3-D lines (the publication audit; nothing from the record's
    own line numbers): its edges (a band's bottom / top, a vertical's left / right, an opening's four; an edge at
    the wall's end or ground, or an arch's top, does not count), which of them lie within edge_tol_m of a line
    cluster (the clusters the measurement uses: a band's over the whole wall, the others' along the edge) and each
    edge's chance rate (the background within 2 m of the element, away from its own edges) -> {"edges", "confirmed",
    "chance", "p_chance", "ok", "photo_ids" (the photos of the clusters at its confirmed edges)}: ok when >=
    min_line_edges are confirmed and p_chance <= line_null_p. The lines are
    selected near `ref` -- the points' reference plane, as the measurement selects them (independent_frame's "ref";
    verify-r1 adversarial crowd: with the frame level 0.24 m off the reference the two planes selected different
    clusters and the audit failed an honest record) -- else near `frame`."""
    g = DEFAULT_GUARDS
    out = {"edges": 0, "confirmed": 0, "chance": [], "p_chance": 1.0, "ok": False}
    fr = ref or frame or r.get("frame") or {}
    try:
        u0, u1, zb, zt = (float(r[k]) for k in ("u0", "u1", "z_bottom", "z_top"))
        WL = wall_lines(wall, lines, {"c0": float(fr["c0"]), "c1": float(fr["c1"])}, prm)
    except (KeyError, TypeError, ValueError):
        return out
    if WL is None:
        return out
    L, tol = float(wall.length), prm.edge_tol_m
    zhi = float(wall.z1) + prm.top_extra_m
    t = r.get("type")
    if t == "band":
        strong = line_clusters(WL, "h", 0.0, L, wall.z0 - 0.2, zhi, prm.strong_cover, prm.strong_photos, prm)
        ch = chance_rate(strong, wall.z0, zhi, tol, exclude=(zb, zt))
        edges = [(zb, zb <= wall.z0 + 0.15, strong, ch), (zt, zt >= zhi - 0.05, strong, ch)]
    elif t == "vertical":
        vloc = line_clusters(WL, "v", zb, zt, u0 - 0.5, u1 + 0.5, prm.local_cover, prm.line_min_photos, prm)
        ch = _background_chance(WL, "v", zb, zt, u0, u1, 0.0, L, prm)
        edges = [(u0, u0 <= 0.25, vloc, ch), (u1, u1 >= L - 0.25, vloc, ch)]
    elif t == "opening":
        arched = bool(r.get("arched")) and r.get("spring_z") is not None
        ztop_ = float(r["spring_z"]) if arched else zt
        vloc = line_clusters(WL, "v", zb, ztop_, u0 - 0.5, u1 + 0.5, prm.local_cover, prm.line_min_photos, prm)
        hloc = line_clusters(WL, "h", u0, u1, zb - 0.5, zt + 0.5, prm.local_cover, prm.line_min_photos, prm)
        cj = _background_chance(WL, "v", zb, ztop_, u0, u1, 0.0, L, prm)
        cs = _background_chance(WL, "h", u0, u1, zb, zt, wall.z0, zhi, prm)
        edges = [(u0, False, vloc, cj), (u1, False, vloc, cj), (zb, zb <= wall.z0 + 0.3, hloc, cs),
                 (zt, arched, hloc, cs)]
    else:
        return out
    real = [(pos, cl, ch) for pos, limit, cl, ch in edges if not limit]
    near_ = [nearest_cluster(cl, pos, tol) for pos, cl, _ in real]
    conf = sum(1 for c in near_ if c is not None)
    ps = [ch for _, _, ch in real]
    p = tail_prob(ps, conf) if real else 1.0
    out.update(edges=len(real), confirmed=int(conf), chance=[round(x, 3) for x in ps], p_chance=round(p, 4),
               ok=bool(conf >= g.min_line_edges and p <= g.line_null_p),
               photo_ids=sorted({x for c in near_ if c is not None for x in c["photos"]}, key=str))
    return out


def _count_check(wall, points, track, prm: ReliefParams, others) -> tuple[bool, int, float]:
    """The cheap part of G-R1 (enough points near the wall), to decide whether 3-D lines are worth computing:
    (passes, points, points per m2)."""
    U, _ = wall_points(wall, points, track, prm, others)
    area = max(wall.length * (wall.z1 - wall.z0), 1e-9)
    return bool(len(U) >= prm.min_points and len(U) >= prm.min_density * area), int(len(U)), round(len(U) / area, 1)


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
        enough, rep["points"], rep["points_per_m2"] = _count_check(wall, P, trk, prm, walls)
        if getattr(wall, "profile", None) is not None:
            rep["refused"] = "a gable / skillion wall (top profile): relief is measured on rectangular walls only"
        elif rep["part"] is not None and rep["part"] in stubs:
            rep["refused"] = "a 'height pending' stub part"
        elif min(rep["photos_per_half"]) < g.min_photos_per_half:
            rep["refused"] = (f"G-R1: photos seeing the wall per photo half {rep['photos_per_half']} "
                              f"(< {g.min_photos_per_half})")
        elif tracks is None or not len(tracks):
            rep["refused"] = "no point tracks: the split-half check (G-R2) cannot run"
        elif not enough:
            rep["refused"] = (f"G-R1: too few SfM points for relief ({rep['points']}, {rep['points_per_m2']} per m2; "
                              f"needs {prm.min_points} and {prm.min_density:g} per m2)")
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
        res.stats["split_points"], res.stats["split_of"], res.stats["split_s"] = [0, 0], 0, 0.0
        for wi, WR in full.items():
            # the interleaved split: THIS wall's points re-triangulated from each half's photos (per wall, so that the
            # publication audit re-runs exactly the same split from the wall's own points; verify-r1: points of the
            # other walls re-triangulated into this wall's window changed a half's measurement)
            ts = time.perf_counter()
            tri = [retriangulate(WR.maps["idx"], tracks, cameras, half == h, g) for h in (0, 1)]
            res.stats["split_points"] = [a + int(np.isfinite(X[:, 0]).sum()) for a, (X, _) in zip(res.stats["split_points"], tri)]
            res.stats["split_of"] += int(len(WR.maps["idx"]))
            res.stats["split_s"] = round(res.stats["split_s"] + time.perf_counter() - ts, 2)
            wall = walls[wi]
            rep = res.walls[wi]
            fac = facades.get(wi)
            U_w = wall_points(wall, P, trk, prm, walls)[0]

            def recheck(r, wall=wall, U_w=U_w):
                return audit_points(r, wall, P, trk, others=walls, prm=prm, U=U_w, g=g, strict=True)

            # things standing in front of the wall at any distance (>= clear_margin_m), for G-R5's clear view
            Ua = to_wall(wall.a, wall.tangent, wall.normal, P)
            fd = Ua[:, 1] - (float(WR.ref["c0"] + WR.frame) + float(WR.ref["c1"]) * Ua[:, 0])
            fsel = (fd > g.clear_margin_m) & (fd < 150.0) & (Ua[:, 0] > -60) & (Ua[:, 0] < wall.length + 60) \
                & (trk >= prm.min_track)
            front, front_d = P[fsel], fd[fsel]
            if len(front):          # surfaces, not the sparse SfM noise around a facade: >= 5 points per 0.2 m voxel
                _, inv, cnt = np.unique(np.floor(front / 0.2).astype(np.int64), axis=0, return_inverse=True,
                                        return_counts=True)
                dense = cnt[inv.ravel()] >= g.clear_voxel_points
                front, front_d = front[dense], front_d[dense]
            if len(front) > 80000:
                k_ = np.random.default_rng(wi).choice(len(front), 80000, replace=False)
                front, front_d = front[k_], front_d[k_]
            ob = observations[wi] if observations is not None and wi < len(observations) else None
            seen_cams = [cameras[ci] for ci, frac, _ in (ob.cameras if ob is not None else [])
                         if frac >= g.photo_wall_frac and 0 <= ci < len(cameras)]

            def clear(r, wall=wall, front=front, front_d=front_d, seen_cams=seen_cams):
                # only what stands clear_margin_m in front of the element's own surface (a cornice's own points
                # do not hide it)
                return clear_view_share(r, wall, front[front_d > max(float(r["offset_m"]), 0.0) + g.clear_margin_m],
                                        seen_cams, g)

            # the viewpoint-disjoint split: the wall's points re-triangulated from the photos left / right of it
            seen_idx = [ci for ci, frac, _ in (ob.cameras if ob is not None else []) if frac >= g.photo_wall_frac]
            side, per_side = side_halves(cameras, wall, seen_idx)
            rep["photos_per_side"] = per_side
            wids = WR.maps["idx"]
            tri_s = [retriangulate(wids, tracks, cameras, side == h, g) for h in (0, 1)]

            def halves_of(cut_boxes, wall=wall, tri=tri, tri_s=tri_s, wids=wids):
                hs, ss = [], []
                for X, c in tri:
                    ok = np.isfinite(X[:, 0])
                    hs.append(measure_wall(wall, X[ok], np.maximum(c[ok], prm.min_track), None, prm, others=walls,
                                           cut_boxes=cut_boxes))
                for X, c in tri_s:
                    ok = np.isfinite(X[:, 0])
                    ss.append(measure_wall(wall, X[ok], np.maximum(c[ok], prm.min_track), None, prm, others=walls,
                                           cut_boxes=cut_boxes) if ok.any() else None)
                return hs, ss
            halves, sides = halves_of(None)
            els = guard_elements(WR, halves, rep["photos_per_half"], None if fac is None else fac.mask, wall, g,
                                 recheck=recheck, clear=clear, sides=sides, per_side=per_side)
            # openings are cut into bays only at verticals the points confirm (both photo splits find them, the raw
            # points re-measure them, no free-standing object at them): a strip the points do not confirm (an object
            # in front, a strip one photo half does not see) must not split a recess into made-up bays, and a confirmed
            # pilaster still separates two bays when only its line evidence is at chance
            acc_v = halves_cutters(els, WR)
            if acc_v is not None:
                WR = measure_wall(wall, P, trk, lines, prm, others=walls, cut_boxes=acc_v)
                halves, sides = halves_of(acc_v)
                els = guard_elements(WR, halves, rep["photos_per_half"], None if fac is None else fac.mask, wall, g,
                                     recheck=recheck, clear=clear, sides=sides, per_side=per_side)
                rep["recut_at_accepted_verticals"] = len(acc_v)
            cut_at = acc_v
            for e in els:
                e["halves_cut_at"] = cut_at      # the verticals the photo halves were cut at (None: all they found)
            rep["split_refused"] = [h.refused for h in halves]
            rep["side_refused"] = [("no points" if h is None else h.refused) for h in sides]
            if WR.stats.get("occluders"):
                rep["occluders"] = WR.stats["occluders"]
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
PROV_D24 = "photo-measured (points, D24)"     # built on the points alone, its 3-D lines at chance (D24, apply_d24)
RELIEF_PROVENANCES = (PROV_MEASURED, PROV_D24)
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
            back = float(e["offset_m"]) if e.get("offset_m") is not None else \
                float(e.get("register_frame") or 0.0) - float(e["depth"])
            if e.get("arched") and e.get("spring_z") is not None and e["spring_z"] > c + 0.1:
                poly = _arch_polygon(a, b, c, min(float(e["spring_z"]), d), d)
            else:
                poly = _rect(a, b, c, d)
            level1.append({"poly": poly, "d": back, "parent": None, "el": i})
        elif e["type"] in ("band", "vertical"):
            if float(e["depth"]) <= 0:
                continue
            # the front at the element's own offset from the wall's photographed frame surface (a pilaster on a
            # set-back register stands on that register's frame), which is what its record says
            p = float(e["offset_m"]) if e.get("offset_m") is not None else float(e["depth"])
            if p <= 0.02:
                skipped.append((i, "not in front of the wall face"))
                continue
            if p > RELIEF_MAX_DEPTH:
                skipped.append((i, "over the depth limit"))
                continue
            a, b = max(e["u0"], u0), min(e["u1"], u1)
            c, d = max(e["z_bottom"], z0w), min(e["z_top"], z1w)
            if b - a >= 0.1 and d - c >= 0.1:
                boxes.append({"u0": a, "u1": b, "z0": c, "z1": d, "p": p, "el": i})
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
        dn = float(e["offset_m"]) if e.get("offset_m") is not None else rec[k]["d"] - float(e["depth"])
        rec.append({"poly": poly, "d": dn, "parent": k, "el": i})
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
    ss = e.get("split_sides")
    out = {"id": e.get("id"), "type": e["type"], "kind": e["kind"], "u0": round(e["u0"], 3), "u1": round(e["u1"], 3),
           "z_bottom": round(e["z_bottom"], 3), "z_top": round(e["z_top"], 3),
           e["depth_key"]: round(float(e["depth"]), 3), "offset_m": e.get("offset_m"),
           "provenance": PROV_D24 if e.get("accepted_by") == "D24" else PROV_MEASURED,
           "channels": {"sfm_points": True,
                        "split_half": {"iou_a": sp.get("iou_a"), "iou_b": sp.get("iou_b"),
                                       "depth_diff_m": sp.get("depth_diff"), "edge_diff_max_m": sp.get("edge_diff_max")},
                        "lines3d": {"edges_confirmed": e["edges_confirmed"], "edges": e["edges_real"],
                                    "photos": len(e.get("line_photos") or []),
                                    "chance": list(e.get("edge_chance") or []),
                                    "p_chance": e.get("line_p_chance")}},
           "photos_per_half": list(e.get("photos_per_half") or []), "line_photos": list(e.get("line_photos") or []),
           "guards": dict(e.get("guards") or {}), "built": bool(built)}
    if ss is not None:
        out["channels"]["split_sides"] = {"iou_a": ss.get("iou_a"), "iou_b": ss.get("iou_b"),
                                          "depth_diff_m": ss.get("depth_diff"), "edge_diff_max_m": ss.get("edge_diff_max")}
        out["photos_per_side"] = list(e.get("photos_per_side") or [])
    if e.get("frame"):
        out["frame"] = e["frame"]
    if e.get("observed_share") is not None or e.get("clear_share") is not None:
        out["view"] = {"observed_share": e.get("observed_share"), "clear_share": e.get("clear_share")}
    if e.get("recheck"):
        ck = e["recheck"]
        out["recheck"] = {k: ck.get(k) for k in ("points", "measured_m", "base_m", "se_m", "tol_m", "ring_m")}
        if ck.get("interior") is not None:
            iv = ck["interior"]
            out["recheck"]["interior"] = {k: iv.get(k) for k in ("v", "h", "v_m", "h_m", "step_m", "member_m", "limits",
                                                                 "ok")}
    if e["type"] == "opening":
        out["nesting"] = int(e.get("level") or 1)
        if e.get("parent_id") is not None:
            out["parent"] = e["parent_id"]
        if e.get("arched") and e.get("spring_z") is not None:
            out["arched"], out["spring_z"] = True, round(float(e["spring_z"]), 3)
    if e.get("sign") is not None:
        out["sign"] = int(e["sign"])
    if e.get("how"):
        out["how"] = e["how"]
    if e.get("accepted_by") == "D24" and e.get("d24"):
        out["d24"] = e["d24"]
    if isinstance(e.get("scale_condition"), dict):
        # the model's scale condition (d24_scale_condition): D24 needs it met; for a line-confirmed element it is a
        # caveat -- where it is not met, its position and depth carry the reconstruction's unverified common scale
        out["scale_condition"] = dict(e["scale_condition"])
    return out


# ---------------------------------------------------------------- QA (called by crowd3d.stylize's checks)


RELIEF_LIMITS = {"band": {"h": (0.1, 6.0)}, "vertical": {"w": (0.2, 2.5)}, "opening": {"w": (0.4, 8.0), "h": (0.6, 30.0)}}


def record_violations(r: dict, g: ReliefGuards = DEFAULT_GUARDS) -> list[str]:
    """What is wrong with one element record (empty: backed): photo-measured, both halves of both photo splits found
    it with its measured edges within split_edge_m, >= 1 line-confirmed edge above chance (p_chance <= line_null_p),
    enough photos per half, every guard passed (re-measured from the points, G-R6; a recess also re-measured for
    structure inside it, interior_structure, and none found), inside the size and depth limits. A D24 record
    (provenance PROV_D24: built on the points alone) needs no line evidence above chance and no G-R3; instead its own
    D24 evidence (d24_violations: D24 on, the scale condition met, its lines AT chance with their statistics, both
    photo splits' measurements of it, the ring beside it)."""
    v = []
    ch = r.get("channels") or {}
    sh = ch.get("split_half") or {}
    d24 = r.get("provenance") == PROV_D24
    if r.get("provenance") not in RELIEF_PROVENANCES:
        v.append(f"element provenance {r.get('provenance')!r}")
    elif d24:
        v += d24_violations(r, g)
    elif r.get("d24") is not None:
        v.append("a D24 evidence block on a record that does not say D24")
    if not (sh.get("iou_a") or 0) >= g.split_iou or not (sh.get("iou_b") or 0) >= g.split_iou:
        v.append("element not measured from both photo halves")
    sd = ch.get("split_sides") or {}
    if not (sd.get("iou_a") or 0) >= g.split_iou or not (sd.get("iou_b") or 0) >= g.split_iou:
        v.append("element not measured from both viewpoint sides")
    for nm, sp_ in (("photo halves", sh), ("viewpoint sides", sd)):
        em = sp_.get("edge_diff_max_m")
        if em is None or float(em) > g.split_edge_m + 1e-9:
            v.append(f"edges of the {nm} {em} m apart (> {g.split_edge_m})")
    l3 = ch.get("lines3d") or {}
    if not (l3.get("edges_confirmed") or 0) >= g.min_line_edges:
        v.append("no line-confirmed edge")
    if l3.get("p_chance") is None or (not d24 and float(l3["p_chance"]) > g.line_null_p + 1e-9):
        v.append(f"line confirmation at chance (p {l3.get('p_chance')})")
    pph = r.get("photos_per_half") or [0, 0]
    if len(pph) < 2 or min(pph) < g.min_photos_per_half:
        v.append(f"photos per half {pph} < {g.min_photos_per_half}")
    gd = r.get("guards")
    if not isinstance(gd, dict) or not gd or not all(bool(x) for k, x in gd.items()
                                                     if not (d24 and k == "G-R3_line_edge")):
        v.append("a guard did not pass")
    elif "G-R6_points_recheck" not in gd:
        v.append("not re-measured from the points (G-R6)")
    if is_recess(r):
        iv = (r.get("recheck") or {}).get("interior") if isinstance(r.get("recheck"), dict) else None
        if not isinstance(iv, dict) or iv.get("ok") is not True:
            v.append("a recess not re-measured for structure inside it (G-R6 interior)" if not isinstance(iv, dict)
                     else "a recess over structure the points show inside it (G-R6 interior)")
    if r.get("built") and not abs(float(r.get("offset_m") or 0.0)) > 0.02:
        v.append("built at the wall face itself")
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


def signed_relief(r: dict) -> float | None:
    """An element's relief relative to its surroundings, from its record's own depth field (outward +): an opening's
    -depth, a band's level, a vertical's projection (a frame strip: over the recesses beside it)."""
    key = {"opening": "depth", "band": "level", "vertical": "projection"}.get(r.get("type"))
    raw = r.get(key) if key else None
    if raw is None and key and r.get("depth_key") == key:          # an element as guard_elements carries it
        raw = r.get("depth")
    try:
        v = None if raw is None else float(raw)
    except (TypeError, ValueError):
        return None
    return None if v is None else (-v if key == "depth" else v)


def record_base(r: dict) -> float | None:
    """The surface an element's relief is measured from (its parent's back, its register's frame, the recesses beside
    a frame strip; the wall frame for a band), relative to the wall's photographed frame surface: offset_m minus its
    signed relief. What the element's box would show if the element were not there."""
    sr = signed_relief(r)
    try:
        return None if sr is None else float(r["offset_m"]) - sr
    except (KeyError, TypeError, ValueError):
        return None


RING_IN_M, RING_OUT_M = 0.1, 0.4     # audit_points' ring: the points 0.1-0.4 m beside an opening's / vertical's sides
INTERIOR_BIN_M, INTERIOR_ACROSS_M = 0.1, 0.25   # interior_structure: bins along a profile, cells across it
INTERIOR_POOL_M = 0.15                          # ... a horizontal row pools the points within this of its centre
INTERIOR_FLANK_M = (0.5, 1.5)                   # ... a member's flanks: the rows this far below / above it
INTERIOR_MIN_CLAIM_M = 1.0    # ... an outline this long along an axis can hold two recesses and the structure between


def is_recess(r: dict) -> bool:
    """A record / element that claims a recess (its interior must not hold structure at the frame plane): an opening,
    or a band set back (a setback, a groove)."""
    return r.get("type") == "opening" or (r.get("type") == "band" and (r.get("sign") or 0) < 0)


def outline_arch(r: dict) -> tuple[float, float, float] | None:
    """(springing line, centre u, radius) of the round arch an element's outline has AS BUILT (relief_plan: a level-1
    opening, arched, its springing line more than 0.1 m above its sill -- the half circle above it, clipped at its
    top), or None: a rectangle."""
    if r.get("type") != "opening" or (r.get("level") or r.get("nesting") or 1) != 1 or not r.get("arched") \
            or r.get("spring_z") is None:
        return None
    try:
        sp, zb, zt = float(r["spring_z"]), float(r["z_bottom"]), float(r["z_top"])
        u0, u1 = float(r["u0"]), float(r["u1"])
    except (KeyError, TypeError, ValueError):
        return None
    if not sp > zb + 0.1:
        return None
    return min(sp, zt), 0.5 * (u0 + u1), 0.5 * (u1 - u0)


def _interior_shrink(r: dict) -> tuple[float, float]:
    w, h = float(r["u1"]) - float(r["u0"]), float(r["z_top"]) - float(r["z_bottom"])
    return min(0.15, 0.25 * max(w, 0.0)), min(0.15, 0.25 * max(h, 0.0))


def interior_box(r: dict) -> tuple[float, float, float, float]:
    """The bounding box of the part of an element's outline its re-measurement uses (audit_points,
    interior_structure; in_interior): the outline AS BUILT (outline_arch: an arched opening is the rectangle up to its
    springing line and the half circle above it, clipped at its top) shrunk by up to 0.15 m on every side. verify-r2
    N10x: the box used to end at the springing line, so a record that put its springing line 0.3 m above its sill was
    re-measured on a 0.15 m band while its built arch spanned two bays and the pilaster between them."""
    su, sz = _interior_shrink(r)
    u0, u1, zb, zt = float(r["u0"]) + su, float(r["u1"]) - su, float(r["z_bottom"]) + sz, float(r["z_top"]) - sz
    a = outline_arch(r)
    if a is not None:
        zt = min(zt, a[0] + max(a[2] - su, 0.0))
    return u0, u1, zb, zt


def in_interior(r: dict, u: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Which (u, z) lie inside the element's outline as built, shrunk by up to 0.15 m on every side (interior_box: a
    rectangle, or the rectangle up to the springing line and, above it, the half circle of radius half the width
    less the shrink)."""
    u, z = np.asarray(u, float), np.asarray(z, float)
    bu0, bu1, bz0, bz1 = interior_box(r)
    m = (u >= bu0) & (u <= bu1) & (z >= bz0) & (z <= bz1)
    a = outline_arch(r)
    if a is not None:
        sp, uc, rad = a
        su = _interior_shrink(r)[0]
        m &= (z <= sp) | ((u - uc) ** 2 + (z - sp) ** 2 <= max(rad - su, 0.0) ** 2)
    return m


def _claim_extent(r: dict) -> tuple[float, float]:
    """The element's outline as built: its width and its height (an arch: up to its crown, clipped at its top)."""
    w = float(r["u1"]) - float(r["u0"])
    a = outline_arch(r)
    zt = float(r["z_top"]) if a is None else min(float(r["z_top"]), a[0] + a[2])
    return w, zt - float(r["z_bottom"])


def interior_structure(u: np.ndarray, off: np.ndarray, z: np.ndarray, r: dict, claimed: float, base: float,
                       g: ReliefGuards = DEFAULT_GUARDS, strict: bool = False) -> dict:
    """Structure INSIDE a claimed recess, from the raw points (u along the wall, off = offset from the frame surface,
    z), independently of anything the record says about its interior, over its outline AS BUILT (in_interior: the
    rectangle, or for an arched opening the rectangle up to the springing line and the half circle above it, shrunk
    by up to 0.15 m; a cell or row counts only the part of the grid inside it):
      vertical strips (piers, pilasters, mullions): 0.1 m columns, cells up them (0.25 m, up to 1 m where the points
        are sparse: ~6 points per cell); a cell (>= 3 points, its median) is STRUCTURE when it stands prouder than
        the claimed back by >= max(half the claimed relief, recheck_min_m), the relief taken to the record's base or
        to the frame plane (off 0), whichever is nearer -- so that a record cannot move its own base to make a pier
        look recessed; a column is structure when >= interior_share of its cells with data (at least half its cells)
        are; a run of such columns >= interior_w_m wide is a strip;
      horizontal strips (an entablature between two orders, a frame band, a transom): 0.1 m rows, each POOLING the
        points within INTERIOR_POOL_M of its centre (one noisy row does not break a band), cells along it (~6
        points each); a row is structure when >= interior_share of its cells with data stand at the frame plane as
        above, OR when it is a MEMBER across the recess: its pooled median rises >= interior_member x the relief
        above the recess's back -- the mean of the pooled medians INTERIOR_FLANK_M below and above it (both
        needed; never below the claimed back) and by half that over each flank on its own (a bump, not a step: the
        top of statues standing in an arch is no member) -- and >= interior_share of its cells stand >= half that
        rise above the back (across the width, not one statue); a run of such rows >= interior_h_m tall is a strip.
    A strip is real structure the claim deletes (two bays merged into one, a lower order and an upper one made into
    one tall order). A recess's interior may hold things of partial extent (statues in an arch, a short impost
    moulding, a window inside it is deeper anyway). strict: the pipeline's own limits (ReliefGuards.interior_guard).
    Strip widths and heights count nominal 0.1 m bins (verify-r2: a 0.5 m pilaster seen as three columns of 0.0997 m
    is a 0.3 m strip, not 0.299 m; a two-row strip is 0.2 m whatever the outline's height makes a row). ok None (unverifiable, never a pass): fewer than 12 points inside it, or an outline at
    least INTERIOR_MIN_CLAIM_M long along an axis (room for two recesses and the structure between them) whose
    interior cannot be measured along it, or one where under half the interior's columns (or rows) have data on half
    their cells (most of the claimed recess unseen: verify-r2 N10x, a check that could not run counted as a pass).
    Returns {"v": [[u0, u1], ...], "h": [[z0, z1], ...], "v_m", "h_m" (widest / tallest strip, m), "step_m" (the
    structure threshold above the claimed back), "member_m" (a member's rise), "limits" (share, width, height,
    member used), "cover" (share of columns / rows with data), "ok", "why"}."""
    share, w_min, h_min, member = (g.interior_guard if strict else
                                   (g.interior_share, g.interior_w_m, g.interior_h_m, g.interior_member))
    out = {"v": [], "h": [], "v_m": 0.0, "h_m": 0.0, "step_m": None, "member_m": None, "ok": True, "why": "",
           "limits": [round(float(share), 3), float(w_min), float(h_min), float(member)]}
    rel = float(claimed) - float(base)
    if rel >= 0 or not is_recess(r):
        return out                                   # not a recess: nothing to check inside
    rr = min(abs(rel), abs(float(claimed)))
    step = max(0.5 * rr, g.recheck_min_m)
    mstep = max(float(member) * rr, g.recheck_min_m)
    out["step_m"], out["member_m"] = round(step, 3), round(mstep, 3)
    u0, u1, z0, z1 = interior_box(r)
    u, off, z = np.asarray(u, float), np.asarray(off, float), np.asarray(z, float)
    m = in_interior(r, u, z)
    cw, chh = _claim_extent(r)
    blind = [f"{nm} {ext:.2f} m inside an outline {cl:.2f} m {nm[0] == 'w' and 'wide' or 'tall'}"
             for nm, ext, cl in (("width", u1 - u0, cw), ("height", z1 - z0, chh))
             if ext < 2 * INTERIOR_BIN_M and cl >= INTERIOR_MIN_CLAIM_M]
    if blind or m.sum() < 12:
        out.update(ok=None, why="its interior cannot be measured: " + ("; ".join(blind) if blind
                                                                       else f"{int(m.sum())} points inside it (< 12)"))
        return out
    if u1 - u0 < 2 * INTERIOR_BIN_M or z1 - z0 < 2 * INTERIOR_BIN_M:
        return out                                   # an outline too small to hold two recesses and structure between
    uu, oo, zz = u[m], off[m], z[m]
    dens = float(m.sum()) / max((u1 - u0) * (z1 - z0), 1e-9)
    nbins_w = int(math.ceil(float(w_min) / INTERIOR_BIN_M - 1e-6))
    nbins_h = int(math.ceil(float(h_min) / INTERIOR_BIN_M - 1e-6))
    # ---- vertical strips: 0.1 m columns, cells up them (INTERIOR_ACROSS_M, up to 1 m where the points are sparse)
    across = float(np.clip(6.0 / max(dens * INTERIOR_BIN_M, 1e-9), INTERIOR_ACROSS_M, 1.0))
    out["cell_m"] = round(across, 3)
    na = max(1, int(round((u1 - u0) / INTERIOR_BIN_M)))
    nb = max(1, int(round((z1 - z0) / across)))
    bw, bzc = (u1 - u0) / na, (z1 - z0) / nb
    ia = np.clip(((uu - u0) / (u1 - u0) * na).astype(int), 0, na - 1)
    ib = np.clip(((zz - z0) / (z1 - z0) * nb).astype(int), 0, nb - 1)
    med, cnt = _group_median(ia * nb + ib, oo, na * nb)
    med, cnt = med.reshape(na, nb), cnt.reshape(na, nb)
    gu, gz = np.meshgrid(u0 + (np.arange(na) + 0.5) * bw, z0 + (np.arange(nb) + 0.5) * bzc, indexing="ij")
    cin = in_interior(r, gu, gz)                     # the cells of the grid inside the outline
    nin = cin.sum(1)
    has = cnt >= 3
    nd = has.sum(1)
    elig = (nin > 0) & (nd >= 0.5 * np.maximum(nin, 1))
    sh = np.where(elig, (has & (med - float(claimed) >= step)).sum(1) / np.maximum(nd, 1), 0.0)
    out["v"] = [[round(u0 + s * bw, 3), round(u0 + e * bw, 3)] for s, e in _runs(list((sh >= share).astype(int)), 1)
                if e - s >= nbins_w]
    # ---- horizontal strips: 0.1 m rows, each pooling the points within INTERIOR_POOL_M of its centre
    nr = max(1, int(round((z1 - z0) / INTERIOR_BIN_M)))
    bh = (z1 - z0) / nr
    zc = z0 + (np.arange(nr) + 0.5) * bh
    o_ = np.argsort(zz, kind="stable")
    zs, us_, os_ = zz[o_], uu[o_], oo[o_]
    lo = np.searchsorted(zs, zc - INTERIOR_POOL_M, "left")
    hi = np.searchsorted(zs, zc + INTERIOR_POOL_M, "right")
    across_h = float(np.clip(6.0 / max(dens * 2 * INTERIOR_POOL_M, 1e-9), INTERIOR_ACROSS_M, 1.0))
    nc = max(1, int(round((u1 - u0) / across_h)))
    ic = np.clip(((us_ - u0) / (u1 - u0) * nc).astype(int), 0, nc - 1)
    C = np.full((nr, nc), np.nan)
    prof = np.full(nr, np.nan)
    for k in range(nr):
        a_, b_ = int(lo[k]), int(hi[k])
        if b_ - a_ < 5:
            continue
        mk, ck = _group_median(ic[a_:b_], os_[a_:b_], nc)
        C[k] = np.where(ck >= 3, mk, np.nan)
        prof[k] = float(np.median(os_[a_:b_]))
    hz, hu = np.meshgrid(zc, u0 + (np.arange(nc) + 0.5) * (u1 - u0) / nc, indexing="ij")
    rin = in_interior(r, hu, hz).sum(1)              # the cells of each row inside the outline
    hasr = np.isfinite(C)
    ndr = hasr.sum(1)
    enough = (rin > 0) & (ndr >= 0.5 * np.maximum(rin, 1))
    out["cover"] = [round(float(elig[nin > 0].mean()) if (nin > 0).any() else 0.0, 3),
                    round(float(enough[rin > 0].mean()) if (rin > 0).any() else 0.0, 3)]
    if min(out["cover"]) < 0.5:
        out.update(ok=None, why=f"its interior cannot be measured: columns / rows with data {out['cover']} (< 0.5)")
        return out
    Cz = np.where(hasr, C, -np.inf)
    row = enough & ((Cz - float(claimed) >= step).sum(1) >= share * np.maximum(ndr, 1) - 1e-9)
    fin = np.isfinite(prof)
    for k in np.nonzero(enough & fin & ~row)[0]:
        d = zc - zc[k]
        flanks = []
        for sgn in (-1.0, 1.0):
            q = fin & (sgn * d >= INTERIOR_FLANK_M[0] - 1e-9) & (sgn * d <= INTERIOR_FLANK_M[1] + 1e-9)
            if q.sum() >= 3:
                flanks.append(float(np.median(prof[q])))
        if len(flanks) < 2:
            continue                                 # a member stands BETWEEN two parts of the recess
        back = max(0.5 * (flanks[0] + flanks[1]), float(claimed))
        # a bump, not a step: it rises over EACH flank too (by half the rise) -- the top of a tall partial object
        # (statues standing in an arch) or a recess whose lower part is shallower is no member between two recesses
        if prof[k] - back >= mstep - 1e-9 and prof[k] - max(flanks) >= 0.5 * mstep - 1e-9 \
                and (Cz[k] - back >= 0.5 * mstep).sum() >= share * ndr[k] - 1e-9:
            row[k] = True
    out["h"] = [[round(z0 + s * bh, 3), round(z0 + e * bh, 3)] for s, e in _runs(list(row.astype(int)), 1)
                if e - s >= nbins_h]
    for axis in ("v", "h"):
        out[axis + "_m"] = round(max((y - x for x, y in out[axis]), default=0.0), 3)
    out["ok"] = not out["v"] and not out["h"]
    return out


def audit_points(r: dict, wall, points: np.ndarray, track: np.ndarray | None, others=None, min_points: int = 12,
                 prm: ReliefParams = DEFAULT_RELIEF, U: np.ndarray | None = None,
                 g: ReliefGuards = DEFAULT_GUARDS, frame: dict | None = None, strict: bool = False) -> dict:
    """Independent re-measurement of one element from the raw SfM points (no relief map, no levels, no cells): the
    median offset of the points inside its outline as built (in_interior: shrunk by up to 0.15 m; an arch's head
    included -- verify-r2 N10x: a springing line set just above the sill must not shrink what is re-measured; within
    max(0.3 m, |relief|) of the claimed or the base level) from the frame surface -- `frame` when given (the
    publication audit passes the wall's frame re-derived from the points, independent_frame: never the record's),
    else the record's own (the pipeline's G-R6, on the frame it just measured; strict: its stricter interior limits,
    ReliefGuards.interior_guard). ok when
      (1) it agrees with the claimed offset within max(recheck_abs_m, recheck_rel x |relief|);
      (2) it differs from the element's base (record_base: what the box would show without the element) toward the
          claimed side by more than max(recheck_k x its standard error, recheck_min_m) -- a shallow claim on a flat
          wall fails (2) and usually (1);
      (3) for an opening or a vertical, the points RIGHT BESIDE it (the ring: RING_IN_M-RING_OUT_M outside its left /
          right side over its height, shrunk as its interior, per side) show its step: on every side with points, the box minus the side
          lies toward the claimed side by at least ring_step_share of the claimed relief (and the minimum of (2)) --
          a surface the record's frame or base made up has no step to its surroundings; a larger step (a recess
          beside a proud pier, a pier beside a recessed bay) does not contradict the claim, a smaller one comes from
          mouldings at a jamb or a frame strip taller than the bays beside it, and (1) bounds the claim itself
          against the frame (bands span the wall: their reference is the frame alone);
      (4) for a recess (an opening, a setback, a groove), no structure at the frame plane INSIDE it
          (interior_structure, on the same points: a pier, a pilaster, a mullion or an entablature across >= 2/3 of
          it means the claim spans real structure -- two bays merged, the wall between them deleted).
    ok None (unverifiable) with fewer than min_points points in the box, or none beside it, or an interior that
    cannot be measured (interior_structure ok None). U: the wall's points in
    wall coordinates (wall_points), when already computed. Returns {"points", "measured_m", "claimed_m", "base_m",
    "se_m", "tol_m", "ring_m" (per side: box minus side, or None), "interior" (interior_structure), "ok", "why"}."""
    claimed = r.get("offset_m")
    fr = frame if frame is not None else (r.get("frame") or {})
    base = record_base(r)
    out = {"points": 0, "measured_m": None, "claimed_m": claimed, "base_m": None if base is None else round(base, 3),
           "se_m": None, "tol_m": None, "ring_m": None, "interior": None, "ok": False, "why": ""}
    if claimed is None or "c0" not in fr or base is None:
        out["why"] = "no frame / offset / depth"
        return out
    if U is None:
        U, _ = wall_points(wall, points, np.full(len(points), prm.min_track) if track is None else track, prm, others)
    _, sz = _interior_shrink(r)
    rel = float(claimed) - base
    wnd = max(0.3, abs(rel))
    lo_w, hi_w = min(float(claimed), base) - wnd, max(float(claimed), base) + wnd
    o_all = U[:, 1] - (float(fr["c0"]) + float(fr["c1"]) * U[:, 0])
    # only points near the two levels in question (the claimed surface and the base): an object standing in front of
    # the wall (or a hole far behind it) says nothing about which of the two the wall shows
    near = (o_all >= lo_w) & (o_all <= hi_w)

    def offsets(m):
        return o_all[m & near]

    zin = (U[:, 2] >= float(r["z_bottom"]) + sz) & (U[:, 2] <= float(r["z_top"]) - sz)     # the ring's height
    inbox = in_interior(r, U[:, 0], U[:, 2])
    off = offsets(inbox)
    n = int(len(off))
    out["points"] = n
    if n < min_points:
        out.update(ok=None, why=f"{n} points in its box (< {min_points})")
        return out
    meas = float(np.median(off))
    sig = 1.4826 * float(np.median(np.abs(off - meas)))
    se = 1.2533 * sig / math.sqrt(n)
    tol = max(g.recheck_abs_m, g.recheck_rel * abs(rel))
    sgn = 1.0 if rel >= 0 else -1.0
    agree = abs(meas - float(claimed)) <= tol
    contrast = (meas - base) * sgn
    need = max(g.recheck_k * se, g.recheck_min_m)
    ok = bool(agree and contrast >= need)
    why = "" if ok else ("measured {:+.3f} m vs claimed {:+.3f} m (tolerance {:.3f} m)".format(meas, float(claimed), tol)
                         if not agree else "measured relief {:.3f} m over its base, under {:.3f} m".format(contrast, need))
    out.update(measured_m=round(meas, 3), se_m=round(se, 4), tol_m=round(tol, 3))
    if r.get("type") in ("opening", "vertical"):
        L = float(wall.length)
        sides = []
        for a, b in ((r["u0"] - RING_OUT_M, r["u0"] - RING_IN_M), (r["u1"] + RING_IN_M, r["u1"] + RING_OUT_M)):
            a, b = max(a, 0.0), min(b, L)
            if b - a < 0.1:
                sides.append(None)
                continue
            so = offsets((U[:, 0] >= a) & (U[:, 0] <= b) & zin)
            sides.append(round(meas - float(np.median(so)), 3) if len(so) >= min_points else None)
        out["ring_m"] = sides
        got = [x for x in sides if x is not None]
        if not got:
            out.update(ok=None, why=f"no points beside it (< {min_points} per side)")
            return out
        step_min = max(g.ring_step_share * abs(rel), g.recheck_min_m)
        if ok and not all(x * sgn >= step_min for x in got):
            ok = False
            why = "beside it the points show a step of {} m, not the claimed {:+.3f} m (tolerance {:.3f} m)".format(
                [f"{x:+.3f}" for x in got], rel, tol)
    if is_recess(r):
        sel = inbox & near
        inner = interior_structure(U[sel, 0], o_all[sel], U[sel, 2], r, float(claimed), base, g, strict)
        out["interior"] = inner
        if inner["ok"] is None and ok:       # a definite failure above stands; a pass needs the interior measured
            out.update(ok=None, why=inner.get("why") or "its interior cannot be measured")
            return out
        if ok and not inner["ok"]:
            ok = False
            why = "inside it the points show structure at the frame plane (a {}): the claim spans it".format(
                "; ".join([f"vertical strip u {a:.2f}-{b:.2f} m" for a, b in inner["v"]]
                          + [f"horizontal strip z {a:.2f}-{b:.2f} m" for a, b in inner["h"]]))
    out.update(ok=ok, why=why)
    return out


# ---------------------------------------------------------------- geometry against its records (the audits)

PLANE_TOL_M = 1e-3        # a wall node without relief records: every vertex on its plane
FRAME_TOL_M = 0.1         # a record's frame against the wall's frame re-derived from the points (independent_frame)
GEOM_TOL_M = 0.005        # relief geometry against its records: (u, z) and offset (records keep mm; the GLB float32)


def plane_violation(pos) -> float:
    """Largest distance of a node's vertices from their best-fit plane (m)."""
    p = np.asarray(pos, float).reshape(-1, 3)
    if len(p) < 4:
        return 0.0
    c = p.mean(0)
    vt = np.linalg.svd(p - c, full_matrices=False)[2]
    return float(np.abs((p - c) @ vt[-1]).max())


def geometry_violations(u, d, z, tris, recs: list, tol: float = GEOM_TOL_M) -> list[str]:
    """A relief node's geometry against its records, in wall coordinates (u along the wall, d outward from the wall
    plane, z up): every vertex off the wall plane lies inside the (u, z) box of a record marked built, at that
    record's offset or its parent's (a nested opening's reveal starts at its parent's back); every built record has a
    face (a triangle) at its offset inside its box. Geometry without records, deepened or shifted geometry and records
    without geometry all fail."""
    v: list[str] = []
    u, d, z = (np.asarray(x, float).reshape(-1) for x in (u, d, z))
    T = np.asarray(tris, np.int64).reshape(-1, 3)
    recs = [r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else []
    by_id = {r.get("id"): r for r in recs}
    boxes = []
    for r in recs:
        if not r.get("built"):
            continue
        try:
            b = tuple(float(r[k]) for k in ("u0", "u1", "z_bottom", "z_top", "offset_m"))
        except (KeyError, TypeError, ValueError):
            v.append(f"built record {r.get('id')} without its box or offset")
            continue
        po = None
        par = by_id.get(r.get("parent")) if r.get("parent") is not None else None
        if par is not None and par.get("offset_m") is not None:
            po = float(par["offset_m"])
        boxes.append((r, b, po))
    explained = np.zeros(len(u), bool)
    inbox = []
    for r, (u0, u1, z0, z1, o), po in boxes:
        inb = (u >= u0 - tol) & (u <= u1 + tol) & (z >= z0 - tol) & (z <= z1 + tol)
        lev = np.abs(d - o) <= tol
        if po is not None:
            lev = lev | (np.abs(d - po) <= tol)
        explained |= inb & lev
        inbox.append(inb & (np.abs(d - o) <= tol))
    bad = (np.abs(d) > PLANE_TOL_M) & ~explained
    if bad.any():
        v.append(f"{int(bad.sum())} vertices off the wall plane that no built record explains "
                 f"(max |d| {float(np.abs(d[bad]).max()):.3f} m)")
    for (r, (u0, u1, z0, z1, o), po), at in zip(boxes, inbox):
        ok = False
        if len(T) and abs(o) > PLANE_TOL_M:
            sel = at[T].all(axis=1)
            if sel.any():
                tu, tz = u[T[sel]], z[T[sel]]
                area = 0.5 * np.abs((tu[:, 1] - tu[:, 0]) * (tz[:, 2] - tz[:, 0]) - (tu[:, 2] - tu[:, 0]) * (tz[:, 1] - tz[:, 0]))
                ok = bool((area > 1e-4).any())
        if not ok:
            v.append(f"built record {r.get('id')}: no face at its offset {o:+.3f} m inside its box")
    return v


def record_signature(r: dict) -> tuple:
    """Coordinates and relief of a record to the mm: two independent measurements on two walls never share it."""
    sr = signed_relief(r)
    try:
        box = tuple(round(float(r[k]), 3) for k in ("u0", "u1", "z_bottom", "z_top"))
    except (KeyError, TypeError, ValueError):
        box = None
    return (r.get("type"), box, None if sr is None else round(sr, 3))


def copied_records(nodes: list[tuple[str, Any, list]]) -> list[dict]:
    """Records copied between walls: nodes = [(node name, its wall index, its relief records)]. A record whose
    signature (record_signature) appears on ANOTHER wall is a copy, not a measurement (the pieces of one wall share
    their elements legitimately), and a record's id names the wall it was measured on (w<wall>-...)."""
    first: dict[tuple, Any] = {}
    out = []
    for name, wall, recs in nodes:
        for r in recs if isinstance(recs, list) else []:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id") or "")
            if wall is not None and not rid.startswith(f"w{wall}-"):
                out.append({"node": name, "element": r.get("id"), "why": f"record id {rid!r} does not name wall {wall}"})
            sig = record_signature(r)
            if sig[1] is None:
                continue
            w0 = first.setdefault(sig, (wall, name))
            if w0[0] != wall:
                out.append({"node": name, "element": r.get("id"),
                            "why": f"same coordinates and relief as a record on {w0[1]}: a copy, not a measurement"})
    return out
