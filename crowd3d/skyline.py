"""Skyline evidence for structure parts (docs/STRUCTURE_FIRST.md §2.3, §2.4.2-§2.4.3).

Hover reads roof outlines from lateral photos; here every structure part (crowd3d.structure) is projected into
the posed photos and its upper edge is compared with where the photo's sky ends. This only ever produces
evidence about parameters of parts the photos show against the sky; crowd3d.judge decides what the evidence
may change (in v0: nothing automatically, D15).

- Sky masks: A+R, a classical method (smooth top-connected seeds -> Lab colour model -> GrabCut, then a
  full-resolution GrabCut band refinement of the boundary). OpenCV only, no learned weights, commercial-use safe.
  Other masks (learned ones, after a licence review) plug in as plain bool arrays.
- Rasterizer: per camera and part, the first image row the part's silhouette covers in every column
  (near-plane clipped triangles, cv2.fillConvexPoly).
- Cost per photo (the probe's measured "forward + reverse" cost): in the upper band of the landmark (its
  outline extruded to 1.35 x the tallest part, minus the lowest 6 m where people and buses live), the model's
  top row per column against the photo's sky boundary: forward = distance from the model edge to the nearest
  sky / non-sky boundary pixel, reverse = row distance from the photo's top-most sky -> building transition to the
  model edge, both truncated at 40 px, summed over the band's columns and divided by their number.
- Profiles: one parameter of one part at a time, every other part at its map value (a part never inherits an
  unaccepted change of another), on a grid of multipliers (heights and roof heights 0.70-1.30 in 0.05 steps,
  outline scale 0.90-1.10 in 0.02 steps, dome / onion radius 0.70-1.30); children follow the part (they stand
  on it). 200 bootstrap resamples of the photos give a 90 % interval. Photos counted at the map value: the part is
  the model skyline over >= 10 band columns ("visible"), and of those, it has sky 6 px above it ("sky").
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import structure as S
from .stylize import PosedCamera

TRUNC_PX = 40.0          # cost truncation
LOW_BAND_M = 6.0         # the lowest metres of the landmark are not skyline (people, kiosks, buses)
CAP_FACTOR = 1.35        # the band reaches 1.35 x the tallest part
MIN_SKY_COLS = 10        # a photo sees a part as skyline / against the sky with at least this many columns
SKY_MARGIN_PX = 6        # "sky above": the mask is sky this many pixels above the part's edge
GRID = np.round(np.arange(0.70, 1.3001, 0.05), 3)
GRID_OUTLINE = np.round(np.arange(0.90, 1.1001, 0.02), 3)


# ---------------------------------------------------------------- sky masks (A+R)


def _resize_long(rgb: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    H, W = rgb.shape[:2]
    s = long_side / max(H, W)
    return cv2.resize(rgb, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA), s


def _grabcut_from_prob(rgb_small: np.ndarray, prob: np.ndarray, iters: int = 4) -> np.ndarray:
    m = np.full(prob.shape, cv2.GC_PR_BGD, np.uint8)
    m[prob >= 0.5] = cv2.GC_PR_FGD
    hi = cv2.erode((prob >= 0.9).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lo = cv2.erode((prob <= 0.1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m[hi], m[lo] = cv2.GC_FGD, cv2.GC_BGD
    if hi.sum() < 30 or lo.sum() < 30:
        return prob >= 0.5
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR), m, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return prob >= 0.5
    return (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)


def sky_mask_classical(rgb: np.ndarray, work: int = 640, g_thr: float = 10.0) -> np.ndarray:
    """A: smooth regions touching the image top (mostly in its upper part) seed a Lab colour model of the sky;
    similar smooth pixels connected to the seeds form the candidate; GrabCut decides; islands not connected to the
    seeds are dropped. Returns a bool mask at the input resolution (all False when no sky is found)."""
    H, W = rgb.shape[:2]
    small, _ = _resize_long(rgb, work)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    g = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((g < g_thr).astype(np.uint8), connectivity=4)
    top_rows = max(2, int(0.03 * h))
    seeds = np.zeros((h, w), bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_TOP] <= top_rows and stats[k, cv2.CC_STAT_AREA] >= 0.004 * h * w:
            comp = lbl == k
            if np.median(np.nonzero(comp)[0]) < 0.6 * h:
                seeds |= comp
    if seeds.sum() < 0.003 * h * w:
        return np.zeros((H, W), bool)
    px = lab[seeds]
    mu = np.median(px, 0)
    ic = np.linalg.inv(np.cov((px - mu).T) + np.diag([25.0, 4.0, 4.0]))
    d = lab.reshape(-1, 3) - mu
    md = np.sqrt(np.einsum("ij,jk,ik->i", d, ic, d)).reshape(h, w)
    cand = ((md < 3.5) & (g < 2.5 * g_thr)).astype(np.uint8)
    _, lbl2 = cv2.connectedComponents(cand | seeds.astype(np.uint8), connectivity=4)
    keep = np.unique(lbl2[seeds])
    region = np.isin(lbl2, keep[keep > 0])
    prob = np.where(region, 0.7, 0.3).astype(np.float32)
    prob[cv2.erode(seeds.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = 0.95
    far = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3) > 12
    prob[far & (g > g_thr)] = 0.05
    m = _grabcut_from_prob(small, prob)
    _, lbl3 = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    keep = np.unique(lbl3[seeds & m])
    m = np.isin(lbl3, keep[keep > 0])
    return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5


def band_refine(rgb: np.ndarray, mask: np.ndarray, r: int = 6, iters: int = 2) -> np.ndarray:
    """R: GrabCut at full resolution in a +-r px band around the mask boundary (outside it the input decides):
    snaps a coarse boundary to the photo's own edges."""
    m8 = mask.astype(np.uint8)
    if m8.min() == m8.max():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    band = (cv2.dilate(m8, k) > 0) & ~(cv2.erode(m8, k) > 0)
    gc = np.where(mask, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
    gc[band & mask] = cv2.GC_PR_FGD
    gc[band & ~mask] = cv2.GC_PR_BGD
    ys, xs = np.nonzero(band)
    y0, y1 = max(0, ys.min() - 2 * r), min(mask.shape[0], ys.max() + 2 * r + 1)
    x0, x1 = max(0, xs.min() - 2 * r), min(mask.shape[1], xs.max() + 2 * r + 1)
    sub = np.ascontiguousarray(gc[y0:y1, x0:x1])
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2BGR), sub, None, bgd, fgd, iters,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask.copy()
    out = mask.copy()
    out[y0:y1, x0:x1] = (sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)
    return out


def sky_mask(rgb: np.ndarray) -> np.ndarray:
    """A+R (the commercial-use-safe mask)."""
    return band_refine(rgb, sky_mask_classical(rgb))


# ---------------------------------------------------------------- rasterizer


def cam_space(cam: PosedCamera, xyz: np.ndarray) -> np.ndarray:
    return (np.asarray(xyz, float).reshape(-1, 3) - cam.center) @ cam.rot.T


def clip_near(tc: np.ndarray, near: float = 0.3) -> list[np.ndarray]:
    """Clip camera-space triangles (n, 3, 3) against z >= near; returns polygons (k, 3)."""
    out = []
    z = tc[:, :, 2]
    allin = (z >= near).all(1)
    out.extend(tc[allin])
    for t in tc[~allin & (z >= near).any(1)]:
        poly = []
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            ina, inb = a[2] >= near, b[2] >= near
            if ina:
                poly.append(a)
            if ina != inb:
                poly.append(a + (near - a[2]) / (b[2] - a[2]) * (b - a))
        if len(poly) >= 3:
            out.append(np.array(poly))
    return out


def to_px(cam: PosedCamera, pc: np.ndarray) -> np.ndarray:
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    if cam.k1 or cam.k2:
        r2 = x * x + y * y
        f = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2
        x, y = x * f, y * f
    return np.stack([cam.fx * x + cam.cx, cam.fy * y + cam.cy], 1)


def raster_mask(cam: PosedCamera, tris: np.ndarray, shape: tuple[int, int], near: float = 0.3) -> np.ndarray:
    """uint8 silhouette (0/1) of ENU triangles (n, 3, 3) in this camera at image size `shape` (H, W)."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    if not len(tris):
        return m
    tc = cam_space(cam, tris.reshape(-1, 3)).reshape(-1, 3, 3)
    tc = tc[(tc[:, :, 2] >= near).any(1)]
    for poly in clip_near(tc, near):
        px = np.clip(to_px(cam, poly), -1e5, 1e5)
        cv2.fillConvexPoly(m, np.round(px * 4).astype(np.int32), 1, lineType=cv2.LINE_8, shift=2)
    return m


def column_tops(cam: PosedCamera, tris: np.ndarray, H: int, W: int, near: float = 0.3) -> np.ndarray:
    """(W,) first image row the triangles' silhouette covers (H where none); rasterizes only their box."""
    out = np.full(W, H, np.int32)
    if not len(tris):
        return out
    tc = cam_space(cam, tris.reshape(-1, 3)).reshape(-1, 3, 3)
    tc = tc[(tc[:, :, 2] >= near).any(1)]
    if not len(tc):
        return out
    pxs = [np.clip(to_px(cam, p), -1e5, 1e5) for p in clip_near(tc, near)]
    allp = np.concatenate(pxs)
    x0, x1 = int(max(0, np.floor(allp[:, 0].min()))), int(min(W, np.ceil(allp[:, 0].max()) + 1))
    y0, y1 = int(max(0, np.floor(allp[:, 1].min()))), int(min(H, np.ceil(allp[:, 1].max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return out
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], float)
    for px in pxs:
        cv2.fillConvexPoly(m, np.round((px - off) * 4).astype(np.int32), 1, lineType=cv2.LINE_8, shift=2)
    any_ = m.any(0)
    t = np.argmax(m > 0, 0) + y0
    out[x0:x1][any_] = t[any_]
    return out


def prism_triangles(ring: np.ndarray, z0: float, z1: float) -> np.ndarray:
    from .stylize import triangulate_polygon
    ring = np.asarray(ring, float)
    tris = []
    n = len(ring)
    for k in range(n):
        a, b = ring[k], ring[(k + 1) % n]
        tris += [(np.r_[a, z0], np.r_[b, z0], np.r_[b, z1]), (np.r_[a, z0], np.r_[b, z1], np.r_[a, z1])]
    for i, j, k in triangulate_polygon(ring):
        tris.append((np.r_[ring[i], z1], np.r_[ring[j], z1], np.r_[ring[k], z1]))
    return np.array(tris, float)


# ---------------------------------------------------------------- per-photo data and cost


class CamData:
    """One photo against one landmark: the upper band (ROI), the distance to the sky boundary (D2) and the photo's
    top-most sky -> building row per band column (r_o)."""

    def __init__(self, cam: PosedCamera, sky: np.ndarray, hull_ring: np.ndarray, cap: float, index: int = 0) -> None:
        self.cam, self.sky, self.index = cam, sky, index
        H, W = sky.shape
        self.H, self.W = H, W
        hi = raster_mask(cam, prism_triangles(hull_ring, 0.0, cap), (H, W)) > 0
        lo = raster_mask(cam, prism_triangles(hull_ring, 0.0, LOW_BAND_M), (H, W)) > 0
        self.roi = hi & ~lo
        self.roi_cols = self.roi.any(0)
        self.n_roi_cols = int(self.roi_cols.sum())
        b = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        self.D2 = np.minimum(cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3), TRUNC_PX).astype(np.float32)
        trans = np.zeros_like(sky)
        trans[1:] = sky[:-1] & ~sky[1:]
        trans &= self.roi
        r_o = np.argmax(trans, 0).astype(float)
        r_o[~trans.any(0)] = np.nan
        self.r_o = r_o


def score(cd: CamData, r_m: np.ndarray) -> dict[str, Any]:
    """Cost of a model skyline r_m (W,) in one photo: forward (model edge to the photo's sky boundary) plus reverse
    (photo skyline to the model edge), per band column, truncated, over the band's columns."""
    H, W = cd.H, cd.W
    cols = cd.roi_cols
    has_m = (r_m < H) & (r_m > 0) & cols
    rr = np.clip(r_m, 0, H - 1).astype(int)
    ef = np.where(has_m, cd.D2[rr, np.arange(W)], 0.0)
    has_o = ~np.isnan(cd.r_o) & cols
    er = np.where(has_o, np.minimum(TRUNC_PX, np.abs(np.nan_to_num(cd.r_o) - r_m)), 0.0)
    n = max(cd.n_roi_cols, 1)
    return {"cost": float((ef.sum() + er.sum()) / n), "fwd": float(ef.sum() / n), "n_model_cols": int(has_m.sum()),
            "fwd_mean": float(ef[has_m].mean()) if has_m.any() else None}


def sky_cols_against(cd: CamData, r_m: np.ndarray, part_top: np.ndarray, margin: int = SKY_MARGIN_PX) -> np.ndarray:
    """Band columns where this part is the model's top and the photo shows sky `margin` px above it."""
    own = (part_top <= r_m + 1) & (part_top < cd.H) & (part_top > margin) & cd.roi_cols
    above = np.clip(part_top - margin, 0, cd.H - 1).astype(int)
    return own & cd.sky[above, np.arange(cd.W)]


# ---------------------------------------------------------------- profiles at the map value


def params_of(b: S.BuiltPart) -> list[str]:
    """The parameters a skyline can speak about for a built part: its wall extent ('eave'), its roof height
    ('roof', pitched roofs) and its outline scale ('scale', domes / onions and parts of >= 30 m2)."""
    ks = []
    if b.eave - b.z0 > 0.5:
        ks.append("eave")
    if b.roof.shape != "flat" and b.roof.height_m > 0:
        ks.append("roof")
    if b.roof.shape in ("dome", "onion") or b.spec.area >= 30.0:
        ks.append("scale")
    return ks


def grid_for(b: S.BuiltPart, key: str) -> np.ndarray:
    return GRID_OUTLINE if key == "scale" and b.roof.shape not in ("dome", "onion") else GRID


def _pick(grid: np.ndarray, tot: np.ndarray) -> int:
    """argmin with ties (flat profiles) broken toward the map value (1.0)."""
    tot = np.asarray(tot, float)
    return int(np.argmin(tot + 1e-6 * np.abs(np.asarray(grid) - 1.0) * max(abs(tot.min()), 1.0)))


def _boot(G: np.ndarray, per: np.ndarray, n: int = 200, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nc = per.shape[1]
    tot = per.sum(1)
    k = _pick(G, tot)
    if nc == 0:
        return {"argmin": 1.0, "p05": 1.0, "p95": 1.0, "at_bound": False, "lo_bound": False, "hi_bound": False,
                "curve": [round(float(t), 1) for t in tot]}
    b = np.array([float(G[_pick(G, per[:, rng.integers(0, nc, nc)].sum(1))]) for _ in range(n)])
    span = float(tot.max() - tot.min())
    at_bound = bool(k in (0, len(G) - 1) and span > 1e-6)
    return {"argmin": float(G[k]), "p05": float(np.quantile(b, 0.05)), "p95": float(np.quantile(b, 0.95)),
            "at_bound": at_bound, "lo_bound": at_bound and k == 0, "hi_bound": at_bound and k == len(G) - 1,
            "curve": [round(float(t), 1) for t in tot]}


@dataclass
class PartProfile:
    id: str
    n_vis: int                           # photos where the part is the model skyline (>= MIN_SKY_COLS) at the map value
    n_sky: int                           # ... and has sky just above it
    fwd_px: float | None                 # mean forward distance over its skyline columns at the map value
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    map_top: float = 0.0
    map_eave: float = 0.0
    z0: float = 0.0


class SkylineProblem:
    """All photos of one landmark against its structure (cameras in the structure's ENU frame).

    masks: per camera index, a bool sky mask at the camera's resolution (None: the photo is not used)."""

    def __init__(self, model: S.StructureModel, cameras: list[PosedCamera], masks: dict[int, np.ndarray],
                 max_cameras_per_part: int = 60, min_band_cols: int = 20) -> None:
        t0 = time.perf_counter()
        self.model = model
        self.built = {b.id: b for b in model.built}
        self.ids = [b.id for b in model.built]
        cap = CAP_FACTOR * max(b.top for b in model.built)
        self.cds: list[CamData] = []
        for i, cam in enumerate(cameras):
            sky = masks.get(i)
            if sky is None or not sky.any():
                continue
            if sky.shape != (cam.height, cam.width):
                sky = cv2.resize(sky.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST) > 0
            cd = CamData(cam, sky, model.main_rings[0], cap, i)
            if cd.n_roi_cols >= min_band_cols:
                self.cds.append(cd)
        self.tris = {pid: self.built[pid].triangles(base=False) for pid in self.ids}
        self.tops = {pid: [column_tops(cd.cam, self.tris[pid], cd.H, cd.W) for cd in self.cds] for pid in self.ids}
        self.union = [self._union(c) for c in range(len(self.cds))]
        self.max_cams = max_cameras_per_part
        self.children = {pid: [] for pid in self.ids}
        for p in model.specs:
            if p.parent in self.children and p.id in self.built:
                self.children[p.parent].append(p.id)
        self.setup_s = time.perf_counter() - t0

    def _union(self, c: int, override: dict[str, np.ndarray] | None = None) -> np.ndarray:
        r = np.full(self.cds[c].W, self.cds[c].H, np.int32)
        for pid in self.ids:
            t = override[pid] if override and pid in override else self.tops[pid][c]
            r = np.minimum(r, t)
        return r

    def family(self, pid: str) -> list[str]:
        out, todo = [], [pid]
        while todo:
            q = todo.pop()
            if q in out:
                continue
            out.append(q)
            todo += self.children.get(q, [])
        return out

    def stats_at_map(self, pid: str) -> tuple[int, int, float | None, list[int]]:
        n_vis = n_sky = 0
        e, cams = [], []
        for c, cd in enumerate(self.cds):
            r_m, t = self.union[c], self.tops[pid][c]
            own = (t <= r_m + 1) & (t < cd.H) & (t > SKY_MARGIN_PX) & cd.roi_cols
            if own.sum() >= MIN_SKY_COLS:
                n_vis += 1
                cams.append(c)
            if sky_cols_against(cd, r_m, t).sum() >= MIN_SKY_COLS:
                n_sky += 1
            fwd = (t <= r_m) & (t < cd.H) & (t > 0) & cd.roi_cols
            if fwd.any():
                e.append(cd.D2[np.clip(t[fwd], 0, cd.H - 1), np.nonzero(fwd)[0]])
        e = np.concatenate(e) if e else np.zeros(0)
        return n_vis, n_sky, (float(e.mean()) if len(e) else None), cams

    def _cams_for(self, fam: list[str], key_mult_hi: dict[str, dict[str, float]]) -> list[int]:
        """Photos where the part family (at its largest trial) is in view; at most max_cams, spread in bearing."""
        big = self.model.rebuild(key_mult_hi, ids=fam, closed=False)
        bb = {b.id: b for b in big}
        cams = []
        for c, cd in enumerate(self.cds):
            for pid in fam:
                if pid in bb and (column_tops(cd.cam, bb[pid].triangles(base=False), cd.H, cd.W) < cd.H).any():
                    cams.append(c)
                    break
        if len(cams) > self.max_cams:
            cen = S.poly_centroid(self.built[fam[0]].outer)
            ang = np.array([math.atan2(*(self.cds[c].cam.center[:2] - cen)[::-1]) for c in cams])
            order = np.argsort(ang)
            cams = [cams[i] for i in order[np.linspace(0, len(order) - 1, self.max_cams).round().astype(int)]]
        return sorted(set(cams))

    def profile_part(self, pid: str, n_boot: int = 200) -> PartProfile:
        b = self.built[pid]
        n_vis, n_sky, fwd, _ = self.stats_at_map(pid)
        prof = PartProfile(pid, n_vis, n_sky, fwd, map_top=b.top, map_eave=b.eave, z0=b.z0)
        fam = self.family(pid)
        for key in params_of(b):
            G = grid_for(b, key)
            cams = self._cams_for(fam, {pid: {key: float(G[-1])}})
            FR, FO, metres = [], [], []
            for v in G:
                built = {q.id: q for q in self.model.rebuild({pid: {key: float(v)}}, ids=fam, closed=False)}
                per_r, per_o = [], []
                tri = {q: built[q].triangles(base=False) for q in fam if q in built}
                for c in cams:
                    cd = self.cds[c]
                    o = {q: column_tops(cd.cam, tri[q], cd.H, cd.W) for q in tri}
                    s = score(cd, self._union(c, o))
                    per_r.append(s["cost"])
                    per_o.append(s["fwd"])
                FR.append(per_r)
                FO.append(per_o)
                me = built.get(pid)
                metres.append({"top": round(me.top, 2), "eave": round(me.eave, 2)} if me is not None else {"top": None, "eave": None})
            FR, FO = np.array(FR, float).reshape(len(G), -1), np.array(FO, float).reshape(len(G), -1)
            prof.params[key] = {"grid": [float(g) for g in G], "n_cams": len(cams), "fr": _boot(G, FR, n_boot),
                                "fo": _boot(G, FO, n_boot), "metres": metres}
        return prof

    def profile_all(self, ids: list[str] | None = None, log=None) -> dict[str, PartProfile]:
        out = {}
        for pid in ids or self.ids:
            t = time.perf_counter()
            out[pid] = self.profile_part(pid)
            if log:
                p = out[pid]
                log(f"  {pid} vis {p.n_vis} sky {p.n_sky} " + " ".join(
                    f"{k}:{v['fr']['argmin']}{'*' if v['fr']['at_bound'] else ''}" for k, v in p.params.items())
                    + f" ({time.perf_counter() - t:.1f}s)")
        return out
