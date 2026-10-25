"""D25's measurements (docs/STRUCTURE_FIRST.md §2.4.8): the numbers crowd3d.correct decides on.

Everything is measured by multi-view geometry on the photos' registration (never guessed, never generative):

- skyline (both sky masks): a part's top by the photos' sky boundary, on a grid that widens in stages when the
  minimum hits a bound (x0.70-1.30, then up to x1.6 / 2.0 / 2.5 and down to x0.4 / 0.1); pitched roofs by an
  eave x roof grid; a 'height pending' stub by its top in metres (3-45 m). The cost is crowd3d.skyline's (forward +
  reverse, truncated at 40 px over the landmark's upper band), the 90 % interval a 200-sample bootstrap over the
  photos, and the photo counts at the map value and at the measured value (the part is the model skyline in a photo
  over >= 10 band columns; sky 6 px above its edge).
- SfM points (track >= 3): a pitched part's roof-envelope fit (per 1 m cell the 2nd-highest point against the part's
  surface for trial eaves x roof heights: inliers within 0.75 m minus cells above, bootstrap over cells) and every
  part's all-round top (each quadrant's q98 over the exposed area).
- 3-D lines (crowd3d.lines3d, re-assigned to each version of the structure): apex / ridge / eave records.
- the registration box: every component of the registration moved to +-3 sigma (crowd3d.correct.box_poses) and the
  agreeing channels measured again. A registration error T moves photos, points and lines together; the skyline is
  measured with the model moved by T^-1 instead (the same relative motion: the cameras and their sky masks stay as
  they are, so a sample costs rasterizing the solids, not recomputing the masks).
- the scale-free anchor (opposite walls), the measured ground, the local ground and its tilt-injection test, the
  sky-carving upper bound and the lower-bound readings of D25-B.

correct_landmark() runs the rounds: judge at the map value, parents and taller disputed neighbours first, the box
for every candidate, the corrections written (crowd3d.correct.apply_*), the neighbours measured again on the new
model, until nothing changes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import correct as C
from . import judge as J
from . import skyline as K
from . import structure as S
from . import stylize as st

UP_STAGES = (1.30, 1.60, 2.00, 2.50)
DOWN_STAGES = (0.70, 0.40, 0.10)
STEP = 0.05
RM_GRID = tuple(float(x) for x in np.round(np.arange(0.5, 4.0001, 0.25), 3))
STUB_GRID = tuple(float(x) for x in np.arange(3.0, 45.01, 0.5))
N_BOOT = 200
FIT_TAU = 0.75
ADJ_GAP_M = 1.5


# ---------------------------------------------------------------- the registration perturbation T


def _lean_R(gx: float, gy: float) -> np.ndarray:
    t = math.hypot(gx, gy)
    if t < 1e-12:
        return np.eye(3)
    R, _ = C.pose_matrix({"tilt_deg": t, "azimuth_deg": math.degrees(math.atan2(gx, gy)), "pivot": [0, 0, 0]})
    return R


def pose_RT(pose: dict[str, float] | None, pivot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T(X) = R (X - c) + c + d: R = lean(gx, gy) @ yaw, d = (x, y, 0); c the pivot (the outline's centroid, z = 0)."""
    p = pose or {}
    y = math.radians(float(p.get("yaw", 0.0)))
    Ry = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    R = _lean_R(float(p.get("gx", 0.0)), float(p.get("gy", 0.0))) @ Ry
    return R, np.asarray(pivot, float), np.array([float(p.get("x", 0.0)), float(p.get("y", 0.0)), 0.0])


def transform(X: np.ndarray, pose: dict[str, float] | None, pivot, inverse: bool = False) -> np.ndarray:
    if not pose or all(abs(float(v)) < 1e-12 for v in pose.values()):
        return np.asarray(X, float)
    R, c, d = pose_RT(pose, pivot)
    A = np.asarray(X, float)
    sh = A.shape
    A = A.reshape(-1, 3)
    out = (A - c - d) @ R + c if inverse else (A - c) @ R.T + c + d
    return out.reshape(sh)


def transform_lines(lines: Any, pose, pivot):
    """A crowd3d.lines3d result (or bare Lines3D) moved by T (a scene fact under another registration)."""
    from . import lines3d as L3
    if lines is None or not pose:
        return lines
    res = lines
    L = res.lines if hasattr(res, "lines") and not hasattr(res, "A") else res
    L2 = L3.Lines3D(transform(L.A, pose, pivot), transform(L.B, pose, pivot), L.n_images, L.photos, L.support, L.resid)
    if L is res:
        return L2
    pts = None if res.points is None else transform(res.points, pose, pivot)
    return L3.LinesResult(res.parts, L2, res.stats, res.timings_s, res.detector, res.selection, pts)


# ---------------------------------------------------------------- the evidence


@dataclass
class Evidence25:
    """One landmark's photo evidence in the structure's ENU frame (after the snap)."""
    cameras: list
    masks: dict[str, dict[int, np.ndarray]]
    points: np.ndarray | None = None
    quality: np.ndarray | None = None
    lines: Any = None                     # LinesResult / Lines3D, or callable(model, pose) -> {pid: records}
    lines_params: Any = None
    snap_scale: float = 1.0
    grade: str | None = None              # 'snap-accepted' / 'crowd' / None (no correction)
    snap_meta: dict[str, Any] | None = None
    gps_residual_median_m: float | None = None
    reported_yaw_sigma_deg: float | None = None
    gravity_vp: dict[str, Any] | None = None     # the scene-verticals gravity reference (crowd3d.tilt) when measured
    max_cams: int = 60
    sky_eval: Any = None

    def track_points(self) -> np.ndarray:
        if self.points is None or not len(self.points):
            return np.zeros((0, 3))
        P = np.asarray(self.points, float)
        if self.quality is not None and len(self.quality) == len(P):
            P = P[np.asarray(self.quality)[:, 0] >= 3]
        return P


# ---------------------------------------------------------------- skyline


class SkyEval:
    """Per photo, both sky masks' data once (distance to the sky boundary, the sky -> building transitions); per
    registration sample a Frame (the landmark's band and every part's column tops with the model moved by T^-1)."""

    def __init__(self, cameras: list, masks: dict[str, dict[int, np.ndarray]]):
        self.cams = list(cameras)
        self.names = sorted(masks)
        self.data: dict[int, dict[str, Any]] = {}
        for i, cam in enumerate(self.cams):
            ms = {n: masks[n].get(i) for n in self.names}
            if any(m is None or not np.asarray(m).any() for m in ms.values()):
                continue
            row = {"H": cam.height, "W": cam.width, "masks": {}}
            for n, m in ms.items():
                m = np.asarray(m, bool)
                if m.shape != (cam.height, cam.width):
                    m = cv2.resize(m.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST) > 0
                b = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                D2 = np.minimum(cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3), K.TRUNC_PX).astype(np.float32)
                trans = np.zeros_like(m)
                trans[1:] = m[:-1] & ~m[1:]
                row["masks"][n] = {"sky": m, "D2": D2, "trans": trans}
            self.data[i] = row

    def frame(self, model: S.StructureModel, pose: dict[str, float] | None, pivot) -> Frame:
        return Frame(self, model, pose, pivot)


class Frame:
    def __init__(self, ev: SkyEval, model: S.StructureModel, pose, pivot, min_band_cols: int = 20):
        self.ev, self.model, self.pose, self.pivot = ev, model, pose, np.asarray(pivot, float)
        cap = K.CAP_FACTOR * max(b.top for b in model.built)
        hull = model.main_rings[0]
        hi_t = self.tf(K.prism_triangles(hull, 0.0, cap))
        lo_t = self.tf(K.prism_triangles(hull, 0.0, K.LOW_BAND_M))
        self.cd: dict[int, dict[str, Any]] = {}
        for i, row in ev.data.items():
            cam = ev.cams[i]
            H, W = row["H"], row["W"]
            roi = (K.raster_mask(cam, hi_t, (H, W)) > 0) & ~(K.raster_mask(cam, lo_t, (H, W)) > 0)
            cols = roi.any(0)
            if int(cols.sum()) < min_band_cols:
                continue
            r_o = {}
            for n, m in row["masks"].items():
                t = m["trans"] & roi
                r = np.argmax(t, 0).astype(float)
                r[~t.any(0)] = np.nan
                r_o[n] = r
            self.cd[i] = {"cols": cols, "n": int(cols.sum()), "r_o": r_o, "H": H, "W": W}
        self.cams = sorted(self.cd)
        self.built = {b.id: b for b in model.built}
        self._tops: dict[tuple[str, int], np.ndarray] = {}
        self._tris = {pid: self.tf(b.triangles(base=False)) for pid, b in self.built.items()}
        self._box = {pid: (t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)) for pid, t in self._tris.items() if len(t)}

    def tf(self, tris: np.ndarray) -> np.ndarray:
        if not len(tris):
            return tris
        return transform(tris, self.pose, self.pivot, inverse=True)

    def tops(self, pid: str, c: int) -> np.ndarray:
        key = (pid, c)
        if key not in self._tops:
            cam = self.ev.cams[c]
            self._tops[key] = K.column_tops(cam, self._tris.get(pid, np.zeros((0, 3, 3))), self.cd[c]["H"], self.cd[c]["W"])
        return self._tops[key]

    def _xrange(self, pid: str, c: int) -> tuple[float, float] | None:
        if pid not in self._box:
            return None
        lo, hi = self._box[pid]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        cam = self.ev.cams[c]
        uv, z = cam.project(corners)
        if (z <= 0.3).any():
            return (-1e9, 1e9)
        return float(uv[:, 0].min()), float(uv[:, 0].max())

    def union_except(self, fam: set[str], c: int, xr: tuple[float, float] | None = None) -> np.ndarray:
        W, H = self.cd[c]["W"], self.cd[c]["H"]
        r = np.full(W, H, np.int32)
        for pid in self.built:
            if pid in fam:
                continue
            if xr is not None:
                pr = self._xrange(pid, c)
                if pr is None or pr[1] < xr[0] - 2 or pr[0] > xr[1] + 2:
                    continue
            r = np.minimum(r, self.tops(pid, c))
        return r

    def cost(self, c: int, r_m: np.ndarray) -> dict[str, float]:
        cd = self.cd[c]
        row = self.ev.data[c]
        H, W = cd["H"], cd["W"]
        cols = cd["cols"]
        has_m = (r_m < H) & (r_m > 0) & cols
        rr = np.clip(r_m, 0, H - 1).astype(int)
        out = {}
        for n, m in row["masks"].items():
            ef = np.where(has_m, m["D2"][rr, np.arange(W)], 0.0)
            ro = cd["r_o"][n]
            has_o = ~np.isnan(ro) & cols
            er = np.where(has_o, np.minimum(K.TRUNC_PX, np.abs(np.nan_to_num(ro) - r_m)), 0.0)
            out[n] = float((ef.sum() + er.sum()) / max(cd["n"], 1))
        return out

    def counts(self, c: int, r_m: np.ndarray, t: np.ndarray) -> tuple[bool, dict[str, bool]]:
        cd = self.cd[c]
        own = (t <= r_m + 1) & (t < cd["H"]) & (t > K.SKY_MARGIN_PX) & cd["cols"]
        vis = int(own.sum()) >= K.MIN_SKY_COLS
        sky = {}
        above = np.clip(t - K.SKY_MARGIN_PX, 0, cd["H"] - 1).astype(int)
        for n, m in self.ev.data[c]["masks"].items():
            sky[n] = int((own & m["sky"][above, np.arange(cd["W"])]).sum()) >= K.MIN_SKY_COLS
        return vis, sky


def _family(model: S.StructureModel, pid: str) -> list[str]:
    built = {b.id for b in model.built}
    return [pid] + [q for q in S.descendants(model, pid) if q in built]


class PartSky:
    """The skyline of one part on one Frame: trial solids (multipliers on the part, its family following) scored on
    the photos that see the family at its largest trial (at most max_cams, spread in bearing)."""

    def __init__(self, frame: Frame, pid: str, big: dict[str, float], max_cams: int = 60):
        self.f, self.pid = frame, pid
        self.model = frame.model
        self.fam = _family(self.model, pid)
        self.famset = set(self.fam)
        bigb = {b.id: b for b in self.model.rebuild({pid: big}, ids=self.fam, closed=False)}
        cams = []
        self.xr = {}
        for c in frame.cams:
            hit = False
            xs = []
            for q in self.fam:
                if q not in bigb:
                    continue
                t = K.column_tops(frame.ev.cams[c], frame.tf(bigb[q].triangles(base=False)), frame.cd[c]["H"], frame.cd[c]["W"])
                cols = np.nonzero(t < frame.cd[c]["H"])[0]
                if len(cols):
                    hit = True
                    xs += [cols.min(), cols.max()]
            if hit:
                cams.append(c)
                self.xr[c] = (float(min(xs)), float(max(xs)))
        if len(cams) > max_cams:
            cen = S.poly_centroid(frame.built[pid].outer)
            ang = np.array([math.atan2(*(frame.ev.cams[c].center[:2] - cen)[::-1]) for c in cams])
            order = np.argsort(ang)
            cams = [cams[i] for i in order[np.linspace(0, len(order) - 1, max_cams).round().astype(int)]]
        self.cams = sorted(set(cams))
        self.others = {c: frame.union_except(self.famset, c, self.xr.get(c)) for c in self.cams}
        self._cache: dict[tuple, tuple] = {}

    def trial(self, ov: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        key = tuple(sorted((k, round(v, 6)) for k, v in ov.items()))
        if key in self._cache:
            return self._cache[key]
        built = {b.id: b for b in self.model.rebuild({self.pid: ov}, ids=self.fam, closed=False)}
        tris = {q: self.f.tf(built[q].triangles(base=False)) for q in self.fam if q in built}
        costs = {n: np.zeros(len(self.cams)) for n in self.f.ev.names}
        for j, c in enumerate(self.cams):
            cd = self.f.cd[c]
            r = self.others[c].copy()
            for q, tq in tris.items():
                r = np.minimum(r, K.column_tops(self.f.ev.cams[c], tq, cd["H"], cd["W"]))
            for n, v in self.f.cost(c, r).items():
                costs[n][j] = v
        me = built.get(self.pid)
        geo = {"top": float(me.top), "eave": float(me.eave)} if me is not None else {"top": math.nan, "eave": math.nan}
        self._cache[key] = (costs, geo)
        return costs, geo

    def stats(self, ov: dict[str, float] | None) -> tuple[int, dict[str, int]]:
        """Photos (of all the frame's photos) where the part, rebuilt with ov, is the model skyline / has sky above."""
        built = {b.id: b for b in self.model.rebuild({self.pid: ov or {}}, ids=self.fam, closed=False)}
        tris = {q: self.f.tf(built[q].triangles(base=False)) for q in self.fam if q in built}
        n_vis, n_sky = 0, {n: 0 for n in self.f.ev.names}
        for c in self.f.cams:
            cd = self.f.cd[c]
            tq = {q: K.column_tops(self.f.ev.cams[c], t, cd["H"], cd["W"]) for q, t in tris.items()}
            if not (tq[self.pid] < cd["H"]).any():
                continue
            r = self.f.union_except(self.famset, c)
            for t in tq.values():
                r = np.minimum(r, t)
            vis, sky = self.f.counts(c, r, tq[self.pid])
            n_vis += vis
            for n in sky:
                n_sky[n] += sky[n]
        return n_vis, n_sky


def _grid(lo: float, hi: float, step: float = STEP) -> list[float]:
    return [round(float(x), 4) for x in np.arange(lo, hi + 1e-9, step)]


def _boot(per: np.ndarray, n: int = N_BOOT, seed: int = 0) -> tuple[int, list[int]]:
    """argmin of the summed cost over a (grid, photos) matrix and its argmins over photo resamples."""
    rng = np.random.default_rng(seed)
    nc = per.shape[-1]
    k = int(np.argmin(per.sum(-1)))
    if nc == 0:
        return k, [k] * n
    return k, [int(np.argmin(per[..., rng.integers(0, nc, nc)].sum(-1))) for _ in range(n)]


def _subgrid(curve: np.ndarray, k: int) -> float:
    if 0 < k < len(curve) - 1:
        c0, c1, c2 = curve[k - 1], curve[k], curve[k + 1]
        den = c0 - 2 * c1 + c2
        if den > 1e-9:
            return float(np.clip(0.5 * (c0 - c2) / den, -0.5, 0.5))
    return 0.0


def _mask_result_1d(G: list[float], per: np.ndarray, tops: list[float], eaves: list[float], at_global: tuple[bool, bool]
                    ) -> dict[str, Any]:
    k, ks = _boot(per)
    curve = per.sum(1)
    flat = bool(curve.max() - curve.min() <= 1e-6)
    frac = _subgrid(curve, k)
    i0, i1 = max(k - 1, 0), min(k + 1, len(G) - 1)
    step_m = (tops[i1] - tops[i0]) / max(i1 - i0, 1)
    bt = [tops[x] for x in ks]
    at_lo, at_hi = (k == 0 and at_global[0]), (k == len(G) - 1 and at_global[1])
    return {"top": round(tops[k] + frac * step_m, 3), "top_grid": round(tops[k], 3), "eave": round(eaves[k], 3),
            "p05": round(float(np.quantile(bt, 0.05)), 3), "p95": round(float(np.quantile(bt, 0.95)), 3),
            "at_bound": bool((at_lo or at_hi) and not flat), "flat": flat, "mult": G[k], "grid": [G[0], G[-1]]}


def search_1d(ps: PartSky, key: str, start: float | None = None, window: int = 6) -> dict[str, dict[str, Any]]:
    """The widened 1-D search of one parameter ('eave' / 'roof'), both masks on the same trial solids. Without
    start: from x0.70-1.30, widening in stages while a mask's minimum sits on the grid's end; with start (a box
    sample): a window of +-window steps around it, moved / grown while a minimum sits on its edge (within the stages'
    outer bounds)."""
    lo_lim, hi_lim = DOWN_STAGES[-1], UP_STAGES[-1]
    if start is None:
        G = _grid(0.70, 1.30)
        su, sd = 0, 0
    else:
        G = _grid(max(lo_lim, start - window * STEP), min(hi_lim, start + window * STEP))
    while True:
        R = [ps.trial({key: g}) for g in G]
        per = {n: np.array([r[0][n] for r in R]) for n in ps.f.ev.names}
        ks = {n: int(np.argmin(per[n].sum(1))) for n in per}
        grow_hi = any(k == len(G) - 1 for k in ks.values()) and G[-1] < hi_lim - 1e-9
        grow_lo = any(k == 0 for k in ks.values()) and G[0] > lo_lim + 1e-9
        if start is None:
            if grow_hi and su + 1 < len(UP_STAGES):
                su += 1
                G = sorted(set(G + _grid(G[-1] + STEP, UP_STAGES[su])))
                continue
            if grow_lo and sd + 1 < len(DOWN_STAGES):
                sd += 1
                G = sorted(set(_grid(DOWN_STAGES[sd], G[0] - STEP) + G))
                continue
        else:
            if grow_hi:
                G = sorted(set(G + _grid(G[-1] + STEP, min(hi_lim, G[-1] + window * STEP))))
                continue
            if grow_lo:
                G = sorted(set(_grid(max(lo_lim, G[0] - window * STEP), G[0] - STEP) + G))
                continue
        break
    tops = [r[1]["top"] for r in R]
    eaves = [r[1]["eave"] for r in R]
    # the loop only stops with a minimum on the grid's end when that end is the widened search's outer bound
    out = {}
    for n in per:
        res = _mask_result_1d(G, per[n], tops, eaves, (True, True))
        if key == "roof":
            res.pop("eave", None)
        out[n] = {**res, "search": f"1-D {key} {G[0]:g}-{G[-1]:g}"}
    return out


def search_2d(ps: PartSky, start: tuple[float, float] | None = None, window: int = 4) -> dict[str, dict[str, Any]]:
    """Pitched roofs: eave x roof multipliers (eave widening in stages as search_1d; roof 0.5-4.0 in 0.25 steps)."""
    lo_lim, hi_lim = DOWN_STAGES[-1], UP_STAGES[-1]
    RM = list(RM_GRID)
    if start is None:
        EM = _grid(0.70, 1.30)
        rm = RM
        su, sd = 0, 0
    else:
        EM = _grid(max(lo_lim, start[0] - window * STEP), min(hi_lim, start[0] + window * STEP))
        j0 = int(np.argmin(np.abs(np.array(RM) - start[1])))
        rm = RM[max(0, j0 - window):j0 + window + 1]
    while True:
        R = {(e, r): ps.trial({"eave": e, "roof": r}) for e in EM for r in rm}
        per = {n: np.array([[R[(e, r)][0][n] for r in rm] for e in EM]) for n in ps.f.ev.names}
        ij = {n: divmod(int(np.argmin(per[n].sum(-1))), len(rm)) for n in per}
        e_hi = any(i == len(EM) - 1 for i, _ in ij.values()) and EM[-1] < hi_lim - 1e-9
        e_lo = any(i == 0 for i, _ in ij.values()) and EM[0] > lo_lim + 1e-9
        r_hi = start is not None and any(j == len(rm) - 1 for _, j in ij.values()) and rm[-1] < RM[-1] - 1e-9
        r_lo = start is not None and any(j == 0 for _, j in ij.values()) and rm[0] > RM[0] + 1e-9
        if start is None:
            if e_hi and su + 1 < len(UP_STAGES):
                su += 1
                EM = sorted(set(EM + _grid(EM[-1] + STEP, UP_STAGES[su])))
                continue
            if e_lo and sd + 1 < len(DOWN_STAGES):
                sd += 1
                EM = sorted(set(_grid(DOWN_STAGES[sd], EM[0] - STEP) + EM))
                continue
        else:
            if e_hi or e_lo or r_hi or r_lo:
                if e_hi:
                    EM = sorted(set(EM + _grid(EM[-1] + STEP, min(hi_lim, EM[-1] + window * STEP))))
                if e_lo:
                    EM = sorted(set(_grid(max(lo_lim, EM[0] - window * STEP), EM[0] - STEP) + EM))
                if r_hi or r_lo:
                    j0, j1 = RM.index(rm[0]), RM.index(rm[-1])
                    rm = RM[max(0, j0 - (window if r_lo else 0)):j1 + (window if r_hi else 0) + 1]
                continue
        break
    tops = np.array([[R[(e, r)][1]["top"] for r in rm] for e in EM])
    eaves = np.array([[R[(e, r)][1]["eave"] for r in rm] for e in EM])
    out = {}
    for n in per:
        flatp = per[n].reshape(len(EM) * len(rm), -1)
        k, ks = _boot(flatp)
        i, j = divmod(k, len(rm))
        curve = flatp.sum(1)
        bt, be = [tops.flat[x] for x in ks], [eaves.flat[x] for x in ks]
        at_b = (i == 0 and EM[0] <= lo_lim + 1e-9) or (i == len(EM) - 1 and EM[-1] >= hi_lim - 1e-9) or \
            (j == 0 and rm[0] <= RM[0] + 1e-9) or (j == len(rm) - 1 and rm[-1] >= RM[-1] - 1e-9)
        out[n] = {"top": round(float(tops[i, j]), 3), "eave": round(float(eaves[i, j]), 3),
                  "p05": round(float(np.quantile(bt, 0.05)), 3), "p95": round(float(np.quantile(bt, 0.95)), 3),
                  "eave_p05": round(float(np.quantile(be, 0.05)), 3), "eave_p95": round(float(np.quantile(be, 0.95)), 3),
                  "at_bound": bool(at_b and curve.max() - curve.min() > 1e-6), "flat": bool(curve.max() - curve.min() <= 1e-6),
                  "mult": [EM[i], rm[j]], "search": f"2-D eave {EM[0]:g}-{EM[-1]:g} x roof {rm[0]:g}-{rm[-1]:g}"}
    return out


def search_stub(ps: PartSky, b: S.BuiltPart, start: float | None = None, window: int = 8) -> dict[str, dict[str, Any]]:
    """A 'height pending' stub's flat top in metres (3-45 m, 0.5 m steps)."""
    L = max(b.eave - b.z0, 1e-6)
    Hs = list(STUB_GRID)
    if start is not None:
        j = int(np.argmin(np.abs(np.array(Hs) - start)))
        lo_i, hi_i = max(0, j - window), min(len(Hs) - 1, j + window)
    else:
        lo_i, hi_i = 0, len(Hs) - 1
    while True:
        H = Hs[lo_i:hi_i + 1]
        R = [ps.trial({"eave": float((h - b.z0) / L)}) for h in H]
        per = {n: np.array([r[0][n] for r in R]) for n in ps.f.ev.names}
        ks = {n: int(np.argmin(per[n].sum(1))) for n in per}
        if start is not None and any(k == 0 for k in ks.values()) and lo_i > 0:
            lo_i = max(0, lo_i - window)
            continue
        if start is not None and any(k == len(H) - 1 for k in ks.values()) and hi_i < len(Hs) - 1:
            hi_i = min(len(Hs) - 1, hi_i + window)
            continue
        break
    out = {}
    for n in per:
        k, ks_ = _boot(per[n])
        curve = per[n].sum(1)
        bt = [H[x] for x in ks_]
        at_b = (k == 0 and lo_i == 0) or (k == len(H) - 1 and hi_i == len(Hs) - 1)
        out[n] = {"top": float(H[k]), "p05": round(float(np.quantile(bt, 0.05)), 3), "p95": round(float(np.quantile(bt, 0.95)), 3),
                  "at_bound": bool(at_b and curve.max() - curve.min() > 1e-6), "flat": bool(curve.max() - curve.min() <= 1e-6),
                  "mult": float((H[k] - b.z0) / L), "search": f"stub metres {H[0]:g}-{H[-1]:g}"}
    return out


def part_kind(b: S.BuiltPart) -> str:
    if b.spec.params["height"].source == st.PROV_DEFAULT:
        return "stub"
    if b.roof.shape not in ("flat", "dome", "onion") and b.roof.height_m > C.ROOF_MIN_M:
        return "pitched"
    if b.roof.shape in ("dome", "onion") and b.roof.height_m > 0:
        return "dome"
    return "flat"


def skyline_channel(frame: Frame, b: S.BuiltPart, start: dict[str, Any] | None = None, max_cams: int = 60,
                    counts: bool = True) -> dict[str, Any]:
    """The skyline channel's raw numbers for one part (both masks), in crowd3d.correct's format."""
    kind = part_kind(b)
    t0 = time.perf_counter()
    if kind == "pitched":
        big = {"eave": UP_STAGES[-1], "roof": RM_GRID[-1]}
    elif kind == "dome":
        big = {"roof": UP_STAGES[-1]}
    elif kind == "stub":
        big = {"eave": float((STUB_GRID[-1] - b.z0) / max(b.eave - b.z0, 1e-6))}
    else:
        big = {"eave": UP_STAGES[-1]}
    ps = PartSky(frame, b.id, big, max_cams)
    if not ps.cams:
        return {"masks": {}, "n_vis_map": 0, "n_sky_map": 0, "why": "no photo sees the part"}
    if kind == "pitched":
        ms = search_2d(ps, None if start is None else tuple(start["mult"]))
        ovs = {n: {"eave": m["mult"][0], "roof": m["mult"][1]} for n, m in ms.items()}
    elif kind == "dome":
        ms = search_1d(ps, "roof", None if start is None else start["mult"])
        ovs = {n: {"roof": m["mult"]} for n, m in ms.items()}
    elif kind == "stub":
        ms = search_stub(ps, b, None if start is None else start["top"])
        ovs = {n: {"eave": m["mult"]} for n, m in ms.items()}
    else:
        ms = search_1d(ps, "eave", None if start is None else start["mult"])
        ovs = {n: {"eave": m["mult"]} for n, m in ms.items()}
    out = {"masks": ms, "kind": kind, "n_cams": len(ps.cams)}
    if counts:
        nv0, ns0 = ps.stats(None)
        out["n_vis_map"], out["n_sky_map"] = nv0, min(ns0.values()) if ns0 else 0
        for n, m in ms.items():
            nv, ns = ps.stats(ovs[n])
            m["n_vis_at"], m["n_sky_at"] = nv, ns[n]
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def skyline_sample(sk: dict[str, Any]) -> dict[str, Any] | None:
    """A box sample's skyline value (both masks, interior minima) and half its union interval."""
    ms = sk.get("masks") or {}
    if len(ms) < 2 or any(m.get("at_bound") or m.get("flat") for m in ms.values()):
        return None
    tops = [float(m["top"]) for m in ms.values()]
    lo, hi = min(float(m["p05"]) for m in ms.values()), max(float(m["p95"]) for m in ms.values())
    return {"value": round(float(np.mean(tops)), 3), "half": round(0.5 * (hi - lo), 3)}


def skyline_eave_sample(sk: dict[str, Any]) -> dict[str, Any] | None:
    ms = sk.get("masks") or {}
    es = [m.get("eave") for m in ms.values()]
    if len(es) < 2 or any(e is None for e in es) or any(m.get("at_bound") for m in ms.values()) or \
            abs(es[0] - es[1]) > C.SKY_MASKS_M or any(m.get("eave_p05") is None for m in ms.values()):
        return None
    lo, hi = min(m["eave_p05"] for m in ms.values()), max(m["eave_p95"] for m in ms.values())
    return {"value": round(float(np.mean(es)), 3), "half": round(0.5 * (hi - lo), 3)}


# ---------------------------------------------------------------- SfM points


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
    solid block at that value leaves nothing to see (trees or a crane over a roof that really is at the map value;
    verify-r1 trees-ns / ring). With fewer than INTERIOR_MIN_CELLS interior cells every cell counts (edge cells hold
    wall points too: stricter, never looser)."""
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
    c, z2, q, m, g = exposed_cells(model, b, P)
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
    out.update(below_share(b, ch, zh, bb.surface_z(ch), C.tol_of(b.top)))
    return out


def sfm_quad(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray, cell: float = 0.5, qq: float = 0.98,
             min_pts: int = C.SFM_QUAD_POINTS) -> dict[str, Any]:
    """Per quadrant of the part's exposed area (about its centre), the points' q98."""
    c, z2, q, m, g = exposed_cells(model, b, P, cell, min_pts=1)
    cen = g.centres()[m].mean(0) if m.any() else S.poly_centroid(b.outer, b.holes)
    vals, counts = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            s = (np.sign(q[:, 0] - cen[0]) == sx) & (np.sign(q[:, 1] - cen[1]) == sy)
            counts.append(int(s.sum()))
            vals.append(round(float(np.quantile(q[s, 2], qq)), 3) if s.sum() >= min_pts else None)
    return {"quadrant_q98": vals, "quadrant_points": counts, "points": int(len(q))}


def sfm_channel(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray, start: dict[str, Any] | None = None,
                ceiling: bool = False) -> dict[str, Any]:
    """The all-round top (quadrants' q98) and, for every roof type, the part's own roof-surface fit (the only SfM
    value that may join an agreeing set: crowd3d.correct.sfm_verdict)."""
    out = {"quad": sfm_quad(model, b, P), "ceiling": bool(ceiling)}
    kind = part_kind(b)
    if kind != "stub":
        out["fit"] = sfm_fit(model, b, P, None if start is None or not (start.get("fit") or {}).get("mult")
                             else tuple(start["fit"]["mult"]), kind=kind)
    return out


def sfm_sample(sf: dict[str, Any], how: str) -> dict[str, Any] | None:
    if how in ("roof-fit", "surface-fit"):
        f = sf.get("fit") or {}
        if f.get("top") is None or f.get("at_bound"):
            return None
        return {"value": f["top"], "half": round(0.5 * (f["top_p95"] - f["top_p05"]), 3)}
    v = C.quad_top(sf.get("quad"))
    return None if v is None else {"value": round(v, 3), "half": 0.0}


def sfm_eave_sample(sf: dict[str, Any]) -> dict[str, Any] | None:
    f = sf.get("fit") or {}
    if f.get("eave") is None or f.get("at_bound"):
        return None
    return {"value": f["eave"], "half": round(0.5 * (f["eave_p95"] - f["eave_p05"]), 3)}


# ---------------------------------------------------------------- 3-D lines


def lines_records(ev: Evidence25, model: S.StructureModel, pose=None, pivot=None) -> dict[str, list[dict[str, Any]]]:
    if ev.lines is None:
        return {}
    if callable(ev.lines):
        return ev.lines(model, pose) or {}
    from . import lines3d as L3
    res = transform_lines(ev.lines, pose, pivot)
    if isinstance(res, L3.Lines3D):
        P = ev.track_points()
        res = L3.LinesResult({}, res, {}, {}, "", {}, transform(P, pose, pivot) if len(P) else None)
    return L3.remeasure(res, model, ev.lines_params or L3.DEFAULT_LINES, ev.snap_scale).evidence()


def lines_channel(recs: dict[str, list[dict[str, Any]]], pid: str) -> dict[str, Any]:
    keep = ("parameter", "how", "value", "implied_top", "interval", "n_photos", "n_lines", "at_bound", "vote",
            "vote_withheld", "search_m", "map")
    return {"records": [{k: r.get(k) for k in keep} for r in recs.get(pid, [])
                        if r.get("parameter") in ("top", "eave") and r.get("kind", "lines3d") == "lines3d"]}


def lines_sample(ln: dict[str, Any]) -> dict[str, Any] | None:
    r = C.lines_record(ln.get("records"))
    if r is None or r.get("at_bound"):
        return None
    v = float(r.get("implied_top") if r.get("implied_top") is not None else r["value"])
    iv = r.get("interval") or [v, v]
    return {"value": round(v, 3), "half": round(0.5 * (float(iv[1]) - float(iv[0])), 3)}


def lines_eave_sample(ln: dict[str, Any]) -> dict[str, Any] | None:
    for r in ln.get("records") or []:
        if r.get("parameter") == "eave" and r.get("value") is not None and not r.get("at_bound"):
            iv = r.get("interval") or [r["value"], r["value"]]
            return {"value": round(float(r["value"]), 3), "half": round(0.5 * (float(iv[1]) - float(iv[0])), 3)}
    return None


# ---------------------------------------------------------------- the part's own walls (U2)


def wall_top_run(z: np.ndarray, cell: np.ndarray, ncell: int, z_from: float) -> dict[str, Any] | None:
    """How high a wall surface's points rise contiguously: 0.5 m height bins from z_from, a bin 'seen' when its points
    fall in >= max(2, 20 %) of the edge's ncell free 1 m cells (cell: each point's cell along the edge; a narrow thing
    touching the wall -- a tree, a pole, a drainpipe -- covers too few cells to bridge a gap); from the lowest seen bin
    upwards while the next seen bin lies <= 2 m above; the run's top is the highest point of its last bin."""
    if not len(z):
        return None
    ncell = max(1, int(ncell))
    need = max(C.WALL_COVER_MIN, int(math.ceil(C.WALL_COVER * ncell)))
    zb = np.arange(z_from, float(z.max()) + C.WALL_BIN_M + 1e-9, C.WALL_BIN_M)
    if len(zb) < 2:
        return None
    k = np.clip(((z - z_from) / C.WALL_BIN_M).astype(int), 0, len(zb) - 2)
    seen = np.zeros(len(zb) - 1, bool)
    for i in np.unique(k):
        seen[i] = len(np.unique(cell[k == i])) >= need
    idx = np.nonzero(seen)[0]
    if not len(idx):
        return None
    top = int(idx[0])
    for i in idx[1:]:
        if zb[i] - zb[top + 1] > C.WALL_GAP_M + 1e-9:
            break
        top = int(i)
    zz = z[k == top]
    return {"from_m": round(float(zb[idx[0]]), 2), "top_m": round(float(zz.max()), 2), "cells": ncell, "need_cells": need}


def _mad(x: np.ndarray) -> float:
    m = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - m)))


def wall_planar(z: np.ndarray, d: np.ndarray, cell: np.ndarray, ncell: int, z_from: float, eave: float
                ) -> tuple[np.ndarray, dict[str, Any]]:
    """Above the map's eave a 0.5 m height bin of the wall slab (d: each point's offset from the plane chosen below
    the eave) counts as the wall only while it stays on it (crowd3d.correct WALL_ABOVE_*): its robust spread (1.4826
    MAD of d) <= min(0.25, max(0.15, 1.25 x the median spread of the seen bins below the eave)); its median within
    0.2 m of the last accepted bin's (starting from the wall's last 2 m below the eave: a wall may lean or recede
    gradually, a canopy hanging out steps); its 1 m cells >= max(the run's need, half the median below the eave).
    Tree crowns or a canopy over the eave fill the +-0.5 m slab with points (verify-r3 crowns: MAD 0.2-0.47 m); their
    bins are dropped, so the run stops at the wall's real top. Without a seen bin below the eave nothing is dropped
    (the run then starts above the eave and is refused for that). Returns (the points kept, the numbers)."""
    keep = np.ones(len(z), bool)
    ncell = max(1, int(ncell))
    need = max(C.WALL_COVER_MIN, int(math.ceil(C.WALL_COVER * ncell)))
    if not len(z):
        return keep, {"reference": None}
    zb = np.arange(z_from, float(z.max()) + C.WALL_BIN_M + 1e-9, C.WALL_BIN_M)
    if len(zb) < 2:
        return keep, {"reference": None}
    k = np.clip(((z - z_from) / C.WALL_BIN_M).astype(int), 0, len(zb) - 2)
    above = zb[:-1] + 0.5 * C.WALL_BIN_M > eave + 1e-9
    stats = {}
    for i in np.unique(k):
        m = k == i
        stats[int(i)] = (int(len(np.unique(cell[m]))), float(np.median(d[m])), _mad(d[m]))
    below = [i for i, (nc, _, _) in stats.items() if not above[i] and nc >= need]
    if not below:
        return keep, {"reference": None}
    mad_ref = float(np.median([stats[i][2] for i in below]))
    cells_ref = float(np.median([stats[i][0] for i in below]))
    near = np.isin(k, [i for i in below if zb[i] >= eave - C.WALL_REF_M - 1e-9])
    ref = float(np.median(d[near])) if near.any() else stats[max(below)][1]
    thr = min(C.WALL_ABOVE_MAD_MAX, max(C.WALL_ABOVE_MAD_MIN, C.WALL_ABOVE_MAD_REL * mad_ref))
    cneed = max(need, C.WALL_ABOVE_CELLS_REL * cells_ref)
    dropped = []
    for i in sorted(i for i in stats if above[i]):
        nc, med, mad = stats[i]
        ok = nc >= cneed - 1e-9 and mad <= thr + 1e-9 and abs(med - ref) <= C.WALL_ABOVE_STEP_M + 1e-9
        if ok:
            ref = med
        else:
            keep[k == i] = False
            dropped.append(round(float(zb[i]), 2))
    return keep, {"reference": round(ref, 3), "mad_ref": round(mad_ref, 3), "mad_max": round(thr, 3),
                  "cells_ref": round(cells_ref, 1), "dropped_bins": dropped}


def wall_reading(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray, wall_out_m: float = 3.0,
                 band_m: float = 0.5, sel: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(U2, crowd3d.correct.walls_bound) How high the part's own walls are seen. Per eave edge (an outline edge whose
    wall top is the eave): its 1 m cells with a taller part (top above the eave - 0.5 m, base under the eave) 0.5,
    1.5 or 2.5 m outside are dropped (their points are that part's wall: verify-r3, a tower abutting 6 of an annex's
    14 m), and so is the edge when < 2 m of it is left; the wall surface is the densest 0.1 m offset bin of the
    remaining points (track >= 3) in the edge's window (min(6 m, a quarter of the narrowest width) inside, 3 m
    outside) between the part's base + 1 m and the map's eave -- where its wall surely is -- refined to their median
    within 0.25 m (>= 30 points); its points within band_m of that plane, above the eave only the height bins that
    stay on it (wall_planar), give the contiguous wall top (wall_top_run), which must start at or below the map's
    eave. The reading is the highest edge's top (one wall seen rising to a height shows the part's wall is that
    tall; the rule compares it with the eave a correction builds).
    Returns (reading, selection). sel: the selection of the registration's base pose (per edge the wall points'
    indices into P, their 1 m cells and the edge's free cell count); given, the runs are recomputed on those points'
    heights in P -- a registration sample moves the scene rigidly, so its tilt changes the heights while which points
    are the part's wall stays the base pose's (re-finding it on an outline shifted by 3 m picks up whatever stands 3 m
    away)."""
    from . import lines3d as L3
    P = np.asarray(P, float).reshape(-1, 3)
    rows: list[dict[str, Any]] = []
    if sel is None:
        sel = []
        pred = L3.predicted_edges(b)
        tall = [o for o in model.built if o.id != b.id and o.top > b.eave - 0.5 and o.z0 < b.eave]
        w_in = min(L3.DEFAULT_LINES.wall_in_m, L3.DEFAULT_LINES.wall_in_share * max(b.spec.min_width(), 0.1))
        W = C.WALL_CELL_M
        for e in pred["eave"]:
            a2, b2, Le, n = e["a"], e["b"], float(e["len"]), e["normal"]
            u = (b2 - a2) / max(Le, 1e-9)
            ncell_all = max(1, int(math.ceil(Le / W)))
            lo_c = np.arange(ncell_all) * W
            hi_c = np.minimum(lo_c + W, Le)
            tmid = 0.5 * (lo_c + hi_c)
            covered = np.zeros(ncell_all, bool)
            for off in C.WALL_COVER_OFFSETS:
                probe = a2 + tmid[:, None] * u + off * n
                for o in tall:
                    covered |= st.points_in_polygon(probe, o.outer)
            free_len = float(np.sum((hi_c - lo_c)[~covered]))
            if free_len < 2.0:
                continue                         # behind taller parts (or too short): not this part's wall to read
            row: dict[str, Any] = {"edge": int(e["edge"]), "len_m": round(Le, 2)}
            if covered.any():
                row["cells_covered"] = int(covered.sum())
            if not len(P):
                sel.append({"row": dict(row, why="no points")})
                continue
            rp = P[:, :2] - a2
            t, d = rp @ u, rp @ n
            along = (t >= 0) & (t <= Le)
            cidx = np.clip((t / W).astype(int), 0, ncell_all - 1)
            along &= ~covered[cidx]
            band = along & (d >= -w_in) & (d <= wall_out_m) & (P[:, 2] >= b.z0 + 1.0) & (P[:, 2] <= b.eave)
            if int(band.sum()) < C.WALL_BAND_MIN:
                sel.append({"row": dict(row, why=f"{int(band.sum())} wall-band points (< {C.WALL_BAND_MIN})")})
                continue
            bins = np.arange(-w_in, wall_out_m + 0.1, 0.1)
            h, ed = np.histogram(d[band], bins=bins)
            j = int(np.argmax(h))
            mode = 0.5 * (ed[j] + ed[j + 1])
            near = band & (np.abs(d - mode) <= 0.25)
            plane = float(np.median(d[near])) if near.any() else float(mode)
            idx = np.nonzero(along & (np.abs(d - plane) <= band_m) & (P[:, 2] >= b.z0))[0]
            nfree = int((~covered).sum())
            keep, pl = wall_planar(P[idx, 2], d[idx] - plane, cidx[idx], nfree, float(b.z0), float(b.eave))
            row.update(offset_m=round(plane, 2), points=int(keep.sum()))
            if pl.get("dropped_bins"):
                row["off_plane_bins"] = pl["dropped_bins"]
            sel.append({"row": row, "idx": idx[keep], "cell": cidx[idx][keep], "ncell": nfree})
    for s_ in sel:
        row = dict(s_["row"])
        if "idx" in s_:
            run = wall_top_run(P[s_["idx"], 2], s_["cell"], s_["ncell"], float(b.z0))
            if run is None:
                row["why"] = "no 0.5 m height bin covers enough of the edge"
            else:
                row.update(run)
                if run["from_m"] > b.eave + 1e-9:
                    row["why"] = "the contiguous run starts above the map's eave"
        rows.append(row)
    tops = [r["top_m"] for r in rows if r.get("top_m") is not None and not r.get("why")]
    out: dict[str, Any] = {"edges": rows, "value": round(float(max(tops)), 2) if tops else None}
    if not tops:
        out["why"] = ("no eave edge's wall surface seen" if rows else "no eave edge in view (all behind taller parts)")
    return out, sel


# ---------------------------------------------------------------- anchors, ground


def wall_pairs(points: np.ndarray, quality, ring: np.ndarray, min_points: int = C.WALL_MIN_POINTS,
               min_sep: float = 10.0) -> list[dict[str, Any]]:
    """Opposite outline walls: per edge with >= min_points wall points (3-16 m, track >= 3) on its middle 80 %, the
    median offset; per pair of edges facing opposite ways, the measured / map separation (a shift cancels)."""
    ring = S.ccw(ring)
    xy = J._wall_points(points, quality, 3.0, 16.0)
    k, d, sgn = J._edge_assignment(xy, ring)
    rows = []
    for e in range(len(ring)):
        a, b = ring[e], ring[(e + 1) % len(ring)]
        L = float(np.linalg.norm(b - a))
        m = k == e
        if L < 3.0 or m.sum() < min_points:
            continue
        u = (b - a) / L
        s = (xy[m] - a) @ u
        mid = (s > 0.1 * L) & (s < 0.9 * L)
        if mid.sum() < min_points:
            continue
        n = np.array([u[1], -u[0]])
        rows.append({"edge": e, "a": a, "b": b, "n": n, "normal_deg": math.degrees(math.atan2(n[1], n[0])) % 360,
                     "points": int(mid.sum()), "offset": float(np.median(sgn[m][mid]))})
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r1, r2 = rows[i], rows[j]
            if abs(((r1["normal_deg"] - r2["normal_deg"]) % 360) - 180) > C.WALL_ANGLE_TOL:
                continue
            nbar = r1["n"] - r2["n"]
            nbar = nbar / np.linalg.norm(nbar)
            m1, m2 = 0.5 * (r1["a"] + r1["b"]), 0.5 * (r2["a"] + r2["b"])
            D = float(nbar @ (m1 - m2))
            if D < min_sep:
                continue
            Dm = D + r1["offset"] * float(r1["n"] @ nbar) - r2["offset"] * float(r2["n"] @ nbar)
            out.append({"edges": [r1["edge"], r2["edge"]], "normals_deg": [round(r1["normal_deg"], 1), round(r2["normal_deg"], 1)],
                        "points": [r1["points"], r2["points"]], "offsets_m": [round(r1["offset"], 3), round(r2["offset"], 3)],
                        "map_sep_m": round(D, 2), "measured_sep_m": round(Dm, 3), "k": round(Dm / D, 4)})
    return out


def ground_measured(model: S.StructureModel, P: np.ndarray, cameras: list) -> dict[str, Any]:
    """'The ground is measured': points outside every building outline (within 40 m of the main outline) have their
    1 % quantile within 0.5 m of z = 0, and the cameras' median height is 1-2.5 m."""
    rings = [model.main_rings[0]] + [p.outer for p in model.specs]
    out_ = np.ones(len(P), bool)
    for r in rings:
        out_ &= ~st.points_in_polygon(P[:, :2], r)
    near = st.distance_to_rings(P[:, :2], [model.main_rings[0]])[0] <= 40.0
    q = P[out_ & near]
    q01 = float(np.quantile(q[:, 2], 0.01)) if len(q) >= 50 else None
    cz = float(np.median([c.center[2] for c in cameras])) if cameras else None
    ok = q01 is not None and abs(q01) <= 0.5 and cz is not None and 1.0 <= cz <= 2.5
    return {"measured": bool(ok), "q01_m": None if q01 is None else round(q01, 3), "points": int(len(q)),
            "camera_median_z_m": None if cz is None else round(cz, 3)}


def local_ground(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray) -> dict[str, Any]:
    """The ground next to a part: points 2-15 m outside its outline and in no building outline, by 45-degree sector;
    every sector with >= 30 points has its 5 % quantile within 0.5 m of z = 0, and two such sectors >= 90 deg apart."""
    lg = C.LOCAL_GROUND
    rings = [model.main_rings[0]] + [p.outer for p in model.specs]
    d = st.distance_to_rings(P[:, :2], [b.outer])[0]
    inb = st.points_in_polygon(P[:, :2], b.outer)
    sel = (~inb) & (d >= lg["r_in_m"]) & (d <= lg["r_out_m"])
    for r in rings:
        sel &= ~st.points_in_polygon(P[:, :2], r)
    q = P[sel]
    cen = S.poly_centroid(b.outer, b.holes)
    ang = (np.degrees(np.arctan2(q[:, 1] - cen[1], q[:, 0] - cen[0])) % 360) if len(q) else np.zeros(0)
    w = 360.0 / lg["sectors"]
    sectors = []
    for s in range(lg["sectors"]):
        m = (ang >= s * w) & (ang < (s + 1) * w)
        if m.sum() >= lg["min_points"]:
            sectors.append({"sector_deg": s * w + w / 2, "points": int(m.sum()),
                            "q05_m": round(float(np.quantile(q[m, 2], 0.05)), 3)})
    good = [x for x in sectors if abs(x["q05_m"]) <= lg["z_tol_m"]]
    bad = [x for x in sectors if abs(x["q05_m"]) > lg["z_tol_m"]]
    sep = max((abs(((a["sector_deg"] - b_["sector_deg"]) + 180) % 360 - 180) for a in good for b_ in good), default=0.0)
    ok = bool(good and not bad and sep >= lg["min_sep_deg"])
    return {"passed": ok, "sectors": sectors, "max_sep_deg": round(sep, 1),
            "why": "" if ok else ("a populated sector's ground is off z = 0" if bad else "no two ground sectors >= 90 deg apart")}


def estimate_up_boot(cameras: list, n: int = 200, seed: int = 0) -> dict[str, Any]:
    """The bootstrap spread (per horizontal axis, degrees) of the cameras' up estimate (reconstruct.estimate_up)."""
    from .reconstruct import estimate_up
    if len(cameras) < 5:
        return {"sigma_deg": None}
    ups = np.array([-np.asarray(c.rot)[1] for c in cameras])
    rights = np.array([np.asarray(c.rot)[0] for c in cameras])
    cen = np.array([c.center for c in cameras])
    rng = np.random.default_rng(seed)
    tang = []
    for _ in range(n):
        i = rng.integers(0, len(cameras), len(cameras))
        u = estimate_up(ups[i], rights[i], cen[i])
        tang.append(np.degrees(np.arctan2(u[:2], u[2])))
    tang = np.array(tang)
    u0 = estimate_up(ups, rights, cen)
    return {"sigma_deg": round(float(tang.std(0).max()), 4), "up_tangent_deg": np.degrees(np.arctan2(u0[:2], u0[2])).round(3).tolist()}


# ---------------------------------------------------------------- D25-B readings


def block_lines(lines: Any, b: S.BuiltPart, gap: float = 0.6) -> dict[str, Any]:
    """The 3-D lines' lower bound in a block's footprint: horizontal lines (< 10 deg, >= 1 m) wholly inside; clusters
    in height (gap <= 0.6 m); the highest cluster with >= 3 lines from >= 3 photos; its lowest line."""
    out: dict[str, Any] = {"valid": False}
    L = getattr(lines, "lines", lines) if lines is not None and not callable(lines) else None
    if L is None or not hasattr(L, "A"):
        out["why"] = "no 3-D lines"
        return out
    A, B = np.asarray(L.A, float), np.asarray(L.B, float)
    if not len(A):
        out["why"] = "no 3-D lines"
        return out
    D = B - A
    Ln = np.linalg.norm(D, axis=1)
    slope = np.degrees(np.arcsin(np.clip(np.abs(D[:, 2]) / np.maximum(Ln, 1e-9), 0, 1)))
    ina = st.points_in_polygon(A[:, :2], b.outer) & st.points_in_polygon(B[:, :2], b.outer)
    idx = np.nonzero(ina & (slope < 10.0) & (Ln >= 1.0))[0]
    z = 0.5 * (A[idx, 2] + B[idx, 2])
    o = np.argsort(z)
    clusters, cur = [], []
    for k in o:
        if cur and z[k] - z[cur[-1]] > gap:
            clusters.append(cur)
            cur = []
        cur.append(k)
    if cur:
        clusters.append(cur)
    good = []
    for cl in clusters:
        photos = {p for k in cl for p in L.photos[idx[k]]}
        if len(cl) >= 3 and len(photos) >= 3:
            good.append((cl, photos))
    out["clusters"] = [[round(float(z[cl[0]]), 2), round(float(z[cl[-1]]), 2), len(cl)] for cl in clusters]
    if not good:
        out["why"] = "no horizontal cluster of >= 3 lines from >= 3 photos inside"
        return out
    cl, photos = good[-1]
    out.update(valid=True, value=round(float(z[cl].min()), 3), n_lines=len(cl), n_photos=len(photos),
               z_range=[round(float(z[cl].min()), 2), round(float(z[cl].max()), 2)])
    return out


def block_roof(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray) -> dict[str, Any]:
    """(U1 for D25-B, crowd3d.correct.block_roof_problem) The block's own roof over its interior: the 1 m cells of its
    exposed area (exposed_cells: minus the parts on it and 1.5 m around taller parts) whose centres lie >= INTERIOR_M
    inside its outline -- a scaffold net or a hoarding at the outline puts no point there -- each cell holding a point
    (track >= 3, above the base) giving its highest point ('tops'); 'roof_tops' the cells whose points do not reach
    into the lowest quarter of their height above the base (those are a vertical surface standing in the interior,
    a net or a wall, never a roof: a solid block has no point there); 'cells_footprint' the interior cells of the
    exposed area, seen or not."""
    P = np.asarray(P, float).reshape(-1, 3)
    c, z2, q, m, g = exposed_cells(model, b, P)
    cen = g.centres().reshape(g.ny, g.nx, 2)
    fc = cen[m]
    if len(fc):
        fi = st.points_in_polygon(fc, b.outer) & (st.distance_to_rings(fc, [b.outer, *b.holes])[0] >= INTERIOR_M)
    else:
        fi = np.zeros(0, bool)
    out: dict[str, Any] = {"interior_m": INTERIOR_M, "cells_footprint": int(fi.sum()), "tops": [], "roof_tops": []}
    if not len(q) or not fi.any():
        return out
    ix = np.clip(((q[:, 0] - g.lo[0]) / 1.0).astype(int), 0, g.nx - 1)
    iy = np.clip(((q[:, 1] - g.lo[1]) / 1.0).astype(int), 0, g.ny - 1)
    key = iy * g.nx + ix
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], q[order, 2]
    uk, start = np.unique(ks, return_index=True)
    zmax, zmin = np.maximum.reduceat(zs, start), np.minimum.reduceat(zs, start)
    cc = np.array([cen[k // g.nx, k % g.nx] for k in uk])
    inner = st.points_in_polygon(cc, b.outer) & (st.distance_to_rings(cc, [b.outer, *b.holes])[0] >= INTERIOR_M)
    vertical = zmin < b.z0 + C.BLOCK_ROOF_LOW * (zmax - b.z0)
    out["tops"] = sorted(round(float(z), 3) for z in zmax[inner])
    out["roof_tops"] = sorted(round(float(z), 3) for z in zmax[inner & ~vertical])
    return out


def block_sfm(model: S.StructureModel, b: S.BuiltPart, P: np.ndarray) -> dict[str, Any]:
    """SfM's lower bound on a block's footprint: points within 1 m of taller parts removed; per quadrant (about the
    centroid) q98 with >= 20 points; the 3rd-highest of >= 3 quadrants."""
    inside = st.points_in_polygon(P[:, :2], b.outer)
    Q = P[inside]
    keep = np.ones(len(Q), bool)
    for o in model.built:
        if o is b or o.top <= b.top + 1.0:
            continue
        keep &= ~(st.points_in_polygon(Q[:, :2], o.outer) | (st.distance_to_rings(Q[:, :2], [o.outer])[0] < 1.0))
    Q = Q[keep]
    cen = S.poly_centroid(b.outer, b.holes)
    vals, counts = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            s = (np.sign(Q[:, 0] - cen[0]) == sx) & (np.sign(Q[:, 1] - cen[1]) == sy)
            counts.append(int(s.sum()))
            vals.append(round(float(np.quantile(Q[s, 2], 0.98)), 3) if s.sum() >= C.SFM_QUAD_POINTS else None)
    v = C.quad_top({"quadrant_q98": vals, "quadrant_points": counts})
    out = {"quadrant_q98": vals, "quadrant_points": counts, "points": int(len(Q)), "valid": v is not None}
    if v is not None:
        out["value"] = round(v, 3)
    else:
        out["why"] = "fewer than 3 quadrants with >= 20 points"
    return out


def block_skyline(sk: dict[str, Any], need_counts: bool = True) -> dict[str, Any]:
    """The skyline's lower bound: the lower end of the 90 % interval (both masks agreeing, off the bounds, the model
    skyline in >= 10 photos and sky above in >= 5 at the value)."""
    ms = sk.get("masks") or {}
    out: dict[str, Any] = {"valid": False, "masks": {n: {k: m.get(k) for k in ("top", "p05", "p95", "n_vis_at", "n_sky_at",
                                                                             "at_bound")} for n, m in ms.items()}}
    if len(ms) < 2 or any(m.get("at_bound") or m.get("flat") for m in ms.values()):
        out["why"] = "one mask, a bound or a flat profile"
        return out
    tops = [float(m["top"]) for m in ms.values()]
    if max(tops) - min(tops) > max(C.SKY_MASKS_M, C.SKY_MASKS_REL * max(tops)):
        out["why"] = "the masks disagree"
        return out
    if need_counts and (min(int(m.get("n_vis_at") or 0) for m in ms.values()) < C.SKY_VIS_AT or
                        min(int(m.get("n_sky_at") or 0) for m in ms.values()) < C.SKY_ABOVE):
        out["why"] = "too few skyline / sky photos at the value"
        return out
    out.update(valid=True, value=round(min(float(m["p05"]) for m in ms.values()), 3), top=round(float(np.mean(tops)), 3))
    return out


def carve_upper_bound(b: S.BuiltPart, cameras: list, masks: dict[str, dict[int, np.ndarray]], vox: float = 0.5,
                      zmax: float = 60.0, erode: int = 3, min_votes: int = 3, min_frac: float = 0.2) -> dict[str, Any]:
    """Sky carving over a block's footprint: a voxel is empty when >= 3 photos and >= 20 % of those that see it show
    (eroded) sky there; per 0.5 m cell the lowest empty voxel bounds the block from above; the 25 % quantile (the
    minimum over masks)."""
    lo, hi = b.outer.min(0), b.outer.max(0)
    xs = np.arange(lo[0], hi[0] + 1e-9, vox) + vox / 2
    ys = np.arange(lo[1], hi[1] + 1e-9, vox) + vox / 2
    XX, YY = np.meshgrid(xs, ys)
    cells = np.c_[XX.ravel(), YY.ravel()]
    cells = cells[st.points_in_polygon(cells, b.outer)]
    zs = np.arange(0, zmax, vox) + vox / 2
    nz = len(zs)
    if not len(cells):
        return {"p25": None}
    Pv = np.c_[np.repeat(cells, nz, 0), np.tile(zs, len(cells))]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    per = {}
    for name, mm in masks.items():
        sky_v = np.zeros(len(Pv), np.int32)
        obs_v = np.zeros(len(Pv), np.int32)
        for i, sky in mm.items():
            cam = cameras[i]
            skye = cv2.erode(np.asarray(sky, np.uint8), k) > 0
            uv, z = cam.project(Pv)
            ok = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width - 0.5) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height - 0.5)
            u = np.round(uv[ok, 0]).astype(int)
            v = np.round(uv[ok, 1]).astype(int)
            if skye.shape != (cam.height, cam.width):
                skye = cv2.resize(skye.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST) > 0
            obs_v[ok] += 1
            sky_v[np.nonzero(ok)[0][skye[v, u]]] += 1
        carved = (sky_v >= min_votes) & (sky_v >= min_frac * np.maximum(obs_v, 1))
        Cg = carved.reshape(len(cells), nz)
        first = np.where(Cg.any(1), np.argmax(Cg, 1), nz)
        ub = np.where(first < nz, zs[np.minimum(first, nz - 1)] - vox / 2, np.inf)
        fin = np.isfinite(ub)
        per[name] = {"cells": int(len(cells)), "bounded": int(fin.sum()),
                     "p25": round(float(np.quantile(ub[fin], 0.25)), 3) if fin.any() else None,
                     "median": round(float(np.median(ub[fin])), 3) if fin.any() else None}
    ps = [v["p25"] for v in per.values() if v["p25"] is not None]
    return {"p25": min(ps) if len(ps) == len(per) and ps else None, "masks": per}


# ---------------------------------------------------------------- the driver


@dataclass
class Measurer:
    """Measures one landmark's parts under a registration sample (pose)."""
    ev: Evidence25
    model: S.StructureModel
    pivot: np.ndarray = field(default=None)
    _frames: dict = field(default_factory=dict)
    _lines: dict = field(default_factory=dict)
    _sfm_rows: dict | None = None
    _wall_sel: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pivot is None:
            c = st.polygon_centroid(S.ccw(self.model.main_rings[0]))
            self.pivot = np.array([c[0], c[1], 0.0])
        if self.ev.sky_eval is None and self.ev.masks:
            self.ev.sky_eval = SkyEval(self.ev.cameras, self.ev.masks)

    def key(self, pose):
        return tuple(sorted((k, round(float(v), 6)) for k, v in (pose or {}).items() if abs(float(v)) > 1e-12))

    def frame(self, pose):
        k = self.key(pose)
        if k not in self._frames:
            if len(self._frames) > 2:
                self._frames.clear()
            self._frames[k] = self.ev.sky_eval.frame(self.model, pose, self.pivot)
        return self._frames[k]

    def points(self, pose):
        return transform(self.ev.track_points(), pose, self.pivot)

    def lines(self, pose):
        k = self.key(pose)
        if k not in self._lines:
            if len(self._lines) > 2:
                self._lines.clear()
            self._lines[k] = lines_records(self.ev, self.model, pose, self.pivot)
        return self._lines[k]

    def ceiling(self, b: S.BuiltPart) -> bool:
        """The points reach the map's top all round and no higher (judge.channel_tops' ceiling), at the base pose."""
        if self._sfm_rows is None:
            self._sfm_rows = J.sfm_channel(self.model, self.points(None), None)
        row = self._sfm_rows.get(b.id) or {}
        if row.get("q98") is None or row.get("model_q98") is None:
            return False
        v = b.top + row["q98"] - row["model_q98"]
        return bool(v <= b.top and v >= b.top - C.tol_of(b.top) and (row.get("near_top_quadrants") or 0) >= 3
                    and (row.get("near_top_points") or 0) >= 20)

    def channels(self, pid: str, pose=None, base: dict[str, Any] | None = None, counts: bool = True) -> dict[str, Any]:
        b = self.model.built_part(pid)
        P = self.points(pose)
        out: dict[str, Any] = {}
        if self.ev.sky_eval is not None:
            st_ = None if base is None else _sky_start(base.get(C.SKYLINE))
            out[C.SKYLINE] = skyline_channel(self.frame(pose), b, st_, self.ev.max_cams, counts=counts)
        if len(P):
            ceil = self.ceiling(b) if base is None else False
            out[C.SFM] = sfm_channel(self.model, b, P, None if base is None else base.get(C.SFM), ceiling=ceil)
        out[C.LINES] = lines_channel(self.lines(pose), pid)
        # (U2) not a channel: how high its own walls are seen (the surface chosen at the base pose, its heights moved
        # with each registration sample)
        if base is None or pid not in self._wall_sel:
            out["walls"], self._wall_sel[pid] = wall_reading(self.model, b, P)
        else:
            out["walls"], _ = wall_reading(self.model, b, P, sel=self._wall_sel[pid])
        return out


def _sky_start(sk: dict[str, Any] | None) -> dict[str, Any] | None:
    ms = (sk or {}).get("masks") or {}
    if not ms:
        return None
    m = next(iter(ms.values()))
    return {"mult": m.get("mult"), "top": m.get("top")}


def sample_values(ch: dict[str, Any], verdicts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A box sample's value and half interval per channel (and the eave readings)."""
    how = (verdicts.get(C.SFM) or {}).get("how") or "all-round q98"
    vals = {C.SKYLINE: skyline_sample(ch.get(C.SKYLINE) or {}),
            C.SFM: sfm_sample(ch.get(C.SFM) or {}, how),
            C.LINES: lines_sample(ch.get(C.LINES) or {})}
    eave = {C.SKYLINE: skyline_eave_sample(ch.get(C.SKYLINE) or {}),
            C.SFM: sfm_eave_sample(ch.get(C.SFM) or {}),
            C.LINES: lines_eave_sample(ch.get(C.LINES) or {})}
    return vals, eave


def adjacency(model: S.StructureModel, gap: float = ADJ_GAP_M) -> dict[str, list[str]]:
    out = {}
    for b in model.built:
        out[b.id] = [o.id for o in model.built if o is not b
                     and float(st.distance_to_rings(o.outer, [b.outer])[0].min()) <= gap]
    return out


def registration_inputs(ev: Evidence25, model: S.StructureModel, P: np.ndarray) -> dict[str, Any]:
    gm = ground_measured(model, P, ev.cameras) if len(P) else {"measured": False}
    eb = estimate_up_boot(ev.cameras)
    snap = ev.snap_meta or {}
    out = {"ground": gm, "estimate_up": eb, "snap_yaw_sigma_deg": (snap.get("search") or {}).get("yaw_sigma_deg"),
           **snap_alternative_poses(snap)}
    if ev.grade == C.GRADE_SNAP and model.main_rings:
        # the snap gate accepts symmetric aliases (REAL_WORLD_EVAL §7 item 1, task #73, unfixed): every alias of the
        # main outline is a registration alternative the snap-grade box samples (yaw about the outline's centroid)
        out["aliases"] = C.outline_aliases(model.main_rings[0])
        out["alternatives"] = list(out["alternatives"]) + [{"yaw": float(a)} for a in out["aliases"]]
    return out


def snap_alternative_poses(snap: dict[str, Any]) -> dict[str, Any]:
    """The box's 'every snap alternative within the ambiguity margin' (§2.4.8 part 7) as registration samples
    (pose offsets x, y, yaw from the accepted placement). The snap-accepted grade needs an UNAMBIGUOUS snap: a rival
    placement within the margin makes the snap ambiguous (stylize._rival) and the registration crowd grade, where
    D25-H / D25-B are off. So a snap-grade box has none by construction; a pose-format alternative a snap meta
    carries ({'pose': {'x', 'y', 'yaw'}, 'within_margin': True}) is sampled as given."""
    alts = [a for a in (snap.get("search") or {}).get("alternatives") or [] if a.get("within_margin") and a.get("pose")]
    out: dict[str, Any] = {"alternatives": [{k: float(v) for k, v in a["pose"].items() if k in ("x", "y", "yaw")}
                                            for a in alts]}
    if not alts:
        out["alternatives_note"] = ("no snap alternative within the ambiguity margin (an accepted unambiguous snap has "
                                    "none by construction)")
    return out


def make_box(meas: Measurer, pid: str, base_ch: dict[str, Any], pre: dict[str, Any], reg: dict[str, Any],
             local: dict[str, Any] | None, log=None) -> dict[str, Any]:
    """The evidence-grade box for one candidate: poses, and every agreeing channel measured again on each."""
    ev = meas.ev
    vp = ev.gravity_vp or {}
    two = vp.get("diff_deg") if vp.get("passed") else None
    use_local = bool(local and local.get("passed") and local.get("injection_test_passed"))
    sigma = C.box_sigma(ev.grade, snap_yaw_sigma_deg=reg.get("snap_yaw_sigma_deg"),
                        gps_residual_median_m=ev.gps_residual_median_m, reported_yaw_sigma_deg=ev.reported_yaw_sigma_deg,
                        ground_measured=bool((reg.get("ground") or {}).get("measured")),
                        estimate_up_boot_deg=(reg.get("estimate_up") or {}).get("sigma_deg"),
                        two_gravity_diff_deg=two, local_ground=use_local)
    poses = C.box_poses(sigma, alternatives=reg.get("alternatives") or None)
    samples = []
    t0 = time.perf_counter()
    for i, s in enumerate(poses):
        if s["kind"] == "base":
            ch = base_ch
        else:
            ch = meas.channels(pid, s["pose"], base=base_ch, counts=False)
        vals, eave = sample_values(ch, pre["verdicts"])
        samples.append({**s, "values": {c: vals.get(c) for c in pre["agreeing"]},
                        "eave": {c: v for c, v in eave.items() if v is not None},
                        "walls": (ch.get("walls") or {}).get("value")})
        if log:
            log(f"    box {pid} {i + 1}/{len(poses)} {s['kind']} {s['pose']}: "
                + ", ".join(f"{c} {v['value'] if v else None}" for c, v in samples[-1]["values"].items())
                + f", walls {samples[-1]['walls']}"
                + f" ({time.perf_counter() - t0:.0f}s)")
    return C._plain({"grade": ev.grade, "k": C.K_BOX, "sigma": sigma, "samples": samples,
                     "local_ground": local, "gravity_sampled": sigma.get("gravity_deg") is not None,
                     "inputs": box_inputs(ev, reg)})


def box_inputs(ev: Evidence25, reg: dict[str, Any]) -> dict[str, Any]:
    """What the box's sigma was sized from, recorded with it (crowd3d.correct.box_sigma_problem recomputes it)."""
    vp = ev.gravity_vp or {}
    return {"ground": reg.get("ground"), "estimate_up": reg.get("estimate_up"),
            "gravity_vp": vp if vp.get("passed") else None, "alternatives": reg.get("alternatives") or [],
            "aliases": reg.get("aliases") or []}


def injection_test(meas: Measurer, pid: str, base_ch: dict[str, Any], pre: dict[str, Any], LB: float, UB: float,
                   log=None) -> dict[str, Any]:
    """The local-ground route's test: whole-scene tilts of +-1 / +-3 deg about both horizontal axes; wherever the
    agreeing channels' value leaves [LB, UB], the local ground check must fail on the tilted points."""
    rows = []
    passed = True
    for g in (1.0, 3.0):
        for gx, gy in ((g, 0.0), (-g, 0.0), (0.0, g), (0.0, -g)):
            pose = {"x": 0.0, "y": 0.0, "yaw": 0.0, "gx": gx, "gy": gy}
            ch = meas.channels(pid, pose, base=base_ch, counts=False)
            vals, _ = sample_values(ch, pre["verdicts"])
            vs = [v["value"] for c, v in vals.items() if c in pre["agreeing"] and v]
            V = float(np.mean(vs)) if vs else None
            lg = local_ground(meas.model, meas.model.built_part(pid), meas.points(pose))
            outside = V is None or not (LB - 1e-6 <= V <= UB + 1e-6)
            ok = (not outside) or (not lg["passed"])
            passed &= ok
            rows.append({"gx": gx, "gy": gy, "V": None if V is None else round(V, 3), "outside": bool(outside),
                         "local_ground_passed": lg["passed"], "ok": bool(ok)})
            if log:
                log(f"    injection {pid} gx {gx:+g} gy {gy:+g}: V {V} outside {outside} local ground {lg['passed']}")
    return {"passed": bool(passed), "rows": rows, "interval": [LB, UB]}


def _height_part(model: S.StructureModel, pid: str) -> dict[str, Any]:
    b = model.built_part(pid)
    g = C._part_geometry(b.spec)
    return {"id": pid, "map_top": float(b.top), "z0": float(b.z0), "roof_h": float(b.roof.height_m),
            "pitched": g["pitched"], "min_width": g["min_width"]}


def decide_height(meas: Measurer, pid: str, base: dict[str, Any], part: dict[str, Any], pre: dict[str, Any],
                  scale: dict[str, Any], reg: dict[str, Any], log=None) -> dict[str, Any]:
    """The box for one D25-H candidate: the local-ground route first (gravity not sampled) when the ground next to the
    part passes its check AND the tilt-injection test against that box's [LB, UB]; otherwise gravity sampled per the
    table. Returns {'final', 'box', 'eave', ...}."""
    say = log or (lambda *a: None)
    dec: dict[str, Any] = {}
    probe = C.rule_h_box(pre, {"grade": meas.ev.grade, "samples": []}, scale, part)
    if not str(probe.get("why", "")).startswith("registration box"):
        dec["final"] = probe                       # the scale, V / k or the crowd switch refuse before any box
        return dec
    tb = time.perf_counter()
    P = meas.points(None)
    local = local_ground(meas.model, meas.model.built_part(pid), P) if len(P) else {"passed": False, "why": "no points"}
    box = None
    if local.get("passed"):
        box_l = make_box(meas, pid, base, pre, reg, dict(local, injection_test_passed=True), log=say)
        bb = C.box_bounds(box_l, pre["agreeing"], float(pre["V"]), float(scale["k_err"]))
        if "LB" in bb:
            local["injection"] = injection_test(meas, pid, base, pre, bb["LB"], bb["UB"], log=say)
            local["injection_test_passed"] = bool(local["injection"]["passed"])
        else:
            local["injection_test_passed"] = False
            local["injection"] = {"passed": False, "why": bb.get("why")}
        if local["injection_test_passed"]:
            box_l["local_ground"] = local
            box = box_l
    else:
        local["injection_test_passed"] = False
    if box is None:
        box = make_box(meas, pid, base, pre, reg, None, log=say)
        box["local_ground"] = local
    fin = C.rule_h_box(pre, box, scale, part)
    dec.update(final=fin, box=box, box_s=round(time.perf_counter() - tb, 1))
    if fin.get("fires") and fin.get("adopted") == "measured":
        dec["eave"] = C.rule_eave(base, part, pre["direction"], pre["V"], box, scale)
    return dec


def correct_landmark(model: S.StructureModel, ev: Evidence25, *, log=None, max_rounds: int = 4,
                     only: list[str] | None = None, apply: bool = True, targets: list[str] | None = None) -> dict[str, Any]:
    """D25 on one landmark after the judge (D15) ran: the preconditions, the scale anchor, rounds of D25-H (the
    disputed parts) and D25-B (the 'height pending' stubs), each candidate's box; corrections written (apply) and the
    parts they touch (deferred parts, neighbours, children) judged again on the new model, until nothing changes.
    targets: parts judged by D25-H whatever their D15 state (a survey, e.g. of a crowd model no D15 judge ran on; the
    rule and its switches are the same; a survey never writes, so targets need apply=False)."""
    if targets and apply:
        raise ValueError("targets is a survey of parts D15 did not dispute: it never applies corrections")
    say = log or (lambda *a: None)
    t0 = time.perf_counter()
    out: dict[str, Any] = {"rule": C.RULE, "grade": ev.grade, "rounds": [], "decisions": {}, "records": []}
    if ev.grade not in (C.GRADE_SNAP, C.GRADE_CROWD):
        out["why"] = "no evidence-grade registration (not georegistered, or no points): no correction"
        return out
    if any(t.get("kind") == "frame" and t.get("open", True) for t in model.tickets):
        out["why"] = "the frame ticket is open (G11 or a similarity in question): no correction"
        return out
    P = ev.track_points()
    meas = Measurer(ev, model)
    pairs = wall_pairs(P, None, model.main_rings[0]) if len(P) else []
    W = 2.0 * S.obb(S.ccw(model.main_rings[0])).W
    sim = C.scale_from_similarity(J.scale_status(model.frame, ev.snap_scale), W)
    scale = C.scale_anchor([C.scale_from_walls(pairs), sim])
    reg = registration_inputs(ev, model, P)
    out.update(scale=scale, registration_inputs=reg)
    say(f"D25: grade {ev.grade}; scale {scale.get('route')} k {scale.get('k')} ({scale.get('why')}); ground "
        f"{reg['ground']}; estimate_up boot {reg['estimate_up'].get('sigma_deg')}")
    corrected: set[str] = set()
    todo: set[str] | None = None                   # None: everything in round 1
    for r in range(1, max_rounds + 1):
        built = {b.id for b in model.built}
        heights = [p.id for p in model.specs if p.id in built and (p.params["height"].state == S.STATE_DISPUTED
                                                                   or (targets and p.id in targets
                                                                       and p.params["height"].source != st.PROV_DEFAULT))
                   and p.id not in corrected and (todo is None or p.id in todo) and (not only or p.id in only)]
        stubs = [p.id for p in model.specs if p.id in built and p.params["height"].source == st.PROV_DEFAULT
                 and p.id not in corrected and (todo is None or p.id in todo) and (not only or p.id in only)]
        if not heights and not stubs:
            break
        rt = time.perf_counter()
        meas = Measurer(ev, model, meas.pivot)
        adj = adjacency(model)
        base, pres = {}, {}
        for pid in heights:
            t1 = time.perf_counter()
            base[pid] = meas.channels(pid)
            part = _height_part(model, pid)
            pres[pid] = (part, C.rule_h_pre(part, base[pid]))
            say(f"  r{r} {pid} map {part['map_top']:.2f}: " + ", ".join(
                f"{c} {v.get('value')}{'' if v['valid'] else ' x'}" for c, v in pres[pid][1]["verdicts"].items())
                + f" -> {pres[pid][1]['why'][:160]} ({time.perf_counter() - t1:.0f}s)")
        # parents first; taller adjacent parts that may still go up (disputed upward, not refused for good) first
        open_up = {pid for pid, (_, pr) in pres.items() if pr.get("direction") == "up" or pr.get("candidate")}
        deferred: set[str] = set()
        for pid, (part, pr) in list(pres.items()):
            b = model.built_part(pid)
            par = b.spec.parent
            if par and model.part(par).params["height"].state in (S.STATE_DISPUTED, S.STATE_STUB):
                part["parent_ok"], part["parent_why"] = False, f"its parent {par} is {model.part(par).params['height'].state}"
            pend = [q for q in adj.get(pid, []) if q in open_up and q != par and model.built_part(q).top > b.top]
            if pend:
                part["pending"] = pend
            if not part.get("parent_ok", True) or pend:
                pres[pid] = (part, C.rule_h_pre(part, base[pid]))
                if pres[pid][1].get("deferred"):
                    deferred.add(pid)
        fired: list[str] = []
        for pid, (part, pr) in pres.items():
            dec: dict[str, Any] = {"pre": pr, "round": r, "part": part}
            if pr.get("candidate"):
                dec.update(decide_height(meas, pid, base[pid], part, pr, scale, reg, log=say))
                say(f"  r{r} {pid}: {dec['final'].get('why')}")
                if dec["final"].get("fires"):
                    fired.append(pid)
            else:
                dec["final"] = {"fires": False, "why": pr.get("why")}
            out["decisions"][pid] = dec
        changed: set[str] = set()
        if apply:
            for pid in fired:
                dec = out["decisions"][pid]
                part, pr = pres[pid]
                recs = C.apply_height(model, pid, pr, dec["final"], base[pid], dec["box"], scale, eave=dec.get("eave"),
                                      round_=r, snap_scale=ev.snap_scale, rule_part=part)
                dec["records"] = [x["id"] for x in recs]
                out["records"] += dec["records"]
                corrected.add(pid)
                changed.add(pid)
        # blocks after this round's heights, measured on the current model
        meas = Measurer(ev, model, meas.pivot)
        adj = adjacency(model)
        for pid in stubs:
            b = model.built_part(pid)
            pend = [q for q in adj.get(pid, []) if q in changed or q in deferred
                    or (q in open_up and q not in corrected and q not in out["decisions"])]
            part = {"id": pid, "min_width": float(b.spec.min_width()), "z0": float(b.z0), "map_top": float(b.top),
                    **({"pending": pend} if pend else {})}
            dec = {"round": r, "part": part}
            if pend:
                dec["final"] = C.rule_b(part, {}, None, None, scale)
                dec["final"]["why"] = f"taller adjacent parts {pend} are corrected first: re-judged after them"
                deferred.add(pid)
                out["decisions"][pid] = dec
                say(f"  r{r} block {pid}: {dec['final']['why']}")
                continue
            res = block_readings(meas, pid)
            carve = carve_upper_bound(b, ev.cameras, ev.masks) if ev.masks else {"p25": None}
            valid = [c for c, v in res.items() if C.block_reading_problem(c, v) is None]
            box = None
            if len(valid) >= C.BLOCK_MIN_READINGS and C.LINES in valid and scale.get("established") and \
                    (ev.grade == C.GRADE_SNAP or C.CROWD_HB_ENABLED):
                box = make_block_box(meas, pid, res, reg, log=say)
            roof = block_roof(meas.model, b, meas.points(None))
            fin = C.rule_b(part, res, carve.get("p25"), box if box is not None else
                           {"grade": ev.grade, "k": C.K_BOX, "sigma": {}, "samples": []}, scale, roof)
            dec.update(readings=res, carve=carve, final=fin, box=box, roof=roof)
            say(f"  r{r} block {pid}: readings {fin.get('readings')} minima {fin.get('box_minima')} carve "
                f"{carve.get('p25')} interior roof {len(roof['roof_tops'])}/{roof['cells_footprint']} cells -> "
                f"{fin.get('why')}")
            out["decisions"][pid] = dec
            if fin.get("fires") and apply:
                rec = C.apply_block(model, pid, fin, res, box, scale, round_=r, snap_scale=ev.snap_scale,
                                    depends_on=sorted(changed & set(adj.get(pid, []))))
                dec["records"] = [rec["id"]]
                out["records"].append(rec["id"])
                corrected.add(pid)
                changed.add(pid)
        out["rounds"].append({"round": r, "heights": heights, "stubs": stubs, "changed": sorted(changed),
                              "deferred": sorted(deferred), "seconds": round(time.perf_counter() - rt, 1)})
        if not changed:
            break
        # judged again: what waited for a parent or a neighbour, and every part touching or resting on a change
        adj = adjacency(model)
        todo = set(deferred)
        for q in changed:
            todo |= set(adj.get(q, [])) | set(S.descendants(model, q))
        todo -= corrected
    if apply and out["records"]:
        gate = g5_gate(model, out["records"], ev.cameras, ev.masks)
        out["g5"] = gate["g5"]
        say(f"D25 G5: {gate['g5'].get('why')} parts "
            f"{({k: v.get('why') for k, v in (gate['g5'].get('parts') or {}).items()})}")
        if gate["reverted"]:
            out["records_reverted_by_g5"] = gate["reverted"]
        out["records"] = gate["records"]
    out["total_s"] = round(time.perf_counter() - t0, 1)
    return out


def g5_gate(model: S.StructureModel, records: list[str], cameras: list, masks: dict[str, dict[int, np.ndarray]],
            rounds: int = 3) -> dict[str, Any]:
    """G5 per corrected part against the model before this run's D25 corrections (D15 kept); a part that regresses
    keeps its old geometry (§2.8: its records reverted, with what depends on them), the rest are measured again
    until nothing more is reverted. Returns {'g5', 'records' (kept, with their part's G5 attached), 'reverted'}."""
    tg = time.perf_counter()
    reverted: list[str] = []
    g5: dict[str, Any] = {}

    def live() -> list[str]:
        return [r for r in records if any(a.get("id") == r and a.get("active", True) for a in model.adoptions)]

    for _ in range(rounds):
        cur = live()
        if not cur:
            break
        parts = sorted({a["part"] for a in model.adoptions if a.get("id") in set(cur)})
        if masks and len(masks) >= C.G5_MIN_MASKS:
            act = {a["id"]: a for a in model.adoptions if a.get("active", True)}
            sky = {a["part"] for a in act.values() if a.get("id") in set(cur) and C.skyline_seen(a, act)}
            g5 = g5_measure(before_d25(model, cur), model, cameras, masks, parts=parts, skyline_parts=sky)
        else:
            why = C.g5_problem({"masks": {}})
            g5 = {"baseline": G5_BASELINE, "masks": {}, "passed": False, "why": why,
                  "parts": {pid: {"masks": {}, "passed": False, "why": why} for pid in parts}}
        bad = {pid for pid, pg in (g5.get("parts") or {}).items() if not pg.get("passed")}
        if not bad:
            break
        for rid in reversed(cur):                      # 'not met: keep the old geometry' (§2.8)
            a = next(x for x in model.adoptions if x.get("id") == rid)
            if a["part"] in bad and a.get("active", True):
                reverted += [x for x in J.revert_adoption(model, rid) if x not in reverted]
                if rid not in reverted:
                    reverted.append(rid)
    kept = live()
    g5["seconds"] = round(time.perf_counter() - tg, 1)
    g5["reverted"] = list(reverted)
    if kept:
        C.attach_g5(model, kept, g5)
    return {"g5": g5, "records": kept, "reverted": [r for r in records if r not in set(kept)]}


def _sky_problems(model: S.StructureModel, cameras: list, masks: dict[str, dict[int, np.ndarray]]) -> dict[str, Any]:
    return {name: K.SkylineProblem(model, cameras, masks[name], max_cameras_per_part=10 ** 6) for name in sorted(masks)}


def skyline_distances(model: S.StructureModel, cameras: list, masks: dict[str, dict[int, np.ndarray]],
                      sps: dict[str, Any] | None = None) -> dict[str, dict[int, float]]:
    """Per sky mask and photo, the model's mean forward skyline distance (px, <= 40 truncated; crowd3d.skyline.score):
    G5's consistency number (never 'likeness')."""
    sps = sps or _sky_problems(model, cameras, masks)
    out: dict[str, dict[int, float]] = {}
    for name in sorted(masks):
        sp = sps[name]
        row = {}
        for c, cd in enumerate(sp.cds):
            r = K.score(cd, sp.union[c])
            if r["fwd_mean"] is not None:
                row[int(cd.index)] = float(r["fwd_mean"])
        out[name] = row
    return out


def _fwd_cols(cd: Any, r_m: np.ndarray, cols: np.ndarray) -> float | None:
    has = (r_m < cd.H) & (r_m > 0) & cols
    if not has.any():
        return None
    rr = np.clip(r_m, 0, cd.H - 1).astype(int)
    return float(cd.D2[rr[has], np.nonzero(has)[0]].mean())


def part_skyline_distances(sp_a: Any, sp_b: Any, pid: str) -> dict[int, tuple[float, float]]:
    """Per photo, the mean forward skyline distance of two versions of a model over the band columns where the part is
    the model's skyline in either version (>= K.MIN_SKY_COLS columns, both versions with a model edge there)."""
    out: dict[int, tuple[float, float]] = {}
    if pid not in sp_a.tops or pid not in sp_b.tops:
        return out
    ia = {int(cd.index): c for c, cd in enumerate(sp_a.cds)}
    for cb, cdb in enumerate(sp_b.cds):
        ca = ia.get(int(cdb.index))
        if ca is None:
            continue
        cda = sp_a.cds[ca]
        ta, ua = sp_a.tops[pid][ca], sp_a.union[ca]
        tb, ub = sp_b.tops[pid][cb], sp_b.union[cb]
        cols = (((ta <= ua + 1) & (ta < cda.H)) | ((tb <= ub + 1) & (tb < cdb.H))) & cda.roi_cols & cdb.roi_cols
        if int(cols.sum()) < K.MIN_SKY_COLS:
            continue
        fa, fb = _fwd_cols(cda, ua, cols), _fwd_cols(cdb, ub, cols)
        if fa is not None and fb is not None:
            out[int(cdb.index)] = (fa, fb)
    return out


G5_BASELINE = "the model before this run's D25 corrections (D15's adoptions kept)"


def g5_measure(prior: S.StructureModel, corrected: S.StructureModel, cameras: list,
               masks: dict[str, dict[int, np.ndarray]], parts: list[str] | None = None,
               skyline_parts: set[str] | None = None) -> dict[str, Any]:
    """G5 on the same photos: per corrected part (crowd3d.correct.g5_problem decides each), per mask, the mean forward
    skyline distance of the baseline and of the corrected version over the photos where the part is the model's
    skyline in either version (its columns only); and, for reference, the whole model's over the photos where both
    versions have a model edge."""
    spa, spb = _sky_problems(prior, cameras, masks), _sky_problems(corrected, cameras, masks)
    a, b = skyline_distances(prior, cameras, masks, spa), skyline_distances(corrected, cameras, masks, spb)
    out: dict[str, Any] = {"baseline": G5_BASELINE, "masks": {}, "parts": {}}
    for name in sorted(masks):
        common = sorted(set(a.get(name, {})) & set(b.get(name, {})))
        out["masks"][name] = {"photos": len(common),
                              "prior_px": round(float(np.mean([a[name][i] for i in common])), 3) if common else None,
                              "corrected_px": round(float(np.mean([b[name][i] for i in common])), 3) if common else None}
    for pid in parts or []:
        pm = {}
        for name in sorted(masks):
            d = part_skyline_distances(spa[name], spb[name], pid)
            pm[name] = {"photos": len(d),
                        "prior_px": round(float(np.mean([v[0] for v in d.values()])), 3) if d else None,
                        "corrected_px": round(float(np.mean([v[1] for v in d.values()])), 3) if d else None}
        g = {"scope": C.G5_SCOPE, "part": pid, "masks": pm}
        why = C.g5_problem(g, pid, pid in (skyline_parts or set()))
        out["parts"][pid] = {"masks": pm, "passed": why is None,
                             "why": why or "no worse than the baseline (+0.5 px) on the part's own skyline columns in "
                                           "both masks"}
    bad = [p for p, g in out["parts"].items() if not g["passed"]]
    out.update(passed=not bad, why=(f"G5 fails for {bad}" if bad else "every corrected part no worse than the baseline "
                                                                      "(+0.5 px) in both masks"))
    return out


def before_d25(model: S.StructureModel, rids: list[str]) -> S.StructureModel:
    """G5's baseline: a copy with this run's D25 corrections reverted (D15's adoptions kept: verify-r2 -- a D15
    frame improvement in the baseline would mask a D25 regression)."""
    import copy as _copy
    base = _copy.deepcopy(model)
    for rid in reversed(list(rids)):
        if any(a.get("id") == rid and a.get("active", True) for a in base.adoptions):
            J.revert_adoption(base, rid)
    return base


def map_prior(model: S.StructureModel) -> S.StructureModel:
    """The map prior: a copy with every active adoption and correction reverted (equal to the fresh OSM model)."""
    import copy as _copy
    prior = _copy.deepcopy(model)
    for a in reversed([a for a in prior.adoptions if a.get("active", True)]):
        if a.get("active", True):
            J.revert_adoption(prior, a["id"])
    return prior


def block_readings(meas: Measurer, pid: str, pose=None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    b = meas.model.built_part(pid)
    P = meas.points(pose)
    out: dict[str, Any] = {}
    lines = transform_lines(meas.ev.lines, pose, meas.pivot) if meas.ev.lines is not None and not callable(meas.ev.lines) else None
    out[C.LINES] = block_lines(lines, b)
    out[C.SFM] = block_sfm(meas.model, b, P) if len(P) else {"valid": False, "why": "no points"}
    if meas.ev.sky_eval is not None:
        st_ = None if base is None else _sky_start((base.get(C.SKYLINE) or {}).get("raw"))
        raw = skyline_channel(meas.frame(pose), b, st_, meas.ev.max_cams, counts=base is None)
        sk = block_skyline(raw, need_counts=base is None)
        sk["raw"] = {"masks": raw.get("masks"), "n_vis_map": raw.get("n_vis_map")}
        out[C.SKYLINE] = sk
    return out


def make_block_box(meas: Measurer, pid: str, base: dict[str, Any], reg: dict[str, Any], log=None) -> dict[str, Any]:
    ev = meas.ev
    vp = ev.gravity_vp or {}
    sigma = C.box_sigma(ev.grade, snap_yaw_sigma_deg=reg.get("snap_yaw_sigma_deg"),
                        gps_residual_median_m=ev.gps_residual_median_m, reported_yaw_sigma_deg=ev.reported_yaw_sigma_deg,
                        ground_measured=bool((reg.get("ground") or {}).get("measured")),
                        estimate_up_boot_deg=(reg.get("estimate_up") or {}).get("sigma_deg"),
                        two_gravity_diff_deg=vp.get("diff_deg") if vp.get("passed") else None)
    samples = []
    t0 = time.perf_counter()
    valid = [c for c, v in base.items() if C.block_reading_problem(c, v) is None]
    for i, s in enumerate(C.box_poses(sigma, alternatives=reg.get("alternatives") or None)):
        rd = base if s["kind"] == "base" else block_readings(meas, pid, s["pose"], base)
        samples.append({**s, "values": {c: ({"value": rd[c]["value"], "half": 0.0} if rd.get(c, {}).get("valid") else None)
                                        for c in valid}})
        if log:
            log(f"    block box {pid} {i + 1} {s['kind']}: " + ", ".join(f"{c} {(v or {}).get('value')}" for c, v in samples[-1]["values"].items())
                + f" ({time.perf_counter() - t0:.0f}s)")
    return C._plain({"grade": ev.grade, "k": C.K_BOX, "sigma": sigma, "samples": samples, "local_ground": None,
                     "gravity_sampled": True, "inputs": box_inputs(ev, reg)})
