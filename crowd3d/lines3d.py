"""3-D line evidence for structure parts (docs/STRUCTURE_FIRST.md §1, §2.4.2): the "structure half" of Hover's stack.

Hover's per-photo networks did not recover landmark roofs in the hover-models probe; multi-view 3-D line
triangulation did (Baptistery: 7/8 eaves, 6/8 hips, apex 0.18 m / 0.54 m off its SfM-fitted reference). This is
that numpy "LIMAP-lite", ported from the probe, plus what the design asks of it: per OSM part, the 3-D lines are
assigned to the edges the part's map geometry predicts and turned into evidence records in the judge's format.
Lines are EVIDENCE ONLY: they never become geometry, never colour anything, and in v0 never change a map value
(crowd3d.judge decides: D15 adopts a value only when another channel agrees with it).

Pipeline (lines_channel):
 1. 2-D segments per photo: OpenCV FLD (cv2.ximgproc, opencv-contrib) when present, else core OpenCV's LSD (the
    design's fallback); endpoints undistorted to the pinhole model; only segments inside the landmark's projected
    volume; the longest max_segments per photo; optional per-photo cache (a photo is processed once).
 2. Photos per part: SfM points of the part's prism seen by each photo (coarse z-buffer); at most
    max_cameras_per_part photos per part, the best-viewing ones spread over 8 bearing sectors (CPU bound: the
    matching cost grows with the photos), at most max_cameras in all.
 3. Matching: each photo i against its neighbours j (most shared SfM points of the parts it was chosen for, median
    triangulation angle >= 3 deg): a segment pair's back-projected planes intersect in the 3-D candidate (planes >= 2
    deg apart, >= 50 % overlap in j, in front of both cameras, inside the volume, and when >= 3 SfM points lie
    within 3 px of the segment, at their depth (q10 / 1.15 .. q90 * 1.15): the SfM depth prior);
 4. verification: a candidate needs >= min_support further photos (besides i and j) with a segment within 2.5 px
    and 2 deg of its projection (>= 40 % overlap);
 5. merging (collinear within 2.5 deg, max(0.25 m, 0.4 % of the depth), gaps < 0.5 m) and filtering: seen in >=
    keep_min_images photos and >= 50 % of the line within 0.4 m of an SfM point (track >= 3): "S+V", the variant the
    design uses (the Baptistery's 5/8 eaves and 5/8 hips).

Measuring (measure_part), per built part at its MAP geometry:
 - eave: per outline edge whose wall top is the eave, the horizontal lines parallel to it, inside a window around
   it (frame errors: up to min(6 m, a quarter of the part's width) inside, 3 m outside), on the wall plane they
   themselves define; the edge's eave is the highest height cluster on that plane (>= 15 % of the edge covered, >= 3
   photos). Edges without such lines are listed as unobserved: nothing is said about them. With SfM points, an edge
   is not measured (ambiguous) when the wall's own points rise above that cluster (a string course, not the top) or
   when the cluster lies more than the eave's tolerance above the wall's own surface -- where its points are densest
   (wall_surface_top()) -- (scaffolding just outside, a structure in front: not this wall's top), when that surface
   lies more than 1 m outside the edge (a scaffold net or hoarding is itself the densest surface: verify-r2), or when
   the cluster's plane is not on that surface (> 0.6 m off). A net closer than 1 m to the wall is not told apart
   from the wall by geometry. Eave edges that disagree by > max(1.5 m, 7.5 %) withhold the vote.
 - apex (pointed roofs: pyramidal, cone, hipped to a point): RANSAC over pairs of sloped lines rising toward the
   map's apex, IRLS refinement; needs >= 4 lines from >= 3 photos coming from two directions >= 45 deg apart, within
   max(2 m, 15 % of the width) of the map's apex in plan. Its position is reported (a frame question), its height
   votes. The hip pitch of its lines is reported beside the map's.
 - ridge (gabled, hipped with a ridge, ...): the highest cluster of horizontal lines along the map's ridge.
Each measurement has a 90 % interval (200 bootstrap resamples of its lines), n_photos, and a vote: 'up' / 'down'
when it is off the map by more than max(2 m, 10 %) and its interval excludes the map value (lines measure heights
directly, so they vote both ways), else None. Fewer than 3 photos: refused. Every record carries the snap scale; a
snap scale correction > 3 % marks it scale_corrected (it cannot decide alone; the judge applies that). Each record
states the height window it searched (search_m) and whether its value sits at the window's edge (at_bound: a value
at a fit bound, never adopted by D15).

Pure numpy + OpenCV (no scipy in the repo): nearest-neighbour work runs on sorted integer cell keys.
Coordinates: the landmark ENU frame of the structure (stylize.SceneInput after the snap).
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import structure as S
from . import stylize as st

KIND = "lines3d"
SNAP_SCALE_TOL = 0.03


@dataclass(frozen=True)
class LinesParams:
    detector: str = "auto"               # 'auto' (FLD if opencv-contrib is installed, else LSD) | 'fld' | 'lsd'
    min_len_px: float = 30.0
    max_segments: int = 600              # longest segments per photo
    max_cameras_per_part: int = 40       # CPU bound (design §5.1): best-viewing photos per part
    max_cameras: int = 200               # ... and in all
    min_part_points: int = 20            # a photo "sees" a part with at least this many of its SfM points
    covis_points: int = 40000            # SfM points (subsample) for covisibility and photo choice
    prior_points: int = 150000           # SfM points (subsample) for the depth prior
    neighbours: int = 8
    verify_pool: int = 10
    min_shared: int = 20
    min_tri_angle_deg: float = 3.0
    plane_sep_deg: float = 2.0
    pair_overlap: float = 0.5
    prior_px: float = 3.0
    prior_slack: float = 1.15
    top_per_segment: int = 3
    verify_px: float = 2.5
    verify_deg: float = 2.0
    verify_overlap: float = 0.4
    min_support: int = 2                 # verifying photos besides the matched pair
    merge_deg: float = 2.5
    merge_tol_m: float = 0.25
    merge_tol_rel: float = 0.004
    merge_gap_m: float = 0.5
    merge_cell_m: float = 0.6
    min_images: int = 4                  # a merged line is kept when seen in this many photos ...
    keep_min_images: int = 6             # ... and used when seen in this many (S+V)
    support_radius_m: float = 0.4
    support_step_m: float = 0.25
    support_min: float = 0.5
    support_min_track: int = 3
    volume_margin_m: float = 8.0
    cap_factor: float = 1.4              # the volume reaches this x the tallest part
    threads: int = 2
    # measuring
    min_photos: int = 3
    horiz_deg: float = 5.0
    parallel_deg: float = 7.0
    wall_in_m: float = 6.0
    wall_in_share: float = 0.25
    wall_out_m: float = 3.0
    on_wall_m: float = 0.6
    wall_surface_out_m: float = 1.0      # the wall's own (densest) surface no farther outside the outline edge than this
    cluster_gap_m: float = 0.4
    cluster_min_len_m: float = 1.0
    cluster_min_share: float = 0.15
    slope_min_deg: float = 15.0
    slope_max_deg: float = 80.0
    sloped_min_len_m: float = 1.5
    apex_line_m: float = 0.5
    apex_end_m: float = 4.0
    apex_min_lines: int = 4
    apex_max_off_m: float = 2.0
    apex_max_off_share: float = 0.15
    apex_min_spread_deg: float = 45.0
    apex_ransac: int = 3000
    boot: int = 200


DEFAULT_LINES = LinesParams()


# ---------------------------------------------------------------- cameras


def _pinhole(cam: st.PosedCamera, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ENU -> undistorted pixel coordinates (segments are undistorted at detection) and depth."""
    pc = (np.asarray(X, float) - cam.center) @ cam.rot.T
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[..., 0] / z + cam.cx
        v = cam.fy * pc[..., 1] / z + cam.cy
    return np.stack([u, v], -1), z


def undistort_px(cam: st.PosedCamera, uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, float).reshape(-1, 2)
    if not (cam.k1 or cam.k2):
        return uv
    x, y = (uv[:, 0] - cam.cx) / cam.fx, (uv[:, 1] - cam.cy) / cam.fy
    xu, yu = x.copy(), y.copy()
    for _ in range(10):
        r2 = xu * xu + yu * yu
        f = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2
        xu, yu = x / f, y / f
    return np.stack([cam.fx * xu + cam.cx, cam.fy * yu + cam.cy], 1)


def _seg_planes(cam: st.PosedCamera, segs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World normals of the segments' back-projected planes and the unit rays of their endpoints."""
    def norm_coords(p):
        return np.stack([(p[:, 0] - cam.cx) / cam.fx, (p[:, 1] - cam.cy) / cam.fy, np.ones(len(p))], 1)
    p1, p2 = norm_coords(segs[:, :2]), norm_coords(segs[:, 2:])
    n = np.cross(p1, p2) @ cam.rot                    # R^T (p1 x p2)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    r1, r2 = p1 @ cam.rot, p2 @ cam.rot
    return n, r1 / np.linalg.norm(r1, axis=1, keepdims=True), r2 / np.linalg.norm(r2, axis=1, keepdims=True)


def visible_points(cam: st.PosedCamera, P: np.ndarray, cell: int = 6, tol: float = 0.06) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of points in front of the camera, in frame and not behind nearer points (coarse z-buffer: a point is
    kept when its depth is within `tol` of the nearest point in its cell)."""
    uv, z = _pinhole(cam, P)
    ok = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] < cam.height)
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return idx, uv, z
    gx = (uv[idx, 0] // cell).astype(np.int64)
    gy = (uv[idx, 1] // cell).astype(np.int64)
    key = gy * (cam.width // cell + 2) + gx
    zmin = np.full(int(key.max()) + 1, np.inf)
    np.minimum.at(zmin, key, z[idx])
    return idx[z[idx] <= zmin[key] * (1 + tol)], uv, z


# ---------------------------------------------------------------- 2-D segments


def detector_name(detector: str = "auto") -> str:
    if detector == "lsd":
        return "lsd"
    if hasattr(cv2, "ximgproc"):
        return "fld"
    if detector == "fld":
        raise RuntimeError("FLD needs the opencv-contrib build (cv2.ximgproc); use detector='lsd' or 'auto'")
    return "lsd"


def detect_segments(gray: np.ndarray, detector: str = "auto", min_len_px: float = 30.0) -> tuple[np.ndarray, str]:
    """(n, 4) float32 x1 y1 x2 y2 segments of at least min_len_px, and the detector used."""
    name = detector_name(detector)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    if name == "fld":
        fld = cv2.ximgproc.createFastLineDetector(10, 1.414, 50, 50, 3, False)
        s = fld.detect(gray)
    else:
        s = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    s = np.zeros((0, 4), np.float32) if s is None else np.asarray(s, np.float32).reshape(-1, 4)
    L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    return s[L >= min_len_px], name


def _cache_file(cache_dir: Path, cam: st.PosedCamera, name: str) -> Path:
    """Per photo (its id: the same photo in another model's job hits the cache; a MegaDepth camera's id is its name)."""
    key = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{cam.photo_id or cam.name}-{cam.width}x{cam.height}")
    return cache_dir / f"{name}-{key}.npy"


def photo_segments(cams: list[st.PosedCamera], ids: list[int], params: LinesParams = DEFAULT_LINES,
                   cache_dir: str | Path | None = None) -> tuple[dict[int, np.ndarray], str, float]:
    """Raw segments (detector pixels) per camera index, detected on `params.threads` threads; cached per photo."""
    name = detector_name(params.detector)
    cdir = Path(cache_dir) if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def one(i: int) -> tuple[int, np.ndarray]:
        cam = cams[i]
        if cdir:
            f = _cache_file(cdir, cam, name)
            if f.exists():
                return i, np.load(f)
        s, _ = detect_segments(cam.load(), name, params.min_len_px)
        if cdir:
            np.save(_cache_file(cdir, cam, name), s)
        return i, s

    with ThreadPoolExecutor(max_workers=max(1, params.threads)) as ex:
        out = dict(ex.map(one, ids))
    return out, name, time.perf_counter() - t0


# ---------------------------------------------------------------- volume, photos per part


def _volume(model: S.StructureModel, params: LinesParams) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    pts = np.vstack([b.outer for b in model.built])
    lo, hi = pts.min(0) - params.volume_margin_m, pts.max(0) + params.volume_margin_m
    zmax = params.cap_factor * max(b.top for b in model.built)
    ring = S.ccw(S.convex_hull(np.vstack([pts + params.volume_margin_m * np.array([math.cos(a), math.sin(a)])
                                          for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)])))
    return lo, hi, -3.0, zmax, ring


def _in_volume(X: np.ndarray, vol) -> np.ndarray:
    lo, hi, z0, z1 = vol[:4]
    return (X[..., 0] > lo[0]) & (X[..., 0] < hi[0]) & (X[..., 1] > lo[1]) & (X[..., 1] < hi[1]) & (X[..., 2] > z0) & (X[..., 2] < z1)


def _part_members(model: S.StructureModel, P: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for b in model.built:
        inside = st.points_in_polygon(P[:, :2], b.outer)
        for h in b.holes:
            inside &= ~st.points_in_polygon(P[:, :2], h)
        zhi = b.top + max(3.0, 0.6 * (b.top - b.z0))
        out[b.id] = inside & (P[:, 2] >= b.z0 - 0.5) & (P[:, 2] <= zhi)
    return out


def select_cameras(model: S.StructureModel, cams: list[st.PosedCamera], V: np.ndarray, members: dict[str, np.ndarray],
                   params: LinesParams = DEFAULT_LINES) -> tuple[dict[str, list[int]], dict[str, np.ndarray]]:
    """Per built part, at most max_cameras_per_part photos seeing >= min_part_points of its SfM points: the best ones
    (most points) taken round-robin over 8 bearing sectors about the part."""
    sel, scores = {}, {}
    for b in model.built:
        m = members[b.id]
        sc = V[:, m].sum(1) if m.any() else np.zeros(len(cams), int)
        scores[b.id] = sc
        cand = [int(c) for c in np.nonzero(sc >= params.min_part_points)[0]]
        if len(cand) > params.max_cameras_per_part:
            cen = S.poly_centroid(b.outer, b.holes)
            bins: dict[int, list[int]] = {}
            for c in sorted(cand, key=lambda c: (-sc[c], c)):
                d = cams[c].center[:2] - cen
                bins.setdefault(int(((math.degrees(math.atan2(d[0], d[1])) % 360) + 22.5) % 360 // 45), []).append(c)
            pick = []
            while len(pick) < params.max_cameras_per_part and any(bins.values()):
                for k in sorted(bins):
                    if bins[k] and len(pick) < params.max_cameras_per_part:
                        pick.append(bins[k].pop(0))
            cand = pick
        sel[b.id] = sorted(cand)
    return sel, scores


# ---------------------------------------------------------------- matching


def _depth_image(cam: st.PosedCamera, P: np.ndarray) -> np.ndarray:
    vis, uv, z = visible_points(cam, P)
    D = np.full((cam.height, cam.width), np.inf, np.float32)
    if len(vis):
        u = np.clip(uv[vis, 0].astype(int), 0, cam.width - 1)
        v = np.clip(uv[vis, 1].astype(int), 0, cam.height - 1)
        np.minimum.at(D, (v, u), z[vis].astype(np.float32))
    return D


def seg_depth_prior(D: np.ndarray, segs: np.ndarray, px: float = 3.0, min_points: int = 3) -> np.ndarray:
    """q10, q90 of the SfM depths (per-pixel nearest, D) within px of each segment (NaN with < min_points)."""
    out = np.full((len(segs), 2), np.nan)
    H, W = D.shape
    offs = np.arange(-int(px), int(px) + 1, dtype=float)
    for k, s in enumerate(segs):
        p, q = s[:2], s[2:]
        L = float(np.hypot(*(q - p)))
        if L < 1:
            continue
        t = np.linspace(0, 1, int(L) + 1)
        nrm = np.array([-(q - p)[1], (q - p)[0]]) / L
        base = p[None] + t[:, None] * (q - p)[None]
        xy = (base[:, None, :] + offs[None, :, None] * nrm[None, None, :]).reshape(-1, 2)
        u, v = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
        ok = (u >= 0) & (v >= 0) & (u < W) & (v < H)
        d = D[v[ok], u[ok]]
        d = d[np.isfinite(d)]
        if len(d) >= min_points:
            out[k] = np.percentile(d, [10, 90])
    return out


def support_in_view(cam: st.PosedCamera, segs_v: np.ndarray, A: np.ndarray, B: np.ndarray, tol_px: float = 2.5,
                    tol_deg: float = 2.0, min_ov: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Candidate 3-D segments A-B (m, 3) supported in this photo: a segment within tol_px of the projected line at both
    ends, within tol_deg, overlapping >= min_ov. Returns (supported, best segment index)."""
    m = len(A)
    out, best = np.zeros(m, bool), np.full(m, -1)
    if not len(segs_v) or not m:
        return out, best
    pa, za = _pinhole(cam, A)
    pb, zb = _pinhole(cam, B)
    ok = (za > 0.5) & (zb > 0.5)
    inside = ok & (np.maximum(pa[:, 0], pb[:, 0]) > 0) & (np.minimum(pa[:, 0], pb[:, 0]) < cam.width) & \
        (np.maximum(pa[:, 1], pb[:, 1]) > 0) & (np.minimum(pa[:, 1], pb[:, 1]) < cam.height)
    idx = np.nonzero(inside)[0]
    if not len(idx):
        return out, best
    pa, pb = pa[idx], pb[idx]
    d = pb - pa
    L = np.linalg.norm(d, axis=1)
    good = L > 10
    idx, pa, pb, d, L = idx[good], pa[good], pb[good], d[good], L[good]
    if not len(idx):
        return out, best
    u = d / L[:, None]
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    q1, q2 = segs_v[:, :2], segs_v[:, 2:]
    dq = q2 - q1
    Lq = np.linalg.norm(dq, axis=1)
    uq = dq / np.maximum(Lq, 1e-9)[:, None]
    cos = np.abs(u @ uq.T)
    base_n = np.einsum("mk,mk->m", nrm, pa)[:, None]
    e1 = np.abs(nrm @ q1.T - base_n)
    e2 = np.abs(nrm @ q2.T - base_n)
    base_u = np.einsum("mk,mk->m", u, pa)[:, None]
    a1, a2 = u @ q1.T - base_u, u @ q2.T - base_u
    lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)
    ov = np.clip(np.minimum(hi, L[:, None]) - np.maximum(lo, 0), 0, None) / np.minimum(L[:, None], Lq[None])
    okm = (cos > math.cos(math.radians(tol_deg))) & (e1 < tol_px) & (e2 < tol_px) & (ov >= min_ov)
    anyok = okm.any(axis=1)
    out[idx] = anyok
    best[idx] = np.where(anyok, np.where(okm, e1 + e2, np.inf).argmin(axis=1), -1)
    return out, best


@dataclass
class RawLines:
    A: np.ndarray
    B: np.ndarray
    seed: np.ndarray                     # the photo the segment came from
    obs: list                            # per line: camera indices that see it (seed, partner, verifiers)


def triangulate(cams: list[st.PosedCamera], segs: dict[int, np.ndarray], used: list[int], shared: np.ndarray,
                angle_of, P_prior: np.ndarray, vol, params: LinesParams = DEFAULT_LINES,
                log=None) -> tuple[RawLines, dict[str, Any]]:
    """Epipolar matching with the SfM depth prior and verification in further photos (the probe's run loop)."""
    stats = {"pairs": 0, "cands_epi": 0, "cands_kept": 0, "verified": 0}
    As, Bs, seeds, obs = [], [], [], []
    planes = {i: _seg_planes(cams[i], segs[i]) for i in used if len(segs.get(i, ()))}
    pos = {c: k for k, c in enumerate(used)}
    t0 = time.perf_counter()
    for n_i, i in enumerate(used):
        ci = cams[i]
        Si = segs.get(i, np.zeros((0, 4)))
        if not len(Si):
            continue
        ni, r1, r2 = planes[i]
        prior = seg_depth_prior(_depth_image(ci, P_prior), Si, params.prior_px)
        row = shared[pos[i]]
        order = [used[k] for k in np.argsort(-row, kind="stable") if used[k] != i and row[k] >= params.min_shared]
        nbr = [j for j in order if angle_of(i, j) >= params.min_tri_angle_deg][:params.neighbours]
        pool = order[:params.neighbours + params.verify_pool]
        best: dict[int, tuple[int, np.ndarray, np.ndarray, list]] = {}
        for j in nbr:
            cj = cams[j]
            Sj = segs.get(j, np.zeros((0, 4)))
            if not len(Sj):
                continue
            nj = planes[j][0]
            stats["pairs"] += 1
            num = nj @ (cj.center - ci.center)
            with np.errstate(divide="ignore", invalid="ignore"):
                l1 = num[None] / (r1 @ nj.T)
                l2 = num[None] / (r2 @ nj.T)
            sep = np.abs(ni @ nj.T) < math.cos(math.radians(params.plane_sep_deg))
            ok = sep & (l1 > 1) & (l2 > 1) & (l1 < 2000) & (l2 < 2000)
            si, ti = np.nonzero(ok)
            if not len(si):
                continue
            A = ci.center + l1[si, ti, None] * r1[si]
            B = ci.center + l2[si, ti, None] * r2[si]
            inside = _in_volume(A, vol) & _in_volume(B, vol)
            si, ti, A, B = si[inside], ti[inside], A[inside], B[inside]
            if not len(si):
                continue
            pa, za = _pinhole(cj, A)
            pb, zb = _pinhole(cj, B)
            t1_, t2_ = Sj[ti, :2], Sj[ti, 2:]
            dt = t2_ - t1_
            Lt = np.linalg.norm(dt, axis=1)
            ut = dt / np.maximum(Lt, 1e-9)[:, None]
            a1 = np.einsum("nk,nk->n", pa - t1_, ut)
            a2 = np.einsum("nk,nk->n", pb - t1_, ut)
            lo_, hi_ = np.minimum(a1, a2), np.maximum(a1, a2)
            ov = np.clip(np.minimum(hi_, Lt) - np.maximum(lo_, 0), 0, None) / np.maximum(np.minimum(hi_ - lo_, Lt), 1e-6)
            keep = (za > 0.5) & (zb > 0.5) & (ov >= params.pair_overlap)
            dA = ((A - ci.center) @ ci.rot.T)[:, 2]
            dB = ((B - ci.center) @ ci.rot.T)[:, 2]
            pr = prior[si]
            has = ~np.isnan(pr[:, 0])
            dm = 0.5 * (dA + dB)
            with np.errstate(invalid="ignore"):
                keep &= ~has | ((dm > pr[:, 0] / params.prior_slack) & (dm < pr[:, 1] * params.prior_slack))
            si, ti, A, B, ov = si[keep], ti[keep], A[keep], B[keep], ov[keep]
            stats["cands_epi"] += len(si)
            if not len(si):
                continue
            order2 = np.lexsort((-ov, si))
            si, ti, A, B = si[order2], ti[order2], A[order2], B[order2]
            first = np.r_[True, si[1:] != si[:-1]]
            grp = np.cumsum(first) - 1
            rank = np.arange(len(si)) - np.flatnonzero(first)[grp]
            k3 = rank < params.top_per_segment
            si, ti, A, B = si[k3], ti[k3], A[k3], B[k3]
            stats["cands_kept"] += len(si)
            sup = np.zeros(len(si), int)
            sup_lists: list[list[int]] = [[] for _ in range(len(si))]
            for v in pool:
                if v == j or not len(segs.get(v, ())):
                    continue
                okv, _ = support_in_view(cams[v], segs[v], A, B, params.verify_px, params.verify_deg, params.verify_overlap)
                sup += okv
                for k in np.nonzero(okv)[0]:
                    sup_lists[k].append(int(v))
            for k in np.nonzero(sup >= params.min_support)[0]:
                s_ = int(si[k])
                if s_ not in best or sup[k] > best[s_][0]:
                    best[s_] = (int(sup[k]), A[k], B[k], [i, int(j)] + sup_lists[k])
        for s_, (_, A_, B_, sl) in best.items():
            As.append(A_)
            Bs.append(B_)
            seeds.append(i)
            obs.append(sl)
        stats["verified"] += len(best)
        if log and n_i % 20 == 0:
            log(f"  lines3d photo {n_i}/{len(used)}: raw {len(As)} candidates {stats['cands_epi']} "
                f"({time.perf_counter() - t0:.0f} s)")
    raw = RawLines(np.array(As, float).reshape(-1, 3), np.array(Bs, float).reshape(-1, 3), np.array(seeds, int), obs)
    return raw, stats


# ---------------------------------------------------------------- neighbours on integer cell keys (no scipy)


_OFF27 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def _cell_keys(q: np.ndarray, dims: np.ndarray) -> np.ndarray:
    return (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]


def close_owner_pairs(X: np.ndarray, owner: np.ndarray, cell: float, test=None, chunk: int = 1_000_000) -> np.ndarray:
    """Pairs (a < b) of owners with samples in the same or adjacent cells of size `cell` (a superset of the pairs
    with samples closer than `cell`), optionally filtered chunk by chunk with test(a, b) -> bool."""
    if not len(X):
        return np.zeros((0, 2), np.int64)
    q = np.floor(X / cell).astype(np.int64)
    q -= q.min(0) - 1
    dims = q.max(0) + 2
    ko = np.unique(np.stack([_cell_keys(q, dims), owner.astype(np.int64)], 1), axis=0)
    K, O = ko[:, 0], ko[:, 1]
    uk, start, cnt = np.unique(K, return_index=True, return_counts=True)
    out = []
    for dx, dy, dz in [o for o in _OFF27 if o > (0, 0, 0)] + [(0, 0, 0)]:
        off = (dx * dims[1] + dy) * dims[2] + dz
        pos = np.searchsorted(uk, uk + off)
        hit = (pos < len(uk)) & (uk[np.minimum(pos, len(uk) - 1)] == uk + off)
        ga, gb = np.nonzero(hit)[0], pos[hit]
        tot = cnt[ga] * cnt[gb]
        if off == 0:
            keep = cnt[ga] > 1
            ga, gb, tot = ga[keep], gb[keep], tot[keep]
        if not len(ga):
            continue
        csum = np.cumsum(tot)
        lo = 0
        while lo < len(ga):
            hi = int(np.searchsorted(csum, (csum[lo - 1] if lo else 0) + chunk, side="right"))
            hi = max(hi, lo + 1)
            g1, g2, tt = ga[lo:hi], gb[lo:hi], tot[lo:hi]
            T = int(tt.sum())
            rep = np.repeat(np.arange(len(tt)), tt)
            r = np.arange(T) - np.repeat(np.cumsum(tt) - tt, tt)
            cb = cnt[g2][rep]
            ia = start[g1][rep] + r // cb
            ib = start[g2][rep] + r % cb
            a, b = O[ia], O[ib]
            m = a < b if off == 0 else a != b
            a, b = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
            if test is not None and len(a):
                m = test(a, b)
                a, b = a[m], b[m]
            if len(a):
                out.append(np.unique(np.stack([a, b], 1), axis=0))
            lo = hi
    return np.unique(np.vstack(out), axis=0) if out else np.zeros((0, 2), np.int64)


def components(n: int, pairs: np.ndarray) -> np.ndarray:
    """Connected-component labels (the smallest member index) of n items under undirected pairs."""
    lab = np.arange(n)
    if not len(pairs):
        return lab
    a, b = pairs[:, 0], pairs[:, 1]
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        if np.array_equal(new, lab):
            return lab
        lab = new


class PointGrid:
    """SfM points on integer cells: distance to the nearest point within r (at most `cap` points per cell are kept,
    deterministically; enough for a support test)."""

    def __init__(self, P: np.ndarray, cell: float, cap: int = 32) -> None:
        self.cell, self.cap = cell, cap
        P = np.asarray(P, float)
        self.lo = P.min(0) - 2 * cell if len(P) else np.zeros(3)
        q = np.floor((P - self.lo) / cell).astype(np.int64)
        self.dims = (q.max(0) + 3) if len(P) else np.ones(3, np.int64)
        key = _cell_keys(q, self.dims)
        order = np.argsort(key, kind="stable")
        self.P, key = P[order], key[order]
        self.uk, self.start, cnt = np.unique(key, return_index=True, return_counts=True)
        self.cnt = np.minimum(cnt, cap)

    def nearest(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        best = np.full(len(X), np.inf)
        if not len(self.uk) or not len(X):
            return best
        q = np.floor((X - self.lo) / self.cell).astype(np.int64)
        valid = (q >= 1).all(1) & (q < self.dims - 1).all(1)
        for dx, dy, dz in _OFF27:
            k = _cell_keys(q + np.array([dx, dy, dz]), self.dims)
            pos = np.searchsorted(self.uk, k)
            hit = valid & (pos < len(self.uk)) & (self.uk[np.minimum(pos, len(self.uk) - 1)] == k)
            idx = np.nonzero(hit)[0]
            if not len(idx):
                continue
            s, c = self.start[pos[idx]], self.cnt[pos[idx]]
            for j in range(int(c.max())):
                m = c > j
                ii = idx[m]
                d = np.linalg.norm(self.P[s[m] + j] - X[ii], axis=1)
                best[ii] = np.minimum(best[ii], d)
        return best


# ---------------------------------------------------------------- merging, support


@dataclass
class Lines3D:
    A: np.ndarray                        # (n, 3)
    B: np.ndarray
    n_images: np.ndarray
    photos: list                         # per line: sorted camera indices
    support: np.ndarray                  # share of the line within support_radius of an SfM point
    resid: np.ndarray

    def __len__(self) -> int:
        return len(self.A)

    def subset(self, m: np.ndarray) -> Lines3D:
        idx = np.nonzero(m)[0] if m.dtype == bool else np.asarray(m, int)
        return Lines3D(self.A[idx], self.B[idx], self.n_images[idx], [self.photos[k] for k in idx], self.support[idx],
                       self.resid[idx])

    def to_json(self) -> list[dict[str, Any]]:
        return [{"a": np.round(self.A[k], 3).tolist(), "b": np.round(self.B[k], 3).tolist(), "n_images": int(self.n_images[k]),
                 "support": round(float(self.support[k]), 3)} for k in range(len(self))]


def merge_lines(raw: RawLines, cams: list[st.PosedCamera], params: LinesParams = DEFAULT_LINES) -> Lines3D:
    n = len(raw.A)
    if not n:
        z = np.zeros((0, 3))
        return Lines3D(z, z.copy(), np.zeros(0, int), [], np.zeros(0), np.zeros(0))
    A, B = raw.A, raw.B
    D = B - A
    L = np.linalg.norm(D, axis=1)
    U = D / np.maximum(L, 1e-9)[:, None]
    M = 0.5 * (A + B)
    C = np.array([c.center for c in cams])
    depth = np.linalg.norm(M - C[raw.seed], axis=1)
    ns = np.clip((L / 1.0).astype(int) + 1, 2, 40)
    owner = np.repeat(np.arange(n), ns)
    tt = np.concatenate([np.linspace(0, 1, k) for k in ns])
    Xs = A[owner] + tt[:, None] * D[owner]
    cosmin = math.cos(math.radians(params.merge_deg))

    def pdist(P_, A_, U_):
        r = P_ - A_
        return np.linalg.norm(r - np.einsum("nk,nk->n", r, U_)[:, None] * U_, axis=1)

    def test(a, b):
        ok = np.abs(np.einsum("nk,nk->n", U[a], U[b])) > cosmin
        tol = np.maximum(params.merge_tol_m, params.merge_tol_rel * np.minimum(depth[a], depth[b]))
        dmax = np.max(np.stack([pdist(A[a], A[b], U[b]), pdist(B[a], A[b], U[b]), pdist(A[b], A[a], U[a]),
                                pdist(B[b], A[a], U[a])]), axis=0)
        ok &= dmax < tol
        ta1 = np.einsum("nk,nk->n", A[a] - A[b], U[b])
        ta2 = np.einsum("nk,nk->n", B[a] - A[b], U[b])
        gap = np.maximum(np.minimum(ta1, ta2) - L[b], -np.maximum(ta1, ta2))
        return ok & (gap < params.merge_gap_m)

    pairs = close_owner_pairs(Xs, owner, params.merge_cell_m, test)
    lab = components(n, pairs)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.r_[True, lab[order][1:] != lab[order][:-1]])
    outA, outB, nim, photos, resid = [], [], [], [], []
    for gi, c0 in enumerate(cuts):
        c1 = cuts[gi + 1] if gi + 1 < len(cuts) else len(order)
        g = order[c0:c1]
        E = np.vstack([A[g], B[g]])
        c = E.mean(0)
        u = U[g[0]] if len(g) == 1 else np.linalg.svd(E - c, full_matrices=False)[2][0]
        t = (E - c) @ u
        outA.append(c + t.min() * u)
        outB.append(c + t.max() * u)
        ph = sorted({v for k in g for v in raw.obs[k]})
        photos.append(ph)
        nim.append(len(ph))
        resid.append(float(np.median(np.linalg.norm((E - c) - np.outer(t, u), axis=1))))
    return Lines3D(np.array(outA), np.array(outB), np.array(nim, int), photos, np.zeros(len(outA)), np.array(resid))


def line_support(lines: Lines3D, grid: PointGrid, params: LinesParams = DEFAULT_LINES) -> np.ndarray:
    """Share of each line's samples (every support_step_m) within support_radius_m of an SfM point."""
    if not len(lines):
        return np.zeros(0)
    L = np.linalg.norm(lines.B - lines.A, axis=1)
    ns = np.maximum((L / params.support_step_m).astype(int), 2)
    owner = np.repeat(np.arange(len(lines)), ns)
    tt = np.concatenate([np.linspace(0, 1, k) for k in ns])
    X = lines.A[owner] + tt[:, None] * (lines.B - lines.A)[owner]
    near = grid.nearest(X) <= params.support_radius_m
    return np.bincount(owner, weights=near.astype(float), minlength=len(lines)) / ns


# ---------------------------------------------------------------- predicted edges of a part


def predicted_edges(b: S.BuiltPart) -> dict[str, Any]:
    """The edges a part's map geometry predicts: eave edges (outline edges whose wall top is the eave), ridges
    (horizontal roof edges above the eave), hips (sloped edges between two roof planes), rakes (sloped edges over a
    gable wall) and the apex (one top point where >= 3 sloped edges meet). Curved roofs (dome, onion) give their
    eave ring only."""
    eaves = []
    for w in b.walls:
        if w.flat_top and abs(w.z1 - b.eave) < 0.05 and w.length >= 1.0:
            eaves.append({"edge": len(eaves), "ring": w.ring, "a": w.a.copy(), "b": w.b.copy(), "len": w.length,
                          "z": float(b.eave), "normal": w.normal()[:2]})
    out: dict[str, Any] = {"eave": eaves, "ridge": [], "hip": [], "rake": [], "apex": None}
    if b.roof.shape == "flat" or b.roof.shape in ("dome", "onion"):
        return out
    faces = b.roof.faces
    edges: dict[tuple, list[int]] = {}
    for fi, f in enumerate(faces):
        for poly in f.polys:
            for k in range(len(poly)):
                p, q = poly[k], poly[(k + 1) % len(poly)]
                if np.linalg.norm(q - p) < 0.05:
                    continue
                key = tuple(sorted([tuple(np.round(p, 2)), tuple(np.round(q, 2))]))
                edges.setdefault(key, []).append(fi)
    sloped_at_top: dict[tuple, int] = {}
    for key, fis in edges.items():
        p, q = np.array(key[0]), np.array(key[1])
        normals = {tuple(np.round(faces[fi].normal, 3)) for fi in fis}
        if len(fis) >= 2 and len(normals) == 1:
            continue                                     # a split inside one plane, not an edge
        dz = q[2] - p[2]
        if abs(dz) < 0.05:
            if abs(0.5 * (p[2] + q[2]) - b.eave) < 0.05:
                continue                                 # an eave (taken from the walls)
            out["ridge"].append({"a": p, "b": q, "len": float(np.linalg.norm(q - p)), "z": float(0.5 * (p[2] + q[2]))})
            continue
        lo_, hi_ = (p, q) if p[2] < q[2] else (q, p)
        out["hip" if len(normals) >= 2 else "rake"].append({"a": lo_, "b": hi_, "len": float(np.linalg.norm(q - p)),
                                                           "slope_deg": float(math.degrees(math.asin(min(1.0, abs(dz) / np.linalg.norm(q - p)))))})
        if abs(hi_[2] - b.top) < 0.05:
            k2 = tuple(np.round(hi_[:2], 1))
            sloped_at_top[k2] = sloped_at_top.get(k2, 0) + 1
    tops = np.array([v for f in faces for poly in f.polys for v in poly if abs(v[2] - b.top) < 0.05])
    if len(tops) and np.ptp(tops[:, 0]) < 0.3 and np.ptp(tops[:, 1]) < 0.3 and max(sloped_at_top.values(), default=0) >= 3:
        out["apex"] = np.r_[tops[:, :2].mean(0), b.top]
    return out


# ---------------------------------------------------------------- measuring a part


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


def _boot_wmedian(v: np.ndarray, w: np.ndarray, n: int, seed: int = 0) -> tuple[float, float]:
    if len(v) < 2:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        k = rng.integers(0, len(v), len(v))
        b.append(_wmedian(v[k], w[k]))
    return float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))


def _clusters_top(z: np.ndarray, w: np.ndarray, photos: list, gap: float, min_len: float, min_photos: int) -> np.ndarray | None:
    """The highest height cluster (sorted heights split at gaps > gap) with total length >= min_len and lines from
    >= min_photos photos: the mask of its members."""
    if not len(z):
        return None
    o = np.argsort(z)
    zs = z[o]
    cut = np.flatnonzero(np.diff(zs) > gap) + 1
    groups = np.split(o, cut)
    for g in groups[::-1]:
        if w[g].sum() >= min_len and len({p for k in g for p in photos[k]}) >= min_photos:
            m = np.zeros(len(z), bool)
            m[g] = True
            return m
    return None


def _closest_point(Aa: np.ndarray, Uu: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(Aa)) if w is None else w
    M = np.eye(3)[None] - Uu[:, :, None] * Uu[:, None, :]
    Am = (w[:, None, None] * M).sum(0)
    bm = (w[:, None, None] * M @ Aa[:, :, None])[:, :, 0].sum(0)
    return np.linalg.lstsq(Am, bm, rcond=None)[0]


def _point_line_dist(X: np.ndarray, Aa: np.ndarray, Uu: np.ndarray) -> np.ndarray:
    r = X[None] - Aa
    return np.linalg.norm(r - np.einsum("nk,nk->n", r, Uu)[:, None] * Uu, axis=1)


def _irls_apex(Aa: np.ndarray, Uu: np.ndarray, it: int = 10) -> tuple[np.ndarray, np.ndarray]:
    w = np.ones(len(Aa))
    X = _closest_point(Aa, Uu, w)
    for _ in range(it):
        r = _point_line_dist(X, Aa, Uu)
        w = np.where(r < 0.5, 1.0, 0.5 / np.maximum(r, 1e-9))
        X = _closest_point(Aa, Uu, w)
    return X, _point_line_dist(X, Aa, Uu)


@dataclass
class PartLines:
    id: str
    cameras: list[int] = field(default_factory=list)       # photos chosen for the part
    measurements: dict[str, dict[str, Any]] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)  # parameter -> why nothing is said
    lines_near: int = 0

    def evidence(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self.measurements.values()]


def _mark_bound(rec: dict[str, Any], value: float, lo: float | None, hi: float | None, params: LinesParams) -> None:
    """The height window the measurement searched (lines outside it were never considered) and whether the value
    sits at its edge (within one cluster gap): a value at a fit bound, which D15 never adopts (the truth may lie
    beyond the window)."""
    g = params.cluster_gap_m
    rec["search_m"] = [None if lo is None else round(float(lo), 2), None if hi is None else round(float(hi), 2)]
    rec["at_bound"] = bool((lo is not None and value <= lo + g) or (hi is not None and value >= hi - g))


def wall_surface_top(dp: np.ndarray, tp: np.ndarray, z: np.ndarray, length: float, lo: float, hi: float,
                     eave: float, band_m: float = 0.25, min_band: int = 30, min_surface: int = 10) -> dict[str, Any] | None:
    """The wall's own surface along one outline edge, from SfM points (dp: offset along the outward normal, tp:
    position along the edge, z: height): the densest 0.1 m offset bin of the points within the edge's window [lo, hi]
    and below the map's eave (the wall band), refined to the median offset of the points within band_m of it; then
    the 0.98 height quantile of every point within band_m of that plane. None when too few points."""
    along = (tp >= 0) & (tp <= length)
    band = along & (dp >= lo) & (dp <= hi) & (z <= eave)
    if band.sum() < min_band:
        return None
    bins = np.arange(lo, hi + 0.1, 0.1)
    if len(bins) < 2:
        return None
    h, e = np.histogram(dp[band], bins=bins)
    mode = 0.5 * (e[int(np.argmax(h))] + e[int(np.argmax(h)) + 1])
    near = band & (np.abs(dp - mode) <= band_m)
    plane = float(np.median(dp[near])) if near.sum() else mode
    surf = along & (np.abs(dp - plane) <= band_m)
    if surf.sum() < min_surface:
        return None
    return {"offset_m": round(plane, 2), "points": int(surf.sum()), "q98": round(float(np.quantile(z[surf], 0.98)), 2)}


def _vote(value: float, interval: list[float], mapv: float, tol: float) -> str | None:
    if value - mapv > tol and interval[0] > mapv:
        return "up"
    if mapv - value > tol and interval[1] < mapv:
        return "down"
    return None


def measure_part(b: S.BuiltPart, model: S.StructureModel, lines: Lines3D, params: LinesParams = DEFAULT_LINES,
                 snap_scale: float = 1.0, detector: str = "", cameras: list[int] | None = None,
                 points: np.ndarray | None = None) -> PartLines:
    """Assign the 3-D lines to the edges the part's map geometry predicts and measure its eave, apex / ridge and hip
    pitch (see the module docstring). Nothing is measured on an edge no line was assigned to. points: SfM points
    (track >= 3); an edge whose own wall-plane points rise above its highest line cluster is not measured there (that
    cluster is a string course, not the wall top)."""
    out = PartLines(b.id, sorted(cameras or []))
    scaled = abs(snap_scale - 1.0) > SNAP_SCALE_TOL
    base = {"kind": KIND, "detector": detector, "snap_scale": round(float(snap_scale), 4), "scale_corrected": bool(scaled),
            "can_vote": ["up", "down"]}
    if not len(lines):
        why = "no 3-D lines"
        if cameras is not None and len(cameras) < params.min_photos:
            why = f"seen in {len(cameras)} (< {params.min_photos}) photos: no 3-D lines"
        out.refused["eave"] = out.refused["top"] = why
        return out
    A, B = lines.A, lines.B
    D = B - A
    Ln = np.linalg.norm(D, axis=1)
    U = D / np.maximum(Ln, 1e-9)[:, None]
    slope = np.degrees(np.arcsin(np.clip(np.abs(D[:, 2]) / np.maximum(Ln, 1e-9), 0, 1)))
    zmid = 0.5 * (A[:, 2] + B[:, 2])
    pred = predicted_edges(b)
    others = [o for o in model.built if o.id != b.id]
    minw = max(b.spec.min_width(), 0.1)
    w_in = min(params.wall_in_m, params.wall_in_share * minw)
    tol_top = max(2.0, 0.1 * b.top)

    def owned_elsewhere(idx: np.ndarray) -> np.ndarray:
        """Lines whose both ends lie inside another part's outline (inset 0.25 m) within its height range."""
        bad = np.zeros(len(idx), bool)
        for o in others:
            ring = S.inset_ring(o.outer, 0.25)
            if ring is None or len(ring) < 3:
                continue
            ina = st.points_in_polygon(A[idx, :2], ring) & st.points_in_polygon(B[idx, :2], ring)
            zr = (np.minimum(A[idx, 2], B[idx, 2]) >= o.z0 - 0.5) & (np.maximum(A[idx, 2], B[idx, 2]) <= o.top + 0.5)
            bad |= ina & zr
        return bad

    # ------------------------------------------------ eave (wall tops)
    horiz = (slope < params.horiz_deg) & (Ln >= 1.0)
    hz = np.nonzero(horiz)[0]
    hz = hz[~owned_elsewhere(hz)] if len(hz) else hz
    zlo = b.z0 + 0.5 * (b.eave - b.z0)
    zhi = b.eave + max(max(2.0, 0.1 * b.eave), 0.6 * (b.eave - b.z0))
    per_edge, members, edges_unobs, edges_amb = [], [], [], []
    cosp = math.cos(math.radians(params.parallel_deg))
    Pn = np.zeros((0, 3))
    if points is not None and len(points):
        lo2 = b.outer.min(0) - (w_in + params.wall_out_m + 1.0)
        hi2 = b.outer.max(0) + (w_in + params.wall_out_m + 1.0)
        P_ = np.asarray(points, float)
        Pn = P_[(P_[:, 0] > lo2[0]) & (P_[:, 0] < hi2[0]) & (P_[:, 1] > lo2[1]) & (P_[:, 1] < hi2[1]) & (P_[:, 2] >= zlo)]
    for e in pred["eave"]:
        a2, b2 = e["a"], e["b"]
        Le = e["len"]
        u = (b2 - a2) / max(Le, 1e-9)
        n_out = e["normal"]
        mid = 0.5 * (a2 + b2) + 0.5 * n_out
        hidden = any(o.top > b.eave - 0.5 and st.points_in_polygon(mid[None], o.outer)[0] and o.z0 < b.eave for o in others)
        if hidden or Le < 2.0 or not len(hz):
            if not hidden:
                edges_unobs.append(e["edge"])
            continue
        ra, rb = A[hz, :2] - a2, B[hz, :2] - a2
        ta, tb = ra @ u, rb @ u
        da, db = ra @ n_out, rb @ n_out
        par = np.abs(U[hz, :2] @ u) / np.maximum(np.linalg.norm(U[hz, :2], axis=1), 1e-9) > cosp
        dd = 0.5 * (da + db)
        win = (np.minimum(da, db) >= -w_in) & (np.maximum(da, db) <= params.wall_out_m)
        ovl = np.clip(np.minimum(np.maximum(ta, tb), Le) - np.maximum(np.minimum(ta, tb), 0), 0, None)
        okov = (ovl >= 0.5) & (ovl >= 0.3 * Ln[hz])
        cand = par & win & okov & (zmid[hz] >= b.z0 + 1.0) & (zmid[hz] <= zhi)
        if cand.sum() == 0:
            edges_unobs.append(e["edge"])
            continue
        d_wall = _wmedian(dd[cand], ovl[cand])
        onw = cand & (np.abs(dd - d_wall) <= params.on_wall_m) & (zmid[hz] >= zlo)
        idx = np.nonzero(onw)[0]
        if not len(idx):
            edges_unobs.append(e["edge"])
            continue
        top = _clusters_top(zmid[hz][idx], ovl[idx], [lines.photos[k] for k in hz[idx]], params.cluster_gap_m,
                            max(params.cluster_min_len_m, params.cluster_min_share * Le), params.min_photos)
        if top is None:
            edges_unobs.append(e["edge"])
            continue
        gi = idx[top]
        zed = _wmedian(zmid[hz][gi], ovl[gi])
        row = {"edge": e["edge"], "z": round(zed, 2), "wall_offset_m": round(float(d_wall), 2), "lines": int(len(gi)),
               "covered_m": round(float(ovl[gi].sum()), 2), "len_m": round(Le, 2)}
        if len(gi) < 2 and ovl[gi].sum() < 0.5 * Le:
            edges_amb.append({**row, "why": "one line covering < 50 % of the edge"})
            continue
        if len(Pn):
            rp = Pn[:, :2] - a2
            tp, dp = rp @ u, rp @ n_out
            wp = (np.abs(dp - d_wall) <= 0.5) & (tp >= 0) & (tp <= Le)
            if wp.sum() >= 10:
                ztop = float(np.quantile(Pn[wp, 2], 0.98))
                row["wall_points_q98"] = round(ztop, 2)
                if ztop > zed + max(1.0, 0.05 * zed):
                    edges_amb.append({**row, "why": f"the wall's own SfM points reach {ztop:.1f} m, above its highest "
                                                    f"line cluster: not the wall top"})
                    continue
            # the other way round: the cluster must lie on the part's OWN wall. The wall's surface is where its SfM
            # points are densest (in the wall band below the map's eave: scaffold boards, poles and things in front
            # are sparser than a whole facade); when that surface's points stop well below the cluster, the cluster
            # is something else's top (scaffolding standing just outside, a structure in front), not this wall's
            surf_top = wall_surface_top(dp, tp, Pn[:, 2], Le, -w_in, params.wall_out_m, b.eave)
            if surf_top is not None:
                row["wall_surface"] = surf_top
                if zed > surf_top["q98"] + max(2.0, 0.1 * zed):        # more than the eave's tolerance above it
                    edges_amb.append({**row, "why": f"the highest line cluster ({zed:.1f} m) lies above the wall's own "
                                                    f"surface, whose SfM points stop at {surf_top['q98']:.1f} m "
                                                    f"(offset {surf_top['offset_m']:+.2f} m): not this wall's top"})
                    continue
                # a surface standing more than 1 m OUTSIDE the outline edge is not shown to be the part's own wall (a
                # scaffold net or a hoarding hides the wall behind it and is the densest surface itself: verify-r2
                # net, 1-1.3 m out); and the line cluster must lie on the surface it is checked against. A net
                # closer than 1 m is not told apart here (D25 needs the part's roof for an upward change, §2.4.8)
                if surf_top["offset_m"] > params.wall_surface_out_m:
                    edges_amb.append({**row, "why": f"the densest wall surface lies {surf_top['offset_m']:+.2f} m outside "
                                                    f"the outline edge (> {params.wall_surface_out_m:g} m): something "
                                                    f"standing in front of the wall (a scaffold net, a hoarding), not "
                                                    f"shown to be this wall"})
                    continue
                if abs(float(d_wall) - surf_top["offset_m"]) > params.on_wall_m:
                    edges_amb.append({**row, "why": f"the line cluster's plane ({float(d_wall):+.2f} m) is not on the "
                                                    f"wall surface ({surf_top['offset_m']:+.2f} m) it is checked against"})
                    continue
        per_edge.append(row)
        members.append((hz[gi], ovl[gi]))
    if per_edge:
        ids = np.concatenate([m[0] for m in members])
        wts = np.concatenate([m[1] for m in members])
        photos = sorted({p for k in ids for p in lines.photos[k]})
        if len(photos) >= params.min_photos:
            val = _wmedian(zmid[ids], wts)
            iv = _boot_wmedian(zmid[ids], wts, params.boot)
            tol_e = max(2.0, 0.1 * b.eave)
            zs = [p["z"] for p in per_edge]
            rec = {**base, "parameter": "eave", "value": round(val, 2), "map": round(float(b.eave), 2),
                   "interval": [round(iv[0], 2), round(iv[1], 2)], "n_photos": len(photos), "n_lines": int(len(ids)),
                   "edges_observed": [p["edge"] for p in per_edge], "edges_unobserved": sorted(edges_unobs),
                   "edges_ambiguous": [p["edge"] for p in edges_amb], "edges_total": len(pred["eave"]),
                   "per_edge": per_edge + edges_amb, "tol_m": round(tol_e, 2), "implied_top": round(val + (b.top - b.eave), 2)}
            _mark_bound(rec, val, zlo, zhi, params)
            rec["vote"] = _vote(val, rec["interval"], b.eave, tol_e)
            spread_tol = max(1.5, 0.075 * val)
            if rec["vote"] and len(ids) < 2:
                rec["vote_withheld"], rec["vote"] = f"one line only (would be {rec['vote']})", None
            elif rec["vote"] and max(zs) - min(zs) > spread_tol:
                rec["vote_withheld"] = f"edges disagree: {min(zs):.1f}-{max(zs):.1f} m (> {spread_tol:.1f} m; would be {rec['vote']})"
                rec["vote"] = None
            out.measurements["eave"] = rec
        else:
            out.refused["eave"] = f"eave lines from {len(photos)} (< {params.min_photos}) photos"
    elif not pred["eave"]:
        out.refused["eave"] = "no wall below the roof (a roof-only part): no eave edge"
    elif edges_amb:
        out.refused["eave"] = "; ".join(f"edge {p['edge']}: {p['why']}" for p in edges_amb[:3])
    else:
        out.refused["eave"] = "no horizontal lines on the wall plane of any eave edge"

    # ------------------------------------------------ apex (pointed roofs) and hip pitch
    if pred["apex"] is not None:
        X0 = pred["apex"]
        sl = np.nonzero((slope > params.slope_min_deg) & (slope < params.slope_max_deg) & (Ln >= params.sloped_min_len_m))[0]
        if len(sl):
            up = np.where((A[sl, 2] > B[sl, 2])[:, None], A[sl], B[sl])
            lo_ = np.where((A[sl, 2] > B[sl, 2])[:, None], B[sl], A[sl])
            reach = minw / 2 + 2.0
            near = (np.linalg.norm(up[:, :2] - X0[:2], axis=1) <= reach) & \
                   (np.linalg.norm(up[:, :2] - X0[:2], axis=1) < np.linalg.norm(lo_[:, :2] - X0[:2], axis=1)) & \
                   (up[:, 2] >= b.z0 + 0.5 * (b.eave - b.z0))
            sl = sl[near]
            if len(sl):
                kids = [o for o in others if o.spec.parent == b.id]
                bad = np.zeros(len(sl), bool)
                for o in kids:
                    ring = S.inset_ring(o.outer, 0.25)
                    if ring is not None and len(ring) >= 3:
                        bad |= st.points_in_polygon(A[sl, :2], ring) & st.points_in_polygon(B[sl, :2], ring)
                sl = sl[~bad]
        apex_rec = _apex(b, X0, sl, A, U, Ln, slope, lines, pred, params, minw)
        if isinstance(apex_rec, str):
            out.refused["top"] = apex_rec
        else:
            out.lines_near = int(len(sl))
            rec = {**base, "parameter": "top", "how": "apex", "map": round(float(b.top), 2), "tol_m": round(tol_top, 2),
                   **apex_rec["top"]}
            _mark_bound(rec, rec["value"], b.z0 + 0.5 * (b.eave - b.z0), None, params)
            rec["vote"] = _vote(rec["value"], rec["interval"], b.top, tol_top)
            rec["implied_top"] = rec["value"]
            out.measurements["top"] = rec
            out.measurements["apex_xy"] = {**base, "parameter": "apex_xy", "can_vote": [], "vote": None, **apex_rec["xy"]}
            if apex_rec.get("pitch"):
                p = apex_rec["pitch"]
                rh = b.top - b.eave
                implied = b.eave + rh * math.tan(math.radians(p["value"])) / max(math.tan(math.radians(p["map"])), 1e-6)
                pr = {**base, "parameter": "pitch", **p, "implied_top": round(implied, 2), "tol_m": round(tol_top, 2)}
                pr["vote"] = None                       # the apex itself votes on the top; pitch is recorded
                out.measurements["pitch"] = pr
    elif pred["ridge"]:
        rr = _ridge(b, pred, A, B, U, Ln, slope, zmid, lines, params, minw)
        if isinstance(rr, str):
            out.refused["top"] = rr
        else:
            rec = {**base, "parameter": "top", "how": "ridge", "map": round(float(b.top), 2), "tol_m": round(tol_top, 2), **rr}
            _mark_bound(rec, rec["value"], b.eave + 0.25 * (b.top - b.eave),
                        b.top + max(2.0, 0.1 * b.top, 0.5 * (b.top - b.z0)), params)
            rec["vote"] = _vote(rec["value"], rec["interval"], b.top, tol_top)
            if rec["vote"] and rec["n_lines"] < 2:
                rec["vote_withheld"], rec["vote"] = f"one line only (would be {rec['vote']})", None
            rec["implied_top"] = rec["value"]
            out.measurements["top"] = rec
    elif b.roof.shape not in ("flat",):
        out.refused["top"] = f"{b.roof.shape}: no straight edges at the top to measure (eave only)"
    return out


def _apex(b, X0, sl, A, U, Ln, slope, lines: Lines3D, pred, params: LinesParams, minw: float) -> dict[str, Any] | str:
    if len(sl) < params.apex_min_lines:
        return f"{len(sl)} (< {params.apex_min_lines}) sloped lines rising toward the map's apex"
    Aa, Uu = A[sl], U[sl]
    up = np.where((Uu[:, 2] > 0)[:, None], Aa + Ln[sl, None] * Uu, Aa)
    max_off = max(params.apex_max_off_m, params.apex_max_off_share * minw)
    zmin = b.z0 + 0.5 * (b.eave - b.z0)
    pairs = [(i, j) for i in range(len(sl)) for j in range(i + 1, len(sl))]
    if len(pairs) > params.apex_ransac:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(pairs), params.apex_ransac, replace=False)
        pairs = [pairs[k] for k in sorted(pick)]
    best = None
    for i, j in pairs:
        if abs(Uu[i] @ Uu[j]) > 0.995:
            continue
        X = _closest_point(Aa[[i, j]], Uu[[i, j]])
        if not np.isfinite(X).all() or X[2] < zmin or np.linalg.norm(X[:2] - X0[:2]) > max_off:
            continue
        d = _point_line_dist(X, Aa, Uu)
        inl = (d < params.apex_line_m) & (np.linalg.norm(up - X, axis=1) < params.apex_end_m)
        sc = (int(inl.sum()), float(Ln[sl][inl].sum()))
        if best is None or sc > best[0]:
            best = (sc, inl)
    if best is None or best[0][0] < params.apex_min_lines:
        return f"no apex supported by >= {params.apex_min_lines} sloped lines within {max_off:.1f} m of the map's apex"
    inl = best[1]
    X, r = _irls_apex(Aa[inl], Uu[inl])
    ids = sl[inl]
    photos = sorted({p for k in ids for p in lines.photos[k]})
    if len(photos) < params.min_photos:
        return f"apex lines from {len(photos)} (< {params.min_photos}) photos"
    if np.linalg.norm(X[:2] - X0[:2]) > max_off or X[2] < zmin:
        return f"the refined apex lies {np.linalg.norm(X[:2] - X0[:2]):.1f} m from the map's (> {max_off:.1f} m)"
    lowend = np.where((Uu[inl][:, 2] > 0)[:, None], Aa[inl], Aa[inl] + Ln[ids, None] * Uu[inl])
    az = np.degrees(np.arctan2(lowend[:, 0] - X[0], lowend[:, 1] - X[1])) % 360
    azs = np.sort(az)
    gaps = np.diff(np.r_[azs, azs[0] + 360])
    spread = 360.0 - float(gaps.max()) if len(azs) > 1 else 0.0
    if spread < params.apex_min_spread_deg:
        return f"apex lines from one direction only (spread {spread:.0f} < {params.apex_min_spread_deg:.0f} deg)"
    rng = np.random.default_rng(0)
    zs, xs = [], []
    for _ in range(params.boot):
        k = rng.integers(0, int(inl.sum()), int(inl.sum()))
        Xb = _closest_point(Aa[inl][k], Uu[inl][k])
        if np.isfinite(Xb).all():
            zs.append(Xb[2])
            xs.append(np.linalg.norm(Xb[:2] - X[:2]))
    out = {"top": {"value": round(float(X[2]), 2), "interval": [round(float(np.quantile(zs, 0.05)), 2), round(float(np.quantile(zs, 0.95)), 2)],
                   "n_photos": len(photos), "n_lines": int(inl.sum()), "azimuth_spread_deg": round(spread, 1),
                   "apex": [round(float(v), 2) for v in X]},
           "xy": {"value": round(float(np.linalg.norm(X[:2] - X0[:2])), 2), "offset_m": [round(float(v), 2) for v in X[:2] - X0[:2]],
                  "map": [round(float(v), 2) for v in X0[:2]], "n_photos": len(photos), "n_lines": int(inl.sum()),
                  "xy_spread_m": round(float(np.quantile(xs, 0.95)), 2) if xs else None}}
    hips = pred["hip"] + pred["rake"]
    if hips:
        ms = float(np.median(slope[ids]))
        mp = float(np.median([h["slope_deg"] for h in hips]))
        bs = [float(np.median(slope[ids][rng.integers(0, len(ids), len(ids))])) for _ in range(params.boot)]
        out["pitch"] = {"value": round(ms, 1), "map": round(mp, 1), "interval": [round(float(np.quantile(bs, 0.05)), 1),
                                                                                round(float(np.quantile(bs, 0.95)), 1)],
                        "n_photos": len(photos), "n_lines": int(len(ids)), "unit": "deg (hip / rake slope)"}
    return out


def _ridge(b, pred, A, B, U, Ln, slope, zmid, lines: Lines3D, params: LinesParams, minw: float) -> dict[str, Any] | str:
    ridges = pred["ridge"]
    zhi = b.top + max(2.0, 0.1 * b.top, 0.5 * (b.top - b.z0))
    zlo = b.eave + 0.25 * (b.top - b.eave)
    cosp = math.cos(math.radians(params.parallel_deg))
    ids_all, w_all = [], []
    for r in ridges:
        a2, b2 = r["a"][:2], r["b"][:2]
        Lr = float(np.linalg.norm(b2 - a2))
        if Lr < 1.0:
            continue
        u = (b2 - a2) / Lr
        n = np.array([u[1], -u[0]])
        hz = np.nonzero((slope < params.horiz_deg) & (Ln >= 1.0) & (zmid >= zlo) & (zmid <= zhi))[0]
        if not len(hz):
            continue
        ra, rb = A[hz, :2] - a2, B[hz, :2] - a2
        ta, tb = ra @ u, rb @ u
        off = np.maximum(np.abs(ra @ n), np.abs(rb @ n))
        par = np.abs(U[hz, :2] @ u) / np.maximum(np.linalg.norm(U[hz, :2], axis=1), 1e-9) > cosp
        ovl = np.clip(np.minimum(np.maximum(ta, tb), Lr) - np.maximum(np.minimum(ta, tb), 0), 0, None)
        ok = par & (off <= max(1.0, 0.1 * minw)) & (ovl >= 0.5) & (ovl >= 0.3 * Ln[hz])
        if ok.any():
            ids_all.append(hz[ok])
            w_all.append(ovl[ok])
    if not ids_all:
        return "no horizontal lines along the map's ridge"
    ids, w = np.concatenate(ids_all), np.concatenate(w_all)
    order = np.lexsort((-w, ids))                 # a ridge split at T-junctions: count each line once (its best overlap)
    ids, w = ids[order], w[order]
    first = np.r_[True, ids[1:] != ids[:-1]]
    ids, w = ids[first], w[first]
    top = _clusters_top(zmid[ids], w, [lines.photos[k] for k in ids], params.cluster_gap_m, params.cluster_min_len_m,
                        params.min_photos)
    if top is None:
        return f"ridge lines from < {params.min_photos} photos or too short"
    ids, w = ids[top], w[top]
    photos = sorted({p for k in ids for p in lines.photos[k]})
    val = _wmedian(zmid[ids], w)
    iv = _boot_wmedian(zmid[ids], w, params.boot)
    return {"value": round(val, 2), "interval": [round(iv[0], 2), round(iv[1], 2)], "n_photos": len(photos),
            "n_lines": int(len(ids))}


# ---------------------------------------------------------------- the channel


@dataclass
class LinesResult:
    parts: dict[str, PartLines]
    lines: Lines3D
    stats: dict[str, Any]
    timings_s: dict[str, float]
    detector: str
    selection: dict[str, list[int]]
    points: np.ndarray | None = None     # the SfM points (track >= 3) the measurements check walls against

    def evidence(self) -> dict[str, list[dict[str, Any]]]:
        """{part id: lines3d evidence records} for crowd3d.judge.decide(lines=...)."""
        return {pid: p.evidence() for pid, p in self.parts.items()}

    def summary(self) -> dict[str, Any]:
        return {"detector": self.detector, "stats": self.stats, "timings_s": self.timings_s,
                "parts": {pid: {"cameras": len(p.cameras), "measurements": p.measurements, "refused": p.refused}
                          for pid, p in self.parts.items()}}


def lines_channel(model: S.StructureModel, cameras: list[st.PosedCamera], points: np.ndarray | None,
                  quality: np.ndarray | None = None, snap_scale: float = 1.0, params: LinesParams = DEFAULT_LINES,
                  segments: dict[int, np.ndarray] | None = None, cache_dir: str | Path | None = None,
                  log=None) -> LinesResult:
    """The lines3d evidence channel for every built part (see the module docstring). cameras and points in the
    structure's ENU frame (after the snap); quality: per point (track length, angle). segments: precomputed raw
    segments per camera index (detector pixels), else detected (and cached in cache_dir)."""
    T: dict[str, float] = {}
    t = time.perf_counter()
    parts = {b.id: PartLines(b.id) for b in model.built}
    empty = Lines3D(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int), [], np.zeros(0), np.zeros(0))
    det = detector_name(params.detector)
    if points is None or len(points) < 50 or not model.built:
        for p in parts.values():
            p.refused["eave"] = p.refused["top"] = "no SfM points: 3-D lines need the SfM depth prior and support"
        return LinesResult(parts, empty, {"cameras": 0}, T, det, {})
    vol = _volume(model, params)
    P = np.asarray(points, float)
    inv = _in_volume(P, vol)
    track = np.asarray(quality)[:, 0] if quality is not None and len(quality) == len(P) else None
    rng = np.random.default_rng(0)
    idx_v = np.nonzero(inv)[0]
    sub = idx_v if len(idx_v) <= params.covis_points else np.sort(rng.choice(idx_v, params.covis_points, replace=False))
    Pc = P[sub]
    V = np.zeros((len(cameras), len(Pc)), bool)
    for c, cam in enumerate(cameras):
        vis, _, _ = visible_points(cam, Pc)
        V[c, vis] = True
    members = _part_members(model, Pc)
    sel, scores = select_cameras(model, cameras, V, members, params)
    used = sorted({c for v in sel.values() for c in v})
    if len(used) > params.max_cameras:
        tot = sum(scores.values())
        used = sorted(sorted(used, key=lambda c: (-tot[c], c))[:params.max_cameras])
        sel = {k: [c for c in v if c in set(used)] for k, v in sel.items()}
    for pid, p in parts.items():
        p.cameras = sel.get(pid, [])
    T["covisibility_s"] = round(time.perf_counter() - t, 2)
    if log:
        log(f"lines3d: {len(used)} photos for {len(parts)} parts (<= {params.max_cameras_per_part} per part), "
            f"{len(Pc)} covisibility points ({T['covisibility_s']} s)")
    # segments
    t = time.perf_counter()
    if segments is None:
        raw_segs, det, td = photo_segments(cameras, used, params, cache_dir)
    else:
        raw_segs, td = {i: segments[i] for i in used if i in segments}, 0.0
    T["detect_s"] = round(td, 2)
    hull = vol[4]
    segs: dict[int, np.ndarray] = {}
    from . import skyline as K                         # lazy: only for the volume's silhouette
    n_raw = 0
    for i in used:
        s = np.asarray(raw_segs.get(i, np.zeros((0, 4))), float).reshape(-1, 4)
        n_raw += len(s)
        if not len(s):
            segs[i] = s
            continue
        cam = cameras[i]
        s = np.c_[undistort_px(cam, s[:, :2]), undistort_px(cam, s[:, 2:])]
        mask = K.raster_mask(cam, K.prism_triangles(hull, vol[2], vol[3]), (cam.height, cam.width))
        mask = cv2.dilate(mask, np.ones((31, 31), np.uint8)) > 0
        mid = np.clip(np.round(0.5 * (s[:, :2] + s[:, 2:])).astype(int), 0, [cam.width - 1, cam.height - 1])
        s = s[mask[mid[:, 1], mid[:, 0]]]
        L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
        s = s[L >= params.min_len_px]
        L = L[L >= params.min_len_px]
        if len(s) > params.max_segments:
            s = s[np.argsort(-L, kind="stable")[:params.max_segments]]
        segs[i] = s
    T["segments_s"] = round(time.perf_counter() - t, 2)
    # neighbours: shared SfM points of the parts a photo was chosen for
    t = time.perf_counter()
    pos = {c: k for k, c in enumerate(used)}
    Vsel = np.zeros((len(used), len(Pc)), bool)
    for pid, cs in sel.items():
        for c in cs:
            Vsel[pos[c]] |= V[c] & members[pid]
    shared = Vsel.astype(np.int32) @ Vsel.T.astype(np.int32)
    ang_cache: dict[tuple[int, int], float] = {}

    def angle_of(i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in ang_cache:
            k = Vsel[pos[i]] & Vsel[pos[j]]
            if k.sum() < 3:
                ang_cache[key] = 0.0
            else:
                a, b2 = Pc[k] - cameras[i].center, Pc[k] - cameras[j].center
                cosv = np.einsum("nk,nk->n", a, b2) / np.linalg.norm(a, axis=1) / np.linalg.norm(b2, axis=1)
                ang_cache[key] = float(np.degrees(np.median(np.arccos(np.clip(cosv, -1, 1)))))
        return ang_cache[key]

    pri = idx_v if len(idx_v) <= params.prior_points else np.sort(rng.choice(idx_v, params.prior_points, replace=False))
    raw, mstats = triangulate(cameras, segs, used, shared, angle_of, P[pri], vol, params, log)
    T["match_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    merged = merge_lines(raw, cameras, params)
    T["merge_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    kept = merged.subset(merged.n_images >= params.min_images)
    sp = P[inv] if track is None else P[inv & (track >= params.support_min_track)]
    grid = PointGrid(sp, params.support_radius_m)
    cand = kept.n_images >= params.keep_min_images
    sup = np.zeros(len(kept))
    sup[cand] = line_support(kept.subset(cand), grid, params)
    kept.support = sup
    final = kept.subset((kept.n_images >= params.keep_min_images) & (sup >= params.support_min))
    T["support_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    for b in model.built:
        parts[b.id] = measure_part(b, model, final, params, snap_scale, det, sel.get(b.id, []), sp)
    T["measure_s"] = round(time.perf_counter() - t, 2)
    T["total_s"] = round(sum(v for k, v in T.items() if k != "total_s"), 2)
    stats = {"cameras": len(used), "segments_raw": int(n_raw), "segments_used": int(sum(len(v) for v in segs.values())),
             **mstats, "raw_lines": int(len(raw.A)), "merged": int(len(merged)), "kept_n_images": int(len(kept)),
             "final_SV": int(len(final))}
    if log:
        log(f"lines3d: {stats} timings {T}")
    return LinesResult(parts, final, stats, T, det, sel, sp)


def remeasure(result: LinesResult, model: S.StructureModel, params: LinesParams = DEFAULT_LINES,
              snap_scale: float = 1.0) -> LinesResult:
    """The same 3-D lines measured against another version of the structure (e.g. after a curator's frame): no
    re-triangulation (lines are photo facts in the scene frame; only the assignment to map edges changes)."""
    parts = {b.id: measure_part(b, model, result.lines, params, snap_scale, result.detector, result.selection.get(b.id, []),
                                result.points) for b in model.built}
    return LinesResult(parts, result.lines, result.stats, result.timings_s, result.detector, result.selection, result.points)
