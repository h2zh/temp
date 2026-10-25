"""D25-T's measurements (docs/STRUCTURE_FIRST.md §2.4.8 part 5): the lean of a round tall stack, by three channels,
against two independent gravity references, in one reconstruction's frame (z = that frame's up, x east, y north).

Channels (lean vectors in degrees, east and north components of the axis' tangent dx/dz, dy/dz):
- points axis: the axis of a surface of revolution by radial-profile SHARPNESS (all azimuths at once: for a candidate
  axis every point's distance to it at its own height; the (height band, radius bin) histogram is sharpest when the
  axis is right). Started from a robust line through RANSAC circles per height band; block bootstrap over bands.
- silhouette: the stack's outline swept over in-image leans in every photo that sees the whole stack; the score is
  the image gradient along the outline's near-vertical sides; per photo the best lean, then the lean vector by robust
  least squares over the photos (each sees the component across its line of sight); bootstrap over photos.
- 3-D lines: the near-vertical 3-D lines (crowd3d.lines3d) inside the stack's cylinder; their robust mean direction,
  cluster bootstrap over 1.5 m cells.
Gravity references (each a lean vector of the measured vertical in the same frame):
- camera: reconstruct.estimate_up on the cameras whose roll is <= 5 deg (photos whose pixels were stored rotated do not
  report it in their pose); the frame's own z is not trusted.
- scene verticals: the vertical vanishing direction of the photos' straight segments OUTSIDE the stack (a generous
  cylinder around it), with the known camera rotations (each segment's plane through its camera centre contains the
  vertical: IRLS on asin|n . v|), cluster bootstrap over photos.
measure() returns crowd3d.correct.rule_t's inputs.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from . import correct as C


def nelder_mead(f, x0, step, iters: int = 400, tol: float = 1e-10):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + 1e-12):
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# ---------------------------------------------------------------- points axis


def _fit_circle_ls(xy: np.ndarray, c0):
    cx, cy, r = c0
    x, y = xy[:, 0], xy[:, 1]
    for _ in range(10):
        dx, dy = x - cx, y - cy
        d = np.hypot(dx, dy) + 1e-12
        J = np.c_[-dx / d, -dy / d, -np.ones(len(x))]
        step = np.linalg.lstsq(J, -(d - r), rcond=None)[0]
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-9:
            break
    dx, dy = x - cx, y - cy
    d = np.hypot(dx, dy)
    res = d - r
    J = np.c_[-dx / (d + 1e-12), -dy / (d + 1e-12), -np.ones(len(x))]
    s2 = float(res @ res) / max(len(x) - 3, 1)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)
    return np.array([cx, cy, r]), cov


def _ransac_circle(xy, rmin, rmax, tol, centre, win, seed=0, iters=400):
    rng = np.random.default_rng(seed)
    n = len(xy)
    if n < 20:
        return None
    best, bestc = -1, None
    for _ in range(iters):
        p1, p2, p3 = xy[rng.choice(n, 3, replace=False)]
        a = np.array([[p2[0] - p1[0], p2[1] - p1[1]], [p3[0] - p1[0], p3[1] - p1[1]]]) * 2
        if abs(np.linalg.det(a)) < 1e-12:
            continue
        c = np.linalg.solve(a, np.array([p2 @ p2 - p1 @ p1, p3 @ p3 - p1 @ p1]))
        r = np.linalg.norm(p1 - c)
        if not (rmin <= r <= rmax) or np.linalg.norm(c - centre) > win:
            continue
        cnt = int((np.abs(np.linalg.norm(xy - c, axis=1) - r) < tol).sum())
        if cnt > best:
            best, bestc = cnt, (c[0], c[1], r)
    if bestc is None:
        return None
    p = np.array(bestc)
    for _ in range(3):
        inl = np.abs(np.linalg.norm(xy - p[:2], axis=1) - p[2]) < tol
        if inl.sum() < 10:
            return None
        p, cov = _fit_circle_ls(xy[inl], p)
    ang = np.arctan2(xy[inl, 1] - p[1], xy[inl, 0] - p[0])
    cov_ = float((np.histogram(ang, bins=36, range=(-math.pi, math.pi))[0] >= 3).mean())
    return {"c": p[:2], "r": float(p[2]), "cov": cov[:2, :2], "coverage": cov_}


def _band_axis(P, zlo, zhi, dz, centre, rmin, rmax):
    bands = []
    z, k = zlo, 0
    while z < zhi - 1e-9:
        xy = P[(P[:, 2] >= z) & (P[:, 2] < z + dz), :2]
        xy = xy[np.linalg.norm(xy - centre, axis=1) < 1.8 * rmax]
        res = _ransac_circle(xy, rmin, rmax, 0.03 * 0.5 * (rmin + rmax), centre, 0.5 * rmax, seed=k)
        if res is not None and res["coverage"] >= 0.2:
            res["z"] = z + dz / 2
            bands.append(res)
        z += dz
        k += 1
    if len(bands) < 4:
        return None
    zz = np.array([b["z"] for b in bands])
    cc = np.array([b["c"] for b in bands])
    zref = float(zz.mean())
    A = np.c_[np.ones(len(zz)), zz - zref]
    sx = np.linalg.lstsq(A, cc[:, 0], rcond=None)[0]
    sy = np.linalg.lstsq(A, cc[:, 1], rcond=None)[0]
    return {"c": np.array([sx[0], sy[0]]), "slope": np.array([sx[1], sy[1]]), "zref": zref,
            "R0": float(np.median([b["r"] for b in bands])), "bands": len(bands)}


class SharpAxis:
    def __init__(self, P, c0, s0, zref, R0, zlo, zhi, dz_frac=0.1, dr_frac=0.012, win_frac=1.45, rmin_frac=0.3):
        self.R0, self.zref = R0, zref
        a = c0 + np.outer(P[:, 2] - zref, s0)
        r = np.linalg.norm(P[:, :2] - a, axis=1)
        keep = (r < win_frac * R0) & (r > rmin_frac * R0) & (P[:, 2] >= zlo) & (P[:, 2] < zhi)
        self.P = P[keep]
        self.band = np.floor((self.P[:, 2] - zlo) / (dz_frac * R0)).astype(int)
        self.nb = int(self.band.max()) + 1 if len(self.P) else 0
        self.dr = dr_frac * R0
        self.nr = int(math.ceil(2.0 * R0 / self.dr)) + 2
        self.weights = np.ones(max(self.nb, 1))

    def score(self, x, weights=None):
        a = x[:2] + np.outer(self.P[:, 2] - self.zref, x[2:4])
        r = np.linalg.norm(self.P[:, :2] - a, axis=1)
        w = self.weights if weights is None else weights
        tot = 0.0
        for off in (0.0, 0.5):
            rb = np.clip(np.floor(r / self.dr + off).astype(int), 0, self.nr - 1)
            H = np.bincount(self.band * self.nr + rb, minlength=self.nb * self.nr).reshape(self.nb, self.nr).astype(float)
            tot += float((w[:, None] * H * H).sum())
        return -tot

    def fit(self, x0, weights=None, iters=400):
        step = np.array([0.05 * self.R0, 0.05 * self.R0, 0.01, 0.01])
        f = lambda x: self.score(x, weights)
        x, _ = nelder_mead(f, x0, step, iters=iters)
        x, v = nelder_mead(f, x, step * 0.2, iters=iters)
        return x, v


def points_axis(P: np.ndarray, centre, R0_guess: float, zlo: float, zhi: float, n_boot: int = 60) -> dict[str, Any]:
    """The points-axis channel: lean (deg east, north), bands, bootstrap sigma (deg)."""
    out: dict[str, Any] = {"channel": "points"}
    ax = _band_axis(P, zlo, zhi, 0.12 * R0_guess, np.asarray(centre, float), 0.5 * R0_guess, 1.3 * R0_guess)
    if ax is None:
        out["why"] = "fewer than 4 height bands with a ring"
        return out
    zref = 0.5 * (zlo + zhi)
    c0 = ax["c"] + ax["slope"] * (zref - ax["zref"])
    sa = SharpAxis(P, c0, ax["slope"], zref, ax["R0"], zlo, zhi)
    if sa.nb < 3:
        out["why"] = "too few points around the axis"
        return out
    x, _ = sa.fit(np.r_[c0, ax["slope"]])
    rng = np.random.default_rng(0)
    B = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, sa.nb, sa.nb), minlength=sa.nb).astype(float)
        # every bootstrap fit starts from a perturbed axis (centre +-3 % R0, lean +-0.5 deg): a flat direction of the
        # score (a partial arc trades the axis against the radius) shows in the spread instead of staying at x
        x0 = x + np.r_[rng.normal(0.0, BOOT_START_CENTRE * sa.R0, 2), np.tan(np.radians(rng.normal(0.0, BOOT_START_LEAN, 2)))]
        xb, _ = sa.fit(x0, weights=w, iters=200)
        B.append(xb)
    B = np.array(B)
    lean = np.degrees(np.arctan(x[2:]))
    lb = np.degrees(np.arctan(B[:, 2:]))
    cov = azimuth_coverage(sa.P, x, zref)
    boot = float(np.max(lb.std(0)))
    out.update(lean_en_deg=[round(float(lean[0]), 4), round(float(lean[1]), 4)], bands=int(sa.nb),
               sigma_boot_deg=round(boot, 4), azimuth_coverage=round(cov, 3),
               sigma_deg=round(boot / max(cov, 1.0 / AZ_BINS), 4), R0=round(ax["R0"], 3), centre=x[:2].round(3).tolist(),
               slope=[round(float(v), 6) for v in x[2:]], zref=round(zref, 3), points=int(len(sa.P)),
               base_centre=(x[:2] + x[2:] * (zlo - zref)).round(3).tolist())
    return out


AZ_BINS, AZ_MIN_SHARE = 36, 0.1
BOOT_START_CENTRE, BOOT_START_LEAN = 0.03, 0.5


def azimuth_coverage(P: np.ndarray, x: np.ndarray, zref: float) -> float:
    """The share of the stack's azimuth the points cover about the fitted axis: 10-degree sectors holding >= 10 % of
    a uniform ring's share. A one-face reconstruction sees an arc; the points-axis sigma is divided by this share
    (verify-r1: a 60-degree arc missed the full ring's lean by 1.8 deg with a bootstrap sigma of 0.10)."""
    if not len(P):
        return 0.0
    a = x[:2] + np.outer(P[:, 2] - zref, x[2:4])
    ang = np.arctan2(P[:, 1] - a[:, 1], P[:, 0] - a[:, 0])
    h = np.histogram(ang, bins=AZ_BINS, range=(-math.pi, math.pi))[0]
    return float(np.mean(h >= AZ_MIN_SHARE * len(P) / AZ_BINS))


def axis_centre(ch: dict[str, Any], z: float, fallback) -> np.ndarray:
    """Where a measured axis passes at height z (the points channel's centre at zref plus its slope)."""
    if ch.get("centre") is None or ch.get("slope") is None:
        return np.asarray(fallback, float)
    return np.asarray(ch["centre"], float) + np.asarray(ch["slope"], float) * (float(z) - float(ch["zref"]))


# ---------------------------------------------------------------- 3-D lines


def lines_vertical(A: np.ndarray, B: np.ndarray, photos: list, centre, radius: float, max_angle: float = 15.0,
                   cell: float = 1.5, n_boot: int = 300, slope=None, zref: float = 0.0) -> dict[str, Any]:
    """The near-vertical 3-D lines inside the stack's cylinder (about the axis centre + slope (z - zref) when a slope
    is given): their robust mean direction (lean deg east, north)."""
    out: dict[str, Any] = {"channel": "lines3d"}
    D = B - A
    Ln = np.linalg.norm(D, axis=1)
    ang = np.degrees(np.arccos(np.clip(np.abs(D[:, 2]) / np.maximum(Ln, 1e-9), 0, 1)))
    mid = 0.5 * (A + B)
    c_at = np.asarray(centre, float) + (np.outer(mid[:, 2] - zref, np.asarray(slope, float)) if slope is not None else 0.0)
    sel = (ang < max_angle) & (np.linalg.norm(mid[:, :2] - c_at, axis=1) < radius)
    out["n_lines"] = int(sel.sum())
    out["n_photos"] = len({p for i in np.flatnonzero(sel) for p in photos[i]})
    if sel.sum() < 5:
        out["why"] = f"{sel.sum()} near-vertical lines in the cylinder"
        return out
    D = D[sel] * np.sign(D[sel][:, 2:3])
    t = D[:, :2] / D[:, 2:3]
    w = Ln[sel]

    def rmean(w_):
        m = np.average(t, axis=0, weights=w_)
        for _ in range(10):
            r = np.linalg.norm(t - m, axis=1)
            s = 1.4826 * np.median(r) + 1e-9
            ww = w_ * np.clip(1 - (r / (4.685 * s)) ** 2, 0, None) ** 2
            if ww.sum() <= 0:
                break
            m = np.average(t, axis=0, weights=ww)
        return m
    m = rmean(w)
    key = np.floor(mid[sel] / cell).astype(int)
    _, grp = np.unique(key, axis=0, return_inverse=True)
    grp = grp.ravel()
    ng = int(grp.max()) + 1
    rng = np.random.default_rng(0)
    bs = np.array([rmean(w * np.bincount(rng.integers(0, ng, ng), minlength=ng)[grp]) for _ in range(n_boot)])
    lean = np.degrees(np.arctan(m))
    out.update(lean_en_deg=[round(float(lean[0]), 4), round(float(lean[1]), 4)],
               sigma_deg=round(float(np.degrees(bs.std(0)).max()), 4), groups=ng)
    return out


# ---------------------------------------------------------------- silhouette


def _side_score(G: np.ndarray, mask: np.ndarray, border: int = 8) -> float:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return math.nan
    c = max(cnts, key=len)[:, 0, :].astype(float)
    d = np.roll(c, -5, 0) - np.roll(c, 5, 0)
    side = np.abs(d[:, 1]) > 2 * np.abs(d[:, 0])
    h, w = G.shape
    ins = (c[:, 0] > border) & (c[:, 0] < w - border) & (c[:, 1] > border) & (c[:, 1] < h - border)
    k = side & ins
    if k.sum() < 50:
        return math.nan
    q = np.round(c[k]).astype(int)
    return float(G[q[:, 1], q[:, 0]].mean())


def silhouette(cameras: list, prisms: list[np.ndarray], pivot, taus=None, load=None, top=None) -> dict[str, Any]:
    """The silhouette channel: prisms (each (2n, 3): a part's ring at its base and top, in the frame) swept over
    in-image leans tau about the pivot; per photo seeing the whole stack (>= 40 % of the image tall) the best tau
    (sub-step parabola), its contrast; the lean vector by robust least squares over photos with contrast >= 0.2 off
    the sweep's bounds; bootstrap over photos. load(cam) -> grayscale image (default: cam.load()). top: the measured
    axis' top point (a photo must see the axis from the pivot to it; default: straight above the pivot)."""
    taus = np.arange(-8.0, 8.01, 0.5) if taus is None else np.asarray(taus, float)
    pivot = np.asarray(pivot, float)
    allv = np.vstack(prisms)
    zt = float(allv[:, 2].max())
    rows = []
    for cam in cameras:
        ends = np.array([pivot, np.asarray(top, float) if top is not None else [pivot[0], pivot[1], zt]])
        uv, z = cam.project(ends)
        if not ((z > 0).all() and (uv[:, 0] > 0).all() and (uv[:, 0] < cam.width).all() and (uv[:, 1] > 0).all()
                and (uv[:, 1] < cam.height).all()) or np.linalg.norm(uv[0] - uv[1]) < 0.4 * max(cam.width, cam.height):
            continue
        try:
            g = load(cam) if load is not None else cv2.cvtColor(np.asarray(cam.load()), cv2.COLOR_RGB2GRAY)
        except Exception:                                              # noqa: BLE001 -- a photo that cannot be read
            continue
        if g is None or g.shape[:2] != (cam.height, cam.width):
            continue
        g = g.astype(np.float32)
        G = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1.5)
        dxy = pivot[:2] - cam.center[:2]
        view_az = math.degrees(math.atan2(dxy[0], dxy[1])) % 360
        perp = (view_az + 90) % 360
        sc = []
        for t in taus:
            R, c = C.pose_matrix({"tilt_deg": abs(t), "azimuth_deg": perp if t >= 0 else (perp + 180) % 360,
                                  "pivot": pivot.tolist()})
            m = np.zeros((cam.height, cam.width), np.uint8)
            for V in prisms:
                X = (V - c) @ R.T + c
                uvp, zp = cam.project(X)
                if (zp <= 0.5).any():
                    continue
                cv2.fillConvexPoly(m, cv2.convexHull(np.round(uvp * 4).astype(np.int32)), 255, shift=2)
            sc.append(_side_score(G, m))
        sc = np.array(sc)
        if np.isnan(sc).all():
            continue
        i = int(np.nanargmax(sc))
        tb = float(taus[i])
        if 0 < i < len(taus) - 1 and np.isfinite(sc[i - 1:i + 2]).all():
            a, b_, cc = sc[i - 1], sc[i], sc[i + 1]
            den = a - 2 * b_ + cc
            if den < 0:
                tb = taus[i] + 0.5 * (taus[1] - taus[0]) * (a - cc) / den
        med = float(np.nanmedian(sc))
        rows.append({"photo": cam.photo_id or cam.name, "perp_az": round(perp, 2), "tau": round(tb, 3),
                     "contrast": round(float((sc[i] - med) / max(med, 1e-6)), 3), "at_bound": i in (0, len(taus) - 1)})
    out: dict[str, Any] = {"channel": "silhouette", "rows": rows}
    use = [r for r in rows if r["contrast"] >= C.TILT_SIL_CONTRAST and not r["at_bound"]]
    out["n_photos"] = len(use)
    if len(use) < 4:
        out["why"] = f"{len(use)} photos see the stack in profile with contrast"
        return out

    def solve(rs):
        A = np.array([[math.sin(math.radians(r["perp_az"])), math.cos(math.radians(r["perp_az"]))] for r in rs])
        b = np.array([r["tau"] for r in rs])
        L = np.linalg.lstsq(A, b, rcond=None)[0]
        k = np.ones(len(b), bool)
        for _ in range(5):
            res = b - A @ L
            mad = 1.4826 * np.median(np.abs(res)) + 1e-6
            k = np.abs(res) < 3 * mad
            if k.sum() < 3:
                break
            L = np.linalg.lstsq(A[k], b[k], rcond=None)[0]
        return L, k
    L, k = solve(use)
    rng = np.random.default_rng(0)
    B = []
    for _ in range(300):
        try:
            B.append(solve([use[j] for j in rng.integers(0, len(use), len(use))])[0])
        except np.linalg.LinAlgError:
            pass
    B = np.array(B)
    out.update(lean_en_deg=[round(float(L[0]), 4), round(float(L[1]), 4)], inlier_share=round(float(k.mean()), 3),
               contrast=round(float(np.median([r["contrast"] for r in use])), 3),
               sigma_deg=round(float(np.max(B.std(0))), 4) if len(B) else 9.0)
    return out


# ---------------------------------------------------------------- gravity references


def camera_roll(cameras: list, up=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    rights = np.array([np.asarray(c.rot)[0] for c in cameras])
    return np.degrees(np.arcsin(np.clip(rights @ np.asarray(up, float), -1, 1)))


def gravity_camera(cameras: list, roll_max: float = C.TILT_ROLL_MAX, n_boot: int = 200) -> dict[str, Any]:
    """estimate_up on the cameras with |roll| <= 5 deg (roll measured against the first estimate on all of them);
    the lean of that up in the frame (deg east, north)."""
    from .reconstruct import estimate_up
    ups = np.array([-np.asarray(c.rot)[1] for c in cameras])
    rights = np.array([np.asarray(c.rot)[0] for c in cameras])
    cen = np.array([c.center for c in cameras])
    u0 = estimate_up(ups, rights, cen)
    keep = np.abs(np.degrees(np.arcsin(np.clip(rights @ u0, -1, 1)))) <= roll_max
    out: dict[str, Any] = {"cameras": len(cameras), "rolled_removed": int((~keep).sum()), "rolled_removed_checked": True,
                           "roll_max_deg": roll_max}
    if keep.sum() < 5:
        out["why"] = f"{keep.sum()} cameras with |roll| <= {roll_max:g} deg"
        out["rolled_removed_checked"] = False
        return out
    u = estimate_up(ups[keep], rights[keep], cen[keep])
    u = u if u[2] > 0 else -u
    out["lean_en_deg"] = [round(float(v), 4) for v in np.degrees(np.arctan2(u[:2], u[2]))]
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(keep)
    bs = []
    for _ in range(n_boot):
        i = rng.choice(idx, len(idx))
        ub = estimate_up(ups[i], rights[i], cen[i])
        ub = ub if ub[2] > 0 else -ub
        bs.append(np.degrees(np.arctan2(ub[:2], ub[2])))
    out["boot_sigma_deg"] = round(float(np.array(bs).std(0).max()), 4)        # recorded; never used as the error
    return out


def _vp_solve(N, W, v0, rounds=(10.0, 5.0, 3.0, 2.0, 2.0, 2.0)):
    v = np.asarray(v0, float) / np.linalg.norm(v0)
    for c in rounds:
        r = np.degrees(np.arcsin(np.clip(np.abs(N @ v), 0, 1)))
        w = W * np.clip(1 - (r / c) ** 2, 0, None) ** 2
        if w.sum() <= 0:
            break
        _, V = np.linalg.eigh((N * w[:, None]).T @ N)
        u = V[:, 0]
        v = u if u @ v >= 0 else -u
    return v, np.degrees(np.arcsin(np.clip(np.abs(N @ v), 0, 1)))


def gravity_scene_verticals(cameras: list, base, radius: float, z0: float, z1: float, load=None,
                            min_len_px: float = 40.0, n_boot: int = 300) -> dict[str, Any]:
    """The vertical vanishing direction of the photos' segments outside the stack's cylinder (base xy, radius,
    z0-z1), with the known camera rotations; inliers within 2 deg; cluster bootstrap over photos."""
    from . import lines3d as L
    base = np.asarray(base, float)
    Ns, Ws, owner = [], [], []
    for i, cam in enumerate(cameras):
        try:
            g = load(cam) if load is not None else cv2.cvtColor(np.asarray(cam.load()), cv2.COLOR_RGB2GRAY)
        except Exception:                                              # noqa: BLE001
            continue
        if g is None:
            continue
        segs, _ = L.detect_segments(g, "auto", min_len_px)
        if not len(segs):
            continue
        segs = np.c_[L.undistort_px(cam, segs[:, :2]), L.undistort_px(cam, segs[:, 2:])]
        a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        ring = np.c_[base[0] + radius * np.cos(a), base[1] + radius * np.sin(a)]
        X = np.vstack([np.c_[ring, np.full(48, z0)], np.c_[ring, np.full(48, z1)]])
        uv, z = cam.project(X)
        mask = np.zeros((cam.height, cam.width), np.uint8)
        if (z > 0.5).all():
            cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
        elif (z > 0.5).any():
            continue
        if mask.mean() > 0.9 * 255:
            continue
        mid = np.clip(np.round(0.5 * (segs[:, :2] + segs[:, 2:])).astype(int), 0, [cam.width - 1, cam.height - 1])
        segs = segs[mask[mid[:, 1], mid[:, 0]] == 0]
        if not len(segs):
            continue
        n, _, _ = L._seg_planes(cam, segs)
        Ns.append(n)
        Ws.append(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))
        owner.append(np.full(len(n), i))
    out: dict[str, Any] = {"photos_used": len(Ns)}
    if not Ns:
        out["why"] = "no segments outside the stack"
        return out
    N, W, own = np.concatenate(Ns), np.concatenate(Ws), np.concatenate(owner)
    v, r = _vp_solve(N, W, np.array([0, 0, 1.0]))
    inl = r < 2.0
    rng = np.random.default_rng(0)
    ids = np.unique(own)
    B = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.choice(ids, len(ids)), minlength=own.max() + 1)
        vb, _ = _vp_solve(N, W * cnt[own], v, rounds=(3.0, 2.0, 2.0))
        B.append(np.degrees(vb[:2] / vb[2]))
    B = np.array(B)
    out.update(lean_en_deg=[round(float(x), 4) for x in np.degrees(v[:2] / v[2])], inliers=int(inl.sum()),
               photos=int(len(np.unique(own[inl]))), sigma_deg=round(float(B.std(0).max()), 4), segments=int(len(N)))
    return out


def measure(cameras: list, points: np.ndarray, stack_prisms: list[np.ndarray], centre, R0_guess: float, zlo: float,
            zhi: float, lines: Any = None, load=None, silhouette_photos: bool = True) -> dict[str, Any]:
    """Every channel and both gravity references for one reconstruction (frame z = its up). lines: a crowd3d.lines3d
    result / Lines3D of the stack (or None)."""
    ch: dict[str, Any] = {"points": points_axis(points, centre, R0_guess, zlo, zhi)}
    pa = ch["points"]
    c_ax = axis_centre(pa, zlo, centre)
    if lines is not None:
        Lz = getattr(lines, "lines", lines)
        ch["lines3d"] = lines_vertical(np.asarray(Lz.A), np.asarray(Lz.B), Lz.photos, pa.get("centre", centre),
                                       1.35 * R0_guess, slope=pa.get("slope"), zref=float(pa.get("zref", 0.0)))
    if silhouette_photos:
        ch["silhouette"] = silhouette(cameras, stack_prisms, np.r_[c_ax, zlo], load=load)
    grav = {"camera": gravity_camera(cameras),
            "scene-verticals": gravity_scene_verticals(cameras, c_ax, 1.6 * R0_guess + 0.15 * (zhi - zlo), zlo - 2.0,
                                                       zlo + 1.2 * (zhi - zlo), load=load)}
    return {"channels": ch, "gravity": grav}
