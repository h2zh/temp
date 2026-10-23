"""Judging structure parts against the photos (docs/STRUCTURE_FIRST.md §2.2, §2.4.2-§2.4.3, §2.8).

Photos never shape a part here: they check the map. Everything is decided AT THE MAP VALUE.

- frame_gate(): the per-edge residual gate (G11). Every outline edge >= 5 m long with >= 200 wall points (2-20 m
  high, nearest edge within 10 m, points far inside counted too) must have a median distance <= 1.5 m. When it
  fails, a frame ticket opens and no parameter is verified or fitted (the evidence is still recorded).
- frame_similarity(): one similarity for the whole building (outline scale k about its centre, shift t) from the
  wall planes (least squares on the edges' median wall positions). Only reported and put in the ticket: a
  change of the map's outline is a curator's decision (D15 / D16); single-sided data cannot separate k from t,
  so it needs walls facing >= 180 deg of directions including two opposite ones.
- sfm_channel(): the SfM-point channel: the 0.98 quantile of point heights over the part's exposed area (minus the
  parts standing on it and 1.5 m around taller neighbours) against the model's own surface over the same area.
  Points can only say "the map is too low" (a roof nobody photographed has no points; trees and neighbours sit
  above parts too, so it never decides alone).
- decide(): the rules of §2.4.3 per part, from one or two sky masks' skyline profiles (crowd3d.skyline) and the
  SfM channel; tolerance TOL = max(2 m, 10 % of the map top); each channel votes only for the direction it can
  observe; a skyline alone never raises anything ("up?" counts only with another "up"); slender parts (own height /
  narrowest width >= 2 and width <= 10 m) never move; verification needs two masks agreeing, a stable profile off
  its bounds, the frame gate passed and a snap scale correction <= 3 %. States: map, photo-verified, disputed
  (ticket), photo-fitted (D15 auto-adopt, off by default: reported as 'would fit').
- apply_decisions() writes states, evidence and tickets into the structure's parameters (massing.json v2);
  check_decisions() is G4 / G9: no verified or fitted value without the evidence the rules need.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import stylize as st
from . import structure as S

FRAME_EDGE_MIN_M = 5.0
FRAME_EDGE_MIN_POINTS = 200
FRAME_MAX_MEDIAN_M = 1.5
FRAME_K_TOL = 0.03
FRAME_T_TOL_M = 1.5
SNAP_SCALE_TOL = 0.03
MIN_PHOTOS = 3


def _wall_points(points: np.ndarray, quality: np.ndarray | None, z_lo: float = 2.0, z_hi: float = 20.0,
                 min_track: int = 3) -> np.ndarray:
    p = np.asarray(points, float)
    sel = (p[:, 2] > z_lo) & (p[:, 2] < z_hi)
    if quality is not None and len(quality) == len(p):
        sel &= np.asarray(quality)[:, 0] >= min_track
    return p[sel, :2]


def _edge_assignment(xy: np.ndarray, ring: np.ndarray, max_d: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest outline edge of each point (within max_d), its distance, and the signed offset (outward +)."""
    n = len(ring)
    D = np.full((len(xy), n), np.inf)
    Sg = np.zeros((len(xy), n))
    for k in range(n):
        a, b = ring[k], ring[(k + 1) % n]
        t = b - a
        L = float(np.linalg.norm(t))
        if L < 1e-9:
            continue
        u = t / L
        s = np.clip((xy - a) @ u, 0, L)
        D[:, k] = np.linalg.norm(xy - (a + s[:, None] * u), axis=1)
        Sg[:, k] = (xy - a) @ np.array([u[1], -u[0]])
    k = D.argmin(1)
    d = D[np.arange(len(xy)), k]
    sgn = Sg[np.arange(len(xy)), k]
    k[d > max_d] = -1
    return k, d, sgn


def frame_gate(points: np.ndarray, rings: list[np.ndarray], quality: np.ndarray | None = None) -> dict[str, Any]:
    """G11, the per-edge residual gate (design §2.2 step 1)."""
    ring = S.ccw(rings[0])
    xy = _wall_points(points, quality)
    k, d, _ = _edge_assignment(xy, ring)
    edges, failing = [], []
    for e in range(len(ring)):
        m = k == e
        L = float(np.linalg.norm(ring[(e + 1) % len(ring)] - ring[e]))
        if m.sum() < FRAME_EDGE_MIN_POINTS:
            continue
        row = {"edge": e, "len_m": round(L, 2), "points": int(m.sum()), "median_m": round(float(np.median(d[m])), 2)}
        edges.append(row)
        if L >= FRAME_EDGE_MIN_M and row["median_m"] > FRAME_MAX_MEDIAN_M:
            failing.append(row)
    return {"passed": not failing, "edges": edges, "failing": failing,
            "rule": f"edges >= {FRAME_EDGE_MIN_M:g} m with >= {FRAME_EDGE_MIN_POINTS} wall points: median <= {FRAME_MAX_MEDIAN_M:g} m"}


def frame_similarity(points: np.ndarray, rings: list[np.ndarray], quality: np.ndarray | None = None,
                     min_points: int = FRAME_EDGE_MIN_POINTS) -> dict[str, Any]:
    """Whole-building similarity from the wall planes (design §2.2 step 2, the wall-plane method): per edge with
    enough wall points (3-16 m high), the median wall position along its outward normal n_i; least squares for
    position_i = k * d_i + n_i . t, d_i the edge's distance from the outline's centroid. 'measured' needs walls facing
    >= 180 deg of directions with two opposite ones (else scale and shift cannot be separated)."""
    ring = S.ccw(rings[0])
    c = st.polygon_centroid(ring)
    xy = _wall_points(points, quality, 3.0, 16.0)
    k, d, sgn = _edge_assignment(xy, ring)
    A, y, w, used, normals = [], [], [], [], []
    for e in range(len(ring)):
        a, b = ring[e], ring[(e + 1) % len(ring)]
        L = float(np.linalg.norm(b - a))
        m = k == e
        if L < 3.0 or m.sum() < min_points:
            continue
        u = (b - a) / L
        s = (xy[m] - a) @ u
        mid = (s > 0.1 * L) & (s < 0.9 * L)
        if mid.sum() < min_points // 2:
            continue
        n = np.array([u[1], -u[0]])
        di = float(n @ (a - c))
        pos = di + float(np.median(sgn[m][mid]))
        A.append([di, n[0], n[1]])
        y.append(pos)
        w.append(math.sqrt(mid.sum()))
        normals.append(math.degrees(math.atan2(n[1], n[0])) % 360)
        used.append({"edge": e, "points": int(mid.sum()), "offset_m": round(pos - di, 2)})
    out: dict[str, Any] = {"edges": used, "centre": [round(float(v), 3) for v in c], "measured": False}
    if len(A) < 3:
        out["reason"] = f"{len(A)} edges with wall points (< 3)"
        return out
    A, y, w = np.array(A), np.array(y), np.array(w)
    sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    kk, tx, ty = (float(v) for v in sol)
    res = y - A @ sol
    ang = np.sort(np.array(normals))
    gaps = np.diff(np.r_[ang, ang[0] + 360])
    opposite = any(abs(((a1 - a2) % 360) - 180) <= 30 for a1 in normals for a2 in normals)
    covered = bool(gaps.max() <= 181.0 and opposite)          # walls facing >= 180 deg (1 deg slack)
    out.update({"k": round(kk, 4), "t": [round(tx, 3), round(ty, 3)], "residual_max_m": round(float(np.abs(res).max()), 3),
                "coverage_ok": covered, "directions_deg": [round(float(v), 1) for v in ang],
                "measured": covered})
    out["ticket"] = bool(covered and (abs(kk - 1) > FRAME_K_TOL or math.hypot(tx, ty) > FRAME_T_TOL_M))
    return out


def sfm_channel(model: S.StructureModel, points: np.ndarray, quality: np.ndarray | None = None, cell: float = 0.5,
                min_points: int = 50) -> dict[str, dict[str, Any]]:
    """Per built part: the SfM points' 0.98 height quantile over its exposed area vs the model's own surface there."""
    P = np.asarray(points, float)
    if quality is not None and len(quality) == len(P):
        P = P[np.asarray(quality)[:, 0] >= 3]
    built = {b.id: b for b in model.built}
    out: dict[str, dict[str, Any]] = {}
    for b in model.built:
        lo, hi = b.outer.min(axis=0), b.outer.max(axis=0)
        g = S._Grid(lo - cell, hi + cell, cell)
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
                r = max(1, int(round(1.5 / cell)))
                om = cv2.dilate(om.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            m &= ~om
        row: dict[str, Any] = {"exposed_m2": round(float(m.sum()) * cell * cell, 1), "points": 0, "q98": None, "model_q98": None}
        if m.sum() >= 2:
            sel = (P[:, 0] >= g.lo[0]) & (P[:, 1] >= g.lo[1]) & (P[:, 0] < g.lo[0] + g.nx * cell) & (P[:, 1] < g.lo[1] + g.ny * cell)
            q = P[sel]
            ix = ((q[:, 0] - g.lo[0]) / cell).astype(int)
            iy = ((q[:, 1] - g.lo[1]) / cell).astype(int)
            ok = m[np.clip(iy, 0, g.ny - 1), np.clip(ix, 0, g.nx - 1)] & (q[:, 2] > b.z0)
            row["points"] = int(ok.sum())
            if ok.sum() >= min_points:
                row["q98"] = round(float(np.quantile(q[ok, 2], 0.98)), 2)
            cen = g.centres()[m]
            z = built[b.id].surface_z(cen)
            z = z[np.isfinite(z)]
            if len(z):
                row["model_q98"] = round(float(np.quantile(z, 0.98)), 2)
        out[b.id] = row
    return out


@dataclass
class Decision:
    id: str
    state: str
    why: str
    votes: dict[str, Any] = field(default_factory=dict)
    verified: dict[str, bool] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    n_vis: int = 0
    n_sky: int = 0
    slender: bool = False
    would_fit: dict[str, float] | None = None
    evidence: list = field(default_factory=list)


def decide(model: S.StructureModel, profiles: dict[str, dict[str, Any]], sfm: dict[str, dict[str, Any]] | None = None,
           frame: dict[str, Any] | None = None, snap_scale: float = 1.0, auto_adopt: bool = False) -> dict[str, Decision]:
    """The §2.4.3 rules for every built part. profiles: {mask name: {part id: skyline.PartProfile}} (one or two
    masks; verification needs two)."""
    masks = list(profiles)
    sfm = sfm or {}
    frame_open = bool(frame and not frame.get("passed", True))
    scaled = abs(snap_scale - 1.0) > SNAP_SCALE_TOL
    out: dict[str, Decision] = {}
    for b in model.built:
        pid = b.id
        profs = [profiles[m][pid] for m in masks if pid in profiles[m]]
        if not profs:
            continue
        top, z0 = b.top, b.z0
        tol = max(2.0, 0.1 * top)
        minw = max(b.spec.min_width(), 0.1)
        hw = (top - z0) / minw
        slender = hw >= 2.0 and minw <= 10.0
        n_vis = min(p.n_vis for p in profs)
        n_sky = min(p.n_sky for p in profs)
        fwd0 = [p.fwd_px for p in profs]

        def pinfo(key: str, p) -> dict[str, Any]:
            pp = p.params[key]
            G = pp["grid"]
            k = G.index(pp["fr"]["argmin"])
            k1 = int(np.argmin(np.abs(np.array(G) - 1.0)))
            if key == "scale":
                L = minw / 2
                val, mapv = L * pp["fr"]["argmin"], L
                topv = pp["metres"][k]["top"]
            else:
                L = b.roof.height_m if key == "roof" else b.eave - z0
                q = "top" if key == "roof" else "eave"
                val, mapv = pp["metres"][k][q], pp["metres"][k1][q]
                topv = pp["metres"][k]["top"]
            step = G[1] - G[0]
            return {"val": round(float(val), 2), "map": round(float(mapv), 2), "top": topv, "bound": pp["fr"]["at_bound"],
                    "lo_bound": pp["fr"]["lo_bound"], "hi_bound": pp["fr"]["hi_bound"],
                    "ci90_m": round((pp["fr"]["p95"] - pp["fr"]["p05"]) * L, 2), "argmin": pp["fr"]["argmin"],
                    "robust": abs(pp["fr"]["argmin"] - pp["fo"]["argmin"]) <= step + 1e-9, "n_cams": pp["n_cams"]}

        keys = sorted(set.intersection(*[set(p.params) for p in profs])) if profs else []
        params = {key: {m: pinfo(key, profiles[m][pid]) for m in masks if pid in profiles[m]} for key in keys}
        verified = {}
        for key, pm in params.items():
            radius = key == "scale"
            lim = 0.75 if radius else 1.5
            vals = [v["val"] for v in pm.values()]
            mapv = next(iter(pm.values()))["map"]
            vt = max(1.0, 0.1 * mapv) if radius else max(2.0, 0.1 * mapv)
            verified[key] = bool(len(pm) >= 2 and not scaled and n_vis >= MIN_PHOTOS and n_sky >= MIN_PHOTOS
                                 and all(v["ci90_m"] <= lim and v["robust"] and not v["bound"] for v in pm.values())
                                 and max(vals) - min(vals) <= lim and all(abs(v - mapv) <= vt for v in vals))
        tops = [k for k in ("eave", "roof") if k in params]
        votes: dict[str, Any] = {}
        if tops:
            dmin = [min(params[k][m]["top"] - top for k in tops) for m in params[tops[0]]]
            dmax = [max(params[k][m]["top"] - top for k in tops) for m in params[tops[0]]]
            lo = [any(params[k][m]["lo_bound"] for k in tops) for m in params[tops[0]]]
            hi = [any(params[k][m]["hi_bound"] for k in tops) for m in params[tops[0]]]
            if n_sky >= MIN_PHOTOS and all(x < -tol or lb for x, lb in zip(dmin, lo)):
                votes["skyline-down"] = round(float(np.mean(dmin)), 1) if all(x < -tol for x in dmin) else "lower bound hit"
            if all(x > tol or hb for x, hb in zip(dmax, hi)):
                votes["skyline-up?"] = round(float(np.mean(dmax)), 1) if all(x > tol for x in dmax) else "upper bound hit"
        if n_vis >= MIN_PHOTOS and n_sky < MIN_PHOTOS and all((f or 0) >= 20 for f in fwd0):
            votes.setdefault("skyline-up?", "no sky above at the map value")
        s = sfm.get(pid, {})
        if s.get("q98") is not None and s.get("model_q98") is not None and s["q98"] - s["model_q98"] > tol:
            votes["sfm-up"] = round(s["q98"] - s["model_q98"], 1)
        up = [v for v in votes if v.endswith("-up")]
        down = [v for v in votes if v.endswith("-down")]
        n_up = len(up) + (1 if "skyline-up?" in votes and up else 0)
        n_down = len(down)
        stable = bool(tops) and all(all(v["ci90_m"] <= 1.5 and v["robust"] and not v["bound"] for v in params[k].values())
                                    and (len(params[k]) < 2 or abs(np.ptp([v["top"] for v in params[k].values()])) <= 1.5)
                                    for k in tops)
        would = None
        if n_vis < MIN_PHOTOS:
            state, why = S.STATE_MAP, f"model skyline in {n_vis} (< {MIN_PHOTOS}) photos at the map value: photos insufficient"
        elif n_up >= 2 or n_down >= 2:
            dirn = "up" if n_up >= 2 else "down"
            vals = []
            if dirn == "down" and isinstance(votes.get("skyline-down"), float):
                vals.append(top + votes["skyline-down"])
            if dirn == "up" and "sfm-up" in votes and s.get("q98") is not None:
                vals.append(s["q98"])
            agree = len(vals) >= 2 and (max(vals) - min(vals)) <= max(1.0, 0.05 * float(np.mean(vals)))
            ok = agree and stable and not slender and not scaled and not frame_open
            would = {"top": round(float(np.mean(vals)), 2)} if ok else None
            state = S.STATE_DISPUTED
            why = f"{dirn}: " + ", ".join(f"{k}={v}" for k, v in votes.items()) + ("; D15 would adopt" if ok else "; ticket")
        elif slender:
            state, why = S.STATE_MAP, f"slender (h/w {hw:.1f}, width {minw:.1f} m): single channels only recorded"
        elif any(verified.values()):
            state = S.STATE_VERIFIED
            why = "verified: " + ", ".join(k for k, v in verified.items() if v)
        else:
            bits = []
            if n_sky < MIN_PHOTOS:
                bits.append(f"sky above in {n_sky} (< {MIN_PHOTOS}) photos")
            if tops and not stable:
                bits.append("skyline profile unstable or at a bound")
            if len(masks) < 2:
                bits.append("one sky mask only: nothing can be verified")
            if scaled:
                bits.append(f"snap scale corrected x{snap_scale:.3f} (> {SNAP_SCALE_TOL:.0%}): not verified")
            if votes:
                bits.append("single channel: " + ", ".join(f"{k}={v}" for k, v in votes.items()))
            state, why = S.STATE_MAP, "; ".join(bits) or "no evidence"
        if frame_open and state in (S.STATE_VERIFIED, S.STATE_DISPUTED):
            why = f"frame ticket open: nothing verified or fitted; would be {state}: {why}"
            state = S.STATE_MAP
        evidence = []
        for m in params[tops[0]] if tops else []:
            evidence.append({"kind": "skyline", "mask": m, "skyline_photos": n_vis, "sky_above_photos": n_sky,
                             "top_at_min": round(float(min(params[k][m]["top"] for k in tops)), 2),
                             "vote": "down" if "skyline-down" in votes else ("up?" if "skyline-up?" in votes else None),
                             "snap_scale": round(snap_scale, 4)})
        if s.get("q98") is not None:
            evidence.append({"kind": "sfm-points", "q98": s["q98"], "model_q98": s.get("model_q98"),
                             "vote": "up" if "sfm-up" in votes else None, "snap_scale": round(snap_scale, 4)})
        out[pid] = Decision(pid, state, why, votes, verified, params, n_vis, n_sky, slender,
                            would, evidence)
    return out


def apply_decisions(model: S.StructureModel, decisions: dict[str, Decision], frame: dict[str, Any] | None = None,
                    auto_adopt: bool = False) -> S.StructureModel:
    """Write the decisions into the structure's parameters (states, evidence, tickets) and the frame result into
    its registry. With auto_adopt (D15, off by default) a disputed part whose conditions hold takes the photos'
    top as 'photo-fitted' (the map value kept beside it); otherwise values never change here."""
    model.tickets = [t for t in model.tickets if t.get("kind") not in ("part", "frame")]
    if frame is not None:
        model.frame = {k: v for k, v in frame.items() if k != "edges"}
        if not frame.get("passed", True) or (frame.get("similarity") or {}).get("ticket"):
            model.tickets.append({"id": f"T-{model.landmark}-frame", "kind": "frame", "open": True,
                                  "failing_edges": frame.get("failing", []), "similarity": frame.get("similarity")})
    n = 0
    for p in model.specs:
        d = decisions.get(p.id)
        if d is None:
            continue
        h = p.params["height"]
        h.evidence = list(d.evidence)
        if d.state == S.STATE_DISPUTED:
            n += 1
            tid = f"T-{model.landmark}-{n}"
            h.state, h.ticket = S.STATE_DISPUTED, tid
            model.tickets.append({"id": tid, "kind": "part", "part": p.id, "param": "height", "open": True,
                                  "votes": d.votes, "why": d.why,
                                  "strength": len([v for v in d.votes if not v.endswith("?")])})
            if auto_adopt and d.would_fit:
                h.map_value, h.value, h.source, h.state = h.value, d.would_fit["top"], st.PROV_FITTED, S.STATE_FITTED
        elif d.state == S.STATE_VERIFIED:
            if d.verified.get("eave") or d.verified.get("roof"):
                h.state = S.STATE_VERIFIED
            if d.verified.get("roof") and p.params["roof_height"].source not in (st.PROV_DEFAULT,):
                p.params["roof_height"].state = S.STATE_VERIFIED
        p.problems = [x for x in p.problems if not x.startswith("judge:")] + [f"judge: {d.state}: {d.why}"]
    return model


def check_decisions(model: S.StructureModel, decisions: dict[str, Decision] | None = None) -> dict[str, Any]:
    """G4 / G9: every photo-verified or photo-fitted parameter carries the evidence the rules need (two masks,
    >= 3 photos with sky above, snap scale recorded), none while a frame ticket is open, no slender part fitted, no
    fitted child of a part whose own change was not accepted, and no parameter from a forbidden source."""
    bad = []
    frame_open = any(t.get("kind") == "frame" and t.get("open") for t in model.tickets)
    states = {p.id: p.params["height"].state for p in model.specs if "height" in p.params}
    for p in model.specs:
        for k, v in p.params.items():
            if v.source in S.FORBIDDEN_SOURCES:
                bad.append({"id": p.id, "param": k, "why": f"forbidden source {v.source}"})
            if v.state not in (S.STATE_VERIFIED, S.STATE_FITTED):
                continue
            if frame_open:
                bad.append({"id": p.id, "param": k, "why": f"{v.state} while the frame ticket is open"})
            sky = [e for e in v.evidence if e.get("kind") == "skyline"]
            if len({e.get("mask") for e in sky}) < 2:
                bad.append({"id": p.id, "param": k, "why": f"{v.state} without two sky masks' evidence"})
            if any(e.get("sky_above_photos", 0) < MIN_PHOTOS for e in sky):
                bad.append({"id": p.id, "param": k, "why": f"{v.state} with < {MIN_PHOTOS} photos showing sky above"})
            if any("snap_scale" not in e for e in v.evidence):
                bad.append({"id": p.id, "param": k, "why": "evidence without its snap scale"})
            if v.state == S.STATE_FITTED:
                minw = p.min_width()
                if (p.top - p.z0) / max(minw, 0.1) >= 2.0 and minw <= 10.0:
                    bad.append({"id": p.id, "param": k, "why": "a slender part was fitted"})
                if p.parent is not None and states.get(p.parent) in (S.STATE_DISPUTED,):
                    bad.append({"id": p.id, "param": k, "why": "fitted on a parent whose change is not accepted"})
                if v.map_value is None:
                    bad.append({"id": p.id, "param": k, "why": "fitted without the map value kept"})
    if decisions:
        for pid, d in decisions.items():
            if d.state == S.STATE_VERIFIED and d.n_sky < MIN_PHOTOS:
                bad.append({"id": pid, "why": "verified with too few sky photos"})
    return {"violations": bad, "passed": not bad}


def apply_frame(model: S.StructureModel, k: float, t: tuple[float, float], centre: np.ndarray | None = None,
                note: str = "curator") -> S.StructureModel:
    """A curator-approved whole-building similarity (scale k about the outline's centre, then shift t) applied
    rigidly to every part and the main outline (never one edge or one part alone); outlines become 'curator'."""
    c = np.asarray(centre if centre is not None else st.polygon_centroid(S.ccw(model.main_rings[0])), float)
    tt = np.asarray(t, float)

    def tf(r):
        return c + (np.asarray(r, float) - c) * k + tt

    for p in model.specs:
        p.outer = tf(p.outer)
        p.holes = [tf(h) for h in p.holes]
        p.outline_source = st.PROV_CURATOR
        p.problems.append(f"{note}: whole-building similarity k={k:.4f}, t=({tt[0]:.2f}, {tt[1]:.2f}) m")
    model.main_rings = [tf(r) for r in model.main_rings]
    model.built = S.build_all(model.specs, model.params)
    model.frame = {**model.frame, "curator_similarity": {"k": k, "t": [float(tt[0]), float(tt[1])], "centre": c.tolist(),
                                                         "note": note}}
    model.tickets = [dict(tk, open=False, resolved_by=note) if tk.get("kind") == "frame" else tk for tk in model.tickets]
    return model


def decision_table(decisions: dict[str, Decision], model: S.StructureModel) -> list[dict[str, Any]]:
    rows = []
    for b in model.built:
        d = decisions.get(b.id)
        if d is None:
            continue
        rows.append({"id": b.id, "name": b.spec.name, "roof": b.roof.shape, "z0": round(b.z0, 2), "eave": round(b.eave, 2),
                     "top": round(b.top, 2), "min_width_m": round(b.spec.min_width(), 2), "slender": d.slender,
                     "n_vis": d.n_vis, "n_sky": d.n_sky,
                     "profile": {k: {m: f"{v['val']}{'*' if v['bound'] else ''}" for m, v in pm.items()} for k, pm in d.params.items()},
                     "votes": d.votes, "state": d.state, "why": d.why, "would_fit": d.would_fit})
    return rows
