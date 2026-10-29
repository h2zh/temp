"""Task #73 (geo-anchor stage 2, docs/STYLIZED_MODEL.md 3.2): behind StyleParams flags (stylize.SNAP_GATE_73) the snap
gate reads the yaw sigma from the model's GeoAnchor, refuses a window the outline's symmetry clipped or a model that
reports no yaw sigma, and refuses snapped cameras far from their own GPS; a snap stored with an earlier model version
(scripts/stylize_model.py --snap-from) is checked again before it is reused.

The adversarial alias cases are the round-3 verifier's (a model turned by a symmetry period about the outline or its
cameras, the reported sigma missing, large, or one the pipeline can report, GPS noise up to 18 m), ported here as
run_case. tests/fixtures/alias73.json lists every case of those three sets (2,688) that passed the gate wrongly with
the flags off at the time of the fix: 69."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from crowd3d import scene_io
from crowd3d import stylize as st
from crowd3d.anchor import GeoAnchor
from crowd3d.geo import LocalENU

REPO = Path(__file__).resolve().parent.parent
ON = dataclasses.replace(st.DEFAULT_PARAMS, **st.SNAP_GATE_73["on"])
OFF = dataclasses.replace(st.DEFAULT_PARAMS, **st.SNAP_GATE_73["off"])
ALIAS_CASES = json.loads((REPO / "tests" / "fixtures" / "alias73.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the verifier's case generator (adv_r3.run_case)


def regular_polygon(n: int, radius: float, phase_deg: float = 0.0) -> np.ndarray:
    a = np.radians(phase_deg + 360.0 * np.arange(n) / n)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], 1)


def rotm(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def rect(w: float, h: float, rot_deg: float = 0.0) -> np.ndarray:
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ rotm(rot_deg).T


def baptistery() -> np.ndarray:
    ts = json.loads((REPO / "eval" / "testsets" / "commons-ten-v1" / "testset.json").read_text(encoding="utf-8"))
    b = next(x for x in ts["buildings"] if x["slug"] == "florence-baptistery")
    fp, _ = scene_io.load_footprint_rings(REPO / "eval" / "testsets" / "commons-ten-v1" / b["osm_outline"]["file"],
                                          LocalENU(b["centre"]["lat"], b["centre"]["lon"]))
    return fp


OUTLINES = {"octagon20": lambda: regular_polygon(8, 20.0, 22.5), "octagon12": lambda: regular_polygon(8, 12.0),
            "square20": lambda: rect(20, 20, 5.0), "hexagon15": lambda: regular_polygon(6, 15.0),
            "rect34x13": lambda: rect(34, 13.1, 10.0), "12gon20": lambda: regular_polygon(12, 20.0),
            "12gon12": lambda: regular_polygon(12, 12.0), "16gon20": lambda: regular_polygon(16, 20.0),
            "pentagon15": lambda: regular_polygon(5, 15.0), "baptistery": baptistery}


def outward(fp: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = fp[k], fp[(k + 1) % len(fp)]
    d = b - a
    return a, d, np.array([d[1], -d[0]]) / np.linalg.norm(d)


def adv_walls(fp: np.ndarray, edges, rng, step: float = 0.4) -> np.ndarray:
    pts = []
    for k in edges:
        a, d, _ = outward(fp, k)
        for t in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(d) / step))):
            for h in rng.uniform(2.5, 18.0, 4):
                pts.append([*(a + t * d), h])
    return np.array(pts)


def run_case(fp, faces, yaw_err, about_kind, prior_sigma, shift_m, scale_err, cam_range, gps_noise, scale_sigma, rng,
             params) -> dict:
    """The truth: wall points on `faces` (SfM noise), a tree clump and a neighbour block that move with the model,
    five cameras per face, their EXIF GPS = truth + noise. The model: the truth turned by yaw_err about the outline's
    centre or the cameras' centroid, shifted and scaled. The gate as stylize composes it: snap_to_footprint with the
    product's SnapPrior, then registration_check with the snapped cameras. Wrong: rotation error > 5 deg, cameras
    > 1.5 m (median) or walls > 1.5 m (90th percentile) off."""
    true_walls = adv_walls(fp, faces, rng)
    true_walls[:, :2] += rng.normal(0, 0.15, (len(true_walls), 2))
    a, d, nrm = outward(fp, faces[0])
    n_c = max(10, int(0.08 * len(true_walls)))
    extra = [np.c_[(a + 0.5 * d + nrm * 6.0) + rng.normal(0, 1.5, (n_c, 2)), rng.uniform(2, 9, n_c)]]
    cen = st.polygon_centroid(fp)
    nb_c = cen + nrm * (np.linalg.norm(cen - a) + 30.0)
    extra.append(np.c_[nb_c + rng.uniform(-6, 6, (n_c, 2)), rng.uniform(3, 15, n_c)])
    true_all = np.vstack([true_walls] + extra)
    cams_true = []
    for k in faces:
        a_, d_, n_ = outward(fp, k)
        for _ in range(5):
            cams_true.append(a_ + rng.uniform(0.1, 0.9) * d_ + n_ * rng.uniform(*cam_range) + rng.normal(0, 3.0, 2))
    cams_true = np.array(cams_true)
    cam_z_true = 1.6 + rng.normal(0, 0.3, len(cams_true))
    about = {"walls": true_walls[:, :2].mean(0), "cams": cams_true.mean(0), "centre": cen}[about_kind]
    ang = rng.uniform(0, 2 * math.pi)
    sh = shift_m * np.array([math.cos(ang), math.sin(ang)])

    def err_tf(xy):
        return scale_err * (xy - about) @ rotm(yaw_err).T + about + sh

    model = true_all.copy()
    model[:, :2] = err_tf(true_all[:, :2])
    model[:, 2] *= scale_err
    cams_model = err_tf(cams_true)
    gps = cams_true + rng.normal(0, gps_noise, cams_true.shape)
    prior = st.SnapPrior(prior_sigma, cams_model, gps, scale_sigma, georegistered=True,
                         camera_xyz=np.c_[cams_model, cam_z_true * scale_err])
    snap = st.snap_to_footprint(model, fp, params, prior=prior)
    if not snap.accepted:
        return {"gate": False, "reason": snap.reason}
    s_cams = snap.scale * cams_model @ snap.rotation()[:2, :2].T + snap.shift
    reg = st.registration_check(st.apply_snap_points(snap, model), [fp], st.DEFAULT_PARAMS, cameras_xy=s_cams)
    nw = len(true_walls)
    rot_err = (yaw_err + snap.rotation_deg + 180.0) % 360.0 - 180.0
    snapped = st.apply_snap_points(snap, model)
    wall_p90 = float(np.percentile(np.linalg.norm(snapped[:nw, :2] - true_walls[:, :2], axis=1), 90))
    cam_med = float(np.median(np.linalg.norm(s_cams - cams_true, axis=1)))
    return {"gate": bool(reg["accepted"]), "reason": reg.get("reason") or "",
            "wrong": abs(rot_err) > 5.0 or cam_med > 1.5 or wall_p90 > 1.5}


def alias_case(row: dict, params: st.StyleParams) -> dict:
    """A fixture row run again: its set's seed (the exact-alias set drew the error's offset first)."""
    fp = st.ensure_ccw(OUTLINES[row["outline"]]())
    rng = np.random.default_rng(row["idx"] + {"exact": 0, "reported": 7, "gps": 1007}[row["set"]])
    if row["set"] == "exact":
        rng.uniform(-3, 3)
    return run_case(fp, list(range(row["faces"])), row["err"], row["about"], row["sigma"], row["shift"],
                    row["scale_err"], tuple(row["cams"]), row["gps"], row["scale_sigma"], rng, params)


def test_the_adversarial_aliases_pass_wrongly_without_the_flags_and_never_with_them():
    """One case of each set (exact alias with no sigma, a pentagon with the reported 20.3 deg, 12-18 m GPS noise)."""
    for name in ("exact", "reported", "gps"):
        row = next(r for r in ALIAS_CASES if r["set"] == name)
        off = alias_case(row, OFF)
        assert off["gate"] and off["wrong"], (row, off)
        on = alias_case(row, ON)
        assert not on["gate"], (row, on)
        assert "alias" in on["reason"] or "own GPS" in on["reason"], on["reason"]


@pytest.mark.slow
def test_every_recorded_adversarial_wrong_pass_is_refused_with_the_flags():
    """The whole list: 69 wrong passes with the flags off, 0 with them on."""
    assert len(ALIAS_CASES) == 69
    off = [alias_case(r, OFF) for r in ALIAS_CASES]
    assert sum(o["gate"] and o["wrong"] for o in off) == 69
    on = [alias_case(r, ON) for r in ALIAS_CASES]
    assert sum(o["gate"] for o in on) == 0, [(r, o) for r, o in zip(ALIAS_CASES, on) if o["gate"]]


# ---------------------------------------------------------------- each flag


def octagon_facade(turn: float = 9.0) -> np.ndarray:
    octagon = regular_polygon(8, 20.0, 22.5)
    rng = np.random.default_rng(0)
    pts = adv_walls(octagon, (5, 6, 7), rng)
    pts[:, :2] = (pts[:, :2] - pts[:, :2].mean(0)) @ rotm(turn).T + pts[:, :2].mean(0) + np.array([3.0, -2.0])
    return pts


def anchor_with(yaw_sigma: float, xy_sigma: float = 3.5, log_scale: float = 0.05) -> GeoAnchor:
    return GeoAnchor("job", "full", 1.0, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, [0.0, 0.0, 0.0],
                     {"yaw_deg": yaw_sigma, "log_scale": log_scale, "east_m": xy_sigma, "north_m": xy_sigma,
                      "up_m": 1.5, "tilt_deg": 2.0}, {}, {}, {"id": "test"})


def test_flags_off_change_nothing():
    """Off, a snap records exactly what it did before the flags existed (no new search or GPS keys), and a SnapPrior
    without an anchor records the same meta."""
    assert all(getattr(st.DEFAULT_PARAMS, k) == v for k, v in st.SNAP_GATE_73["off"].items())
    octagon = regular_polygon(8, 20.0, 22.5)
    prior = st.SnapPrior(yaw_sigma_deg=5.0, anchor=anchor_with(30.0))       # an anchor nobody reads with the flags off
    snap = st.snap_to_footprint(octagon_facade(), octagon, OFF, prior=prior)
    period = snap.search["symmetry_period_deg"]                            # ~37 deg: it fits itself from there on
    assert snap.accepted and snap.search["window_deg"] == pytest.approx(min(15.0, period / 2 - 5.0))
    assert not {"yaw_sigma_from", "window_clipped"} & set(snap.search)
    assert set(st.SnapPrior(5.0).to_meta()) == {"yaw_sigma_deg", "scale_sigma", "gps_cameras", "cameras", "georegistered"}


def test_the_yaw_window_comes_from_the_anchor_and_a_clipped_one_is_refused():
    """With snap_use_anchor the window is 3 x the anchor's yaw sigma (here 7.9 deg, the frozen floors' median), which on
    an octagon (P about 37 deg) is clipped at P/2 - 5 deg: the alias one period away lies inside 3 sigma, so the snap
    is refused. The legacy sigma (4 deg) on the same model is not clipped: accepted."""
    octagon = regular_polygon(8, 20.0, 22.5)
    pts = octagon_facade()
    prior = st.SnapPrior(yaw_sigma_deg=4.0, anchor=anchor_with(7.9))
    legacy = st.snap_to_footprint(pts, octagon, dataclasses.replace(ON, snap_use_anchor=False), prior=prior)
    assert legacy.accepted and legacy.search["window_clipped"] is False and "yaw_sigma_from" not in legacy.search
    anchored = st.snap_to_footprint(pts, octagon, ON, prior=prior)
    assert anchored.search["yaw_sigma_from"] == "anchor" and anchored.search["yaw_sigma_deg"] == 7.9
    assert anchored.search["window_clipped"] is True
    assert not anchored.accepted and "alias one period away" in anchored.reason
    # the clipped window alone (no anchor): a reported 5 deg is clipped too; off, the clipped window is searched
    five = st.SnapPrior(yaw_sigma_deg=5.0)
    assert not st.snap_to_footprint(pts, octagon, ON, prior=five).accepted
    assert st.snap_to_footprint(pts, octagon, OFF, prior=five).accepted
    # a box (P = 180 deg) is not clipped by the anchor's sigma
    box = rect(30.0, 16.0)
    rng = np.random.default_rng(1)
    walls = adv_walls(box, (0, 1), rng)
    walls[:, :2] = walls[:, :2] @ rotm(4.0).T + np.array([1.0, 1.5])
    wide = st.snap_to_footprint(walls, box, ON, prior=st.SnapPrior(4.0, scale_sigma=0.01, anchor=anchor_with(7.9)))
    assert wide.search["window_deg"] == pytest.approx(23.7) and wide.search["window_clipped"] is False


def test_a_pipeline_model_without_a_yaw_sigma_is_refused_and_a_scene_without_a_prior_is_not():
    octagon = regular_polygon(8, 20.0, 22.5)
    pts = octagon_facade()
    missing = st.snap_to_footprint(pts, octagon, ON, prior=st.SnapPrior(yaw_sigma_deg=None))
    assert not missing.accepted and "reports no yaw sigma" in missing.reason
    assert not st.snap_to_footprint(pts, octagon, ON, prior=st.SnapPrior(yaw_sigma_deg=float("nan"))).accepted
    # a reference scene (no SnapPrior: GT georeferencing) keeps the default window, clipped, as before
    ref = st.snap_to_footprint(pts, octagon, ON)
    period = ref.search["symmetry_period_deg"]
    assert ref.accepted and ref.search["window_deg"] == pytest.approx(period / 2 - 5.0)
    assert "window_clipped" not in ref.search


def test_snapped_cameras_far_from_their_own_gps_are_refused():
    """Right walls, but every photo's GPS lies 40 deg round the walls from where its camera stood (an alias's
    cameras): each GPS is as far from the walls as its camera, so the depth check agrees; the median check (20 m)
    does not. An anchor whose horizontal sigma at the cameras is larger widens the limit to 3 x that sigma."""
    box = rect(30.0, 16.0)
    rng = np.random.default_rng(2)
    walls = adv_walls(box, (0, 1), rng)
    at = np.median(walls[:, :2], axis=0)
    cams = np.array([[x, -40.0] for x in np.linspace(-12, 12, 6)] + [[45.0, y] for y in np.linspace(-6, 6, 4)])
    turn, shift = 2.0, np.array([1.0, -1.5])
    walls[:, :2] = walls[:, :2] @ rotm(turn).T + shift
    cams_model = cams @ rotm(turn).T + shift
    prior = dict(yaw_sigma_deg=4.0, camera_xy=cams_model, scale_sigma=0.01, georegistered=True,
                 camera_xyz=np.c_[cams_model, np.full(len(cams), 1.6)])
    near = st.snap_to_footprint(walls, box, ON, prior=st.SnapPrior(gps_xy=cams, **prior))
    assert near.accepted, near.reason
    assert near.gps["camera_gps_median_m"] < 1.0 and near.gps["camera_gps_limit_m"] == 20.0
    round_walls = (cams - at) @ rotm(40.0).T + at
    far = st.snap_to_footprint(walls, box, ON, prior=st.SnapPrior(gps_xy=round_walls, **prior))
    assert far.gps["agrees"] and far.gps["camera_gps_median_m"] > 20.0
    assert not far.accepted and "from their own GPS" in far.reason
    assert st.snap_to_footprint(walls, box, OFF, prior=st.SnapPrior(gps_xy=round_walls, **prior)).accepted
    loose = st.snap_to_footprint(walls, box, ON, prior=st.SnapPrior(gps_xy=round_walls, anchor=anchor_with(4.0, 15.0),
                                                                    **prior))
    assert loose.gps["camera_gps_limit_m"] >= 45.0 and loose.accepted, loose.reason


# ---------------------------------------------------------------- --snap-from: a stored snap is checked again


def test_a_stored_snap_is_reused_only_when_this_codes_gate_accepts_it():
    """scripts/stylize_model.py --snap-from reused an earlier version's snap without any of the later checks (README,
    task #73). Now it is reused (exactly) only when this code's snap lands within snap_ambiguity_m of it."""
    box = rect(30.0, 16.0)
    rng = np.random.default_rng(3)
    walls = adv_walls(box, (0, 1, 2), rng)
    walls[:, :2] = walls[:, :2] @ rotm(3.0).T + np.array([1.0, -1.0])
    prior = st.SnapPrior(4.0, scale_sigma=0.01)
    fresh = st.snap_to_footprint(walls, box, prior=prior)
    assert fresh.accepted, fresh.reason
    meta = json.loads(json.dumps(fresh.to_meta()))
    meta["shift_m"] = [v + 0.2 for v in meta["shift_m"]]                      # stored by an earlier, close version
    stored = st.FootprintSnap.from_meta(meta)
    assert not stored.verified
    kept = st.recheck_stored_snap(stored, walls, box, [], prior)
    assert kept.accepted and kept.verified and np.allclose(kept.shift, meta["shift_m"])     # its exact placement
    far = dict(meta, rotation_deg=meta["rotation_deg"] + 45.0)                  # an earlier gate's alias
    moved = st.recheck_stored_snap(st.FootprintSnap.from_meta(far), walls, box, [], prior)
    assert not moved.accepted and "places the points" in moved.reason and moved.rotation_deg == 0.0
    refused = st.recheck_stored_snap(stored, walls, box, [], st.SnapPrior(4.0, scale_sigma=0.5))   # scale now refused
    assert not refused.accepted and "gate refuses the snap" in refused.reason
    # through register_scene: the registration is refused with that reason, the GLB is clay
    scene = st.SceneInput("box", box, walls, np.full((len(walls), 3), 200, np.uint8), [], snap_override=st.FootprintSnap
                          .from_meta(far), snap_prior=prior)
    reg = st.register_scene(scene)
    assert not reg.photo_ok and "stored snap is not reused" in reg.registration["reason"]
