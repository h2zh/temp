"""crowd3d.lines3d (the 3-D line evidence channel) on synthetic scenes: a known structure's faces are rendered (flat
shading) into synthetic posed photos, its visible surfaces sampled as the SfM points; the channel must recover the
eave, apex and pitch the truth has, vote both ways on a wrong map, refuse with too few photos, say nothing about
sides no photo saw, and feed crowd3d.judge as a third channel (two channels agreeing to verify; v0 changes nothing)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from crowd3d import judge as J
from crowd3d import lines3d as L3
from crowd3d import skyline as K
from crowd3d import structure as S
from crowd3d import stylize as st

EAVE, APEX, R_OCT = 18.0, 30.0, 13.0


def look_at(center, target, w=640, h=480, f=520.0, name="cam") -> st.PosedCamera:
    c = np.asarray(center, float)
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return st.PosedCamera(name, w, h, f, f, w / 2, h / 2, np.stack([right, down, fwd]), c)


def ring(bearings_deg, radii=(45.0, 70.0), heights=(1.6, 6.0), target=(0.0, 0.0, 16.0)) -> list[st.PosedCamera]:
    """Photos around the landmark at alternating distances and heights (ground photos: line triangulation needs
    some spread, as crowd photos have)."""
    out = []
    for i, b in enumerate(bearings_deg):
        r, h = radii[i % len(radii)], heights[(i // len(radii)) % len(heights)]
        out.append(look_at([r * math.sin(math.radians(b)), r * math.cos(math.radians(b)), h], target, name=f"c{i}"))
    return out


def _faces(model: S.StructureModel):
    return [(tri, f.normal) for b in model.built for f in b.faces(base=False) for tri in f.triangles()]


def render(model: S.StructureModel, cams: list[st.PosedCamera], seed: int = 0):
    """Flat-shaded photos of the truth (painter's order, back faces culled) and per-pixel face ids; the photos are
    attached to the cameras."""
    rng = np.random.default_rng(seed)
    faces = _faces(model)
    light = np.array([0.4, -0.3, 0.85]) / np.linalg.norm([0.4, -0.3, 0.85])
    ids = []
    for cam in cams:
        H, W = cam.height, cam.width
        v, u = np.mgrid[0:H, 0:W]
        rays = np.stack([(u + 0.5 - cam.cx) / cam.fx, (v + 0.5 - cam.cy) / cam.fy, np.ones(u.shape)], -1) @ cam.rot
        img = np.where((rays[..., 2] > 0)[..., None], np.array([200, 215, 235], np.uint8),
                       np.array([95, 100, 90], np.uint8)).astype(np.uint8)
        fid = np.full((H, W), -1, np.int32)
        order = []
        for k, (tri, n) in enumerate(faces):
            pc = (tri - cam.center) @ cam.rot.T
            if n @ (cam.center - tri.mean(0)) <= 1e-6 or (pc[:, 2] < 0.5).any():
                continue
            order.append((-float(np.linalg.norm(tri.mean(0) - cam.center)), k, pc))
        for _, k, pc in sorted(order, key=lambda o: (o[0], o[1])):
            n = faces[k][1]
            g = int(np.clip(70 + 150 * max(0.0, n @ light) + 25 * (abs(n[2]) > 0.2), 0, 255))
            px = np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], 1)
            q = np.round(px * 16).astype(np.int32)
            cv2.fillConvexPoly(img, q, (g, int(g * 0.95), int(g * 0.88)), lineType=cv2.LINE_AA, shift=4)
            cv2.fillConvexPoly(fid, q, int(k), lineType=cv2.LINE_8, shift=4)
        cam.image = np.clip(img.astype(float) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        ids.append(fid)
    return ids


def sfm_points(model: S.StructureModel, cams: list[st.PosedCamera], ids, step: float = 0.6, seed: int = 0):
    """Surface samples seen by >= 2 photos (track = number of photos), 2 cm noise: what SfM would triangulate."""
    rng = np.random.default_rng(seed)
    faces = _faces(model)
    P, F = [], []
    for k, (tri, _) in enumerate(faces):
        a, b, c = tri
        m = max(1, int(0.5 * np.linalg.norm(np.cross(b - a, c - a)) / step ** 2))
        r1, r2 = rng.random(m), rng.random(m)
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        P.append(a + r1[:, None] * (b - a) + r2[:, None] * (c - a))
        F.append(np.full(m, k))
    P, F = np.vstack(P), np.concatenate(F)
    seen = np.zeros(len(P), int)
    for cam, fid in zip(cams, ids):
        uv, z = cam.project(P)
        ok = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] < cam.height)
        u = np.clip(uv[:, 0].astype(int), 0, cam.width - 1)
        v = np.clip(uv[:, 1].astype(int), 0, cam.height - 1)
        seen += ok & (fid[v, u] == F)
    keep = seen >= 2
    return P[keep] + rng.normal(0, 0.02, (int(keep.sum()), 3)), np.stack([seen[keep], np.full(int(keep.sum()), 5.0)], 1)


def octagon(r: float = R_OCT) -> np.ndarray:
    return np.array([[r * math.cos(math.radians(22.5 + 45 * k)), r * math.sin(math.radians(22.5 + 45 * k))] for k in range(8)])


def oct_model(eave: float = EAVE, top: float = APEX) -> S.StructureModel:
    o = octagon()
    spec = S.PartSpec("w1", "yes", o.copy(), [], {"height": f"{top:g}", "roof:shape": "pyramidal", "roof:height": f"{top - eave:g}"})
    return S.build_structure("synthetic-oct", [spec], [o.copy()], {})


@pytest.fixture(scope="module")
def all_round():
    truth = oct_model()
    cams = ring(np.linspace(0, 360, 24, endpoint=False))
    ids = render(truth, cams)
    P, q = sfm_points(truth, cams, ids)
    return truth, cams, P, q


def test_lines_recover_the_eave_apex_and_pitch_of_a_known_roof(all_round):
    truth, cams, P, q = all_round
    res = L3.lines_channel(oct_model(), cams, P, q)
    p = res.parts["w1"]
    m = p.measurements
    assert set(m) >= {"eave", "top", "apex_xy", "pitch"}, p.refused
    assert m["eave"]["value"] == pytest.approx(EAVE, abs=0.3) and m["eave"]["n_photos"] >= 6
    assert len(m["eave"]["edges_observed"]) >= 6                 # photos all round: (nearly) every eave edge
    assert m["top"]["value"] == pytest.approx(APEX, abs=0.4) and m["top"]["how"] == "apex"
    assert m["apex_xy"]["value"] <= 0.3
    hip = math.degrees(math.atan2(APEX - EAVE, R_OCT))           # the hips' true slope (to the octagon's corners)
    assert m["pitch"]["value"] == pytest.approx(hip, abs=1.5) and m["pitch"]["map"] == pytest.approx(hip, abs=0.1)
    assert all(r["vote"] is None for r in m.values())            # a right map: no vote
    assert all(r["snap_scale"] == 1.0 and not r["scale_corrected"] for r in m.values())
    assert res.stats["cameras"] == 24 and res.timings_s["total_s"] >= 0 and res.detector in ("lsd", "fld")
    # lines are evidence about the part only: every recovered line lies on the truth's surface
    tris = np.array([t for t, _ in _faces(truth)])
    mid = 0.5 * (res.lines.A + res.lines.B)
    assert len(mid) and np.median([_dist_to_tris(x, tris) for x in mid]) < 0.15


def _dist_to_tris(x: np.ndarray, tris: np.ndarray) -> float:
    """Distance from a point to the nearest of the triangles (inside the face: to its plane; else to its edges)."""
    best = np.inf
    for a, b, c in tris:
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        p = x - ((x - a) @ n) * n
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        v, w = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        if v >= 0 and w >= 0 and v + w <= 1:
            best = min(best, abs(float((x - a) @ n)))
            continue
        for e0, e1 in ((a, b), (b, c), (c, a)):
            t = np.clip((x - e0) @ (e1 - e0) / ((e1 - e0) @ (e1 - e0)), 0, 1)
            best = min(best, float(np.linalg.norm(e0 + t * (e1 - e0) - x)))
    return best


def test_lines_vote_both_ways_on_a_wrong_map(all_round):
    """The same photos against a map 4 m too low and one 6 m too high: the lines measure the truth and vote up /
    down (they see heights directly); the map's values are untouched."""
    _, cams, P, q = all_round
    low = L3.lines_channel(oct_model(15.0, 26.0), cams, P, q).parts["w1"].measurements
    assert low["top"]["value"] == pytest.approx(APEX, abs=0.4) and low["top"]["vote"] == "up"
    assert low["eave"]["value"] == pytest.approx(EAVE, abs=0.3) and low["eave"]["vote"] == "up"
    high = L3.lines_channel(oct_model(21.6, 36.0), cams, P, q).parts["w1"].measurements
    assert high["top"]["value"] == pytest.approx(APEX, abs=0.4) and high["top"]["vote"] == "down"
    assert high["eave"]["value"] == pytest.approx(EAVE, abs=0.3) and high["eave"]["vote"] == "down"
    assert high["top"]["map"] == 36.0 and high["eave"]["map"] == 21.6


def test_refused_with_fewer_than_three_photos():
    truth = oct_model()
    for bearings in ([30.0, 40.0], [30.0, 40.0, 50.0]):
        cams = ring(bearings)
        ids = render(truth, cams)
        P, q = sfm_points(truth, cams, ids)
        p = L3.lines_channel(oct_model(), cams, P, q).parts["w1"]
        assert not p.measurements and p.refused["eave"] and p.refused["top"], (bearings, p.refused)
    # no SfM points: no depth prior, no support -> refused
    p = L3.lines_channel(oct_model(), cams, None).parts["w1"]
    assert not p.measurements and "no SfM points" in p.refused["eave"]


def test_no_lines_and_no_measurement_on_sides_no_photo_saw():
    """Photos from the north-east only (bearings 20-70 deg): lines exist only on faces those photos face, the eave
    is measured on the edges facing them and the others are listed as unobserved; the apex (seen from one side only)
    is refused."""
    truth = oct_model()
    cams = ring(np.linspace(20, 70, 12))
    ids = render(truth, cams)
    P, q = sfm_points(truth, cams, ids)
    res = L3.lines_channel(oct_model(), cams, P, q)
    p = res.parts["w1"]
    assert len(res.lines) > 0
    C = np.array([c.center for c in cams])
    faces = _faces(truth)
    for a, b in zip(res.lines.A, res.lines.B):
        x = 0.5 * (a + b)
        near = [n for t, n in faces if _dist_to_tris(x, t[None]) < 0.3]
        assert near and any(((C - x) @ n > 0).any() for n in near), x       # on a face some photo faces
    eave = p.measurements["eave"]
    walls = {w.edge: w.normal()[:2] for w in truth.built[0].walls}
    seen_dir = np.array([math.sin(math.radians(45)), math.cos(math.radians(45))])
    assert eave["edges_observed"] and all(walls[e] @ seen_dir > 0 for e in eave["edges_observed"])
    far = [e for e, n in walls.items() if n @ seen_dir < -0.5]
    assert far and set(far) <= set(eave["edges_unobserved"]) and not set(far) & set(eave["edges_observed"])
    assert "top" not in p.measurements and p.refused["top"]


def test_the_snap_scale_guard_marks_lines_and_they_cannot_decide_alone(all_round):
    """Under a snap scale correction > 3 % every record is scale_corrected; the judge counts a lines vote only beside
    another strong vote in its direction (never alone, never with the weak skyline-up? alone)."""
    _, cams, P, q = all_round
    recs = L3.lines_channel(oct_model(15.0, 26.0), cams, P, q, snap_scale=1.10).evidence()
    assert all(r["scale_corrected"] and r["snap_scale"] == 1.1 for r in recs["w1"])
    model = oct_model(15.0, 26.0)
    prof = _flat_profile(model, "w1")
    d = J.decide(model, {"A": prof, "B": prof}, lines=recs, snap_scale=1.10)["w1"]
    assert "lines-up" in d.votes and d.state == S.STATE_MAP and d.ticket is None and "cannot decide alone" in d.why
    # the judge stamps its own snap scale on the records it keeps (a record made without it is still marked)
    unmarked = {"w1": [dict(r, snap_scale=1.0, scale_corrected=False) for r in recs["w1"]]}
    d2 = J.decide(model, {"A": prof, "B": prof}, lines=unmarked, snap_scale=1.10)["w1"]
    kept = [e for e in d2.evidence if e["kind"] == "lines3d"]
    assert kept and all(e["scale_corrected"] and e["snap_scale"] == 1.1 for e in kept)
    d = J.decide(model, {"A": prof, "B": prof}, lines=recs, snap_scale=1.0)["w1"]          # unscaled: a low ticket
    assert d.state == S.STATE_MAP and d.ticket == "low" and d.measured["lines3d"] == pytest.approx(APEX, abs=0.4)
    sfm = {"w1": {"q98": 29.8, "model_q98": 25.9}}
    d = J.decide(model, {"A": prof, "B": prof}, sfm, lines=recs, snap_scale=1.10)["w1"]    # beside sfm-up: counted
    assert d.state == S.STATE_DISPUTED and {"lines-up", "sfm-up"} <= set(d.votes), d.why
    assert set(d.agreeing) == {"sfm-points", "lines3d"} and d.agreed == pytest.approx(29.9, abs=0.5)
    assert d.would_fit is None                                                             # scaled: D15 never adopts


def _flat_profile(model: S.StructureModel, pid: str) -> dict:
    """A skyline profile with no opinion (flat costs, every photo sees the part with sky above): isolates the
    3-D line channel in the judge."""
    b = model.built_part(pid)
    prof = K.PartProfile(pid, n_vis=12, n_sky=12, fwd_px=1.0, map_top=b.top, map_eave=b.eave, z0=b.z0)
    for key in K.params_of(b):
        G = [float(g) for g in K.grid_for(b, key)]
        metres = [{"top": round(b.top + ((g - 1.0) * b.roof.height_m if key == "roof" else
                                         (g - 1.0) * (b.eave - b.z0) if key == "eave" else 0.0), 2),
                   "eave": round(b.eave + ((g - 1.0) * (b.eave - b.z0) if key == "eave" else 0.0), 2)} for g in G]
        flat = {"argmin": 1.0, "p05": 1.0, "p95": 1.0, "at_bound": False, "lo_bound": False, "hi_bound": False,
                "curve": [0.0] * len(G)}
        prof.params[key] = {"grid": G, "n_cams": 12, "fr": dict(flat), "fo": dict(flat), "metres": metres}
    return {pid: prof}


# ---------------------------------------------------------------- the channel in the judge, end to end

BOX = np.array([[-15.0, -10.0], [15.0, -10.0], [15.0, 10.0], [-15.0, 10.0]])
TOWER = np.array([[-6.0, -6.0], [6.0, -6.0], [6.0, 6.0], [-6.0, 6.0]])


def tower_model(top: float) -> S.StructureModel:
    specs = [S.PartSpec("w1", "yes", BOX.copy(), [], {"height": "12"}),
             S.PartSpec("w2", "yes", TOWER.copy(), [], {"height": f"{top:g}"})]
    return S.build_structure("synthetic", specs, [BOX.copy()], {})


def ideal_sky(model: S.StructureModel, cams: list[st.PosedCamera]) -> dict[int, np.ndarray]:
    tris = np.concatenate([b.triangles() for b in model.built])
    out = {}
    for i, cam in enumerate(cams):
        v, u = np.mgrid[0:cam.height, 0:cam.width]
        rays = np.stack([(u + 0.5 - cam.cx) / cam.fx, (v + 0.5 - cam.cy) / cam.fy, np.ones(u.shape)], -1) @ cam.rot
        out[i] = (rays[..., 2] > 0) & (K.raster_mask(cam, tris, (cam.height, cam.width)) == 0)
    return out


@pytest.fixture(scope="module")
def tower_scene():
    truth = tower_model(30.0)
    cams = ring(np.linspace(0, 360, 16, endpoint=False), radii=(55.0, 80.0), target=(0.0, 0.0, 14.0))
    ids = render(truth, cams)
    P, q = sfm_points(truth, cams, ids)
    return truth, cams, P, q, ideal_sky(truth, cams)


def test_the_judge_verifies_only_when_skyline_and_lines_agree_and_disputes_with_their_values(tower_scene):
    truth, cams, P, q, sky = tower_scene
    model = tower_model(30.0)
    lines = L3.lines_channel(model, cams, P, q)
    eave = lines.parts["w2"].measurements["eave"]
    assert eave["value"] == pytest.approx(30.0, abs=0.3) and eave["vote"] is None
    prof = {"w2": K.SkylineProblem(model, cams, sky).profile_part("w2")}
    d = J.decide(model, {"A": prof, "B": prof}, frame={"passed": True}, lines=lines.evidence())
    assert d["w2"].state == S.STATE_VERIFIED and d["w2"].agreement["eave"]["agree"], d["w2"].why
    J.apply_decisions(model, d, frame={"passed": True})
    assert model.part("w2").params["height"].state == S.STATE_VERIFIED and J.check_decisions(model, d)["passed"]
    # the publish gate rechecks the agreement from the records: a forged lines value 4 m off is caught
    h = model.part("w2").params["height"]
    forged = next(e for e in h.evidence if e["kind"] == "lines3d" and e["parameter"] == "eave")
    forged["value"] = 34.0
    assert any("second channel" in v["why"] for v in J.check_decisions(model)["violations"])
    forged["value"], forged["scale_corrected"] = eave["value"], True
    assert any("second channel" in v["why"] for v in J.check_decisions(model)["violations"])
    h.evidence = [e for e in h.evidence if e["kind"] != "agreement"]
    forged["scale_corrected"] = False
    assert any("second channel" in v["why"] for v in J.check_decisions(model)["violations"])

    # the map 4 m too low: the lines say up (a measured value), the skyline 'up?'; two channels -> a dispute whose
    # ticket carries the measured values; v0 changes nothing
    low = tower_model(26.0)
    ll = L3.lines_channel(low, cams, P, q)
    assert ll.parts["w2"].measurements["eave"]["vote"] == "up"
    prof = {"w2": K.SkylineProblem(low, cams, sky).profile_part("w2")}
    sfm = J.sfm_channel(low, P, q)
    d = J.decide(low, {"A": prof, "B": prof}, sfm, frame={"passed": True}, lines=ll.evidence())
    assert d["w2"].state == S.STATE_DISPUTED and "lines-up" in d["w2"].votes, d["w2"].why
    assert d["w2"].measured["lines3d"] == pytest.approx(30.0, abs=0.3)
    J.apply_decisions(low, d, frame={"passed": True})
    h = low.part("w2").params["height"]
    assert h.state == S.STATE_DISPUTED and h.value == 26.0 and low.built_part("w2").top == pytest.approx(26.0)
    tk = next(t for t in low.tickets if t["kind"] == "part" and t["part"] == "w2")
    assert tk["measured"]["lines3d"] == pytest.approx(30.0, abs=0.3) and tk["map_value"] == 26.0
    assert "lines3d" in tk["agreeing_channels"] and tk["agreed_value"] == pytest.approx(30.0, abs=0.6)
    assert J.check_decisions(low, d)["passed"]


def test_a_verified_pitched_roof_carries_its_evidence_on_both_parameters():
    """Skyline and 3-D lines agreeing on the eave and the apex verify the height and the roof height; both parameters
    carry the evidence, so the publish gate (G4) passes (the Baptistery after its frame ticket hit a roof height
    marked verified without any evidence)."""
    model = oct_model()
    prof = _flat_profile(model, "w1")
    recs = {"w1": [{"kind": "lines3d", "parameter": "top", "value": 29.8, "interval": [29.6, 30.0], "n_photos": 9,
                    "vote": None, "implied_top": 29.8},
                   {"kind": "lines3d", "parameter": "eave", "value": 18.3, "interval": [18.1, 18.5], "n_photos": 9,
                    "vote": None, "implied_top": 30.3}]}
    d = J.decide(model, {"A": prof, "B": prof}, frame={"passed": True}, lines=recs)
    assert d["w1"].state == S.STATE_VERIFIED and d["w1"].verified["roof"] and d["w1"].verified["eave"], d["w1"].why
    J.apply_decisions(model, d, frame={"passed": True})
    p = model.part("w1")
    assert p.params["height"].state == p.params["roof_height"].state == S.STATE_VERIFIED
    assert any(e["kind"] == "agreement" and e["parameter"] == "top" for e in p.params["roof_height"].evidence)
    assert J.check_decisions(model, d)["passed"], J.check_decisions(model, d)
    p.params["roof_height"].evidence = []                    # what apply_decisions wrote before
    assert not J.check_decisions(model)["passed"]


def test_a_contrary_eave_measurement_blocks_verification():
    """Nothing is verified while a 3-D line record votes against (here the eave, with the top agreeing)."""
    model = oct_model()
    prof = _flat_profile(model, "w1")
    recs = {"w1": [{"kind": "lines3d", "parameter": "top", "value": 30.1, "interval": [29.9, 30.3], "n_photos": 9,
                    "vote": None, "implied_top": 30.1},
                   {"kind": "lines3d", "parameter": "eave", "value": 21.0, "interval": [20.8, 21.2], "n_photos": 9,
                    "vote": "up", "implied_top": 33.0}]}
    d = J.decide(model, {"A": prof, "B": prof}, lines=recs)["w1"]
    assert d.verified.get("roof") and d.state == S.STATE_MAP and "eave:up" in d.why, d.why


def _hlines(z_list, y=-10.0, x0=-14.0, x1=14.0, photos=range(8)) -> L3.Lines3D:
    """Horizontal 3-D lines along the south wall (y = -10) of the box, one per height."""
    A = np.array([[x0, y, z] for z in z_list], float)
    B = np.array([[x1, y, z] for z in z_list], float)
    n = len(z_list)
    return L3.Lines3D(A, B, np.full(n, len(list(photos))), [sorted(photos)] * n, np.ones(n), np.zeros(n))


def test_a_string_course_below_the_wall_top_is_not_taken_for_the_eave():
    """The highest line cluster on a wall is the eave only if the wall's own SfM points do not rise above it: a string
    course at 10 m on an 18 m wall (its top edge not detected) would read 'down'; with the wall's points it is not
    measured at all. A single line never votes."""
    box = S.build_structure("b", [S.PartSpec("w1", "yes", BOX.copy(), [], {"height": "18"})], [BOX.copy()], {})
    b = box.built[0]
    lines = _hlines([10.0])
    m = L3.measure_part(b, box, lines).measurements["eave"]                  # no points: the one line, no vote
    assert m["value"] == pytest.approx(10.0) and m["vote"] is None and "one line" in m["vote_withheld"]
    lines2 = _hlines([10.0, 10.1])
    assert L3.measure_part(b, box, lines2).measurements["eave"]["vote"] == "down"
    gx, gz = np.meshgrid(np.linspace(-14, 14, 40), np.linspace(2, 17.8, 30))
    wall = np.c_[gx.ravel(), np.full(gx.size, -10.0), gz.ravel()]           # the wall's points reach 17.8 m
    p = L3.measure_part(b, box, lines2, points=wall)
    assert "eave" not in p.measurements and "not the wall top" in p.refused["eave"], p.refused
    p = L3.measure_part(b, box, _hlines([10.0, 17.9, 18.0]), points=wall)    # the real top edge seen: measured
    assert p.measurements["eave"]["value"] == pytest.approx(18.0, abs=0.15) and p.measurements["eave"]["vote"] is None
    p = L3.measure_part(b, box, _hlines([17.9, 18.0], photos=range(2)), points=wall)    # lines from 2 photos only
    assert "eave" not in p.measurements and p.refused["eave"]


def test_a_vote_needs_its_interval_clear_of_the_map_value():
    assert L3._vote(25.0, [24.0, 26.0], 18.0, 2.0) == "up" and L3._vote(12.0, [11.0, 13.0], 18.0, 2.0) == "down"
    assert L3._vote(25.0, [17.0, 26.0], 18.0, 2.0) is None and L3._vote(12.0, [11.0, 18.5], 18.0, 2.0) is None
    assert L3._vote(19.5, [19.4, 19.6], 18.0, 2.0) is None                  # inside the tolerance


def test_an_apex_seen_from_one_direction_only_is_refused():
    """Sloped lines that meet at the map's apex but all come from one side (spread < 45 deg) do not measure it."""
    model = oct_model()
    b = model.built[0]
    az = np.radians([0.0, 8.0, 16.0, 24.0, 32.0])
    low = np.stack([10.0 * np.sin(az), 10.0 * np.cos(az), np.full(len(az), EAVE + 2.0)], 1)
    apex = np.array([0.0, 0.0, APEX])
    n = len(az)
    lines = L3.Lines3D(low, np.repeat(apex[None], n, 0) - 0.05 * (np.repeat(apex[None], n, 0) - low), np.full(n, 8),
                       [list(range(8))] * n, np.ones(n), np.zeros(n))
    p = L3.measure_part(b, model, lines)
    assert "top" not in p.measurements and "one direction" in p.refused["top"], p.refused
    az = np.radians([0.0, 90.0, 180.0, 270.0, 45.0])                         # the same lines from all round: measured
    low = np.stack([10.0 * np.sin(az), 10.0 * np.cos(az), np.full(len(az), EAVE + 2.0)], 1)
    lines = L3.Lines3D(low, np.repeat(apex[None], n, 0) - 0.05 * (np.repeat(apex[None], n, 0) - low), np.full(n, 8),
                       [list(range(8))] * n, np.ones(n), np.zeros(n))
    assert L3.measure_part(b, model, lines).measurements["top"]["value"] == pytest.approx(APEX, abs=0.05)


def test_a_photo_pair_alone_makes_no_line():
    """A 3-D line needs its two matched photos plus >= 2 further photos agreeing: three photos make none, even with
    the merged-line photo counts relaxed to 2."""
    truth = oct_model()
    cams = ring([30.0, 40.0, 50.0])
    ids = render(truth, cams)
    P, q = sfm_points(truth, cams, ids)
    from dataclasses import replace
    loose = replace(L3.DEFAULT_LINES, min_images=2, keep_min_images=2, support_min=0.0)
    assert len(L3.lines_channel(oct_model(), cams, P, q, params=loose).lines) == 0
    cams = ring([20.0, 30.0, 40.0, 50.0, 60.0, 70.0])                       # six photos: lines, each seen by >= 4
    ids = render(truth, cams)
    P, q = sfm_points(truth, cams, ids)
    res = L3.lines_channel(oct_model(), cams, P, q, params=loose)
    assert len(res.lines) > 0 and res.lines.n_images.min() >= 4


# ---------------------------------------------------------------- building blocks


def test_close_pairs_components_and_point_grid_match_brute_force():
    rng = np.random.default_rng(3)
    X = rng.random((600, 3)) * 10
    owner = rng.integers(0, 150, 600)
    pairs = {tuple(p) for p in L3.close_owner_pairs(X, owner, 0.6).tolist()}
    d = np.linalg.norm(X[:, None] - X[None], axis=2)
    for i, j in zip(*np.nonzero(d < 0.6)):
        if owner[i] != owner[j]:
            assert (min(owner[i], owner[j]), max(owner[i], owner[j])) in pairs
    lab = L3.components(6, np.array([[0, 1], [1, 2], [4, 5]]))
    assert lab.tolist() == [0, 0, 0, 3, 4, 4]
    grid = L3.PointGrid(X, 0.4, cap=1000)
    Q = rng.random((300, 3)) * 10
    near = grid.nearest(Q)
    bf = np.linalg.norm(Q[:, None] - X[None], axis=2).min(1)
    assert np.allclose(np.where(bf <= 0.4, bf, np.inf), np.where(near <= 0.4, near, np.inf))


def test_predicted_edges_of_roof_shapes():
    pe = L3.predicted_edges(oct_model().built[0])
    assert len(pe["eave"]) == 8 and len(pe["hip"]) == 8 and pe["apex"] is not None and pe["apex"][2] == pytest.approx(APEX)
    rect = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])
    g = S.build_structure("g", [S.PartSpec("g1", "yes", rect.copy(), [], {"height": "15", "roof:shape": "gabled",
                                                                         "roof:height": "5"})], [rect.copy()], {})
    pe = L3.predicted_edges(g.built[0])
    assert len(pe["ridge"]) == 1 and pe["ridge"][0]["z"] == pytest.approx(15.0) and len(pe["rake"]) == 4
    assert pe["apex"] is None and len(pe["eave"]) == 2                    # the gable walls are not eaves
