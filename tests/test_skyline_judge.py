"""crowd3d.skyline + crowd3d.judge on synthetic scenes: a known massing's silhouette rendered into synthetic cameras
as the sky masks; a perturbed map is profiled, judged and (D15 off) never changed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from crowd3d import judge as J
from crowd3d import skyline as K
from crowd3d import structure as S
from crowd3d import stylize as st

BOX = np.array([[-15.0, -10.0], [15.0, -10.0], [15.0, 10.0], [-15.0, 10.0]])
TOWER = np.array([[-6.0, -6.0], [6.0, -6.0], [6.0, 6.0], [-6.0, 6.0]])
TRUE_TOP = 30.0
# the frame gate passes and a whole-building similarity is measured at k = 1: the common scale is established
FRAME_OK = {"passed": True, "similarity": {"measured": True, "k": 1.0, "t": [0.0, 0.0], "ticket": False,
                                           "at_bound": False}}


def look_at(center, target, w=320, h=240, f=230.0, name="cam") -> st.PosedCamera:
    c = np.asarray(center, float)
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return st.PosedCamera(name, w, h, f, f, w / 2, h / 2, np.stack([right, down, fwd]), c)


def structure(tower_top: float, tower_w: np.ndarray = TOWER) -> S.StructureModel:
    specs = [S.PartSpec("w1", "yes", BOX.copy(), [], {"height": "12"}),
             S.PartSpec("w2", "yes", tower_w.copy(), [], {"height": f"{tower_top:g}"})]
    return S.build_structure("synthetic", specs, [BOX.copy()], {})


def cameras(n: int = 12, radius: float = 75.0, bearings=None) -> list[st.PosedCamera]:
    angs = bearings if bearings is not None else np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [look_at([radius * math.cos(a), radius * math.sin(a), 1.6], [0.0, 0.0, 14.0], name=f"c{i}")
            for i, a in enumerate(angs)]


def sky_masks(model: S.StructureModel, cams: list[st.PosedCamera]) -> dict[int, np.ndarray]:
    """What an ideal sky mask of the true building shows: above the horizon and not the building."""
    tris = np.concatenate([b.triangles() for b in model.built])
    out = {}
    for i, cam in enumerate(cams):
        v, u = np.mgrid[0:cam.height, 0:cam.width]
        rays = np.stack([(u + 0.5 - cam.cx) / cam.fx, (v + 0.5 - cam.cy) / cam.fy, np.ones(u.shape)], -1) @ cam.rot
        out[i] = (rays[..., 2] > 0) & (K.raster_mask(cam, tris, (cam.height, cam.width)) == 0)
    return out


@pytest.fixture(scope="module")
def truth() -> S.StructureModel:
    return structure(TRUE_TOP)


@pytest.fixture(scope="module")
def masks(truth):
    cams = cameras()
    return cams, sky_masks(truth, cams)


def tower_points(top: float) -> np.ndarray:
    gx, gy = np.meshgrid(np.linspace(-5.5, 5.5, 12), np.linspace(-5.5, 5.5, 12))
    return np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, top)]


def lines_ev(pid: str, value: float, parameter: str = "eave", vote: str | None = None, n_photos: int = 12,
             snap_scale: float = 1.0) -> dict[str, list[dict]]:
    """A 3-D line measurement in crowd3d.lines3d's record format (the full pipeline runs in tests/test_lines3d.py);
    the tower is flat, so its eave is its top."""
    return {pid: [{"kind": "lines3d", "parameter": parameter, "value": value, "interval": [value - 0.2, value + 0.2],
                   "n_photos": n_photos, "n_lines": 8, "vote": vote, "implied_top": value, "can_vote": ["up", "down"],
                   "snap_scale": snap_scale, "scale_corrected": abs(snap_scale - 1.0) > 0.03}]}


def test_sky_mask_classical_finds_a_plain_sky():
    rng = np.random.default_rng(0)
    img = np.full((200, 300, 3), (150, 200, 245), np.uint8)
    img[120:] = (120, 110, 100)                                   # a wall below
    img[60:120, 100:160] = (180, 170, 160)                        # a tower
    img[60:] = np.clip(img[60:].astype(int) + rng.integers(-18, 18, img[60:].shape), 0, 255).astype(np.uint8)
    img[60:120:6, 100:160] = (90, 85, 80)                         # stone courses: texture, as real walls have
    m = K.sky_mask(img)
    assert m[:50].mean() > 0.95 and not m[130:].any() and not m[70:115, 105:155].any()


def test_profile_recovers_a_wrong_height(masks):
    cams, sky = masks
    model = structure(26.0)                                       # the map says 26 m, the photos show 30 m
    prob = K.SkylineProblem(model, cams, sky)
    p = prob.profile_part("w2")
    assert p.n_vis >= 3 and p.n_sky == 0                          # the true tower stands above the map's: no sky there
    eave = p.params["eave"]
    best = eave["metres"][eave["grid"].index(eave["fr"]["argmin"])]["top"]
    assert abs(best - TRUE_TOP) <= 0.05 * 26.0 + 1e-6, best         # within one grid step
    assert not eave["fr"]["at_bound"]
    # and a map that is right is where the cost is lowest
    p_ok = K.SkylineProblem(structure(TRUE_TOP), cams, sky).profile_part("w2")
    assert p_ok.params["eave"]["fr"]["argmin"] == 1.0


def test_judge_verifies_a_right_map_with_two_masks_and_a_second_channel(masks):
    """Verification needs >= 2 channels agreeing within max(1 m, 5 %): the skyline (two masks) and the 3-D lines.
    The skyline alone, one mask, lines that disagree with it, or a snap scale correction > 3 % verify nothing."""
    cams, sky = masks
    model = structure(TRUE_TOP)
    prob = K.SkylineProblem(model, cams, sky)
    prof = {"w2": prob.profile_part("w2")}
    d = J.decide(model, {"A": prof, "B": prof})
    assert d["w2"].state == S.STATE_MAP and "no second channel agrees" in d["w2"].why, d["w2"].why
    assert d["w2"].skyline_ok["eave"]                                  # the skyline's own conditions hold
    d = J.decide(model, {"A": prof, "B": prof}, lines=lines_ev("w2", 30.2))
    assert d["w2"].state == S.STATE_VERIFIED, d["w2"].why
    assert d["w2"].agreement["eave"]["agree"] and d["w2"].agreement["eave"]["values"]["lines3d"] == 30.2
    J.apply_decisions(model, d, frame={"passed": True})
    h = model.part("w2").params["height"]
    assert h.state == S.STATE_VERIFIED and h.value == TRUE_TOP
    assert {e["kind"] for e in h.evidence} >= {"skyline", "lines3d", "agreement"}
    assert J.check_decisions(model, d)["passed"]
    # the 3-D lines 1.8 m below the skyline (> max(1 m, 5 %)) but within the map's tolerance: no agreement, no vote
    d2 = J.decide(structure(TRUE_TOP), {"A": prof, "B": prof}, lines=lines_ev("w2", 28.2))["w2"]
    assert d2.state == S.STATE_MAP and not d2.votes and "lines 28.2 vs skyline" in d2.why, d2.why
    # one mask only: nothing can be verified
    d1 = J.decide(structure(TRUE_TOP), {"A": prof}, lines=lines_ev("w2", 30.2))
    assert d1["w2"].state == S.STATE_MAP and "one sky mask" in d1["w2"].why
    # a snap scale corrected by more than 3 %: not verified either
    assert J.decide(model, {"A": prof, "B": prof}, snap_scale=1.10, lines=lines_ev("w2", 30.2))["w2"].state == S.STATE_MAP


def test_two_channels_against_the_map_open_a_ticket_and_change_nothing(masks):
    cams, sky = masks
    model = structure(26.0)
    prob = K.SkylineProblem(model, cams, sky)
    prof = {"w2": prob.profile_part("w2")}
    sfm = J.sfm_channel(model, tower_points(TRUE_TOP))
    assert sfm["w2"]["q98"] == pytest.approx(TRUE_TOP, abs=0.01)
    d = J.decide(model, {"A": prof, "B": prof}, sfm)
    assert d["w2"].state == S.STATE_DISPUTED and "sfm-up" in d["w2"].votes and "skyline-up?" in d["w2"].votes
    J.apply_decisions(model, d, frame={"passed": True})
    h = model.part("w2").params["height"]
    assert h.state == S.STATE_DISPUTED and h.ticket and h.value == 26.0          # D15 off: the map value stays
    assert any(t["kind"] == "part" and t["part"] == "w2" for t in model.tickets)
    # the skyline alone never raises anything
    d_sky = J.decide(structure(26.0), {"A": prof, "B": prof})
    assert d_sky["w2"].state == S.STATE_MAP


def test_refused_when_fewer_than_three_photos_see_the_part(truth):
    cams = cameras(bearings=[0.0, 0.3])                           # two photos only
    sky = sky_masks(truth, cams)
    model = structure(26.0)
    prob = K.SkylineProblem(model, cams, sky)
    p = prob.profile_part("w2")
    assert p.n_vis <= 2
    d = J.decide(model, {"A": {"w2": p}, "B": {"w2": p}}, J.sfm_channel(model, tower_points(TRUE_TOP)))
    assert d["w2"].state == S.STATE_MAP and "insufficient" in d["w2"].why


def test_frame_ticket_blocks_verification(masks):
    cams, sky = masks
    model = structure(TRUE_TOP)
    prof = {"w2": K.SkylineProblem(model, cams, sky).profile_part("w2")}
    d = J.decide(model, {"A": prof, "B": prof}, frame={"passed": False}, lines=lines_ev("w2", TRUE_TOP))
    assert d["w2"].state == S.STATE_MAP and "frame ticket" in d["w2"].why and "photo-verified" in d["w2"].why


def test_frame_gate_and_similarity_find_a_scaled_shifted_outline():
    oct_map = np.array([[17.0 * math.cos(math.radians(22.5 + 45 * k)), 17.0 * math.sin(math.radians(22.5 + 45 * k))]
                        for k in range(8)])
    k_true, t_true = 0.82, np.array([3.0, -1.0])
    real = oct_map * k_true + t_true
    pts = []
    for i in range(8):
        a, b = real[i], real[(i + 1) % 8]
        s = np.linspace(0.05, 0.95, 60)
        for z in np.linspace(3.5, 15.5, 6):
            pts.append(np.c_[a + s[:, None] * (b - a), np.full(len(s), z)])
    pts = np.vstack(pts)
    gate = J.frame_gate(pts, [oct_map])
    assert not gate["passed"] and gate["failing"]
    sim = J.frame_similarity(pts, [oct_map])
    assert sim["measured"] and sim["ticket"]
    # the frame ticket puts the photos' heights beside k for the curator (an outline error or a mis-scaled model?)
    m0 = S.build_structure("oct", [S.PartSpec("w9", "yes", oct_map.copy(), [], {"height": "18"})], [oct_map.copy()], {})
    dec = {"w9": J.Decision("w9", S.STATE_MAP, "", evidence=[{"kind": "sfm-points", "q98": 14.8, "model_q98": 18.0,
                                                                 "snap_scale": 1.0}])}
    J.apply_decisions(m0, dec, frame={**gate, "similarity": sim})
    hyp = next(t for t in m0.tickets if t["kind"] == "frame")["scale_hypothesis"]
    assert hyp["outline_k"] == sim["k"] and hyp["height_ratio_median"] == pytest.approx(14.8 / 18.0, abs=1e-3)
    assert "agree with k" in hyp["reading"] and "map_top_vs_wikidata" not in hyp
    hyp = J.scale_hypothesis(m0, dec, sim, wikidata_height_m=18.5)          # Wikidata's total height beside the map's
    assert hyp["wikidata_height_m"] == 18.5 and hyp["map_top_vs_wikidata"] == pytest.approx(18.0 / 18.5, abs=1e-3)
    assert sim["k"] == pytest.approx(k_true, abs=0.01) and np.allclose(sim["t"], t_true, atol=0.1)
    # one side photographed only: scale and shift cannot be told apart -> not measured
    half = pts[pts[:, 0] > 2.0]
    assert not J.frame_similarity(half, [oct_map])["measured"]
    # a curator applies it to every part rigidly
    model = S.build_structure("oct", [S.PartSpec("w9", "yes", oct_map.copy(), [], {"height": "18"})], [oct_map.copy()], {})
    J.apply_frame(model, sim["k"], tuple(sim["t"]), centre=np.zeros(2))
    assert np.allclose(model.specs[0].outer, real, atol=0.15) and model.specs[0].outline_source == st.PROV_CURATOR
    assert J.frame_gate(pts, model.main_rings)["passed"]


def test_injected_violations_are_caught(masks):
    """G9: forged photo-fitted values, verification under an open frame ticket, a fitted slender part, a
    Wikidata height -- all caught."""
    model = structure(26.0)
    h = model.part("w2").params["height"]
    h.state, h.source = S.STATE_FITTED, st.PROV_FITTED                    # no evidence, no map value kept
    r = J.check_decisions(model)
    whys = " ".join(v["why"] for v in r["violations"])
    assert not r["passed"] and "adoption record" in whys and "map value" in whys
    m2 = structure(TRUE_TOP)
    m2.part("w2").params["height"].state = S.STATE_VERIFIED
    m2.tickets.append({"id": "T-frame", "kind": "frame", "open": True})
    assert any("frame ticket" in v["why"] for v in J.check_decisions(m2)["violations"])
    m3 = structure(26.0, tower_w=np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]))   # slender
    p3 = m3.part("w2").params["height"]
    p3.state, p3.source, p3.map_value = S.STATE_FITTED, st.PROV_FITTED, 26.0
    p3.evidence = [{"kind": "skyline", "mask": m, "sky_above_photos": 9, "snap_scale": 1.0} for m in ("A", "B")]
    assert any("slender" in v["why"] for v in J.check_decisions(m3)["violations"])
    m4 = structure(26.0)
    m4.part("w2").params["height"] = S.Param(56.0, st.PROV_WIKIDATA)
    assert any("forbidden" in v["why"] for v in J.check_decisions(m4)["violations"])


# ---------------------------------------------------------------- review fixes: tolerance, contrary votes, slender, stubs


def test_a_map_too_high_is_not_verified_and_one_strong_down_vote_opens_a_low_ticket(masks):
    """True top 30 m. A map of 34 m (13 % too high) was verified: its 5 % grid point 30.6 m lay exactly at the
    tolerance 3.4 m. At 36 m the outline scale verified and made the decision 'photo-verified' over a skyline-down
    vote. Only the skyline can say 'too high' today, so one strong down vote opens a low-strength ticket."""
    cams, sky = masks
    for map_top in (34.0, 36.0):
        model = structure(map_top)
        prof = {"w2": K.SkylineProblem(model, cams, sky).profile_part("w2")}
        d = J.decide(model, {"A": prof, "B": prof}, frame={"passed": True})
        dd = d["w2"]
        assert dd.state == S.STATE_MAP and not dd.verified["eave"], (map_top, dd.why)
        assert "skyline-down" in dd.votes and dd.ticket == "low", (map_top, dd.votes, dd.why)
        eave = dd.params["eave"]["A"]
        assert abs(eave["val_sub"] - TRUE_TOP) < abs(eave["val"] - TRUE_TOP) + 1e-6      # the sub-grid minimum is nearer
        J.apply_decisions(model, d, frame={"passed": True})
        h = model.part("w2").params["height"]
        assert h.state == S.STATE_MAP and h.value == map_top and h.ticket                # value and state: the map's
        tk = [t for t in model.tickets if t["kind"] == "part"]
        assert len(tk) == 1 and tk[0]["strength"] == "low" and tk[0]["part"] == "w2"
        assert J.check_decisions(model, d)["passed"]
    # a right map (and one within tolerance) still verifies with a second channel agreeing, without a ticket
    for map_top in (30.0, 32.0):
        model = structure(map_top)
        prof = {"w2": K.SkylineProblem(model, cams, sky).profile_part("w2")}
        dd = J.decide(model, {"A": prof, "B": prof}, frame={"passed": True}, lines=lines_ev("w2", TRUE_TOP))["w2"]
        assert dd.state == S.STATE_VERIFIED and dd.ticket is None, (map_top, dd.why)
    # a decision verified over a contrary vote is caught (G4)
    forged = J.Decision("w2", S.STATE_VERIFIED, "forged", votes={"skyline-down": -5.4}, n_sky=9)
    assert any("contrary" in v["why"] for v in J.check_decisions(structure(36.0), {"w2": forged})["violations"])
    # ... and from the parameter alone (the publish gate sees the model, not the decisions): a verified height whose
    # own evidence carries a vote against it
    model = structure(36.0)
    h = model.part("w2").params["height"]
    h.state = S.STATE_VERIFIED
    h.evidence = [{"kind": "skyline", "mask": m, "sky_above_photos": 9, "snap_scale": 1.0, "vote": v,
                   "values": {"eave": 35.6}} for m, v in (("A", None), ("B", None))]
    h.evidence += lines_ev("w2", 35.8)["w2"]
    h.evidence.append({"kind": "agreement", "parameter": "eave", "channels": ["skyline", "lines3d"],
                       "values": {"skyline": 35.6, "lines3d": 35.8}, "vote": None, "snap_scale": 1.0})
    assert J.check_decisions(model)["passed"], J.check_decisions(model)
    h.evidence[1]["vote"] = "down"
    assert any("contrary vote in its own evidence" in v["why"] for v in J.check_decisions(model)["violations"])


def test_a_slender_part_is_never_verified_even_with_perfect_evidence():
    """Own height / narrowest width >= 2 and width <= 10 m: single channels are only recorded, never a verification
    (the design's flowchart); check_decisions rejects a verified slender part."""
    thin = np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]])          # 5 m wide, 30 m tall
    truth = structure(TRUE_TOP, thin)
    cams = cameras()
    sky = sky_masks(truth, cams)
    model = structure(TRUE_TOP, thin)
    prof = {"w2": K.SkylineProblem(model, cams, sky).profile_part("w2")}
    d = J.decide(model, {"A": prof, "B": prof}, frame={"passed": True}, lines=lines_ev("w2", TRUE_TOP))["w2"]
    assert any(d.verified.values()), d.why                    # the evidence (two channels agreeing) would verify it ...
    assert d.slender and d.state == S.STATE_MAP and "slender" in d.why     # ... the slender rule keeps it on the map
    h = model.part("w2").params["height"]
    h.state = S.STATE_VERIFIED
    h.evidence = [{"kind": "skyline", "mask": m, "sky_above_photos": 9, "snap_scale": 1.0} for m in ("A", "B")]
    assert any("slender part was verified" in v["why"] for v in J.check_decisions(model)["violations"])


def test_the_judge_never_moves_a_height_pending_stub_out_of_stub():
    """A 3 m 'height pending' stub (default-style) under a dispute keeps its state (stub: dashed top, '高度待定'), gets
    the ticket and the evidence, unless an adoption record that passes D15 (two channels agreeing) gives it a measured
    height: a decision carrying only would_fit (one channel, no record) adopts nothing, even with the switch on;
    check_decisions rejects any other state on a default-style source."""
    specs = [S.PartSpec("w1", "yes", np.array([[0.0, -10.0], [15.0, -10.0], [15.0, 10.0], [0.0, 10.0]]), [], {"height": "30"})]
    model = S.build_structure("synthetic", specs, [BOX.copy()], {})
    stub = next(p for p in model.specs if p.kind == "remainder")
    assert stub.params["height"].source == st.PROV_DEFAULT and stub.params["height"].state == S.STATE_STUB
    ev = [{"kind": "sfm-points", "q98": 17.4, "model_q98": 3.0, "vote": "up", "snap_scale": 1.0}]
    d = {stub.id: J.Decision(stub.id, S.STATE_DISPUTED, "up: sfm-up=14.4, skyline-up?=7.8",
                             votes={"sfm-up": 14.4, "skyline-up?": 7.8}, evidence=ev, would_fit={"top": 17.4})}
    J.apply_decisions(model, d, frame={"passed": True}, auto_adopt=True)
    h = stub.params["height"]
    assert h.state == S.STATE_STUB and h.value == 3.0 and h.source == st.PROV_DEFAULT       # not disputed, not fitted
    assert h.ticket and h.evidence == ev and any(t["kind"] == "part" and t["part"] == stub.id for t in model.tickets)
    assert stub.state() == S.STATE_STUB and J.check_decisions(model)["passed"]
    assert model.compact_extras()["parts"][-1]["params"]["height"][1] == S.STATE_STUB
    h.state = S.STATE_DISPUTED                                                             # what the old code wrote
    assert any("stub" in v["why"] for v in J.check_decisions(model)["violations"])
    g2 = S.check_structure(model)                                                          # and the structure gate G2
    assert not g2["passed"] and any(r["id"] == stub.id and "stub" in r.get("why", "") for r in g2["G2_bad_sources"])


def test_decide_keeps_a_stub_a_stub_under_two_channels_against_it():
    """decide() itself (not only apply_decisions) never makes a 'height pending' stub disputed or verified, and fits
    it only when D15 holds: the real block over the gap is 17 m, the skyline and the SfM points both say "higher" (two
    votes: a dispute for a mapped part) but only the points give a value (the skyline's profile of a 3 m stub stops at
    3.9 m), so the stub stays a stub with its ticket and no value to adopt (Aachen's remainder13)."""
    east = np.array([[0.0, -10.0], [15.0, -10.0], [15.0, 10.0], [0.0, 10.0]])
    west = np.array([[-15.0, -10.0], [0.0, -10.0], [0.0, 10.0], [-15.0, 10.0]])
    truth = S.build_structure("synthetic", [S.PartSpec("w1", "yes", east.copy(), [], {"height": "12"}),
                                            S.PartSpec("w9", "yes", west.copy(), [], {"height": "17"})], [BOX.copy()], {})
    model = S.build_structure("synthetic", [S.PartSpec("w1", "yes", east.copy(), [], {"height": "12"})], [BOX.copy()], {})
    stub = next(p for p in model.specs if p.kind == "remainder")
    assert stub.params["height"].source == st.PROV_DEFAULT
    cams = cameras()
    sky = sky_masks(truth, cams)
    prof = {stub.id: K.SkylineProblem(model, cams, sky).profile_part(stub.id)}
    sfm = {stub.id: {"q98": 17.0, "model_q98": 3.0, "points": 300, "min_track": 3}}
    d = J.decide(model, {"A": prof, "B": prof}, sfm, frame={"passed": True})[stub.id]
    assert "sfm-up" in d.votes and any(v.startswith("skyline-up") for v in d.votes), (d.votes, d.why)
    assert d.state == S.STATE_STUB and d.would_fit is None and d.ticket == "dispute", (d.state, d.why)
    assert d.adopt is None and d.adopt_why.startswith("1 channel(s)"), d.adopt_why   # the skyline: at its 3.9 m bound
    assert "height pending stub" in d.why
    J.apply_decisions(model, {stub.id: d}, frame={"passed": True}, auto_adopt=True)
    assert stub.params["height"].state == S.STATE_STUB and stub.params["height"].value == 3.0
    assert J.check_decisions(model, {stub.id: d})["passed"]


def _adopt_record(judged_against: float, value: float, direction: str = "up", **chan) -> dict:
    """A D15 adoption record as decide() writes it (two channels agreeing)."""
    ch = chan or {"skyline": {"ok": True, "value": value + 0.1, "photos": 12, "masks": 2, "sky_above_photos": 9},
                  "sfm-points": {"ok": True, "value": value - 0.1, "photos": 3, "points": 144}}
    return {"rule": "D15", "param": "height", "direction": direction, "judged_against": judged_against,
            "value": value, "channels": ch, "snap_scale": 1.0, "scale": J.scale_status(FRAME_OK), "scale_free": None,
            "slender": False, "provenance": st.PROV_FITTED_AUTO}


def test_an_adopted_top_is_drawn_the_map_value_kept_and_the_part_on_it_follows_and_it_reverts():
    """D15 (on by default): an adopted top is drawn, not only written into the parameter (a part labelled
    'photo-fitted' with its solid at the map value would misreport its shape); the map value is kept beside it; a
    part resting on it follows (D16 ②: its base and top move by the change, its own 3 m extent is kept) with a carry
    record; the reversible list restores everything, and the ticket reopens."""
    model = structure(24.0)
    cap = S.PartSpec("w3", "yes", np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]), [],
                     {"height": "27", "min_height": "24"})
    model = S.build_structure("synthetic", [p for p in model.specs if p.kind != "remainder"] + [cap], [BOX.copy()], {})
    assert model.part("w3").parent == "w2" and model.built_part("w2").top == pytest.approx(24.0)
    ev = [{"kind": "skyline", "mask": m, "sky_above_photos": 9, "snap_scale": 1.0} for m in ("A", "B")]
    d = {"w2": J.Decision("w2", S.STATE_DISPUTED, "up", votes={"sfm-up": 6.0, "skyline-up?": 6.0}, evidence=ev,
                          direction="up", adopt=_adopt_record(24.0, 30.0))}
    J.apply_decisions(model, d, frame={"passed": True}, auto_adopt=False)    # the switch off: nothing changes
    assert model.part("w2").params["height"].value == 24.0 and model.built_part("w2").top == pytest.approx(24.0)
    assert not model.adoptions
    J.apply_decisions(model, d, frame={"passed": True})                      # D15 on (the default)
    h = model.part("w2").params["height"]
    assert h.state == S.STATE_FITTED and h.value == 30.0 and h.map_value == 24.0 and h.source == st.PROV_FITTED_AUTO
    assert model.built_part("w2").top == pytest.approx(30.0)                  # drawn where it was adopted
    c3 = model.part("w3")
    assert model.built_part("w3").z0 == pytest.approx(30.0) and model.built_part("w3").top == pytest.approx(33.0)
    assert c3.params["min_height"].map_value == 24.0 and c3.params["height"].map_value == 27.0
    assert any(e["kind"] == "carried" and e["parent"] == "w2" for e in c3.params["height"].evidence)
    assert S.check_structure(model)["passed"] and J.check_decisions(model)["passed"], J.check_decisions(model)
    (a,) = model.adoptions
    assert a["active"] and a["map_value"] == 24.0 and a["value"] == 30.0 and a["carried"][0]["part"] == "w3"
    assert a["provenance"] == "photo-fitted (auto)" and a["label"] == "照片量出（自动采用）"
    tk = next(t for t in model.tickets if t["kind"] == "part")
    assert not tk["open"] and tk["resolved_by"] == a["id"]
    reg = model.to_registry()
    assert reg["adoptions_active"] == 1 and model.compact_extras()["adoptions"][0]["id"] == a["id"]
    # forged: a carried value that does not add up, or an adoption no longer in the list, is caught (G4)
    c3.params["height"].value = 34.0
    assert any("carried" in v["why"] for v in J.check_decisions(model)["violations"])
    c3.params["height"].value = 33.0
    # the reversible list: everything comes back, the ticket reopens
    assert J.revert_adoption(model, a["id"]) == [a["id"]]
    assert model.part("w2").params["height"].value == 24.0 and model.part("w2").params["height"].source == st.PROV_S3DB
    assert model.built_part("w3").z0 == pytest.approx(24.0) and c3.params["height"].source == st.PROV_S3DB
    assert not model.adoptions[0]["active"] and model.adoptions[0]["reverted"]
    assert next(t for t in model.tickets if t["kind"] == "part")["open"]
    assert J.check_decisions(model)["passed"] and S.check_structure(model)["passed"]


def test_skyline_down_and_lines_down_dispute_and_d15_adopts_their_agreed_value(masks):
    """True top 30 m, map 36 m: the skyline says 'down' and the 3-D lines measure 30.1 m ('down'): two channels, a
    dispute whose ticket carries both measured values and their agreement; with the switch off the map's value and
    solid stay, with D15 on (the default) the agreed value is adopted and drawn."""
    cams, sky = masks
    model = structure(36.0)
    prof = {"w2": K.SkylineProblem(model, cams, sky).profile_part("w2")}
    lines = lines_ev("w2", 30.1, vote="down")
    d = J.decide(model, {"A": prof, "B": prof}, frame=FRAME_OK, lines=lines)["w2"]
    assert d.state == S.STATE_DISPUTED and {"skyline-down", "lines-down"} <= set(d.votes), d.why
    assert set(d.agreeing) == {"skyline", "lines3d"} and abs(d.agreed - 30.0) < 1.5, (d.agreeing, d.agreed)
    J.apply_decisions(model, {"w2": d}, frame={"passed": True}, auto_adopt=False)
    h = model.part("w2").params["height"]
    assert h.state == S.STATE_DISPUTED and h.value == 36.0 and model.built_part("w2").top == pytest.approx(36.0)
    tk = next(t for t in model.tickets if t["kind"] == "part")
    assert tk["strength"] == 2 and tk["measured"]["lines3d"] == 30.1 and set(tk["agreeing_channels"]) == {"skyline", "lines3d"}
    assert J.check_decisions(model, {"w2": d})["passed"]
    assert d.adopt and d.would_fit["top"] == d.agreed and set(d.adopt["channels"]) == {"skyline", "lines3d"}
    m2 = structure(36.0)
    J.apply_decisions(m2, {"w2": d}, frame={"passed": True})
    h2 = m2.part("w2").params["height"]
    assert h2.state == S.STATE_FITTED and h2.map_value == 36.0 and h2.value == pytest.approx(d.agreed)
    assert m2.built_part("w2").top == pytest.approx(d.agreed) and J.check_decisions(m2, {"w2": d})["passed"]
