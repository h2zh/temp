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


def test_judge_verifies_a_right_map_with_two_masks(masks):
    cams, sky = masks
    model = structure(TRUE_TOP)
    prob = K.SkylineProblem(model, cams, sky)
    prof = {"w2": prob.profile_part("w2")}
    d = J.decide(model, {"A": prof, "B": prof})
    assert d["w2"].state == S.STATE_VERIFIED, d["w2"].why
    J.apply_decisions(model, d, frame={"passed": True})
    assert model.part("w2").params["height"].state == S.STATE_VERIFIED
    assert J.check_decisions(model, d)["passed"]
    # one mask only: nothing can be verified
    d1 = J.decide(structure(TRUE_TOP), {"A": prof})
    assert d1["w2"].state == S.STATE_MAP and "one sky mask" in d1["w2"].why
    # a snap scale corrected by more than 3 %: not verified either
    assert J.decide(model, {"A": prof, "B": prof}, snap_scale=1.10)["w2"].state == S.STATE_MAP


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
    d = J.decide(model, {"A": prof, "B": prof}, frame={"passed": False})
    assert d["w2"].state == S.STATE_MAP and "frame ticket" in d["w2"].why


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
    assert not r["passed"] and "two sky masks" in whys and "map value" in whys
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


def test_mask_crosscheck():
    a = np.zeros((100, 120), bool)
    a[:40] = True
    b = a.copy()
    cols = np.ones(120, bool)
    assert K.mask_crosscheck(a, b, cols)["ok"]
    b2 = np.zeros_like(a)
    b2[:60] = True                                                # 20 px apart
    r = K.mask_crosscheck(a, b2, cols)
    assert not r["ok"] and r["med_px"] == 20.0
