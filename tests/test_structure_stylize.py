"""crowd3d.stylize with the structure-first massing (crowd3d.structure parts): clay parts, gates, roof lines."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import test_stylize as ts   # the synthetic south-facade scene (pytest puts tests/ on the path)

from crowd3d import structure as S
from crowd3d import stylize as st


def box_structure(roof: str = "gabled", extra: list[S.PartSpec] | None = None) -> S.StructureModel:
    tags = {"height": "14", "roof:shape": roof, "roof:height": "4"} if roof != "flat" else {"height": "10"}
    specs = [S.PartSpec("w100", "yes", ts.BOX.copy(), [], tags, name="box")] + list(extra or [])
    return S.build_structure("Test Box", specs, [ts.BOX.copy()], {})


@pytest.fixture(scope="module")
def parts_model() -> st.StylizedModel:
    scene = ts.south_scene()
    scene.structure = box_structure()
    return st.stylize_landmark(scene)


def test_the_parts_are_the_massing_and_every_check_passes(parts_model):
    s = parts_model.summary
    assert s["massing_source"] == "osm-parts"
    qa = s["qa"]
    assert qa["passed"] and qa["audit"]["passed"] and qa["glb"]["passed"], qa
    pr = parts_model.massing.prisms
    assert len(pr) == 1 and pr[0].cap == "faces" and pr[0].roof_shape == "gabled" and pr[0].part == "w100"
    walls = parts_model.massing.walls
    assert len(walls) == 4 and sum(w.profile is not None for w in walls) == 2       # two gable ends
    assert parts_model.observations[ts.SOUTH].observed                            # the photographed long wall
    assert s["structure"]["version"] == 2 and s["structure"]["parts"][0]["params"]["height"]["source"] == st.PROV_S3DB


def test_gable_walls_and_roof_faces_in_the_glb(parts_model):
    gltf, binary = st.read_glb(parts_model.glb)
    extras = gltf["scenes"][0]["extras"]
    assert extras["structure"]["parts"][0]["roof"] == "gabled"
    assert any("OpenStreetMap" in a for a in extras["attribution"])
    nodes = {n["name"]: n for n in gltf["nodes"]}
    gables = [p for p in parts_model.parts if p.extras.get("kind") == "wall" and p.extras.get("profiled")]
    assert gables
    for p in gables:
        pos = p.positions
        assert len(pos) == 5 and len(p.indices) == 3                               # pentagon: 3 triangles
        assert pos[0][2] == pytest.approx(0.0) and pos[1][2] == pytest.approx(0.0)   # bottom-left, bottom-right first
        assert pos[2][2] == pytest.approx(10.0) and pos[3][2] == pytest.approx(14.0)  # top-right, the ridge, top-left
        assert p.extras["provenance"]["outline"] == st.PROV_S3DB and p.extras["part"] == "w100"
        assert not nodes[p.name]["extras"]["observed"] and "locked" not in nodes[p.name]["extras"]
    roofs = [p for p in parts_model.parts if p.extras.get("kind") == "roof"]
    assert roofs and all(p.extras["cap"] == "gabled" and p.extras["roof"]["shape"] == "gabled" for p in roofs)
    assert sum(len(p.extras["faces"]) for p in roofs) == 2                          # two slopes
    tri = np.concatenate([p.positions[p.indices.reshape(-1)] for p in roofs])
    assert tri[:, 2].max() == pytest.approx(14.0) and tri[:, 2].min() == pytest.approx(10.0)
    for p in parts_model.parts:
        assert all(p.extras["provenance"][k] in st.PROVENANCES for k in st.PROVENANCE_KEYS)


def test_a_dome_and_a_raised_lantern_close_and_show_their_underside():
    dome = S.PartSpec("w200", "yes", np.array([[5.0 * math.cos(a), 5.0 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]),
                      [], {"height": "13", "roof:shape": "dome", "roof:height": "3", "min_height": "10"})
    model = box_structure("flat", [dome])
    assert model.part("w200").parent == "w100"
    scene = ts.south_scene()
    scene.structure = model
    m = st.stylize_landmark(scene)
    assert m.summary["qa"]["passed"], m.summary["qa"]
    pr = next(p for p in m.massing.prisms if p.part == "w200")
    assert pr.cap == "faces" and pr.num_faces() == 12 * 6 and pr.z0 == pytest.approx(10.0)
    assert pr.roof_height(np.array([[0.2, 0.1]]))[0] == pytest.approx(13.0, abs=0.3)
    assert any(p.extras.get("kind") == "base" and p.extras.get("part") == "w200" for p in m.parts)
    face = pr.face_of(np.array([[0.0, 0.0], [4.9, 0.0]]))
    assert face[0] != face[1]


def test_a_part_roof_is_one_clay_node_and_keeps_its_shape(parts_model):
    """D29: no roof counts as photographed. A part's roof is one clay node with all its faces and their point
    coverage; a pitched face without points stays pitched."""
    roofs = [p for p in parts_model.parts if p.extras.get("kind") == "roof"]
    assert len(roofs) == 1 and roofs[0].extras["faces"] == [0, 1] and roofs[0].name == "roof-0-clay-0"
    assert not roofs[0].extras["observed"] and roofs[0].rgb == st.CLAY_ROOF_RGB
    assert roofs[0].extras["provenance"]["appearance"] == st.PROV_DEFAULT and len(roofs[0].extras["coverage"]) == 2
    assert max(p.positions[:, 2].max() for p in roofs) == pytest.approx(14.0)
    assert "point_colors" not in parts_model.summary and "roof_observed_area_m2" not in parts_model.summary


def ideal_sky(scene: st.SceneInput) -> dict[int, np.ndarray]:
    """The sky mask of the procedural south-facade photos (their sky is one flat colour), for the judge's skyline
    channel (test_d15). The classical mask takes the synthetic, textureless wall for sky; real walls have texture
    (test_skyline_judge)."""
    return {i: np.all(c.image == (150, 200, 250), axis=-1) for i, c in enumerate(scene.cameras)}


# ---------------------------------------------------------------- review fixes


def stub_scene() -> st.SceneInput:
    """The photographed box, but OSM maps only its east half (10 m); the west half is a main-outline gap with no
    height fact: a 3 m 'height pending' stub. Points stand over the gap at 1.5-4.5 m (Aachen's remainder13 had 267 of
    them in that band, 3.1 % of the 8,594 over it), and a photo from a neighbouring tower looks down on it."""
    scene = ts.south_scene()
    east = np.array([[0.0, -6.0], [10.0, -6.0], [10.0, 6.0], [0.0, 6.0]])
    scene.structure = S.build_structure("Test Box", [S.PartSpec("w300", "yes", east, [], {"height": "10"})],
                                        [ts.BOX.copy()], {})
    rng = np.random.default_rng(5)
    low = np.stack([rng.uniform(-9.5, -0.5, 1500), rng.uniform(-5.5, 5.5, 1500), rng.uniform(1.5, 4.5, 1500)], 1)
    scene.points = np.vstack([scene.points, low])
    scene.colors = np.vstack([scene.colors, np.tile([[150, 80, 60]], (len(low), 1))]).astype(np.uint8)
    tower = ts.look_at([-5.0, -24.0, 26.0], [-5.0, -1.0, 3.0], name="tower")
    tower.image = ts.render_south(tower)
    scene.cameras.append(tower)
    # ... and the points over the gap near its west and north edges are seen from there (grey photos): points on a
    # face no camera faces cannot lie there (the registration gate refuses them, STYLIZED_MODEL 3.2)
    north = ts.look_at([-5.0, 32.0, 1.6], [-5.0, 6.0, 3.0], name="side-north")
    north.image = np.full((north.height, north.width, 3), 128, np.uint8)
    scene.cameras += [ts.side_cameras()[1], north]
    return scene


def test_a_height_pending_stub_is_always_clay_and_every_check_says_so():
    """A 3 m 'height pending' stub gets nothing from the few points in its 1.5 m band (no roof coverage, no
    photographed wall): every face of it is clay, and check_honesty and audit_glb say so."""
    scene = stub_scene()
    stub = next(p for p in scene.structure.specs if p.kind == "remainder")
    assert stub.params["height"].source == st.PROV_DEFAULT and stub.drawn
    m = st.stylize_landmark(scene)
    qa = m.summary["qa"]
    assert qa["passed"], qa
    mine = [p for p in m.parts if p.extras.get("part") == stub.id]
    assert mine and all(p.extras["provenance"]["appearance"] == st.PROV_DEFAULT and not p.extras["observed"]
                        and p.rgb in (st.CLAY_RGB, st.CLAY_ROOF_RGB) for p in mine)
    pr = next(p for p in m.massing.prisms if p.part == stub.id)
    assert pr.stub and not any(m.roof.coverage.get((pr.index, f), 0.0) for f in range(pr.num_faces()))
    assert not any(pr.face_coverage) and not m.massing.roof_grids[pr.index].covered.any()    # no point "sees" its cap
    stub_walls = [w.index for w in m.massing.walls if w.prism == pr.index]
    assert all("stub" in m.observations[i].reason for i in stub_walls)
    assert any(m.observations[w.index].observed for w in m.massing.walls if w.prism != pr.index)   # the rest is seen
    # the stub's roof claiming photos: check_honesty and the bytes audit catch it
    roof = next(p for p in mine if p.extras["kind"] == "roof")
    roof.rgb = (196, 78, 56)                                                           # not a clay colour
    roof.extras.update(observed=True, provenance={**roof.extras["provenance"], "appearance": st.PROV_PHOTO})
    assert not st.check_honesty(m.parts)["stub_faces_not_clay"]                        # without the stub ids: blind
    h = st.check_honesty(m.parts, stub_parts=st.stub_part_ids(scene.structure))
    assert not h["passed"] and h["stub_faces_not_clay"] == 1
    gltf, _ = st.read_glb(m.glb)
    g = st.audit_glb(st.write_glb(m.parts, gltf["scenes"][0]["extras"]))
    assert not g["passed"] and g["stub_nodes_not_clay"] == [roof.name]


def test_structure_gates_are_part_of_the_publish_qa():
    """Review finding 2: G1-G12 were not in the publish QA. A floating part opens a G10 ticket and is withheld; an
    unsupported part without its ticket, or a parameter verified without evidence (G4 / G9), stops publication."""
    floating = S.PartSpec("w400", "yes", np.array([[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]]), [],
                          {"height": "20", "min_height": "16"})                     # 6 m above the 10 m box
    scene = ts.south_scene()
    scene.structure = box_structure("flat", [floating])
    m = st.stylize_landmark(scene)
    qa = m.summary["qa"]
    assert qa["passed"] and qa["structure"]["passed"], qa["structure"]
    assert qa["structure"]["gates"]["G10_withheld"] == ["w400"]
    assert any(t["kind"] == "support" and t["part"] == "w400" and t["action"] == "withheld"
               for t in m.summary["structure"]["tickets"])
    assert not any(p.extras.get("part") == "w400" for p in m.parts)
    assert st.read_glb(m.glb)[0]["scenes"][0]["extras"]["structure"]["withheld"] == ["w400"]
    # the same part drawn again without its ticket (a model edited after the build): not published
    model = box_structure("flat", [S.PartSpec("w400", "yes", floating.outer.copy(), [], dict(floating.tags))])
    model.part("w400").drawn, model.part("w400").support = True, None
    model.tickets = [t for t in model.tickets if t["kind"] != "support"]
    model.built = S.build_all(model.specs, model.params)
    scene.structure = model
    qa = st.stylize_landmark(scene).summary["qa"]
    assert not qa["passed"] and qa["structure"]["gates"]["G10_without_ticket"] == ["w400"]
    # a parameter photo-verified without two masks' evidence: not published
    model = box_structure("flat")
    model.part("w100").params["height"].state = S.STATE_VERIFIED
    scene.structure = model
    qa = st.stylize_landmark(scene).summary["qa"]
    assert not qa["passed"] and not qa["structure"]["decisions"]["passed"]


def test_the_frame_gate_opens_a_ticket_and_keeps_the_walls_on_its_edge_clay():
    """G11 in the publish run: points 5.5 m inside the south edge (counted by the per-edge gate, beyond the facade
    plane search) push its median over 1.5 m. A frame ticket opens and the south wall, otherwise photographed, gets no
    photo evidence."""
    scene = ts.south_scene()
    scene.structure = box_structure("flat")
    ok = st.stylize_landmark(scene)
    assert ok.summary["qa"]["structure"]["frame"]["passed"] and ok.observations[ts.SOUTH].observed
    rng = np.random.default_rng(11)
    inner = np.stack([rng.uniform(-4, 4, 4000), np.full(4000, -0.5), rng.uniform(2.5, 9.5, 4000)], 1)
    scene.points = np.vstack([scene.points, inner])
    scene.colors = np.vstack([scene.colors, np.full((len(inner), 3), 128, np.uint8)])
    scene.structure = box_structure("flat")
    m = st.stylize_landmark(scene)
    sq = m.summary["qa"]["structure"]
    assert not sq["frame"]["passed"] and [f["edge"] for f in sq["frame"]["failing"]] == [0]
    assert any(t["kind"] == "frame" and t["open"] for t in m.summary["structure"]["tickets"])
    south = [w.index for w in m.massing.walls if w.edge == 0]
    assert sq["frame_blocked_walls"] == south and all("G11" in m.observations[i].reason for i in south)
    assert not any(p.extras.get("observed") for p in m.parts if p.extras.get("wall") in south)
    assert m.summary["qa"]["passed"] and sq["frame_ok"]
    # nothing may be verified while the frame ticket is open
    scene.structure = box_structure("flat")
    h = scene.structure.part("w100").params["height"]
    h.state = S.STATE_VERIFIED
    h.evidence = [{"kind": "skyline", "mask": k, "sky_above_photos": 9, "snap_scale": 1.0} for k in ("A", "B")]
    sq = st.stylize_landmark(scene).summary["qa"]["structure"]
    assert not sq["passed"] and any("frame ticket" in v["why"] for v in sq["decisions"]["violations"])


def test_points_standing_5_m_above_a_part_roof_do_not_cover_it():
    """A structure part's roof is a map fact: only points within 1.5 m of it count as its coverage (a roof the map has
    too low gets no coverage from the points of the real one above it)."""
    rng = np.random.default_rng(2)
    xy = np.stack([rng.uniform(-9.5, 9.5, 3000), rng.uniform(-5.5, 5.5, 3000)], 1)
    for z, covered in ((15.0, False), (10.3, True)):
        pts = np.c_[xy, np.full(len(xy), z)]
        massing = st.build_massing_from_structure(box_structure("flat"), pts, st.DEFAULT_PARAMS)
        cov = st.roof_coverage(massing, pts).coverage[(0, 0)]
        assert (cov > 0.8) if covered else cov == 0.0, (z, cov)
        massing.roof_grids.clear()                                    # recomputed from the points: the same
        assert st.roof_coverage(massing, pts).coverage[(0, 0)] == cov


def test_an_unregistered_model_runs_no_frame_gate_and_opens_no_frame_ticket():
    """G11 measures the registered points against the outline. With the registration rejected nothing shows photos,
    so the gate does not run: points 4 m south of the footprint (inside the per-edge band, registration rejected)
    must not open a frame ticket or block walls in a model that is clay anyway."""
    scene = ts.south_scene()
    scene.points = scene.points + np.array([0.0, -4.0, 0.0])
    for c in scene.cameras:
        c.center = c.center + np.array([0.0, -4.0, 0.0])
    scene.snap = False
    scene.structure = box_structure("flat")
    m = st.stylize_landmark(scene)
    assert not m.summary["registration"]["accepted"]
    sq = m.summary["qa"]["structure"]
    assert sq["frame"]["passed"] and "no registered points" in sq["frame"]["note"] and not sq["frame_blocked_walls"]
    assert not any(t["kind"] == "frame" for t in m.summary["structure"]["tickets"])
    assert all(p.extras["provenance"]["appearance"] == st.PROV_DEFAULT for p in m.parts) and m.summary["qa"]["passed"]
    # the same points, gated as if registered, fail G11 (what the old code did with them)
    from crowd3d import judge as J
    assert not J.frame_gate(np.asarray(scene.points, float), scene.structure.main_rings)["passed"]


def test_every_node_of_a_structure_part_names_its_part():
    """check_honesty and audit_glb find a stub's nodes by their part id, so every node a structure part can produce
    carries it -- corner strips too (a bevelled build)."""
    from dataclasses import replace
    model = box_structure("flat")
    massing = st.build_massing_from_structure(model, None, st.DEFAULT_PARAMS)
    parts = st.build_parts(massing, {}, st.RoofLook(), replace(st.DEFAULT_PARAMS, bevel_m=0.5), [])
    assert {p.extras["kind"] for p in parts} >= {"wall", "corners", "roof"}
    assert all(p.extras.get("part") == "w100" for p in parts), [p.name for p in parts if "part" not in p.extras]


def test_the_roof_lists_its_structural_lines_and_3d_lines_verify_them(parts_model):
    from crowd3d.lines3d import Lines3D
    sl = parts_model.summary["structure_lines"]
    assert sl["by_type"] == {"ridge": 1, "eave": 2, "rake": 4} and sl["photo_verified"] == 0 and sl["convention"] == 0
    ridge = next(r for r in sl["lines"] if r["type"] == "ridge")
    a, b = np.array(ridge["a"]), np.array(ridge["b"])
    assert abs(a[2] - 14.0) < 1e-6 and abs(b[2] - 14.0) < 1e-6 and abs(np.linalg.norm(b - a) - 20.0) < 0.5
    ex = st.read_glb(parts_model.glb)[0]["scenes"][0]["extras"]["structure_lines"]
    row = next(r for r in ex["lines"] if ex["types"][r[6]] == "ridge")
    assert len(ex["lines"]) == 7 and row[1] == row[4] == 14.0                                    # glTF y is up

    def seen(p, q, photos=5):
        return Lines3D(np.array([p], float), np.array([q], float), np.array([photos]), [list(range(photos))], np.ones(1), np.zeros(1))
    assert st.confirm_segment(a, b, seen(a + 0.2 * (b - a), a + 0.8 * (b - a)))["photos"] == 5
    assert st.confirm_segment(a, b, seen(a + [0, 1.0, 0], b + [0, 1.0, 0])) is None                # 1 m beside it
    assert st.confirm_segment(a, b, seen(a + [0, 0, 0.5], b + [0, 0, 0.5])) is None                # 0.5 m above it
    assert st.confirm_segment(a, b, seen(a, a + 0.1 * (b - a))) is None                            # covers 10 %
    assert st.confirm_segment(a, b, seen(a, b, photos=2)) is None                                   # 2 photos
    turn = np.array([[np.cos(0.26), -np.sin(0.26), 0], [np.sin(0.26), np.cos(0.26), 0], [0, 0, 1]])
    assert st.confirm_segment(a, b, seen(a, a + turn @ (b - a))) is None                            # 15 deg off


def test_a_hipped_roof_has_four_hips_and_a_conventional_one_is_marked_for_dashes():
    scene = ts.south_scene()
    scene.structure = box_structure("hipped")
    sl = st.stylize_landmark(scene).summary["structure_lines"]
    assert sl["by_type"] == {"ridge": 1, "hip": 4, "eave": 4} and sl["convention"] == 0
    specs = [S.PartSpec("w100", "yes", ts.BOX.copy(), [], {"height": "14", "roof:shape": "hipped"}, name="box")]
    scene.structure = S.build_structure("Test Box", specs, [ts.BOX.copy()], {})
    m = st.stylize_landmark(scene)
    sl = m.summary["structure_lines"]
    assert sl["n"] == 9 and sl["convention"] == 9                     # roof height by the S3DB convention (D17)
    assert m.summary["structure_metrics"]["census"]["roof_lines"] == {"n": 9, "convention": 9, "photo_verified": 0}


def test_a_roof_line_is_verified_only_above_its_per_line_chance(parts_model):
    """verify-r1: displaced copies of the drawn roof lines were 'confirmed' about as often as the lines themselves.
    A line is photo-verified only when the 3-D lines lie along it better than along its displaced copies (1.5 m
    up / down / out / in, turned 15 deg), and the verification carries that chance rate."""
    from crowd3d.lines3d import Lines3D
    ridge = next(r for r in parts_model.summary["structure_lines"]["lines"] if r["type"] == "ridge")
    a, b = np.array(ridge["a"]), np.array(ridge["b"])
    mk = lambda segs: Lines3D(np.array([p for p, q in segs], float), np.array([q for p, q in segs], float),
                              np.full(len(segs), 6), [list(range(6))] * len(segs), np.ones(len(segs)), np.zeros(len(segs)))
    along = [(a + 0.1 * (b - a), a + 0.9 * (b - a))]
    ver, chk = st.verify_segment(a, b, mk(along))
    assert ver is not None and ver["null"]["confirmed"] == 0 and ver["null"]["chance"] == 0.0
    # a dense field of parallel lines (a striped facade, a tiled roof): the copies 1.5 m up and down are 'confirmed'
    # as well as the line itself -> confirmed, but not verified
    field = along + [(p + [0, 0, dz], q + [0, 0, dz]) for p, q in along for dz in (-1.5, 1.5)]
    ver, chk = st.verify_segment(a, b, mk(field))
    assert ver is None and chk["confirmed"] and chk["null"]["confirmed"] >= 2 and chk["null"]["chance"] > 0
    assert len(st.line_alternatives(a, b)) == 14


def test_the_glb_audit_checks_the_roof_lines_records(parts_model):
    gltf, _ = st.read_glb(parts_model.glb)
    sl = gltf["scenes"][0]["extras"]["structure_lines"]
    assert st.audit_structure_lines(sl) == [] and st.audit_glb(parts_model.glb)["structure_lines"] == []
    claimed = {**sl, "lines": [r[:8] + [1] + r[9:] for r in sl["lines"]]}          # verified, without evidence
    assert all("without its evidence" in w for w in st.audit_structure_lines(claimed))
    glb = st.with_scene_extras(parts_model.glb, {"structure_lines": claimed})
    assert not st.audit_glb(glb)["passed"]
    backed = {**claimed, "verified": {str(i): [2, 5, 0.6, 14, 0, 0.0] for i in range(len(sl["lines"]))}}
    assert st.audit_structure_lines(backed) == []
    assert st.audit_structure_lines(sl, stub_ids={"w100"})                         # no line on a stub part
    T = sl["types"]                                                                # a type its own ends contradict
    hip = {**sl, "lines": [r[:6] + [T.index("hip")] + r[7:] if T[r[6]] == "ridge" else r for r in sl["lines"]]}
    assert [w for w in st.audit_structure_lines(hip) if "a hip that is level" in w]


def test_a_roof_line_the_photos_see_displaced_reports_an_offset_not_a_verification(parts_model):
    """verify-r2: the only photo-verified roof line of the Baptistery sat 0.8 m below the eave the held-out photos
    show; its null had copies 1.5 m away only. With near copies (0.4 / 0.8 m), 3-D lines along the edge 0.8 m above
    the drawn line confirm the copy better than the line: line_check status 'offset' (where the photos put it), not
    verified; lines right on it still verify it."""
    from crowd3d.lines3d import Lines3D
    eave = next(r for r in parts_model.summary["structure_lines"]["lines"] if r["type"] == "eave")
    a, b = np.array(eave["a"]), np.array(eave["b"])
    mk = lambda segs: Lines3D(np.array([p for p, q in segs], float), np.array([q for p, q in segs], float),
                              np.full(len(segs), 6), [list(range(6))] * len(segs), np.ones(len(segs)), np.zeros(len(segs)))
    up = np.array([0.0, 0.0, 0.8])
    # most of the photos' edge 0.8 m up, a short piece within the 0.3 m tolerance of the drawn line
    lines = mk([(a + 0.05 * (b - a) + up, a + 0.95 * (b - a) + up), (a + 0.1 * (b - a) + [0, 0, 0.25], a + 0.5 * (b - a) + [0, 0, 0.25])])
    ver, chk = st.verify_segment(a, b, lines)
    assert ver is None and chk["confirmed"] and chk["offset"]["shift"] == "up0.8", chk
    ver, chk = st.verify_segment(a, b, mk([(a + 0.05 * (b - a), a + 0.95 * (b - a))]))
    assert ver is not None and chk["offset"] is None and ver["null"]["alternatives"] == 14


def test_the_glb_audit_checks_every_node_against_its_parts_record(parts_model):
    """The frame in the GLB is what the structure record says was built: a roof apex raised 4 m in the mesh or
    lowered 1.5 m, a wall node moved 1 m along the map, a provenance that is no known value and a part without its
    built record all fail audit_glb."""
    gltf, _ = st.read_glb(parts_model.glb)
    sx = gltf["scenes"][0]["extras"]
    assert st.audit_glb(parts_model.glb)["frame"] == []
    assert all(p.get("built") and p.get("rings") and p.get("outline_source") for p in sx["structure"]["parts"])

    def audit(parts, extras=None):
        return st.audit_glb(st.write_glb(parts, extras or {k: v for k, v in sx.items() if k != "structure_metrics"}))

    assert audit(parts_model.parts)["passed"]
    P = copy.deepcopy(parts_model.parts)
    k = next(i for i, p in enumerate(P) if p.extras.get("kind") == "roof")
    z = P[k].positions[:, 2]
    P[k].positions = P[k].positions.copy()
    P[k].positions[z > z.max() - 0.05, 2] += 4.0
    a = audit(P)
    assert not a["passed"] and any(x["why"].startswith("roof from") for x in a["frame"]), a["frame"]
    P = copy.deepcopy(parts_model.parts)                            # ... or lowered 1.5 m: it no longer reaches its top
    P[k].positions = P[k].positions.copy()
    P[k].positions[z > z.max() - 0.05, 2] -= 1.5
    a = audit(P)
    assert not a["passed"] and any("not reaching" in x["why"] for x in a["frame"]), a["frame"]
    P = copy.deepcopy(parts_model.parts)
    k = next(i for i, p in enumerate(P) if p.extras.get("kind") == "wall")
    w = parts_model.massing.walls[P[k].extras["wall"]]
    P[k].positions = P[k].positions + np.r_[w.tangent, 0.0][None] * 1.0
    a = audit(P)
    assert not a["passed"] and any("off its part's outline" in x["why"] for x in a["frame"]), a["frame"]
    P = copy.deepcopy(parts_model.parts)
    P[k].extras["provenance"] = dict(P[k].extras["provenance"], height="measured")
    a = audit(P)
    assert not a["passed"] and any("not a known value" in x["why"] for x in a["frame"]), a["frame"]
    stripped = copy.deepcopy({k_: v for k_, v in sx.items() if k_ != "structure_metrics"})
    for p in stripped["structure"]["parts"]:
        p.pop("built")
    a = audit(parts_model.parts, stripped)
    assert not a["passed"] and any("without its built heights" in x["why"] for x in a["frame"])


def test_every_roof_crease_is_typed_by_its_own_ends():
    """A crease between two roof faces may be several pieces, some level, some sloped (Frauenkirche's hipped part
    w462645945 on a 35-vertex outline: one face pair meets along 12 pieces, 0.07 m of rise per metre in all). Each
    piece is a ridge when level and a hip when sloped, as the GLB audit checks it (the pair's type made 9 level
    'hips' and failed every Frauenkirche OSM-parts model's audit_glb)."""
    from pathlib import Path

    from crowd3d.geo import LocalENU

    repo = Path(__file__).resolve().parent.parent
    e = next(x for x in json.loads((repo / "eval" / "landmarks.json").read_text()) if x["slug"] == "frauenkirche-dresden")
    enu = LocalENU(e["lat"], e["lon"])
    rings, _ = S.main_outline_from_geojson(repo / e["osm"]["file"], enu)
    specs, _, _ = S.parts_from_osm(S.parse_osm(repo / e["osm_parts"]["file"]), enu, rings, S.DEFAULT_STRUCTURE)
    spec = next(p for p in specs if p.id == "w462645945")
    model = S.build_structure("frauenkirche", [spec], [spec.outer.copy()], {})
    recs = st.structure_lines(st.build_massing_from_structure(model, None, st.DEFAULT_PARAMS))
    assert {r["type"] for r in recs} >= {"ridge", "hip"}
    assert st.audit_structure_lines(st.structure_lines_extras(recs)) == []
    for r in recs:
        if r["type"] in ("ridge", "hip"):
            a, b = np.array(r["a"]), np.array(r["b"])
            assert (abs(b[2] - a[2]) < 0.05 * np.linalg.norm(b - a) + 0.02) == (r["type"] == "ridge"), r
