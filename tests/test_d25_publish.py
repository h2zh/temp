"""D25 in the published model (docs/STRUCTURE_FIRST.md §2.4.8 parts 10-11): a correction travels with its record into
the registry, the GLB (scene extras and every node of its part) and structure_metrics; the publish audits re-run its
rule from the record and fail a forged one; the tilt is one node matrix on every node, checked against its record; a
refused registration withdraws the corrections whose box was sized for a registration the model no longer has."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import test_d25 as td
import test_stylize as ts

from crowd3d import correct as C
from crowd3d import judge as J
from crowd3d import structure as S
from crowd3d import stylize as st
from crowd3d import structmetrics as SM

SK, SF, LN = C.SKYLINE, C.SFM, C.LINES


def corrected_box_scene(map_top: float = 5.0) -> st.SceneInput:
    """The photographed 10 m box (south facade and roof) whose map says 5 m, corrected by D25-H from numbers a
    measurement of it gives (skyline 10.0 m, 3-D lines 10.05 m, SfM points 10.0 m all round)."""
    scene = ts.south_scene()
    model = S.build_structure("Test Box", [S.PartSpec("w100", "yes", ts.BOX.copy(), [], {"height": f"{map_top:g}"})],
                              [ts.BOX.copy()], {})
    part = {"id": "w100", "map_top": map_top, "z0": 0.0, "roof_h": 0.0, "pitched": False, "min_width": 12.0}
    ch = {SK: td.sky(top=10.0, p05=9.8, p95=10.2), SF: {"quad": td.quad((10.0, 10.0, 10.0, 10.0))},
          LN: td.lines(10.05, (9.95, 10.15))}
    pre = C.rule_h_pre(part, ch)
    box = td.make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.02,
                      halves={c: 0.1 for c in pre["agreeing"]})
    fin = C.rule_h_box(pre, box, td.SCALE, part)
    assert fin["fires"] and fin["adopted"] == "measured", fin["why"]
    C.apply_height(model, "w100", pre, fin, ch, box, td.SCALE)
    scene.structure = model
    return scene


@pytest.fixture(scope="module")
def corrected_model():
    scene = corrected_box_scene()
    return scene, st.stylize_landmark(scene)


def test_a_corrected_height_is_published_with_its_record_and_passes_every_check(corrected_model):
    scene, m = corrected_model
    s = m.summary
    assert s["registration"]["accepted"] and s["qa"]["passed"], s["qa"]
    assert s["qa"]["glb"]["d25"] == [] and s["qa"]["audit"]["d25"] == [] and s["qa"]["structure"]["passed"]
    reg = s["structure"]
    h = next(p for p in reg["parts"] if p["id"] == "w100")["params"]["height"]
    assert (h["value"], h["state"], h["source"], h["map_value"]) == (pytest.approx(10.025, abs=0.01), S.STATE_CORRECTED,
                                                                      C.PROV_FULL, 5.0)
    nodes = [p for p in m.parts if p.extras.get("part") == "w100"]
    assert nodes and all(p.extras["provenance"]["height"] == C.PROV_FULL for p in nodes if p.extras.get("kind") == "wall")
    (cor,) = nodes[0].extras["corrected"]
    assert cor["map"] == 5.0 and cor["adopted"] == "measured" and cor["record"].startswith("D25-")
    assert cor["range"][0] < cor["value"] < cor["range"][1]
    gx = st.read_glb(m.glb)[0]["scenes"][0]["extras"]["structure"]
    (rec,) = [a for a in gx["adoptions"] if a["kind"] == C.KIND_H]
    assert rec["osm_edited"] is False and rec["registration_box"]["samples"] and C.check_record(rec) is None
    sm = s["structure_metrics"]
    assert sm["honesty"]["unbacked"] == 0 and sm["shares"]["params_counts"].get("photo-corrected") == 1
    assert sm["shares"]["numbers_counts"].get("photo", 0) >= 1
    assert any(f.get("kind") == C.KIND_H for f in sm["detail"]["photo_fit"])
    # the OSM fix report for mappers
    rep = C.osm_fix_suggestions(scene.structure)
    assert rep["never_uploaded"] and rep["suggestions"][0]["current"] == 5.0


def _replace_structure(glb: bytes, mutate) -> bytes:
    gltf, _ = st.read_glb(glb)
    sx = copy.deepcopy(gltf["scenes"][0]["extras"]["structure"])
    mutate(sx)
    return st.with_scene_extras(glb, {"structure": sx})


@pytest.mark.parametrize("forge,why", [
    (lambda sx: next(a for a in sx["adoptions"] if a["kind"] == C.KIND_H).update(V=30.0), "V"),
    (lambda sx: next(a for a in sx["adoptions"] if a["kind"] == C.KIND_H)["registration_box"].update(samples=[]), "box"),
    (lambda sx: next(a for a in sx["adoptions"] if a["kind"] == C.KIND_H)["scale"].update(established=False), "scale"),
    (lambda sx: sx.update(adoptions=[]), "no record"),
    (lambda sx: next(p for p in sx["parts"] if p["id"] == "w100")["params"]["height"].__setitem__(0, 12.0), "value"),
    (lambda sx: next(a for a in sx["adoptions"] if a["kind"] == C.KIND_H).update(provenance=C.PROV_LB), "provenance"),
])
def test_a_forged_correction_in_the_glb_fails_the_glb_audit(corrected_model, forge, why):
    _, m = corrected_model
    assert st.audit_glb(m.glb)["passed"]
    bad = st.audit_glb(_replace_structure(m.glb, forge))
    assert not bad["passed"] and bad["d25"], why


@pytest.mark.parametrize("param,value", [("min_height", 0.0), ("roof_height", 0.0)])
def test_a_photo_corrected_parameter_without_its_own_record_or_carry_fails_the_glb_audit(corrected_model, param, value):
    """verify-r1 F1b: once any D25 record was active, audit_glb exempted every other photo-corrected parameter as
    'carried' without checking. Now only a carry listed by an active D25-H record of an ancestor is accepted."""
    _, m = corrected_model

    def forge(sx):
        next(p for p in sx["parts"] if p["id"] == "w100")["params"][param] = [value, "photo-corrected", C.PROV_FULL]
    bad = st.audit_glb(_replace_structure(m.glb, forge))
    assert not bad["passed"] and any(v.get("param") == param for v in bad["d25"]), bad["d25"]


def test_the_glb_carry_check_follows_the_published_parent_chain():
    rec = {"id": "D25-x-1", "kind": C.KIND_H, "part": "w2", "param": "height", "value": 30.0, "judged_against": 20.0,
           "carried": [{"part": "w3", "dz": 10.0}],
           "before": [{"part": "w2", "param": "height", "value": 20.0}, {"part": "w3", "param": "height", "value": 24.0},
                      {"part": "w3", "param": "min_height", "value": 20.0}]}
    parents = {"w1": None, "w2": None, "w3": "w2", "w4": "w3"}
    assert st._audit_glb_d25_carry([rec], parents, "w3", "height", 34.0, C.PROV_FULL) is None
    assert "snapshot plus" in st._audit_glb_d25_carry([rec], parents, "w3", "height", 36.0, C.PROV_FULL)
    assert "does not rest" in st._audit_glb_d25_carry([rec], dict(parents, w3=None), "w3", "height", 34.0, C.PROV_FULL)
    assert "without a matching" in st._audit_glb_d25_carry([rec], parents, "w1", "height", 24.0, C.PROV_FULL)
    assert "provenance" in st._audit_glb_d25_carry([rec], parents, "w3", "height", 34.0, C.PROV_LB)
    assert "only heights" in st._audit_glb_d25_carry([rec], parents, "w3", "roof_height", 3.0, C.PROV_FULL)
    deep = dict(rec, carried=[{"part": "w4", "dz": 10.0}], before=[{"part": "w4", "param": "height", "value": 30.0}])
    assert st._audit_glb_d25_carry([deep], parents, "w4", "height", 40.0, C.PROV_FULL) is None      # a grandchild
    assert "change" in st._audit_glb_d25_carry([dict(deep, carried=[{"part": "w4", "dz": 9.0}])], parents, "w4",
                                               "height", 39.0, C.PROV_FULL)


def test_a_correction_without_its_evidence_fails_the_publish_gate():
    scene = corrected_box_scene()
    rec = scene.structure.adoptions[-1]
    rec["channels"][SK]["masks"]["A+R"]["n_vis_at"] = 2          # the skyline no longer a channel: one left
    m = st.stylize_landmark(scene)
    assert not m.summary["qa"]["passed"] and not m.summary["qa"]["structure"]["passed"]
    assert m.summary["qa"]["audit"]["d25"] and m.summary["qa"]["glb"]["d25"]


@pytest.mark.parametrize("refusal,kept", [("snap rejected, georegistered", False), ("prior only", False)])
def test_a_refused_registration_withdraws_corrections_sized_for_a_registration_the_model_lost(refusal, kept):
    scene = corrected_box_scene()
    if refusal == "prior only":
        scene.snap_prior = st.SnapPrior(georegistered=False)
    else:
        scene.snap_prior = st.SnapPrior(georegistered=True)
        scene.snap_override = st.FootprintSnap(0.0, np.zeros(2), 0.0, 0.0, len(scene.points), False, "test: rejected")
    m = st.stylize_landmark(scene)
    s = m.summary
    assert not s["registration"]["accepted"] and s["qa"]["passed"], s["qa"]
    h = next(p for p in s["structure"]["parts"] if p["id"] == "w100")["params"]["height"]
    assert (h["value"], h["source"]) == (5.0, st.PROV_S3DB)
    assert not any(a["active"] for a in s["structure"]["adoptions"])
    assert scene.structure.adoptions[-1]["active"]                 # the caller's structure is left as it was


def test_a_photo_measured_block_is_flat_and_published_with_its_record():
    import test_d15 as t15
    scene = t15.photographed_stub_scene()
    model = scene.structure
    stub = next(p for p in model.specs if p.kind == "remainder")
    part = {"id": stub.id, "min_width": stub.min_width(), "z0": 0.0, "map_top": 3.0}
    read = {LN: td.blk_lines(9.9, n_lines=5, n_photos=6), SF: td.blk_sfm(9.95), SK: td.blk_sky(9.7)}
    box = td.make_box({c: r["value"] for c, r in read.items()}, sens=0.02, halves={c: 0.0 for c in read})
    res = C.rule_b(part, read, 10.4, box, td.SCALE)
    assert res["fires"], res["why"]
    C.apply_block(model, stub.id, res, read, box, td.SCALE)
    m = st.stylize_landmark(scene)
    s = m.summary
    assert s["qa"]["passed"], s["qa"]
    p = next(x for x in s["structure"]["parts"] if x["id"] == stub.id)
    assert p["params"]["height"]["source"] == C.PROV_LB and p["shape"]["roof"] == "flat"
    nodes = [q for q in m.parts if q.extras.get("part") == stub.id]
    assert nodes[0].extras["corrected"][0]["kind"] == C.KIND_BLOCK
    bad = st.audit_glb(_replace_structure(m.glb, lambda sx: next(r for r in sx["parts"] if r["id"] == stub.id)
                                          .update(roof="gabled")))
    assert any("roof shape" in v["why"] for v in bad["d25"])


def test_the_tilt_is_one_matrix_on_every_node_and_the_glb_audit_checks_it():
    """A round tower tilted by D25-T (a crowd-grade registration: kept when the colour gate refuses but the model is
    georegistered); every node carries the pose's matrix, its vertices stay upright and match the records; a node
    with another matrix, a pose without its record, or a tilt on a box fail the GLB audit; reverting gives the upright
    tower's GLB nodes back without any matrix."""
    scene = ts.south_scene()
    ring = td.circle(8.0)
    model = S.build_structure("Tower", [S.PartSpec("w1", "yes", ring.copy(), [], {"height": "56"})], [ring.copy()], {})
    reg = {"grade": C.GRADE_CROWD, "reported_yaw_sigma_deg": 1.0}
    res = C.rule_t(td.TILT_CH, td.TILT_G, C.tilt_sigma_yaw(reg), C.tilt_shape_gate(S.ccw(model.main_rings[0]), 56.0))
    assert res["fires"], res["why"]
    rec = C.apply_tilt(model, res, td.TILT_CH, td.TILT_G, registration=reg,
                       consensus=C.tilt_consensus([dict(res, reconstruction="r")]))
    scene.structure = model
    scene.footprint = ring.copy()
    scene.snap_prior = st.SnapPrior(georegistered=True)
    scene.snap_override = st.FootprintSnap(0.0, np.zeros(2), 0.0, 0.0, len(scene.points), False, "test: rejected")
    m = st.stylize_landmark(scene)
    assert m.summary["qa"]["passed"], m.summary["qa"]
    gltf, _ = st.read_glb(m.glb)
    want = st.pose_node_matrix(model.pose)
    assert want is not None and all(n.get("matrix") == want for n in gltf["nodes"])
    sx = gltf["scenes"][0]["extras"]["structure"]
    assert sx["pose"]["adoption"] == rec["id"] and m.parts[0].extras["tilt"]["tilt_deg"] == model.pose["tilt_deg"]
    assert m.summary["structure_metrics"]["detail"]["photo_dof"] >= 2
    # one node with another matrix
    g2, b2 = st.read_glb(m.glb)
    g2["nodes"][0]["matrix"] = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    js = json.dumps(g2, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    import struct
    forged = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(b2)) + struct.pack("<II", len(js), 0x4E4F534A) \
        + js + struct.pack("<II", len(b2), 0x004E4942) + b2
    assert any("matrix" in v["why"] for v in st.audit_glb(forged)["d25"])
    # verify-r1 T1: the pivot moved 200 m up, consistently in the pose, the record and every node matrix
    g3, b3 = st.read_glb(m.glb)
    sx3 = g3["scenes"][0]["extras"]["structure"]
    sx3["pose"]["pivot"] = [sx3["pose"]["pivot"][0], sx3["pose"]["pivot"][1], sx3["pose"]["pivot"][2] + 200.0]
    next(a for a in sx3["adoptions"] if a["kind"] == C.KIND_TILT)["value"]["pivot"] = list(sx3["pose"]["pivot"])
    for n in g3["nodes"]:
        n["matrix"] = st.pose_node_matrix(sx3["pose"])
    js = json.dumps(g3, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    forged = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(b3)) + struct.pack("<II", len(js), 0x4E4F534A) \
        + js + struct.pack("<II", len(b3), 0x004E4942) + b3
    assert any("pivot" in v["why"] for v in st.audit_glb(forged)["d25"])
    # a pose without its record
    bad = st.audit_glb(_replace_structure(m.glb, lambda s: s.update(adoptions=[])))
    assert any("pose" in v["why"] or "record" in v["why"] for v in bad["d25"])
    # reverted: no matrix anywhere, the same upright vertices
    J.revert_adoption(model, rec["id"])
    m2 = st.stylize_landmark(scene)
    g3, b3 = st.read_glb(m2.glb)
    assert not any("matrix" in n for n in g3["nodes"]) and m2.summary["qa"]["passed"]
    pos = lambda g, b: np.concatenate([st.glb_accessor(g, b, g["meshes"][n["mesh"]]["primitives"][0]["attributes"]["POSITION"])
                                       for n in g["nodes"]])
    assert np.array_equal(pos(g3, b3), pos(*st.read_glb(m.glb)))


def test_the_tilt_on_a_box_fails_the_glb_audit():
    scene = ts.south_scene()
    ring = td.circle(8.0)
    model = S.build_structure("Tower", [S.PartSpec("w1", "yes", ring.copy(), [], {"height": "56"})], [ring.copy()], {})
    reg = {"grade": C.GRADE_CROWD, "reported_yaw_sigma_deg": 1.0}
    res = C.rule_t(td.TILT_CH, td.TILT_G, C.tilt_sigma_yaw(reg), C.tilt_shape_gate(S.ccw(model.main_rings[0]), 56.0))
    assert res["fires"], res["why"]
    rec = C.apply_tilt(model, res, td.TILT_CH, td.TILT_G, registration=reg,
                       consensus=C.tilt_consensus([dict(res, reconstruction="r")]))
    box = S.build_structure("Box", [S.PartSpec("w1", "yes", ts.BOX.copy(), [], {"height": "56"})], [ts.BOX.copy()], {})
    box.adoptions.append(copy.deepcopy(rec))
    box.pose = copy.deepcopy(model.pose)
    scene.structure = box
    scene.snap_prior = st.SnapPrior(georegistered=True)
    scene.snap_override = st.FootprintSnap(0.0, np.zeros(2), 0.0, 0.0, len(scene.points), False, "test: rejected")
    m = st.stylize_landmark(scene)
    assert not m.summary["qa"]["passed"]
    assert any("shape gate" in v["why"] for v in m.summary["qa"]["glb"]["d25"])
