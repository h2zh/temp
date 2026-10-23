"""crowd3d.structmetrics: structural census, OSM / photo provenance shares, detail score and honesty."""

from __future__ import annotations

import copy

import pytest
from metrics_glb import box_part, recess_wall, write_glb   # (pytest puts tests/ on the path)

from crowd3d import structmetrics as SM

L, D, H = 10.0, 6.0, 10.0


def reg(height_state="map", height_src="osm-s3db", evidence=None, outline="osm-s3db", adoptions=(), roof="flat", rh=0.0):
    hp = {"value": H, "state": height_state, "source": height_src}
    if evidence is not None:
        hp["evidence"] = evidence
    return {"landmark": "t", "adoptions": list(adoptions), "parts": [{
        "id": "p1", "drawn": True, "outline_source": outline, "shape": {"roof": roof, "source": "osm-s3db"},
        "params": {"height": hp, "min_height": {"value": 0.0, "state": "map", "source": "osm-default-ground"},
                   "roof_height": {"value": rh, "state": "map", "source": "osm-s3db"}}}]}


def element(**kw):
    e = {"type": "opening", "kind": "bay", "u0": 2.0, "u1": 4.0, "z_bottom": 1.0, "z_top": 5.0, "depth": 0.3,
         "provenance": "photo-measured",
         "channels": {"sfm_points": True, "split_half": {"iou_a": 0.9, "iou_b": 0.9, "depth_diff_m": 0.01},
                      "lines3d": {"edges_confirmed": 2, "edges": 4, "photos": 12}},
         "photos_per_half": [10, 9], "line_photos": ["a.jpg", "b.jpg", "c.jpg"], "guards": {"G-R1": True},
         "built": True}
    e.update(kw)
    return e


def model(tmp_path, relief=None, observed=True, roof="flat", apex=None, extra_nodes=(), name="m.glb"):
    nodes = box_part("p1", 0, 0, L, D, 0, H, roof=roof, apex=apex)
    if relief is not None:
        nodes[0] = recess_wall("p1", 0, L, H, relief[0]["u0"], relief[0]["u1"], relief[0]["z_bottom"],
                               relief[0]["z_top"], relief[0]["depth"], {"relief": relief, "observed": observed,
                               "provenance": {"outline": "osm-s3db", "height": "osm-s3db",
                                              "appearance": "photo-observed" if observed else "default-style"}})
    nodes += list(extra_nodes)
    p = tmp_path / name
    write_glb(p, nodes)
    return p


def test_flat_box_census_is_all_osm(tmp_path):
    r = SM.structure_metrics(reg(), model(tmp_path))
    c, m = r["census"]["elements_n"], r["metrics"]
    assert c["part"] == 1 and c["wall_face"] == 4 and c["roof_face"] == 1
    assert c["line_corner"] == 4 and c["line_eave"] == 4
    assert m["struct_params"] == 4 and m["struct_share_osm_params"] == 1.0
    assert m["struct_photo_dof"] == 0 and m["struct_unbacked"] == 0
    assert r["shares"]["volume_m3"] == pytest.approx(L * D * H, rel=1e-3)
    assert r["shares"]["area"]["osm"] == 1.0


def test_pyramid_roof_faces_and_hips(tmp_path):
    r = SM.structure_metrics(reg(roof="pyramidal", rh=4.0), model(tmp_path, roof="pyramidal", apex=H + 4.0))
    c = r["census"]["elements_n"]
    assert c["roof_face"] == 4 and c["line_hip"] == 4 and c["line_eave"] == 4 and c["line_corner"] == 4


def test_backed_relief_is_rewarded_and_frame_lines_kept(tmp_path):
    r = SM.structure_metrics(reg(), model(tmp_path, relief=[element()]))
    m, c = r["metrics"], r["census"]["elements_n"]
    assert m["struct_facade_elements"] == 1 and m["struct_unbacked"] == 0
    assert r["detail"]["photo_dof_by_axis"] == {"u": 2, "z": 2, "depth": 1}
    assert m["struct_photo_dof"] == 5
    assert c["line_corner"] == 4 and c["line_eave"] == 4 and c["wall_face"] == 4
    assert r["shares"]["area"]["photo-measured"] > 0
    assert m["struct_share_photo_added_elements"] > 0
    assert r["detail"]["photo_relief_volume_m3"] == pytest.approx(2 * 4 * 0.3)


@pytest.mark.parametrize("bad", [
    {"observed": False},                                              # not photographed
    {"el": {"provenance": "inferred"}},                               # invented
    {"el": {"provenance": "convention"}},
    {"el": {"channels": {"sfm_points": True}}},                       # one channel
    {"el": {"photos_per_half": [3, 20]}},                             # too few photos in one half
    {"el": {"guards": {"G-R1": True, "G-R2": False}}},                # a guard failed
    {"el": {"channels": {"sfm_points": True, "split_half": {"iou_a": 0.9, "iou_b": 0.9, "depth_diff_m": 0.01},
                         "lines3d": {"edges_confirmed": 2, "edges": 4, "photos": 12, "p_chance": 0.5}}}},   # M19: at chance
])
def test_unbacked_elements_are_never_rewarded(tmp_path, bad):
    e = element(**bad.get("el", {}))
    r = SM.structure_metrics(reg(), model(tmp_path, relief=[e], observed=bad.get("observed", True)))
    m = r["metrics"]
    assert m["struct_facade_elements"] == 0 and m["struct_photo_dof"] == 0 and m["struct_photo_edge_m"] == 0
    assert m["struct_unbacked"] == 1
    assert m["struct_share_photo_area"] == pytest.approx(r["shares"]["area"]["unbacked"] * 0 + m["struct_share_photo_area"])


def test_copied_record_on_another_wall_is_not_a_measurement(tmp_path):
    e = element()
    # the same record pasted onto a second (photographed) box's wall
    other = box_part("p2", 20, 0, 20 + L, D, 0, H)
    w = recess_wall("p2", 0, L, H, 2, 4, 1, 5, 0.3, {"relief": [copy.deepcopy(e)]})
    w = (w[0], w[1], [tuple((x + 20, y, z) for x, y, z in t) for t in w[2]])
    other[0] = w
    rg = reg()
    rg["parts"].append(copy.deepcopy(rg["parts"][0]))
    rg["parts"][1]["id"] = "p2"
    r = SM.structure_metrics(rg, model(tmp_path, relief=[e], extra_nodes=other))
    assert r["metrics"]["struct_facade_elements"] == 1
    assert r["metrics"]["struct_photo_dof"] == 5
    assert r["metrics"]["struct_unbacked"] == 1


def test_repeated_values_count_once(tmp_path):
    e1 = element()
    e2 = element(u0=6.0, u1=8.0, line_photos=["d.jpg", "e.jpg"])      # same z levels and depth as e1
    r = SM.structure_metrics(reg(), model(tmp_path, relief=[e1, e2]))
    assert r["detail"]["photo_dof_by_axis"] == {"u": 4, "z": 2, "depth": 1}
    assert r["metrics"]["struct_facade_elements"] == 2


def test_values_on_the_wall_bounds_are_not_measurements(tmp_path):
    e = element(u0=0.0, u1=L, z_bottom=-0.1, z_top=H)                 # a full-wall recess: only its depth is new
    r = SM.structure_metrics(reg(), model(tmp_path, relief=[e]))
    assert r["detail"]["photo_dof_by_axis"] == {"u": 0, "z": 0, "depth": 1}


def test_stub_part_is_stub_and_cannot_carry_elements(tmp_path):
    r = SM.structure_metrics(reg(height_state="stub", height_src="default-style"), model(tmp_path, relief=[element()]))
    m = r["metrics"]
    assert m["struct_facade_elements"] == 0 and m["struct_unbacked"] == 1
    assert r["shares"]["elements_existence"]["stub"] > 0.9
    assert m["struct_share_stub_params"] == 0.25


def test_photo_verified_needs_two_channels_and_adds_no_dof(tmp_path):
    one = [{"kind": "skyline", "skyline_photos": 40}]
    two = one + [{"kind": "lines3d", "n_photos": 20}]
    r1 = SM.structure_metrics(reg(height_state="photo-verified", evidence=one), model(tmp_path))
    assert r1["metrics"]["struct_unbacked"] == 1
    r2 = SM.structure_metrics(reg(height_state="photo-verified", evidence=two), model(tmp_path, name="b.glb"))
    m = r2["metrics"]
    assert m["struct_unbacked"] == 0 and m["struct_photo_dof"] == 0
    assert m["struct_share_photo_checked_params"] == 0.25 and m["struct_share_photo_value_params"] == 0.0
    # a verified (kept OSM) height is never counted as photo share
    assert m["struct_share_photo_elements"] == 0.0 and m["struct_share_photo_area"] == 0.0
    assert m["struct_share_photo_volume"] == 0.0
    assert m["struct_share_photo_checked_area"] > 0 and m["struct_share_photo_checked_elements"] > 0


def test_d15_outline_adoption_counts_three_numbers_only_when_backed(tmp_path):
    ad = {"id": "A1", "kind": "frame", "param": "outline", "value": {"k": 0.9, "t": [1.0, -1.0], "centre": [0, 0]},
          "active": True, "channels": {"sfm-points": {"measured": True, "ok": True}, "lines3d": {"measured": True, "ok": True}}}
    r = SM.structure_metrics(reg(outline="photo-fitted (auto)", adoptions=[ad]), model(tmp_path))
    assert r["metrics"]["struct_photo_dof"] == 3 and r["metrics"]["struct_unbacked"] == 0
    assert r["shares"]["elements_existence"]["osm"] == 1.0           # the elements still exist because OSM maps them
    # per parameter: the outline is photo-fitted, the heights stay OSM's -> neither all photo nor all OSM
    el = r["shares"]["elements"]
    assert 0.2 < el["photo-fitted"] < 0.8 and 0.2 < el["osm"] < 0.8, el
    assert r["detail"]["photo_fit"] == [{"param": "outline", "part": None, "k": 0.9, "tx": 1.0, "ty": -1.0, "numbers": 3,
                                         "channels": ["lines3d", "sfm-points"]}]
    m = r["metrics"]
    assert m["struct_share_photo_elements"] == pytest.approx(el["photo-fitted"]) and m["struct_share_photo_checked_elements"] == 0
    ad1 = copy.deepcopy(ad)
    del ad1["channels"]["lines3d"]
    r = SM.structure_metrics(reg(outline="photo-fitted (auto)", adoptions=[ad1]), model(tmp_path, name="c.glb"))
    assert r["metrics"]["struct_photo_dof"] == 0 and r["metrics"]["struct_unbacked"] == 1


def test_shares_sum_to_one(tmp_path):
    r = SM.structure_metrics(reg(), model(tmp_path, relief=[element()]))
    for k in ("elements", "elements_existence", "params", "area", "area_plan", "area_elevation", "volume"):
        assert sum(r["shares"][k].values()) == pytest.approx(1.0, abs=2e-3), k


def test_glb_only_photo_claims_are_unbacked(tmp_path):
    p = tmp_path / "g.glb"
    write_glb(p, box_part("p1", 0, 0, L, D, 0, H), scene_extras={"structure": {"parts": [
        {"id": "p1", "state": "photo-verified", "roof": "flat",
         "params": {"height": [H, "photo-verified", "osm-s3db"], "min_height": [0.0, "map", "osm-default-ground"],
                    "roof_height": [0.0, "map", "osm-default-ground"]}}]}})
    r = SM.structure_metrics(None, p)
    assert r["metrics"]["struct_unbacked"] == 1 and r["metrics"]["struct_photo_dof"] == 0


def test_metric_specs_cover_flat_metrics(tmp_path):
    r = SM.structure_metrics(reg(), model(tmp_path))
    names = {s["name"] for s in SM.METRIC_SPECS}
    assert set(r["metrics"]) <= names
    for s in SM.METRIC_SPECS:
        assert s["unit"] and s["direction"] in ("higher", "lower", "none") and s["description"]


def test_the_raw_number_share_counts_outline_coordinates_heights_and_photo_numbers(tmp_path):
    """verify-r2: the headline share is the raw number count. A box part: 4 outline vertices (8 coordinates) and 3
    heights from OSM; an adopted outline similarity adds 3 photo numbers (k, tx, ty) while the coordinates stay the
    map's: 3 of 14. Without rings in the GLB the count is not made up (None)."""
    rings = {"structure": {"parts": [{"id": "p1", "rings": [[[0, 0], [L, 0], [L, D], [0, D]]]}]}}
    p = tmp_path / "n.glb"
    write_glb(p, box_part("p1", 0, 0, L, D, 0, H), rings)
    r = SM.structure_metrics(reg(), p)
    assert r["shares"]["numbers_counts"] == {"osm": 11, "photo": 0} and r["metrics"]["struct_numbers"] == 11
    assert r["metrics"]["struct_share_photo_numbers"] == 0.0
    ad = {"param": "outline", "part": "*", "active": True, "value": {"k": 0.82, "t": [3.35, -1.18]},
          "channels": {"sfm-points": {"measured": True}, "lines3d": {"measured": True}}}
    r = SM.structure_metrics(reg(outline="photo-fitted (auto)", adoptions=[ad]), p)
    assert r["shares"]["numbers_counts"] == {"osm": 11, "photo": 3}
    assert r["metrics"]["struct_share_photo_numbers"] == pytest.approx(3 / 14, abs=1e-4)
    assert SM.structure_metrics(reg(), model(tmp_path))["shares"]["numbers"] is None


def test_a_default_flat_roof_is_a_convention_and_a_stubs_roof_a_stub(tmp_path):
    """verify-r2: no roof:shape tag -> the S3DB default (flat) is a convention, not an OSM fact; on a 'height pending'
    stub it is a placeholder."""
    r0 = reg()
    r0["parts"][0]["shape"] = {"roof": "flat", "source": "osm-default-ground"}
    assert SM.structure_metrics(r0, model(tmp_path))["census"]["roof_shape_source"] == {"convention": 1}
    r1 = copy.deepcopy(r0)
    r1["parts"][0]["params"]["height"] = {"value": H, "state": "stub", "source": "default-style"}
    assert SM.structure_metrics(r1, model(tmp_path))["census"]["roof_shape_source"] == {"stub": 1}
