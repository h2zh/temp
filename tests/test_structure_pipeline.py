"""crowd3d.structure_pipeline: the one entry that judges a landmark's OSM parts with one model's photos and stylizes it
(product masks: A+R only). Scenes: the rendered octagon of tests/test_lines3d.py (truth eave 18 m, apex 30 m; the
snapshot maps 15 / 27 m) and the south-facade box of tests/test_stylize.py."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import test_lines3d as tl            # rendered photos, triangulated points
import test_stylize as ts
from test_structure import ENU, LAT, LON, Osm

from crowd3d import correct as C
from crowd3d import structure_pipeline as SP
from crowd3d import stylize as st


def write_snapshot(folder: Path, top: float = 27.0, eave: float = 15.0) -> tuple[Path, Path]:
    """An OSM snapshot of the octagon (one building:part, the main building way) and its main outline GeoJSON."""
    o = Osm()
    ring = [tuple(p) for p in tl.octagon()]
    o.way(ring, {"building:part": "yes", "height": f"{top:g}", "roof:shape": "pyramidal",
                 "roof:height": f"{top - eave:g}"}, wid=10)
    o.way(ring, {"building": "yes", "height": f"{top:g}"}, wid=11)
    osm, outline = folder / "oct.parts.osm", folder / "oct.geojson"
    osm.write_bytes(o.xml())
    lonlat = [list(ENU.to_geodetic(x, y, 0.0)[:2])[::-1] for x, y in ring]
    outline.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"osm_type": "way", "osm_id": 11},
        "geometry": {"type": "Polygon", "coordinates": [lonlat + [lonlat[0]]]}}]}))
    return osm, outline


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> SP.StructureSource:
    d = tmp_path_factory.mktemp("snapshot")
    osm, outline = write_snapshot(d)
    catalog = d / "eval" / "landmarks.json"
    catalog.parent.mkdir()
    catalog.write_text(json.dumps([{"slug": "oct", "osm": {"file": outline.name},          # relative to the repo root
                                    "osm_parts": {"file": osm.name, "sha256": SP._sha256(osm)}}]))
    src = SP.find_source("oct", LAT, LON, catalog=catalog)
    assert src is not None and src.osm_parts.resolve() == osm.resolve() and src.outline.resolve() == outline.resolve()
    assert SP.find_source("elsewhere", LAT, LON, catalog=catalog) is None      # bound by slug only
    return src


@pytest.fixture(scope="module")
def oct_scene() -> st.SceneInput:
    truth = tl.oct_model()
    cams = tl.ring(np.linspace(0, 360, 24, endpoint=False))
    P, q = tl.sfm_points(truth, cams, tl.render(truth, cams))
    return st.SceneInput("oct", tl.octagon(), P, np.full((len(P), 3), 180, np.uint8), cams, snap=False, point_quality=q)


@pytest.fixture(scope="module")
def judged(oct_scene, source, tmp_path_factory):
    """Judged once. The cache already holds the sky of 23 photos (ideal masks, to save 15 s of A+R on renders: with
    one mask the skyline says nothing either way); the first photo's A+R is computed."""
    cache = SP.PhotoCache(tmp_path_factory.mktemp("cache"))
    sky = tl.ideal_sky(tl.oct_model(), oct_scene.cameras)
    for i, cam in enumerate(oct_scene.cameras[1:], 1):
        f = cache.mask_file(cam)
        f.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(f), sky[i].astype(np.uint8) * 255)
    srun = SP.build_structure(oct_scene, source, cache)
    return srun, cache


def test_no_source_is_stylize_landmark_unchanged():
    """(a) Without a snapshot the entry is the footprint stylization, byte for byte."""
    scene = ts.south_scene()
    assert SP.run(scene, None).glb == st.stylize_landmark(ts.south_scene()).glb


def test_an_unregistered_model_gets_the_osm_parts_as_clay_and_nothing_is_judged(source, oct_scene, tmp_path):
    """(b) Registration refused (a prior-only model): no mask, no 3-D lines, no judge; the OSM parts are the massing,
    every wall clay, no number from the photos."""
    scene = replace(oct_scene, snap_prior=st.SnapPrior(georegistered=False))
    cache = SP.PhotoCache(tmp_path)
    m = SP.run(scene, source, cache)
    rec = m.summary["structure_run"]
    assert m.summary["massing_source"] == "osm-parts" and m.summary["observed_walls"] == 0
    assert rec["judge"]["status"] == "skipped" and "registration not accepted" in rec["judge"]["reason"]
    assert rec["masks"] == {} and rec["lines3d"] is None and rec["adoptions"] == [] and not any(tmp_path.iterdir())
    assert cache.stats == {"masks_read": 0, "masks_computed": 0}
    sm = m.summary["structure_metrics"]["metrics"]
    assert sm["struct_share_photo_numbers"] == 0.0 and sm["struct_unbacked"] == 0 and m.summary["qa"]["passed"]
    assert rec["limits"]["within"] and rec["total"]["seconds"] >= rec["seconds"]


def test_a_registered_model_is_judged_with_the_one_licensed_mask(judged):
    """(c) Registered: the judge runs with exactly one sky mask (A+R); the skyline channel says so, the frame's scale
    check has one channel, and D25 does not run (G5 needs two masks). What D15 adopts comes from the SfM points and the
    3-D lines with the part's own roof reading (D27): the octagon's apex at 30 m (the map says 27 m)."""
    srun, _ = judged
    rec = srun.record
    assert rec["masks"] == {"A+R": 24} and rec["judge"]["status"] == "ran" and rec["judge"]["parts_judged"] == 1
    assert any("one sky mask only" in w for w in rec["judge"]["skyline"]), rec["judge"]
    assert rec["d25"]["ran"] is False and "G5 needs 2 sky masks" in rec["d25"]["why"]
    assert rec["gate"] == {"check_decisions": True, "check_structure": True, "revert_all_equals_map": True}
    ad = [a for a in rec["adoptions"] if a["active"]]
    assert [(a["part"], a["rule"]) for a in ad] == [("w10", "D15")], rec["adoptions"]
    assert set(ad[0]["channels"]) >= {"sfm-points", "lines3d"} and "skyline" not in ad[0]["channels"]
    assert srun.structure.built_part("w10").top == pytest.approx(30.0, abs=1.0)
    json.dumps(rec)                                                  # JSON-safe


def test_the_judged_structure_is_published_and_reverts_to_the_map(judged, oct_scene, source):
    """(e) The adoption reaches the GLB with its provenance; reverting it gives back the mapped structure."""
    srun, _ = judged
    m = SP.stylize(oct_scene, srun)
    assert m.summary["qa"]["passed"] and m.summary["massing_source"] == "osm-parts"
    reg = m.summary["structure"]
    assert reg["adoptions_active"] == 1 and m.summary["structure_metrics"]["metrics"]["struct_unbacked"] == 0
    mapped = SP.load_structure(source)
    assert SP._params_of(srun.structure) != SP._params_of(mapped) and SP.revert_all_equals_map(srun.structure, mapped)
    rv = copy.deepcopy(srun.structure)
    for a in reversed(rv.adoptions):
        SP.J.revert_adoption(rv, a["id"])
    assert rv.built_part("w10").top == pytest.approx(mapped.built_part("w10").top)


def test_the_photo_cache_is_hit_on_the_second_run(judged, oct_scene, source):
    """(f) Masks and line segments are computed once per photo; a second model of the same photos reads them."""
    srun, cache = judged
    assert cache.stats == {"masks_read": 23, "masks_computed": 1}
    first = cv2.imread(str(cache.mask_file(oct_scene.cameras[0])), cv2.IMREAD_GRAYSCALE) > 127
    assert np.array_equal(first, srun.masks["A+R"][0]) and first.shape == (480, 640)
    # (A+R's GrabCut draws from OpenCV's RNG: a recomputation differs in a few pixels, the cache keeps the first)
    assert (first == SP.PRODUCT_MASKS["A+R"](oct_scene.cameras[0].load())).mean() > 0.999
    segs = sorted(p.name for p in cache.segments_dir.iterdir())
    assert segs and all(re.fullmatch(r"lsd-c\d+-640x480\.npy", s) for s in segs)
    again = SP.build_structure(oct_scene, source, cache)
    assert cache.stats == {"masks_read": 47, "masks_computed": 1}
    assert sorted(p.name for p in cache.segments_dir.iterdir()) == segs
    assert again.record["adoptions"] == srun.record["adoptions"]


def test_only_the_licensed_mask_is_accepted(oct_scene):
    """(d) crowd3d computes and accepts A+R only: another name, or the same mask under two names, is refused."""
    with pytest.raises(ValueError, match="not licensed"):
        SP.check_mask_names(["A+R", "Dm+R"])
    with pytest.raises(ValueError, match="not licensed"):
        SP.build_structure(oct_scene, None, limits=SP.Limits(masks=("A+R", "A+R copy")))
    with pytest.raises(ValueError, match="once"):
        SP.check_mask_names(["A+R", "A+R"])
    with pytest.raises(ValueError, match="not licensed"):
        SP.PhotoCache(None).masks(oct_scene.cameras[:1], "B0")
    assert list(SP.PRODUCT_MASKS) == ["A+R"] and C.G5_MIN_MASKS > len(SP.PRODUCT_MASKS)


def test_no_crowd3d_code_loads_a_learned_sky_mask():
    """The learned masks (onnx models) are research only: nothing in crowd3d references them."""
    pkg = Path(SP.__file__).parent
    for f in pkg.glob("*.py"):
        text = f.read_text(encoding="utf-8").lower()
        assert "onnx" not in text and "dm+r" not in text, f


def test_limits_skip_the_photo_stages_and_say_why(oct_scene, source):
    """Past the time budget no photo stage starts; above the part limit nothing is judged (OSM-parts massing)."""
    m = SP.run(oct_scene, source, limits=SP.Limits(max_seconds=0.0))
    rec = m.summary["structure_run"]
    assert rec["judge"]["status"] == "skipped" and rec["judge"]["reason"].startswith("time limit")
    assert rec["masks"] == {} and not rec["limits"]["within"] and rec["limits"]["over"][0].startswith("time")
    rec = SP.build_structure(oct_scene, source, limits=SP.Limits(max_parts_judged=0)).record
    assert rec["judge"]["reason"] == "1 built parts > 0: OSM-parts massing as mapped" and rec["masks"] == {}
    with pytest.raises(ValueError, match="d25"):
        SP.build_structure(oct_scene, source, limits=SP.Limits(d25="on"))


def test_the_cli_takes_the_structure_path_when_the_landmark_has_a_snapshot(tmp_path):
    """scripts/stylize_model.py: the snapshot's parts are the massing by default (here a job whose registration is
    refused: clay, nothing judged, said in structure_run); --no-structure keeps the footprint massing."""
    pycolmap = pytest.importorskip("pycolmap")
    import stylize_model as sm
    from PIL import Image

    osm, outline = write_snapshot(tmp_path)
    catalog = tmp_path / "landmarks.json"
    catalog.write_text(json.dumps([{"slug": "oct", "name": "Octagon", "lat": LAT, "lon": LON,
                                    "osm": {"file": str(outline)},
                                    "osm_parts": {"file": str(osm), "sha256": SP._sha256(osm)}}]))
    job = tmp_path / "job"
    (job / "sparse" / "0").mkdir(parents=True)
    (job / "images").mkdir()
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera(model="SIMPLE_PINHOLE", width=64, height=48,
                                                    params=[50.0, 32.0, 24.0], camera_id=1))
    cams = {}
    for i in range(4):
        im = pycolmap.Image(name=f"{i:05d}_p{i}.jpg", camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(im, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)),
                                                              np.array([-4.0 * i, 30.0, 0.0])))
        Image.new("RGB", (64, 48), (120, 120, 120)).save(job / "images" / im.name)
        cams[f"p{i}"] = {"enu": [4.0 * i, -30.0, 0.0], "gps_en": [4.0 * i, -30.0], "flags": []}
    rec.add_point3D(np.array([0.0, 0.0, 5.0]), pycolmap.Track(), np.array([10, 20, 30], np.uint8))
    rec.write(str(job / "sparse" / "0"))
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"georegistered": True, "yaw_sigma_deg": 4.0, "scale_sigma": 0.02, "cameras": cams}))
    common = ["--job-dir", str(job), "--meta", str(meta), "--landmark", "oct", "--catalog", str(catalog), "--no-debug"]
    assert sm.main(common + ["--out", str(tmp_path / "on")]) == 0
    on = json.loads((tmp_path / "on" / "oct.json").read_text())
    run = on["structure_run"]
    assert on["massing_source"] == "osm-parts" and run["source"]["osm_parts_sha256"] == SP._sha256(osm)
    assert run["judge"]["status"] == "skipped" and not on["registration"]["accepted"]
    assert run["d25"] == {"mode": "auto", "ran": False, "records": [], "why": "nothing was judged"}
    assert sm.main(common + ["--out", str(tmp_path / "off"), "--no-structure"]) == 0
    off = json.loads((tmp_path / "off" / "oct.json").read_text())
    assert off["massing_source"] == "height-field" and "structure_run" not in off
    with pytest.raises(SystemExit):
        sm.main(common + ["--out", str(tmp_path / "x"), "--no-structure", "--osm-parts", str(osm)])
