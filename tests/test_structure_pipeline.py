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

pytestmark = pytest.mark.structure


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
    model, srun = SP.run(scene, None)
    assert srun is None and model.glb == st.stylize_landmark(ts.south_scene()).glb


def test_an_unregistered_model_gets_the_osm_parts_as_clay_and_nothing_is_judged(source, oct_scene, tmp_path):
    """(b) Registration refused (a prior-only model): no mask, no 3-D lines, no judge; the OSM parts are the massing,
    every wall clay, no number from the photos."""
    scene = replace(oct_scene, snap_prior=st.SnapPrior(georegistered=False))
    cache = SP.PhotoCache(tmp_path)
    m, _ = SP.run(scene, source, cache)
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
    assert again.record["cache"] == {"masks_read": 24, "masks_computed": 0}        # per model
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
    m, _ = SP.run(oct_scene, source, limits=SP.Limits(max_seconds=0.0))
    rec = m.summary["structure_run"]
    assert rec["judge"]["status"] == "skipped" and rec["judge"]["reason"].startswith("time limit")
    assert rec["masks"] == {} and not rec["limits"]["within"] and rec["limits"]["over"][0].startswith("time")
    rec = SP.build_structure(oct_scene, source, limits=SP.Limits(max_parts_judged=0)).record
    assert rec["judge"]["reason"] == "1 built parts > 0: OSM-parts massing as mapped" and rec["masks"] == {}
    with pytest.raises(ValueError, match="d25"):
        SP.build_structure(oct_scene, source, limits=SP.Limits(d25="on"))


def tiny_job(folder: Path) -> tuple[Path, Path]:
    """A kept job dir (sparse/0, images) of 4 photos 30 m south of the landmark centre and 1 point, and its model
    JSON (georegistered): too little for the snap, so its registration is refused."""
    pycolmap = pytest.importorskip("pycolmap")
    from PIL import Image

    job = folder / "job"
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
    meta = folder / "m.json"
    meta.write_text(json.dumps({"georegistered": True, "yaw_sigma_deg": 4.0, "scale_sigma": 0.02, "cameras": cams}))
    return job, meta


def test_the_cli_takes_the_structure_path_when_the_landmark_has_a_snapshot(tmp_path):
    """scripts/stylize_model.py: the snapshot's parts are the massing by default (here a job whose registration is
    refused: clay, nothing judged, said in structure_run); --no-structure keeps the footprint massing."""
    import stylize_model as sm

    osm, outline = write_snapshot(tmp_path)
    catalog = tmp_path / "landmarks.json"
    catalog.write_text(json.dumps([{"slug": "oct", "name": "Octagon", "lat": LAT, "lon": LON,
                                    "osm": {"file": str(outline)},
                                    "osm_parts": {"file": str(osm), "sha256": SP._sha256(osm)}}]))
    job, meta = tiny_job(tmp_path)
    common = ["--job-dir", str(job), "--meta", str(meta), "--landmark", "oct", "--catalog", str(catalog), "--no-debug"]
    assert sm.main(common + ["--out", str(tmp_path / "on")]) == 0
    on = json.loads((tmp_path / "on" / "oct.json").read_text())
    run = on["structure_run"]
    assert on["massing_source"] == "osm-parts" and run["source"]["osm_parts_sha256"] == SP._sha256(osm)
    assert run["judge"]["status"] == "skipped" and not on["registration"]["accepted"]
    assert run["d25"] == {"mode": "auto", "ran": False, "records": [], "why": "nothing was judged"}
    assert sm.main([c for c in common if c != "--no-debug"] + ["--out", str(tmp_path / "off"), "--no-structure"]) == 0
    off = json.loads((tmp_path / "off" / "oct.json").read_text())
    assert off["massing_source"] == "height-field" and "structure_run" not in off
    assert off["debug_files"] == ["oct-topdown.png"] and (tmp_path / "off" / "oct-topdown.png").is_file()
    with pytest.raises(SystemExit):
        sm.main(common + ["--out", str(tmp_path / "x"), "--no-structure", "--osm-parts", str(osm)])


def test_the_product_child_stylizes_a_saved_map_and_never_opens_the_database(tmp_path):
    """run_model, as crowd3d.jobs runs it (`python -m crowd3d.structure_pipeline run --spec`): a saved map's sparse
    model and the photos' processed copies by image name (one photo missing: its camera is left out); it writes only
    into its out dir (GLB, summary, result) and the photo cache, and imports neither crowd3d.store nor crowd3d.jobs."""
    import subprocess
    import sys

    osm, outline = write_snapshot(tmp_path)
    catalog = tmp_path / "landmarks.json"
    catalog.write_text(json.dumps([{"slug": "oct", "name": "Octagon", "lat": LAT, "lon": LON,
                                    "osm": {"file": str(outline), "attribution": "outline (ODbL)"},
                                    "osm_parts": {"file": str(osm), "sha256": SP._sha256(osm),
                                                  "attribution": "parts (ODbL)"}}]))
    job, meta = tiny_job(tmp_path)
    images = {p.name: str(p) for p in sorted((job / "images").iterdir())[1:]}
    spec = {"catalog": str(catalog), "slug": "oct", "landmark": {"id": "oct", "name": "Octagon", "lat": LAT, "lon": LON},
            "model": {"id": "m1", "meta": str(meta), "sparse": str(job / "sparse" / "0"), "images": images},
            "out": str(tmp_path / "out"), "cache": str(tmp_path / "cache"), "max_seconds": 1200, "threads": 2}
    root = Path(SP.__file__).resolve().parent.parent
    probe = ("import sys; from crowd3d.structure_pipeline import main; rc = main(sys.argv[1:]); "
             "print([m for m in sys.modules if m in ('crowd3d.store', 'crowd3d.jobs')]); sys.exit(rc)")

    def child(spec_: dict) -> tuple[int, str, dict]:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec_))
        done = subprocess.run([sys.executable, "-c", probe, "run", "--spec", str(path)], cwd=root,
                              capture_output=True, text=True, timeout=300)
        return done.returncode, done.stdout, json.loads((Path(spec_["out"]) / "result.json").read_text())

    rc, out, result = child(spec)
    assert rc == 0 and out.strip().splitlines()[-1] == "[]", out
    assert result["ok"] and result["error"] is None and result["registration"]["accepted"] is False
    assert (result["massing_source"], result["parts_built"], result["judge"]["status"]) == ("osm-parts", 1, "skipped")
    assert result["source"] == {"slug": "oct", "osm_parts_sha256": SP._sha256(osm)} and result["within_limits"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["m1.glb", "m1.style.json", "result.json"]
    summary = json.loads((tmp_path / "out" / "m1.style.json").read_text())
    assert summary["source"]["camera_fit_residual_max_m"] < 0.01 and summary["observed_walls"] == 0
    assert result["snap"] == summary["snap"]             # the job runner records it for the public photo poses (D30)
    gltf, _ = st.read_glb((tmp_path / "out" / "m1.glb").read_bytes())
    extras = gltf["scenes"][0]["extras"]
    assert extras["attribution"][:3] == ["outline (ODbL)", "parts (ODbL)", SP.PHOTO_ATTRIBUTION], extras["attribution"]
    assert extras["registration"]["accepted"] is False
    # a slug the catalogue does not list: no GLB, the error in the result, exit status 1
    rc, _, result = child({**spec, "slug": "elsewhere", "out": str(tmp_path / "out2")})
    assert rc == 1 and not result["ok"] and "no OSM parts snapshot" in result["error"]
    assert sorted(p.name for p in (tmp_path / "out2").iterdir()) == ["result.json"]
    # (a failure the step expects: not marked unexpected; its traceback is kept like any other's)
    assert "unexpected" not in result and result["error_type"] == "ValueError" and result["traceback"][-1] == result["error"]
    # an exception nobody expected (D34: the child died without a result and the job said "exited with status 1"):
    # here the model JSON holds a list. The result is written all the same, with the exception and the end of its
    # traceback; the whole traceback is on stderr (the job runner's child.log); exit status 1
    broken = tmp_path / "broken.json"
    broken.write_text("[1, 2, 3]")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({**spec, "model": {**spec["model"], "meta": str(broken)}, "out": str(tmp_path / "out3")}))
    done = subprocess.run([sys.executable, "-m", "crowd3d.structure_pipeline", "run", "--spec", str(path)], cwd=root,
                          capture_output=True, text=True, timeout=300)
    result = json.loads((tmp_path / "out3" / "result.json").read_text())
    assert done.returncode == 1 and sorted(p.name for p in (tmp_path / "out3").iterdir()) == ["result.json"]
    assert result["ok"] is False and result["unexpected"] is True and result["model_id"] == "m1"
    assert result["error_type"] in ("AttributeError", "TypeError") and result["error"].startswith(result["error_type"] + ": ")
    assert 1 <= len(result["traceback"]) <= SP.TRACEBACK_LINES and result["traceback"][-1].startswith(result["error_type"])
    assert "Traceback (most recent call last)" in done.stderr and result["traceback"][-1] in done.stderr


def bad_catalog(folder: Path) -> Path:
    """A catalogue whose landmark "pal" has a snapshot with a part that cannot be built: the octagon, and a block
    beside it whose courtyard runs across its outline."""
    o = Osm()
    ring = [tuple(p) for p in tl.octagon()]
    o.way(ring, {"building:part": "yes", "height": "27", "roof:shape": "pyramidal", "roof:height": "12"}, wid=10)
    x0 = max(x for x, _ in ring) + 2.0
    w1 = o.way([(x0, -5), (x0 + 10, -5), (x0 + 10, 5), (x0, 5)], wid=20)
    w2 = o.way([(x0 + 5, -2), (x0 + 14, -2), (x0 + 14, 2), (x0 + 5, 2)], wid=21)
    o.rel([(w1, "outer"), (w2, "inner")], {"type": "multipolygon", "building:part": "yes", "height": "8"})
    main = [(min(x for x, _ in ring), -20), (x0 + 10, -20), (x0 + 10, 20), (min(x for x, _ in ring), 20)]
    o.way(main, {"building": "yes"}, wid=11)
    osm, outline = folder / "pal.parts.osm", folder / "pal.geojson"
    osm.write_bytes(o.xml())
    lonlat = [list(ENU.to_geodetic(x, y, 0.0)[:2])[::-1] for x, y in main]
    outline.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"osm_type": "way", "osm_id": 11},
        "geometry": {"type": "Polygon", "coordinates": [lonlat + [lonlat[0]]]}}]}))
    catalog = folder / "landmarks.json"
    catalog.write_text(json.dumps([{"slug": "pal", "name": "Palace", "lat": LAT, "lon": LON, "osm": {"file": str(outline)},
                                    "osm_parts": {"file": str(osm), "sha256": SP._sha256(osm)}}]))
    return catalog


def test_a_model_that_fails_its_qa_says_which_gate_and_which_part(tmp_path):
    """D34: a QA failure left "failed its QA: structure" and nothing else. The result now names the gates
    (qa_failed) and what they failed on (qa_detail: the part that was not built and why); the summary with the
    whole QA is written beside it, no GLB."""
    job, meta = tiny_job(tmp_path)
    spec = {"catalog": str(bad_catalog(tmp_path)), "slug": "pal",
            "landmark": {"id": "pal", "name": "Palace", "lat": LAT, "lon": LON},
            "model": {"id": "m1", "meta": str(meta), "sparse": str(job / "sparse" / "0"), "images": {}},
            "out": str(tmp_path / "out"), "max_seconds": 1200, "threads": 2}
    result = SP.run_model(spec)
    assert result == json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["ok"] is False and result["error"] == "the stylized model failed its QA: structure"
    assert result["qa_failed"] == ["structure"] and "unexpected" not in result and result["glb_bytes"] is None
    assert result["qa_detail"].startswith("structure: G13_not_built [r5000: 1 of 1 courtyard(s) cannot be joined")
    assert len(result["qa_detail"]) <= SP.QA_DETAIL_CHARS and result["parts_built"] >= 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["m1.style.json", "result.json"]
    qa = json.loads((tmp_path / "out" / "m1.style.json").read_text())["qa"]
    assert qa["passed"] is False and qa["structure"]["gates"]["G13_not_built"][0]["id"] == "r5000"
    assert qa["mesh"] == {"passed": True, "inexact_triangulations": []} and qa["audit"]["passed"] and qa["glb"]["passed"]
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    assert SP.main(["run", "--spec", str(spec_path)]) == 1                         # the child's exit status


def test_run_model_writes_a_result_whatever_is_raised(tmp_path, monkeypatch, capsys):
    """Whatever is raised, wherever in the step, is recorded with its type and the end of its traceback (the whole of
    it on stderr), and the result file is there in every case; a failure the step expects is not marked unexpected."""
    spec = {"catalog": str(tmp_path / "none.json"), "slug": "oct", "landmark": {"id": "oct", "name": "O", "lat": LAT, "lon": LON},
            "model": {"id": "m1", "meta": str(tmp_path / "m.json"), "sparse": str(tmp_path), "images": {}},
            "out": str(tmp_path / "out")}
    result = SP.run_model(spec)                                      # (the catalogue file is missing: an OSError)
    assert result["ok"] is False and result["error"].startswith("FileNotFoundError: ") and "unexpected" not in result
    assert result["error_type"] == "FileNotFoundError" and result["traceback"][-1] == result["error"]
    assert json.loads((tmp_path / "out" / "result.json").read_text()) == result
    err = capsys.readouterr().err                                    # (the whole traceback: which read it was)
    assert "Traceback (most recent call last)" in err and "in catalog_entry" in err

    def boom(*a, **k):
        raise IndexError("index 7 is out of bounds for axis 0 with size 7")

    monkeypatch.setattr(SP, "find_source", boom)
    result = SP.run_model(spec)
    assert result == json.loads((tmp_path / "out" / "result.json").read_text())
    assert (result["ok"], result["error"], result["error_type"], result["unexpected"]) == \
        (False, "IndexError: index 7 is out of bounds for axis 0 with size 7", "IndexError", True)
    assert result["traceback"][-1] == "IndexError: index 7 is out of bounds for axis 0 with size 7"
    assert any("in boom" in line for line in result["traceback"]) and result["seconds"] >= 0
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err and "in boom" in err
    # a spec without its model block: the same (it was a KeyError before the try)
    result = SP.run_model({"out": str(tmp_path / "out4")})
    assert result["unexpected"] and result["error_type"] == "KeyError" and result["model_id"] is None
    assert (tmp_path / "out4" / "result.json").is_file()


def _planted(code: str):
    """A function that raises what `code` raises (a real numpy or builtin error, not one made by hand)."""
    def planted_fault(*a, **k):
        exec(code)
        raise AssertionError("the planted code did not raise")
    return planted_fault


NUMPY_VALUE_ERROR = "import numpy as np; np.zeros(0).max()"
VALUE_ERRORS = {     # what a geometry bug raises as a ValueError or an OSError: (code, type, start of the message)
    "numpy shape": (NUMPY_VALUE_ERROR, "ValueError", "zero-size array to reduction operation maximum"),
    "numpy linalg": ("import numpy as np; np.linalg.inv(np.zeros((2, 2)))", "LinAlgError", "Singular matrix"),
    "empty max": ("max([])", "ValueError", "max() "),
    "disk full": ("raise OSError(28, 'No space left on device')", "OSError", "[Errno 28] No space left on device"),
}


def test_only_reading_the_inputs_may_fail_as_expected_and_every_failure_keeps_its_traceback(tmp_path, monkeypatch, capsys):
    """D34, review round 2. run_model took every ValueError and OSError of the whole step for a failure it expects:
    one line, no traceback, nothing on stderr. A numpy shape error, numpy's LinAlgError and max() of an empty sequence
    are ValueErrors, so a geometry bug in the judge or the stylization left no trace of where it happened (36 of 42
    planted faults). Now only reading the inputs may fail as expected (and StylizeError, which says what it means),
    and whatever is raised keeps its type, the end of its traceback and the whole traceback on stderr."""
    from crowd3d import scene_io

    osm, outline = write_snapshot(tmp_path)
    catalog = tmp_path / "landmarks.json"
    catalog.write_text(json.dumps([{"slug": "oct", "name": "Octagon", "lat": LAT, "lon": LON, "osm": {"file": str(outline)},
                                    "osm_parts": {"file": str(osm), "sha256": SP._sha256(osm)}}]))
    job, meta = tiny_job(tmp_path)
    spec = {"catalog": str(catalog), "slug": "oct", "landmark": {"id": "oct", "name": "Octagon", "lat": LAT, "lon": LON},
            "model": {"id": "m1", "meta": str(meta), "sparse": str(job / "sparse" / "0"), "images": {}},
            "out": str(tmp_path / "out"), "max_seconds": 1200}

    def run_with(module, attr: str, code: str) -> dict:
        with monkeypatch.context() as mp:
            mp.setattr(module, attr, _planted(code))
            SP._LOADED.clear()                                      # (load_structure memoises the snapshot's parts)
            capsys.readouterr()
            result = SP.run_model(spec)
        err = capsys.readouterr().err
        assert result == json.loads((tmp_path / "out" / "result.json").read_text()) and result["ok"] is False
        # the failing function is named in the result and on stderr, the exception is the traceback's last line
        assert any("in planted_fault" in line for line in result["traceback"]), result
        # (numpy's LinAlgError is printed with its module: the last line ends with the error)
        assert 1 <= len(result["traceback"]) <= SP.TRACEBACK_LINES and result["traceback"][-1].endswith(result["error"])
        assert "Traceback (most recent call last)" in err and "in planted_fault" in err
        assert result["error"].startswith(result["error_type"] + ": ") and "glb_bytes" not in result
        return result

    # the model stage: the snapshot's parts, the registration (before the judge), inside stylize_landmark
    stage = ((SP.S, "load_osm_structure"), (st, "register_scene"), (st, "build_parts"))
    for module, attr in stage:
        for name, (code, kind, message) in VALUE_ERRORS.items():
            result = run_with(module, attr, code)
            assert (result.get("unexpected"), result["error_type"]) == (True, kind), (attr, name, result)
            assert result["error"].startswith(f"{kind}: {message}"), (attr, name, result)
        # a SystemExit there does not end the child without a result, and is not expected either
        result = run_with(module, attr, "raise SystemExit('planted exit')")
        assert (result.get("unexpected"), result["error"]) == (True, "SystemExit: planted exit"), attr
        # StylizeError says what it means wherever it is raised (neither a usable footprint nor points)
        result = run_with(module, attr, "raise st.StylizeError('no usable footprint')")
        assert "unexpected" not in result and result["error"] == "StylizeError: no usable footprint", attr
    # reading the inputs: the same classes are failures the step expects (a file that cannot be read, load_job's
    # SystemExit); they keep their traceback all the same, and anything else is unexpected there too
    for module, attr in ((SP, "find_source"), (scene_io, "load_footprint_rings"), (scene_io, "load_job")):
        for name, (code, kind, message) in VALUE_ERRORS.items():
            result = run_with(module, attr, code)
            assert "unexpected" not in result and result["error"].startswith(f"{kind}: {message}"), (attr, name, result)
        result = run_with(module, attr, "raise SystemExit('fewer than 3 camera positions')")
        assert "unexpected" not in result and result["error"] == "SystemExit: fewer than 3 camera positions", attr
        result = run_with(module, attr, "[][3]")
        assert (result.get("unexpected"), result["error_type"]) == (True, "IndexError"), attr
    # nothing planted: the same spec gives a model (the faults above were the only reason for the failures)
    SP._LOADED.clear()
    assert SP.run_model(spec)["ok"] is True and (tmp_path / "out" / "m1.glb").is_file()


def test_a_numpy_valueerror_inside_stylize_landmark_leaves_its_traceback_in_the_real_child(tmp_path):
    """The same in the product's child process (`crowd3d.structure_pipeline.main run --spec`, as the job runner starts
    it): a numpy ValueError raised inside stylize_landmark. result.json has its type, "unexpected" and the end of the
    traceback with the failing function; the whole traceback is on stderr, which the job runner writes to child.log
    (it was empty); the exit status is 1 and no GLB is written."""
    import subprocess
    import sys

    osm, outline = write_snapshot(tmp_path)
    catalog = tmp_path / "landmarks.json"
    catalog.write_text(json.dumps([{"slug": "oct", "name": "Octagon", "lat": LAT, "lon": LON, "osm": {"file": str(outline)},
                                    "osm_parts": {"file": str(osm), "sha256": SP._sha256(osm)}}]))
    job, meta = tiny_job(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "catalog": str(catalog), "slug": "oct", "landmark": {"id": "oct", "name": "Octagon", "lat": LAT, "lon": LON},
        "model": {"id": "m1", "meta": str(meta), "sparse": str(job / "sparse" / "0"), "images": {}},
        "out": str(tmp_path / "out"), "max_seconds": 1200, "threads": 2}))
    # build_parts is called by stylize_landmark, after the registration and the massing
    probe = ("import sys\n"
             "import crowd3d.stylize as st\n"
             "def planted_fault(*a, **k):\n"
             f"    {NUMPY_VALUE_ERROR}\n"
             "st.build_parts = planted_fault\n"
             "from crowd3d.structure_pipeline import main\n"
             "sys.exit(main(sys.argv[1:]))\n")
    done = subprocess.run([sys.executable, "-c", probe, "run", "--spec", str(spec)],
                          cwd=Path(SP.__file__).resolve().parent.parent, capture_output=True, text=True, timeout=300)
    assert done.returncode == 1 and sorted(p.name for p in (tmp_path / "out").iterdir()) == ["result.json"], done.stderr
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert (result["ok"], result["unexpected"], result["error_type"], result["model_id"]) == (False, True, "ValueError", "m1")
    assert result["error"] == "ValueError: zero-size array to reduction operation maximum which has no identity"
    assert result["traceback"][-1] == result["error"] and len(result["traceback"]) == SP.TRACEBACK_LINES
    assert any("in planted_fault" in line for line in result["traceback"])
    assert any("in stylize_landmark" in line for line in result["traceback"])
    assert "Traceback (most recent call last)" in done.stderr and "in planted_fault" in done.stderr
    assert "in stylize_landmark" in done.stderr and done.stderr.rstrip().endswith(result["error"])


def test_the_failure_detail_names_parts_and_nodes_within_its_bound():
    qa = {"passed": False, "faces_without_provenance": 0, "relief_unbacked": 2,
          "audit": {"passed": True, "violations": []},
          "mesh": {"passed": False, "inexact_triangulations": [{"node": "roof-3", "part": "w7", "why": "2 reversed triangle(s)"}]},
          "structure": {"passed": False, "frame_ok": True, "decisions": {"passed": True},
                        "gates": {"passed": False, "G1_not_watertight": ["w1"], "G10_unsupported": [{"id": "w9"}],
                                  "G13_not_built": [{"id": f"w{k}", "kind": "exception", "why": "TypeError: x"}
                                                    for k in range(8)]}}}
    assert SP._qa_failures(qa) == ["mesh", "structure", "relief_unbacked"]
    detail = SP.qa_failure_detail(qa)
    assert detail == ("mesh: inexact_triangulations [roof-3: 2 reversed triangle(s)] | structure: G1_not_watertight [w1], "
                      "G13_not_built [w0: TypeError: x; w1: TypeError: x; w2: TypeError: x; w3: TypeError: x; "
                      "w4: TypeError: x, +3 more] | relief_unbacked = 2")
    long = {"passed": False, "structure": {"passed": False, "gates": {
        "G13_not_built": [{"id": f"w{k}", "kind": "exception", "why": "x" * 400} for k in range(5)]}}}
    assert len(SP.qa_failure_detail(long)) == SP.QA_DETAIL_CHARS and SP.qa_failure_detail(long).endswith("...")
    assert SP.qa_failure_detail({"passed": True, "audit": {"passed": True}}) == ""
