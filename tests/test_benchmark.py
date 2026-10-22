"""The parameter mechanism (crowd3d.params), pipeline configs and scripts/benchmark.py (no COLMAP)."""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import math
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from crowd3d import params, qa, reconstruct, validation
from crowd3d.config import Settings
from crowd3d.exif import build_exif_bytes
from crowd3d.reconstruct import ReconstructionResult
from crowd3d.records import JOB_FACE, Landmark, PhotoRecord

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "scripts"))


# ---------------------------------------------------------------- params: defaults, overrides, effective dump

# The values the modules hard-coded before the mechanism existed (reconstruct.py, validation.py, qa.py).
SHIPPED = {
    "gps_sigma_m": 6.0, "gps_sigma_min_m": 2.0, "gps_inlier_sigmas": 3.0, "gps_inlier_min_m": 10.0,
    "max_gps_scale_error": 0.35, "compass_sigma_deg": 15.0, "compass_yaw_min_sigma_deg": 3.0,
    "compass_outlier_sigmas": 3.0, "pseudo_compass_sigma_deg": 25.0, "pseudo_compass_min_sigma_deg": 10.0,
    "max_yaw_conflict_deg": 40.0, "up_prior_weight": 0.01, "up_min_eigen_gap": 0.02,
    "up_common_height_fraction": 0.25, "exhaustive_max_images": 60, "spatial_max_neighbors": 30,
    "spatial_max_distance_m": 80.0, "focal_refine_min_images": 12, "mapping_attempts": 3,
    "mapping_attempts_large": 2, "small_set_images": 30, "min_candidate_points": 100,
    "retry_flagged_fraction": 0.05, "retry_compass_error_deg": 20.0, "gross_compass_sigmas": 3.0,
    "far_from_gps_factor": 3.0, "focal_ratio_min": 0.25, "focal_ratio_max": 12.0, "analysis_max_side_px": 800,
    "qa_max_facing_offset_deg": 90.0, "qa_full_sector_min_registered": 2,
}


def test_defaults_are_the_shipped_constants():
    assert dataclasses.asdict(params.DEFAULTS) == SHIPPED
    assert params.current() == params.DEFAULTS
    assert params.DEFAULTS.gross_compass_deg == 45.0
    assert params.DEFAULTS.focal_ratio_range == (0.25, 12.0)
    assert reconstruct.GPS_INLIER_M == 18.0
    # max(3 sigma, 10 m) with the 2 m floor on a reported accuracy, as before.
    assert [reconstruct.gps_inlier_threshold_m(a) for a in (None, float("nan"), 1.0, 5.0, 8.0)] == [
        18.0, 18.0, 10.0, 15.0, 24.0]
    assert type(params.DEFAULTS.mapping_attempts) is int


def test_overrides_take_effect_in_every_module_and_are_restored():
    assert reconstruct.camera_flags(5.0, 18.0, 50.0, 1.0) == []
    with params.override(gps_inlier_min_m=30.0, gross_compass_sigmas=2.0, focal_ratio_max=0.5):
        assert params.current().gps_inlier_min_m == 30.0
        assert reconstruct.gps_inlier_threshold_m(None) == 30.0
        # 50 deg off the compass is now gross (2 x 15), 20 m from GPS beyond an 18 m threshold, focal 1.0 odd.
        assert reconstruct.camera_flags(20.0, 18.0, 50.0, 1.0) == ["gps_and_compass", "odd_focal"]
    assert params.current() == params.DEFAULTS
    assert reconstruct.gps_inlier_threshold_m(None) == 18.0

    # validation: blur and dHash are measured on a copy of analysis_max_side_px.
    buf = io.BytesIO()
    Image.new("RGB", (1200, 900), (90, 120, 150)).save(buf, "JPEG")
    assert max(validation._analysis_gray(buf.getvalue()).size) == 800
    with params.override(analysis_max_side_px=400):
        assert max(validation._analysis_gray(buf.getvalue()).size) == 400

    # compass_yaw drops headings beyond compass_outlier_sigmas x sigma from the mean.
    view = np.array([[0.0, 1.0, 0.0]] * 5)
    headings = [0.0, 2.0, -2.0, 1.0, 40.0]
    assert reconstruct.compass_yaw(view, headings)[1] == 5
    with params.override(compass_outlier_sigmas=1.0):
        assert reconstruct.compass_yaw(view, headings)[1] == 4


def test_qa_reads_its_tunables():
    lm = Landmark("lm", "LM", 31.0, 121.0, 150.0, 0.0)
    photos = [PhotoRecord(id=f"p{i}", landmark_id="lm", user_id="u", status="accepted", uploaded_at=0.0)
              for i in range(4)]
    # Camera 20 m east of the centre looking north-east: 45 deg away from "towards the landmark" (west) + 90.
    cams = {p.id: {"enu": [20.0, float(i), 1.6], "heading_deg": 45.0, "flags": []} for i, p in enumerate(photos)}

    def flagged(**changes: Any) -> int:
        result = ReconstructionResult(ok=True, message="", num_input=4, num_registered=4, num_points=500,
                                      cameras=json.loads(json.dumps(cams)))
        with params.override(**changes):
            qa.assess(result, JOB_FACE, photos, lm, Settings())
        return sum(bool(c["flags"]) for c in result.cameras.values())

    assert flagged() == 4                                   # 135 deg off > 90
    assert flagged(qa_max_facing_offset_deg=150.0) == 0


def test_unknown_names_and_bad_values_are_refused():
    with pytest.raises(ValueError, match="unknown algorithm parameter"):
        params.with_overrides({"gps_sigma": 5.0})
    with pytest.raises(ValueError, match="integer"):
        params.with_overrides({"mapping_attempts": 1.5})
    with pytest.raises(ValueError):
        params.with_overrides({"gps_sigma_m": "5"})
    with pytest.raises(ValueError):
        params.with_overrides({"gps_sigma_m": True})
    with pytest.raises(ValueError):
        params.with_overrides({"gps_sigma_m": float("nan")})
    assert params.with_overrides({"mapping_attempts": 2.0}).mapping_attempts == 2
    assert params.with_overrides({"gps_sigma_m": 5}).gps_sigma_m == 5.0
    for bad in ({"data_dir": "x"}, {"inline_jobs": True}, {"reconstructor": "fake"}, {"no_such": 1}):
        with pytest.raises(ValueError):
            params.settings_with(Settings(), bad)
    s = params.settings_with(Settings(data_dir=Path("d"), inline_jobs=True), {"processing_max_side_px": 1024})
    assert s.processing_max_side_px == 1024 and s.inline_jobs and s.data_dir == Path("d")


def test_the_effective_dump_is_complete():
    eff = params.effective(Settings(reconstructor="colmap"))
    assert eff["reconstructor"] == "colmap"
    assert set(eff["params"]) == set(SHIPPED)
    tunable = {f.name for f in dataclasses.fields(Settings)} - params.SETTINGS_NOT_TUNABLE
    assert set(eff["settings"]) == tunable and len(tunable) >= 20
    assert all(eff["settings"][k] == getattr(Settings(), k) for k in tunable)
    json.dumps(eff)    # serialisable as it is


def test_activate_is_process_wide():
    seen = []
    previous = params.activate(params.with_overrides({"mapping_attempts": 1}))
    try:
        t = threading.Thread(target=lambda: seen.append(params.current().mapping_attempts))
        t.start()
        t.join()
    finally:
        params.activate(previous)
    assert seen == [1] and params.current() == params.DEFAULTS
    with pytest.raises(TypeError):
        params.activate({"mapping_attempts": 1})


def test_reconstructors_plug_in_through_the_registry():
    class Dummy:
        name = "dummy-test"

        def __init__(self, settings=None):
            self.settings = settings

    reconstruct.register_reconstructor("dummy-test", Dummy)
    try:
        r = reconstruct.get_reconstructor("dummy-test", Settings())
        assert isinstance(r, Dummy) and isinstance(r.settings, Settings)
        reconstruct.register_reconstructor("dummy-test", Dummy)          # same factory again: fine
        with pytest.raises(ValueError, match="already registered"):
            reconstruct.register_reconstructor("dummy-test", lambda s: None)
    finally:
        reconstruct.RECONSTRUCTORS.pop("dummy-test")
    assert isinstance(reconstruct.get_reconstructor("fake"), reconstruct.FakeReconstructor)
    with pytest.raises(ValueError, match="unknown reconstructor"):
        reconstruct.get_reconstructor("dummy-test")


# ---------------------------------------------------------------- pipeline configs


def test_committed_configs_load_and_differ_only_where_they_say(bench):
    ev = bench.ev
    default = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "default.json"))
    fast = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "fast.json"))
    assert default["params"] == SHIPPED
    assert default["settings"] == params.effective(Settings())["settings"]
    assert default["reconstructor"] == "colmap" and default["harness"] == {"arrival": "session", "seed": 0}
    diff = {k for k in default["settings"] if default["settings"][k] != fast["settings"][k]}
    assert diff == {"processing_max_side_px"} and fast["settings"]["processing_max_side_px"] < 1600
    assert default["params"] == fast["params"]


@pytest.mark.parametrize("bad", [{"settings": {"inline_jobs": False}}, {"params": {"gps_sigma": 1}},
                                 {"harness": {"arrival": "random"}}, {"harness": {"speed": 1}}, {"extra": 1}])
def test_bad_configs_are_refused(bench, tmp_path, bad):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"config_id": "c", **bad}))
    with pytest.raises(ValueError):
        bench.ev.load_pipeline_config(path)


# ---------------------------------------------------------------- compare, aggregate, noise


def _harness():
    if "eval_real_landmark" not in sys.modules:
        spec = importlib.util.spec_from_file_location("eval_real_landmark", ROOT / "scripts" / "eval_real_landmark.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["eval_real_landmark"]


def metrics(**values: Any) -> dict[str, Any]:
    """A building's metrics: every METRICS name, None unless given."""
    out = {m.name: None for m in _harness().METRICS}
    out.update(values)
    return out


def result(buildings: dict[str, dict[str, Any]], *, sha: str = "t1", schema: int = 1, noise: dict | None = None,
           name: str = "r", inputs: dict[str, str] | None = None, effective: dict | None = None) -> dict[str, Any]:
    eff = effective or {"reconstructor": "colmap", "settings": {"processing_max_side_px": 1600}, "params": {}}
    return {
        "result_schema_version": 1, "metric_schema_version": schema, "run_name": name,
        "testset": {"id": "ts", "sha256": sha}, "noise": noise,
        "config": {"id": "default", "effective": eff, "effective_sha256": json.dumps(eff, sort_keys=True)},
        "provenance": {"started": f"2026-10-22T00:00:{len(name):02d}+00:00", "git": {"commit": "abc", "dirty": False}},
        "buildings": {s: {"status": "ok", "metrics": m, "inputs_sha256": (inputs or {}).get(s, "same")}
                      for s, m in buildings.items()},
        "aggregate": {},
    }


def test_judge_gives_direction_and_significance(bench):
    by = bench.ev.METRIC_BY_NAME
    j = bench.judge(by["models_published"], 2, 3, None, 1.0)
    assert (j["verdict"], j["significant"], j["delta"]) == ("better", True, 1.0)
    assert bench.judge(by["outline_shift_m"], 4.0, 6.0, None, 1.0)["verdict"] == "worse"
    assert bench.judge(by["outline_shift_m"], 4.0, 3.0, None, 1.0)["verdict"] == "better"
    assert bench.judge(by["photos_accepted"], 40, 41, None, 1.0)["verdict"] == "changed"
    assert bench.judge(by["outline_shift_m"], 4.0, 4.0, None, 1.0)["verdict"] == "same"
    gone = bench.judge(by["outline_shift_m"], 4.0, None, None, 1.0)
    assert gone["verdict"] == "disappeared" and gone["significant"]
    assert not bench.judge(by["outline_shift_m"], None, 4.0, None, 1.0, presence_varies=True)["significant"]
    # A delta equal to the recorded spread is within the noise (no float-rounding false alarm).
    assert not bench.judge(by["outline_shift_m"], 7.85, 10.97, 3.12, 1.0)["significant"]
    assert bench.judge(by["outline_shift_m"], None, None, None, 1.0)["significant"] is False
    # Within the recorded noise: not significant; beyond it: significant; the factor scales the noise.
    assert not bench.judge(by["outline_shift_m"], 4.0, 5.0, 1.5, 1.0)["significant"]
    assert bench.judge(by["outline_shift_m"], 4.0, 6.0, 1.5, 1.0)["significant"]
    assert not bench.judge(by["outline_shift_m"], 4.0, 6.0, 1.5, 2.0)["significant"]
    # Below the metric's own minimum change (0.2 m) nothing is significant, noise or not.
    assert not bench.judge(by["outline_shift_m"], 4.0, 4.15, None, 1.0)["significant"]
    # Runtime: a relative floor of 25 %.
    assert not bench.judge(by["wall_time_s"], 100.0, 120.0, None, 1.0)["significant"]
    assert bench.judge(by["wall_time_s"], 100.0, 140.0, None, 1.0)["significant"]


def test_aggregate_uses_each_metrics_rule_and_skips_missing(bench):
    agg = bench.aggregate({
        "a": metrics(models_published=2, registration_rate=0.5, outline_shift_m=1.0, gt_cameras=None),
        "b": metrics(models_published=3, registration_rate=1.0, outline_shift_m=3.0, gt_cameras=None),
        "c": metrics(models_published=1, registration_rate=None, outline_shift_m=10.0, gt_cameras=4),
    })
    assert agg["buildings"] == 3 and agg["models_published"] == 6
    assert agg["registration_rate"] == 0.75 and agg["outline_shift_m"] == 3.0 and agg["gt_cameras"] == 4
    assert bench.aggregate({})["models_published"] is None


def test_noise_from_repeats(bench):
    noise = bench.noise_from_repeats({
        "a": [metrics(outline_shift_m=1.0, models_published=2), metrics(outline_shift_m=1.4, models_published=2)],
        "b": [metrics(outline_shift_m=5.0, models_published=1), metrics(outline_shift_m=4.0, models_published=2)],
    })
    assert noise["repeats"] == 2 and noise["buildings"] == ["a", "b"]
    assert noise["per_building"]["a"]["outline_shift_m"] == pytest.approx(0.4)
    assert noise["per_building"]["b"]["outline_shift_m"] == pytest.approx(1.0)
    assert noise["per_metric"]["outline_shift_m"] == pytest.approx(1.0)
    assert noise["per_metric"]["models_published"] == 1.0 and noise["per_building"]["a"]["models_published"] == 0.0
    # The aggregate (median of 2 buildings = mean) moved from 3.0 to 2.7.
    assert noise["aggregate"]["outline_shift_m"] == pytest.approx(0.3)
    assert noise["per_metric"]["gt_cameras"] is None
    flaky = bench.noise_from_repeats({"a": [metrics(outline_shift_m=1.0), metrics(), metrics(outline_shift_m=2.0)]})
    assert flaky["presence_varies"] == {"a": ["outline_shift_m"]}


def test_compare_flags_only_changes_beyond_the_noise(bench):
    noise = bench.noise_from_repeats({"a": [metrics(outline_shift_m=2.0), metrics(outline_shift_m=3.0)]})
    a = result({"a": metrics(outline_shift_m=2.0, models_published=2, wall_time_s=100.0)}, noise=noise, name="a")
    b = result({"a": metrics(outline_shift_m=2.5, models_published=3, wall_time_s=105.0)}, name="b")
    c = bench.compare_results(a, b, bench.noise_sources([a, b], [], Path("/nonexistent")))
    rows = c["per_building"]["a"]["metrics"]
    assert rows["outline_shift_m"]["noise"] == 1.0 and not rows["outline_shift_m"]["significant"]
    assert rows["models_published"]["verdict"] == "better" and rows["models_published"]["significant"]
    assert not rows["wall_time_s"]["significant"]
    assert c["significant_better"] == 2 and c["significant_worse"] == 0     # building + aggregate rows
    assert c["effective_identical"] and c["noise_sources"] == ["a (default, 2 repeats)"]
    text = bench.format_compare(c)
    assert "models_published" in text and "Effective parameters identical" in text


def test_compare_refuses_other_testsets_and_metric_schemas(bench):
    a = result({"a": metrics(models_published=1)})
    for other in (result({"a": metrics(models_published=1)}, sha="t2"),
                  result({"a": metrics(models_published=1)}, schema=2)):
        with pytest.raises(ValueError, match="force"):
            bench.compare_results(a, other, [])
        forced = bench.compare_results(a, other, [], force=True)
        assert forced["problems"] and "WARNING" in bench.format_compare(forced)


def test_compare_leaves_buildings_with_other_inputs_out_of_the_aggregate(bench):
    eff_b = {"reconstructor": "colmap", "settings": {"processing_max_side_px": 1024}, "params": {}}
    a = result({"x": metrics(models_published=2), "y": metrics(models_published=5)}, name="a")
    b = result({"x": metrics(models_published=2), "y": metrics(models_published=0)}, name="b",
               inputs={"y": "other"}, effective=eff_b)
    c = bench.compare_results(a, b, [])
    assert c["buildings_inputs_differ"] == ["y"]
    assert c["aggregate"]["models_published"]["verdict"] == "same"           # only x counts
    assert c["per_building"]["y"]["metrics"]["models_published"]["verdict"] == "worse"
    assert c["parameter_differences"] == {"settings.processing_max_side_px": [1600, 1024]}
    assert not c["effective_identical"]


def test_leaderboard_lists_runs_and_defines_the_metrics(bench):
    r = result({"x": metrics(models_published=2, wall_time_s=12.0)}, name="default-1")
    r["aggregate"] = bench.aggregate({"x": r["buildings"]["x"]["metrics"]})
    r["noise"] = bench.noise_from_repeats({"x": [metrics(wall_time_s=12.0), metrics(wall_time_s=14.0)]})
    md = bench.build_leaderboard([r])
    assert "| ts | default-1 | default |" in md
    assert "### ts · default · default-1" in md and "| x | ok |" in md
    assert "wall_time_s 2" in md
    assert all(f"| {m.name} |" in md for m in bench.ev.METRICS)


# ---------------------------------------------------------------- integrity and test sets


def _jpeg(exif: bytes | None) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 48), (10, 200, 30))
    img.save(buf, "JPEG", **({"exif": exif} if exif else {}))
    return buf.getvalue()


def test_jpeg_exif_segment(bench):
    blob = build_exif_bytes(lat=43.77, lon=11.25, heading_deg=10.0)
    seg = bench.jpeg_exif_segment(_jpeg(blob))
    assert seg is not None and seg.startswith(b"Exif\0\0") and seg[6:] == blob[6:]
    assert bench.jpeg_exif_segment(_jpeg(None)) is None
    assert bench.jpeg_exif_segment(b"not a jpeg") is None


def test_verify_photos_and_the_inputs_fingerprint(bench, tmp_path):
    files = {"a.jpg": b"aaa", "b.jpg": b"bbb", "c.jpg": b"ccc"}
    b = {"photos": [{"key": k, "file": k, "sha256": bench.sha256_bytes(v)} for k, v in files.items()]}
    for k, v in files.items():
        (tmp_path / k).write_bytes(v)
    ok = bench.verify_photos(b, tmp_path)
    assert (ok["ok"], ok["changed"], ok["missing"]) == (3, 0, 0)
    fp = bench.inputs_fingerprint(ok, False, "o", {})
    (tmp_path / "b.jpg").write_bytes(b"BBB")
    (tmp_path / "c.jpg").unlink()
    bad = bench.verify_photos(b, tmp_path)
    assert (bad["ok"], bad["changed"], bad["missing"]) == (1, 1, 1)
    assert bad["changed_keys"] == ["b.jpg"] and bad["missing_keys"] == ["c.jpg"]
    spec = bench.building_spec({"slug": "s", "centre": {"lat": 1.0, "lon": 2.0}, **b}, tmp_path, bad, None, False)
    assert [p["key"] for p in spec["photos"]] == ["a.jpg"] and spec["radius_m"] == 150.0
    assert [p["key"] for p in bench.building_spec({"slug": "s", "lat": 1.0, "lon": 2.0, **b}, tmp_path, bad,
                                                  None, True)["photos"]] == ["a.jpg", "b.jpg"]
    # Skipping the changed photo leaves a different input set than before, and using it another one again.
    assert len({fp, bench.inputs_fingerprint(bad, False, "o", {}), bench.inputs_fingerprint(bad, True, "o", {})}) == 3
    assert bench.inputs_fingerprint(ok, False, "o", {}) == fp


def test_commons_original_url(bench):
    assert bench.commons_original_url("Example.jpg") == "https://upload.wikimedia.org/wikipedia/commons/a/a9/Example.jpg"


@pytest.mark.parametrize("testset_id", ["synthetic-v1", "mdx-two-v1"])
def test_committed_test_sets_are_complete(bench, testset_id):
    ts, path, sha = bench.load_testset(testset_id)
    assert ts["testset_id"] == testset_id and len(sha) == 64
    assert ts["created"] and ts["description"] and ts["licence"]
    for b in ts["buildings"]:
        lat, lon = bench.building_centre(b)
        assert -90 <= lat <= 90 and -180 <= lon <= 180 and bench.building_radius(b) > 0
        outline, want = bench.building_outline(b, path.parent)
        assert outline.is_file() and bench.sha256_file(outline) == want
        keys = [bench.photo_key(p) for p in b["photos"]]
        assert len(keys) == len(set(keys)) and len(keys) >= 10
        assert all(len(p["sha256"]) == 64 and p["file"] and "exif" in p for p in b["photos"])
        if testset_id == "mdx-two-v1":
            # The product rule: real EXIF GPS on every photo; reference poses and licences recorded.
            assert all(p["exif"]["lat"] is not None and p["exif"]["lon"] is not None for p in b["photos"])
            assert all(p["licence"] and p["source"]["kind"] == "megadepth-x" for p in b["photos"])
            ref = b["gt_reference"]
            assert set(ref["sparse_files"]) >= {"cameras.bin", "images.bin", "points3D.bin"}
            assert set(ref["images"]) == set(keys)


def test_metric_definitions_are_consistent(bench):
    names = [m.name for m in bench.ev.METRICS]
    assert len(names) == len(set(names))
    for m in bench.ev.METRICS:
        assert m.direction in ("higher", "lower", "none") and m.aggregate in ("sum", "mean", "median")
        assert m.description and m.unit and m.min_delta >= 0 and 0 <= m.rel_floor < 1
    assert math.isclose(bench.ev.METRIC_BY_NAME["wall_time_s"].rel_floor, 0.25)


def test_reference_image_maps_keep_exact_names_and_components(bench, tmp_path):
    sub = "commons/WLMUS_2021_images/0/pictures/Lincoln at night.jpg"
    b = {"slug": "lm", "photos": [
        {"key": "a.jpg", "file": "photos/a.jpg"},
        {"key": "b.jpg", "file": "photos/b.jpg", "gt_image": sub, "gt_orientation": 6},
        {"key": "c.jpg", "file": "photos/c.jpg", "gt_additional": {"component": "1", "name": "x/c.jpg",
                                                                   "orientation": 8}},
    ]}
    ref = {"component": "0", "images": {"a.jpg": {"name_in_megadepth_x": "Scene/0/images/sub/a1.jpg",
                                                  "orientation": 3}},
           "additional_components": [{"component": "1", "sparse_files": {}}]}
    maps = bench.gt_image_maps(b, ref)
    assert maps["0"] == {"a.jpg": {"name": "sub/a1.jpg", "orientation": 3}, "b.jpg": {"name": sub, "orientation": 6}}
    assert maps["1"] == {"c.jpg": {"name": "x/c.jpg", "orientation": 8}}
    # Only components whose sparse files are present (and unchanged) are handed to the harness.
    sparse = tmp_path / "lm" / "gt" / "0" / "sparse"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_bytes(b"x")
    ref["sparse_files"] = {"images.bin": {"sha256": bench.sha256_bytes(b"x")}}
    ref["additional_components"][0]["sparse_files"] = {"images.bin": {"sha256": "0" * 64}}
    gt, integ = bench.resolve_gt({**b, "gt_reference": ref}, tmp_path / "lm", tmp_path, None)
    assert [c["component"] for c in gt["components"]] == ["0"]
    assert integ["components"]["1"]["status"] == "missing" and integ["status"] == "missing"


def test_median_metrics_of_repeated_runs(bench):
    runs = [metrics(models_published=1, outline_shift_m=11.8), metrics(models_published=0),
            metrics(models_published=1, outline_shift_m=15.7)]
    med = bench.median_metrics(runs)
    assert med["models_published"] == 1 and type(med["models_published"]) is int
    assert med["outline_shift_m"] == pytest.approx(13.75)
    assert bench.median_metrics(runs[:2] + [metrics(models_published=0)])["outline_shift_m"] is None
