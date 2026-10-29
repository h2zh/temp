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
    "max_yaw_conflict_deg": 40.0, "yaw_systematic_sigma_deg": 3.0, "up_prior_weight": 0.01, "up_min_eigen_gap": 0.02,
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
        assert m.direction in ("higher", "lower", "none") and m.aggregate in ("sum", "mean", "median", "max", "pooled")
        assert m.description and m.unit and m.min_delta >= 0 and 0 <= m.rel_floor < 1
        # a pooled metric names a numerator and a denominator that are metrics themselves
        assert (m.aggregate == "pooled") == (m.pooled is not None)
        assert m.pooled is None or all(x in bench.ev.METRIC_BY_NAME for x in m.pooled)
    assert math.isclose(bench.ev.METRIC_BY_NAME["wall_time_s"].rel_floor, 0.25)
    assert bench.ev.METRIC_SCHEMA_VERSION == 6 and all(n.startswith("style_") for n in bench.ev.STYLE_METRICS)
    assert "style_ref_camera_error_worst_m" not in bench.ev.METRIC_BY_NAME           # renamed in 5
    assert "style_registration_pass_rate" in bench.ev.STYLE_METRICS
    # 6: the photo_pose_* group; its rates pool over photos (uploaded, or those with a reference pose)
    assert all(n.startswith("photo_pose_") for n in bench.ev.POSE_METRICS) and len(bench.ev.POSE_METRICS) == 36
    pooled = {m.name: m.pooled for m in bench.ev.METRICS if m.group == "poses" and m.pooled}
    assert pooled["photo_pose_determined_rate"] == ("photo_pose_determined", "photos_uploaded")
    assert pooled["photo_pose_correct_rate"] == ("photo_pose_correct", "photo_pose_gt")
    # per test set (never-uploaded photos fail) and the bound pair over every test-set photo
    assert pooled["photo_pose_determined_rate_testset"] == ("photo_pose_determined", "photo_pose_testset_photos")
    assert pooled["photo_pose_correct_rate_testset"] == ("photo_pose_correct", "photo_pose_gt_testset")
    assert pooled["photo_pose_pass_rate_low_testset"] == ("photo_pose_correct", "photo_pose_testset_photos")
    assert pooled["photo_pose_pass_rate_high_testset"] == ("photo_pose_pass", "photo_pose_testset_photos")
    # per source
    for src in ("sfm", "exif"):
        assert pooled[f"photo_pose_{src}_uncertainty_coverage"] == (f"photo_pose_{src}_within_uncertainty",
                                                                    f"photo_pose_{src}_gt")
    assert (bench.ev.POSE_RIGHT_M, bench.ev.POSE_RIGHT_DEG, bench.ev.POSE_STRICT_M, bench.ev.POSE_STRICT_DEG) == (
        10.0, 15.0, 5.0, 10.0)


def test_pooled_and_max_aggregates(bench):
    agg = bench.aggregate({
        "a": metrics(style_registration_passed=1, style_models_eligible=4, style_registration_pass_rate=0.25,
                     style_glb_kb_gz=20.0),
        "b": metrics(style_registration_passed=3, style_models_eligible=4, style_registration_pass_rate=0.75,
                     style_glb_kb_gz=55.5),
        "c": metrics(style_registration_passed=0, style_models_eligible=0, style_registration_pass_rate=None),
    })
    # passing / eligible over all models (4/8), not the mean of the per-building rates; a building without
    # eligible models adds nothing
    assert agg["style_registration_pass_rate"] == 0.5 and agg["style_glb_kb_gz"] == 55.5
    none = bench.aggregate({"c": metrics(style_registration_passed=0, style_models_eligible=0)})
    assert none["style_registration_pass_rate"] is None


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


def test_compact_style_keeps_the_gate_and_drops_per_wall_rows(bench):
    rec = {"status": "ok", "eligible": True, "gate": {"passed": False, "reason": "x"},
           "snap": {"accepted": False, "rotation_deg": 0.0, "meta": {"big": 1}},
           "registration": {"accepted": False}, "walls": {"total": 4, "coloured": 0, "coloured_walls": [{"wall": 1}]},
           "observed_pixel_area_fraction": 0.0, "roof_observed_area_fraction": 0.0, "fill": {"area_m2": 0.0},
           "qa": {"passed": True}, "glb_bytes": 1, "glb_gz_bytes": 1, "triangles": 2, "time_s": 0.5, "glb": "/x.glb",
           "snap_candidate": {"rotation_deg": 78.0}, "diagnosis": {"cause": "symmetric_alias"}}
    c = bench.compact_style(rec)
    assert "meta" not in c["snap"] and "coloured_walls" not in c["walls"]
    assert c["diagnosis"]["cause"] == "symmetric_alias" and c["snap_candidate"]["rotation_deg"] == 78.0
    assert bench.compact_style({"status": "skipped (time cap)", "eligible": True}) == {
        "status": "skipped (time cap)", "eligible": True}


def test_resume_reuses_only_a_run_with_the_same_key(bench, tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "summary.json").write_text(json.dumps({"metrics": {"models_published": len(calls)}}))
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)          # whatever the host's free disk
    spec, cfg = {"slug": "b", "photos": []}, tmp_path / "cfg.json"
    key = bench.run_key(spec, code_sha="c1", testset_sha="t", cfg_sha="f", effective_sha="e", inputs_sha="i",
                        stylize=True, time_cap_s=60.0)
    first = bench.run_building_process(spec, cfg, tmp_path / "run", 60.0, key=key, resume=True)
    assert len(calls) == 1 and first["status"] == "ok" and not first.get("resumed")
    again = bench.run_building_process(spec, cfg, tmp_path / "run", 60.0, key=key, resume=True)
    assert len(calls) == 1 and again["resumed"] and again["summary"] == first["summary"]
    assert again["wall_s"] == first["wall_s"]
    other = dict(key, code_sha256="c2")                       # other code: runs again and replaces the key
    bench.run_building_process(spec, cfg, tmp_path / "run", 60.0, key=other, resume=True)
    assert len(calls) == 2
    bench.run_building_process(spec, cfg, tmp_path / "run", 60.0, key=other, resume=False)
    assert len(calls) == 3                                    # without --resume it always runs
    assert bench.run_key(dict(spec, photos=[{"key": "x"}]), code_sha="c1", testset_sha="t", cfg_sha="f",
                         effective_sha="e", inputs_sha="i", stylize=True, time_cap_s=60.0) != key


def test_resume_never_reuses_a_failed_run(bench, tmp_path, monkeypatch):
    calls = []

    def failing(cmd, **kwargs):
        calls.append(cmd)
        return type("Proc", (), {"returncode": 1})()

    monkeypatch.setattr(bench.subprocess, "run", failing)
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)
    spec = {"slug": "b", "photos": []}
    key = bench.run_key(spec, code_sha="c", testset_sha="t", cfg_sha="f", effective_sha="e", inputs_sha="i",
                        stylize=True, time_cap_s=60.0)
    res = bench.run_building_process(spec, tmp_path / "cfg.json", tmp_path / "run", 60.0, key=key, resume=True)
    assert res["status"].startswith("failed") and not (tmp_path / "run" / bench.RUN_KEY_FILE).exists()
    bench.run_building_process(spec, tmp_path / "cfg.json", tmp_path / "run", 60.0, key=key, resume=True)
    assert len(calls) == 2


def test_a_building_run_needs_free_disk(bench, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bench.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda p: type("U", (), {"free": 2 * 2**30})())
    spec = {"slug": "b", "photos": []}
    key = bench.run_key(spec, code_sha="c", testset_sha="t", cfg_sha="f", effective_sha="e", inputs_sha="i",
                        stylize=True, time_cap_s=60.0)
    with pytest.raises(SystemExit, match="2.0 GiB free"):
        bench.run_building_process(spec, tmp_path / "cfg.json", tmp_path / "run", 60.0, key=key, resume=True)
    assert not calls and not (tmp_path / "run" / "spec.json").exists()
    # a finished run that --resume reuses needs no disk
    (tmp_path / "run" / "summary.json").write_text(json.dumps({"metrics": {}}))
    (tmp_path / "run" / bench.RUN_KEY_FILE).write_text(json.dumps({"key": key, "wall_s": 1.0}))
    assert bench.run_building_process(spec, tmp_path / "cfg.json", tmp_path / "run", 60.0, key=key,
                                      resume=True)["resumed"]


def test_the_split_file_and_its_checks(bench, tmp_path):
    ts, _, sha = bench.load_testset("commons-ten-v1")
    split = bench.load_split(ts, sha)
    assert split["id"] == "commons-ten-v1-holdout-v1" and split["path"] == "eval/splits/commons-ten-v1-holdout-v1.json"
    assert split["dev"] == ["florence-baptistery", "aachen-cathedral", "brandenburg-gate", "leaning-tower-pisa",
                            "frauenkirche-dresden"]          # the core five, the only buildings ever run
    assert sorted(split["test"]) == sorted({b["slug"] for b in ts["buildings"]} - set(split["dev"]))
    assert split["sha256"] == bench.sha256_file(bench.REPO_ROOT / split["path"])
    assert bench.load_split(ts, sha, "commons-ten-v1-holdout-v1") == split
    syn, _, syn_sha = bench.load_testset("synthetic-v1")
    assert bench.load_split(syn, syn_sha) is None                       # a test set without a split: all dev
    with pytest.raises(SystemExit, match="not commons-ten-v1"):
        bench.load_split(ts, "0" * 64)                                    # another version of the test set
    doc = json.loads((bench.REPO_ROOT / split["path"]).read_text())
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**doc, "test": doc["test"] + ["florence-baptistery"]}))
    with pytest.raises(SystemExit, match="exactly one"):
        bench.load_split(ts, sha, str(bad))
    with pytest.raises(SystemExit, match="no split"):
        bench.load_split(ts, sha, "nosuch")


def test_the_split_photographers_are_the_test_set_s(bench):
    """The photographers of test photos seen in dev, and the photo counts per stratum, follow from testset.json."""
    ts, _, _ = bench.load_testset("commons-ten-v1")
    doc = json.loads((bench.SPLITS_DIR / "commons-ten-v1-holdout-v1.json").read_text())
    who = {b["slug"]: [bench.mt.normalize_author(p.get("author_url"), p.get("author")) for p in b["photos"]]
           for b in ts["buildings"]}
    dev = {k for s in doc["dev"] for k in who[s] if k}
    test = [k for s in doc["test"] for k in who[s]]
    ph = doc["photographers"]
    assert ph["test_photographers_seen_in_dev"] == sorted({k for k in test if k in dev})
    assert (ph["test_photos"], ph["test_photos_by_seen"], ph["test_photos_without_author"]) == (
        len(test), sum(k in dev for k in test), sum(k is None for k in test))
    assert ph["test_photos_by_unseen"] == ph["test_photos"] - ph["test_photos_by_seen"] - ph["test_photos_without_author"]
    assert doc["contamination"] and doc["rules"]


def test_the_osm_parts_snapshot_is_an_input(bench, tmp_path):
    rows = {"rows": [{"key": "a", "sha256": "x", "status": "ok"}]}
    gt = {"files": {"0": {"images.bin": "ok"}}}
    parts = bench.osm_parts_sha("florence-baptistery")
    catalog = json.loads(bench.CATALOG.read_text())
    assert parts == next(e for e in catalog if e["slug"] == "florence-baptistery")["osm_parts"]["sha256"]
    assert bench.osm_parts_sha("synthetic-two-sides") is None
    # without a snapshot the fingerprint is the one recorded before snapshots counted (old results stay comparable)
    legacy = bench.canonical_sha256({"photos": [("a", "x")], "outline": "o", "gt": gt["files"]})
    assert bench.inputs_fingerprint(rows, False, "o", gt) == bench.inputs_fingerprint(rows, False, "o", gt, None) == legacy
    assert bench.inputs_fingerprint(rows, False, "o", gt, parts) not in (legacy, bench.inputs_fingerprint(
        rows, False, "o", gt, "0" * 64))
    edited = [dict(e, osm_parts={**e["osm_parts"], "sha256": "0" * 64}) if e["slug"] == "india-gate" else e
              for e in catalog]
    (tmp_path / "landmarks.json").write_text(json.dumps(edited))
    with pytest.raises(SystemExit, match="not the snapshot the catalog lists"):
        bench.osm_parts_sha("india-gate", tmp_path / "landmarks.json")


def test_the_fingerprint_keeps_the_reference_of_a_valid_cache(bench, tmp_path):
    """Deleting the reference models once the reference is cached does not change inputs_sha256."""
    live = {"status": "missing", "files": {"0": {"images.bin": "missing"}}}
    when_built = {"status": "ok", "files": {"0": {"images.bin": "ok"}}}
    cache = {"path": str(tmp_path / "s.json"), "key": "k1"}
    assert bench.reference_integrity(live, cache) == live                 # no cache yet
    (tmp_path / "s.json").write_text(json.dumps({"key": "k1", "integrity": when_built}))
    assert bench.reference_integrity(live, cache) == when_built
    assert bench.reference_integrity(live, {**cache, "key": "k2"}) == live     # a stale cache is not used
    (tmp_path / "s.json").write_text("{not json")
    assert bench.reference_integrity(live, cache) == live


def test_code_fingerprint_covers_the_code_a_run_executes(bench):
    sha = bench.code_fingerprint()
    assert sha is not None and len(sha) == 64 and sha == bench.code_fingerprint()


def test_compare_does_not_read_a_cut_short_stylization_as_a_change(bench):
    """Under load a building once lost its whole style record to the time cap (verifier r2: Frauenkirche 'style
    gate 0/0'); its style metrics are marked incomplete, not 'disappeared', and the aggregate leaves it out."""
    a = result({"x": metrics(style_registration_passed=3, style_models_eligible=18, style_walls_coloured=3,
                             style_incomplete=0, models_published=20),
                "y": metrics(style_registration_passed=1, style_models_eligible=4, style_walls_coloured=2,
                             style_incomplete=0)}, name="a")
    b = result({"x": metrics(style_registration_passed=0, style_models_eligible=0, style_walls_coloured=None,
                             style_incomplete=20, models_published=20),
                "y": metrics(style_registration_passed=1, style_models_eligible=4, style_walls_coloured=2,
                             style_incomplete=0)}, name="b")
    c = bench.compare_results(a, b, [])
    rows = c["per_building"]["x"]["metrics"]
    assert rows["style_walls_coloured"]["verdict"] == "incomplete" and not rows["style_walls_coloured"]["significant"]
    assert rows["style_registration_passed"]["verdict"] == "incomplete"
    assert rows["style_incomplete"]["verdict"] == "worse"            # the cut itself is reported
    assert rows["models_published"]["verdict"] == "same"             # the pipeline's metrics are judged as usual
    assert c["aggregate"]["style_registration_passed"]["verdict"] == "same"      # only y counts
    assert c["significant_worse"] == 2                               # style_incomplete, building and aggregate


# ---------------------------------------------------------------- rescore: photo poses from a recorded run


def test_rescore_scores_a_recorded_run(bench, synthetic_dataset, tmp_path):
    import argparse
    import shutil

    ev = bench.ev
    ds = synthetic_dataset
    files = [p["file"] for p in ds.manifest["photos"]]
    tid = "tiny-v1"
    for slug in ("s", "t"):
        (tmp_path / "data" / tid / slug).mkdir(parents=True)
        for f in files:
            shutil.copyfile(ds.path / f, tmp_path / "data" / tid / slug / f)
    building = {"name": "S", "centre": {"lat": ds.lat, "lon": ds.lon}, "radius_m": 150.0,
                "photos": [{"key": f, "file": f, "author": p["user_id"]} for f, p in zip(files, ds.manifest["photos"])]}
    ts = {"schema_version": 1, "testset_id": tid, "integrity_policy": "skip",
          "buildings": [{"slug": "s", **building}, {"slug": "t", **building}]}
    ts_path = tmp_path / "testset.json"
    ts_path.write_text(json.dumps(ts))
    _, _, ts_sha = bench.load_testset(ts_path)
    # a recorded run of building s (fake reconstructor), with the synthetic truth as its reference
    gt = {"kind": "poses", "poses": ev.synthetic_gt_poses(ds.manifest)}
    integ = bench.verify_photos(ts["buildings"][0], tmp_path / "data" / tid / "s")
    spec = bench.building_spec(ts["buildings"][0], tmp_path, integ, gt, False)
    run_root = tmp_path / "work" / tid / "src"
    # cut by the time cap after its first upload request: the photos it never uploaded fail in the *_testset rates
    summary = ev.run_crowd(ev.building_from_spec(spec), run_root / "s", reconstructor="fake", soft_time_cap_s=0.0)
    assert summary["truncated"]
    src = {"result_schema_version": 1, "metric_schema_version": ev.METRIC_SCHEMA_VERSION, "run_name": "src",
           "testset": {"id": tid, "sha256": ts_sha}, "config": {"id": "default"},
           "provenance": {"repeat": 1, "stylize": False, "recon_output_version": bench.RECON_OUTPUT_VERSION},
           "buildings": {"s": {"status": "ok", "metrics": summary["metrics"]}}, "artefacts_dir": str(run_root)}
    src_path = tmp_path / "src.json"
    src_path.write_text(json.dumps(src))
    # the reference cache (the synthetic set has no MegaDepth-X reference to build): the same truth for both
    # buildings (t has no run in the source result)
    cache = tmp_path / "cache"
    for slug in ("s", "t"):
        cb = ev.building_from_spec({**spec, "slug": slug,
                                    "reference_cache": bench.reference_cache_spec(cache, ts_sha, slug, {})})
        assert ev.building_reference(cb, {}, None)[0] is not None
    args = argparse.Namespace(testset=str(ts_path), source=str(src_path), name="poses", buildings=None,
                              work_root=tmp_path / "work", reference_cache=cache, gt_root=None, restylize=False,
                              allow_recon_diff=False, note="test", overwrite=False, data_root=tmp_path / "data",
                              results_dir=tmp_path / "res", split=None)
    assert bench.cmd_rescore(args) == 0
    out = json.loads((tmp_path / "res" / tid / "poses.json").read_text())
    s, t = out["buildings"]["s"], out["buildings"]["t"]
    # the rescored run says exactly what the run itself measured; its other metrics are copied
    assert {k: s["metrics"][k] for k in ev.POSE_METRICS} == {k: summary["metrics"][k] for k in ev.POSE_METRICS}
    assert s["metrics"]["models_published"] == summary["metrics"]["models_published"]
    assert s["poses"]["reference"]["cached"]
    # the test-set denominator: every photo the run planned to upload, also the ones the cap never uploaded
    run = s["poses"]["runs"][0]
    assert run["testset_photos"] == run["run_planned_uploads"] == s["metrics"]["photo_pose_testset_photos"]
    assert s["metrics"]["photo_pose_testset_photos"] > s["metrics"]["photos_uploaded"]
    assert s["metrics"]["photo_pose_not_uploaded"] == run["not_uploaded"] > 0
    assert t == {"status": "skipped (no run in the source result)", "metrics": None}
    assert out["metric_schema_version"] == ev.METRIC_SCHEMA_VERSION
    assert out["provenance"]["rescored_from"]["run_name"] == "src"
    assert out["provenance"]["recon_output_version"] == bench.RECON_OUTPUT_VERSION
    assert out["testset"]["split"] is None and s["integrity"]["osm_parts_sha256"] is None
    assert len(s["inputs_sha256"]) == 64
    assert out["aggregate"]["photo_pose_determined"] == s["metrics"]["photo_pose_determined"]
    assert out["aggregate"]["photo_pose_testset_photos"] == len(files) - sum(  # every file but the no-GPS ones
        p["kind"] == "no_gps" for p in ds.manifest["photos"])
    md = bench.build_leaderboard([out])
    assert "## 照片位姿：每张照片的拍摄位置和朝向" in md and "### tiny-v1 · poses" in md
    assert "| s | ok |" in md and "| **合计** |" in md
    for col in ("photo_pose_testset_photos", "photo_pose_correct_rate_testset", "photo_pose_pass_rate_low_testset",
                "photo_pose_pass_rate_high_testset"):
        assert col in md
    with pytest.raises(SystemExit):                      # an existing result is never overwritten silently
        bench.cmd_rescore(args)
    # artefacts of another reconstruction output version are refused unless asked for
    src["provenance"]["recon_output_version"] = bench.RECON_OUTPUT_VERSION + 1
    src_path.write_text(json.dumps(src))
    args.name = "other"
    with pytest.raises(SystemExit, match="reconstruction output version"):
        bench.cmd_rescore(args)
    args.allow_recon_diff = True
    assert bench.cmd_rescore(args) == 0
    other = json.loads((tmp_path / "res" / tid / "other.json").read_text())
    assert other["provenance"]["recon_output_version"] == bench.RECON_OUTPUT_VERSION + 1    # what it scored


def test_results_recorded_before_the_recon_output_version_map_by_commit(bench):
    for f in sorted((bench.RESULTS_DIR).glob("*/*.json")):
        prov = json.loads(f.read_text())["provenance"]
        v = bench.source_recon_output_version({"provenance": prov})
        # v1-merged and v1-after-fixes ran this reconstruction code (the rescore sources); v0 runs did not
        assert v == (1 if f.stem in ("v1-merged", "v1-after-fixes") else None), f


def test_rescore_refuses_a_result_of_another_test_set(bench, tmp_path):
    import argparse

    src = tmp_path / "src.json"
    src.write_text(json.dumps({"result_schema_version": 1, "testset": {"id": "other", "sha256": "x"}}))
    args = argparse.Namespace(testset="synthetic-v1", source=str(src), name="x", buildings=None,
                              work_root=tmp_path, reference_cache=None, gt_root=None, split=None,
                              note=None, overwrite=False, data_root=tmp_path, results_dir=tmp_path)
    with pytest.raises(SystemExit, match="not synthetic-v1"):
        bench.cmd_rescore(args)
