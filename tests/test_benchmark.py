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
from crowd3d import stylize as st
from crowd3d.config import Settings
from crowd3d.exif import build_exif_bytes
from crowd3d.reconstruct import ReconstructionResult
from crowd3d.records import JOB_FACE, Landmark, PhotoRecord

pytestmark = pytest.mark.harness

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
    "mapping_attempts_large": 2, "small_set_images": 30, "min_candidate_points": 100, "feature_cache": True,
    "map_continue": False, "map_rebuild_ratio": 1.5, "map_max_continuations": 4, "map_fix_existing": False,
    "map_continue_min_kept": 0.95,
    "retry_flagged_fraction": 0.05, "retry_compass_error_deg": 20.0, "gross_compass_sigmas": 3.0,
    "far_from_gps_factor": 3.0, "focal_ratio_min": 0.25, "focal_ratio_max": 12.0, "analysis_max_side_px": 800,
    "qa_max_facing_offset_deg": 90.0, "qa_full_sector_min_registered": 2,
    "anchor_yaw_floor_deg": 7.35, "anchor_scale_floor": 0.352, "anchor_xy_floor_m": 3.1, "anchor_tilt_floor_deg": 1.94,
    "anchor_z_sigma_m": 1.5, "anchor_fallback_log_scale_sigma": 0.7, "anchor_calibration_id": "dev-core5-v1",
    # visual localization (crowd3d.localize, D30): VISUAL_LOCALIZATION 3.4-3.9
    "loc_min_inliers": 30, "loc_min_inlier_ratio": 0.75, "loc_max_reproj_frac": 0.0025, "loc_ransac_frac": 0.012,
    "loc_second_ratio": 0.15, "loc_second_m": 5.0, "loc_second_deg": 10.0, "loc_focal_mismatch": 0.2,
    "loc_min_focal_ratio": 0.3, "loc_max_focal_ratio": 8.0, "loc_across_half": 0.5, "loc_across_deg": 25.0,
    "loc_min_height_m": -15.0, "loc_max_height_m": 120.0, "loc_max_look_deg": 75.0, "loc_max_pairs": 200,
    "loc_withhold_gps_conflict": False,
    # how georegister weighs its evidence (D34): the span switch and "ignored" are on since the offline replay of
    # the recorded runs (the smallest arm the dev rule lets through; the held-out replay passed its frozen
    # criteria). The yaw arbitration is off: the dev rule does not select it and it decided nothing on the
    # held-out split (eval/criteria/d34-v1.json with its addendum, docs/BENCHMARK.md 6.13)
    "gps_span_model_side": True, "compass_other": "ignored", "compass_other_sigma_deg": 45.0,
    "yaw_arbitration": False, "yaw_arb_min_inliers": 8, "yaw_arb_gps_sigmas": 5.0, "yaw_arb_min_readings": 3.0,
    "yaw_arb_max_scatter": 2.0,
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
    assert params.with_overrides({"anchor_calibration_id": "refit"}).anchor_calibration_id == "refit"
    with pytest.raises(ValueError, match="string"):
        params.with_overrides({"anchor_calibration_id": 2})
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


def test_a_replay_config_may_only_change_params_read_after_sfm(bench, tmp_path):
    """rescore --replay makes a run's models again from its kept SfM models: a config whose changes act before or
    inside SfM (or on Settings, or with another reconstructor) is refused, the replay would not show what it does."""
    arms = sorted((ROOT / "eval" / "configs" / "d34-regeo").glob("*.json"))
    assert [a.stem for a in arms] == ["a", "ab-ignored", "abc-ignored", "abc-weak", "ac", "b-ignored", "b-weak",
                                      "bc-ignored", "c", "off"]
    switches = {"gps_span_model_side", "compass_other", "yaw_arbitration"}
    for arm in arms:                                     # every arm of the D34 replay names all three switches
        cfg = bench.load_replay_config(arm)
        assert cfg["id"] == f"d34-regeo-{arm.stem}" and set(cfg["params"]) == switches
        assert cfg["path"] == f"eval/configs/d34-regeo/{arm.name}" and len(cfg["sha256"]) == 64
    off = bench.load_replay_config(ROOT / "eval" / "configs" / "d34-regeo" / "off.json")["params"]
    assert off == {"gps_span_model_side": False, "compass_other": "trusted", "yaw_arbitration": False}
    # the arm the defaults are (the smallest one the dev rule lets through): a replay with it changes nothing a
    # replay under the defaults would not
    shipped = bench.load_replay_config(ROOT / "eval" / "configs" / "d34-regeo" / "ab-ignored.json")["params"]
    assert shipped == {k: getattr(params.DEFAULTS, k) for k in switches}
    assert bench.load_replay_config("default")["params"] == {}

    def config(**parts):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"config_id": "c", **parts}), encoding="utf-8")
        return path

    ok = bench.load_replay_config(config(params={"qa_max_facing_offset_deg": 80.0, "gross_compass_sigmas": 2.0,
                                                 "max_yaw_conflict_deg": 30.0, "up_prior_weight": 0.02}))
    assert set(ok["params"]) == {"qa_max_facing_offset_deg", "gross_compass_sigmas", "max_yaw_conflict_deg",
                                 "up_prior_weight"}
    for bad in ({"params": {"mapping_attempts": 2}}, {"params": {"feature_cache": False}},
                {"params": {"anchor_yaw_floor_deg": 1.0}}, {"params": {"loc_min_inliers": 10}},
                {"settings": {"qa_min_points": 10}}, {"reconstructor": "fake"}):
        with pytest.raises(SystemExit, match="a replay only honours Params read after SfM"):
            bench.load_replay_config(config(**bad))
    assert all(name.startswith(bench.REPLAYABLE_PARAM_PREFIXES) or name.startswith(
        ("anchor_", "loc_", "exhaustive_", "spatial_", "focal_refine_", "mapping_", "small_set_", "min_candidate_",
         "feature_cache", "map_", "retry_", "analysis_")) for name in params.PARAM_NAMES)    # every Params field is placed


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


def test_the_cache_config_pins_the_feature_cache_the_default_now_has(bench):
    ev = bench.ev
    default = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "default.json"))
    cache = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "cache.json"))
    assert ev.load_pipeline_config(ROOT / "eval" / "configs" / "cache.json")["params"] == {"feature_cache": True}
    assert cache["params"] == default["params"] and cache["settings"] == default["settings"]


def test_the_continue_config_turns_on_continuing_and_nothing_else(bench):
    ev = bench.ev
    default = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "default.json"))
    cont = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "continue.json"))
    assert {k for k in default["params"] if default["params"][k] != cont["params"][k]} == (
        set() if params.DEFAULTS.map_continue else {"map_continue"})
    assert cont["params"]["map_continue"] is True and cont["params"]["feature_cache"] is True
    assert cont["settings"] == default["settings"]


def test_the_d30_localize_config_turns_on_localization_and_nothing_else(bench):
    # the config of the D30 fresh runs (eval/results/commons-ten-v1/d30-localize/d30-fresh-loc.json records its sha256)
    ev = bench.ev
    default = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "default.json"))
    loc = ev.config_effective(ev.load_pipeline_config(ROOT / "eval" / "configs" / "d30-localize.json"))
    assert {k for k in default["settings"] if default["settings"][k] != loc["settings"][k]} == {"localize_photos"}
    assert loc["settings"]["localize_photos"] is True and default["settings"]["localize_photos"] is False
    assert loc["params"] == default["params"] and loc["harness"] == default["harness"]
    assert loc["reconstructor"] == default["reconstructor"]


def test_request_caps_per_building_and_in_the_run_key(bench, tmp_path, monkeypatch):
    slugs = ["a", "b", "c"]
    assert bench.request_caps(None, slugs) == {"a": None, "b": None, "c": None}
    assert bench.request_caps(["b=104", "5"], slugs) == {"a": 5, "b": 104, "c": 5}     # a building's own cap wins
    assert bench.request_caps(["5", "b=104"], slugs) == {"a": 5, "b": 104, "c": 5}
    for bad in (["x=3"], ["b=0"], ["b=two"]):
        with pytest.raises(SystemExit, match="max-requests"):
            bench.request_caps(bad, slugs)
    spec = {"slug": "b", "photos": []}

    def key(cap):
        return bench.run_key(spec, code_sha="c", testset_sha="t", cfg_sha="f", effective_sha="e", inputs_sha="i",
                             stylize=False, time_cap_s=60.0, max_requests=cap)

    assert key(104) != key(None) != key(31) and key(104)["max_requests"] == 104
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("Proc", (), {"returncode": 1})()

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)
    bench.run_building_process(spec, tmp_path / "cfg.json", tmp_path / "run", 60.0, key=key(104), max_requests=104)
    bench.run_building_process(spec, tmp_path / "cfg.json", tmp_path / "run", 60.0, key=key(None))
    assert calls[0][calls[0].index("--max-requests") + 1] == "104" and "--max-requests" not in calls[1]


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


def test_compare_reads_a_metric_that_appears_as_appeared_not_worse(bench):
    """A metric the earlier result could not measure (struct_* before the structure pipeline ran in the benchmark, a
    schema-7 score of a rescored run) is "appeared": never counted as worse, whichever its direction."""
    a = result({"x": metrics(models_published=2)}, name="a")
    b = result({"x": metrics(models_published=2, struct_incorrect=3, struct_qa_failed=1, cov_local_wrong_m_1p5=9.0)},
               name="b")
    c = bench.compare_results(a, b, [])
    rows = c["per_building"]["x"]["metrics"]
    assert {rows[k]["verdict"] for k in ("struct_incorrect", "struct_qa_failed", "cov_local_wrong_m_1p5")} == {"appeared"}
    assert c["aggregate"]["struct_incorrect"]["verdict"] == "appeared"
    assert c["significant_worse"] == 0 and c["significant_better"] == 0


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


def test_compare_leaves_out_excluded_groups(bench):
    """--exclude-group style: a baseline stylized by older code is not judged on its style metrics."""
    a = result({"x": metrics(models_published=2, style_registration_passed=3, style_registration_right=4)}, name="a")
    b = result({"x": metrics(models_published=3, style_registration_passed=0, style_registration_right=0)}, name="b")
    assert bench.compare_results(a, b, [])["significant_worse"] == 4           # 2 style metrics, building + aggregate
    c = bench.compare_results(a, b, [], exclude_groups=("style",))
    assert c["significant_worse"] == 0 and c["significant_better"] == 2 and c["excluded_groups"] == ["style"]
    assert not any(n.startswith("style_") for n in c["aggregate"]) and "style_" not in bench.format_compare(c)


def _split(run: dict[str, Any], dev: list[str], test: list[str]) -> dict[str, Any]:
    run["testset"]["split"] = {"id": "ts-holdout-v1", "path": "eval/splits/x.json", "sha256": "s", "dev": dev,
                               "test": test}
    return run


def test_compare_aggregates_each_split_and_refuses_a_building_that_moved(bench):
    noise = bench.noise_from_repeats({"x": [metrics(models_published=2), metrics(models_published=3)],
                                      "y": [metrics(models_published=1), metrics(models_published=1)],
                                      "z": [metrics(models_published=4), metrics(models_published=6)]})
    runs = {"x": metrics(models_published=2), "y": metrics(models_published=1), "z": metrics(models_published=4)}
    a = _split(result(runs, name="a", noise=noise), ["x", "y"], ["z"])
    b = _split(result({**runs, "x": metrics(models_published=5), "z": metrics(models_published=5)}, name="b"),
               ["x", "y"], ["z"])
    c = bench.compare_results(a, b, bench.noise_sources([a, b], [], Path("/nonexistent")))
    assert c["split_buildings"] == {"dev": ["x", "y"], "test": ["z"]}
    dev, test = c["splits"]["dev"]["models_published"], c["splits"]["test"]["models_published"]
    # each split's aggregate over its own buildings; its noise bounded by their spreads (a sum: 1 + 0, and 2)
    assert (dev["a"], dev["b"], dev["noise"], dev["significant"]) == (3, 6, 1.0, True)
    assert (test["a"], test["b"], test["noise"], test["significant"]) == (4, 5, 2.0, False)
    assert c["per_building"]["z"]["split"] == "test" and "[test]" in bench.format_compare(c)
    # one split only: its aggregate is the whole comparison's
    only = bench.compare_results(result(runs, name="a"), result(runs, name="b"), [])
    assert list(only["splits"]) == ["dev"] and only["splits"]["dev"] is only["aggregate"]
    moved = _split(result(runs, name="m"), ["x"], ["y", "z"])
    with pytest.raises(ValueError, match="y is dev in A, test in B"):
        bench.compare_results(a, moved, [])
    assert bench.compare_results(a, moved, [], force=True)["problems"]


def _review_metrics(cov: float, **kw: Any) -> dict[str, Any]:
    """A building's metrics with every value the review-v1 rules read: guards at 0, 12 calibrated models."""
    values = dict(cov_local_supported_m_1p5=cov, cov_geo_supported_m_5=10.0, cov_geo_supported_m_1p5=0.0,
                  style_registration_wrong=0, style_qa_failed=0, style_incomplete=0,
                  struct_incorrect=0, struct_qa_failed=0, stale_overwrites=0, cov_local_wrong_m_1p5=5.0,
                  cov_geo_wrong_m_5=10.0, wrong_face_cameras_shown=1, wrong_face_models_published=0,
                  photo_pose_wrong=3, gt_gross_shown=0, cpu_time_s=1000.0, jobs_cpu_s=800.0, peak_rss_jobs_mb=1200.0,
                  request_latency_p90_s=40.0, photo_time_to_pose_p50_s=60.0, disk_run_mb=300.0, models_published=4)
    for comp in ("yaw", "scale", "xy", "tilt"):
        values.update({f"calib_{comp}_n": 12, f"calib_{comp}_within2": 11, f"calib_{comp}_beyond3": 0})
    values.update(kw)
    return metrics(**values)


def _review(bench, a_runs: dict, b_runs: dict, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    a = _split(result(a_runs, name="b0"), ["x", "y"], ["z", "w"])
    b = _split(result(b_runs, name="f"), ["x", "y"], ["z", "w"])
    c = bench.compare_results(a, b, [], exclude_groups=exclude)
    return bench.apply_criteria(bench.load_criteria("review-v1"), c, b)


def _outcomes(cr: dict[str, Any], metric: str) -> dict[str, str]:
    return {r["split"]: r["outcome"] for r in cr["rules"] if r["metric"] == metric}


def test_review_criteria_pass_and_fail_as_the_program_says(bench):
    base = {s: _review_metrics(30.0) for s in ("x", "y", "z", "w")}
    better = {**{s: _review_metrics(40.0, jobs_cpu_s=300.0) for s in ("x", "y")},
              **{s: _review_metrics(31.0, jobs_cpu_s=900.0) for s in ("z", "w")}}
    cr = _review(bench, base, better)
    assert cr["verdict"] == "PASS" and cr["splits"] == ["dev", "test"], [r for r in cr["rules"] if r["outcome"] not in
                                                                         ("PASS", "REPORT")]
    assert _outcomes(cr, "cov_local_supported_m_1p5") == {"dev": "PASS", "test": "PASS"}
    assert _outcomes(cr, "models_published") == {"dev": "REPORT", "test": "REPORT"}
    # publishing more models, nothing else: never a pass (the primary did not improve)
    more = {s: _review_metrics(30.0, models_published=9, jobs_cpu_s=300.0) for s in base}
    cr = _review(bench, base, more)
    assert cr["verdict"] == "FAIL" and _outcomes(cr, "cov_local_supported_m_1p5")["dev"] == "FAIL"
    # a hard guard above 0 on one held-out building fails, whatever the rest does
    bad = {**better, "w": _review_metrics(31.0, jobs_cpu_s=900.0, style_registration_wrong=1)}
    cr = _review(bench, base, bad)
    assert cr["verdict"] == "FAIL" and _outcomes(cr, "style_registration_wrong") == {"dev": "PASS", "test": "FAIL"}
    assert next(r for r in cr["rules"] if r["metric"] == "style_registration_wrong" and r["split"] == "test")[
        "detail"] == "w 1"
    # a guard beyond noise on a single building fails (per building), a coverage loss on one does not (pooled)
    one = {**better, "x": _review_metrics(25.0, jobs_cpu_s=300.0, wrong_face_cameras_shown=4)}
    cr = _review(bench, base, one)
    assert _outcomes(cr, "cov_local_supported_m_1p5")["dev"] == "PASS"          # 55 vs 60: pooled still better
    assert _outcomes(cr, "wrong_face_cameras_shown")["dev"] == "FAIL"
    # cost: +25 % at most; the persistent map must cut the jobs' CPU time on dev
    slow = {**better, "z": _review_metrics(31.0, jobs_cpu_s=900.0, cpu_time_s=1700.0)}
    assert _outcomes(_review(bench, base, slow), "cpu_time_s") == {"dev": "PASS", "test": "FAIL"}
    same_jobs = {s: dict(m, jobs_cpu_s=800.0) for s, m in better.items()}
    assert _outcomes(_review(bench, base, same_jobs), "jobs_cpu_s") == {"dev": "FAIL", "test": "PASS"}


def test_review_zero_guards_read_every_repeat(bench):
    """A hard guard is 0 in every repeat of every building: the median over 3 repeats (0, 0, 1) would hide one."""
    base = {s: _review_metrics(30.0) for s in ("x", "y", "z", "w")}
    better = {**{s: _review_metrics(40.0, jobs_cpu_s=300.0) for s in ("x", "y")},
              **{s: _review_metrics(31.0, jobs_cpu_s=900.0) for s in ("z", "w")}}
    a = _split(result(base, name="b0"), ["x", "y"], ["z", "w"])
    b = _split(result(better, name="f"), ["x", "y"], ["z", "w"])
    b["buildings"]["x"]["repeats"] = [{"status": "ok", "metrics": dict(better["x"], style_registration_wrong=v)}
                                      for v in (0, 0, 1)]
    b["buildings"]["y"]["repeats"] = [{"status": "ok", "metrics": better["y"]},
                                      {"status": "ok", "metrics": dict(better["y"], style_incomplete=2)},
                                      {"status": "timeout", "metrics": None}]
    cr = bench.apply_criteria(bench.load_criteria("review-v1"), bench.compare_results(a, b, []), b)
    row = next(r for r in cr["rules"] if r["metric"] == "style_registration_wrong" and r["split"] == "dev")
    assert (row["outcome"], row["detail"]) == ("FAIL", "x 1 (largest of 3 repeats)")
    assert _outcomes(cr, "style_qa_failed") == {"dev": "INSUFFICIENT", "test": "PASS"}   # y: one repeat cut short
    assert _outcomes(cr, "stale_overwrites") == {"dev": "PASS", "test": "PASS"}


def test_review_criteria_say_insufficient_without_the_data(bench):
    base = {s: _review_metrics(30.0) for s in ("x", "y", "z", "w")}
    better = {**{s: _review_metrics(40.0, jobs_cpu_s=300.0) for s in ("x", "y")},
              **{s: _review_metrics(31.0, jobs_cpu_s=900.0) for s in ("z", "w")}}
    # struct_* before the structure pipeline runs in the benchmark: insufficient, unless the group is left out
    pre = {s: dict(m, struct_incorrect=None, struct_qa_failed=None) for s, m in better.items()}
    cr = _review(bench, base, pre)
    assert cr["verdict"] == "INSUFFICIENT" and _outcomes(cr, "struct_incorrect") == {"dev": "INSUFFICIENT",
                                                                                     "test": "INSUFFICIENT"}
    cr = _review(bench, base, pre, exclude=("struct",))
    assert cr["verdict"] == "PASS" and _outcomes(cr, "struct_qa_failed") == {"dev": "EXCLUDED", "test": "EXCLUDED"}
    # calibration: fewer than 20 models say nothing; enough overconfident ones fail
    few = {s: dict(m, calib_xy_n=None, calib_xy_within2=None, calib_xy_beyond3=None) for s, m in better.items()}
    few["x"].update(calib_xy_n=12, calib_xy_within2=12, calib_xy_beyond3=0)
    cr = _review(bench, base, few)
    assert _outcomes(cr, "calib_xy_cover2") == {"dev": "INSUFFICIENT", "test": "INSUFFICIENT"}
    assert cr["verdict"] == "INSUFFICIENT"
    over = {s: dict(m, calib_scale_within2=7, calib_scale_beyond3=3) for s, m in better.items()}
    cr = _review(bench, base, over)
    assert _outcomes(cr, "calib_scale_cover2") == {"dev": "FAIL", "test": "FAIL"}     # 14/24, Wilson up to 0.76
    assert _outcomes(cr, "calib_scale_beyond3") == {"dev": "FAIL", "test": "FAIL"}    # 6/24 > 5 %
    # 20/24 = 0.83: its Wilson interval reaches 0.85, but the share does not (overconfident); all inside passes
    near = {s: dict(m, calib_yaw_within2=10, calib_tilt_within2=12) for s, m in better.items()}
    cr = _review(bench, base, near)
    assert _outcomes(cr, "calib_yaw_cover2") == {"dev": "FAIL", "test": "FAIL"}
    assert _outcomes(cr, "calib_tilt_cover2") == {"dev": "PASS", "test": "PASS"}
    # a stylization cut short has not checked every model: a style guard at 0 is not proven
    cut = {**better, "y": _review_metrics(40.0, jobs_cpu_s=300.0, style_incomplete=3)}
    assert _outcomes(_review(bench, base, cut), "style_registration_wrong")["dev"] == "INSUFFICIENT"
    # cost without a baseline value (a rescored run older than schema 7)
    old = {s: dict(m, cpu_time_s=None) for s, m in base.items()}
    assert _outcomes(_review(bench, old, better), "cpu_time_s") == {"dev": "INSUFFICIENT", "test": "INSUFFICIENT"}


def test_the_review_criteria_file_and_its_checks(bench, tmp_path):
    crit = bench.load_criteria("review-v1")
    assert crit["path"] == "eval/criteria/review-v1.json" and crit["sha256"] == bench.sha256_file(
        bench.REPO_ROOT / crit["path"])
    zero = {r["metric"] for r in crit["rules"] if r.get("require") == "zero"}
    # D29: style_coloured_walls_wrong left with colouring (edited in place, recorded in the file's decisions)
    assert zero == {"style_registration_wrong", "style_qa_failed", "struct_incorrect", "struct_qa_failed",
                    "stale_overwrites"}
    assert any(d.startswith("D29") for d in crit["decisions"])
    primary = [r for r in crit["rules"] if r["role"] == "primary"]
    assert [r["metric"] for r in primary] == ["cov_local_supported_m_1p5"]     # owner question 1
    assert all(r["role"] in bench.REPORT_ROLES for r in crit["rules"]
               if r["metric"] in ("models_published", "facades_published", "style_registration_pass_rate"))
    assert bench.wilson_interval(17, 20) == pytest.approx((0.6396, 0.9476), abs=1e-4)
    doc = json.loads((bench.REPO_ROOT / crit["path"]).read_text())
    for rule, msg in (({"metric": "nosuch", "role": "guard", "require": "zero"}, "unknown metric"),
                      ({"metric": "photo_pose_wrong", "role": "guard", "require": {"interval": [0, 1]}}, "pooled"),
                      ({"metric": "photo_pose_wrong", "role": "guard"}, "needs one"),
                      ({"metric": "photo_pose_wrong", "role": "guard", "require": "less"}, "unknown requirement")):
        (tmp_path / "c.json").write_text(json.dumps({**doc, "rules": [rule]}))
        with pytest.raises(SystemExit, match=msg):
            bench.load_criteria(str(tmp_path / "c.json"))


def _d31(bench, a_runs: dict, b_runs: dict) -> dict[str, Any]:
    a = _split(result(a_runs, name="f"), ["x", "y"], ["z", "w"])
    b = _split(result(b_runs, name="d31"), ["x", "y"], ["z", "w"])
    return bench.apply_criteria(bench.load_criteria("d31-v1"), bench.compare_results(a, b, []), b)


def _rate_row(cr: dict[str, Any], split: str) -> dict[str, Any]:
    return next(r for r in cr["rules"] if r["metric"] == "photo_pose_wrong" and r["role"] == "guard"
                and r["split"] == split)


def test_rate_worse_is_newcombes_interval_of_the_difference(bench):
    # India Gate as predicted for D31 (reference version 2): 0 of 3 -> 4 of 19 is sampling noise
    worse, end = bench.rate_worse(0, 3, 4, 19, higher_is_better=False)
    assert not worse and end == pytest.approx(-0.365, abs=1e-3)
    worse, end = bench.rate_worse(14, 70, 15, 71, higher_is_better=False)       # Brandenburg r1
    assert not worse and end == pytest.approx(-0.123, abs=1e-3)
    assert bench.rate_worse(14, 70, 40, 70, higher_is_better=False)[0]           # 0.20 -> 0.57 on 70 photos
    assert not bench.rate_worse(14, 70, 40, 70, higher_is_better=True)[0]        # the same move is better there
    assert bench.rate_worse(40, 70, 14, 70, higher_is_better=True)[0]
    assert not bench.rate_worse(5, 20, 5, 20, higher_is_better=False)[0]


def test_d31_criteria_judge_the_wrong_rate_not_the_raw_count(bench):
    """D31 shows more poses by design: twice the determined poses at the same wrong rate passes d31-v1, while
    review-v1's raw-count guard fails; a building whose rate rises beyond sampling noise fails and is named."""
    base = {s: _review_metrics(30.0, photo_pose_wrong=10, photo_pose_gt_determined=50) for s in ("x", "y", "z", "w")}
    more = {s: dict(m, photo_pose_wrong=20, photo_pose_gt_determined=100) for s, m in base.items()}
    cr = _d31(bench, base, more)
    assert {r["outcome"] for r in cr["rules"] if r["metric"] == "photo_pose_wrong"} == {"PASS", "REPORT"}
    row = _rate_row(cr, "dev")
    assert (row["a"], row["b"]) == (0.2, 0.2) and "nearest building" in row["detail"]
    assert _outcomes(_review(bench, base, more), "photo_pose_wrong") == {"dev": "FAIL", "test": "FAIL"}
    bad = {**more, "z": dict(more["z"], photo_pose_wrong=70)}                    # 0.2 -> 0.7 on 100
    cr = _d31(bench, base, bad)
    assert (_rate_row(cr, "dev")["outcome"], _rate_row(cr, "test")["outcome"]) == ("PASS", "FAIL")
    assert _rate_row(cr, "test")["detail"].startswith("worse beyond sampling noise: pooled; z 10/50 = 0.200 -> 70/100")
    assert cr["verdict"] == "FAIL"
    # no determined pose in A (no reference, nothing posed): that building has no rate, the rest is judged
    none = {**base, "y": dict(base["y"], photo_pose_wrong=0, photo_pose_gt_determined=0)}
    row = _rate_row(_d31(bench, none, more), "dev")
    assert row["outcome"] == "PASS" and row["detail"].endswith("no rate: y")
    empty = {s: dict(m, photo_pose_wrong=None, photo_pose_gt_determined=None) for s, m in base.items()}
    assert _rate_row(_d31(bench, empty, more), "dev")["outcome"] == "INSUFFICIENT"


def test_d31_criteria_file_against_the_f_results(bench, tmp_path):
    """The rules D31's verification plan names, and the acceptance the checks expect agree with F's recorded runs:
    the same uploads, every reason but not_subject F's own."""
    crit = bench.load_criteria("d31-v1")
    assert crit["path"] == "eval/criteria/d31-v1.json"
    rules = {(r["metric"], r["role"]): r for r in crit["rules"]}
    assert {r["metric"] for r in crit["rules"] if r.get("require") == "zero"} == {
        "style_registration_wrong", "style_qa_failed", "struct_incorrect", "struct_qa_failed", "stale_overwrites",
        "wrong_face_models_published"}
    assert rules[("cov_local_supported_m_1p5", "primary")]["require"] == "not_worse"      # an owner rule
    assert rules[("photo_pose_wrong", "guard")]["require"] == {"rate_of": "photo_pose_gt_determined"}
    assert ("photo_pose_wrong", "info") in rules
    assert all(rules[(m, "guard")]["require"] == "not_worse_per_building"
               for m in ("wrong_face_cameras_shown", "gt_gross_shown", "cov_local_wrong_m_1p5", "cov_geo_wrong_m_5"))
    assert all(r["require"] == {"max_ratio": 1.25} for r in crit["rules"] if r["role"] == "cost")
    checks = {c["id"]: c for c in crit["checks"]}
    assert set(checks) == {"reference_version_2", "acceptance_equals_replay", "pose_correct_pooled_rise",
                           "count_path_models", "wrong_face_reference_v1", "review_v1_raw_count"}
    assert checks["count_path_models"]["require"]["min_share_right"] == 0.745
    expected = checks["acceptance_equals_replay"]["expected"]
    for tid, files in (("commons-ten-v1", ("f-dev", "f-test")), ("synthetic-v1", ("f",)), ("mdx-two-v1", ("f",))):
        seen = set()
        for name in files:
            r = json.loads((bench.REPO_ROOT / "eval" / "results" / tid / f"{name}.json").read_text())
            for slug, rec in r["buildings"].items():
                if not rec.get("details"):
                    continue
                want, funnel = expected[tid][slug], rec["details"]["funnel"]
                ours = {k: v for k, v in want["rejected_by_reason"].items() if k != "not_subject"}
                assert (want["uploaded"], ours) == (funnel["uploaded"], funnel["rejected_by_reason"]), slug
                assert want["accepted"] <= funnel["accepted"], slug
                seen.add(slug)
        assert seen == set(expected[tid])
    assert expected["synthetic-v1"]["synthetic-ring"]["accepted"] == 32     # no outline: F's acceptance
    for rule, msg in (({"metric": "photo_pose_wrong", "role": "guard", "require": {"rate_of": "nosuch"}}, "rate_of"),
                      ({"metric": "photo_pose_correct_rate", "role": "guard",
                        "require": {"rate_of": "photo_pose_gt"}}, "rate_of")):
        (tmp_path / "c.json").write_text(json.dumps({**json.loads((bench.REPO_ROOT / crit["path"]).read_text()),
                                                     "rules": [rule]}))
        with pytest.raises(SystemExit, match=msg):
            bench.load_criteria(str(tmp_path / "c.json"))
    (tmp_path / "c.json").write_text(json.dumps({"criteria_id": "x", "rules": [], "checks": [{"id": "x"}]}))
    with pytest.raises(SystemExit, match="needs an id and a requirement"):
        bench.load_criteria(str(tmp_path / "c.json"))


def _d30_metrics(**kw: Any) -> dict[str, Any]:
    """A building's metrics with every value the d30-v1 rules read (A: no localizer)."""
    values = dict(photo_pose_loc_wrong=0, photo_pose_loc_wrong_face=0, photo_pose_wrong=10, photo_pose_gt=100,
                  photo_pose_gt_determined=50, photo_pose_wrong_share=0.2, photo_pose_loc_cpu_s=0.0,
                  cpu_time_s=1000.0, photo_pose_correct=40, photo_pose_gt_testset=100,
                  photo_pose_correct_rate_testset=0.4, photo_pose_loc=0, photo_pose_loc_gt=0, photo_pose_loc_correct=0,
                  photo_pose_loc_within_uncertainty=0, photo_pose_loc_uncertainty_coverage=None,
                  photo_pose_loc_queries=0)
    values.update(kw)
    return metrics(**values)


def _d30(bench, a_runs: dict, b_runs: dict, b_repeats: dict | None = None) -> dict[str, Any]:
    """x and y are dev, every other building test."""
    test = [s for s in a_runs if s not in ("x", "y")]
    a = _split(result(a_runs, name="d30-a"), ["x", "y"], test)
    b = _split(result(b_runs, name="d30-b"), ["x", "y"], test)
    for slug, reps in (b_repeats or {}).items():
        b["buildings"][slug]["repeats"] = [{"metrics": m} for m in reps]
    return bench.apply_criteria(bench.load_criteria("d30-v1"), bench.compare_results(a, b, [], factor=0.0), b)


def _d30_no_reference(bench, **kw: Any) -> dict[str, Any]:
    """A building without a reference pose for any photo (Sydney Opera House): its photo_pose_* metrics as
    pose_metrics gives them, every reference metric None."""
    return metrics(**{**bench.ev.pose_metrics([], False, kw), "cpu_time_s": 1000.0})


def test_d30_criteria_file_is_the_design_gate(bench):
    """d30-v1 holds the D30 verification plan's gate: zero wrong and zero wrong-face localized poses, the wrong share
    not worse than A, localization CPU within 25 % of the run's; coverage and the correct-rate gain only reported."""
    crit = bench.load_criteria("d30-v1")
    assert crit["path"] == "eval/criteria/d30-v1.json"
    rules = {(r["metric"], r["role"]): r for r in crit["rules"]}
    assert {r["metric"] for r in crit["rules"] if r.get("require") == {"zero": True, "skip_without": "photo_pose_gt"}
            } == {"photo_pose_loc_wrong", "photo_pose_loc_wrong_face"}
    assert rules[("photo_pose_wrong_share", "guard")]["require"] == "not_worse_per_building"
    assert rules[("photo_pose_loc_cpu_s", "cost")]["require"] == {"max_share": 0.25, "of": "cpu_time_s"}
    assert ("photo_pose_loc_uncertainty_coverage", "info") in rules
    assert ("photo_pose_correct_rate_testset", "info") in rules
    assert not [r for r in crit["rules"] if r["role"] in ("primary", "calibration")]
    checks = {c["id"]: c for c in crit["checks"]}
    assert set(checks) == {"paired_rescores", "only_loc_changes", "loc_gps_conflict", "loc_coverage_pooled",
                           "synthetic_truth", "fresh_run_display_equals_score", "privacy_tests", "loc_cost_live",
                           "loc_report"}
    assert checks["loc_gps_conflict"]["require"] == {"wrong": 0} and checks["loc_gps_conflict"]["role"] == "guard"
    assert "--localize" in crit["baseline"]["B"] and "d31" in crit["baseline"]["A"]["commons-ten-v1 test"]


def test_d34_criteria_file_freezes_the_nominated_replay_arm(bench):
    """d34-v1 (D34 step 6, docs/BENCHMARK.md 6.13): the dev replay's outcome under the rule written down before it,
    the one arm that rule let through frozen with its config's sha256, and what is replayed once on the held-out
    split and judged by the same rule. Changing a frozen config or the rule after the fact fails here."""
    crit = bench.load_criteria("d34-v1")
    assert crit["path"] == "eval/criteria/d34-v1.json"
    arms = crit["dev_outcome"]["arms"]
    assert list(arms) == ["off", "a", "b-weak", "b-ignored", "c", "abc-weak", "abc-ignored"]
    assert all(v["qualifies"] == (v["helps"] and v["no_harm"]) for v in arms.values())
    assert crit["dev_outcome"]["qualifying"] == [a for a, v in arms.items() if v["qualifies"]] == ["abc-ignored"]
    assert not arms["off"]["helps"] and arms["off"]["no_harm"] and arms["off"]["yaw_rms_change_pct"] == 0.0
    held = crit["held_out"]
    assert held["nominated"]["arm"] == crit["dev_outcome"]["nominated"] == "abc-ignored"
    assert held["nominated"]["params"] == {"gps_span_model_side": True, "compass_other": "ignored",
                                           "yaw_arbitration": True}
    assert held["control"]["params"] == {"gps_span_model_side": False, "compass_other": "trusted",
                                         "yaw_arbitration": False}
    for frozen in (held["nominated"], held["control"]):
        cfg = bench.load_replay_config(ROOT / frozen["config"])
        assert (cfg["sha256"], cfg["params"]) == (frozen["sha256"], frozen["params"]), frozen["arm"]
    # the control is what a run recorded before the switches is replayed with
    assert params.AS_BEFORE == held["control"]["params"]
    assert [c.split("--name ")[1].split()[0] for c in held["run_once"]] == ["d34-plain-test", "d34-off-test",
                                                                           "d34-abc-ignored-test"]
    assert all("d31-test.json" in c and "--restylize" in c for c in held["run_once"])
    assert len(crit["dev_rule"]["helps_any_of"]) == 4 and len(crit["dev_rule"]["no_harm_all_of"]) == 7
    rules = {(r["metric"], r["role"]): r for r in crit["rules"]}
    assert rules[("wrong_face_models_published", "guard")]["require"] == "zero"
    assert rules[("photo_pose_wrong", "guard")]["require"] == {"rate_of": "photo_pose_gt_determined"}
    assert not [r for r in crit["rules"] if r["role"] in ("primary", "calibration", "cost")]
    checks = {c["id"]: c for c in crit["checks"]}
    assert list(checks) == ["control_reproduces_plain", "helps", "no_harm", "stylization_not_worse",
                            "no_other_arm_on_held_out"]
    assert "dev_rule.helps_any_of" in checks["helps"]["require"]
    assert "dev_rule.no_harm_all_of" in checks["no_harm"]["require"]


def test_d34_criteria_addendum_names_the_arm_the_dev_rule_chooses_and_the_defaults_are_that_arm(bench):
    """The addendum to d34-v1 (after the step's review; the frozen fields above it are untouched, see the test
    above): the two-switch arms run on dev afterwards, the rule applied to them, and what the product ships. By
    dev_rule.choice the qualifying arm with the fewest switches is (a) + (b) ignored; the nominated three-switch
    arm is not better again by a helps measure; the held-out replay of the nominated arm overruled no prior, so it
    is the held-out replay of the two-switch arm as well. The defaults are that arm: the yaw arbitration is off."""
    crit = bench.load_criteria("d34-v1")
    assert list(crit)[-3:] == ["addendum", "path", "sha256"]          # the addendum comes after every frozen field
    add = crit["addendum"]
    arms = add["dev_arms"]
    assert list(arms) == ["ab-ignored", "bc-ignored", "ac"]
    assert all(v["qualifies"] == (v["helps"] and v["no_harm"]) for v in arms.values())
    assert [a for a, v in arms.items() if v["qualifies"]] == ["ab-ignored"]
    assert arms["bc-ignored"]["harm"] == ["wrong_face_cameras_not_higher"] and not arms["ac"]["helps"]
    expected = {"ab-ignored": (True, "ignored", False), "bc-ignored": (False, "ignored", True),
                "ac": (True, "trusted", True)}
    for arm, v in arms.items():                    # each arm's committed config and result are the ones it names
        cfg = bench.load_replay_config(ROOT / v["config"])
        assert cfg["sha256"] == v["sha256"]
        assert tuple(cfg["params"][k] for k in ("gps_span_model_side", "compass_other", "yaw_arbitration")
                     ) == expected[arm]
        result = json.loads((ROOT / v["result"]).read_text(encoding="utf-8"))
        assert result["run_name"] == f"d34-{arm}-dev" and result["provenance"]["git"]["dirty"] is False
        assert result["provenance"]["replay"]["sha256"] == v["sha256"]          # the bytes the replay ran with
        assert result["provenance"]["replay"]["params"] == cfg["params"]
    again = add["dev_rule_applied_again"]
    assert again["qualifying"] == ["abc-ignored", "ab-ignored"]
    assert again["fewest_switches"] == again["chosen"] == "ab-ignored"
    larger = again["abc-ignored_against_ab-ignored"]
    assert larger["helps"] is False and larger["helps_by"] == []
    assert larger["photo_pose_correct"][0] == larger["photo_pose_correct"][1]
    assert larger["photo_pose_wrong"][0] == larger["photo_pose_wrong"][1]
    arb = add["yaw_arbitration_in_the_nominated_arm"]
    assert arb["held_out"]["replayed"] == arb["held_out"]["models"] == 144
    assert arb["held_out"]["answered_overrule"] == arb["held_out"]["models_overruled"] == 0
    assert arb["dev"]["answered_overrule"] == arb["dev"]["models_overruled"] == 6
    # what the product ships: the chosen arm; the nominated arm differs from it by the yaw arbitration only
    shipped = {k: getattr(params.DEFAULTS, k) for k in add["defaults"]}
    assert shipped == add["defaults"] == bench.load_replay_config(ROOT / arms["ab-ignored"]["config"])["params"]
    nominated = crit["held_out"]["nominated"]["params"]
    assert {k for k in nominated if nominated[k] != shipped[k]} == {"yaw_arbitration"}
    assert shipped["yaw_arbitration"] is False


def test_d34b_criteria_file_freezes_the_fresh_evaluation(bench):
    """d34b-v1 (docs/BENCHMARK.md 6.14), written before any fresh D34 run: the two configs a fresh run may use with
    their sha256 (the defaults with and without Settings.facing_away_warns_other_refs), the guards of the fresh pair
    against the recorded D31 runs, the checks judged outside compare, the funnels predicted from the D31 runs' photo
    rows, and how the two verdicts set the defaults. Changing a frozen config or a rule after the fact fails here."""
    crit = bench.load_criteria("d34b-v1")
    assert crit["path"] == "eval/criteria/d34b-v1.json"
    cfgs = crit["configs"]
    for key, settings in (("d_on", {"facing_away_warns_other_refs": True}), ("d_off", {})):
        cfg = bench.ev.load_pipeline_config(ROOT / cfgs[key]["config"])
        assert bench.sha256_file(ROOT / cfgs[key]["config"]) == cfgs[key]["sha256"], key
        assert cfg["settings"] == cfgs[key]["settings"] == settings and cfg["params"] == cfgs[key]["params"] == {}
    on, off = (bench.ev.config_effective(bench.ev.load_pipeline_config(ROOT / cfgs[k]["config"]))
               for k in ("d_on", "d_off"))
    # the config under test is the defaults with the one setting on: nothing else differs
    assert on["params"] == off["params"] and on["harness"] == off["harness"]
    assert {k for k in on["settings"] if on["settings"][k] != off["settings"][k]} == {"facing_away_warns_other_refs"}
    old = bench.load_replay_config(ROOT / cfgs["old_georegistration"]["config"])
    assert (old["sha256"], old["params"]) == (cfgs["old_georegistration"]["sha256"],
                                              cfgs["old_georegistration"]["params"])
    assert params.AS_BEFORE == old["params"]
    # A: the D31 fresh runs at the metric schema of today, each with its own two-repeat noise block
    for split, slugs in (("dev", 5), ("test", 5)):
        a = bench.load_result(ROOT / crit["baseline"]["A"][f"commons-ten-v1 {split}"])
        assert a["metric_schema_version"] == bench.ev.METRIC_SCHEMA_VERSION and a["noise"]["repeats"] == 2
        assert a["provenance"]["rescored_from"]["run_name"] == f"d31-{split}" and len(a["noise"]["buildings"]) == slugs
        assert "--config d34-heading-warn" in crit["baseline"]["B"]["commons-ten-v1 dev"]
        assert "--repeat 2" in crit["baseline"]["B"][f"commons-ten-v1 {split}"]
    assert "--max-requests brandenburg-gate=104 frauenkirche-dresden=31" in crit["baseline"]["B"]["commons-ten-v1 dev"]
    guards = {r["metric"]: r["require"] for r in crit["rules"] if r["role"] == "guard"}
    skip = {"zero": True, "skip_without": "struct_models"}
    assert guards == {
        "wrong_face_models_published": "zero", "style_registration_wrong": "zero", "style_qa_failed": "zero",
        "struct_incorrect": skip, "struct_qa_failed": skip, "stale_overwrites": "zero",
        "models_published": "not_worse", "facades_published": "not_worse",
        "wrong_face_cameras_shown": "not_worse_per_building", "gt_gross_shown": "not_worse_per_building",
        "photo_pose_determined": "not_worse", "photo_pose_wrong": {"rate_of": "photo_pose_gt_determined"},
        "geo_scale_error_median_pct": "not_worse", "geo_yaw_error_median_deg": "not_worse"}
    assert not [r for r in crit["rules"] if r["role"] in ("primary", "calibration", "cost")]
    checks = {c["id"]: c["role"] for c in crit["checks"]}
    assert checks == {"runs_comparable": "precondition", "accepted_photos_rise": "guard",
                      "shown_pose_coverage": "guard", "georeg_errors": "guard", "newly_accepted_photos": "guard",
                      "fresh_pair": "guard", "side_by_side": "report"}
    # the two count checks read metrics compare gives no direction: a rule could not judge them
    assert bench.ev.METRIC_BY_NAME["photos_accepted"].direction == "none"
    assert bench.ev.METRIC_BY_NAME["photo_pose_uncertainty_coverage"].direction == "none"
    pred = crit["predicted_acceptance"]
    assert pred["dev"]["split_total"] == {"uploaded": 508, "accepted_before": 460, "newly_accepted": 12,
                                          "accepted_predicted": 472}
    assert pred["test"]["split_total"] == {"uploaded": 513, "accepted_before": 399, "newly_accepted": 30,
                                           "accepted_predicted": 429}
    assert {s: v["newly_accepted"] for s, v in pred["dev"].items() if s != "split_total"} == {
        "florence-baptistery": 3, "aachen-cathedral": 0, "brandenburg-gate": 3, "leaning-tower-pisa": 1,
        "frauenkirche-dresden": 5}
    assert {s: v["newly_accepted"] for s, v in pred["test"].items() if s != "split_total"} == {
        "sagrada-familia": 15, "hungarian-parliament": 4, "lincoln-memorial": 2, "india-gate": 6,
        "sydney-opera-house": 3}
    for split in ("dev", "test"):                # what was accepted before is what A recorded
        a = bench.load_result(ROOT / crit["baseline"]["A"][f"commons-ten-v1 {split}"])
        for slug, v in pred[split].items():
            if slug != "split_total":
                assert a["buildings"][slug]["metrics"]["photos_accepted"] == v["accepted_before"], slug
                assert v["accepted_predicted"] == v["accepted_before"] + v["newly_accepted"]
                assert v["facing_away_predicted"] == v["facing_away_before"] - v["newly_accepted"] >= 0
        # the stated thresholds are compare's own on today's noise sources
        t = crit["thresholds_today"][split]
        assert t["models_published"]["threshold_today"] == 1.0 and t["facades_published"]["threshold_today"] == 0.0
        assert t["photo_pose_determined"]["threshold_today"] == 13.0
    held = crit["held_out"]
    assert sum(c.startswith("the B command") for c in held["run_once"]) == 1          # one COLMAP run of the split
    assert [c.split("--name ")[1].split()[0] for c in held["run_once"] if c.startswith("rescore")] == [
        "d34-fresh-plain-test", "d34-fresh-own-test", "d34-fresh-off-test"]
    assert any("--criteria d34-v1 --noise-factor 0" in c for c in held["run_once"])
    assert any("--criteria d34b-v1" in c and "d30-a-test.json" in c for c in held["run_once"])
    assert "verdict_d" in held and "verdict_georegistration" in held and len(held["defaults"]) == 5


D34B_FROZEN = ("criteria_id", "description", "baseline", "configs", "decisions", "predicted_acceptance",
               "thresholds_today", "georeg_baseline", "dev_rule", "held_out", "rules", "checks")
D34B_FROZEN_SHA = "2c271a494dadc22c425783bdf4f062d63430e9cf31e2258939a17b20aca8c3c9"
D34B_FROZEN_COMMIT = "e3928f0035892206c6bb20a21a44bc69e7a39761"
# dev_outcome as committed before the held-out run (canonical sha256), and that commit
D34B_DEV_OUTCOME_SHA = "000541b4bc0ced0ecc13e039e5524a4da16b33dd5f3e16902ca4ee72665477bb"
D34B_DEV_OUTCOME_COMMIT = "62131601a6e70a82a5cd6d5319a293f4a837f16f"
# held_out_outcome without its addendum, as committed with the held-out results (canonical sha256), and that commit;
# the addendum written after the step's review
D34B_HELD_OUT_SHA = "5453fde9ba63efc60d06a9ca5088df484426ef9636021b1ef75f94a3dc27cdd3"
D34B_HELD_OUT_COMMIT = "776b0e77265c201a024040c04c42d9c985bcc11d"
D34B_HELD_OUT_ADDENDUM_SHA = "151cde56e06aad45a16b2393af3091d5b448a639a364e7ef866fa245cac13dc8"
# the erratum of dev_outcome: per building (accepted, photos carrying heading_away_unverified, facing_away rejections)
# with Settings.facing_away_warns_other_refs on, recomputed from the photo rows of the recorded D31 runs
D34B_FUNNELS = {
    "dev": {"florence-baptistery": (54, 3, 2), "aachen-cathedral": (75, 0, 16), "brandenburg-gate": (139, 4, 4),
            "leaning-tower-pisa": (116, 1, 5), "frauenkirche-dresden": (88, 5, 0)},
    "test": {"sagrada-familia": (143, 15, 5), "hungarian-parliament": (90, 4, 2), "lincoln-memorial": (60, 3, 12),
             "india-gate": (71, 10, 1), "sydney-opera-house": (65, 4, 10)}}


def _d34b_compare(bench, pair: dict) -> tuple[dict, dict, dict, list[dict]]:
    """compare A B --criteria d34b-v1 made again from the committed result files, with the noise sources the recorded
    comparison names (results committed later would otherwise change the thresholds of an old pair). Returns A, B,
    the comparison and its guard rows as an outcome records them: metric, A, B, outcome and detail."""
    a, b = bench.load_result(ROOT / pair["A"]), bench.load_result(ROOT / pair["B"])
    named = {s.split(" (")[0] for s in pair["noise_sources"]}
    sources = [s for s in bench.noise_sources([a, b], [], bench.RESULTS_DIR) if s["run"] in named]
    assert {s["run"] for s in sources} == named
    c = bench.compare_results(a, b, sources)
    assert c["noise_sources"] == pair["noise_sources"]
    rows = bench.apply_criteria(bench.load_criteria("d34b-v1"), c, b)["rules"]
    return a, b, c, [{"metric": r["metric"], "A": r["a"], "B": r["b"], "outcome": r["outcome"], "detail": r["detail"]}
                     for r in rows if r["role"] == "guard"]


def _d34b_georeg_errors(bench, result: dict, split: str) -> tuple[dict, dict]:
    """benchmark.py georeg's errors over a result's distinct checked models of the split, and the same over each
    repeat's (check georeg_errors of d34b-v1)."""
    rows = [row for s, rec in result["buildings"].items() if bench.split_of(result, s) == split
            for row in rec.get("calibration") or []]
    return (bench.georeg_report(result, split)["overall"]["errors"],
            {run: bench.georeg_errors([row for row in rows if row["run"] == run]) for run in ("r1", "r2")})


def _d34b_range_coverage(result: dict) -> dict:
    """The figures of photoposes.MEASURED_COVERAGE from a result file: per source, the photos with a reference pose
    whose public row lies inside its own range, both repeats counted, with the lowest and the highest landmark-run
    (the first of equal shares)."""
    out = {}
    for src in ("sfm", "exif"):
        runs = [(int(m[f"photo_pose_{src}_within_uncertainty"]), int(m[f"photo_pose_{src}_gt"]), slug, f"{slug} r{i}")
                for slug, rec in result["buildings"].items() for i, x in enumerate(rec.get("repeats") or [], 1)
                if (m := x["metrics"]).get(f"photo_pose_{src}_gt")]
        low, high = min(runs, key=lambda x: x[0] / x[1]), max(runs, key=lambda x: x[0] / x[1])
        out[src] = {"within": sum(x[0] for x in runs), "n": sum(x[1] for x in runs),
                    "landmarks": len({x[2] for x in runs}),
                    "lowest": {"within": low[0], "n": low[1], "landmark_run": low[3]},
                    "highest": {"within": high[0], "n": high[1], "landmark_run": high[3]}}
    return out


def test_d34b_dev_outcome_applies_the_frozen_dev_rule_and_names_the_held_out_config(bench):
    """The dev outcome of d34b-v1 (added after the fresh dev runs, before the held-out run; docs/BENCHMARK.md 6.14).
    The twelve frozen fields are the file of the commit that froze them. The two fresh dev runs it names were made
    from that commit's clean tree with the two frozen configs and accepted the photos predicted. The guard rows it
    records are what compare gives on the committed result files. The nomination is the frozen dev rule's: (d) is
    nominated only when no guard is held against it on the frozen pair, or when the run with (d) on passes the same
    guards against the fresh run with (d) off."""
    crit = bench.load_criteria("d34b-v1")
    assert tuple(crit)[:len(D34B_FROZEN)] == D34B_FROZEN and list(crit)[len(D34B_FROZEN)] == "dev_outcome"
    assert bench.canonical_sha256({k: crit[k] for k in D34B_FROZEN}) == D34B_FROZEN_SHA
    dev = crit["dev_outcome"]
    pred = crit["predicted_acceptance"]["dev"]
    runs = ((dev["run"], "d_on", "d34-dev", "eval/results/commons-ten-v1/d34-dev.json", True),
            (dev["exception_run"]["run"], "d_off", "d34-dev-doff",
             "eval/results/commons-ten-v1/d34-fresh/d34-dev-doff.json", False))
    for run, key, name, rel, warns in runs:
        result = bench.load_result(ROOT / run["result"])
        prov = result["provenance"]
        assert run["result"] == rel and bench.sha256_file(ROOT / rel) == run["sha256"]
        assert result["run_name"] == run["run_name"] == name and "rescored_from" not in prov       # a fresh run
        assert prov["git"]["commit"] == run["commit"] == D34B_FROZEN_COMMIT and prov["git"]["dirty"] is False
        assert run["config"] == key and result["config"]["sha256"] == run["config_sha256"] == crit["configs"][key]["sha256"]
        assert result["config"]["effective"]["settings"]["facing_away_warns_other_refs"] is warns
        assert prov["code_sha256"] == run["code_sha256"] == dev["run"]["code_sha256"]            # the same code
        assert prov["recon_output_version"] == 4 and prov["repeat"] == 2 and prov["time_cap_s"] == 3600.0
        assert prov["stylize"] is True and prov["max_requests"] == {"brandenburg-gate": 104,
                                                                    "frauenkirche-dresden": 31}
        for slug, v in pred.items():
            if slug == "split_total":
                continue
            rec = result["buildings"][slug]
            assert rec["status"] == "ok" and len(rec["repeats"]) == 2 and not rec["details"]["truncated"], slug
            accepted, _, facing_away = D34B_FUNNELS["dev"][slug] if warns else (
                v["accepted_before"], 0, v["facing_away_before"])
            assert [r["metrics"]["photos_accepted"] for r in rec["repeats"]] == [accepted] * 2, slug
            assert rec["details"]["funnel"]["rejected_by_reason"].get("facing_away", 0) == facing_away, slug
    # the fresh run with (d) off is kept where compare takes no noise from: the frozen file names d34-dev as a noise
    # source of the held-out pair, not this run
    assert not (ROOT / "eval/results/commons-ten-v1/d34-dev-doff.json").exists()
    # the erratum: the corrected funnels, on both splits, and what they correct
    err = dev["erratum"]["predicted_acceptance"]
    assert {sp: {s: (v["accepted"], v["warned"], v["facing_away"]) for s, v in err[sp].items()}
            for sp in err} == D34B_FUNNELS
    for split in ("dev", "test"):
        for slug, v in crit["predicted_acceptance"][split].items():
            if slug != "split_total":
                accepted, warned, facing_away = D34B_FUNNELS[split][slug]
                assert accepted == v["accepted_predicted"]                      # the accepted counts were right
                assert facing_away == v["of_them_tagged_true_north"]
                # every facing_away photo not tagged true north carries the warning, accepted or not
                assert warned == v["facing_away_before"] - facing_away >= v["newly_accepted"]
    # each pair: the guard rows are the frozen file's, in order, and compare gives them on the committed files
    guards = [r["metric"] for r in crit["rules"] if r["role"] == "guard"]
    frozen_pair, exception_pair = dev["frozen_pair"], dev["exception_run"]["pair"]
    assert (frozen_pair["A"], frozen_pair["B"]) == ("eval/results/commons-ten-v1/d30-a-dev.json", dev["run"]["result"])
    assert (exception_pair["A"], exception_pair["B"]) == (dev["exception_run"]["run"]["result"], dev["run"]["result"])
    for pair in (frozen_pair, exception_pair):
        recorded = [(r["metric"], r["outcome"]) for r in pair["rules"]["rows"]]
        assert [m for m, _ in recorded] == guards
        assert pair["rules"]["rows"] == _d34b_compare(bench, pair)[3]       # every value, outcome and detail
        failed = [m for m, o in recorded if o == "FAIL"]
        rule = pair["dev_rule"]
        assert [r["guard"] for r in rule["guard_rules_not_passed"] if r["outcome"] == "FAIL"] == failed
        held = [r["guard"] for r in rule["guard_rules_not_passed"] if r["held_strict"]]
        # noise_on_dev: only a not_worse guard on a summed count whose aggregate alone fails, within the sum of the
        # buildings' recorded spreads, is not held against (d)
        for r in rule["guard_rules_not_passed"]:
            if r["outcome"] == "FAIL" and not r["held_strict"]:
                assert r["require"] == "not_worse" and not r["buildings_significantly_worse"]
                agg = r["aggregate"]
                assert agg["fails_by_compare"] and abs(agg["delta"]) <= agg["sum_of_building_spreads"]
        check_fails = [k for k in ("accepted_photos_rise", "shown_pose_coverage", "georeg_errors",
                                   "newly_accepted_photos") if pair["checks"][k]["verdict"] == "FAIL"]
        assert rule["guards_held_against_d"] == held + check_fails
        assert rule["passes_the_guards"] == (not rule["guards_held_against_d"]
                                             and pair["checks"]["runs_comparable"]["verdict"] == "PASS")
        assert pair["checks"]["accepted_photos_rise"]["verdict"] == "PASS"      # with the erratum's funnels
    # the frozen pair as this commit records it
    assert [r["guard"] for r in frozen_pair["dev_rule"]["guard_rules_not_passed"]] == [
        "models_published", "facades_published", "gt_gross_shown"]
    assert frozen_pair["dev_rule"]["allowed_by_noise_on_dev"] == ["models_published", "facades_published"]
    assert frozen_pair["dev_rule"]["guards_held_against_d"] == ["gt_gross_shown"]
    # the exception pair: the same guard is held against (d) with the fresh d-off run as A
    assert [r["guard"] for r in exception_pair["dev_rule"]["guard_rules_not_passed"]] == [
        "facades_published", "gt_gross_shown"]
    assert exception_pair["dev_rule"]["allowed_by_noise_on_dev"] == ["facades_published"]
    assert exception_pair["dev_rule"]["guards_held_against_d"] == ["gt_gross_shown"]
    for pair in (frozen_pair, exception_pair):
        gross = next(r for r in pair["dev_rule"]["guard_rules_not_passed"] if r["guard"] == "gt_gross_shown")
        assert sorted(gross["buildings_significantly_worse"]) == ["brandenburg-gate", "frauenkirche-dresden"]
        assert not gross["aggregate"]["within_that_sum"]
    # the nomination: the frozen rule and nothing else
    nom = dev["nomination"]
    assert nom["frozen_pair_passes_the_guards"] is frozen_pair["dev_rule"]["passes_the_guards"] is False
    assert nom["exception_pair_passes_the_guards"] is exception_pair["dev_rule"]["passes_the_guards"] is False
    assert dev["nominated"] == ("d_on" if nom["frozen_pair_passes_the_guards"]
                                or nom["exception_pair_passes_the_guards"] else "d_off") == "d_off"
    assert dev["held_out_config"] == {k: crit["configs"][dev["nominated"]][k] for k in ("config", "sha256")}
    assert bench.sha256_file(ROOT / dev["held_out_config"]["config"]) == dev["held_out_config"]["sha256"]


def test_d34b_held_out_outcome_is_the_one_frozen_run_and_sets_the_defaults(bench):
    """The held-out outcome of d34b-v1 (docs/BENCHMARK.md 6.14), added after the one fresh held-out run. Neither the
    frozen fields nor dev_outcome changed. The run is the config dev_outcome nominated, made from the commit that
    recorded dev_outcome, with the code of the dev runs. (d) was not nominated, so its verdict is not judged and the
    switch stays off; the georegistration verdict is d34-v1 on the run's own replays, made again here from the three
    committed rescores, and the product's defaults are what the frozen file says that verdict gives. The guard rows
    and the checks reported for the fresh run against the recorded D31 run are made again from the committed files
    (every value and detail, the row that reads FAIL included), as are the yaw values behind that row, the ranges'
    measured coverage and the job tallies of repeat 1. What only the work dirs can give (the job tallies of repeat 2
    and of the levels) and the texts are held by the canonical sha256 of the outcome as it was committed."""
    from crowd3d import photoposes, reconstruct
    from crowd3d.config import Settings

    crit = bench.load_criteria("d34b-v1")
    assert tuple(crit)[:len(D34B_FROZEN) + 2] == D34B_FROZEN + ("dev_outcome", "held_out_outcome")
    assert bench.canonical_sha256({k: crit[k] for k in D34B_FROZEN}) == D34B_FROZEN_SHA
    dev, out = crit["dev_outcome"], crit["held_out_outcome"]
    assert bench.canonical_sha256(dev) == D34B_DEV_OUTCOME_SHA
    run = out["run"]
    result = bench.load_result(ROOT / run["result"])
    prov = result["provenance"]
    assert (run["started"], run["wall_s"], run["repeat"], run["max_requests"], run["dirty"]) == (
        prov["started"], prov["wall_s"], prov["repeat"], prov["max_requests"], prov["git"]["dirty"])
    assert run["result"] == "eval/results/commons-ten-v1/d34-test.json" and result["run_name"] == "d34-test"
    assert bench.sha256_file(ROOT / run["result"]) == run["sha256"] and "rescored_from" not in prov
    assert prov["git"]["commit"] == run["commit"] == D34B_DEV_OUTCOME_COMMIT and prov["git"]["dirty"] is False
    # the nominated config, and the code the dev runs ran (the commit between them changed no code)
    assert run["config"] == dev["nominated"] == "d_off"
    assert result["config"]["sha256"] == run["config_sha256"] == dev["held_out_config"]["sha256"]
    assert result["config"]["effective"]["settings"]["facing_away_warns_other_refs"] is False
    assert prov["code_sha256"] == run["code_sha256"] == dev["run"]["code_sha256"]
    assert prov["recon_output_version"] == 4 and prov["repeat"] == 2 and prov["time_cap_s"] == 3600.0
    assert prov["stylize"] is True and not prov["max_requests"]
    held = ("sagrada-familia", "hungarian-parliament", "lincoln-memorial", "india-gate", "sydney-opera-house")
    assert tuple(s for s, r in result["buildings"].items() if r.get("metrics")) == held
    a = bench.load_result(ROOT / "eval/results/commons-ten-v1/d30-a-test.json")
    for slug in held:                       # without (d) the same photos are accepted as in the recorded run
        rec = result["buildings"][slug]
        assert rec["status"] == "ok" and len(rec["repeats"]) == 2, slug
        assert [r["metrics"]["photos_accepted"] for r in rec["repeats"]] == [
            a["buildings"][slug]["metrics"]["photos_accepted"]] * 2, slug
    # (d): not nominated, not judged, off
    assert out["verdict_d"] == "NOT JUDGED" and Settings().facing_away_warns_other_refs is False
    # (a) + (b): d34-v1 on the three rescores of the run
    fresh = out["fresh_pair"]
    parts = [fresh[k] for k in ("own_replay_equals_plain_rescore", "helps", "no_harm", "stylization_not_worse",
                                "rules_of_d34_v1")]
    assert out["verdict_georegistration"] == fresh["verdict"] == (
        "FAIL" if "FAIL" in parts else "INSUFFICIENT" if "INSUFFICIENT" in parts else "PASS")
    for arm, replay in (("plain", None), ("own", "default"), ("off", "d34-regeo-off")):
        f = fresh["results"][arm]
        r = bench.load_result(ROOT / f["file"])
        assert f["file"] == f"eval/results/commons-ten-v1/d34-fresh/d34-fresh-{arm}-test.json"
        assert bench.sha256_file(ROOT / f["file"]) == f["sha256"] and r["run_name"] == f"d34-fresh-{arm}-test"
        assert r["provenance"]["rescored_from"]["run_name"] == "d34-test"
        assert r["provenance"]["rescored_from"]["sha256"] == run["sha256"]
        assert r["provenance"]["git"]["commit"] == run["commit"] and r["provenance"]["git"]["dirty"] is False
        assert (r["provenance"].get("replay") or {}).get("id") == replay, arm
    # the own-parameter replay moved nothing; the old-georegistration replay is the control of d34-v1
    own = fresh["own_replay"]
    assert own["replayed"] == own["models"] and not (own["transform_changed"] or own["level_changed"]
                                                     or own["published_changed"])
    assert fresh["off_replay"]["replayed"] == fresh["off_replay"]["models"] == own["models"]
    # the recorded numbers are what `benchmark.py georeg --split test` gives on the committed rescores, and the
    # recorded parts of the verdict are d34-v1's arithmetic on them (its dev_rule.helps_any_of and no_harm_all_of)
    loaded = {arm: bench.load_result(ROOT / fresh["results"][arm]["file"]) for arm in ("plain", "own", "off")}
    overall = {arm: bench.georeg_report(loaded[arm], "test")["overall"] for arm in ("own", "off")}
    covers = ("calib_yaw_cover2", "calib_scale_cover2", "calib_xy_cover2", "calib_tilt_cover2")
    counted = ("models_published", "photo_pose_determined", "photo_pose_correct", "photo_pose_wrong",
               "shown_beyond_2sigma", "wrong_face_models_published", "wrong_face_cameras_shown", *covers)
    moved = ("models", "replayed", "left_as_recorded", "transform_changed", "level_changed", "published_changed")
    for arm in ("own", "off"):
        e, c, rec = overall[arm]["errors"], overall[arm]["counts"], fresh[arm]
        assert rec == {**rec, "checked_models": e["models"], "scale_rms_ln": e["scale_rms_ln"],
                       "scale_off": f"{e['scale_off']}/{e['scale_n']}", "yaw_rms_deg": e["yaw_rms_deg"],
                       "yaw_median_deg": e["yaw_median_deg"], "yaw_off": f"{e['yaw_off']}/{e['yaw_n']}",
                       "pivot_error_median_m": e["xy_median_m"], **{k: c[k] for k in counted}}, arm
        assert {k: overall[arm]["replay"][k] for k in moved} == {k: fresh[f"{arm}_replay"][k] for k in moved}, arm
        # the change against the old-georegistration replay, in percent (the off arm against itself: none)
        assert {k: rec[k] for k in ("yaw_rms_change_pct", "scale_rms_change_pct")} == {
            f"{k}_change_pct": round(100 * (e[f"{k}_{unit}"] - overall["off"]["errors"][f"{k}_{unit}"])
                                     / overall["off"]["errors"][f"{k}_{unit}"], 1)
            for k, unit in (("yaw_rms", "deg"), ("scale_rms", "ln"))}, arm
    assert {k: overall["own"]["replay"][k] for k in ("runs", "message_differs")} == {
        k: own[k] for k in ("runs", "message_differs")}
    assert fresh["own_replay_largest_camera_difference_m"] == max(
        (rp.get("max_faithful_diff_m") or 0.0) for rec in loaded["own"]["buildings"].values()
        for rp in rec.get("replay") or [])
    (e1, c1), (e0, c0) = ((overall[arm]["errors"], overall[arm]["counts"]) for arm in ("own", "off"))
    helps = {"scale_off_fewer_by_2": e0["scale_off"] - e1["scale_off"] >= 2,
             "scale_rms_lower_by_5pct": e1["scale_rms_ln"] <= 0.95 * e0["scale_rms_ln"],
             "yaw_rms_lower_by_5pct": e1["yaw_rms_deg"] <= 0.95 * e0["yaw_rms_deg"],
             "yaw_off_fewer_by_2": e0["yaw_off"] - e1["yaw_off"] >= 2,
             "wrong_lower_by_2_right_not_lower": (c0["photo_pose_wrong"] - c1["photo_pose_wrong"] >= 2
                                                   and c1["photo_pose_correct"] >= c0["photo_pose_correct"]),
             "right_higher_by_2_wrong_not_higher": (c1["photo_pose_correct"] - c0["photo_pose_correct"] >= 2
                                                     and c1["photo_pose_wrong"] <= c0["photo_pose_wrong"])}
    assert fresh["helps_by"] == [k for k, met in helps.items() if met]
    assert fresh["helps"] == ("PASS" if fresh["helps_by"] else "FAIL")
    assert all(c1[k] is not None and c0[k] is not None for k in covers) and fresh["coverages_not_judged"] == []
    no_harm = {"published_not_lower_by_more_than_1": c1["models_published"] >= c0["models_published"] - 1,
               "shown_not_lower_by_more_than_1pct": c1["photo_pose_determined"] >= 0.99 * c0["photo_pose_determined"],
               "right_not_lower_by_more_than_1": c1["photo_pose_correct"] >= c0["photo_pose_correct"] - 1,
               "beyond_2sigma_not_higher_by_more_than_1": c1["shown_beyond_2sigma"] <= c0["shown_beyond_2sigma"] + 1,
               "wrong_face_models_not_higher": c1["wrong_face_models_published"] <= c0["wrong_face_models_published"],
               "wrong_face_cameras_not_higher": c1["wrong_face_cameras_shown"] <= c0["wrong_face_cameras_shown"],
               **{f"{k}_not_lower_by_more_than_0.02": c1[k] >= c0[k] - 0.02 for k in covers},
               "scale_off_not_higher_by_more_than_1": e1["scale_off"] <= e0["scale_off"] + 1,
               "yaw_off_not_higher_by_more_than_1": e1["yaw_off"] <= e0["yaw_off"] + 1}
    assert fresh["harm"] == [k for k, ok in no_harm.items() if not ok]
    assert fresh["no_harm"] == ("FAIL" if fresh["harm"] else "PASS")
    # d34-v1's own rules on the pair, without noise (the two replays score the same SfM output)
    rows = bench.apply_criteria(bench.load_criteria("d34-v1"),
                                bench.compare_results(loaded["off"], loaded["own"], [], 0.0), loaded["own"])
    assert rows["verdict"] == fresh["rules_of_d34_v1"]
    assert fresh["d34_v1_rule_rows"] == [
        {"metric": r["metric"], "role": r["role"], "A": r["a"], "B": r["b"], "outcome": r["outcome"],
         "detail": r["detail"]} for r in rows["rules"] if r["role"] == "guard"]
    assert len(fresh["d34_v1_rule_rows"]) == 2
    # in the place of d34-v1's precondition: the replay under the run's own parameters scores as the plain rescore
    # (every metric of every repeat but the timings and sizes of processes, every calibration row), and the
    # stylization is not worse
    def timing(name: str) -> bool:
        return name.endswith(("_s", "_mb")) or any(part in name for part in ("cpu", "rss", "latency", "time"))

    for slug in held:
        for arm, other in (("plain", "own"), ("off", "own")):
            for x, y in zip(loaded[arm]["buildings"][slug]["repeats"], loaded[other]["buildings"][slug]["repeats"],
                            strict=True):
                if arm == "plain":
                    assert {k: v for k, v in x["metrics"].items() if not timing(k)} == {
                        k: v for k, v in y["metrics"].items() if not timing(k)}, slug
                else:
                    for k in ("style_registration_wrong", "style_qa_failed", "struct_incorrect", "struct_qa_failed"):
                        assert (x["metrics"].get(k) is None) == (y["metrics"].get(k) is None), (slug, k)
                        assert x["metrics"].get(k) is None or y["metrics"][k] <= x["metrics"][k], (slug, k)
        assert loaded["plain"]["buildings"][slug].get("calibration") == loaded["own"]["buildings"][slug].get(
            "calibration"), slug
    assert sum(len(loaded["own"]["buildings"][s].get("calibration") or []) for s in held) == 47
    assert fresh["own_replay_equals_plain_rescore"] == fresh["stylization_not_worse"] == "PASS"
    # reported, not a verdict: this file's guard rows for the fresh run against the recorded D31 run are what compare
    # gives on the committed files; the one that reads FAIL is recorded as such and named
    rep = out["reported_not_a_verdict"]
    assert (rep["A"], rep["B"]) == ("eval/results/commons-ten-v1/d30-a-test.json", run["result"])
    _, b, cmp_, guard_rows = _d34b_compare(bench, rep)
    assert rep["rules"]["rows"] == guard_rows and len(guard_rows) == 14       # every value, outcome and detail
    against = out["against_the_recorded_run"]
    assert [r["metric"] for r in guard_rows if r["outcome"] != "PASS"] == [
        against["one_guard_row_reads_fail"]["metric"]] == ["geo_yaw_error_median_deg"]
    assert rep["rules"]["verdict_by_compare"] == "FAIL"
    # the checks of this file that read the two result files and their comparison, made again (the per-photo check
    # and accepted_photos_rise are (d)'s: not judged without it)
    checks, git = rep["checks"], prov["git"]
    comparable = (not git["dirty"] and not cmp_["buildings_inputs_differ"] and not cmp_["buildings_only_in_a"]
                  and not cmp_["buildings_only_in_b"] and all(
                      result["buildings"][s]["status"] == "ok" and not result["buildings"][s]["details"]["truncated"]
                      and [x["status"] for x in result["buildings"][s]["repeats"]] == ["ok", "ok"]
                      and result["buildings"][s]["details"]["requests"] == a["buildings"][s]["details"]["requests"]
                      for s in held))
    assert checks["runs_comparable"] == out["precondition_runs_comparable"] == {
        "verdict": "PASS" if comparable else "INSUFFICIENT",
        "detail": (f"{len(held)} buildings x 2 repeats ran to the end with A's request counts, inputs identical, "
                   f"commit {git['commit'][:7]} clean, config {result['config']['id']} ({run['config']}), "
                   f"code_sha256 {prov['code_sha256'][:12]}")}
    j = cmp_["splits"]["test"]["photos_accepted"]
    assert checks["accepted_photos_rise"] == {
        "verdict": "NOT JUDGED", "detail": f"config d_off: (d) is off; split {j['a']:g} -> {j['b']:g}"}
    j = cmp_["splits"]["test"]["photo_pose_uncertainty_coverage"]
    assert checks["shown_pose_coverage"] == {
        "verdict": "FAIL" if j["delta"] < 0 and j["significant"] else "PASS",
        "detail": f"{j['a']:g} -> {j['b']:g} (delta {j['delta']:+g}, threshold {j['threshold']:g}, noise {j['noise']})"}
    (pa, ra), (pb, rb) = _d34b_georeg_errors(bench, a, "test"), _d34b_georeg_errors(bench, b, "test")
    base = crit["georeg_baseline"]["test"]                 # A's numbers are the frozen file's
    assert (pa["models"], pa["scale_rms_ln"], pa["yaw_rms_deg"], f"{pa['scale_off']}/{pa['scale_n']}",
            f"{pa['yaw_off']}/{pa['yaw_n']}") == tuple(base[k] for k in (
                "checked_models", "scale_rms_ln", "yaw_rms_deg", "scale_off", "yaw_off"))
    parts, worse_in = [], []
    for key, unit in (("scale_rms_ln", ""), ("yaw_rms_deg", " deg")):
        sa, sb = (abs(r["r1"][key] - r["r2"][key]) for r in (ra, rb))
        margin = max(0.05 * pa[key], sa, sb)
        worse = pb[key] - pa[key] > margin * (1 + 1e-9) + 1e-9
        parts.append(f"{key} {pa[key]:g} -> {pb[key]:g}{unit} (margin {margin:.4g}: 5 % of A {0.05 * pa[key]:.4g}, "
                     f"A r1/r2 {sa:.3g}, B r1/r2 {sb:.3g}) {'WORSE' if worse else 'ok'}")
        worse_in += [key] if worse else []
    for key, n in (("scale_off", "scale_n"), ("yaw_off", "yaw_n")):
        worse, end = bench.rate_worse(pa[key], pa[n], pb[key], pb[n], higher_is_better=False)
        parts.append(f"{key} {pa[key]}/{pa[n]} = {pa[key] / pa[n]:.3f} -> {pb[key]}/{pb[n]} = "
                     f"{pb[key] / pb[n]:.3f} (95% end {end:+.3f}) {'WORSE' if worse else 'ok'}")
        worse_in += [key] if worse else []
    assert checks["georeg_errors"] == {
        "verdict": "FAIL" if worse_in else "PASS",
        "detail": f"checked models {pa['models']} -> {pb['models']}; " + "; ".join(parts)}
    assert set(checks) == {"runs_comparable", "accepted_photos_rise", "shown_pose_coverage", "georeg_errors"}
    # the row that reads FAIL: each building's value and the split's, in the recorded D31 run, in this run replayed
    # under the old georegistration, and in this run
    metric = against["one_guard_row_reads_fail"]["metric"]
    for name, r in (("D31_run_recorded", a), ("this_run_replayed_under_the_old_georegistration", loaded["off"]),
                    ("this_run", result)):
        values = {s: r["buildings"][s]["metrics"] for s in held}
        assert against["one_guard_row_reads_fail"][name] == {
            "split": bench.aggregate(values)[metric], **{s: values[s][metric] for s in held}}, name
    assert (guard_rows[-1]["A"], guard_rows[-1]["B"]) == tuple(
        against["one_guard_row_reads_fail"][k]["split"] for k in ("D31_run_recorded", "this_run"))
    # the jobs of the two runs, paired by request, kind, sector and number of photos: repeat 1 from the committed
    # details (repeat 2 and the levels are in the work dirs only), and the totals over what is recorded
    jobs = against["jobs"]
    for slug in held:
        ka, kb = ({(m["request"], m["kind"], m["sector"], m["photos"]): m
                   for m in r["buildings"][slug]["details"]["models"]} for r in (a, result))
        same = set(ka) & set(kb)
        assert {k: v for k, v in jobs["per_building"][slug]["r1"].items() if k != "level_differs"} == {
            "jobs": len(same), "only_in_A": len(set(ka) - set(kb)), "only_in_B": len(set(kb) - set(ka)),
            "registered_differs": sum(ka[k]["registered"] != kb[k]["registered"] for k in same),
            "same_sfm_model": sum((ka[k]["registered"], ka[k]["points"]) == (kb[k]["registered"], kb[k]["points"])
                                  for k in same),
            "published_differs": sum(ka[k]["published"] != kb[k]["published"] for k in same)}, slug
    assert jobs["totals_over_both_repeats"] == {
        k: sum(jobs["per_building"][s][r][k] for s in held for r in ("r1", "r2"))
        for k in jobs["totals_over_both_repeats"]}
    # the measured coverage of the public ranges on this run and on the D31 run; the constant in the code is the
    # D31 run's
    cover = out["public_range_coverage_recomputed"]
    assert cover["this_run"] == _d34b_range_coverage(result)
    assert cover["D31_run_recorded"] == _d34b_range_coverage(a)
    for src, key in ((photoposes.SRC_SFM, "sfm"), (photoposes.SRC_EXIF, "exif")):
        const, measured = photoposes.MEASURED_COVERAGE[src], cover["D31_run_recorded"][key]
        assert (const["within"], const["n"], const["landmarks"]) == (
            measured["within"], measured["n"], measured["landmarks"])
        assert all(const[k] == {x: measured[k][x] for x in ("within", "n")} for k in ("lowest", "highest"))
    # the defaults are what held_out.defaults gives for these verdicts
    ok = out["verdict_georegistration"] == "PASS"
    assert out["defaults"] == {"gps_span_model_side": ok, "compass_other": "ignored" if ok else "trusted",
                               "yaw_arbitration": False, "facing_away_warns_other_refs": False,
                               "recon_output_version": 4 if ok else 5}
    assert {k: getattr(params.DEFAULTS, k) for k in ("gps_span_model_side", "compass_other", "yaw_arbitration")} == {
        k: out["defaults"][k] for k in ("gps_span_model_side", "compass_other", "yaw_arbitration")}
    assert reconstruct.RECON_OUTPUT_VERSION == out["defaults"]["recon_output_version"]
    # the outcome is the one committed with the held-out results: the texts and what only the work dirs can give
    # (the job tallies of repeat 2, the levels) included; the review's addendum comes after it, as its last key
    assert list(out)[-1] == "addendum"
    assert bench.canonical_sha256({k: v for k, v in out.items() if k != "addendum"}) == D34B_HELD_OUT_SHA


def test_d34b_held_out_addendum_narrows_what_the_failing_row_was_said_to_show(bench):
    """The addendum to held_out_outcome (after the step's review; docs/BENCHMARK.md 6.14): the outcome had said the
    one guard row that reads FAIL against the D31 run is no effect of (a) + (b), on the strength of a replay, which
    cannot see a job's candidate choice. The addendum names the two sentences, says what is shown and what is not,
    and records the evidence. What the committed result files hold of it is made again here: the split's value with
    the Hungarian Parliament at its D31 value, the building's value by repeat as the median of its checked W facade
    models, those models' yaw errors, what repeat 1 of both runs kept, that the old georegistration changes no level
    or publication there, and the arithmetic of the 38-photo job. What comes from the harness logs and from judging
    the kept SfM models again (the attempts, the heading tags, the tallies under both parameter sets, repeat 2) is in
    no committed file: the canonical sha256 of the addendum holds it. No verdict and no default follows from it."""
    import statistics

    crit = bench.load_criteria("d34b-v1")
    out = crit["held_out_outcome"]
    add = out["addendum"]
    assert bench.canonical_sha256(add) == D34B_HELD_OUT_ADDENDUM_SHA
    assert bench.canonical_sha256({k: v for k, v in out.items() if k != "addendum"}) == D34B_HELD_OUT_SHA
    assert f"as committed in {D34B_HELD_OUT_COMMIT[:7]}" in add["added"]
    # the sentences it narrows stand where it says, unchanged
    for claim in add["narrows"]["claims"]:
        field = out
        for part in claim["where"].split("."):
            field = field[part]
        assert claim["text"] in field, claim
    slug, metric = "hungarian-parliament", "geo_yaw_error_median_deg"
    files = {"D31_run_recorded": "eval/results/commons-ten-v1/d30-a-test.json",
             "this_run_replayed_under_the_old_georegistration": out["fresh_pair"]["results"]["off"]["file"],
             "this_run": out["run"]["result"]}
    res = {k: bench.load_result(ROOT / f) for k, f in files.items()}
    held = [s for s, r in res["this_run"]["buildings"].items() if r.get("metrics")]
    # the split's value: this run, the D31 run, and this run with this one building at its D31 value
    values = {k: {s: dict(res[k]["buildings"][s]["metrics"]) for s in held} for k in ("D31_run_recorded", "this_run")}
    swapped = {**values["this_run"], slug: values["D31_run_recorded"][slug]}
    assert add["the_row"]["split"] == {
        "D31_run_recorded": bench.aggregate(values["D31_run_recorded"])[metric],
        "this_run": bench.aggregate(values["this_run"])[metric],
        "this_run_with_this_building_at_its_D31_value": bench.aggregate(swapped)[metric]}
    assert add["the_row"]["split"]["this_run_with_this_building_at_its_D31_value"] < add["the_row"]["split"][
        "D31_run_recorded"] < add["the_row"]["split"]["this_run"]
    # the building's value by repeat, the same under the old georegistration, which changes no level or publication
    value = add["building_value"]
    assert value["by_repeat"] == {k: [r["metrics"][metric] for r in res[k]["buildings"][slug]["repeats"]] for k in res}
    assert value["value"] == {k: res[k]["buildings"][slug]["metrics"][metric] for k in res}
    assert value["by_repeat"]["this_run"] == value["by_repeat"]["this_run_replayed_under_the_old_georegistration"]
    # compare's own row for the building: by the noise it knows, the rise is significant (the addendum says so)
    row = _d34b_compare(bench, out["reported_not_a_verdict"])[2]["per_building"][slug]["metrics"][metric]
    recorded = {k: v for k, v in value["compare_row_of_this_building"].items() if k != "what"}
    assert recorded == {k: row[k] for k in recorded} and set(recorded) == {
        "a", "b", "delta", "noise", "threshold", "significant"}
    assert row["significant"] and row["verdict"] == "worse" and row["delta"] > row["threshold"] == max(
        round(abs(v[0] - v[1]), 2) for v in (value["by_repeat"]["D31_run_recorded"], value["by_repeat"]["this_run"]))
    off_reports = res["this_run_replayed_under_the_old_georegistration"]["buildings"][slug]["replay"]
    assert value["old_georegistration_changes_nothing_here"]["by_repeat"] == [
        {"run": rp["run"], "models": rp["models"], "replayed": rp["replayed"],
         "left_as_recorded": len(rp["left_as_recorded"]), "transform_changed": rp["transform_changed"],
         "level_changed": len(rp["level_changed"]), "published_changed": len(rp["published_changed"])}
        for rp in off_reports]
    assert all(rp["level_changed"] == rp["published_changed"] == [] for rp in off_reports)
    # the W facade models: the ones counted are the checked models of the result files, and a repeat's value is
    # their median
    models = add["w_facade_models"]["models"]
    assert [m["request"] for m in models] == [25, 28, 32, 41]
    for label, key, i in (("D31 r1", "D31_run_recorded", 0), ("D31 r2", "D31_run_recorded", 1),
                          ("fresh r1", "this_run", 0), ("fresh r2", "this_run", 1)):
        counted = [m["by_repeat"][label] for m in models if m["by_repeat"][label]["counted"]]
        assert all(r["published"] and r["georegistered"] for r in counted)
        rows = [row for row in res[key]["buildings"][slug]["calibration"] if row["run"] == f"r{i + 1}"]
        assert all((row["kind"], row["sector"]) == ("face", "W") for row in rows)
        assert sorted(r["yaw_error_abs_deg"] for r in counted) == sorted(abs(row["comps"]["yaw"]["e"]) for row in rows)
        assert value["by_repeat"][key][i] == round(statistics.median(r["yaw_error_abs_deg"] for r in counted), 2)
        if key == "this_run":                     # rows recorded since D34 name the registered count
            assert sorted(r["registered"] for r in counted) == sorted(row["registered"] for row in rows)
        for m in models:                          # the attempt kept is one of the attempts, with these counts
            r = m["by_repeat"][label]
            assert [r["kept"], r["registered"], r["flagged"]] in r["attempts_registered_flagged"], (m["request"], label)
        if i == 0:                                # repeat 1: what the committed details say of each kept model
            details = {(d["request"], d["sector"]): d for d in res[key]["buildings"][slug]["details"]["models"]
                       if d["kind"] == "face"}
            for m in models:
                d, r = details[(m["request"], "W")], m["by_repeat"][label]
                assert (m["photos"], r["registered"], r["flagged"], r["published"], r["georegistered"]) == (
                    d["photos"], d["registered"], d["flagged"], d["published"], d["georegistered"]), (
                        m["request"], label)
            # the other W facade models published nothing
            assert not [d for (req, sector), d in details.items() if sector == "W" and d["published"]
                        and req not in (25, 28, 32, 41)]
    # repeat 1 of the fresh run in the two committed replays: every kept model of the building has the same
    # registered, flagged and published under the old georegistration and under the run's own parameters
    own = bench.load_result(ROOT / out["fresh_pair"]["results"]["own"]["file"])
    kept = [[(d["request"], d["kind"], d["sector"], d["registered"], d["flagged"], d["published"], d["georegistered"])
             for d in r["buildings"][slug]["details"]["models"]]
            for r in (res["this_run_replayed_under_the_old_georegistration"], own)]
    assert kept[0] == kept[1] and len(kept[0]) == 15
    tally = add["job_time"]["a_on_the_kept_models"]
    assert tally["models"] == sum(rp["models"] for rp in off_reports) == 30
    assert tally["w_facade_models"] == 2 * sum(k[2] == "W" for k in kept[0]) == 14
    assert tally["models"] == tally["same_unflagged_count_and_same_flagged_cameras"] == tally[
        "same_georegistered_level_and_yaw_source"] == tally["same_retry_verdict"]
    assert tally["w_facade_models"] == tally["w_facade_models_same_in_every_part"]
    # where the two parameter sets do differ: full jobs only, none published
    assert [(x["repeat"], x["request"], x["kind"], x["published"]) for x in tally["compass_error_differs"]] == [
        ("r1", 31, "full", False), ("r1", 35, "full", False), ("r2", 31, "full", False)]
    assert all(x["defaults"] == {"compass_error_deg": None, "cameras_off_both_gps_and_compass": 0}
               for x in tally["compass_error_differs"])
    # the full job where (b) took the twin-pair signal away: what repeat 1 of both runs kept, by the committed details
    lost = add["lost_twin_pair_signal"]["request_31_full_47_photos"]
    for label, key, block in (("D31 r1", "D31_run_recorded", "D31_run"), ("fresh r1", "this_run", "this_run")):
        d = next(x for x in res[key]["buildings"][slug]["details"]["models"]
                 if (x["request"], x["kind"]) == (31, "full"))
        r = lost[block][label]
        attempt = next(x for x in r["attempts_registered_flagged_compass_error"] if x[0] == r["kept"])
        assert (d["photos"], d["registered"], d["flagged"], d["qa_reasons"], d["published"]) == (
            47, attempt[1], attempt[2], r["qa_codes"], False), label
    assert lost["D31_run"]["D31 r1"]["kept"] == "cut" and all(
        x[0] != "cut" for r in lost["this_run"].values() for x in r["attempts_registered_flagged_compass_error"])
    # no W facade photo has a heading (b) acts on
    tags = add["job_time"]["b_cannot_act_in_the_w_jobs"]["photos_by_heading_tag"]
    assert all(set(t) <= {"no heading", "T"} for t in tags.values())
    assert {int(req): sum(t.values()) for req, t in tags.items()} == {
        d["request"]: d["photos"] for d in res["this_run"]["buildings"][slug]["details"]["models"]
        if (d["kind"], d["sector"]) == ("face", "W")}
    # the 38-photo job: unflagged cameras of the kept and the lost attempt, from the attempts recorded above
    big = next(m for m in models if m["request"] == 41)
    assert big["photos"] == 38 > params.DEFAULTS.small_set_images and params.DEFAULTS.mapping_attempts_large == 2
    margins = add["job_time"]["the_38_photo_job"]["unflagged_kept_then_lost"]
    for label, r in big["by_repeat"].items():
        unflagged = {a[0]: a[1] - a[2] for a in r["attempts_registered_flagged"]}
        assert len(unflagged) == 2 and margins[label] == [unflagged.pop(r["kept"]), *unflagged.values()], label
        assert margins[label][0] > margins[label][1]              # the kept attempt has more unflagged cameras
    assert [margins[k][0] - margins[k][1] for k in ("fresh r1", "fresh r2")] == [3, 3]
    # the dev reruns it cites for (a) alone: no candidate with another flag, verdict or level
    runs = add["dev_reruns"]
    assert runs["candidates_b_cannot_act_on"] == 30 <= runs["candidates"] and not any(
        v for k, v in runs.items() if k.startswith("of_them_another"))
    # the addendum decides nothing: the verdicts and the defaults are the outcome's
    assert not {"verdict_d", "verdict_georegistration", "defaults"} & set(add)


def test_d30_criteria_judge_the_localizer_against_the_paired_baseline(bench):
    """compare --criteria d30-v1 --noise-factor 0: a 0.02 gain with no wrong localized pose passes (the gain is
    reported, the coverage with its Wilson interval); one wrong-face pose on one repeat, CPU over a quarter of the
    run's, or a building whose wrong share rises fails."""
    base = {s: _d30_metrics() for s in ("x", "y", "z", "w")}
    good = {s: _d30_metrics(photo_pose_correct=42, photo_pose_correct_rate_testset=0.42, photo_pose_gt_determined=52,
                            photo_pose_wrong_share=round(10 / 52, 4), photo_pose_loc=3, photo_pose_loc_gt=2,
                            photo_pose_loc_correct=2, photo_pose_loc_within_uncertainty=2,
                            photo_pose_loc_uncertainty_coverage=1.0, photo_pose_loc_queries=20,
                            photo_pose_loc_cpu_s=200.0) for s in base}
    cr = _d30(bench, base, good)
    assert cr["verdict"] == "PASS", [r for r in cr["rules"] if r["outcome"] not in ("PASS", "REPORT")]
    gain = next(r for r in cr["rules"] if r["metric"] == "photo_pose_correct_rate_testset" and r["split"] == "dev")
    assert gain["outcome"] == "REPORT" and gain["detail"].startswith("better (significant)")
    cover = next(r for r in cr["rules"] if r["metric"] == "photo_pose_loc_uncertainty_coverage" and r["split"] == "dev")
    assert cover["detail"].endswith("B 4/4, 95% Wilson 0.510-1.000")
    cost = next(r for r in cr["rules"] if r["metric"] == "photo_pose_loc_cpu_s" and r["split"] == "test")
    assert (cost["outcome"], cost["detail"]) == ("PASS", "400/2000 = 0.200")
    # one wrong-face localized pose on one repeat of one building: the median hides it, the zero rule does not
    reps = {"z": [good["z"], dict(good["z"], photo_pose_loc_wrong_face=1, photo_pose_loc_wrong=1)]}
    cr = _d30(bench, base, good, reps)
    assert _outcomes(cr, "photo_pose_loc_wrong_face") == {"dev": "PASS", "test": "FAIL"}
    assert _outcomes(cr, "photo_pose_loc_wrong") == {"dev": "PASS", "test": "FAIL"} and cr["verdict"] == "FAIL"
    # localization CPU over a quarter of the run's on one split
    slow = {**good, "x": dict(good["x"], photo_pose_loc_cpu_s=400.0)}
    assert _outcomes(_d30(bench, base, slow), "photo_pose_loc_cpu_s") == {"dev": "FAIL", "test": "PASS"}
    # a building's wrong share up by more than 0.01 (a wrong localized pose): the pooled share may hide it
    worse = {**good, "w": dict(good["w"], photo_pose_wrong=11, photo_pose_wrong_share=round(11 / 52, 4),
                               photo_pose_loc_wrong=1)}
    cr = _d30(bench, base, worse)
    assert _outcomes(cr, "photo_pose_wrong_share") == {"dev": "PASS", "test": "FAIL"}
    assert _outcomes(cr, "photo_pose_loc_wrong") == {"dev": "PASS", "test": "FAIL"}


def test_d30_zero_rules_leave_out_a_building_without_a_reference_pose(bench):
    """The test split's Sydney Opera House has no reference pose: pose_metrics gives its photo_pose_loc_wrong (every
    reference metric) as None. A plain zero rule would make the test split INSUFFICIENT for any B, which keeps
    localize_photos off whatever the localizer does; d30-v1's zero rules leave such a building out by name, while a
    building with a reference pose and no value, or a split with nothing left, stays INSUFFICIENT."""
    good = {"photo_pose_loc": 1, "photo_pose_loc_gt": 1, "photo_pose_loc_correct": 1, "photo_pose_loc_cpu_s": 100.0,
            "photo_pose_loc_queries": 5}
    base = {s: _d30_metrics() for s in ("x", "y", "z", "w")}
    b_runs = {s: _d30_metrics(**good) for s in base}
    sydney_a = _d30_no_reference(bench, queries=0, cpu_s=0.0)
    sydney_b = _d30_no_reference(bench, queries=0, cpu_s=0.0)
    assert sydney_b["photo_pose_gt"] is None and sydney_b["photo_pose_loc_wrong"] is None
    assert sydney_b["photo_pose_loc_cpu_s"] == 0.0
    reps = {"s": [sydney_b, sydney_b]}
    cr = _d30(bench, {**base, "s": sydney_a}, {**b_runs, "s": sydney_b}, reps)
    assert cr["verdict"] == "PASS", [r for r in cr["rules"] if r["outcome"] not in ("PASS", "REPORT")]
    for name in ("photo_pose_loc_wrong", "photo_pose_loc_wrong_face"):
        row = next(r for r in cr["rules"] if r["metric"] == name and r["split"] == "test")
        assert row["detail"] == "0 on 2 building(s); left out without photo_pose_gt: s"
        dev = next(r for r in cr["rules"] if r["metric"] == name and r["split"] == "dev")
        assert dev["detail"] == "0 on 2 building(s)"
    assert "| photo_pose_loc_wrong | guard | test | zero (buildings without photo_pose_gt left out) |" in "\n".join(
        bench.format_criteria(cr))
    # a wrong localized pose elsewhere still fails, the building left out still named
    bad = {**b_runs, "s": sydney_b, "z": dict(b_runs["z"], photo_pose_loc_wrong=1, photo_pose_loc_correct=0)}
    row = next(r for r in _d30(bench, {**base, "s": sydney_a}, bad, reps)["rules"]
               if r["metric"] == "photo_pose_loc_wrong" and r["split"] == "test")
    assert (row["outcome"], row["detail"]) == ("FAIL", "z 1; left out without photo_pose_gt: s")
    # a building with a reference pose but no value is not left out
    gap = {**b_runs, "s": sydney_b, "w": dict(b_runs["w"], photo_pose_loc_wrong=None)}
    row = next(r for r in _d30(bench, {**base, "s": sydney_a}, gap, reps)["rules"]
               if r["metric"] == "photo_pose_loc_wrong" and r["split"] == "test")
    assert (row["outcome"], row["detail"]) == ("INSUFFICIENT", "no value: w; left out without photo_pose_gt: s")
    # a reference pose in one repeat only: the building is judged (its median has none)
    one = dict(sydney_b, photo_pose_gt=3, photo_pose_loc_wrong=0, photo_pose_loc_wrong_face=0)
    row = next(r for r in _d30(bench, {**base, "s": sydney_a}, {**b_runs, "s": sydney_b},
                               {"s": [sydney_b, one]})["rules"]
               if r["metric"] == "photo_pose_loc_wrong" and r["split"] == "test")
    assert (row["outcome"], row["detail"]) == ("INSUFFICIENT", "no value: s")
    # a split with no building that has a reference pose: nothing judged
    none = {s: _d30_no_reference(bench, queries=0, cpu_s=0.0) for s in ("z", "w")}
    cr = _d30(bench, {**base, **none}, {**b_runs, **none})
    row = next(r for r in cr["rules"] if r["metric"] == "photo_pose_loc_wrong" and r["split"] == "test")
    assert (row["outcome"], row["detail"]) == ("INSUFFICIENT", ("no building with photo_pose_gt; left out without "
                                                                "photo_pose_gt: z, w"))
    assert cr["verdict"] == "INSUFFICIENT"
    # a plain zero rule (d31-v1, review-v1) is unchanged: a building without a value makes it INSUFFICIENT
    plain = bench._absolute("zero", "zero", bench.ev.METRIC_BY_NAME["photo_pose_loc_wrong"],
                            {"z": b_runs["z"], "s": sydney_b}, {}, {})
    assert plain == {"outcome": "INSUFFICIENT", "detail": "no value: s"}


def test_load_criteria_refuses_a_malformed_zero_rule(bench, tmp_path):
    base = json.loads((bench.REPO_ROOT / "eval" / "criteria" / "d30-v1.json").read_text())
    for req in ({"zero": True, "skip_without": "nosuch"}, {"zero": False, "skip_without": "photo_pose_gt"},
                {"zero": True}, {"zero": True, "skip_without": "photo_pose_gt", "min_n": 1}):
        rule = {"metric": "photo_pose_loc_wrong", "role": "guard", "require": req}
        (tmp_path / "c.json").write_text(json.dumps({**base, "rules": [rule]}))
        with pytest.raises(SystemExit, match="zero as a dict"):
            bench.load_criteria(str(tmp_path / "c.json"))


def test_compare_names_the_checks_it_does_not_judge(bench, tmp_path, capsys):
    base = {s: _review_metrics(30.0) for s in ("x", "y")}
    for name in ("f", "d31"):
        (tmp_path / f"{name}.json").write_text(json.dumps(result(base, name=name)))
    args = ["compare", str(tmp_path / "f.json"), str(tmp_path / "d31.json"), "--results-dir", str(tmp_path)]
    bench.main(args + ["--criteria", "d31-v1"])
    out = capsys.readouterr().out
    assert "Not judged here: the verdict above covers the rules only." in out
    assert "acceptance_equals_replay, pose_correct_pooled_rise, count_path_models" in out
    bench.main(args + ["--criteria", "review-v1"])
    assert "Not judged here" not in capsys.readouterr().out


def test_compare_with_criteria_from_the_command_line(bench, tmp_path, capsys):
    base = {s: _review_metrics(30.0) for s in ("x", "y")}
    for name, runs in (("b0", base), ("f", {s: _review_metrics(40.0, jobs_cpu_s=300.0) for s in ("x", "y")}),
                       ("g", {s: _review_metrics(40.0, jobs_cpu_s=300.0, stale_overwrites=1) for s in ("x", "y")})):
        (tmp_path / f"{name}.json").write_text(json.dumps(result(runs, name=name)))
    args = ["compare", str(tmp_path / "b0.json"), "--criteria", "review-v1", "--results-dir", str(tmp_path)]
    assert bench.main(args[:2] + [str(tmp_path / "f.json")] + args[2:]) == 0
    out = capsys.readouterr().out
    assert "Criteria review-v1 (eval/criteria/review-v1.json), splits dev: PASS" in out
    assert bench.main(args[:2] + [str(tmp_path / "g.json")] + args[2:]) == 1
    assert "splits dev: FAIL" in capsys.readouterr().out
    assert bench.main(args[:2] + [str(tmp_path / "f.json"), "--exclude-group", "nosuch"]) == 2


def test_leaderboard_lists_runs_and_defines_the_metrics(bench):
    r = result({"x": metrics(models_published=2, wall_time_s=12.0)}, name="default-1")
    r["aggregate"] = bench.aggregate({"x": r["buildings"]["x"]["metrics"]})
    r["noise"] = bench.noise_from_repeats({"x": [metrics(wall_time_s=12.0), metrics(wall_time_s=14.0)]})
    r["noise"]["per_metric"]["style_observed_area"] = 0.02       # an older result's spread of a removed metric (D29)
    md = bench.build_leaderboard([r])
    assert "| ts | default-1 | default |" in md
    assert "### ts · default · default-1" in md and "| x | dev | ok |" in md
    assert "wall_time_s 2" in md and "style_observed_area" not in md
    assert all(f"| {m.name} |" in md for m in bench.ev.METRICS)


def test_leaderboard_shows_dev_and_test_apart(bench):
    base = {s: _review_metrics(30.0) for s in ("x", "y")}
    r = _split(result({**base, "z": _review_metrics(20.0)}, name="b0", schema=7), ["x", "y"], ["z", "w"])
    r["buildings"]["w"] = {"status": "timeout", "metrics": None, "inputs_sha256": "same"}
    r["aggregate"] = bench.aggregate({s: b["metrics"] for s, b in r["buildings"].items() if b["metrics"]})
    md = bench.build_leaderboard([r])
    assert "| ts | b0 | default | dev | 2026-10-22 00:00 | abc | 2/2 |" in md
    assert "| ts | b0 | default | test | 2026-10-22 00:00 | abc | 1/2 |" in md
    assert "| z | test | ok |" in md and "| w | test | timeout |" in md
    # the review section: per split, its own buildings' aggregate (60 m supported on dev, 20 m on test)
    assert "## 评审指标（指标第 7 版起，按划分汇总）" in md
    assert "| ts | b0 | dev | 2/2 | 60 |" in md and "| ts | b0 | test | 1/2 | 20 |" in md
    # a result without a split is all dev, in one row with its recorded aggregate; schema < 7: no review row
    old = result({"x": metrics(models_published=2)}, name="old")
    old["aggregate"] = bench.aggregate({"x": old["buildings"]["x"]["metrics"]})
    md = bench.build_leaderboard([old])
    assert "| ts | old | default | dev |" in md and "评审指标" not in md


def test_pose_difference_aligns_and_measures(bench):
    rng = np.random.default_rng(3)
    a = {f"p{i}": v for i, v in enumerate(rng.uniform(-60, 60, size=(30, 3)) * [1, 1, 0.2])}
    same = bench.pose_difference(a, a)
    assert same["common"] == 30 and same["shown_median_m"] == same["aligned_p90_m"] == 0.0 and same["scale"] == 1.0
    yaw = math.radians(10.0)
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    b = {k: 1.02 * rot @ v + [3.0, -2.0, 0.5] for k, v in a.items()}
    b["extra"] = np.zeros(3)                                  # shown in B only: not compared
    d = bench.pose_difference(a, b)
    assert (d["photos_a"], d["photos_b"], d["common"]) == (30, 31, 30)
    assert d["shown_median_m"] > 3.0 and d["aligned_median_m"] < 1e-6 and d["scale"] == pytest.approx(1 / 1.02, abs=1e-4)
    few = bench.pose_difference(a, {"p0": a["p0"], "p1": a["p1"]})
    assert few["common"] == 2 and few["aligned_median_m"] is None


def test_posediff_compares_the_final_poses_and_each_runs_repeats(bench, tmp_path, monkeypatch, capsys):
    """posediff reads every run dir's final shown poses (shown_poses, here from a file); B's first run against A's,
    next to A's r1 against r2."""
    def run(root: Path, slug: str, rep: str, poses: dict) -> None:
        d = root / slug / rep
        d.mkdir(parents=True)
        (d / "summary.json").write_text("{}")
        (d / "poses.json").write_text(json.dumps(poses))

    monkeypatch.setattr(bench, "shown_poses", lambda d: {k: np.array(v, dtype=float) for k, v in
                                                         json.loads((d / "poses.json").read_text()).items()})
    pts = {f"p{i}": [float(i), float(i * i % 7), 0.0] for i in range(6)}
    moved = {k: [x + 1.0, y, z] for k, (x, y, z) in pts.items()}
    run(tmp_path / "a", "x", "r1", pts)
    run(tmp_path / "a", "x", "r2", moved)
    run(tmp_path / "b", "x", "", moved)
    a = result({"x": metrics(models_published=1), "y": metrics(models_published=1)}, name="a")
    a.update(artefacts_dir=str(tmp_path / "a"))
    a["provenance"]["repeat"] = 2
    b = result({"x": metrics(models_published=1), "y": metrics(models_published=1)}, name="b")
    b.update(artefacts_dir=str(tmp_path / "b"))
    d = bench.posediff(a, b)
    assert d["y"] == {"status": "artefacts gone"}
    x = d["x"]
    assert x["a_vs_b"]["shown_median_m"] == 1.0 and x["a_vs_b"]["aligned_median_m"] == 0.0
    assert x["a_repeats"] == x["a_vs_b"] and x["b_repeats"] is None
    assert bench.posediff(a, a)["x"]["a_vs_b"]["shown_p90_m"] == 0.0
    for name, r in (("a", a), ("b", b)):
        (tmp_path / f"{name}.json").write_text(json.dumps(r))
    assert bench.main(["posediff", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    out = capsys.readouterr().out
    assert "| x | A r1 vs B r1 | 6 | 1 / 1 | 0 / 0 | 1 |" in out and "| x | A r1 vs A r2 |" in out
    assert "| y | - | artefacts gone |" in out
    (tmp_path / "c.json").write_text(json.dumps(result({"x": metrics()}, sha="t2", name="c")))
    assert bench.main(["posediff", str(tmp_path / "a.json"), str(tmp_path / "c.json")]) == 2


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
    assert bench.ev.METRIC_SCHEMA_VERSION == 8 and all(n.startswith("style_") for n in bench.ev.STYLE_METRICS)
    assert "style_ref_camera_error_worst_m" not in bench.ev.METRIC_BY_NAME           # renamed in 5
    assert "style_registration_pass_rate" in bench.ev.STYLE_METRICS
    # 6: the photo_pose_* group; its rates pool over photos (uploaded, or those with a reference pose)
    # 8 (D30): + the loc source, wrong_share, loc_wrong_face, loc_wrong (the gate's count, in place before any
    # schema-8 result was committed), the localizer's cost and the withheld poses
    assert all(n.startswith("photo_pose_") for n in bench.ev.POSE_METRICS) and len(bench.ev.POSE_METRICS) == 49
    pooled = {m.name: m.pooled for m in bench.ev.METRICS if m.group == "poses" and m.pooled}
    assert pooled["photo_pose_determined_rate"] == ("photo_pose_determined", "photos_uploaded")
    assert pooled["photo_pose_correct_rate"] == ("photo_pose_correct", "photo_pose_gt")
    # per test set (never-uploaded photos fail) and the bound pair over every test-set photo
    assert pooled["photo_pose_determined_rate_testset"] == ("photo_pose_determined", "photo_pose_testset_photos")
    assert pooled["photo_pose_correct_rate_testset"] == ("photo_pose_correct", "photo_pose_gt_testset")
    assert pooled["photo_pose_pass_rate_low_testset"] == ("photo_pose_correct", "photo_pose_testset_photos")
    assert pooled["photo_pose_pass_rate_high_testset"] == ("photo_pose_pass", "photo_pose_testset_photos")
    # per source
    for src in ("sfm", "loc", "exif"):
        assert pooled[f"photo_pose_{src}_uncertainty_coverage"] == (f"photo_pose_{src}_within_uncertainty",
                                                                    f"photo_pose_{src}_gt")
        assert pooled[f"photo_pose_{src}_correct_rate"] == (f"photo_pose_{src}_correct", f"photo_pose_{src}_gt")
    assert pooled["photo_pose_wrong_share"] == ("photo_pose_wrong", "photo_pose_gt_determined")
    assert pooled["photo_pose_loc_cpu_s_per_query"] == ("photo_pose_loc_cpu_s", "photo_pose_loc_queries")
    by_name = bench.ev.METRIC_BY_NAME
    for name in ("photo_pose_loc_wrong_face", "photo_pose_wrong_share", "photo_pose_withheld_right",
                 "photo_pose_loc_cpu_s"):
        assert by_name[name].direction == "lower", name
    assert set(bench.POSE_COLUMNS) <= set(bench.ev.METRIC_BY_NAME)
    assert (bench.ev.POSE_RIGHT_M, bench.ev.POSE_RIGHT_DEG, bench.ev.POSE_STRICT_M, bench.ev.POSE_STRICT_DEG) == (
        10.0, 15.0, 5.0, 10.0)
    # 7: geographic error, coverage in two frames at 1.5 and 5 m, calibration per component, cost, versions, structure
    by, groups = bench.ev.METRIC_BY_NAME, {}
    for m in bench.ev.METRICS:
        groups.setdefault(m.group, []).append(m.name)
    assert all(n.startswith(("geo_", "wrong_face_")) for n in groups["geo"])
    assert all(n.startswith("cov_") for n in groups["coverage"]) and all(n.startswith("calib_") for n in groups["calib"])
    assert {f"cov_{f}_supported_m_{t}" for f in ("local", "geo") for t in ("1p5", "5")} <= set(groups["coverage"])
    for c in bench.ev.CALIB_COMPONENTS:
        assert by[f"calib_{c}_cover2"].pooled == (f"calib_{c}_within2", f"calib_{c}_n")
    assert set(bench.ev.GEO_METRICS) == set(groups["geo"] + groups["coverage"] + groups["calib"])
    assert set(bench.ev.VERSION_METRICS) == {"stale_overwrites", "version_drift_median_m", "version_drift_p90_m"}
    assert set(bench.ev.REQUEST_METRICS) < set(groups["runtime"]) and {"cpu_time_s", "disk_run_mb"} < set(groups["runtime"])
    assert {"struct_incorrect", "struct_qa_failed"} <= set(groups["struct"])
    # the hard guards are counts that must stay 0: lower is better
    for name in ("style_registration_wrong", "style_qa_failed", "struct_incorrect", "struct_qa_failed",
                 "stale_overwrites", "cov_local_wrong_m_1p5", "wrong_face_cameras_shown"):
        assert by[name].direction == "lower", name


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


def test_compact_style_keeps_the_gate_and_drops_the_snap_meta(bench):
    rec = {"status": "ok", "eligible": True, "gate": {"passed": False, "reason": "x"},
           "snap": {"accepted": False, "rotation_deg": 0.0, "meta": {"big": 1}},
           "registration": {"accepted": False}, "walls": {"total": 4, "photographed": 0},
           "qa": {"passed": True}, "glb_bytes": 1, "glb_gz_bytes": 1, "triangles": 2, "time_s": 0.5, "glb": "/x.glb",
           "snap_candidate": {"rotation_deg": 78.0}, "diagnosis": {"cause": "symmetric_alias"}}
    c = bench.compact_style(rec)
    assert "meta" not in c["snap"] and c["walls"] == {"total": 4, "photographed": 0} and c["gate"] == rec["gate"]
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
    a = result({"x": metrics(style_registration_passed=3, style_models_eligible=18, style_walls_total=30,
                             style_incomplete=0, models_published=20),
                "y": metrics(style_registration_passed=1, style_models_eligible=4, style_walls_total=4,
                             style_incomplete=0)}, name="a")
    b = result({"x": metrics(style_registration_passed=0, style_models_eligible=0, style_walls_total=None,
                             style_incomplete=20, models_published=20),
                "y": metrics(style_registration_passed=1, style_models_eligible=4, style_walls_total=4,
                             style_incomplete=0)}, name="b")
    c = bench.compare_results(a, b, [])
    rows = c["per_building"]["x"]["metrics"]
    assert rows["style_walls_total"]["verdict"] == "incomplete" and not rows["style_walls_total"]["significant"]
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
                              results_dir=tmp_path / "res", split=None, regeo=False, structure=None,
                              snap_gate=None)
    assert bench.cmd_rescore(args) == 0
    out = json.loads((tmp_path / "res" / tid / "poses.json").read_text())
    s, t = out["buildings"]["s"], out["buildings"]["t"]
    # the rescored run says exactly what the run itself measured; its other metrics are copied
    assert {k: s["metrics"][k] for k in ev.POSE_METRICS} == {k: summary["metrics"][k] for k in ev.POSE_METRICS}
    again = (*ev.GEO_METRICS, *ev.VERSION_METRICS, *ev.REQUEST_METRICS, "cpu_time_s", "disk_run_mb")
    assert {k: s["metrics"][k] for k in again} == {k: summary["metrics"][k] for k in again}
    assert s["metrics"]["geo_models_checked"] is not None and s["metrics"]["stale_overwrites"] == 0
    assert all("geo" in x for x in s["details"]["models"] if x["published"]) and "coverage" in s["details"]
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
    # the source recorded no fingerprint (no scheme matches it): a marker, so compare says "inputs differ"
    assert s["inputs_unmatched"]["source"] is None and len(s["inputs_unmatched"]["local"]) == 64
    assert s["inputs_sha256"].startswith(bench.UNMATCHED_INPUTS)
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
    src["buildings"]["s"]["inputs_sha256"] = s["inputs_unmatched"]["local"]     # now the source's inputs match
    src_path.write_text(json.dumps(src))
    assert bench.cmd_rescore(args) == 0
    other = json.loads((tmp_path / "res" / tid / "other.json").read_text())
    assert other["provenance"]["recon_output_version"] == bench.RECON_OUTPUT_VERSION + 1    # what it scored
    assert other["buildings"]["s"]["inputs_sha256"] == s["inputs_unmatched"]["local"]
    assert "inputs_unmatched" not in other["buildings"]["s"]
    # --regeo (the fake reconstructor keeps no SfM model: a published model would be left out of the calibration)
    args.name, args.regeo = "regeo", True
    assert bench.cmd_rescore(args) == 0
    regeo = json.loads((tmp_path / "res" / tid / "regeo.json").read_text())
    (rep,) = regeo["buildings"]["s"]["regeo"]
    assert rep["published"] == summary["metrics"]["models_published"] and rep["faithful"] == 0
    assert [x["reason"] for x in rep["left_out"]] == ["no kept sparse/0"] * rep["published"]
    assert regeo["buildings"]["s"]["metrics"]["calib_yaw_n"] == 0 and regeo["buildings"]["s"]["calibration"] == []
    assert regeo["provenance"]["regeo"] and regeo["provenance"]["anchor_calibration"]["id"]
    # --restylize --structure off: the footprint path, recorded in the provenance (struct_* stay empty)
    args.name, args.regeo, args.restylize, args.structure = "restyled", False, True, "off"
    assert bench.cmd_rescore(args) == 0
    restyled = json.loads((tmp_path / "res" / tid / "restyled.json").read_text())
    assert restyled["provenance"]["structure"] == "off" and restyled["provenance"]["restylized"]
    assert all(v is None for k, v in restyled["buildings"]["s"]["metrics"].items() if k.startswith("struct_"))
    assert restyled["provenance"]["snap_gate"] == {k: getattr(st.DEFAULT_PARAMS, k) for k in st.SNAP_GATE_73["on"]}
    # --snap-gate on: the task #73 flags, recorded in the provenance
    args.name, args.snap_gate = "gate", "on"
    assert bench.cmd_rescore(args) == 0
    gated = json.loads((tmp_path / "res" / tid / "gate.json").read_text())
    assert gated["provenance"]["snap_gate"] == st.SNAP_GATE_73["on"]
    args.name, args.restylize, args.structure = "bad", False, None
    with pytest.raises(SystemExit, match="--restylize"):         # --snap-gate only chooses how --restylize works
        bench.cmd_rescore(args)
    args.snap_gate, args.structure = None, "off"
    with pytest.raises(SystemExit, match="--restylize"):         # --structure only chooses how --restylize works
        bench.cmd_rescore(args)


@pytest.mark.slow
def test_rescore_localize_scores_a_recorded_run(bench, tmp_path):
    """D30 rescore --localize on a recorded real-COLMAP run of the synthetic two-sides set (as synthetic-v1): a
    consented photo no published model places is shown from its EXIF without --localize and localized with it,
    within 1 m / 1 deg of the generator's truth; nothing else changes; the run dir is byte-unchanged; loc.json is
    written per repeat and the building's feature store is gone. The map photos' features come from the store when an
    earlier repeat extracted them, and a map photo whose stored keypoints were cut is refused."""
    import argparse
    import hashlib
    import sqlite3
    from contextlib import closing

    from crowd3d import synthetic

    ev = bench.ev
    tid = "two-v1"
    data = tmp_path / "data" / tid / "two"
    manifest = synthetic.generate_dataset(data, sectors=[2, 3], photos_per_sector=6, num_users=3, size=(640, 480),
                                          gps_noise_m=2.0, heading_noise_deg=8.0, include_bad=False, seed=1)
    lat, lon = manifest["landmark"]["lat"], manifest["landmark"]["lon"]
    building = {"slug": "two", "name": "Two", "centre": {"lat": lat, "lon": lon}, "radius_m": 150.0,
                "photos": [{"key": p["file"], "file": p["file"], "author": p["user_id"]} for p in manifest["photos"]]}
    ts_path = tmp_path / "testset.json"
    ts_path.write_text(json.dumps({"schema_version": 1, "testset_id": tid, "integrity_policy": "skip",
                                   "buildings": [building]}))
    _, _, ts_sha = bench.load_testset(ts_path)
    gt = {"kind": "poses", "poses": ev.synthetic_gt_poses(manifest)}
    spec = bench.building_spec(building, tmp_path, bench.verify_photos(building, data), gt, False)
    run_root = tmp_path / "work" / tid / "src"
    summary = ev.run_crowd(ev.building_from_spec(spec), run_root / "two", reconstructor="colmap")
    src = {"result_schema_version": 1, "metric_schema_version": ev.METRIC_SCHEMA_VERSION, "run_name": "src",
           "testset": {"id": tid, "sha256": ts_sha}, "config": {"id": "default", "effective": summary["effective_params"]},
           "provenance": {"repeat": 1, "stylize": False, "recon_output_version": bench.RECON_OUTPUT_VERSION},
           "buildings": {"two": {"status": "ok", "metrics": summary["metrics"]}}, "artefacts_dir": str(run_root)}
    (tmp_path / "src.json").write_text(json.dumps(src))
    cache = tmp_path / "refs"
    cb = ev.building_from_spec({**spec, "reference_cache": bench.reference_cache_spec(cache, ts_sha, "two", {})})
    assert ev.building_reference(cb, {}, None)[0] is not None

    def tree(d: Path) -> dict[str, str]:
        return {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else "dir"
                for f in sorted(d.rglob("*"))}

    before = tree(run_root)
    args = argparse.Namespace(testset=str(ts_path), source=str(tmp_path / "src.json"), name="a", buildings=None,
                              work_root=tmp_path / "work", reference_cache=cache, gt_root=None, restylize=False,
                              allow_recon_diff=False, note="test", overwrite=False, data_root=tmp_path / "data",
                              results_dir=tmp_path / "res", split=None, regeo=False, structure=None,
                              snap_gate=None, localize=False)
    assert bench.cmd_rescore(args) == 0
    args.name, args.localize = "b", True
    assert bench.cmd_rescore(args) == 0
    assert tree(run_root) == before                              # the recorded run is only read
    a, b = (json.loads((tmp_path / "res" / tid / f"{n}.json").read_text()) for n in ("a", "b"))
    ma, mb = a["buildings"]["two"]["metrics"], b["buildings"]["two"]["metrics"]
    (ra,), (rb,) = a["buildings"]["two"]["poses"]["runs"], b["buildings"]["two"]["poses"]["runs"]
    assert "localize" in b["provenance"] and "localize" not in a["provenance"] and "loc" not in ra
    assert b["provenance"]["localize"]["params"]["loc_min_inliers"] == 30
    loc = rb["loc"]
    # the photo uploaded after its face's last job: shown from its EXIF without the localizer, localized with it
    assert ma["photo_pose_loc"] == 0 and ma["photo_pose_exif"] >= 1 and loc["queries"] >= 1
    assert mb["photo_pose_loc"] == loc["shown"] >= 1 and mb["photo_pose_loc_gt"] == loc["gt"] == loc["shown"]
    assert mb["photo_pose_loc_correct"] == loc["correct"] == loc["shown"]
    assert (mb["photo_pose_loc_wrong"], mb["photo_pose_loc_wrong_face"], loc["wrong"]) == (0, 0, 0)
    assert mb["photo_pose_loc_queries"] == loc["queries"] and mb["photo_pose_loc_cpu_s"] > 0
    assert loc["map_images_refused"] == 0 and loc["map_features"]["extracted"] == loc["map_features"]["images"] > 0
    assert sum(loc["replaced"]["exif"].values()) == loc["shown"]
    # the per-map table survives the localizer's counts: each map the poses were shown in, with its own camera error
    assert isinstance(loc["maps"], list) and loc["maps"] and loc["candidate_maps"] >= len(loc["maps"])
    assert sum(row["shown"] for row in loc["maps"]) == loc["shown"]
    assert all(row["wrong"] == 0 and isinstance(row["map_camera_error_median_m"], float) for row in loc["maps"])
    # nothing but the localized photos changed: they left the EXIF source, every other count and metric is A's
    assert mb["photo_pose_exif"] == ma["photo_pose_exif"] - loc["shown"] and mb["photo_pose_sfm"] == ma["photo_pose_sfm"]
    assert {k: v for k, v in ma.items() if not k.startswith("photo_pose_")} == {
        k: v for k, v in mb.items() if not k.startswith("photo_pose_")}
    # loc.json per repeat (exact poses: scratch only): within 1 m of the truth, and its heading within 1 deg of the
    # truth once the map's own heading error is taken off (the generator's compass is 8 deg noisy, so the map's
    # georegistered yaw is a few degrees off; the localizer must add less than 1 deg to it); the store is deleted
    doc = json.loads((tmp_path / "work" / tid / "b" / "two" / "r1" / "loc.json").read_text())
    truth = {p["file"]: p for p in manifest["photos"]}
    key_of = {r["photo_id"]: r["key"] for r in summary["photos"] if r.get("photo_id")}
    _, _, models, _ = ev.load_run_store(run_root / "two" / "data")
    accepted = [r for r in doc["records"] if r["status"] == "accepted"]
    assert len(accepted) == loc["shown"] and doc["maps_key"] and doc["params"]["loc_max_pairs"] == 200
    assert loc["candidate_maps"] == len(doc["maps"]) and {row["model_id"] for row in loc["maps"]} == {
        r["model_id"] for r in accepted} <= set(doc["maps"])
    for r in accepted:
        t = truth[r["key"]]
        cams = next(m for m in models if m.id == r["model_id"]).meta["cameras"]
        offset = float(np.median([ev.angle_diff_deg(c["heading_deg"], truth[key_of[pid]]["true_heading_deg"])
                                  for pid, c in cams.items()]))
        assert math.hypot(r["pose"]["enu"][0] - t["true_enu"][0], r["pose"]["enu"][1] - t["true_enu"][1]) < 1.0
        assert abs(ev.angle_diff_deg(r["pose"]["heading_deg"], t["true_heading_deg"]) - offset) < 1.0
    assert not (tmp_path / "work" / tid / "b" / "two" / "loc-cache").exists()
    # the same repeat localized again from a store an earlier repeat filled: the map features are taken, not
    # extracted, and the poses are the same; a map photo whose stored keypoints were cut is refused, not matched
    store = tmp_path / "store"
    first = ev.rescore_localize(run_root / "two", tmp_path / "loc1" / "r1", store)
    assert first["info"]["map_features"]["extracted"] > 0 and first["info"]["map_images_refused"] == 0
    # the store keeps the map photos' features only (never a query's, never a pair), the repeat's cache is gone
    with closing(sqlite3.connect(store / ev.LOC_STORE_FILE)) as db:
        assert db.execute("SELECT COUNT(*) FROM features").fetchone()[0] == first["info"]["map_features"]["images"]
        assert db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 0
    assert {f.name for f in store.iterdir()} <= {ev.LOC_STORE_FILE, f"{ev.LOC_STORE_FILE}-wal",
                                                 f"{ev.LOC_STORE_FILE}-shm"}
    assert {pid: (r.status, r.pose) for pid, r in first["records"].items()} == {
        r["photo_id"]: (r["status"], r["pose"]) for r in doc["records"]}
    with closing(sqlite3.connect(store / ev.LOC_STORE_FILE)) as db:
        key, rows, cols, kp, drows, dcols, de = db.execute(
            "SELECT feat_key, keypoints_rows, keypoints_cols, keypoints_data, descriptors_rows, descriptors_cols,"
            " descriptors_data FROM features LIMIT 1").fetchone()
        db.execute("UPDATE features SET keypoints_rows = ?, keypoints_data = ?, descriptors_rows = ?,"
                   " descriptors_data = ? WHERE feat_key = ?",
                   (rows - 1, kp[:-cols * 4], drows - 1, de[:-dcols], key))
        db.commit()
    # the photo that row belongs to (the store's key is the processed copy's sha256), and the candidate maps that
    # hold it. The store's rows follow the photo ids, which are random, and the two face maps share one photo: it is
    # refused once in each map that holds it (this test expected 1 and failed whenever the shared photo came first)
    from crowd3d.landmark_map import file_sha256

    data_dir = run_root / "two" / "data"
    (cut,) = [ph.id for ph in ev.load_run_store(data_dir)[1]
              if ph.processed_path and file_sha256(data_dir / ph.processed_path) == key]
    holders = [m for m in doc["maps"]
               if cut in json.loads(next((data_dir / "maps").rglob(f"{m}/map.json")).read_text())["images"].values()]
    assert 1 <= len(holders) <= len(doc["maps"]) == 2
    second = ev.rescore_localize(run_root / "two", tmp_path / "loc2" / "r1", store)
    assert second["info"]["map_features"]["extracted"] == 0
    assert second["info"]["map_features"]["seeded"] == first["info"]["map_features"]["images"]
    assert second["info"]["map_images_refused"] == len(holders) and tree(run_root) == before


def test_a_reference_fix_changes_only_its_own_buildings_cache_key(bench):
    """The reference cache key holds the building's own reference version: the India Gate fix (D31) gives India Gate a
    new key, every other building keeps the key its cache was built with (Sagrada's reference models are deleted:
    a new key would mean fetching them again)."""
    ev = bench.ev
    before = {slug: bench.canonical_sha256({"testset_sha256": "t", "slug": slug, "reference_version": 1})
              for slug in ("sagrada-familia", "india-gate", "florence-baptistery")}
    keys = {slug: bench.reference_cache_spec(Path("/c"), "t", slug, {})["key"] for slug in before}
    assert ev.REFERENCE_VERSION == 1
    assert keys["sagrada-familia"] == before["sagrada-familia"]
    assert keys["florence-baptistery"] == before["florence-baptistery"]
    assert keys["india-gate"] != before["india-gate"] and ev.reference_version("india-gate") == 2
    # the India Gate key of the B0/F reference cache (commons-ten-v1, version 1) is the one it replaces
    ts_sha = "7043a4f3fd6ea5e390953da9e52ae41b4d1551cc54b5dd3667f9ffd2553d8c09"
    assert bench.canonical_sha256({"testset_sha256": ts_sha, "slug": "india-gate", "reference_version": 1}) == (
        "164c651c076a1b9ccb513fb111fc79ceb73370b0092098a07df1f25ff2a75f13")


def test_shown_poses_are_what_the_landmark_page_shows(bench, synthetic_dataset, tmp_path):
    """posediff reads a recorded run's store: one pose per photo the page shows, keyed by the test-set photo key."""
    ev = bench.ev
    ds = synthetic_dataset
    files = [p["file"] for p in ds.manifest["photos"]]
    building = {"slug": "s", "name": "S", "centre": {"lat": ds.lat, "lon": ds.lon}, "radius_m": 150.0,
                "photos": [{"key": f, "file": f, "author": p["user_id"]} for f, p in zip(files, ds.manifest["photos"])]}
    spec = bench.building_spec(building, tmp_path, bench.verify_photos(building, ds.path), None, False)
    summary = ev.run_crowd(ev.building_from_spec(spec), tmp_path / "run" / "s", reconstructor="fake")
    assert summary["metrics"]["models_published"] > 0
    # D31: the recorded details say how each model passed registration (the fake registers every photo)
    details = bench.compact_details(summary)["models"]
    assert details and {x["qa_registration_path"] for x in details} == {"fraction"}
    shown = bench.shown_poses(tmp_path / "run" / "s")
    # the summary's photo rows carry the shown estimate of every photo with an EXIF GPS position
    estimated = {r["key"]: r["est_enu"] for r in summary["photos"] if r.get("est_enu") is not None}
    assert estimated and set(estimated) <= set(shown) <= set(files)
    assert all(np.allclose(shown[k], v, atol=0.01) for k, v in estimated.items())
    run = {"testset": {"id": "tiny-v1", "sha256": "x"}, "run_name": "r", "provenance": {"repeat": 1},
           "artefacts_dir": str(tmp_path / "run"), "buildings": {"s": {"status": "ok", "metrics": summary["metrics"]}}}
    same = bench.posediff(run, run)["s"]
    assert same["status"] == "ok" and same["a_repeats"] is None and same["a_vs_b"]["common"] == len(shown)
    assert same["a_vs_b"]["shown_median_m"] == same["a_vs_b"]["aligned_p90_m"] == 0.0


def test_results_recorded_before_the_recon_output_version_map_by_commit(bench):
    for f in sorted((bench.RESULTS_DIR).glob("**/*.json")):
        prov = json.loads(f.read_text())["provenance"]
        v = bench.source_recon_output_version({"provenance": prov})
        if "recon_output_version" in prov:
            # schema-7 runs and rescores (B0, F, the ablation-v5 cells, D31's baseline A) record the version they scored;
            # D31's own runs recorded 1 but made version 2 (MISRECORDED_RECON_OUTPUT_VERSION); D30's results (rescores
            # of D31's runs and fresh runs at 716de16) recorded 2 and scored version-2 artefacts, and so did D34's
            # replays of D31's runs (rescore --replay, d34-regeo/). D34's fresh runs (d34-dev, d34-test, the d-off dev
            # run and the rescores of all three under d34-fresh/) were made with version 4
            d31_run = f.stem.startswith("d31") and "rescored_from" not in prov
            d34_fresh = f.stem.startswith("d34") and f.parent.name != "d34-regeo"
            d30_or_d34 = f.stem.startswith(("d30", "d34"))
            assert v == (4 if d34_fresh else 2 if d31_run or d30_or_d34 else 1), f
            continue
        # v1-merged and v1-after-fixes ran this reconstruction code (the rescore sources); v0 runs did not
        assert v == (1 if f.stem in ("v1-merged", "v1-after-fixes") else None), f


def test_rescore_refuses_a_result_of_another_test_set(bench, tmp_path):
    import argparse

    src = tmp_path / "src.json"
    src.write_text(json.dumps({"result_schema_version": 1, "testset": {"id": "other", "sha256": "x"}}))
    args = argparse.Namespace(testset="synthetic-v1", source=str(src), name="x", buildings=None,
                              work_root=tmp_path, reference_cache=None, gt_root=None, split=None, regeo=False,
                              note=None, overwrite=False, data_root=tmp_path, results_dir=tmp_path)
    with pytest.raises(SystemExit, match="not synthetic-v1"):
        bench.cmd_rescore(args)


def _calib_result(rows_by_building: dict[str, list[dict]], test: tuple[str, ...] = ()) -> dict:
    return {"result_schema_version": 1, "testset": {"split": {"test": list(test)}},
            "buildings": {b: {"calibration": rows} for b, rows in rows_by_building.items()}}


def _row(key: str, **comps) -> dict:
    return {"key": key, "sigma_from": "anchor", "comps": comps}


def test_calibration_table_counts_each_anchor_once_per_building(bench):
    yaw = lambda e: {"e": e, "core": 1.0, "sigma": 1.0, "ref": 0.0}
    r1 = _calib_result({"a": [_row("k1", yaw=yaw(1.0)), _row("k2", yaw=yaw(-6.0))], "b": [_row("k3", yaw=yaw(0.5))],
                        "h": [_row("k9", yaw=yaw(9.0))]}, test=("h",))
    r2 = _calib_result({"a": [_row("k1", yaw=yaw(1.0))]})               # the same anchor in another repeat
    rows = bench.calibration_rows([r1, r2], "dev")
    assert {s: len(r) for s, r in rows.items()} == {"a": 2, "b": 1}      # held-out h left out, k1 once
    t = bench.calibration_table(rows)["yaw"]
    assert t["buildings"]["a"] == {"n": 2, "error_median": 3.5, "bias": -2.5, "sigma_median": 1.0, "z_median": 3.5,
                                   "cover1": 0.5, "cover2": 0.5, "cover3": 0.5, "beyond3": 0.5}
    assert t["building_weighted"]["cover2"] == 0.75 and t["pooled"]["cover2"] == pytest.approx(0.667, abs=1e-3)
    assert t["building_weighted"]["z_median"] in (1.0, 0.5)           # each building one vote
    refloored = bench.calibration_table(rows, {"yaw_deg": 3.0, "log_scale": 0.0, "xy_m": 0.0, "tilt_deg": 0.0})
    assert refloored["yaw"]["buildings"]["a"]["sigma_median"] == pytest.approx(math.sqrt(10.0), abs=1e-4)
    assert refloored["yaw"]["building_weighted"]["cover2"] == 1.0


def test_fit_floors_reaches_the_building_weighted_target(bench):
    yaw = lambda e: {"e": e, "core": 1.0, "sigma": 1.0, "ref": 0.0}
    xy = lambda e: {"e": e, "core": 1.0, "sigma": 1.0, "ref": 0.5}
    rows = {"a": [_row("k1", yaw=yaw(1.0), xy=xy([0.5, 0.0])), _row("k2", yaw=yaw(-6.0), xy=xy([3.0, 4.0]))],
            "b": [_row("k3", yaw=yaw(0.5), xy=xy([0.0, 1.0]))]}
    fitted = bench.fit_floors(rows)
    # a's 6 deg needs sqrt((6/2)^2 - 1) = 2.83; a's 5 m radius sqrt((5/2.486)^2 - 1 - 0.25) = 1.68 (2-D)
    assert fitted == {"yaw_deg": 2.83, "log_scale": None, "xy_m": 1.68, "tilt_deg": None}
    table = bench.calibration_table(rows, {**fitted, "log_scale": 0.0, "tilt_deg": 0.0})
    assert table["yaw"]["building_weighted"]["cover2"] == table["xy"]["building_weighted"]["cover2"] == 1.0
    just_below = bench.calibration_table(rows, {"yaw_deg": 2.8, "log_scale": 0.0, "xy_m": 1.6, "tilt_deg": 0.0})
    assert just_below["yaw"]["building_weighted"]["cover2"] == just_below["xy"]["building_weighted"]["cover2"] == 0.75


def test_calibration_command_writes_the_tables(bench, tmp_path, capsys):
    import argparse

    yaw = {"e": 4.0, "core": 1.0, "sigma": 3.16, "ref": 0.0}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_calib_result({"a": [_row("k1", yaw=yaw)]})))
    args = argparse.Namespace(results=[path], split="dev", floors="yaw_deg=3,log_scale=0,xy_m=0,tilt_deg=0", fit=True,
                              json=tmp_path / "c.json")
    assert bench.cmd_calibration(args) == 0
    doc = json.loads((tmp_path / "c.json").read_text())
    assert doc["rows"] == {"a": 1} and doc["fitted_floors"]["yaw_deg"] == pytest.approx(math.sqrt(3.0), abs=0.01)
    assert doc["refloored"]["yaw"]["buildings"]["a"]["z_median"] == pytest.approx(4.0 / math.sqrt(10.0), abs=1e-3)
    assert "| yaw | building-weighted |" in capsys.readouterr().out
    args.floors = "yaw_deg=3"
    with pytest.raises(SystemExit, match="--floors"):
        bench.cmd_calibration(args)


# ---------------------------------------------------------------- georeg: errors by model size, what a replay changed


def test_georeg_report_gives_the_checked_models_errors_by_size_and_what_a_replay_changed(bench):
    """benchmark.py georeg (D34): from a result's calibration rows (one per checked model; the same model in another
    repeat counts once) the scale, yaw and pivot errors overall, per building and by registered photos; the split's
    counts of what was published and shown; a replay's own tallies over every repeat."""
    import math

    def row(key, registered, scale, yaw, xy=(3.0, 4.0)):
        return {"key": key, "registered": registered, "comps": {"scale": {"e": -math.log(scale)}, "yaw": {"e": -yaw},
                                                               "xy": {"e": list(xy)}}}

    def metrics(**over):
        m = {name: None for name in bench.ev.METRIC_BY_NAME}
        m.update({"models_published": 4, "photo_pose_determined": 20, "photo_pose_gt_determined": 10,
                  "photo_pose_correct": 6, "photo_pose_wrong": 4, "photo_pose_within_uncertainty": 7,
                  "calib_yaw_n": 2, "calib_yaw_within2": 1, **over})
        return m

    replay = {"levels": {"full": 3, "prior_only": 1}, "published": 4, "models": 4, "replayed": 3,
              "left_as_recorded": [{"model_id": "x", "reason": "no kept sparse/0"}], "message_differs": 1,
              "transform_changed": 2, "level_changed": [{"model_id": "y"}], "published_changed": []}
    result = {"run_name": "arm", "provenance": {"replay": {"id": "d34-regeo-a"}}, "testset": {"split": None},
              "buildings": {
                  "one": {"metrics": metrics(), "replay": [replay, replay],
                          "calibration": [row("k1", 5, 2.0, 20.0), row("k1", 5, 2.0, 20.0),      # r1 and r2: once
                                          row("k2", 8, 1.0 / 1.4, -3.0), row("k3", 25, 1.1, 4.0, (0.0, 1.0))]},
                  "two": {"metrics": metrics(models_published=1), "calibration": [row("k4", 30, 0.9, 16.0)]},
                  "none": {"metrics": None}}}
    rep = bench.georeg_report(result)
    e = rep["overall"]["errors"]
    assert (e["models"], e["scale_off"], e["scale_n"], e["yaw_off"], e["yaw_n"]) == (4, 2, 4, 2, 4)
    logs = [math.log(2.0), math.log(1.4), math.log(1.1), -math.log(0.9)]
    assert e["scale_rms_ln"] == pytest.approx(math.sqrt(sum(v * v for v in logs) / 4), abs=1e-3)
    assert e["yaw_rms_deg"] == pytest.approx(math.sqrt((400 + 9 + 16 + 256) / 4), abs=0.01) and e["yaw_median_deg"] == 10.0
    assert e["xy_median_m"] == 5.0
    bands = rep["overall"]["bands"]
    assert [bands[k]["models"] for k in ("3-6", "7-10", "11-20", "21+")] == [1, 1, 0, 2]
    assert bands["3-6"]["scale_off"] == 1 and bands["21+"]["scale_off"] == 0 and bands["11-20"]["scale_rms_ln"] is None
    c = rep["overall"]["counts"]
    assert (c["models_published"], c["photo_pose_determined"], c["shown_beyond_2sigma"]) == (5, 40, 6)
    assert c["calib_yaw_cover2"] == 0.5
    assert rep["overall"]["replay"] == {"runs": 2, "levels": {"full": 6, "prior_only": 2}, "published": 8, "models": 8,
                                        "replayed": 6, "left_as_recorded": 2, "message_differs": 2,
                                        "transform_changed": 4, "level_changed": 2, "published_changed": 0}
    assert set(rep["buildings"]) == {"one", "two"} and rep["buildings"]["two"]["replay"] is None
    assert rep["buildings"]["two"]["errors"]["models"] == 1 and rep["buildings"]["one"]["errors"]["models"] == 3
    text = bench.format_georeg([rep])
    assert "| arm | 5 | 4 |" in text and "### What each replay changed" in text and "6 / 2 / 0" in text
    # a row recorded before D34 names no registered count: in the totals, in no band
    old = dict(result, buildings={"one": {"metrics": metrics(), "calibration": [{k: v for k, v in row("k", 9, 1.5, 1.0).items()
                                                                               if k != "registered"}]}})
    rep = bench.georeg_report(old)
    assert rep["overall"]["errors"]["models"] == 1 and all(b["models"] == 0 for b in rep["overall"]["bands"].values())
