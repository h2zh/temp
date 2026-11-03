"""The GT-free quality gate (qa.assess) on reconstruction results shaped like the real-photo failures."""

from __future__ import annotations

import math

import pytest

from crowd3d.config import Settings
from crowd3d.coverage import sector_for_bearing
from crowd3d.geo import LocalENU, heading_from_en, normalize_deg
from crowd3d.qa import ANCHOR_CHECKS, FLAG_BEYOND_RADIUS, FLAG_FACING_AWAY, assess
from crowd3d.reconstruct import ReconstructionResult
from crowd3d.records import JOB_FACE, JOB_FULL, PHOTO_ACCEPTED, Landmark, PhotoRecord
from test_anchor import make_anchor

LAT, LON = 43.773056, 11.254722
LM = Landmark(id="lm", name="Baptistry", lat=LAT, lon=LON, radius_m=150.0, created_at=0.0)
S = Settings(data_dir="unused")


def photos_and_result(bearings, *, compass=True, distance=60.0):
    """Accepted photos at `distance` m on the given bearings, and a model that puts every camera
    exactly where its GPS says, looking at the centre: a model with nothing wrong with it."""
    enu = LocalENU(LAT, LON)
    photos, cams = [], {}
    for i, b in enumerate(bearings):
        e, n = distance * math.sin(math.radians(b)), distance * math.cos(math.radians(b))
        lat, lon, _ = enu.to_geodetic(e, n)
        heading = normalize_deg(b + 180.0)
        photos.append(PhotoRecord(id=f"p{i}", landmark_id="lm", user_id="u", status=PHOTO_ACCEPTED,
                                  uploaded_at=float(i), lat=lat, lon=lon, heading_deg=heading if compass else None,
                                  bearing_deg=b, distance_m=distance, sector=sector_for_bearing(b, 8)))
        cams[f"p{i}"] = {"enu": [e, n, 1.6], "heading_deg": heading, "gps_residual_m": 0.0,
                         "compass_offset_deg": 0.0 if compass else None, "focal_ratio": 1.0, "flags": []}
    result = ReconstructionResult(ok=True, message="", num_input=len(photos), num_registered=len(photos),
                                  num_points=5000, mean_reprojection_error_px=0.8, georegistered=True,
                                  cameras=cams, points_median_en=[20.0, 5.0])
    return photos, result


def test_a_sound_facade_model_passes():
    photos, result = photos_and_result(range(70, 112, 2))
    report = assess(result, JOB_FACE, photos, LM, S)
    assert report.passed, report.reasons
    assert not any(c["flags"] for c in result.cameras.values())


def test_twin_facade_cameras_fail_the_model():
    # Baptistry E rebuild: 6 of 18 cameras placed in front of the octagon face 90 deg away.
    photos, result = photos_and_result(range(70, 106, 2))
    for i in range(0, 18, 3):
        result.cameras[f"p{i}"].update(gps_residual_m=60.0, compass_offset_deg=-90.0, flags=["gps_and_compass"])
    report = assess(result, JOB_FACE, photos, LM, S)
    assert not report.passed
    assert any("look-alike facades" in r for r in report.reasons)


def test_a_model_turned_or_dislocated_without_compass_fails():
    # Aachen, real photos without compass: whole models turned 80-170 deg, point clouds 600 m away.
    photos, result = photos_and_result(range(250, 290, 4), compass=False)
    for cam in result.cameras.values():
        cam["heading_deg"] = normalize_deg(cam["heading_deg"] + 170.0)
    result.points_median_en = [480.0, -420.0]
    report = assess(result, JOB_FACE, photos, LM, S)
    assert not report.passed
    assert all(FLAG_FACING_AWAY in c["flags"] for c in result.cameras.values())
    assert any("implausible" in r for r in report.reasons) and any("from the landmark centre" in r
                                                                    for r in report.reasons)


def test_cameras_outside_the_radius_are_flagged_and_too_many_fail():
    photos, result = photos_and_result(range(70, 112, 2))
    far = ["p0", "p1"]
    for pid in far:
        result.cameras[pid]["enu"] = [900.0, 0.0, 1.6]        # one shown 943 km away in the real run
    report = assess(result, JOB_FACE, photos, LM, S)
    assert report.passed                                       # 2 of 21: withheld, model still fine
    assert [pid for pid, c in result.cameras.items() if FLAG_BEYOND_RADIUS in c["flags"]] == far
    for cam in list(result.cameras.values())[:6]:
        cam["enu"] = [900.0, 0.0, 1.6]
    assert not assess(result, JOB_FACE, photos, LM, S).passed


@pytest.mark.parametrize("change,reason", [
    ({"num_registered": 4}, "photos registered"),
    ({"num_points": 95}, "points"),
    ({"mean_reprojection_error_px": 4.0}, "reprojection"),
])
def test_weak_models_fail(change, reason):
    photos, result = photos_and_result(range(70, 112, 2))
    for k, v in change.items():
        setattr(result, k, v)
    report = assess(result, JOB_FACE, photos, LM, S)
    assert not report.passed and any(reason in r for r in report.reasons)


def test_compass_disagreement_fails():
    photos, result = photos_and_result(range(70, 112, 2))
    for cam in result.cameras.values():
        cam["compass_offset_deg"] = 40.0
    report = assess(result, JOB_FACE, photos, LM, S)
    assert not report.passed and any("off the compass" in r for r in report.reasons)


def test_full_model_must_cover_the_required_sides():
    # Aachen augmented: 164 photos -> the largest of 9 fragments, 59 photos of the west front only.
    bearings = [b for centre in (90, 135, 180, 225, 270) for b in range(centre - 15, centre + 15, 3)]
    photos, result = photos_and_result(bearings)
    assert assess(result, JOB_FULL, photos, LM, S).passed
    keep = {p.id for p in photos if p.sector == 6}
    result.cameras = {pid: c for pid, c in result.cameras.items() if pid in keep}
    result.num_registered = len(result.cameras)
    report = assess(result, JOB_FULL, photos, LM, S)
    assert not report.passed
    assert report.checks["covered_sectors"] == 1 and report.checks["required_sectors"] == 4
    result.num_registered = len(photos)       # even if most photos had registered
    assert any("face-ready sides" in r for r in assess(result, JOB_FULL, photos, LM, S).reasons)


def test_heading_towards_the_landmark_is_not_flagged_when_a_compass_vouches():
    # Facing is judged against the compass when there is one, against the centre only without.
    photos, result = photos_and_result(range(70, 112, 2))
    for cam in result.cameras.values():
        cam["heading_deg"] = heading_from_en(1.0, 0.0)
    assert assess(result, JOB_FACE, photos, LM, S).passed


def test_anchor_checks_are_recorded_but_never_decide():
    """qa records the GeoAnchor's level and sigmas (informational): the same models pass and fail, with the same
    reasons, whatever the anchor says; no anchor (or a malformed one) records None."""
    photos, result = photos_and_result(range(70, 112, 2))
    bare = assess(result, JOB_FACE, photos, LM, S)
    assert bare.passed and all(bare.checks[k] is None for k in ANCHOR_CHECKS)
    wide = make_anchor(level="prior_only", sigma={"yaw_deg": 180.0, "log_scale": 0.7, "east_m": 900.0,
                                                  "north_m": 900.0, "up_m": 1.5, "tilt_deg": 30.0})
    result.anchor = wide.to_meta()
    report = assess(result, JOB_FACE, photos, LM, S)
    assert report.passed and report.reasons == []
    h, _ = wide.position_sigma_m([20.0, 5.0])          # at the points' median, not at the pivot
    assert report.checks == {**bare.checks, "anchor_level": "prior_only", "anchor_yaw_sigma_deg": 180.0,
                             "anchor_log_scale_sigma": 0.7, "anchor_position_sigma_m": round(h, 2)}
    result.anchor = {"schema": "crowd3d.anchor/1"}     # malformed: still nothing but None
    assert all(assess(result, JOB_FACE, photos, LM, S).checks[k] is None for k in ANCHOR_CHECKS)

    photos, result = photos_and_result(range(70, 112, 2))
    result.num_points = 10
    weak = assess(result, JOB_FACE, photos, LM, S)
    result.anchor = make_anchor().to_meta()             # a tight anchor does not rescue a weak model
    tight = assess(result, JOB_FACE, photos, LM, S)
    assert (tight.passed, tight.reasons) == (weak.passed, weak.reasons) == (False, weak.reasons)
    assert tight.checks["anchor_level"] == "full" and tight.checks["anchor_yaw_sigma_deg"] == pytest.approx(
        make_anchor().sigma["yaw_deg"], abs=1e-3)


def partial_facade(n_input, n_reg, *, georegistered=True, kind=JOB_FACE):
    """A sound model of which only the first n_reg of n_input photos registered (Lincoln E in F: 10 of 30)."""
    photos, result = photos_and_result([70 + 40 * i / n_input for i in range(n_input)])
    keep = {p.id for p in photos[:n_reg]}
    result.cameras = {pid: c for pid, c in result.cameras.items() if pid in keep}
    result.num_registered, result.georegistered = n_reg, georegistered
    return assess(result, kind, photos, LM, S)


def test_a_facade_passes_on_its_registered_count_with_a_gps_fit():
    """D31: below the fraction, a facade model passes registration with >= qa_face_min_registered registered photos
    when the georegistration fitted GPS; the path is recorded."""
    report = partial_facade(30, 10)
    assert report.passed, report.reasons
    assert report.checks["registration_path"] == "count" and report.checks["registered_fraction"] == 0.333
    assert partial_facade(30, 20).checks["registration_path"] == "fraction"       # today's path comes first
    edge = S.qa_face_min_registered
    assert partial_facade(30, edge).passed and partial_facade(30, edge).checks["registration_path"] == "count"
    below = partial_facade(30, edge - 1)
    assert not below.passed and below.checks["registration_path"] is None


@pytest.mark.parametrize("n_reg,georegistered", [(10, False), (5, True)])
def test_the_count_path_needs_the_gps_fit_and_the_count(n_reg, georegistered):
    report = partial_facade(30, n_reg, georegistered=georegistered)
    assert not report.passed and report.checks["registration_path"] is None
    assert report.reasons == [f"only {n_reg} of 30 photos registered (need 60%, or 6 with a GPS fit)"]
    assert "photos registered" in report.reasons[0]   # eval_real_landmark.QA_REASON_CODES: registered_fraction


def test_the_count_path_keeps_every_other_check():
    photos, result = photos_and_result([70 + 40 * i / 30 for i in range(30)])
    result.cameras = {pid: c for pid, c in result.cameras.items() if int(pid[1:]) < 10}
    result.num_registered, result.num_points = 10, S.qa_min_points - 1
    report = assess(result, JOB_FACE, photos, LM, S)
    assert not report.passed and report.checks["registration_path"] == "count"
    assert [r for r in report.reasons if "registered" in r] == [] and any("points" in r for r in report.reasons)


def test_the_full_model_has_no_count_path():
    """A full model claims every side: a registered count does not show that; its fraction rule is unchanged."""
    bearings = [b for centre in (90, 135, 180, 225, 270) for b in range(centre - 15, centre + 15, 3)]
    photos, result = photos_and_result(bearings * 2)                       # 100 photos
    result.num_registered = 69                                               # 0.69 < 0.7, 69 >> 6, GPS fit
    report = assess(result, JOB_FULL, photos, LM, S)
    assert not report.passed and report.checks["registration_path"] is None
    assert any(r.startswith("only 69 of 100 photos registered (need 70%)") for r in report.reasons)
    result.num_registered = 70
    assert assess(result, JOB_FULL, photos, LM, S).checks["registration_path"] == "fraction"


def test_the_registered_count_is_a_setting():
    import dataclasses

    strict = dataclasses.replace(S, qa_face_min_registered=11)
    photos, result = photos_and_result([70 + 40 * i / 30 for i in range(30)])
    result.cameras = {pid: c for pid, c in result.cameras.items() if int(pid[1:]) < 10}
    result.num_registered = 10
    report = assess(result, JOB_FACE, photos, LM, strict)
    assert not report.passed and report.reasons[0].endswith("(need 60%, or 11 with a GPS fit)")
