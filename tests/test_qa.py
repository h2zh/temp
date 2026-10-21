"""The GT-free quality gate (qa.assess) on reconstruction results shaped like the real-photo failures."""

from __future__ import annotations

import math

import pytest

from crowd3d.config import Settings
from crowd3d.coverage import sector_for_bearing
from crowd3d.geo import LocalENU, heading_from_en, normalize_deg
from crowd3d.qa import FLAG_BEYOND_RADIUS, FLAG_FACING_AWAY, assess
from crowd3d.reconstruct import ReconstructionResult
from crowd3d.records import JOB_FACE, JOB_FULL, PHOTO_ACCEPTED, Landmark, PhotoRecord

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
