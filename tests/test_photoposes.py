"""crowd3d.photoposes: which pose each photo gets and its uncertainty (the benchmark's photo_pose_* metrics)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from crowd3d import photoposes as pp
from crowd3d.api import camera_estimates
from crowd3d.geo import LocalENU
from crowd3d.records import JOB_FACE, JOB_FULL, PHOTO_ACCEPTED, PHOTO_REJECTED, ModelRecord, PhotoRecord

LAT, LON = 43.773, 11.255
ENU = LocalENU(LAT, LON)


def photo(pid, e=None, n=None, heading=None, *, status=PHOTO_ACCEPTED, acc=None):
    lat = lon = None
    if e is not None:
        lat, lon, _ = ENU.to_geodetic(e, n, 0.0)
    return PhotoRecord(id=pid, landmark_id="lm", user_id="u1", status=status, uploaded_at=1.0,
                       reasons=[] if status == PHOTO_ACCEPTED else ["too_blurry"], lat=lat, lon=lon,
                       gps_accuracy_m=acc, heading_deg=heading)


def cam(e, n, heading, flags=(), **kw):
    return {"enu": [e, n, 1.6], "heading_deg": heading, "flags": list(flags), "focal_ratio": 0.8, **kw}


def model(mid, cams, *, kind=JOB_FACE, created=1.0, published=True, **meta):
    base = {"published": published, "cameras": cams, "georegistered": True, "alignment_rmse_m": 4.0,
            "yaw_sigma_deg": 3.0, "message": "georegistered from 9/9 GPS fixes"}
    base.update(meta)
    return ModelRecord(id=mid, landmark_id="lm", job_id="j" + mid, kind=kind, sector=0 if kind == JOB_FACE else None,
                       ply_path=f"models/lm/{mid}.ply", num_photos=len(cams), meta=base, created_at=created)


# ---------------------------------------------------------------- which source wins


def test_published_cameras_pick_what_the_landmark_page_shows():
    models = [
        model("old", {"a": cam(1, 1, 10), "b": cam(2, 2, 20), "c": cam(3, 3, 30)}, created=1.0),
        model("new", {"a": cam(5, 5, 50), "b": cam(6, 6, 60, flags=["gps_and_compass"])}, created=2.0),
        model("full", {"c": cam(7, 7, 70)}, kind=JOB_FULL, created=0.5),
        model("rejected", {"a": cam(9, 9, 90), "d": cam(9, 9, 90)}, created=3.0, published=False),
    ]
    picked = pp.published_cameras(models)
    shown = camera_estimates(models)
    assert {k: v[1] for k, v in picked.items()} == shown
    assert picked["a"][0].id == "new"          # the newest model
    assert picked["b"][0].id == "old"          # a flagged camera is no estimate: the older entry stays
    assert picked["c"][0].id == "full"         # a full model beats facade models
    assert "d" not in picked                   # QA-rejected models show nothing


def test_photo_poses_source_hierarchy():
    photos = [
        photo("sfm", 30, 0, 270.0),
        photo("exif", -30, 0, 90.0),
        photo("gps", 20, 20),
        photo("rej", 10, 10, 0.0, status=PHOTO_REJECTED),
        photo("nogps"),
    ]
    models = [model("m1", {"sfm": cam(28.0, 1.0, 268.0)})]
    poses = {p.photo_id: p for p in pp.photo_poses(photos, models, LAT, LON)}
    assert poses["sfm"].src == "sfm" and poses["sfm"].model_id == "m1"
    assert poses["sfm"].p == [28.0, 1.0, 1.6] and poses["sfm"].yaw == 268.0
    assert poses["exif"].src == "exif" and poses["exif"].yaw == 90.0
    assert math.hypot(poses["exif"].p[0] + 30, poses["exif"].p[1]) < 0.01
    # GPS without a heading: a position only, NOT determined (the camera is never assumed to face the building)
    assert poses["gps"].src == "gps" and poses["gps"].yaw is None and not poses["gps"].determined
    assert poses["gps"].reason == pp.R_GPS_WITHOUT_HEADING
    assert poses["rej"].src is None and poses["rej"].reason == pp.R_REJECTED and poses["rej"].reasons == ["too_blurry"]
    assert poses["nogps"].src is None and poses["nogps"].reason == pp.R_NO_GPS
    assert [p.determined for p in poses.values()] == [True, True, False, False, False]


def test_sfm_camera_moves_with_an_accepted_snap_only():
    photos = [photo("a", 10, 0, 0.0)]
    models = [model("m1", {"a": cam(10.0, 0.0, 0.0)})]
    snap = {"rotation_deg": 90.0, "shift_m": [1.0, 2.0], "scale": 1.0, "accepted": True}
    (pose,) = pp.photo_poses(photos, models, LAT, LON, snaps={"m1": snap})
    assert np.allclose(pose.p, [1.0, 12.0, 1.6], atol=1e-9)
    assert pose.yaw == pytest.approx(270.0)     # north turned 90 deg counter-clockwise looks west
    assert pose.snapped and pose.sp <= 2 * pp.SNAPPED_SIGMA_CAP and pose.sy <= 2 * pp.SNAPPED_SIGMA_CAP
    (pose,) = pp.photo_poses(photos, models, LAT, LON, snaps={"m1": {**snap, "accepted": False}})
    assert pose.p == [10.0, 0.0, 1.6] and pose.yaw == 0.0 and not pose.snapped


def test_apply_snap_matches_stylize():
    st = pytest.importorskip("crowd3d.stylize")
    meta = {"rotation_deg": -7.5, "shift_m": [3.0, -1.5], "scale": 1.04, "accepted": True}
    snap = st.FootprintSnap.from_meta(meta)
    pts = np.array([[10.0, -4.0, 1.6], [-30.0, 22.0, 15.0]])
    want = st.apply_snap_points(snap, pts)
    for p, w in zip(pts, want):
        got, _ = pp.apply_snap(meta, list(p), 0.0)
        assert np.allclose(got, w, atol=1e-9)


# ---------------------------------------------------------------- uncertainty


def test_sfm_sigma_calibrated_rule():
    # floors: the 15 m display circle, 10 deg
    assert pp.sfm_sigma({"alignment_rmse_m": 4.0, "yaw_sigma_deg": 3.0}, [20, 0, 0]) == (7.5, 10.0)
    # a large RMSE widens the circle; a missing yaw sigma means the compass sigma
    pos, head = pp.sfm_sigma({"alignment_rmse_m": 40.0}, [20, 0, 0])
    assert pos == 20.0 and head == pytest.approx(math.hypot(15.0, 1.0))
    # a GPS / compass yaw conflict: d sin(conflict) and conflict / 2
    meta = {"alignment_rmse_m": 4.0, "yaw_sigma_deg": 3.0,
            "message": "georegistered from 8/9 GPS fixes (spread 40.0 m, scale +-5%, yaw +-4 deg from gps+compass; "
                       "GPS yaw - compass yaw = -36 deg)"}
    assert pp.yaw_conflict_deg(meta) == -36.0
    pos, head = pp.sfm_sigma(meta, [60.0, 0.0, 0.0])
    assert pos == pytest.approx(60 * math.sin(math.radians(36)) / 2) and head == 18.0
    # only a compass conflict counts (a landmark-direction pseudo-compass is not a measurement)
    assert pp.yaw_conflict_deg({"message": "...; GPS yaw - landmark-direction yaw = 50 deg)"}) is None
    assert pp.yaw_conflict_deg({"message": "not georegistered: GPS yaw is 63 deg from the compass yaw; compass yaw,"
                                           " approximate scale"}) == 63.0


def test_prior_only_model_is_as_rough_as_phone_gps_and_marked_approx():
    # mdx-two-v1 Aachen's only published model: not georegistered (3 of 4 GPS fixes over 6.9 m), landmark-direction
    # yaw, guessed scale; its 4 cameras sat 23.6-24.8 m from the reference inside a 15 m circle before this rule
    msg = ("registered 4/6 photos, 1065 points; not georegistered: 3/4 consistent GPS fixes spread over 6.9 m cannot"
           " fix the scale; landmark-direction yaw, approximate scale")
    cams = {"a": cam(30.0, 40.0, 200.0), "b": cam(33.0, 41.0, 205.0), "c": cam(29.0, 44.0, 199.0),
            "far": cam(80.0, 40.0, 250.0)}
    prior = model("m1", cams, georegistered=False, alignment_rmse_m=3.0, yaw_sigma_deg=12.9, message=msg)
    photos = [photo(k, c["enu"][0], c["enu"][1], None) for k, c in cams.items()]
    poses = {p.photo_id: p for p in pp.photo_poses(photos, [prior], LAT, LON)}
    assert pp.prior_only(prior.meta) and pp.median_camera_en(prior.meta) == (31.5, 40.5)
    a = poses["a"]
    assert a.src == "sfm" and a.determined and a.approx
    d_med = math.hypot(30.0 - 31.5, 40.0 - 40.5)
    assert a.sp == pytest.approx(math.hypot(pp.PRIOR_ONLY_POS_FLOOR_M, pp.PRIOR_ONLY_SCALE_ERROR * d_med))
    assert a.sp >= 30.0 > 24.8                     # the 24.8 m Aachen error now lies inside
    # the scale term grows with the distance from the model's median camera (the guessed scale stretches it)
    assert poses["far"].sp == pytest.approx(math.hypot(30.0, 0.6 * math.hypot(48.5, 0.5)))
    # the same cameras in a GPS-fitted model: the calibrated 15 m circle, not approximate (mutation guard)
    fitted = model("m1", cams, alignment_rmse_m=3.0, yaw_sigma_deg=12.9)
    b = {p.photo_id: p for p in pp.photo_poses(photos, [fitted], LAT, LON)}["a"]
    assert b.sp == pytest.approx(15.0) and not b.approx
    # snapped (never happens to a prior-only model, the gate needs georegistration): the snap's cap wins, not approx
    snapped = pp.photo_poses(photos[:1], [prior], LAT, LON,
                             snaps={"m1": {"rotation_deg": 0.0, "shift_m": [0, 0], "scale": 1.0, "accepted": True}})[0]
    assert snapped.sp <= 2 * pp.SNAPPED_SIGMA_CAP and not snapped.approx


def test_exif_sigma_by_heading_reference():
    assert pp.exif_sigma(None, "T") == (20.0, 15.0)
    assert pp.exif_sigma(None, "M") == (30.0, 30.0)
    assert pp.exif_sigma(None, None) == (30.0, 30.0)     # unknown reference: as magnetic
    assert pp.exif_sigma(25.0, "T") == (25.0, 15.0)      # the phone's own, larger error
    assert pp.exif_sigma(10.0, "T") == (20.0, 15.0)      # ... but never below the measured spread
    p = photo("p", 0, 30, 180.0)
    (pose,) = pp.photo_poses([p], [], LAT, LON, heading_refs={"p": "M"})
    assert (pose.sp, pose.sy) == (60.0, 60.0)          # 2 sigma
    (pose,) = pp.photo_poses([p], [], LAT, LON, heading_refs={"p": "T"})
    assert (pose.sp, pose.sy) == (40.0, 30.0)
    (pose,) = pp.photo_poses([p], [], LAT, LON)        # no reference known: as magnetic
    assert (pose.sp, pose.sy) == (60.0, 60.0)


def test_exif_circle_is_wide_enough_for_the_measured_exif_errors():
    # the drawn EXIF circle (2 x 12 m = 24 m) held only 37 of the 51 EXIF-drawn reference photos' positions (p80
    # 30 m, p85 38-40 m for 'T' tags); raw EXIF of all 180 heading-tagged reference photos: 'T' p90 23 m, 'M' p90
    # 60 m (docs/BENCHMARK.md 7.7). The circle must reach the drawn photos' p85 and the raw p90.
    sp_t = pp.COVERAGE_SIGMAS * pp.exif_sigma(None, "T")[0]
    sp_m = pp.COVERAGE_SIGMAS * pp.exif_sigma(None, "M")[0]
    assert sp_t >= 38.0 and sp_t >= 23.0
    assert sp_m >= 60.0
    # a position without any heading (src gps) is as wide as an untagged one
    (pose,) = pp.photo_poses([photo("p", 0, 30, None)], [], LAT, LON)
    assert pose.src == "gps" and pose.sp == sp_m

