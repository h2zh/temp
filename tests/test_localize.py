"""crowd3d.localize, the pure part (D30): which photos and maps, pairs, the gates, the choice, the frame, the records."""

from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from crowd3d import localize as loc
from crowd3d import params
from crowd3d import photoposes as pp
from crowd3d.anchor import GeoAnchor, quat_matrix, quat_xyzw
from crowd3d.geo import LocalENU
from crowd3d.params import DEFAULTS, with_overrides
from crowd3d.records import (
    JOB_FACE,
    JOB_FULL,
    LOC_ACCEPTED,
    LOC_REJECTED,
    LOC_SKIPPED,
    LOCATION_CONSENT_V1,
    PHOTO_ACCEPTED,
    PHOTO_REJECTED,
    LocRecord,
    ModelRecord,
    PhotoRecord,
)

LAT, LON = 43.773, 11.255
ENU = LocalENU(LAT, LON)
FV = "feature-version-1"
PRIOR_MSG = ("registered 4/6 photos, 1065 points; not georegistered: 3/4 consistent GPS fixes spread over 6.9 m cannot"
             " fix the scale; landmark-direction yaw, approximate scale")


def photo(pid, e=30.0, n=0.0, heading=None, *, status=PHOTO_ACCEPTED, consent=LOCATION_CONSENT_V1, f35=28.0,
          proc=True, acc=None):
    lat, lon, _ = ENU.to_geodetic(e, n, 0.0)
    return PhotoRecord(id=pid, landmark_id="lm", user_id="u1", status=status, uploaded_at=1.0, lat=lat, lon=lon,
                       gps_accuracy_m=acc, heading_deg=heading, location_consent=consent, focal_35mm=f35,
                       processed_path=f"photos/lm/{pid}.proc.jpg" if proc else None)


def cam_from_enu(heading_deg, pitch_deg, roll_deg):
    """cam_from_enu rows (right, down, forward) of a camera looking along heading / pitch, rolled right side up."""
    h, p, r = (math.radians(v) for v in (heading_deg, pitch_deg, roll_deg))
    f = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    lr = np.cross(f, [0.0, 0.0, 1.0])
    lr /= np.linalg.norm(lr)
    lu = np.cross(lr, f)
    right = math.cos(r) * lr + math.sin(r) * lu
    return np.array([right, np.cross(f, right), f])


def make_anchor(*, yaw=37.0, scale=1.7, t=(12.0, -4.0, 2.5), level="full"):
    a = math.radians(yaw)
    rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return GeoAnchor("job", level, scale, quat_xyzw(rot), list(t), [0.0, 0.0, 1.0], yaw, [0.0, 0.0, 0.0],
                     dict.fromkeys(("yaw_deg", "log_scale", "east_m", "north_m", "up_m", "tilt_deg"), 1.0),
                     dict.fromkeys(("yaw", "scale", "xy", "up", "gravity"), "gps"))


def local_pose(anchor, centre_enu, heading, pitch=0.0, roll=0.0):
    """(cam_from_local rotation, local centre) of a camera at centre_enu looking along heading / pitch / roll."""
    rot = anchor.rotation()
    c_local = rot.T @ (np.asarray(centre_enu, float) - np.asarray(anchor.translation)) / anchor.scale
    return cam_from_enu(heading, pitch, roll) @ rot, c_local


def stored_cam(anchor, centre_enu, heading, pitch=0.0, roll=0.0, flags=()):
    """A model JSON camera entry as reconstruct.check_cameras writes it (enu, heading, local pose)."""
    r, c = local_pose(anchor, centre_enu, heading, pitch, roll)
    enu, _, h = anchor.camera_enu(quat_xyzw(r), c)
    return {"enu": [round(float(x), 3) for x in enu], "heading_deg": round(h, 2), "flags": list(flags),
            "focal_ratio": 1.0, "local": {"q": [round(x, 9) for x in quat_xyzw(r)], "c": [round(float(x), 6) for x in c]}}


def model(mid, cams, *, kind=JOB_FACE, sector=0, created=1.0, published=True, anchor=None, **meta):
    base = {"published": published, "cameras": cams, "georegistered": True, "alignment_rmse_m": 4.0,
            "yaw_sigma_deg": 3.0, "message": "georegistered from 9/9 GPS fixes"}
    if anchor is not None:
        base["anchor"] = anchor.to_meta()
    base.update(meta)
    return ModelRecord(id=mid, landmark_id="lm", job_id="j" + mid, kind=kind,
                       sector=None if kind == JOB_FULL else sector, ply_path=f"models/lm/{mid}.ply",
                       num_photos=len(cams), meta=base, created_at=created)


def doc(images, fv=FV):
    return {"model_id": "x", "feature_version": fv, "images": {f"{i:05d}_{pid}.jpg": pid for i, pid in
                                                               enumerate(images)}}


def mapref(mid="full", kind=JOB_FULL, images=("a", "b"), camera_en=None, anchor=None):
    return loc.MapRef(mid, kind, None if kind == JOB_FULL else 0, 1.0, anchor or make_anchor(),
                      {f"{pid}.jpg": pid for pid in images}, dict(camera_en or {}))


# ---------------------------------------------------------------- parameters


def test_the_loc_parameters_and_their_defaults():
    p = DEFAULTS
    assert (p.loc_min_inliers, p.loc_min_inlier_ratio, p.loc_max_reproj_frac, p.loc_ransac_frac) == \
        (30, 0.75, 0.0025, 0.012)
    assert (p.loc_second_ratio, p.loc_second_m, p.loc_second_deg) == (0.15, 5.0, 10.0)
    assert (p.loc_focal_mismatch, p.loc_min_focal_ratio, p.loc_max_focal_ratio) == (0.2, 0.3, 8.0)
    assert (p.loc_across_half, p.loc_across_deg, p.loc_min_height_m, p.loc_max_height_m, p.loc_max_look_deg) == \
        (0.5, 25.0, -15.0, 120.0, 75.0)
    assert p.loc_max_pairs == 200 and p.loc_withhold_gps_conflict is False
    assert with_overrides({"loc_withhold_gps_conflict": True}).loc_withhold_gps_conflict is True
    with pytest.raises(ValueError):
        with_overrides({"loc_withhold_gps_conflict": 1})
    assert with_overrides({"loc_max_pairs": 50.0}).loc_max_pairs == 50


# ---------------------------------------------------------------- which photos


def test_eligible_photos_and_why_the_others_are_skipped():
    an = make_anchor()
    full = model("full", {"s": stored_cam(an, [30, 0, 1.6], 270), "f": stored_cam(an, [0, 30, 1.6], 180,
                                                                                 flags=["far_from_gps"]),
                          "z": {"enu": None}}, kind=JOB_FULL, anchor=an)
    prior = model("prior", {"w": stored_cam(an, [-30, 0, 1.6], 90)}, georegistered=False, message=PRIOR_MSG)
    maps = [mapref("full", images=("s", "f", "z"), anchor=an)]
    key = loc.maps_key(maps)
    photos = [photo("s"), photo("f"), photo("z"), photo("q"), photo("w", -30.0), photo("n", consent=None),
              photo("v", consent="share-location-v0"), photo("r", status=PHOTO_REJECTED), photo("nf", f35=None),
              photo("n0", f35=0.0), photo("ni", proc=False), photo("t"), photo("t2")]
    records = {"t": LocRecord("t", "lm", LOC_REJECTED, 1.0, reason="too_few_matches", maps_key=key),
               "t2": {"status": LOC_ACCEPTED, "maps_key": "an older key"}}
    to_try, skipped = loc.eligible(photos, [full, prior], maps, records)
    # q a plain query; w's SfM camera is withheld (its model's frame is not fixed); t2 was tried against other maps
    assert to_try == ["q", "w", "t2"]
    assert skipped == {"f": "sfm_flagged", "z": "registered_in_map", "nf": "no_focal", "n0": "no_focal",
                       "ni": "no_image"}
    # s has an SfM pose; n, v (a consent text never shown) and r (rejected) may not be shown: never localized,
    # not even recorded; t was tried against these very maps
    assert not {"s", "n", "v", "r", "t"} & (set(to_try) | set(skipped))
    # no candidate map: nothing to try
    to_try, skipped = loc.eligible(photos, [full, prior], [], records)
    assert to_try == [] and skipped["q"] == skipped["w"] == skipped["t"] == "no_maps"
    assert (skipped["f"], skipped["nf"]) == ("sfm_flagged", "no_focal")
    # an accepted footprint snap fixes the prior-only frame: w is shown as sfm, not localized; a refused one does not
    snap = {"rotation_deg": 0.0, "shift_m": [0.0, 0.0], "scale": 1.0, "accepted": True}
    assert "w" not in loc.eligible(photos, [full, prior], maps, records, snaps={"prior": snap})[0]
    assert "w" in loc.eligible(photos, [full, prior], maps, records, snaps={"prior": {**snap, "accepted": False}})[0]


def test_maps_key_is_the_set_of_maps_and_the_rules(monkeypatch):
    a, b = mapref("a"), mapref("b", JOB_FACE)
    assert loc.maps_key([a, b]) == loc.maps_key([b, a]) == loc.maps_key(["b", "a"]) == loc.maps_key(["a", "b", "a"])
    assert loc.maps_key([a]) != loc.maps_key([a, b]) and loc.maps_key([]) != loc.maps_key([a])
    assert len(loc.maps_key([a])) == 32
    before = loc.maps_key([a, b])
    monkeypatch.setattr(loc, "LOC_VERSION", loc.LOC_VERSION + 1)
    assert loc.maps_key([a, b]) != before                                   # new rules: every photo is tried again


# ---------------------------------------------------------------- which maps


def test_candidate_maps_are_the_latest_published_models_with_a_fixed_frame_and_current_features():
    an = make_anchor()
    full_old = model("full0", {}, kind=JOB_FULL, created=0.5, anchor=an)
    full = model("full", {"a": stored_cam(an, [30, 0, 1.6], 270), "bad": {"enu": [1, "x", 2]}}, kind=JOB_FULL,
                 created=2.0, anchor=an)
    face0 = model("f0", {}, sector=0, anchor=an)
    face2 = model("f2", {}, sector=2, anchor=make_anchor(level="prior_only"), georegistered=False, message=PRIOR_MSG)
    face4 = model("f4", {}, sector=4, anchor=an)
    face5 = model("f5", {}, sector=5)                                 # legacy: georegistered, but no anchor
    face6 = model("f6", {}, sector=6, anchor=an)
    unpub = model("f7", {}, sector=7, anchor=an, published=False)
    docs = {"full0": doc(["a"]), "full": doc(["a", "c"]), "f2": doc(["d"]), "f4": doc(["e"], fv="older"),
            "f5": doc(["g"]), "f6": doc([]), "f7": doc(["h"])}
    models = [full_old, full, face0, face2, face4, face5, face6, unpub]
    maps, skipped = loc.candidate_maps(models, docs, None, FV)
    assert [m.model_id for m in maps] == ["full"]
    assert skipped == {"f0": "no_saved_map", "f2": "frame_not_fixed", "f4": "feature_version", "f5": "no_anchor",
                       "f6": "no_images"}                         # full0 superseded by full; f7 not published
    (m,) = maps
    assert (m.kind, m.sector, m.created_at, m.snapped) == (JOB_FULL, None, 2.0, False)
    assert m.anchor == an and m.photo_ids == {"a", "c"} and m.images["00001_c.jpg"] == "c"
    assert m.camera_en == {"a": pytest.approx((30.0, 0.0), abs=1e-3)}   # cameras with a finite position only
    # an accepted footprint snap fixes the prior-only frame; a refused one does not
    snap = {"rotation_deg": 2.0, "shift_m": [1.0, 0.0], "scale": 1.0, "accepted": True}
    maps, skipped = loc.candidate_maps(models, docs, {"f2": snap}, FV)
    assert [m.model_id for m in maps] == ["full", "f2"] and maps[1].snapped and "f2" not in skipped
    maps, skipped = loc.candidate_maps(models, docs, {"f2": {**snap, "accepted": False}}, FV)
    assert skipped["f2"] == "frame_not_fixed"


def test_select_pairs_takes_every_distinct_map_photo_and_caps_by_gps_distance():
    a = mapref("a", images=("p1", "p2", "q"), camera_en={"p1": (10.0, 0.0), "p2": (50.0, 0.0), "q": (0.0, 0.0)})
    b = mapref("b", JOB_FACE, images=("p2", "p3", "p4"), camera_en={"p2": (2.0, 0.0), "p3": (20.0, 0.0)})
    q = loc.Query("q", 1600, 1200, 28.0, gps_en=(0.0, 0.0))
    assert loc.select_pairs(q, [a, b]) == (["p1", "p2", "p3", "p4"], False)      # one pair per photo, not itself
    # capped: the nearest to the query's GPS first (p2 is 2 m away in map b), a camera-less photo last
    assert loc.select_pairs(q, [a, b], max_pairs=2) == (["p1", "p2"], True)
    assert loc.select_pairs(q, [a, b], max_pairs=3) == (["p1", "p2", "p3"], True)
    with params.override(loc_max_pairs=1):
        assert loc.select_pairs(q, [a, b]) == (["p2"], True)
    # without a GPS fix: by photo id
    assert loc.select_pairs(replace(q, gps_en=None), [a, b], max_pairs=2) == (["p1", "p2"], True)
    assert loc.select_pairs(q, []) == ([], False)


# ---------------------------------------------------------------- the frame


def test_a_pose_goes_through_the_models_anchor_like_its_own_cameras():
    an = make_anchor(yaw=-63.0, scale=0.37, t=(-5.0, 8.0, -1.0))
    m = model("m", {"a": stored_cam(an, [20.0, -30.0, 1.6], 10.0, 12.0, -3.0),
                    "b": stored_cam(an, [-25.0, 5.0, 4.0], 100.0, -2.0, 1.0)}, anchor=an)
    for pid in ("a", "b"):
        cam = m.meta["cameras"][pid]
        r, c = quat_matrix(cam["local"]["q"]), np.asarray(cam["local"]["c"])
        enu, heading, pitch, roll, cfe = loc.to_enu(an, r, c)
        assert enu == pytest.approx(cam["enu"], abs=1e-3) and heading == pytest.approx(cam["heading_deg"], abs=0.01)
        assert (pitch, roll) == pytest.approx(pp.camera_pitch_roll(m.meta, cam), abs=1e-6)
    enu, heading, pitch, roll, _ = loc.to_enu(an, *local_pose(an, [3.0, 4.0, 5.0], 200.0, 7.0, -4.0))
    assert enu == pytest.approx([3.0, 4.0, 5.0]) and (heading, pitch, roll) == pytest.approx((200.0, 7.0, -4.0))
    # the saved map is in the model's frame when its images give back the model's cameras through the anchor
    poses = {pid: (cam["local"]["q"], cam["local"]["c"]) for pid, cam in m.meta["cameras"].items()}
    assert loc.map_frame_ok(an, poses, m.meta["cameras"])
    shifted = {**poses, "b": (poses["b"][0], list(np.asarray(poses["b"][1]) + [0.2, 0.0, 0.0]))}   # 0.074 m
    assert not loc.map_frame_ok(an, shifted, m.meta["cameras"])
    assert not loc.map_frame_ok(make_anchor(), poses, m.meta["cameras"])            # another anchor
    assert not loc.map_frame_ok(an, {"x": poses["a"]}, m.meta["cameras"])          # nothing to compare


def test_make_query_uses_reconstructs_focal_prior_and_the_processed_size():
    ph = photo("q", 30.0, -40.0, 210.0, acc=4.5, f35=26.0)
    q = loc.make_query(ph, 1600, 1200, LAT, LON)
    assert q.gps_en == pytest.approx((30.0, -40.0), abs=1e-3) and (q.heading_deg, q.gps_accuracy_m) == (210.0, 4.5)
    assert q.focal_px == pytest.approx(26.0 / 43.27 * 2000.0) and q.long_side == 1600 and q.aspect == 4 / 3
    assert q.hfov_deg == pytest.approx(pp.hfov_from_35mm(26.0, 4 / 3))
    for bad in (dict(f35=None), dict(f35=0.0), dict(f35=float("nan"))):
        with pytest.raises(ValueError):
            loc.make_query(photo("q", **bad), 1600, 1200, LAT, LON)
    with pytest.raises(ValueError):
        loc.make_query(ph, 0, 1200, LAT, LON)


# ---------------------------------------------------------------- the gates


def passing(**kw):
    """A pose that passes every gate: 1600 px, prior focal 1000 px checked within 5 %, 100 of 120 correspondences."""
    base = dict(model_id="m", kind=JOB_FACE, num_corr=120, long_side_px=1600, focal_prior_px=1000.0,
                focal_px=1000.0, focal_free_px=1050.0, inliers=100, reproj_px=2.0, second_inliers=0,
                enu=[30.0, -40.0, 1.6], heading_deg=320.0, pitch_deg=2.0, roll_deg=0.0, look_deg=10.0)
    return loc.PoseStats(**{**base, **kw})


RADIUS = 150.0


@pytest.mark.parametrize("field,ok,bad,reason", [
    ("num_corr", 40, 29, "too_few_matches"),                     # 30 inliers of 40 pass the ratio too
    ("inliers", 30, 29, "inliers"),
    ("ratio", 75, 74, "inlier_ratio"),                           # of 100 correspondences
    ("reproj_px", DEFAULTS.loc_max_reproj_frac * 1600, DEFAULTS.loc_max_reproj_frac * 1600 + 0.01, "reprojection"),
    ("second_inliers", 15, 16, "ambiguous_in_model"),            # of 100
    ("focal_free_px", 1200.0, 1201.0, "focal_mismatch"),
    ("focal_free_px", 800.0, 799.0, "focal_mismatch"),
    ("focal_free_px", 1000.0, None, "focal_unchecked"),
    ("focal_px", 1000.0, 1000.5, "free_focal"),                  # the main solve must keep the EXIF prior
    ("focal_px", 1000.0, None, "pnp_failed"),
    ("enu", [90.0, -120.0, 1.6], [90.0, -120.01, 1.6], "implausible_position"),     # 150 m = the radius
    ("enu", [30.0, -40.0, 120.0], [30.0, -40.0, 120.01], "implausible_height"),
    ("enu", [30.0, -40.0, -15.0], [30.0, -40.0, -15.01], "implausible_height"),
    ("look_deg", 75.0, 75.01, "looks_away"),
    ("look_deg", 75.0, None, "looks_away"),
])
def test_each_gate_fires_beyond_its_threshold_and_not_at_it(field, ok, bad, reason):
    if field == "num_corr" and reason == "too_few_matches":
        good, worse = passing(num_corr=ok, inliers=30), passing(num_corr=bad, inliers=29)
    elif field == "inliers":
        good, worse = passing(inliers=ok, num_corr=40), passing(inliers=bad, num_corr=38)
    elif field == "ratio":
        good, worse = passing(inliers=ok, num_corr=100), passing(inliers=bad, num_corr=100)
    else:
        good, worse = passing(**{field: ok}), passing(**{field: bad})
    assert loc.check_pose(good, RADIUS) is None and loc.pose_failures(good, RADIUS) == []
    assert loc.check_pose(worse, RADIUS) == reason


def test_the_focal_ratio_must_be_plausible_for_the_prior_and_the_check():
    lo, hi = DEFAULTS.loc_min_focal_ratio * 1600, DEFAULTS.loc_max_focal_ratio * 1600
    for f in (lo, hi):
        assert loc.check_pose(passing(focal_prior_px=f, focal_px=f, focal_free_px=f), RADIUS) is None
    for f in (lo - 1, hi + 1):
        assert loc.check_pose(passing(focal_prior_px=f, focal_px=f, focal_free_px=f), RADIUS) == "implausible_focal"
    # the EXIF prior itself implausible, its check plausible and within 20 %: still no pose
    assert loc.check_pose(passing(focal_prior_px=lo - 1, focal_px=lo - 1, focal_free_px=lo + 20), RADIUS) == \
        "implausible_focal"
    # a free-focal check far off is a mismatch and implausible: both reported, never a pose
    assert loc.pose_failures(passing(focal_free_px=20000.0), RADIUS) == ["implausible_focal", "focal_mismatch"]


def test_a_gps_conflict_is_recorded_and_withholds_only_when_the_flag_says_so():
    s = passing(flags=["far_from_gps"])
    assert s.gps_conflict and loc.check_pose(s, RADIUS) is None
    assert passing(flags=["gps_and_compass"]).gps_conflict and not passing(flags=["odd_focal"]).gps_conflict
    with params.override(loc_withhold_gps_conflict=True):
        assert loc.check_pose(s, RADIUS) == "gps_conflict" and loc.check_pose(passing(), RADIUS) is None


# ---------------------------------------------------------------- one pose from the solves


def solve_at(anchor, centre_enu, heading, pitch=0.0, roll=0.0, *, inliers=100, focal=1000.0, reproj=1.5):
    r, c = local_pose(anchor, centre_enu, heading, pitch, roll)
    return loc.Solve(r, c, inliers, reproj, focal)


def test_pose_stats_puts_the_fixed_focal_solve_on_the_map_and_never_the_free_one():
    an = make_anchor(yaw=20.0, scale=2.0)
    m = mapref("m", JOB_FACE, anchor=an)
    f35 = 1000.0 / 2000.0 * 43.27                          # 1000 px on a 1600 x 1200 copy
    q = loc.Query("q", 1600, 1200, f35, gps_en=(28.0, -38.0), heading_deg=325.0)
    target = [0.0, 0.0, 10.0]
    fixed = solve_at(an, [30.0, -40.0, 1.6], 323.0, 3.0, -1.0)
    free = solve_at(an, [33.0, -44.0, 1.6], 323.5, inliers=101, focal=1080.0)
    s = loc.pose_stats(m, q, 120, fixed, free, None, target, RADIUS)
    assert s.reason is None and s.enu == pytest.approx([30.0, -40.0, 1.6]) and s.focal_px == pytest.approx(1000.0)
    assert (s.heading_deg, s.pitch_deg, s.roll_deg) == pytest.approx((323.0, 3.0, -1.0))
    assert s.focal_free_px == 1080.0 and s.inliers == 100 and s.inlier_ratio == pytest.approx(100 / 120)
    want_look = loc.look_angle_deg(cam_from_enu(323.0, 3.0, -1.0)[2], [30.0, -40.0, 1.6], target)
    assert s.look_deg == pytest.approx(want_look) and 0 < s.look_deg < 75
    assert s.gps_residual_m == pytest.approx(math.hypot(2.0, 2.0)) and s.compass_offset_deg == pytest.approx(-2.0)
    assert s.flags == [] and not s.gps_conflict
    # the free-focal solve handed over as the main one: never a pose
    assert loc.pose_stats(m, q, 120, free, free, None, target, RADIUS).reason == "free_focal"
    # no main solve: pnp_failed, no pose; a mismatched check: focal_mismatch (a cropped photo)
    assert loc.pose_stats(m, q, 120, None, free, None, target, RADIUS).reason == "pnp_failed"
    far_off = solve_at(an, [60.0, -90.0, 1.6], 300.0, focal=1300.0)
    assert loc.pose_stats(m, q, 120, fixed, far_off, None, target, RADIUS).reason == "focal_mismatch"
    # looking away from the model
    away = solve_at(an, [30.0, -40.0, 1.6], 143.0)
    assert loc.pose_stats(m, q, 120, away, free, None, target, RADIUS).reason == "looks_away"
    # 60 m from its own GPS fix: flagged like an SfM camera (3 x 18 m), recorded, not a gate
    gps_far = replace(q, gps_en=(30.0, 20.0))
    s = loc.pose_stats(m, gps_far, 120, fixed, free, None, target, RADIUS)
    assert s.reason is None and s.flags == ["far_from_gps"] and s.gps_conflict
    assert s.gps_residual_m == pytest.approx(60.0)
    no_fix = loc.pose_stats(m, replace(q, gps_en=None, heading_deg=None), 120, fixed, free, None, target, RADIUS)
    assert (no_fix.gps_residual_m, no_fix.compass_offset_deg, no_fix.flags, no_fix.reason) == (None, None, [], None)


def test_the_second_best_pose_counts_only_when_it_is_another_pose_in_metres_and_degrees():
    an = make_anchor(scale=2.0)
    m = mapref("m", JOB_FACE, anchor=an)
    q = loc.Query("q", 1600, 1200, 1000.0 / 2000.0 * 43.27)
    fixed = solve_at(an, [30.0, -40.0, 1.6], 323.0)
    free = solve_at(an, [30.0, -40.0, 1.6], 323.0, focal=1010.0)

    def second(centre, heading, inliers=40):
        return loc.pose_stats(m, q, 120, fixed, free, solve_at(an, centre, heading, inliers=inliers), [0, 0, 5],
                              RADIUS)

    near = second([34.9, -40.0, 1.6], 323.0)               # 4.9 m (2.45 local units x scale 2): the same pose
    assert near.second_inliers == 0 and near.reason is None
    far = second([35.1, -40.0, 1.6], 323.0)                # 5.1 m: another pose with 40 % of the inliers
    assert far.second_inliers == 40 and far.reason == "ambiguous_in_model"
    assert second([30.0, -40.0, 1.6], 333.5).reason == "ambiguous_in_model"    # 10.5 deg rotated
    assert second([30.0, -40.0, 1.6], 332.5).second_inliers == 0               # 9.5 deg: the same pose
    assert second([35.1, -40.0, 1.6], 323.0, inliers=15).reason is None         # 15 % of the best: allowed
    # the alias check is what stops a wrong face: a rotated twin with a fifth of the inliers
    twin = second([-40.0, -30.0, 1.6], 53.0, inliers=20)
    assert twin.reason == "ambiguous_in_model" and twin.second_ratio == pytest.approx(0.2)


# ---------------------------------------------------------------- across maps


def test_choose_takes_the_most_inliers_prefers_full_and_refuses_two_faces():
    a = passing(model_id="a", inliers=100)
    full = passing(model_id="full", kind=JOB_FULL, inliers=100)
    b = passing(model_id="b", inliers=120)
    assert loc.choose([a, b])[0] is b
    assert loc.choose([a, full]) == (full, None) and loc.choose([full, a]) == (full, None)
    # another accepting map with at least half the inliers and a heading more than 25 deg away: ambiguous
    other = passing(model_id="c", inliers=60, heading_deg=320.0 + 25.5)
    assert loc.choose([b, other]) == (None, "ambiguous_across_models")
    assert loc.choose([b, replace(other, inliers=59)])[0] is b                    # under half: fine
    assert loc.choose([b, replace(other, heading_deg=320.0 + 25.0)])[0] is b      # 25 deg: the same view
    assert loc.choose([b, replace(other, heading_deg=320.0 - 30.0)]) == (None, "ambiguous_across_models")
    # nothing accepted: the reason of the map that came closest; nothing tried at all: too_few_matches
    worse = [passing(model_id="x", inliers=10, num_corr=20, reason="too_few_matches"),
             passing(model_id="y", inliers=90, reason="focal_mismatch")]
    assert loc.choose(worse) == (None, "focal_mismatch") and loc.choose([]) == (None, "too_few_matches")


# ---------------------------------------------------------------- records


def test_records_of_a_localized_a_rejected_and_a_skipped_photo():
    q = loc.Query("q", 1600, 1200, 1000.0 / 2000.0 * 43.27)
    ph = photo("q")
    chosen = passing(model_id="full", kind=JOB_FULL, enu=[30.1234, -40.5678, 1.6], heading_deg=320.456,
                     pitch_deg=2.346, roll_deg=-0.567, flags=["far_from_gps"], gps_residual_m=61.0)
    other = passing(model_id="f2", inliers=12, num_corr=31, reason="inliers")
    prev = LocRecord("q", "lm", LOC_REJECTED, 1.0, reason="too_few_matches", maps_key="old", attempts=2)
    rec = loc.tried_record(ph, q, chosen, None, [chosen, other], "k", prev, 9.0, extra={"pairs": 7})
    assert (rec.status, rec.reason, rec.model_id, rec.maps_key, rec.attempts, rec.updated_at) == \
        (LOC_ACCEPTED, None, "full", "k", 3, 9.0)
    assert set(rec.pose) == {"enu", "heading_deg", "pitch_deg", "roll_deg", "hfov_deg", "aspect"}
    assert {k: v for k, v in rec.pose.items() if k != "hfov_deg"} == {
        "enu": [30.123, -40.568, 1.6], "heading_deg": 320.46, "pitch_deg": 2.35, "roll_deg": -0.57, "aspect": 1.3333}
    assert rec.pose["hfov_deg"] == pytest.approx(pp.hfov_from_35mm(q.focal_35mm, 4 / 3), abs=0.006)
    assert rec.stats["gps_conflict"] is True and rec.stats["pairs"] == 7 and rec.stats["chosen"]["model_id"] == "full"
    assert [c["model_id"] for c in rec.stats["candidates"]] == ["full", "f2"]
    assert "enu" not in str(rec.stats["candidates"])                       # no pose of a map that did not accept
    # the pose photoposes shows for it: source loc, the model's sfm sigma with the localizer's own in quadrature
    an = make_anchor()
    full = model("full", {}, kind=JOB_FULL, anchor=an)
    (pose,) = pp.photo_poses([ph], [full], LAT, LON, localizations={"q": rec}, aspects={"q": 4 / 3})
    assert pose.src == "loc" and pose.p == rec.pose["enu"] and pose.yaw == 320.46 and pose.hfov == rec.pose["hfov_deg"]
    sp, sy = pp.sfm_sigma(full.meta, rec.pose["enu"])
    assert pose.sp == pytest.approx(2 * math.hypot(sp, 1.5)) and pose.sy == pytest.approx(2 * math.hypot(sy, 0.7))
    # rejected: no pose, the reason; the first attempt counts 1
    rej = loc.tried_record(ph, q, None, "focal_mismatch", [other], "k", None, 9.0)
    assert (rej.status, rej.reason, rej.pose, rej.model_id, rej.attempts) == (LOC_REJECTED, "focal_mismatch", None,
                                                                              None, 1)
    # a pose that failed a gate is never recorded as a localization
    with pytest.raises(ValueError):
        loc.tried_record(ph, q, other, None, [other], "k", None, 9.0)
    # skipped: no attempt counted
    sk = loc.skipped_record(ph, "no_focal", "k", prev, 9.0)
    assert (sk.status, sk.reason, sk.attempts, sk.pose, sk.maps_key) == (LOC_SKIPPED, "no_focal", 2, None, "k")


# ---------------------------------------------------------------- the COLMAP part (fast: pycolmap on made-up data)


def test_correspondences_are_distinct_pairs_of_query_keypoint_and_3d_point():
    p3d_a = np.array([5, -1, 7, 9])                  # map image a: keypoint -> its 3-D point (-1: none)
    p3d_b = np.array([-1, 5, 8])
    a = np.array([[0, 10], [1, 11], [2, 12], [3, 12], [7, 13]])     # (map keypoint, query keypoint); 7 is past a's
    b = np.array([[1, 10], [2, 14]])                 # query keypoint 10 sees point 5 through a and b: once
    got = loc.correspondences([(a, p3d_a), (b, p3d_b), (np.zeros((0, 2)), p3d_a)])
    assert got.tolist() == [[10, 5], [12, 7], [12, 9], [14, 8]]
    assert loc.correspondences([]).shape == (0, 2)


W, H = 640, 480
F35 = 28.0                                           # prior focal 28 / 43.27 x 800 = 517.7 px


def views(anchor, centre_enu, heading, pitch, focal, n=200, seed=0, depth=(12.0, 14.0), noise_px=0.3):
    """n 3-D points (the anchored model's local frame) a camera at centre_enu sees, and their pixels through a
    distortion-free camera of this focal length, principal point at the centre (COLMAP's keypoint convention)."""
    rng = np.random.default_rng(seed)
    r, c = local_pose(anchor, centre_enu, heading, pitch)
    z = rng.uniform(*depth, n)
    x = rng.uniform(-0.45, 0.45, n) * W / focal * z
    y = rng.uniform(-0.45, 0.45, n) * H / focal * z
    cam = np.stack([x, y, z], axis=1)
    p3 = cam @ r + c                                  # local = R^T cam + c
    p2 = np.stack([focal * x / z + W / 2, focal * y / z + H / 2], axis=1) + rng.normal(0, noise_px, (n, 2))
    return p2, p3


def test_solve_pose_keeps_a_fixed_focal_length_and_estimates_a_free_one():
    pytest.importorskip("pycolmap")
    an = make_anchor(scale=1.0)
    f = 517.0
    p2, p3 = views(an, [20.0, -15.0, 1.6], 310.0, 5.0, f, depth=(8.0, 20.0))
    rng = np.random.default_rng(1)
    p2[:40] = rng.uniform([0, 0], [W, H], (40, 2))   # 20 % outliers
    fixed, mask = loc.solve_pose(p2, p3, W, H, f, 0.012 * W)
    assert fixed.focal_px == f and mask[40:].all() and not mask[:40].any() and fixed.inliers == 160
    assert 0.2 < fixed.reproj_px < 0.6                 # the 0.3 px noise per axis, over the inliers only
    enu, heading, pitch, _, _ = loc.to_enu(an, fixed.rotation, fixed.centre)
    assert enu == pytest.approx([20.0, -15.0, 1.6], abs=0.05) and heading == pytest.approx(310.0, abs=0.2)
    assert pitch == pytest.approx(5.0, abs=0.2)
    free, _ = loc.solve_pose(p2, p3, W, H, None, 0.012 * W)
    assert free.focal_px == pytest.approx(f, rel=0.02) and free.focal_px != f
    assert loc.solve_pose(p2[:3], p3[:3], W, H, f, 0.012 * W) is None


def test_pose_in_map_gives_the_anchored_pose_or_why_not():
    pytest.importorskip("pycolmap")
    an = make_anchor(yaw=25.0, scale=1.7)
    m = mapref("m", JOB_FACE, anchor=an)
    q = loc.Query("q", W, H, F35, gps_en=(31.0, -21.0), heading_deg=300.0)
    centre, heading, pitch = [30.0, -20.0, 1.6], 300.0, 6.0
    p2, p3 = views(an, centre, heading, pitch, q.focal_px)
    target = list(np.median(an.to_enu(p3), axis=0))
    s = loc.pose_in_map(m, q, p2, p3, target, RADIUS)
    assert s.reason is None and s.num_corr == 200 and s.inliers == 200 and s.second_inliers == 0
    assert s.enu == pytest.approx(centre, abs=0.05) and s.heading_deg == pytest.approx(heading, abs=0.2)
    assert s.pitch_deg == pytest.approx(pitch, abs=0.2) and s.focal_px == q.focal_px
    assert s.focal_free_px == pytest.approx(q.focal_px, rel=0.03) and s.look_deg < 30
    # a cropped photo: its EXIF focal length is 1.5 x what the pixels say; the free-focal check refuses the pose
    cropped = replace(q, focal_35mm=F35 * 1.5)
    s = loc.pose_in_map(m, cropped, p2, p3, target, RADIUS)
    assert s.reason == "focal_mismatch" and s.focal_free_px == pytest.approx(q.focal_px, rel=0.03)
    # a second face: 40 of the 200 correspondences agree with another pose 9 m away (the symmetric-alias check)
    o2, o3 = views(an, [21.0, -26.0, 1.6], 340.0, 6.0, q.focal_px, n=40, seed=5)
    s = loc.pose_in_map(m, q, np.vstack([p2[:160], o2]), np.vstack([p3[:160], o3]), target, RADIUS)
    assert s.reason == "ambiguous_in_model" and s.inlier_ratio == pytest.approx(0.8, abs=0.01)
    assert s.second_ratio == pytest.approx(0.25, abs=0.02)


def test_pose_in_map_solves_nothing_below_the_minimum_correspondences(monkeypatch):
    an = make_anchor()
    q = loc.Query("q", W, H, F35)
    p2, p3 = views(an, [30.0, -20.0, 1.6], 300.0, 6.0, q.focal_px, n=DEFAULTS.loc_min_inliers - 1)

    def boom(*a, **k):
        raise AssertionError("no solve below loc_min_inliers")

    monkeypatch.setattr(loc, "solve_pose", boom)
    s = loc.pose_in_map(mapref("m", JOB_FACE, anchor=an), q, p2, p3, [0.0, 0.0, 5.0], RADIUS)
    assert s.reason == "too_few_matches" and s.num_corr == 29 and s.focal_px is None


def _tiny_images(root, n):
    from PIL import Image

    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for k in range(n):
        Image.fromarray((rng.random((48, 64, 3)) * 255).astype(np.uint8)).save(images / f"{k:05d}_p{k}.jpg")
    return images, sorted(p.name for p in images.iterdir())


def _fake_db(path, images, names, pairs):
    """A database of these images (PER_IMAGE cameras) with made-up features and, per (a, b) in pairs (image ids),
    made-up matches and a verified geometry written in that order."""
    pycolmap = pytest.importorskip("pycolmap")
    path.unlink(missing_ok=True)
    pycolmap.Database.open(path).close()
    pycolmap.import_images(path, images, camera_mode=pycolmap.CameraMode.PER_IMAGE, image_names=names)
    db = pycolmap.Database.open(path)
    try:
        for im in db.read_all_images():
            rng = np.random.default_rng(int(im.name[:5]))
            db.write_keypoints(im.image_id, rng.random((20, 6)).astype(np.float32))
            db.write_descriptors(im.image_id, pycolmap.FeatureDescriptors(
                pycolmap.FeatureExtractorType.SIFT, (rng.random((20, 128)) * 255).astype(np.uint8)))
        for a, b in pairs:
            m = np.random.default_rng(10 * a + b).integers(0, 20, (8, 2)).astype(np.uint32)
            db.write_matches(a, b, m)
            g = pycolmap.TwoViewGeometry()
            g.config = 3
            g.inlier_matches = m[:5]
            db.write_two_view_geometry(a, b, g)
    finally:
        db.close()


def test_read_two_view_geometry_gives_the_matches_in_the_order_asked(tmp_path):
    """The localizer reads (map image, query) whatever order the pair was matched in (ColmapLocalizer._poses)."""
    pycolmap = pytest.importorskip("pycolmap")
    images, names = _tiny_images(tmp_path, 2)
    _fake_db(tmp_path / "a.db", images, names, [(2, 1)])
    db = pycolmap.Database.open(tmp_path / "a.db")
    try:
        m21 = np.asarray(db.read_two_view_geometry(2, 1).inlier_matches)
        m12 = np.asarray(db.read_two_view_geometry(1, 2).inlier_matches)
    finally:
        db.close()
    assert len(m21) == 5 and np.array_equal(m12, m21[:, ::-1])


def test_the_cache_seeds_a_pair_matched_either_way_and_harvests_each_in_the_order_computed(tmp_path):
    from crowd3d import landmark_map as lmap

    pycolmap = pytest.importorskip("pycolmap")
    images, names = _tiny_images(tmp_path, 3)
    ex = lmap.extraction_options(pycolmap, 1600)
    ver = pycolmap.TwoViewGeometryOptions()
    ver.ransac.random_seed = 0
    fv, mv = lmap.feature_version(pycolmap, ex), lmap.match_version(pycolmap, ver)
    # a reconstruction job matched (1, 2) and, the other way round, (3, 1)
    job = tmp_path / "job.db"
    _fake_db(job, images, names, [(1, 2), (3, 1)])
    cache = lmap.FeatureCache(tmp_path / "cache.sqlite3")
    keys = {k + 1: lmap.file_sha256(images / n) for k, n in enumerate(names)}
    assert cache.harvest(job, keys, fv, [(1, 2), (3, 1)], lmap.camera_signatures(job), mv) == (3, 2)
    # the localize job's database: the same images in the same order, nothing in it yet
    db_path = tmp_path / "loc.db"
    pycolmap.Database.open(db_path).close()
    pycolmap.import_images(db_path, images, camera_mode=pycolmap.CameraMode.PER_IMAGE, image_names=names)
    lc = loc.LocCache(cache, pycolmap, db_path, {n: images / n for n in names}, ex, ver)
    lc.seed_features()
    assert (lc.stats.features, lc.stats.features_hit) == (3, 3)
    wanted = {lmap.pair_id(1, 2): (1, 2), lmap.pair_id(1, 3): (1, 3), lmap.pair_id(2, 3): (2, 3)}
    lc.seed_pairs(wanted)
    assert lc.ok and lc.seeded == {lmap.pair_id(1, 2), lmap.pair_id(1, 3)}
    assert lc.order == {**wanted, lmap.pair_id(1, 3): (3, 1)}            # as the job that computed its rows
    assert (lc.stats.pairs, lc.stats.pairs_hit) == (3, 2)
    with closing(sqlite3.connect(job)) as a, closing(sqlite3.connect(db_path)) as b:
        for table in ("matches", "two_view_geometries"):
            q = f"SELECT * FROM {table} WHERE pair_id = ?"
            for pid in lc.seeded:
                assert b.execute(q, (pid,)).fetchone() == a.execute(q, (pid,)).fetchone()
    # this job matches (2, 3); the harvest stores that pair in its order and leaves the seeded ones as they were
    db = pycolmap.Database.open(db_path)
    try:
        m = np.array([[0, 1], [2, 3]], dtype=np.uint32)
        db.write_matches(2, 3, m)
        g = pycolmap.TwoViewGeometry()
        g.config = 3
        g.inlier_matches = m
        db.write_two_view_geometry(2, 3, g)
    finally:
        db.close()
    lc.harvest("loc")
    with closing(sqlite3.connect(cache.path)) as c:
        stored = {(r[0], r[1]) for r in c.execute("SELECT key1, key2 FROM pairs")}
    assert stored == {(keys[1], keys[2]), (keys[3], keys[1]), (keys[2], keys[3])}
    assert cache.uses()[-1]["label"] == "loc" and cache.uses()[-1]["pairs_hit"] == 2


def test_localize_skips_photos_without_copies_or_maps_and_stops_before_a_query(tmp_path):
    pytest.importorskip("pycolmap")
    settings = loc.Settings(data_dir=tmp_path, reconstructor="colmap")
    (tmp_path / "photos" / "lm").mkdir(parents=True)
    for pid in ("q1", "q2", "m1"):
        (tmp_path / "photos" / "lm" / f"{pid}.proc.jpg").write_bytes(b"jpeg")
    lm = loc.Landmark("lm", "Tower", LAT, LON, RADIUS, 0.0)
    q1, q2, gone, m1 = photo("q1"), photo("q2"), photo("gone"), photo("m1")
    sm = loc.SparseMap(mapref("m", JOB_FACE, images=("m1",)), None, {"m1": 1}, [0.0, 0.0, 5.0])
    localizer = loc.ColmapLocalizer(settings)
    run = localizer.localize([q1, gone, q2], [sm], [q1, q2, gone, m1], lm, tmp_path / "w", stop=lambda: True)
    assert run.unfinished == ["q1", "q2"] and run.skipped == {"gone": "no_image"} and run.outcomes == {}
    assert not (tmp_path / "w").exists()                     # stopped before COLMAP did anything
    run = localizer.localize([q1, q2], [sm], [q1, q2, m1], lm, tmp_path / "w", deadline=0.0)
    assert run.unfinished == ["q1", "q2"]
    run = localizer.localize([q1], [], [q1], lm, tmp_path / "w")
    assert run.skipped == {"q1": "no_maps"} and run.stats["tried"] == 0
    # every map photo's copy gone: nothing to match against, so not tried (no verdict on the photo), before COLMAP
    gone_map = loc.SparseMap(mapref("m", JOB_FACE, images=("gone",)), None, {"gone": 1}, [0.0, 0.0, 5.0])
    run = localizer.localize([q1, q2], [gone_map], [q1, q2, gone], lm, tmp_path / "w")
    assert run.skipped == {"q1": "no_map_images", "q2": "no_map_images"} and run.outcomes == {}
    assert run.stats["map_photos_missing"] == 1 and not (tmp_path / "w").exists()


def test_a_fault_of_the_inputs_skips_a_query_and_never_rejects_it(tmp_path):
    """A query COLMAP cannot read, one without a feature, and one whose map photos are all unreadable are skipped with
    why (eligible's skipped photos), never rejected as too_few_matches; nothing raises."""
    pytest.importorskip("pycolmap")
    from PIL import Image

    settings = loc.Settings(data_dir=tmp_path, reconstructor="colmap")
    folder = tmp_path / "photos" / "lm"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((240, 320, 3)) * 255).astype(np.uint8)).save(folder / "ok.proc.jpg")
    Image.fromarray(np.full((240, 320, 3), 128, np.uint8)).save(folder / "blank.proc.jpg")
    for pid in ("bad", "m1"):
        (folder / f"{pid}.proc.jpg").write_bytes(b"not a jpeg")
    lm = loc.Landmark("lm", "Tower", LAT, LON, RADIUS, 0.0)
    ok, blank, bad, m1 = photo("ok"), photo("blank"), photo("bad"), photo("m1")
    sm = loc.SparseMap(mapref("m", JOB_FACE, images=("m1",)), None, {"m1": 1}, [0.0, 0.0, 5.0])
    run = loc.ColmapLocalizer(settings).localize([ok, blank, bad], [sm], [ok, blank, bad, m1], lm, tmp_path / "w")
    assert run.skipped == {"bad": "no_image", "blank": "no_features", "ok": "no_map_images"}
    assert run.outcomes == {} and not run.unfinished
    assert (run.stats["map_photos_unreadable"], run.stats["map_photos_missing"], run.stats["tried"]) == (1, 0, 0)
    # skipped, the photos keep their attempts: none was tried
    recs = [loc.skipped_record(ph, run.skipped[ph.id], "k", None, 1.0) for ph in (ok, blank, bad)]
    assert {(r.status, r.attempts) for r in recs} == {(LOC_SKIPPED, 0)}


def test_records_stay_keyed_by_the_candidate_maps_whatever_load_sparse_maps_refused(tmp_path):
    """A candidate map load_sparse_maps refuses is refused again in every job; keyed by the maps it loaded, a record
    would never be the one eligible() compares, and every job would try the photo again."""
    pytest.importorskip("pycolmap")
    settings = loc.Settings(data_dir=tmp_path, reconstructor="colmap")
    (tmp_path / "photos" / "lm").mkdir(parents=True)
    (tmp_path / "photos" / "lm" / "q.proc.jpg").write_bytes(b"jpeg")
    lm = loc.Landmark("lm", "Tower", LAT, LON, RADIUS, 0.0)
    q = photo("q")
    cands = [mapref("gone", JOB_FACE)]
    assert loc.eligible([q], [], cands) == (["q"], {})
    maps, skipped = loc.load_sparse_maps(tmp_path, cands, {})
    assert maps == [] and skipped == {"gone": "unreadable_map"}
    run = loc.ColmapLocalizer(settings).localize([q], maps, [q], lm, tmp_path / "w")
    assert run.skipped == {"q": "no_maps"}
    recs = {pid: loc.skipped_record(q, why, loc.maps_key(cands), None, 1.0) for pid, why in run.skipped.items()}
    assert loc.eligible([q], [], cands, recs) == ([], {})                       # the next job leaves it
    assert loc.eligible([q], [], cands, {"q": replace(recs["q"], maps_key=loc.maps_key(maps))}) == (["q"], {})
    with pytest.raises(TypeError):
        loc.maps_key([loc.SparseMap(cands[0], None, {}, [0.0, 0.0, 0.0])])
    assert loc.maps_key([cands[0]]) == loc.maps_key(["gone"])


def test_a_map_that_is_not_saved_is_not_loaded_and_the_localizer_follows_the_reconstructor(tmp_path):
    pytest.importorskip("pycolmap")
    maps, skipped = loc.load_sparse_maps(tmp_path, [mapref("gone", JOB_FACE)], {})
    assert maps == [] and skipped == {"gone": "unreadable_map"}
    assert isinstance(loc.get_localizer("colmap"), loc.ColmapLocalizer)
    assert loc.get_localizer("fake") is None and loc.get_localizer("nope") is None


# ---------------------------------------------------------------- slow: real COLMAP on a synthetic facade


@pytest.fixture(scope="module")
def facade_map(tmp_path_factory):
    """A face model of sector 2 built by a real reconstruction job from 4 of 6 synthetic photos and published (its map
    saved, its features and pairs harvested into the landmark's feature cache). The GPS fixes and compasses are exact
    (each fix reporting 2 m, the accuracy floor, so that 4 fixes spread over about 5 m georegister the model), so the
    model's anchor is the generator's frame and an error against the generator's truth is the localizer's own (with the
    generator's 8 deg compass noise the model's own cameras are 4.7 deg off, and so is any pose put on it). The 2 photos
    held back (between the others) and one of the far side (sector 6) are consented and were in no job; the second
    held-back photo's copy keeps no EXIF focal length (only its record does), so the localizer must write its focal
    prior itself, as a reconstruction job does."""
    pytest.importorskip("pycolmap")
    import shutil
    import time

    from PIL import Image

    from crowd3d import synthetic as syn
    from crowd3d.config import Settings
    from crowd3d.jobs import JobRunner
    from crowd3d.reconstruct import ColmapReconstructor
    from crowd3d.records import JOB_DONE, JOB_QUEUED, Job, Landmark
    from crowd3d.store import Store

    root = tmp_path_factory.mktemp("loc-facade")
    size = (W, H)
    face = syn.generate_dataset(root / "face", sectors=[2], photos_per_sector=6, size=size, include_bad=False,
                                gps_noise_m=0.0, heading_noise_deg=0.0, seed=3)
    far = syn.generate_dataset(root / "far", sectors=[6], photos_per_sector=1, size=size, include_bad=False,
                               gps_noise_m=0.0, heading_noise_deg=0.0, seed=4)
    shots = [(root / "face", p) for p in face["photos"]] + [(root / "far", p) for p in far["photos"]]
    settings = Settings(data_dir=root / "data", reconstructor="colmap", inline_jobs=True, stylize_published=False)
    store = Store(settings)
    lm = Landmark("lm", "Box", face["landmark"]["lat"], face["landmark"]["lon"], 150.0, 0.0)
    store.create_landmark(lm)
    truth = {}
    for k, (ds, p) in enumerate(shots):
        pid = f"p{k}"
        rel = f"photos/lm/{pid}.proc.jpg"
        (settings.data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        if pid == "p4":
            with Image.open(ds / p["file"]) as im:
                im.save(settings.data_dir / rel, quality=95)           # no EXIF
        else:
            shutil.copyfile(ds / p["file"], settings.data_dir / rel)
        store.add_photo(PhotoRecord(id=pid, landmark_id="lm", user_id="u", status=PHOTO_ACCEPTED,
                                    uploaded_at=1000.0 + k, lat=p["exif_lat"], lon=p["exif_lon"],
                                    heading_deg=p["exif_heading_deg"], gps_accuracy_m=2.0,
                                    focal_35mm=float(syn.focal_35mm_for(size)), processed_path=rel,
                                    location_consent=LOCATION_CONSENT_V1))
        truth[pid] = p
    held, map_ids, far_id = ["p2", "p4"], ["p0", "p1", "p3", "p5"], "p6"
    runner = JobRunner(store, settings, ColmapReconstructor(settings))
    job = runner.run_job(store.create_job(Job(id="j1", landmark_id="lm", kind=JOB_FACE, sector=2, status=JOB_QUEUED,
                                              photo_ids=map_ids, created_at=time.time())))
    assert job.status == JOB_DONE, job.error
    (published,) = [m for m in store.list_models("lm") if m.meta["published"]]
    assert sorted(published.meta["cameras"]) == map_ids and published.meta["anchor"]["level"] == "full"
    try:
        yield settings, store, lm, truth, held, map_ids, far_id
    finally:
        store.close()


def _maps(settings, store):
    from crowd3d.landmark_map import MODELS_DIR, job_feature_version, load_map

    models = store.list_models("lm")
    root = settings.maps_dir / "lm" / MODELS_DIR
    maps, skipped = loc.candidate_maps(models, {m.id: load_map(root, m.id) for m in models}, None,
                                       job_feature_version(settings.processing_max_side_px))
    assert len(maps) == 1 and skipped == {}
    return models, root, maps


@pytest.mark.slow
def test_held_back_photos_are_localized_within_half_a_metre_and_a_degree(facade_map, tmp_path):
    from crowd3d import landmark_map as lmap
    from crowd3d.landmark_map import CACHE_FILE, FeatureCache

    settings, store, lm, truth, held, map_ids, far_id = facade_map
    models, root, maps = _maps(settings, store)
    photos = store.list_photos("lm")
    by = {p.id: p for p in photos}
    to_try, skipped = loc.eligible(photos, models, maps)
    assert to_try == sorted([*held, far_id]) and skipped == {}
    # the same photo without its EXIF focal length is never tried
    assert loc.eligible([replace(by[held[0]], focal_35mm=None)], models, maps) == ([], {held[0]: "no_focal"})
    sparse, lskip = loc.load_sparse_maps(root, maps, {m.id: m.meta["cameras"] for m in models})
    assert len(sparse) == 1 and lskip == {}
    cache = FeatureCache(settings.maps_dir / "lm" / CACHE_FILE)
    localizer = loc.ColmapLocalizer(settings)
    run = localizer.localize([by[p] for p in to_try], sparse, photos, lm, tmp_path / "w1", cache=cache)
    assert run.stats["map_images_refused"] == 0 and run.stats["map_photos"] == 4 and run.stats["images"] == 7
    assert not run.unfinished and not run.skipped and run.stats["localized"] == 2
    assert (run.stats["features"], run.stats["features_seeded"]) == (7, 4)       # the map photos' from the job
    # the pitch carries the model's own gravity error (its 4 cameras face one way, so their right vectors leave the
    # tilt about that axis open): the localizer must add nothing to it
    (published,) = [m for m in models if m.meta["published"]]
    anchor = loc.read_anchor(published.meta)
    model_pitch_err = float(np.median([
        pp.pitch_roll_deg(*anchor.camera_enu(c["local"]["q"], c["local"]["c"])[1][[0, 2]])[0]
        - truth[pid]["true_pitch_deg"] for pid, c in published.meta["cameras"].items()]))
    for pid in held:
        o = run.outcomes[pid]
        assert o.reason is None and o.extra["pairs"] == 4 and not o.extra["pairs_capped"], o.cands[0].summary()
        t = truth[pid]
        assert math.dist(o.chosen.enu, t["true_enu"]) < 0.5, (o.chosen.enu, t["true_enu"])
        assert abs(loc.angle_diff_deg(o.chosen.heading_deg, t["true_heading_deg"])) < 1.0
        assert abs(o.chosen.pitch_deg - t["true_pitch_deg"] - model_pitch_err) < 0.5
        rec = o.record(loc.maps_key(maps), None, 5.0)
        assert rec.status == LOC_ACCEPTED and rec.model_id == maps[0].model_id and rec.pose["enu"] == pytest.approx(
            o.chosen.enu, abs=1e-3)
        (shown,) = [x for x in pp.photo_poses(photos, models, lm.lat, lm.lon, localizations={pid: rec})
                    if x.photo_id == pid]
        assert shown.src == pp.SRC_LOC and shown.p == pytest.approx(rec.pose["enu"])
    assert run.outcomes[far_id].chosen is None and run.outcomes[far_id].reason == "too_few_matches"
    # every image's COLMAP camera has the EXIF focal prior (the copy without EXIF too), as in a reconstruction job
    pycolmap = pytest.importorskip("pycolmap")
    db = pycolmap.Database.open(tmp_path / "w1" / "database.db")
    try:
        cams = {im.name.split("_", 1)[1][:-4]: db.read_camera(im.camera_id) for im in db.read_all_images()}
    finally:
        db.close()
    assert len(cams) == 7 and all(c.has_prior_focal_length for c in cams.values())
    assert cams[held[1]].params[0] == run.outcomes[held[1]].query.focal_px
    # again: every feature and pair from the cache, the same poses
    again = localizer.localize([by[p] for p in to_try], sparse, photos, lm, tmp_path / "w2", cache=cache)
    assert again.stats["pairs"] == run.stats["pairs"] == 12
    assert (again.stats["pairs_seeded"], again.stats["features_seeded"]) == (12, 7)
    for pid in to_try:
        a, b = run.outcomes[pid], again.outcomes[pid]
        assert [c.summary() for c in a.cands] == [c.summary() for c in b.cands]
        assert (a.chosen and a.chosen.enu) == (b.chosen and b.chosen.enu)
    # what the job matched is in the cache in image-id order (the upload order), as an exhaustive reconstruction job
    # of the landmark would match it, so such a job finds it
    uploaded = {lmap.file_sha256(settings.data_dir / p.processed_path): p.uploaded_at for p in photos}
    with closing(sqlite3.connect(cache.path)) as c:
        rows = c.execute("SELECT key1, key2, ids_ascending FROM pairs").fetchall()
    assert len(rows) == 6 + 12 and all(asc == 1 and uploaded[k1] < uploaded[k2] for k1, k2, asc in rows)
    # a stop after the first query leaves the others to a later job
    calls = []
    part = localizer.localize([by[p] for p in to_try], sparse, photos, lm, tmp_path / "w3", cache=cache,
                              stop=lambda: len(calls.append(1) or calls) > 2)
    assert list(part.outcomes) == to_try[:1] and part.unfinished == to_try[1:]
    # a map image whose keypoints are not its points2D is never matched against; a query without its EXIF focal
    # length that got this far is not tried
    real = loc.SparseMap.num_points2d
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loc.SparseMap, "num_points2d", lambda m, pid: real(m, pid) + (pid == map_ids[0]))
        odd = localizer.localize([by[held[0]], replace(by[held[1]], focal_35mm=None)], sparse, photos, lm,
                                 tmp_path / "w4", cache=cache)
    assert odd.stats["map_images_refused"] == 1 and odd.outcomes[held[0]].extra["pairs"] == 3
    assert odd.skipped == {held[1]: "no_focal"} and list(odd.outcomes) == [held[0]]
    # every map image refused: nothing to match against, so the query is not tried (no verdict on it)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loc.SparseMap, "num_points2d", lambda m, pid: real(m, pid) + 1)
        none = localizer.localize([by[held[0]]], sparse, photos, lm, tmp_path / "w5", cache=cache)
    assert none.stats["map_images_refused"] == 4 and none.skipped == {held[0]: "no_map_images"} and not none.outcomes


@pytest.mark.slow
def test_a_map_photo_found_again_from_the_reconstructions_cached_pairs_gives_back_its_camera(facade_map, tmp_path):
    """Localized against the other photos of its own map, a map photo comes back where its SfM camera is, every pair
    seeded from the rows its reconstruction job harvested (that job matched them in its own order); a map whose
    model's cameras are not what its images give back is not used."""
    from crowd3d.landmark_map import CACHE_FILE, FeatureCache

    settings, store, lm, truth, held, map_ids, far_id = facade_map
    models, root, maps = _maps(settings, store)
    cams = {m.id: m.meta["cameras"] for m in models}
    sparse, _ = loc.load_sparse_maps(root, maps, cams)
    photos = store.list_photos("lm")
    by = {p.id: p for p in photos}
    run = loc.ColmapLocalizer(settings).localize([by[map_ids[1]]], sparse, photos, lm, tmp_path / "w",
                                                 cache=FeatureCache(settings.maps_dir / "lm" / CACHE_FILE))
    o = run.outcomes[map_ids[1]]
    assert o.reason is None and o.extra["pairs"] == o.extra["pairs_seeded"] == 3
    cam = cams[maps[0].model_id][map_ids[1]]
    assert math.dist(o.chosen.enu, cam["enu"]) < 0.1 and abs(loc.angle_diff_deg(o.chosen.heading_deg,
                                                                                cam["heading_deg"])) < 0.3
    moved = {mid: {pid: {**c, "enu": [c["enu"][0] + 0.2, *c["enu"][1:]]} for pid, c in cs.items()}
             for mid, cs in cams.items()}
    assert loc.load_sparse_maps(root, maps, moved) == ([], {maps[0].model_id: "map_not_in_model_frame"})


# ---------------------------------------------------------------- slow: the localize job (D30 step 8) with real COLMAP


class _Messages(logging.Handler):
    """Keeps every log message (a module-scoped fixture has no caplog)."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def localize_job(tmp_path_factory):
    """A landmark whose sector-2 facade was built and published by a real reconstruction job from 4 of 6 synthetic
    photos (exact GPS and compass, as facade_map), with Settings.localize_photos on and the product's own localizer
    (localize.get_localizer for the COLMAP reconstructor): the localize job that job's end queued has run. Besides the
    4 map photos: the 2 held back and one of the far side (consented, EXIF focal length), one held-back image again
    without consent, and once more consented without an EXIF focal length. Photo ids are 32 hex digits, as the
    store's, so a log line naming one would show."""
    pytest.importorskip("pycolmap")
    import shutil
    import time
    import uuid

    from crowd3d import synthetic as syn
    from crowd3d.config import Settings
    from crowd3d.jobs import JobRunner
    from crowd3d.reconstruct import ColmapReconstructor
    from crowd3d.records import JOB_DONE, JOB_QUEUED, Job, Landmark
    from crowd3d.store import Store

    root = tmp_path_factory.mktemp("loc-job")
    size = (W, H)
    face = syn.generate_dataset(root / "face", sectors=[2], photos_per_sector=6, size=size, include_bad=False,
                                gps_noise_m=0.0, heading_noise_deg=0.0, seed=3)
    far = syn.generate_dataset(root / "far", sectors=[6], photos_per_sector=1, size=size, include_bad=False,
                               gps_noise_m=0.0, heading_noise_deg=0.0, seed=4)
    shots = [(root / "face", p) for p in face["photos"]] + [(root / "far", p) for p in far["photos"]]
    shots += [shots[2], shots[4]]                 # a held-back image again: without consent; without a focal length
    settings = Settings(data_dir=root / "data", reconstructor="colmap", inline_jobs=True, stylize_published=False,
                        localize_photos=True)
    store = Store(settings)
    lm = Landmark("lm", "Box", face["landmark"]["lat"], face["landmark"]["lon"], 150.0, 0.0)
    store.create_landmark(lm)
    ids = [uuid.uuid4().hex for _ in shots]
    truth = {}
    for k, ((ds, p), pid) in enumerate(zip(shots, ids)):
        rel = f"photos/lm/{pid}.proc.jpg"
        (settings.data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ds / p["file"], settings.data_dir / rel)
        store.add_photo(PhotoRecord(id=pid, landmark_id="lm", user_id="u", status=PHOTO_ACCEPTED,
                                    uploaded_at=1000.0 + k, lat=p["exif_lat"], lon=p["exif_lon"],
                                    heading_deg=p["exif_heading_deg"], gps_accuracy_m=2.0,
                                    focal_35mm=None if k == 8 else float(syn.focal_35mm_for(size)), processed_path=rel,
                                    location_consent=None if k == 7 else LOCATION_CONSENT_V1))
        truth[pid] = p
    names = {"map": [ids[0], ids[1], ids[3], ids[5]], "held": [ids[2], ids[4]], "far": ids[6], "private": ids[7],
             "nofocal": ids[8]}
    messages = _Messages()
    logging.getLogger("crowd3d").addHandler(messages)
    old_level = logging.getLogger("crowd3d").level
    logging.getLogger("crowd3d").setLevel(logging.INFO)
    try:
        runner = JobRunner(store, settings, ColmapReconstructor(settings))
        job = runner.run_job(store.create_job(Job(id="j1", landmark_id="lm", kind=JOB_FACE, sector=2,
                                                  status=JOB_QUEUED, photo_ids=names["map"], created_at=time.time())))
        assert job.status == JOB_DONE, job.error
        assert runner.schedule("lm") == []            # no reconstruction to plan; the localize job queued runs now
        yield settings, store, lm, truth, names, runner, list(messages.messages)
    finally:
        logging.getLogger("crowd3d").removeHandler(messages)
        logging.getLogger("crowd3d").setLevel(old_level)
        store.close()


@pytest.mark.slow
def test_the_localize_job_places_consented_photos_and_the_public_poses_show_them(localize_job):
    """D30 step 8 end to end on real COLMAP: the job a published facade queues localizes the held-back consented
    photos within 0.5 m / 1 deg of the generator's truth (the far-side one is rejected, the one without an EXIF focal
    length skipped, the one without consent and the map photos get no record), every record keyed by the candidate
    maps; the public poses show them as "loc" rows, public_row of their internal poses; the job logs counts, never a
    photo id, and leaves no work dir; nothing new is queued by planning again, a restart or a server start on the same
    data dir; a withdrawal removes the photo's record and row."""
    from fastapi.testclient import TestClient

    from crowd3d import api
    from crowd3d.consent import deployment_salt
    from crowd3d.jobs import JobRunner
    from crowd3d.landmark_map import MODELS_DIR, job_feature_version, load_map
    from crowd3d.reconstruct import ColmapReconstructor
    from crowd3d.records import JOB_DONE, JOB_LOCALIZE

    settings, store, lm, truth, names, runner, messages = localize_job
    (job,) = [j for j in store.list_jobs("lm") if j.kind == JOB_LOCALIZE]
    assert (job.status, job.error, job.photo_ids) == (JOB_DONE, None, [*names["held"], names["far"]])
    assert isinstance(runner._localizer(), loc.ColmapLocalizer)
    models = store.published_models("lm")
    maps, _ = loc.candidate_maps(models, {m.id: load_map(settings.maps_dir / "lm" / MODELS_DIR, m.id) for m in models},
                                 None, job_feature_version(settings.processing_max_side_px))
    key = loc.maps_key(maps)
    recs = store.localizations("lm")
    assert set(recs) == {*names["held"], names["far"], names["nofocal"]}
    assert {r.maps_key for r in recs.values()} == {key}
    for pid in names["held"]:
        r, t = recs[pid], truth[pid]
        assert (r.status, r.model_id, r.attempts) == (LOC_ACCEPTED, models[0].id, 1)
        assert math.dist(r.pose["enu"], t["true_enu"]) < 0.5, (r.pose, t["true_enu"])
        assert abs(loc.angle_diff_deg(r.pose["heading_deg"], t["true_heading_deg"])) < 1.0
    assert (recs[names["far"]].status, recs[names["far"]].reason) == (LOC_REJECTED, "too_few_matches")
    assert (recs[names["nofocal"]].status, recs[names["nofocal"]].reason) == (LOC_SKIPPED, "no_focal")
    # the public poses: the localized photos as "loc" rows, exactly public_row of their internal poses
    salt = deployment_salt(settings.data_dir)
    body = api.landmark_poses(store, settings, lm, salt)
    shared = [p for p in store.list_photos("lm", PHOTO_ACCEPTED) if p.id != names["private"]]
    internal = {row["id"]: row for pose in pp.photo_poses(shared, models, lm.lat, lm.lon, localizations=recs,
                                                          aspects=pp.processed_aspects(shared, settings.data_dir))
                if (row := pp.public_row(pose, salt)) is not None}
    assert {r["id"]: r for r in body["photos"]} == internal
    assert body["counts"]["loc"] == 2 and body["counts"]["sfm"] == 4
    assert {internal[pp.public_id(pid, salt)]["src"] for pid in names["held"]} == {"loc"}
    # counts in the log, no photo id; no work dir left
    (line,) = [m for m in messages if m.startswith(f"job {job.id} (localize")]
    assert "3 photos to try, 3 tried (2 localized, 1 rejected), 1 recorded as skipped, 0 left" in line
    assert "maps: 1 candidates, 1 loaded" in line
    assert not any(pid in m for m in messages for pid in truth)
    assert not (settings.work_dir / job.id).exists()
    # nothing new to do: planning again, a restart, a server start on the same data dir
    runner.schedule("lm")
    JobRunner(store, settings, ColmapReconstructor(settings)).start()
    with TestClient(api.create_app(settings)) as client:
        assert client.get("/api/landmarks/lm/poses").json() == body
        assert client.get(f"/api/jobs/{job.id}").status_code == 404
    assert len([j for j in store.list_jobs("lm") if j.kind == JOB_LOCALIZE]) == 1
    # withdrawn: the record and the row go
    store.withdraw_location_consent(names["held"][0])
    assert store.get_localization(names["held"][0]) is None
    assert pp.public_id(names["held"][0], salt) not in {r["id"] for r in api.landmark_poses(store, settings, lm,
                                                                                             salt)["photos"]}
