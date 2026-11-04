"""crowd3d.localize, the pure part (D30): which photos and maps, pairs, the gates, the choice, the frame, the records."""

from __future__ import annotations

import math
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
