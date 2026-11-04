"""crowd3d.photoposes: which pose each photo gets, its uncertainty, and the rounded row other people see (D30)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from crowd3d import photoposes as pp
from crowd3d.anchor import GeoAnchor, quat_xyzw
from crowd3d.api import camera_estimates
from crowd3d.consent import public_id
from crowd3d.geo import LocalENU
from crowd3d.records import (
    JOB_FACE,
    JOB_FULL,
    LOC_ACCEPTED,
    LOC_REJECTED,
    LOCATION_CONSENT_V1,
    PHOTO_ACCEPTED,
    PHOTO_REJECTED,
    LocRecord,
    ModelRecord,
    PhotoRecord,
)

LAT, LON = 43.773, 11.255
ENU = LocalENU(LAT, LON)
SALT = "s" * 64


def photo(pid, e=None, n=None, heading=None, *, status=PHOTO_ACCEPTED, acc=None, ref=None, consent=None, f35=None):
    lat = lon = None
    if e is not None:
        lat, lon, _ = ENU.to_geodetic(e, n, 0.0)
    return PhotoRecord(id=pid, landmark_id="lm", user_id="u1", status=status, uploaded_at=1.0,
                       reasons=[] if status == PHOTO_ACCEPTED else ["too_blurry"], lat=lat, lon=lon,
                       gps_accuracy_m=acc, heading_deg=heading, heading_ref=ref, location_consent=consent,
                       focal_35mm=f35)


def cam(e, n, heading, flags=(), **kw):
    return {"enu": [e, n, 1.6], "heading_deg": heading, "flags": list(flags), "focal_ratio": 0.8, **kw}


def model(mid, cams, *, kind=JOB_FACE, created=1.0, published=True, **meta):
    base = {"published": published, "cameras": cams, "georegistered": True, "alignment_rmse_m": 4.0,
            "yaw_sigma_deg": 3.0, "message": "georegistered from 9/9 GPS fixes"}
    base.update(meta)
    return ModelRecord(id=mid, landmark_id="lm", job_id="j" + mid, kind=kind, sector=0 if kind == JOB_FACE else None,
                       ply_path=f"models/lm/{mid}.ply", num_photos=len(cams), meta=base, created_at=created)


PRIOR_MSG = ("registered 4/6 photos, 1065 points; not georegistered: 3/4 consistent GPS fixes spread over 6.9 m cannot"
             " fix the scale; landmark-direction yaw, approximate scale")
SNAP = {"rotation_deg": 90.0, "shift_m": [1.0, 2.0], "scale": 1.0, "accepted": True}


def poses_by_id(*args, **kw):
    return {p.photo_id: p for p in pp.photo_poses(*args, **kw)}


# ---------------------------------------------------------------- which source wins


def test_published_cameras_are_the_camera_estimates_of_models_whose_frame_is_fixed():
    models = [
        model("old", {"a": cam(1, 1, 10), "b": cam(2, 2, 20), "c": cam(3, 3, 30)}, created=1.0),
        model("new", {"a": cam(5, 5, 50), "b": cam(6, 6, 60, flags=["gps_and_compass"])}, created=2.0),
        model("full", {"c": cam(7, 7, 70)}, kind=JOB_FULL, created=0.5),
        model("rejected", {"a": cam(9, 9, 90), "d": cam(9, 9, 90)}, created=3.0, published=False),
    ]
    picked = pp.published_cameras(models)
    assert {k: v[1] for k, v in picked.items()} == camera_estimates(models)
    assert picked["a"][0].id == "new"          # the newest model
    assert picked["b"][0].id == "old"          # a flagged camera is no estimate: the older entry stays
    assert picked["c"][0].id == "full"         # a full model beats facade models
    assert "d" not in picked                   # QA-rejected models show nothing
    # a newer prior-only model: still a camera estimate (internal), but its frame is not fixed, so not shown
    prior = model("prior", {"a": cam(8, 8, 80), "e": cam(4, 4, 40)}, created=4.0, georegistered=False,
                  message=PRIOR_MSG)
    picked = pp.published_cameras([*models, prior])
    assert camera_estimates([*models, prior])["a"]["enu"][0] == 8
    assert picked["a"][0].id == "new" and "e" not in picked
    # ... unless a footprint snap was accepted for it: the snap measured it against the OSM walls
    assert pp.published_cameras([*models, prior], {"prior": SNAP})["e"][0].id == "prior"


def test_frame_ok_reads_the_anchor_level_then_the_legacy_flag():
    def m(**meta):
        return model("m", {}, **meta)

    assert pp.anchor_level({"anchor": {"level": "prior_only"}, "georegistered": True}) == "prior_only"
    assert pp.anchor_level({"anchor": {"level": "full"}}) == "full"
    assert pp.anchor_level({"georegistered": True}) == "full"
    assert pp.anchor_level({"georegistered": False}) == "prior_only"
    assert pp.anchor_level({}) is None
    assert pp.frame_ok(m(), {}) and not pp.frame_ok(m(georegistered=False), {})
    assert not pp.frame_ok(m(anchor={"level": "none"}), {})
    assert not pp.frame_ok(model("m", {}, georegistered=None), {})          # nothing says it was fitted
    assert pp.frame_ok(m(georegistered=False), {"m": SNAP})                  # an accepted snap fixes it


def test_a_refused_snap_fixes_no_frame_in_any_helper():
    """FootprintSnap.to_meta writes refused snaps too (accepted False); the stylize record keeps them (D30 C3). None of
    the helpers the localizer and the public builder use may count one as fixing a prior-only model's frame."""
    prior = model("m", {"a": cam(8, 8, 80), "f": cam(1, 1, 10, flags=["far_from_gps"])}, georegistered=False,
                  message=PRIOR_MSG)
    refused = {"m": {**SNAP, "accepted": False, "reason": "ambiguous"}}
    assert not pp.snap_accepted(refused["m"]) and pp.snap_accepted(SNAP) and not pp.snap_accepted(None)
    assert pp.snap_accepted({"rotation_deg": 0.0, "shift_m": [0, 0]})       # an old meta without the flag: accepted
    assert pp.accepted_snaps({"m": refused["m"], "n": SNAP, "o": None}) == {"n": SNAP}
    assert not pp.frame_ok(prior, refused)
    assert pp.published_cameras([prior], refused) == {}
    assert pp.flagged_photos([prior], refused) == set()
    assert {k: v[2] for k, v in pp.withheld_cameras([prior], refused).items()} == {"a": pp.W_FRAME_NOT_FIXED,
                                                                                   "f": pp.W_FRAME_NOT_FIXED}
    assert poses_by_id([photo("a", 8, 8)], [prior], LAT, LON, snaps=refused)["a"].src == pp.SRC_GPS
    # accepted: the same helpers show and flag
    assert set(pp.published_cameras([prior], {"m": SNAP})) == {"a"}
    assert pp.flagged_photos([prior], {"m": SNAP}) == {"f"}


def test_the_public_uncertainty_quotes_measured_coverage_per_source_with_its_spread_and_calibration():
    u = pp.public_uncertainty()
    assert u["sigmas"] == pp.COVERAGE_SIGMAS == 2.0 and u["basis"] == pp.COVERAGE_BASIS
    assert u["measured_coverage"]["loc"] is None                             # no localizer measured yet
    expected = {"sfm": ((146, 165), 4, (1, 10), (13, 13), False),
                "exif": ((23, 28), 3, (1, 2), (1, 1), True)}                 # the EXIF sigma was calibrated on these
    for src, ((within, n), landmarks, lowest, highest, calibrated_here) in expected.items():
        c = u["measured_coverage"][src]
        assert (c["within"], c["n"], c["runs"], c["landmarks"]) == (within, n, 2, landmarks)
        assert c["coverage"] == round(within / n, 3) and c["rule_calibrated_here"] is calibrated_here
        for key, (w, k) in (("lowest", lowest), ("highest", highest)):
            assert (c[key]["within"], c[key]["n"], c[key]["coverage"]) == (w, k, round(w / k, 3))
        assert c["lowest"]["coverage"] <= c["coverage"] <= c["highest"]["coverage"]
        assert c["lowest"]["n"] + c["highest"]["n"] <= n
    # a copy: changing the served block leaves the pin alone
    u["measured_coverage"]["sfm"]["lowest"]["within"] = 99
    assert pp.MEASURED_COVERAGE["sfm"]["lowest"]["within"] == 1
    assert pp.public_rounding() == {"position_m": 1, "angle_deg": 1, "fov_deg": 5, "aspect_decimals": 2,
                                    "sp_rounding_m": 0.71, "sy_rounding_deg": 0.5}


def test_photo_poses_source_hierarchy():
    photos = [
        photo("sfm", 30, 0, 270.0),
        photo("exif", -30, 0, 90.0, ref="T", f35=26.0),
        photo("magnetic", -20, 5, 80.0, ref="M"),
        photo("untagged", -20, 5, 80.0),
        photo("gps", 20, 20),
        photo("rej", 10, 10, 0.0, status=PHOTO_REJECTED),
        photo("nogps"),
    ]
    models = [model("m1", {"sfm": cam(28.0, 1.0, 268.0)})]
    poses = poses_by_id(photos, models, LAT, LON, aspects={"exif": 4 / 3})
    assert poses["sfm"].src == "sfm" and poses["sfm"].model_id == "m1"
    assert poses["sfm"].p == [28.0, 1.0, 1.6] and poses["sfm"].yaw == 268.0 and not poses["sfm"].withheld
    # EXIF only with a true-north compass; the height is not measured, the field of view from the 35 mm focal
    ex = poses["exif"]
    assert ex.src == "exif" and ex.yaw == 90.0 and ex.p[2] is None and ex.pitch is None and ex.roll is None
    assert math.hypot(ex.p[0] + 30, ex.p[1]) < 0.01
    assert ex.hfov == pytest.approx(pp.hfov_from_35mm(26.0, 4 / 3)) and ex.aspect == pytest.approx(4 / 3)
    assert (ex.sp, ex.sy) == (40.0, 30.0)
    # a magnetic or untagged compass: no pose (position only), the heading kept as a withheld EXIF pose
    for k in ("magnetic", "untagged"):
        g = poses[k]
        assert g.src == "gps" and g.yaw is None and not g.determined and g.reason == pp.R_HEADING_NOT_TRUE_NORTH
        assert [(w["src"], w["reason"], w["yaw"]) for w in g.withheld] == [("exif", "heading_not_true_north", 80.0)]
        assert g.withheld[0]["p"][2] is None
    # GPS without a heading: a position only, NOT determined (the camera is never assumed to face the building)
    assert poses["gps"].src == "gps" and poses["gps"].yaw is None and not poses["gps"].determined
    assert poses["gps"].reason == pp.R_GPS_WITHOUT_HEADING and not poses["gps"].withheld
    assert poses["rej"].src is None and poses["rej"].reason == pp.R_REJECTED and poses["rej"].reasons == ["too_blurry"]
    assert poses["nogps"].src is None and poses["nogps"].reason == pp.R_NO_GPS
    assert [p.determined for p in poses.values()] == [True, True, False, False, False, False, False]


def test_heading_ref_comes_from_the_record_then_the_old_store_mapping():
    old = photo("p", 0, 30, 180.0)                         # stored before the store kept the tag
    assert pp.photo_poses([old], [], LAT, LON)[0].src == "gps"
    assert pp.photo_poses([old], [], LAT, LON, heading_refs={"p": "T"})[0].src == "exif"
    # the record's own tag wins over the mapping
    assert pp.photo_poses([photo("p", 0, 30, 180.0, ref="M")], [], LAT, LON, heading_refs={"p": "T"})[0].src == "gps"
    assert pp.photo_poses([photo("p", 0, 30, 180.0, ref="T")], [], LAT, LON, heading_refs={"p": "M"})[0].src == "exif"


def test_prior_only_unsnapped_cameras_are_withheld_not_shown():
    # mdx-two-v1 Aachen's only published model: not georegistered (3 of 4 GPS fixes over 6.9 m), landmark-direction
    # yaw, guessed scale; its 4 cameras sat 23.6-24.8 m from the reference
    cams = {"a": cam(30.0, 40.0, 200.0), "b": cam(33.0, 41.0, 205.0), "c": cam(29.0, 44.0, 199.0)}
    prior = model("m1", cams, georegistered=False, alignment_rmse_m=3.0, yaw_sigma_deg=12.9, message=PRIOR_MSG)
    photos = [photo("a", 30.0, 40.0, 210.0, ref="T"), photo("b", 33.0, 41.0, None), photo("c", 29.0, 44.0, 10.0,
                                                                                           ref="M")]
    poses = poses_by_id(photos, [prior], LAT, LON)
    # a: falls through to its own EXIF (true north); b: GPS only; c: GPS only (magnetic compass withheld too)
    assert poses["a"].src == "exif" and poses["a"].yaw == 210.0
    assert poses["b"].src == "gps" and not poses["b"].determined
    assert poses["c"].src == "gps" and [w["src"] for w in poses["c"].withheld] == ["sfm", "exif"]
    for k in "abc":
        w = poses[k].withheld[0]
        assert (w["src"], w["reason"], w["model_id"]) == ("sfm", "frame_not_fixed", "m1")
        assert w["p"] == cams[k]["enu"] and w["yaw"] == cams[k]["heading_deg"]
    # the anchor's level decides when the model has one
    anchored = model("m1", cams, anchor={"level": "prior_only"})
    assert poses_by_id(photos, [anchored], LAT, LON)["b"].src == "gps"
    # the same cameras in a GPS-fitted model: shown
    fitted = model("m1", cams, alignment_rmse_m=3.0, yaw_sigma_deg=12.9)
    assert {p.src for p in pp.photo_poses(photos, [fitted], LAT, LON)} == {"sfm"}
    # a newer prior-only model over an older fitted one: the older camera is shown, the newer one kept as withheld
    older = model("m0", {"b": cam(34.0, 42.0, 206.0)}, created=0.5)
    b = poses_by_id(photos, [prior, older], LAT, LON)["b"]
    assert b.src == "sfm" and b.model_id == "m0" and b.p == [34.0, 42.0, 1.6]
    assert [(w["src"], w["reason"], w["model_id"]) for w in b.withheld] == [("sfm", "frame_not_fixed", "m1")]
    # an accepted footprint snap fixes the frame: shown, moved with the snap, inside its cap
    snapped = poses_by_id(photos, [prior], LAT, LON, snaps={"m1": SNAP})["b"]
    assert snapped.src == "sfm" and snapped.snapped and not snapped.withheld
    assert np.allclose(snapped.p, [-41.0 + 1.0, 33.0 + 2.0, 1.6])
    assert snapped.sp <= 2 * pp.SNAPPED_SIGMA_CAP and snapped.sy <= 2 * pp.SNAPPED_SIGMA_CAP
    # a refused snap fixes nothing
    refused = poses_by_id(photos, [prior], LAT, LON, snaps={"m1": {**SNAP, "accepted": False}})["b"]
    assert refused.src == "gps"


def test_flagged_cameras_are_withheld():
    models = [model("m1", {"f": cam(10.0, 0.0, 0.0, flags=["far_from_gps"]), "ok": cam(0.0, 10.0, 90.0)})]
    photos = [photo("f", 80.0, 0.0, 5.0, ref="T"), photo("g", 80.0, 0.0, None)]
    models[0].meta["cameras"]["g"] = cam(12.0, 0.0, 3.0, flags=["far_from_gps"])
    poses = poses_by_id(photos, models, LAT, LON)
    assert poses["f"].src == "exif" and poses["g"].src == "gps"
    for k in "fg":
        assert [(w["src"], w["reason"]) for w in poses[k].withheld] == [("sfm", "camera_flagged")]
    assert pp.flagged_photos(models) == {"f", "g"}
    # an unflagged camera of the photo in an older published model is shown instead; the newer flagged one, which
    # the photo would have shown without the rule, stays on record as withheld
    other = model("m0", {"g": cam(13.0, 0.0, 4.0)}, created=0.5)
    g = poses_by_id(photos, [*models, other], LAT, LON)["g"]
    assert g.src == "sfm" and g.model_id == "m0"
    assert [(w["src"], w["reason"], w["model_id"]) for w in g.withheld] == [("sfm", "camera_flagged", "m1")]
    # a flagged camera in an older model than the shown one: nothing was withheld from what the photo shows
    newer = model("m2", {"g": cam(13.0, 0.0, 4.0)}, created=2.0)
    g = poses_by_id(photos, [*models, newer], LAT, LON)["g"]
    assert g.src == "sfm" and g.model_id == "m2" and not g.withheld


# ---------------------------------------------------------------- the localization hook (dormant)


def loc(model_id, enu, heading, status="accepted", **pose):
    return {"status": status, "model_id": model_id, "pose": {"enu": enu, "heading_deg": heading, **pose}}


def test_localized_poses_rank_between_sfm_and_exif():
    full = model("full", {"s": cam(5.0, -30.0, 0.0)}, kind=JOB_FULL, alignment_rmse_m=4.0, yaw_sigma_deg=3.0)
    photos = [photo("s", 5.0, -30.0, 0.0, ref="T"), photo("l", -30.0, 0.0, 90.0, ref="T", f35=28.0),
              photo("e", 0.0, 40.0, 180.0, ref="T")]
    locs = {"s": loc("full", [6.0, -31.0, 1.5], 3.0), "e": loc("full", [0.0, 41.0, 1.5], 182.0, "rejected"),
            "l": loc("full", [-31.0, 0.5, 1.7], 92.0, pitch_deg=8.0, roll_deg=-1.0, hfov_deg=63.0, aspect=1.5)}
    poses = poses_by_id(photos, [full], LAT, LON, localizations=locs, aspects={"l": 4 / 3})
    assert poses["s"].src == "sfm"                       # sfm beats loc
    lp = poses["l"]
    assert lp.src == "loc" and lp.model_id == "full" and lp.p == [-31.0, 0.5, 1.7] and lp.yaw == 92.0
    assert (lp.pitch, lp.roll, lp.hfov, lp.aspect) == (8.0, -1.0, 63.0, 1.5)
    assert poses["e"].src == "exif"                      # a rejected localization gives nothing
    # sigma: the model's own sfm sigma at that point with the localizer's relative error in quadrature, never less
    sp, sy = pp.sfm_sigma(full.meta, [-31.0, 0.5, 1.7])
    assert lp.sp == pytest.approx(2 * math.hypot(sp, pp.LOC_REL_SIGMA_M)) and lp.sp > 2 * sp
    assert lp.sy == pytest.approx(2 * math.hypot(sy, pp.LOC_REL_SIGMA_DEG)) and lp.sy > 2 * sy
    # a record object reads the same; without hfov_deg the 35 mm focal gives it
    rec = SimpleNamespace(status="accepted", model_id="full", pose={"enu": [-31.0, 0.5, 1.7], "heading_deg": 92.0})
    lp2 = poses_by_id(photos, [full], LAT, LON, localizations={"l": rec}, aspects={"l": 4 / 3})["l"]
    assert lp2.src == "loc" and lp2.hfov == pytest.approx(pp.hfov_from_35mm(28.0, 4 / 3)) and lp2.pitch is None
    # the store's records.LocRecord (C5) is such an object: the same pose as its JSON form
    assert pp.LOC_ACCEPTED == LOC_ACCEPTED
    stored = LocRecord("l", "lm", LOC_ACCEPTED, 1.0, model_id="full", pose=locs["l"]["pose"])
    lp3 = poses_by_id(photos, [full], LAT, LON, localizations={"l": stored}, aspects={"l": 4 / 3})["l"]
    assert (lp3.src, lp3.p, lp3.yaw, lp3.pitch, lp3.roll, lp3.hfov, lp3.sp, lp3.sy) == \
        (lp.src, lp.p, lp.yaw, lp.pitch, lp.roll, lp.hfov, lp.sp, lp.sy)
    rejected = LocRecord("l", "lm", LOC_REJECTED, 1.0, reason="too_few_matches")
    assert poses_by_id(photos, [full], LAT, LON, localizations={"l": rejected})["l"].src == "exif"
    # skipped, unknown model, unpublished model, no heading: nothing
    hidden = model("hidden", {}, published=False)
    for bad in (loc("full", [-31.0, 0.5, 1.7], 92.0, "skipped"), loc("nope", [-31.0, 0.5, 1.7], 92.0),
                loc("hidden", [-31.0, 0.5, 1.7], 92.0), loc("full", [-31.0, 0.5, 1.7], None)):
        assert poses_by_id(photos, [full, hidden], LAT, LON, localizations={"l": bad})["l"].src == "exif"


def test_localized_poses_follow_the_frame_rule_and_the_flag():
    prior = model("prior", {"x": cam(1.0, 1.0, 0.0)}, georegistered=False, message=PRIOR_MSG)
    full = model("full", {"f": cam(10.0, 0.0, 0.0, flags=["far_from_gps"])}, kind=JOB_FULL)
    photos = [photo("q", -30.0, 0.0, None), photo("f", 50.0, 0.0, None)]
    into_prior = {"q": loc("prior", [-31.0, 0.5, 1.7], 92.0)}
    q = poses_by_id(photos, [prior, full], LAT, LON, localizations=into_prior)["q"]
    assert q.src == "gps" and [(w["src"], w["reason"]) for w in q.withheld] == [("loc", "frame_not_fixed")]
    # into a snapped prior-only model: shown, moved with the snap like its cameras
    q = poses_by_id(photos, [prior, full], LAT, LON, localizations=into_prior, snaps={"prior": SNAP})["q"]
    assert q.src == "loc" and q.snapped and np.allclose(q.p, [-0.5 + 1.0, -31.0 + 2.0, 1.7])
    assert q.yaw == pytest.approx(2.0)
    # a photo whose own camera is flagged in a shown model is never localized: the localization is withheld
    f = poses_by_id(photos, [prior, full], LAT, LON, localizations={"f": loc("full", [11.0, 0.0, 1.6], 1.0)})["f"]
    assert f.src == "gps" and [(w["src"], w["reason"]) for w in f.withheld] == [("sfm", "camera_flagged"),
                                                                                ("loc", "camera_flagged")]
    # a prior-only photo localized into a fixed model replaces its withheld camera (kept in withheld)
    x = poses_by_id([photo("x", 1.0, 1.0, None)], [prior, full], LAT, LON,
                    localizations={"x": loc("full", [1.5, 1.0, 1.6], 2.0)})["x"]
    assert x.src == "loc" and [(w["src"], w["reason"]) for w in x.withheld] == [("sfm", "frame_not_fixed")]


# ---------------------------------------------------------------- pitch, roll and field of view from the model JSON


def cam_from_enu(heading_deg, pitch_deg, roll_deg):
    """cam_from_enu rows (right, down, forward) of a camera looking along heading / pitch, rolled right side up."""
    h, p, r = (math.radians(v) for v in (heading_deg, pitch_deg, roll_deg))
    f = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    lr = np.cross(f, [0.0, 0.0, 1.0])
    lr /= np.linalg.norm(lr)
    lu = np.cross(lr, f)
    right = math.cos(r) * lr + math.sin(r) * lu
    return np.array([right, np.cross(f, right), f])


def anchored_model(poses_enu, *, yaw=37.0, scale=1.7, t=(12.0, -4.0, 2.5)):
    """A model with a non-trivial GeoAnchor whose cameras carry local poses (as reconstruct.check_cameras writes)."""
    a = math.radians(yaw)
    rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    anchor = GeoAnchor("job", "full", scale, quat_xyzw(rot), list(t), [0.0, 0.0, 1.0], yaw, [0.0, 0.0, 0.0],
                       dict.fromkeys(("yaw_deg", "log_scale", "east_m", "north_m", "up_m", "tilt_deg"), 1.0),
                       dict.fromkeys(("yaw", "scale", "xy", "up", "gravity"), "gps"))
    cams = {}
    for pid, (centre, h, p, r) in poses_enu.items():
        cfe = cam_from_enu(h, p, r)
        c_local = rot.T @ (np.asarray(centre, float) - np.asarray(t)) / scale
        q_local = quat_xyzw(cfe @ rot)
        enu, _, heading = anchor.camera_enu(q_local, c_local)
        cams[pid] = {"enu": [round(float(x), 3) for x in enu], "heading_deg": round(heading, 2), "flags": [],
                     "focal_ratio": 1.2, "local": {"q": [round(x, 9) for x in q_local],
                                                   "c": [round(float(x), 6) for x in c_local]}}
    return model("m", cams, anchor=anchor.to_meta())


def test_pitch_roll_deg():
    assert pp.pitch_roll_deg([1, 0, 0], [0, 1, 0]) == pytest.approx((0.0, 0.0))          # level, looking north
    p, r = pp.pitch_roll_deg(cam_from_enu(90.0, 15.0, 0.0)[0], cam_from_enu(90.0, 15.0, 0.0)[2])
    assert (p, r) == pytest.approx((15.0, 0.0))                                           # tilted up
    p, r = pp.pitch_roll_deg(cam_from_enu(200.0, -5.0, 8.0)[0], cam_from_enu(200.0, -5.0, 8.0)[2])
    assert (p, r) == pytest.approx((-5.0, 8.0))                                           # right side raised: +
    assert pp.pitch_roll_deg([1, 0, 0], [0, 0, 1])[1] == 0.0                              # straight up: no roll


def test_sfm_pitch_roll_and_hfov_come_from_the_stored_model():
    m = anchored_model({"a": ([20.0, -30.0, 1.6], 10.0, 12.0, -3.0), "b": ([-25.0, 5.0, 1.8], 100.0, 0.0, 0.0)})
    photos = [photo("a", 20.0, -30.0, None), photo("b", -25.0, 5.0, None)]
    poses = poses_by_id(photos, [m], LAT, LON, aspects={"a": 4 / 3, "b": 0.75})
    a, b = poses["a"], poses["b"]
    assert (a.pitch, a.roll) == pytest.approx((12.0, -3.0), abs=1e-4) and a.yaw == pytest.approx(10.0, abs=0.01)
    assert (b.pitch, b.roll) == pytest.approx((0.0, 0.0), abs=1e-4)
    # the field of view: focal ratio 1.2 of the longer side; landscape uses the whole width, portrait 0.75 of it
    assert a.hfov == pytest.approx(math.degrees(2 * math.atan(0.5 / 1.2)))
    assert b.hfov == pytest.approx(math.degrees(2 * math.atan(0.375 / 1.2)))
    assert (a.aspect, b.aspect) == pytest.approx((4 / 3, 0.75))
    # no processed-copy aspect: no field of view (never a guessed 60 deg)
    assert poses_by_id(photos, [m], LAT, LON)["a"].hfov is None
    # a snap turns the camera about the vertical: pitch and roll stay
    s = poses_by_id(photos, [m], LAT, LON, snaps={"m": SNAP})["a"]
    assert (s.pitch, s.roll) == pytest.approx((12.0, -3.0), abs=1e-4) and s.yaw == pytest.approx(280.0, abs=0.01)
    # a stored pose the anchor does not give back (another anchor): no pitch / roll rather than a wrong one
    bad = json.loads(json.dumps(m.meta))
    bad["cameras"]["a"]["enu"][0] += 1.0
    assert pp.camera_pitch_roll(bad, bad["cameras"]["a"]) == (None, None)
    bad = json.loads(json.dumps(m.meta))
    bad["cameras"]["a"]["heading_deg"] += 2.0
    assert pp.camera_pitch_roll(bad, bad["cameras"]["a"]) == (None, None)
    # without a local pose or an anchor (the fake reconstructor, models from before anchors): None
    assert pp.camera_pitch_roll({}, m.meta["cameras"]["a"]) == (None, None)
    assert pp.camera_pitch_roll(m.meta, cam(1, 1, 1)) == (None, None)


def test_field_of_view_helpers():
    assert pp.hfov_from_focal_ratio(0.5, 1.5) == pytest.approx(90.0)
    assert pp.hfov_from_focal_ratio(None, 1.5) is None and pp.hfov_from_focal_ratio(0.8, None) is None
    # a 26 mm (35 mm equivalent) phone lens on a 4:3 frame: about 67 deg across
    assert pp.hfov_from_35mm(26.0, 4 / 3) == pytest.approx(67.3, abs=0.2)
    # the full frame itself: 36 mm across at 3:2
    assert pp.hfov_from_35mm(18.0, 1.5) == pytest.approx(math.degrees(2 * math.atan(18.0 / 18.0)), abs=0.3)
    assert pp.hfov_from_35mm(0.0, 1.5) is None and pp.hfov_from_35mm(26.0, None) is None


def test_processed_aspects_read_the_upright_processed_copy(tmp_path):
    from PIL import Image

    (tmp_path / "photos").mkdir()
    Image.new("RGB", (40, 30)).save(tmp_path / "photos" / "a.proc.jpg")
    Image.new("RGB", (30, 40)).save(tmp_path / "photos" / "b.proc.jpg")
    (tmp_path / "photos" / "c.proc.jpg").write_bytes(b"not a jpeg")
    photos = [PhotoRecord(id=k, landmark_id="lm", user_id="u", status=PHOTO_ACCEPTED, uploaded_at=1.0,
                          processed_path=f"photos/{k}.proc.jpg") for k in "abc"]
    photos.append(PhotoRecord(id="d", landmark_id="lm", user_id="u", status=PHOTO_ACCEPTED, uploaded_at=1.0))
    assert pp.processed_aspects(photos, tmp_path) == {"a": 4 / 3, "b": 0.75}


# ---------------------------------------------------------------- what other people see


def test_public_row_rounds_and_keeps_only_the_whitelisted_keys():
    pose = pp.PhotoPose("p1", "sfm", p=[12.49, -12.5, 1.5], yaw=359.6, pitch=-4.5, roll=179.6, hfov=61.0,
                        aspect=4 / 3, sp=7.2, sy=6.0, model_id="m1", snapped=True,
                        withheld=[{"src": "sfm", "reason": "camera_flagged", "model_id": "m0", "p": [1, 2, 3],
                                   "yaw": 4}])
    row = pp.public_row(pose, SALT)
    assert tuple(row) == pp.PUBLIC_KEYS
    assert row == {"id": public_id("p1", SALT), "src": "sfm", "p": [12, -12, 2], "yaw": 0, "pitch": -4, "roll": 180,
                   "hfov": 60, "aspect": 1.33, "sp": math.ceil(math.hypot(7.2, 0.71)), "sy": 7}
    assert row["sp"] == 8 and len(row["id"]) == 12
    # whole numbers, no exact value, no internal field anywhere in the serialized row
    text = json.dumps(row)
    for leak in ("p1", "m1", "m0", "12.49", "359.6", "snapped", "withheld", "reason", "camera_flagged"):
        assert leak not in text
    # steps: half up (12.5 -> 13), yaw mod 360, hfov to 5 deg (62.5 -> 65), roll stays in -179 .. 180
    more = pp.public_row(pp.PhotoPose("p2", "loc", p=[12.5, 0.4999, 0.0], yaw=0.49, pitch=None, roll=-179.6,
                                      hfov=62.5, aspect=None, sp=0.0, sy=0.0), SALT)
    assert more["p"] == [13, 0, 0] and more["yaw"] == 0 and more["roll"] == 180 and more["hfov"] == 65
    assert more["pitch"] is None and more["aspect"] is None and more["sp"] == 1 and more["sy"] == 1
    # an EXIF row keeps its unmeasured height and field of view empty
    exif = pp.public_row(pp.PhotoPose("p3", "exif", p=[-30.2, 4.7, None], yaw=90.4, sp=40.0, sy=30.0), SALT)
    assert exif["p"] == [-30, 5, None] and exif["hfov"] is None and exif["pitch"] is None and exif["sp"] == 41
    assert exif["sy"] == 31
    # rounding never narrows a range: the public sp / sy cover the exact range plus the rounding shift
    for sp in (0.3, 6.0, 7.99, 15.0, 40.0):
        assert pp.public_row(pp.PhotoPose("q", "sfm", p=[0, 0, 0], yaw=0, sp=sp, sy=sp), SALT)["sp"] >= math.hypot(
            sp, math.sqrt(0.5))
    # nothing but a determined pose has a row
    for p in (pp.PhotoPose("g", "gps", p=[1.0, 2.0, None], sp=60.0), pp.PhotoPose("r", None, "rejected"),
              pp.PhotoPose("x", "sfm", p=None, yaw=10.0)):
        assert pp.public_row(p, SALT) is None
    with pytest.raises(ValueError):
        pp.public_row(pose, "")


def test_public_poses_show_only_consented_photos_sorted_by_public_id():
    models = [model("m1", {"a": cam(28.0, 1.0, 268.0), "b": cam(5.0, 5.0, 10.0), "c": cam(9.0, 9.0, 20.0)})]
    photos = [
        photo("a", 30, 0, 270.0, consent=LOCATION_CONSENT_V1),
        photo("b", 5, 5, None, consent=None),                                 # an old row / no consent: hidden
        photo("c", 9, 9, None, consent="share-location-v0"),                  # an unknown consent text: hidden
        photo("e", -30, 0, 90.0, ref="T", consent=LOCATION_CONSENT_V1),
        photo("g", 0, 40, None, consent=LOCATION_CONSENT_V1),                 # GPS only: shared but never shown
        photo("r", 10, 10, 0.0, status=PHOTO_REJECTED, consent=LOCATION_CONSENT_V1),
    ]
    doc = pp.public_poses(photos, models, LAT, LON, salt=SALT)
    assert doc["counts"] == {"photos": 5, "shared": 3, "shown": 2, "sfm": 1, "loc": 0, "exif": 1,
                             "not_determined": 1}
    rows = doc["photos"]
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    assert {r["id"] for r in rows} == {public_id("a", SALT), public_id("e", SALT)}
    # every row is exactly public_row of the internal pose (what the benchmark scores)
    internal = {p.photo_id: p for p in pp.photo_poses(photos, models, LAT, LON)}
    assert rows == sorted((pp.public_row(internal[k], SALT) for k in "ae"), key=lambda r: r["id"])
    text = json.dumps(doc)
    for k in ("u1", '"a"', '"b"', '"c"', '"e"', "m1", "lm"):
        assert k not in text
    # another salt, other ids
    assert {r["id"] for r in pp.public_poses(photos, models, LAT, LON, salt="t" * 64)["photos"]}.isdisjoint(
        {r["id"] for r in rows})


# ---------------------------------------------------------------- snap and uncertainty


def test_sfm_camera_moves_with_an_accepted_snap_only():
    photos = [photo("a", 10, 0, 0.0)]
    models = [model("m1", {"a": cam(10.0, 0.0, 0.0)})]
    (pose,) = pp.photo_poses(photos, models, LAT, LON, snaps={"m1": SNAP})
    assert np.allclose(pose.p, [1.0, 12.0, 1.6], atol=1e-9)
    assert pose.yaw == pytest.approx(270.0)     # north turned 90 deg counter-clockwise looks west
    assert pose.snapped and pose.sp <= 2 * pp.SNAPPED_SIGMA_CAP and pose.sy <= 2 * pp.SNAPPED_SIGMA_CAP
    (pose,) = pp.photo_poses(photos, models, LAT, LON, snaps={"m1": {**SNAP, "accepted": False}})
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
    # the rule's constants are recorded with every result; the prior-only circle is gone with the prior-only poses
    rule = pp.uncertainty_rule()
    assert "prior_only_pos_floor_m" not in rule and rule["loc_rel_sigma_m"] == 1.5
    assert rule["public_sp_rounding_m"] == pytest.approx(math.sqrt(0.5), abs=0.01)


def test_exif_sigma_by_heading_reference():
    assert pp.exif_sigma(None, "T") == (20.0, 15.0)
    assert pp.exif_sigma(None, "M") == (30.0, 30.0)
    assert pp.exif_sigma(None, None) == (30.0, 30.0)     # unknown reference: as magnetic
    assert pp.exif_sigma(25.0, "T") == (25.0, 15.0)      # the phone's own, larger error
    assert pp.exif_sigma(10.0, "T") == (20.0, 15.0)      # ... but never below the measured spread
    (pose,) = pp.photo_poses([photo("p", 0, 30, 180.0, ref="T")], [], LAT, LON)
    assert (pose.src, pose.sp, pose.sy) == ("exif", 40.0, 30.0)          # 2 sigma
    (pose,) = pp.photo_poses([photo("p", 0, 30, 180.0, ref="t ")], [], LAT, LON)
    assert pose.src == "exif"                                            # the tag as written, case and space aside
    (pose,) = pp.photo_poses([photo("p", 0, 30, 180.0, ref="M", acc=45.0)], [], LAT, LON)
    assert (pose.src, pose.sp, pose.sy) == ("gps", 90.0, None)           # magnetic: a position only


def test_exif_circle_is_wide_enough_for_the_measured_exif_errors():
    # the drawn EXIF circle (2 x 12 m = 24 m) held only 37 of the 51 EXIF-drawn reference photos' positions (p80
    # 30 m, p85 38-40 m for 'T' tags); raw EXIF of all 180 heading-tagged reference photos: 'T' p90 23 m
    # (docs/BENCHMARK.md 7.7). The circle must reach the drawn photos' p85 and the raw p90.
    sp_t = pp.COVERAGE_SIGMAS * pp.exif_sigma(None, "T")[0]
    assert sp_t >= 38.0 and sp_t >= 23.0
    # a position without a usable heading (src gps, never shown) is as wide as an untagged one
    (pose,) = pp.photo_poses([photo("p", 0, 30, None)], [], LAT, LON)
    assert pose.src == "gps" and pose.sp == pp.COVERAGE_SIGMAS * pp.EXIF_GPS_SIGMA_OTHER_M
