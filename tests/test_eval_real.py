"""Pure helpers of scripts/eval_real_landmark.py, on synthetic numbers (no COLMAP, no real photos)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from crowd3d.geo import LocalENU, angle_diff_deg, heading_from_en

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_real_landmark.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_real_landmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # dataclasses look their module up while the class is created
    spec.loader.exec_module(module)
    return module


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def look_at(center: np.ndarray, target: np.ndarray, pitch_deg: float = 0.0) -> np.ndarray:
    """cam_from_world of an upright camera (no roll) at center looking at target, tilted up by pitch."""
    fwd = target - center
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    p = math.radians(pitch_deg)
    fwd = math.cos(p) * fwd + math.sin(p) * np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.array([right, down, fwd])


def test_summarize_ignores_missing_values(ev):
    assert ev.summarize([]) == {"n": 0, "median": None, "p90": None, "mean": None, "max": None}
    s = ev.summarize([1.0, None, 3.0, float("nan"), 2.0])
    assert s["n"] == 3 and s["median"] == 2.0 and s["max"] == 3.0 and s["mean"] == 2.0


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_orientation_rotation_matches_the_pixel_transpose(ev, orientation):
    # Where does Pillow (as used by ingest.make_processed_copy) move a raw pixel?
    w, h, u, v = 5, 3, 1, 0
    img = Image.new("RGB", (w, h))
    img.putpixel((u, v), (255, 0, 0))
    img.getexif()[0x0112] = orientation
    upright = ImageOps.exif_transpose(img)
    v2, u2 = np.argwhere(np.asarray(upright)[..., 0] == 255)[0]
    # The same pixel's viewing ray, principal points at the image centres, in both cameras' axes.
    f = 4.0
    ray_raw = np.array([(u + 0.5 - w / 2) / f, (v + 0.5 - h / 2) / f, 1.0])
    ray_up = np.array([(u2 + 0.5 - upright.width / 2) / f, (v2 + 0.5 - upright.height / 2) / f, 1.0])
    np.testing.assert_allclose(ev.exif_orientation_rotation(orientation) @ ray_raw, ray_up, atol=1e-12)


def test_orientation_rotation_is_identity_for_upright_or_unknown(ev):
    for o in (None, 0, 1, 2):
        np.testing.assert_array_equal(ev.exif_orientation_rotation(o), np.eye(3))


def test_umeyama_recovers_a_similarity(ev):
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3)) * 10
    r = random_rotation(rng)
    dst = 2.5 * src @ r.T + np.array([3.0, -7.0, 1.0])
    s, r_est, t = ev.umeyama(src, dst)
    assert s == pytest.approx(2.5)
    np.testing.assert_allclose(r_est, r, atol=1e-9)
    np.testing.assert_allclose(t, [3.0, -7.0, 1.0], atol=1e-9)


def test_relative_pose_errors_are_zero_for_a_similar_copy_and_flag_a_bad_camera(ev):
    rng = np.random.default_rng(2)
    angles = np.radians(np.linspace(20, 160, 12))
    gt_c = np.stack([60 * np.cos(angles), 60 * np.sin(angles), np.full(12, 1.6)], axis=1)
    gt_r = np.array([look_at(c, np.zeros(3), pitch_deg=10) for c in gt_c])
    # The same cameras in an arbitrary SfM frame: x_sfm = s A x_enu + b.
    a, s, b = random_rotation(rng), 0.13, np.array([1.0, 2.0, 3.0])
    est_c = s * gt_c @ a.T + b
    est_r = gt_r @ a.T
    rel = ev.relative_pose_errors(est_r, est_c, gt_r, gt_c)
    assert rel["n"] == 12 and rel["sim3_scale"] == pytest.approx(1 / s)
    assert rel["position_error_m"]["max"] < 1e-6
    assert rel["rotation_error_deg"]["max"] < 1e-4 and rel["pairwise_rotation_error_deg"]["max"] < 1e-4
    assert rel["fraction_within_5deg_5m"] == 1.0

    # One camera turned 30 degrees about its own vertical axis and moved 20 m: only it is flagged.
    bad = est_r.copy()
    turn = np.array([[math.cos(0.5236), 0, -math.sin(0.5236)], [0, 1, 0], [math.sin(0.5236), 0, math.cos(0.5236)]])
    bad[4] = turn @ bad[4]
    moved = est_c.copy()
    moved[4] += s * 20.0 * a @ np.array([0.0, 1.0, 0.0])
    rel = ev.relative_pose_errors(bad, moved, gt_r, gt_c)
    per = rel["per_camera"]
    assert per["rotation_error_deg"][4] == pytest.approx(30.0, abs=3.0)
    assert per["position_error_m"][4] == pytest.approx(20.0, abs=1.0)
    others = [e for k, e in enumerate(per["position_error_m"]) if k != 4]
    assert max(others) < 0.5   # the refit without the outlier keeps the others exact
    assert rel["pairwise_rotation_error_deg"]["median"] < 1e-4
    assert rel["fraction_within_5deg_5m"] == pytest.approx(11 / 12, abs=1e-3)


def test_relative_pose_errors_need_three_cameras(ev):
    assert ev.relative_pose_errors(np.eye(3)[None].repeat(2, 0), np.zeros((2, 3)),
                                   np.eye(3)[None].repeat(2, 0), np.zeros((2, 3)))["n"] == 2


def synthetic_gt(rng: np.random.Generator, n: int = 40):
    """Cameras on the east side of a building at the ENU origin, plus an arbitrary GT SfM frame."""
    bearing = np.radians(rng.uniform(40, 140, n))
    dist = rng.uniform(40, 90, n)
    centers = np.stack([dist * np.sin(bearing), dist * np.cos(bearing), np.full(n, 1.6)], axis=1)
    centers[:4, 2] = [20.0, 35.0, 50.0, 70.0]   # a few elevated viewpoints (towers, terraces)
    rots = np.array([look_at(c, np.array([0.0, 0.0, 10.0]), pitch_deg=rng.uniform(0, 20)) for c in centers])
    a, s, b = random_rotation(rng), 0.12, np.array([0.5, -2.0, 4.0])
    return centers, rots, s * centers @ a.T + b, rots @ a.T, a


def test_fit_gt_frame_recovers_the_enu_frame_from_noisy_gps_and_compass(ev):
    rng = np.random.default_rng(5)
    centers, rots, sfm_c, sfm_r, a = synthetic_gt(rng)
    n = len(centers)
    gps = np.full((n, 2), np.nan)
    tagged = rng.choice(n, 16, replace=False)
    gps[tagged] = centers[tagged, :2] + rng.normal(0, 3.0, (16, 2))
    gps[tagged[0]] += [45.0, 0.0]   # one wild fix
    headings = np.full(n, np.nan)
    true_heading = np.array([heading_from_en(r[2, 0], r[2, 1]) for r in rots])
    headings[tagged[:12]] = true_heading[tagged[:12]] + rng.normal(0, 8.0, 12)

    transform, report = ev.fit_gt_frame(sfm_r, sfm_c, gps, headings)
    pos = ev.apply_similarity(transform, sfm_c)
    horiz = np.linalg.norm(pos[:, :2] - centers[:, :2], axis=1)
    rot = transform[:3, :3] / np.cbrt(np.linalg.det(transform[:3, :3]))
    head = [abs(angle_diff_deg(heading_from_en(*(rot @ r[2])[:2]), h)) for r, h in zip(sfm_r, true_heading)]
    assert np.median(horiz) < 3.0 and np.max(horiz) < 6.0
    assert np.median(head) < 3.0
    assert report["num_gps"] == 16 and report["num_gps_inliers"] == 15
    assert report["gps_residual_m"]["max"] > 30          # the wild fix shows up in the residuals
    assert report["num_compass_used"] == 12
    # Gravity: the true up direction (a @ z in the SfM frame) maps onto +z although some cameras are elevated.
    up = rot @ a @ np.array([0.0, 0.0, 1.0])
    assert math.degrees(math.acos(min(1.0, up[2]))) < 1.5
    assert report["camera_roll_abs_deg"]["max"] < 1.5


def test_fit_gt_frame_needs_two_usable_gps_fixes(ev):
    rng = np.random.default_rng(6)
    centers, rots, sfm_c, sfm_r, _ = synthetic_gt(rng, n=10)
    gps = np.full((10, 2), np.nan)
    gps[0] = centers[0, :2]
    with pytest.raises(ValueError):
        ev.fit_gt_frame(sfm_r, sfm_c, gps, np.full(10, np.nan))


def test_polygon_centroid_and_outline_fit(ev):
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    np.testing.assert_allclose(ev.polygon_centroid(square), [5.0, 5.0])
    np.testing.assert_allclose(ev.polygon_centroid(np.vstack([square, square[:1]])), [5.0, 5.0])
    # Points on three walls of the square, rotated by 8 degrees about its centre and shifted.
    t = np.linspace(0, 1, 40)[:, None]
    walls = np.vstack([square[0] + t * (square[1] - square[0]), square[1] + t * (square[2] - square[1]),
                       square[2] + t * (square[3] - square[2])])
    th = math.radians(8.0)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    moved = (walls - 5.0) @ r.T + 5.0 + np.array([30.0, -4.0])
    d, _ = ev.distance_to_outline(moved, square)
    assert np.median(d) > 3
    fit = ev.fit_outline(moved, square, init_shift=ev.polygon_centroid(square) - moved.mean(axis=0))
    assert fit["rotation_deg"] == pytest.approx(-8.0, abs=0.2)
    assert fit["median_distance_after_m"] < 0.05


def test_best_outline_fit_keeps_the_start_that_fits_closer(ev):
    # An elongated building (76 x 20 m) whose tallest part is at the west end, like Aachen's tower:
    # the top-structure start shifts the points 30 m along the long axis.
    rect = np.array([[-38.0, -10.0], [38.0, -10.0], [38.0, 10.0], [-38.0, 10.0]])
    t = np.linspace(0, 1, 80)[:, None]
    walls = np.vstack([rect[k] + t * (rect[(k + 1) % 4] - rect[k]) for k in range(4)])
    th = math.radians(3.0)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    moved = walls @ r.T + np.array([2.0, 1.5])
    top_shift = np.array([30.0, 0.0])
    fit = ev.best_outline_fit(moved, rect, top_shift)
    zero = ev.fit_outline(moved, rect, np.zeros(2))
    top = ev.fit_outline(moved, rect, top_shift)
    assert fit["median_distance_after_m"] == min(zero["median_distance_after_m"], top["median_distance_after_m"])
    assert fit["rotation_deg"] == pytest.approx(-3.0, abs=0.3)
    assert fit["median_distance_after_m"] < 0.05
    assert fit["start"] in ("zero", "top_structure") and set(fit["other_start"]) == {
        "rotation_deg", "shift_m", "median_distance_after_m"}


def test_distance_to_outline_measures_to_the_nearest_edge(ev):
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    d, closest = ev.distance_to_outline(np.array([[5.0, -2.0], [5.0, 5.0], [13.0, 14.0]]), square)
    np.testing.assert_allclose(d, [2.0, 5.0, 5.0])
    np.testing.assert_allclose(closest[0], [5.0, 0.0])


def test_outputs_must_stay_outside_the_repository(ev, tmp_path):
    with pytest.raises(SystemExit):
        ev.check_outside_repo(ev.REPO_ROOT / "runs" / "x", "--out")
    ev.check_outside_repo(ev.REPO_ROOT / "data" / "eval", "--out")
    ev.check_outside_repo(tmp_path / "run", "--out")


# ---------------------------------------------------------------- crowd protocol helpers


def crowd_photo(ev, key: str, author: str | None, captured_at: str | None):
    from crowd3d.exif import PhotoMeta

    return ev.CrowdPhoto(key=key, path=Path(key), author=author,
                         meta=PhotoMeta(640, 480, 1.0, 2.0, captured_at=captured_at))


def test_pseudo_user_is_the_author_or_one_unknown_user_per_day(ev):
    assert ev.pseudo_user("  Jane   Doe ", "2022:08:31 10:00:00") == "Jane Doe"
    assert ev.pseudo_user(None, "2022:08:31 10:00:00") == "unknown-2022-08-31"
    assert ev.pseudo_user("", None) == "unknown-undated"
    assert len(ev.pseudo_user("x" * 200, None)) == ev.MAX_USER_ID_CHARS


def test_plan_arrival_groups_author_days_in_the_order_the_sessions_ended(ev):
    photos = [
        crowd_photo(ev, "a1", "Ann", "2022:08:31 16:00:00"),
        crowd_photo(ev, "b1", "Bob", "2022:08:31 09:00:00"),
        crowd_photo(ev, "a0", "Ann", "2022:08:31 08:00:00"),
        crowd_photo(ev, "b2", "Bob", "2022:08:31 10:00:00"),
        crowd_photo(ev, "c1", "Ann", "2019:01:02 12:00:00"),
        crowd_photo(ev, "u1", None, "2020:05:05 12:00:00"),
        crowd_photo(ev, "z9", "Zed", None),
    ]
    reqs = ev.plan_arrival(photos, "session")
    assert [[p.key for p, _ in r.items] for r in reqs] == [["c1"], ["u1"], ["b1", "b2"], ["a0", "a1"], ["z9"]]
    assert [r.items[0][1] for r in reqs] == ["Ann", "unknown-2020-05-05", "Bob", "Ann", "Zed"]
    # A long session is split into requests of at most max_per_request photos, still in capture order.
    many = [crowd_photo(ev, f"p{k:02d}", "Ann", f"2022:08:31 10:{k:02d}:00") for k in range(7)]
    split = ev.plan_arrival(many, "session", max_per_request=3)
    assert [len(r.items) for r in split] == [3, 3, 1] and split[0].items[0][0].key == "p00"
    assert [r.items[0][0].key for r in ev.plan_arrival(photos, "photo")] == ["c1", "u1", "a0", "b1", "b2", "a1", "z9"]
    single = ev.plan_arrival(photos, "single")
    assert len(single) == 1 and len(single[0].items) == 7
    with pytest.raises(ValueError):
        ev.plan_arrival(photos, "random")


def test_read_ply_points_round_trips_write_ply(ev, tmp_path):
    from crowd3d.reconstruct import write_ply

    pts = [(1.0, 2.0, 3.0), (-4.5, 0.25, 7.125)]
    write_ply(tmp_path / "m.ply", pts, [(1, 2, 3), (4, 5, 6)])
    np.testing.assert_allclose(ev.read_ply_points(tmp_path / "m.ply"), pts)
    write_ply(tmp_path / "e.ply", [], [])
    assert ev.read_ply_points(tmp_path / "e.ply").shape == (0, 3)


def square_walls(side: float = 20.0, height: float = 12.0, per_edge: int = 30) -> np.ndarray:
    """Points on the four walls of a square building centred at the origin, 1 m to height."""
    half = side / 2
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    t = np.linspace(0, 1, per_edge, endpoint=False)[:, None]
    xy = np.vstack([corners[k] + t * (corners[(k + 1) % 4] - corners[k]) for k in range(4)])
    zs = np.linspace(1.0, height, 6)
    return np.array([[x, y, z] for x, y in xy for z in zs]), corners


def test_wall_points_keep_the_height_band_near_the_outline(ev):
    walls, outline = square_walls()
    far = np.array([[200.0, 0.0, 5.0], [0.0, 0.0, 0.5], [0.0, 0.0, 30.0]])   # far away, ground, spire
    sel, shift = ev.wall_points(np.vstack([walls, far]), outline)
    band = (walls[:, 2] >= ev.FOOTPRINT_WALL_Z_M[0]) & (walls[:, 2] <= ev.FOOTPRINT_WALL_Z_M[1])
    assert len(sel) == band.sum()
    np.testing.assert_allclose(shift, [0.0, 0.0], atol=1e-9)   # the spire is on the centroid


def test_outline_check_measures_the_correction_a_model_needs(ev):
    walls, outline = square_walls()
    th = math.radians(6.0)
    r = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    moved = walls @ r.T + np.array([3.0, -2.0, 0.0])
    check = ev.outline_check(moved, outline)
    fit = check["fit"]
    assert fit["rotation_deg"] == pytest.approx(-6.0, abs=0.2)
    assert fit["shift_m"] == pytest.approx(math.hypot(3.0, 2.0), abs=0.2)
    assert fit["median_distance_after_m"] < 0.05 and not fit["ambiguous"]
    assert set(fit["alternatives"]) == {"zero", "top_structure"} - {fit["start"]}
    assert ev.outline_check(moved[:20], outline)["fit"] is None


def test_pick_outline_fit_reports_the_most_plausible_of_equally_good_fits(ev):
    def fit(rot, tx, resid):
        return {"rotation_deg": rot, "translation_m": [tx, 0.0], "shift_m": abs(tx), "median_distance_after_m": resid}

    # An octagon facade fits the next face 45 degrees on about as well: the smaller rotation is reported.
    picked = ev.pick_outline_fit({"zero": fit(-13.2, 6.5, 0.36), "top_structure": fit(78.2, 34.3, 0.21)})
    assert picked["start"] == "zero" and picked["rotation_deg"] == -13.2 and picked["ambiguous"]
    # Clearly worse fits do not count: Aachen's tower start lands 1.4x as far off but is still tied ...
    picked = ev.pick_outline_fit({"zero": fit(2.7, 6.8, 1.01), "top_structure": fit(-17.9, 18.4, 1.43)})
    assert picked["start"] == "zero"
    # ... while a start that did not converge (3x the distance) is ignored.
    picked = ev.pick_outline_fit({"zero": fit(1.0, 3.0, 0.90), "top_structure": fit(1.5, 9.0, 0.28)})
    assert picked["start"] == "top_structure" and not picked["ambiguous"]
    assert picked["alternatives"]["zero"]["median_distance_after_m"] == 0.90
    # Equal rotations (within 2 deg): the smaller shift, e.g. a corner facade slid along the box.
    picked = ev.pick_outline_fit({"zero": fit(-0.08, 1.44, 0.03), "top_structure": fit(-0.06, 12.56, 0.03)})
    assert picked["start"] == "zero" and picked["shift_m"] == 1.44 and picked["ambiguous"]
    picked = ev.pick_outline_fit({"zero": fit(1.0, 3.0, 0.30), "top_structure": fit(1.0, 3.5, 0.30)})
    assert picked["start"] == "zero" and not picked["ambiguous"]


def test_qa_reason_codes_cover_the_quality_gate(ev):
    from crowd3d.config import Settings
    from crowd3d.qa import assess
    from crowd3d.reconstruct import ReconstructionResult
    from crowd3d.records import JOB_FULL, Landmark, PhotoRecord

    photos = [PhotoRecord(id=f"p{i}", landmark_id="lm", user_id="u", status="accepted", uploaded_at=0.0,
                          heading_deg=0.0, bearing_deg=0.0, sector=0) for i in range(10)]
    cams = {f"p{i}": {"enu": [500.0, 0.0, 1.6], "heading_deg": 180.0, "compass_offset_deg": 180.0,
                      "flags": ["gps_and_compass"]} for i in range(4)}
    result = ReconstructionResult(ok=True, message="", num_input=10, num_registered=4, num_points=10,
                                  mean_reprojection_error_px=9.0, cameras=cams, points_median_en=[400.0, 0.0])
    report = assess(result, JOB_FULL, photos, Landmark("lm", "LM", 1.0, 2.0, 150.0, 0.0), Settings())
    codes = [ev.qa_reason_code(r) for r in report.reasons]
    assert "other" not in codes
    assert set(codes) == {"registered_fraction", "points", "reprojection", "twins", "flagged", "compass",
                          "centre_offset", "coverage"}
    assert ev.qa_reason_code("something new") == "other"


def test_gt_model_check_counts_gross_cameras_and_what_users_see(ev):
    rng = np.random.default_rng(8)
    centers, rots, sfm_c, sfm_r, _ = synthetic_gt(rng, n=8)
    gt = {f"k{i}": (sfm_r[i], sfm_c[i]) for i in range(8)}          # reference in an arbitrary SfM frame
    est_rots = {f"p{i}": rots[i] for i in range(8)}
    est_rots["p3"] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ rots[3]   # turned 90 deg
    cameras = {f"p{i}": {"enu": centers[i].tolist(), "flags": ["gps_and_compass"] if i == 5 else []}
               for i in range(8)}
    cameras["p5"]["enu"] = (centers[5] + [40.0, 0.0, 0.0]).tolist()        # 40 m off and flagged
    photo_key = {f"p{i}": f"k{i}" for i in range(8)} | {"p9": "unknown"}
    out = ev.gt_model_check(cameras, est_rots, gt, photo_key, published=True)
    assert out["n"] == 8 and out["gross"] == 2 and out["gross_keys"] == ["k3", "k5"]
    assert out["gross_flagged"] == 1 and out["gross_shown"] == 1
    assert out["per_camera"]["k0"]["position_error_m"] < 0.01          # metres of the model's (ENU) frame
    assert out["per_camera"]["k3"]["rotation_error_deg"] > 75     # 90 deg, less what it pulls the global alignment
    assert ev.gt_model_check(cameras, est_rots, gt, photo_key, published=False)["gross_shown"] == 0
    positions_only = ev.gt_model_check(cameras, {}, gt, photo_key, published=True)
    assert positions_only["gross_keys"] == ["k5"] and positions_only["per_camera"]["k3"]["rotation_error_deg"] is None
    lone = ev.gt_model_check({"p0": cameras["p0"]}, est_rots, gt, photo_key, True)
    assert (lone["n"], lone["n_with_reference"]) == (0, 1) and lone["rotation_error_deg"]["n"] == 0
    # Two reference frames (components): each is aligned on its own, never mixed.
    two = {k: (r, c * (1.0 if i < 4 else 7.0) + (0 if i < 4 else 100), "0" if i < 4 else "1")
           for i, (k, (r, c)) in enumerate(gt.items())}
    split = ev.gt_model_check(cameras, est_rots, two, photo_key, published=True)
    assert set(split["frames"]) == {"0", "1"} and split["n"] == 8
    assert split["per_camera"]["k0"]["position_error_m"] < 0.01 and split["per_camera"]["k6"]["frame"] == "1"


def test_synthetic_gt_poses_look_along_the_true_heading(ev):
    manifest = {"photos": [{"file": "a.jpg", "true_heading_deg": 90.0, "true_pitch_deg": 10.0,
                            "true_enu": [-20.0, 0.0, 1.6]}]}
    rot, center = (np.asarray(v) for v in ev.synthetic_gt_poses(manifest)["a.jpg"].values())
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    assert heading_from_en(rot[2, 0], rot[2, 1]) == pytest.approx(90.0)     # forward row looks east
    assert rot[1, 2] < 0                                                    # image y (down) points down
    np.testing.assert_allclose(center, [-20.0, 0.0, 1.6])


def test_run_crowd_end_to_end_with_the_fake_reconstructor(ev, synthetic_dataset, tmp_path):
    """The whole crowd protocol on the shared synthetic set (2 sectors + one photo of each invalid kind)."""
    poses = ev.synthetic_gt_poses(synthetic_dataset.manifest)
    photos = [ev.CrowdPhoto(key=p["file"], path=synthetic_dataset.path / p["file"], author=p["user_id"])
              for p in synthetic_dataset.manifest["photos"]]
    b = ev.CrowdBuilding("synthetic", "Synthetic", synthetic_dataset.lat, synthetic_dataset.lon, 150.0, photos,
                         gt={"kind": "poses", "poses": poses}, upload_all=False)
    summary = ev.run_crowd(b, tmp_path / "run", reconstructor="fake")
    m = summary["metrics"]
    assert set(m) == set(ev.METRIC_BY_NAME)
    n_gps = sum(p["exif_lat"] is not None for p in synthetic_dataset.manifest["photos"])
    assert m["photos_listed"] == len(photos) and m["photos_uploaded"] == n_gps     # the no-GPS photo is not uploaded
    good = len(synthetic_dataset.entries("good"))
    assert m["photos_accepted"] == good and summary["funnel"]["no_gps_in_file"] == 1
    assert set(summary["funnel"]["rejected_by_reason"]) == {"too_blurry", "too_far", "facing_away", "duplicate",
                                                             "too_small"}
    assert m["sectors_face_ready"] == 2 and m["facades_published"] == 2 and m["models_qa_rejected"] == 0
    assert m["registration_rate"] == 1.0 and m["estimate_coverage"] == 1.0
    # Facade jobs reach into the neighbouring sector, so a photo can count in both facade models.
    shown = [e for e in summary["models"] if e["latest_published"]]
    assert m["gt_cameras"] == sum(e["gt"]["n"] for e in shown) >= good
    assert m["gt_gross_cameras"] == 0                                 # the fake model sits at the GPS fixes
    assert m["outline_models"] == 0 and m["outline_shift_m"] is None  # no outline given
    for name in ("summary.json", "summary.md", "topdown.png"):
        assert (tmp_path / "run" / name).is_file()
    assert summary["effective_params"]["reconstructor"] == "fake"
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["metrics"] == m
    # schema 7: the run's cost, the versions of its targets and its geographic error against the truth (the fake
    # keeps no orientations: camera error only; no outline: no coverage)
    assert m["cpu_time_s"] > 0 and m["jobs_cpu_s"] >= 0 and m["peak_rss_jobs_mb"] >= m["baseline_rss_mb"] > 0
    assert m["request_latency_p50_s"] is not None and m["photo_time_to_pose_p50_s"] is not None
    assert m["disk_run_mb"] > 0 and m["stale_overwrites"] == 0
    assert all("time_to_pose_s" in r for r in summary["photos"] if r["accepted"])
    assert m["geo_models_checked"] == 2 and m["geo_camera_error_median_m"] < 5.0
    assert m["geo_yaw_error_median_deg"] is None and m["wrong_face_cameras_shown"] == 0 and m["calib_yaw_n"] == 0
    assert m["cov_claimed_m"] is None and all(m[k] is None for k in m if k.startswith("struct_"))

    everything = ev.run_crowd(ev.CrowdBuilding("s2", "S2", b.lat, b.lon, 150.0, photos, upload_all=True),
                              tmp_path / "all", reconstructor="fake", arrival="single")
    assert everything["metrics"]["photos_uploaded"] == len(photos)
    assert "no_gps" in everything["funnel"]["rejected_by_reason"] and everything["metrics"]["gt_cameras"] is None


# ---------------------------------------------------------------- stylized models (style_* metrics)


def rect_walls(w: float = 40.0, d: float = 20.0, step: float = 0.5,
               edges=(0, 1, 2, 3)) -> tuple[np.ndarray, np.ndarray]:
    """Points 2-14 m high on chosen walls of a w x d rectangle centred at the origin, and the outline."""
    corners = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    pts = []
    for k in edges:
        a, b = corners[k], corners[(k + 1) % 4]
        n = int(np.linalg.norm(b - a) / step)
        for t in np.linspace(0.05, 0.95, n):
            for z in (3.0, 6.0, 9.0, 12.0):
                pts.append([*(a + t * (b - a)), z])
    return np.array(pts), corners


def moved(pts: np.ndarray, deg: float, shift) -> np.ndarray:
    th = math.radians(deg)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = pts.copy()
    out[:, :2] = pts[:, :2] @ r.T + np.asarray(shift, float)
    return out


def test_snap_candidate_agrees_with_the_snap_and_shows_what_its_window_hid(ev):
    """snap_candidate is the unconstrained ICP (the diagnostic for aliases); inside the yaw window it lands where
    the windowed snap does, outside it keeps the fit the snap refuses."""
    from crowd3d import stylize as st

    pts, outline = rect_walls(edges=(0, 1))
    off = moved(pts, 4.0, (2.0, -1.5))
    # the GPS fit fixes the scale (two faces alone cannot); a reported 6.7 deg: a +-20 deg window
    tight = st.SnapPrior(6.7, scale_sigma=0.01)
    snap, cand = st.snap_to_footprint(off, outline, prior=tight), ev.snap_candidate(off, outline)
    assert snap.accepted and cand["rotation_deg"] == pytest.approx(snap.rotation_deg, abs=0.1)
    assert cand["shift_m"] == pytest.approx(float(np.linalg.norm(snap.shift)), abs=0.1)
    # A model 30 deg off: beyond the yaw window (+-20 deg) the snap is refused and reset to the identity,
    # the candidate keeps the fit.
    far = moved(pts, 30.0, (0.0, 0.0))
    snap = st.snap_to_footprint(far, outline, prior=tight)
    cand = ev.snap_candidate(far, outline)
    assert not snap.accepted and snap.rotation_deg == 0.0 and "window" in snap.reason
    assert cand["median_after_m"] < 0.2 and abs(cand["rotation_deg"]) > 20.0
    assert ev.snap_candidate(pts[:10], outline) is None


def test_reference_check_tells_a_right_snap_from_a_wrong_one(ev):
    ref_xy = {f"k{i}": np.array([60.0 * math.cos(a), 60.0 * math.sin(a)]) for i, a in
              enumerate(np.radians([200, 230, 260, 290]))}
    reference = {"cameras": {k: {"xy": v, "component": "c", "verified": True} for k, v in ref_xy.items()},
                 "components": {"c": {"verified": True}}}
    photo_key = {f"p{i}": f"k{i}" for i in range(4)}
    off = {f"p{i}": {"enu": [*moved(np.array([[*ref_xy[f'k{i}'], 1.6]]), 8.0, (4.0, 3.0))[0]]} for i in range(4)}
    anchor = np.array([0.0, -10.0])
    # The snap undoes exactly what the georegistration got wrong: right.
    th = math.radians(-8.0)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    undo = {"rotation_deg": -8.0, "shift_m": list(-r @ np.array([4.0, 3.0])), "scale": 1.0, "accepted": True}
    good = ev.reference_check(off, photo_key, reference, undo, anchor)
    assert good["verdict"] == "right" and good["error_after_m"]["max"] < 1e-6
    assert good["correction_before"]["rotation_deg"] == pytest.approx(-8.0, abs=1e-6)
    assert good["correction_before"]["shift_m"] == pytest.approx(5.0, abs=1e-6)
    # A snap that slid the model 6 m along its wall: wrong, though every wall may look fitted.
    slide = dict(undo, shift_m=list(np.asarray(undo["shift_m"]) + [6.0, 0.0]))
    bad = ev.reference_check(off, photo_key, reference, slide, anchor)
    assert bad["verdict"] == "wrong" and bad["correction_after"]["at_walls_m"] == pytest.approx(6.0, abs=1e-6)
    # A model whose scale differs from the reference's about its walls still puts the photos on the right
    # wall: not wrong (the reference's own GPS scale is only known to a few per cent).
    scaled = {k: {"enu": [*(anchor + 1.15 * (np.asarray(v["enu"][:2]) - anchor)), 1.6]} for k, v in off.items()}
    unscaled = {pid: {"enu": [*(anchor + (np.asarray(v["enu"][:2]) - anchor)), 1.6]} for pid, v in off.items()}
    assert ev.reference_check(unscaled, photo_key, reference, undo, anchor)["verdict"] == "right"
    sc = ev.reference_check(scaled, photo_key, reference, None, anchor)
    assert sc["correction_after"]["scale"] == pytest.approx(1 / 1.15, abs=0.05)
    # ... but one 30 % too small would stretch its photos along the walls: wrong
    small = {k: {"enu": [*(anchor + 0.7 * (np.asarray(v["enu"][:2]) - anchor)), 1.6]} for k, v in unscaled.items()}
    assert ev.reference_check(small, photo_key, reference, undo, anchor)["verdict"] == "wrong"
    # Too few reference cameras: no verdict; unverified reference components and flagged cameras do not count.
    few = ev.reference_check({k: off[k] for k in ("p0", "p1")}, photo_key, reference, undo, anchor)
    assert few["verdict"].startswith("unknown") and few["n"] == 2
    reference["cameras"]["k0"]["verified"] = False
    assert ev.reference_check(off, photo_key, reference, undo, anchor)["n"] == 3
    off["p1"]["flags"] = ["far_from_gps"]
    assert ev.reference_check(off, photo_key, reference, undo, anchor)["n"] == 2


def test_the_verdict_is_taken_over_the_wall_points_not_only_at_their_median(ev):
    """A turn about the middle of a long facade leaves its median wall point in place but moves its ends: 4.5 deg
    over a 60 m facade puts the photos at its ends 2.4 m off (90th percentile 2.1 m), a 4 % rescale over 20 m
    only 0.4 m."""
    ref_xy = {f"k{i}": np.array([x, -60.0]) for i, x in enumerate((-30.0, -10.0, 10.0, 30.0))}
    reference = {"cameras": {k: {"xy": v, "component": "c", "verified": True} for k, v in ref_xy.items()},
                 "components": {"c": {"verified": True}}}
    photo_key = {f"p{i}": f"k{i}" for i in range(4)}
    anchor = np.zeros(2)
    walls = np.stack([np.linspace(-30, 30, 121), np.zeros(121)], 1)
    turned = {f"p{i}": {"enu": [*moved(np.array([[*ref_xy[f'k{i}'], 1.6]]), 4.5, (0.0, 0.0))[0]]} for i in range(4)}
    at_anchor = ev.reference_check(turned, photo_key, reference, None, anchor)
    over_walls = ev.reference_check(turned, photo_key, reference, None, anchor, walls)
    # at the anchor alone the turn is invisible; the cameras 60 m away moved 4.7 m (schema 5: wrong for that)
    assert at_anchor["correction_after"]["at_walls_m"] < 1e-6 and ev.correction_is_right(at_anchor["correction_after"])
    assert at_anchor["verdict"] == "wrong" and at_anchor["wrong_because"] == ["cameras"]
    assert over_walls["verdict"] == "wrong" and "walls" in over_walls["wrong_because"]
    assert over_walls["correction_after"]["walls_p90_m"] == pytest.approx(0.9 * 30 * math.radians(4.5), abs=0.1)
    assert over_walls["correction_before"]["walls_p90_m"] == over_walls["correction_after"]["walls_p90_m"]
    short = walls[np.abs(walls[:, 0]) <= 10.0]
    scaled = {k: {"enu": [*(1.04 * np.asarray(ref_xy[photo_key[k]])), 1.6]} for k in photo_key}
    small = ev.reference_check(scaled, photo_key, reference, None, anchor, short)
    # 4 % over 20 m of wall: 0.4 m on the walls, but the cameras 60 m away are 2.4 m off
    assert small["correction_after"]["walls_p90_m"] < 0.5 and ev.correction_is_right(small["correction_after"])
    assert small["verdict"] == "wrong" and small["wrong_because"] == ["cameras"]
    assert not ev.correction_is_right({"rotation_deg": 0.0, "at_walls_m": 0.0, "walls_p90_m": 0.0, "scale": 1.12})
    assert not ev.correction_is_right({"rotation_deg": 0.0, "at_walls_m": 0.0, "walls_p90_m": 0.0, "scale": 1.0}, None, 1.6)
    assert ev.correction_is_right({"rotation_deg": 0.0, "at_walls_m": 0.0, "walls_p90_m": 0.0, "scale": 1.0}, None, 1.5)
    # with a snap, what the georegistration needed is measured at the walls where they were before the snap: a
    # model 20 % too large whose snap rescaled it (a rigid correction moves both the same; a rescale does not)
    far_walls = np.stack([np.linspace(-30, 30, 121), np.full(121, -30.0)], 1)
    big = {k: {"enu": [*(1.2 * ref_xy[photo_key[k]]), 1.6]} for k in photo_key}
    undo = {"rotation_deg": 0.0, "shift_m": [0.0, 0.0], "scale": 1 / 1.2, "accepted": True}
    at = np.median(far_walls, axis=0)
    chk = ev.reference_check(big, photo_key, reference, undo, at, far_walls)
    assert chk["verdict"] == "right" and chk["correction_after"]["walls_p90_m"] < 1e-6
    src = np.array([big[k]["enu"][:2] for k in sorted(photo_key)])
    dst = np.array([ref_xy[photo_key[k]] for k in sorted(photo_key)])
    assert chk["correction_before"]["walls_p90_m"] == pytest.approx(
        ev.camera_correction(src, dst, 1.2 * at, 1.2 * far_walls)["walls_p90_m"], abs=0.01)        # 8.1 m, not 6.7
    assert abs(chk["correction_before"]["walls_p90_m"] - ev.camera_correction(src, dst, at, far_walls)["walls_p90_m"]) > 1.0


def test_wall_anchor_takes_the_points_on_the_outline(ev):
    """The verdict's wall points: wall-height points within reg_edge_limit_m of the outline (within 10 m when
    fewer than 20 are), their median the anchor; none at all: the outline's centroid and no wall points."""
    square = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    xs = np.linspace(-9.0, 9.0, 40)
    south = np.c_[xs, np.full(40, -10.5), np.full(40, 6.0)]
    tree = np.c_[np.full(30, 0.0), np.full(30, -25.0), np.full(30, 6.0)]           # 15 m in front: not a wall
    roof = np.c_[xs[:30], np.zeros(30), np.full(30, 25.0)]                           # above the band
    hedge = np.c_[xs[:25], np.full(25, -16.0), np.full(25, 2.5)]                     # 6 m in front: not either
    anchor, walls = ev.wall_anchor(np.vstack([south, tree, roof, hedge]), square, [])
    assert walls is not None and len(walls) == 40 and np.allclose(anchor, [0.0, -10.5])
    off = south + [0.0, -6.0, 0.0]                                                   # 6.5 m off: the 10 m fallback
    anchor, walls = ev.wall_anchor(off, square, [])
    assert len(walls) == 40 and np.allclose(anchor, [0.0, -16.5])
    anchor, walls = ev.wall_anchor(tree, square, [])
    assert walls is None and np.allclose(anchor, [0.0, 0.0])


def test_camera_correction_ignores_a_camera_kilometres_away(ev):
    rng = np.random.default_rng(3)
    src = rng.uniform(-60, 60, size=(8, 2))
    th = math.radians(3.0)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    dst = src @ r.T + [2.0, 1.0]
    src[0] = [-135000.0, 9000.0]        # a withheld camera far away must not wreck the fit
    c = ev.camera_correction(src, dst, np.zeros(2))
    assert c["cameras_used"] == 7 and c["rotation_deg"] == pytest.approx(3.0, abs=0.01)
    assert c["scale"] == pytest.approx(1.0, abs=1e-3) and c["at_walls_m"] == pytest.approx(math.hypot(2, 1), abs=0.01)


def test_outline_corrected_reference_uses_synthetic_truth_as_is(ev):
    gt = {"kind": "poses", "poses": {"a.jpg": {"rot": np.eye(3).tolist(), "center": [3.0, -4.0, 1.6]}}}
    ref = ev.outline_corrected_reference(gt, {}, np.zeros((4, 2)), [], LocalENU(0.0, 0.0))
    np.testing.assert_allclose(ref["cameras"]["a.jpg"]["xy"], [3.0, -4.0])
    assert ref["cameras"]["a.jpg"]["verified"] and ref["components"]["truth"]["verified"]


def style_record(**kw):
    rec = {"status": "ok", "model_id": kw.pop("id", "m"), "kind": "face", "sector": "E", "latest_published": True,
           "eligible": True, "registered": 8, "gate": {"passed": True, "reason": ""},
           "snap": {"accepted": True, "median_wall_distance_before_m": 4.0, "median_wall_distance_after_m": 0.3,
                    "reason": ""},
           "walls": {"total": 10, "coloured": 2, "unlocked": 1}, "observed_pixel_area_fraction": 0.1,
           "roof_observed_area_fraction": 0.0, "fill": {"area_m2": 0.0}, "glb_gz_bytes": 20480,
           "qa": {"passed": True, "violations": 0}, "time_s": 2.0}
    for k, v in kw.items():
        rec[k] = v
    return rec


def test_style_metrics_pool_the_gate_and_describe_the_best_model(ev):
    right = {"verdict": "right", "error_after_m": {"median": 1.4, "max": 2.6},
             "correction_after": {"rotation_deg": 0.5, "at_walls_m": 0.4, "walls_p90_m": 0.6, "scale": 1.0},
             "correction_3d_after": {"walls_p90_m": 1.1}}
    wrong = {"verdict": "wrong", "error_after_m": {"median": 14.0, "max": 30.0},
             "correction_after": {"rotation_deg": 9.0, "at_walls_m": 6.0, "walls_p90_m": 7.5, "scale": 1.0}}
    records = [
        style_record(id="a", reference=right),
        # a pass whose georegistration was already right: not "refused although placed right"
        style_record(id="a2", reference={"verdict": "right", "error_after_m": {"median": 0.5, "max": 0.9},
                                         "placed_right_before": True,
                                         "correction_before": {"rotation_deg": 0.5, "at_walls_m": 0.4,
                                                               "walls_p90_m": 0.6, "scale": 1.0}}),
        style_record(id="b", reference=wrong, walls={"total": 10, "coloured": 5, "unlocked": 3},
                     observed_pixel_area_fraction=0.3, glb_gz_bytes=40960),
        style_record(id="c", gate={"passed": False, "reason": "snap onto the footprint rejected"},
                     snap={"accepted": False, "median_wall_distance_before_m": 9.0,
                           "median_wall_distance_after_m": 0.4, "reason": "beyond"},
                     walls={"total": 10, "coloured": 0, "unlocked": 0},
                     # refused, although the georegistration alone was already right: a failure too
                     reference={"verdict": "wrong", "placed_right_before": True, "error_after_m": {"median": 2.0},
                                "correction_before": {"rotation_deg": -2.0, "at_walls_m": 1.2, "walls_p90_m": 1.4,
                                                      "scale": 1.05}}),
        style_record(id="c2", gate={"passed": False, "reason": "snap onto the footprint rejected"},
                     snap={"accepted": False, "median_wall_distance_before_m": 4.0,
                           "median_wall_distance_after_m": 0.4, "reason": "beyond"},
                     walls={"total": 10, "coloured": 0, "unlocked": 0},
                     # its median wall point would be right, but the ends of its walls are 2 m off
                     reference={"correction_before": {"rotation_deg": -2.0, "at_walls_m": 1.0, "walls_p90_m": 2.0,
                                                      "scale": 1.0}}),
        style_record(id="d", eligible=False, walls={"total": 10, "coloured": 9, "unlocked": 9},
                     latest_published=False),          # prior_only and superseded: not what users see; its gate
                                                       # passed, which must never happen (Frauenkirche SW#17)
        style_record(id="d2", eligible=False, gate={"passed": False, "reason": "the model is not georegistered"}),
        {"model_id": "e", "status": "skipped (time cap)", "eligible": True},
    ]
    m = ev.style_metrics(records)
    assert set(m) == set(ev.STYLE_METRICS)
    assert (m["style_models_stylized"], m["style_models_eligible"], m["style_registration_passed"]) == (7, 5, 3)
    assert m["style_registration_pass_rate"] == pytest.approx(3 / 5, abs=1e-4)
    assert (m["style_registration_checked"], m["style_registration_wrong"], m["style_registration_right"]) == (3, 1, 2)
    assert m["style_coloured_walls_wrong"] == 5            # b's coloured walls; d's 9 are not a checked pass
    assert m["style_rejected_placed_right"] == 1           # c: 1.4 m over the walls; c2: 2 m is misplaced
    assert m["style_gate_passed_not_eligible"] == 1        # d
    assert m["style_snap_residual_before_m"] == 4.0 and m["style_snap_residual_m"] == 0.3
    assert m["style_ref_camera_error_m"] == pytest.approx(1.4)
    # the largest per-model median, the worst single camera, the worst 3-D wall displacement (schema 5)
    assert m["style_ref_camera_median_worst_m"] == pytest.approx(14.0)
    assert m["style_ref_camera_error_max_m"] == pytest.approx(30.0)
    assert m["style_ref_walls_p90_worst_m"] == pytest.approx(7.5)
    assert m["style_incomplete"] == 1                      # e: the stylization's time cap
    assert ev.best_style_record(records)["model_id"] == "b"
    assert (m["style_walls_coloured"], m["style_walls_unlocked"], m["style_glb_kb_gz"]) == (5, 3, 40.0)
    assert m["style_time_s"] == 14.0 and m["style_qa_failed"] == 0
    empty = ev.style_metrics([])
    assert empty["style_registration_pass_rate"] is None and empty["style_models_eligible"] == 0


def struct_block(**kw):
    rec = {"judge": "skipped", "parts_built": 3, "parts_judged": 0, "registration_accepted": False, "d15_adopted": 0,
           "d25_records": 0, "photo_share": 0.0, "unbacked": 0, "qa_passed": True, "checks": [], "checked": 0,
           "incorrect": 0, "seconds": 2.0, "total_seconds": 3.0, "peak_rss_mb": 300.0, "within_limits": True}
    rec.update(kw)
    return rec


def test_struct_metrics_pool_the_structure_blocks(ev):
    """struct_*: sums over the models that went through the structure pipeline; D15 adoptions in a model the
    registration gate refused count as adopted unregistered; None without a snapshot (schema 7's empty slots)."""
    records = [
        style_record(id="a", structure=struct_block(registration_accepted=True, judge="ran", parts_judged=3,
                                                    d15_adopted=2, photo_share=0.1, checked=2, incorrect=1)),
        style_record(id="b", structure=struct_block(d15_adopted=1, peak_rss_mb=900.0)),
        style_record(id="c", structure=struct_block(unbacked=1, qa_passed=False, d25_records=1)),
        {"model_id": "d", "status": "failed: x", "eligible": True},
    ]
    m = ev.struct_metrics(records, True)
    assert set(m) == {k for k in ev.METRIC_BY_NAME if k.startswith("struct_")}
    assert (m["struct_models"], m["struct_d15_adopted"], m["struct_d25_records"]) == (3, 3, 1)
    assert (m["struct_checked"], m["struct_incorrect"], m["struct_unbacked"]) == (2, 1, 1)
    assert m["struct_adopted_unregistered"] == 1 and m["struct_qa_failed"] == 1
    assert m["struct_photo_share"] == pytest.approx(0.1 / 3, abs=1e-4)
    assert m["struct_time_s"] == 6.0 and m["struct_peak_rss_mb"] == 900.0
    assert all(v is None for v in ev.struct_metrics(records, False).values())
    none_ran = ev.struct_metrics([{"model_id": "d", "status": "failed: x"}], True)
    assert none_ran["struct_models"] == 0 and none_ran["struct_time_s"] is None


def test_published_heights_are_checked_against_the_reference_points(ev):
    """struct_checked / struct_incorrect: a photo-measured height against the reference's verified points over the
    part (0.98 quantile), within max(1 m, 5 %); a lower bound is right when the reference reaches it."""
    import test_structure_stylize as tss
    structure = tss.box_structure("flat")                         # one 10 m flat-roofed box part, w100
    rng = np.random.default_rng(0)
    lo, hi = structure.built[0].outer.min(axis=0), structure.built[0].outer.max(axis=0)
    roof = np.c_[rng.uniform(lo, hi, (400, 2)), np.full(400, 14.0)]          # the reference: the roof at 14 m
    reference = {"components": {"0": {"verified": True}, "1": {"verified": False}},
                 "points": {"0": roof, "1": roof + [0.0, 0.0, 30.0]}}         # an unverified component is ignored
    rows = ev.structure_check(structure, [
        {"id": "A-1", "rule": "D15", "kind": "part-height", "part": "w100", "param": "height", "value": 13.4},
        {"id": "A-2", "rule": "D15", "kind": "part-height", "part": "w100", "param": "height", "value": 11.0},
        {"id": "D-3", "rule": "D25", "kind": "d25-height", "part": "w100", "param": "height", "value": 12.0,
         "adopted": "lower bound"},
        {"id": "D-4", "rule": "D25", "kind": "d25-height", "part": "w100", "param": "height", "value": 16.0,
         "adopted": "lower bound"},
        {"id": "A-5", "rule": "D15", "kind": "frame", "param": "outline", "value": {"k": 1.0}},
        {"id": "A-6", "rule": "D15", "kind": "part-height", "part": "w100", "param": "height", "value": 14.0,
         "active": False}], reference)
    got = {r["id"]: (r["checked"], r.get("incorrect")) for r in rows}
    assert got == {"A-1": (True, False), "A-2": (True, True), "D-3": (True, False), "D-4": (True, True)}
    assert rows[0]["reference_m"] == pytest.approx(14.0) and rows[0]["tol_m"] == 1.0
    empty = ev.structure_check(structure, [{"id": "A-1", "kind": "part-height", "part": "w100", "param": "height",
                                            "value": 13.4}], {"components": {}, "points": {}})
    assert empty[0]["checked"] is False and "no reference reading" in empty[0]["why"]


def test_the_stylize_step_goes_through_the_structure_pipeline(ev, tmp_path):
    """A landmark with an OSM parts snapshot (the catalog's, bound by slug): every published model is stylized
    structure first (here the Baptistery's 3 parts as clay: the tiny job's registration is refused, nothing is
    judged) and the struct_* metrics are filled; structure=False is the footprint path with struct_* empty."""
    import test_structure_pipeline as tsp
    catalog = json.loads((SCRIPT.parent.parent / "eval" / "landmarks.json").read_text())
    entry = next(e for e in catalog if e["slug"] == "florence-baptistery")
    job, meta = tsp.tiny_job(tmp_path)
    (tmp_path / "work").mkdir()
    job.rename(tmp_path / "work" / "j1")
    meta.rename(tmp_path / "m1.json")
    model = {"id": "m1", "kind": "face", "sector": "S", "request": 1, "latest_published": True, "published": True,
             "georegistered": True, "yaw_source": "gps", "num_registered": 4, "num_points": 1, "job_id": "j1",
             "ply_path": str(tmp_path / "m1.ply")}
    b = ev.CrowdBuilding("florence-baptistery", "Baptistery", entry["lat"], entry["lon"], 150.0, [],
                         outline_path=SCRIPT.parent.parent / entry["osm"]["file"])
    kw = dict(work_dir=tmp_path / "work", photo_key={}, metas={}, enu=LocalENU(b.lat, b.lon),
              reference_result=(None, None))
    on = ev.stylize_published([model], b, out_dir=tmp_path / "on", **kw)
    rec = on["records"][0]
    assert rec["status"] == "ok" and on["structure"]["osm_parts_sha256"] == entry["osm_parts"]["sha256"]
    st_ = rec["structure"]
    assert st_["judge"] == "skipped" and st_["parts_built"] == 3 and not st_["registration_accepted"]
    assert st_["qa_passed"] and st_["within_limits"] and st_["d15_adopted"] == 0
    m = on["metrics"]
    assert (m["struct_models"], m["struct_d15_adopted"], m["struct_adopted_unregistered"], m["struct_qa_failed"],
            m["struct_unbacked"], m["struct_photo_share"]) == (1, 0, 0, 0, 0, 0.0)
    assert json.loads(Path(rec["summary_json"]).read_text())["massing_source"] == "osm-parts"
    off = ev.stylize_published([model], b, out_dir=tmp_path / "off", structure=False, **kw)
    assert "structure" not in off["records"][0] and off["structure"] is None
    assert all(v is None for k, v in off["metrics"].items() if k.startswith("struct_"))
    assert json.loads(Path(off["records"][0]["summary_json"]).read_text())["massing_source"] == "height-field"


def test_diagnose_gate_names_the_cause(ev):
    def diag(reason, cand, needed=None, fit=None, window=15.0, mode="window", **kw):
        snap = {"accepted": False, "reason": reason,
                "meta": {"search": {"window_deg": window, "mode": mode, "max_shift_m": 25.0}}}
        rec = style_record(gate={"passed": False, "reason": reason}, snap_candidate=cand, snap=snap,
                           reference={"correction_before": needed} if needed else {}, **kw)
        return ev.diagnose_gate(rec, fit)["cause"]

    cand = {"rotation_deg": 78.0, "shift_m": 37.0, "median_after_m": 0.4}
    assert diag("too few wall points", None) == "too_few_points"
    assert diag("fewer than 2 outline edges carry wall points", cand) == "single_wall"
    assert diag("the verified outline edges are all parallel: ...", cand) == "single_wall"
    assert diag("snap onto the footprint rejected: ambiguous: another placement 14.2 m / +0.0 deg away ...", cand,
                {"rotation_deg": 1.0, "shift_m": 3.0}) == "ambiguous"
    assert diag("snap onto the footprint rejected: the cameras would stand 18% closer to the walls than their"
                " own GPS says ...", cand, {"rotation_deg": 1.0, "shift_m": 3.0}) == "gps_disagrees"
    assert diag("snap onto the footprint rejected: the outline asks for a rescale beyond 1.15x ...", cand) == "shape"
    assert diag("snap onto the footprint rejected: the georegistration leaves the scale uncertain by +-20% ...",
                cand, {"rotation_deg": -11.0, "shift_m": 16.9}) == "scale_uncertain"
    edge = "snap onto the footprint rejected: the best fit lies at the edge of the yaw window (+-15.0 deg)"
    assert diag(edge, cand, {"rotation_deg": -11.0, "shift_m": 16.9}) == "georegistration"
    # the needed correction is outside the search (beyond the window, or the shift bound)
    walls = "snap onto the footprint rejected: walls still 2.9 m off the footprint after snapping"
    assert diag(walls, cand, {"rotation_deg": -18.0, "shift_m": 10.0}) == "georegistration"
    assert diag(walls, cand, {"rotation_deg": -11.0, "shift_m": 35.0}) == "georegistration"
    # inside the window, but an unconstrained ICP would fall into an alias: what the window is for
    assert diag(walls, cand, {"rotation_deg": -11.0, "shift_m": 16.9}) == "symmetric_alias"
    assert diag("only 2/4 supported outline edges have their points within 3.0 m (median)",
                {"rotation_deg": -10.0, "shift_m": 12.0}, {"rotation_deg": -11.0, "shift_m": 12.5}) == "residual"
    assert diag(walls, cand, {"rotation_deg": -11.0, "shift_m": 16.9}, reference_component_verified=False) == \
        "osm_outline"
    # translation only (a near-circle): any needed rotation lies outside the search
    assert diag(walls, cand, {"rotation_deg": 3.5, "shift_m": 4.0}, window=0.0, mode="translation-only") == \
        "georegistration"
    # no reference: the harness's own outline fit is the evidence
    assert diag(walls, cand, None, {"rotation_deg": 2.0, "shift_m": 5.0, "ambiguous": False}) == "symmetric_alias"
    assert diag(walls, cand, None, None) == "unknown"
    # schema 4: a near-circle whose registered yaw is too uncertain, a flat yaw valley (an ambiguity)
    assert diag("snap onto the footprint rejected: yaw unverifiable (near-rotationally symmetric outline): the registered"
                " yaw is only known to +-9.9 deg ...", cand, {"rotation_deg": 3.5, "shift_m": 4.0}, window=0.0,
                mode="translation-only") == "yaw_unverifiable"
    assert diag("snap onto the footprint rejected: ambiguous: the walls turned -3.5 deg about themselves fit as well"
                " ...", cand, {"rotation_deg": 4.2, "shift_m": 3.0}) == "ambiguous"
    assert diag("snap onto the footprint rejected: the walls follow the outline only to 0.51 m on average (more than"
                " 0.45 m): too loose to fix the yaw", cand, {"rotation_deg": 4.2, "shift_m": 3.0}) == "loose_fit"


def test_run_crowd_stylize_records_what_it_could_not_stylize(ev, synthetic_dataset, tmp_path):
    photos = [ev.CrowdPhoto(key=p["file"], path=synthetic_dataset.path / p["file"], author=p["user_id"])
              for p in synthetic_dataset.manifest["photos"]]
    b = ev.CrowdBuilding("synthetic", "Synthetic", synthetic_dataset.lat, synthetic_dataset.lon, 150.0, photos)
    summary = ev.run_crowd(b, tmp_path / "run", reconstructor="fake", stylize=True)
    assert summary["stylize"] and summary["keep_work_dirs"]
    assert summary["style"]["note"] == "no OSM outline"          # nothing to snap onto: no stylization
    m = summary["metrics"]
    assert m["style_models_stylized"] == 0 and m["style_registration_pass_rate"] is None
    assert set(m) == set(ev.METRIC_BY_NAME)


@pytest.mark.slow
def test_run_crowd_stylizes_every_published_colmap_model(ev, tmp_path):
    """Real COLMAP on the synthetic two-sides set (as synthetic-v1): every published model is stylized from its
    kept work dir, gated, checked against the generator's truth and passes the honesty QA."""
    from crowd3d import synthetic

    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import benchmark
    finally:
        sys.path.remove(str(SCRIPT.parent))
    data = tmp_path / "photos"
    manifest = synthetic.generate_dataset(data, sectors=[2, 3], photos_per_sector=6, num_users=3, size=(640, 480),
                                          gps_noise_m=2.0, heading_noise_deg=8.0, include_bad=False, seed=1)
    lat, lon = manifest["landmark"]["lat"], manifest["landmark"]["lon"]
    spec = synthetic.BuildingSpec()
    outline = tmp_path / "box.geojson"
    outline.write_text(json.dumps(benchmark.box_outline_geojson(lat, lon, spec.width_e_m, spec.depth_n_m, "box")))
    photos = [ev.CrowdPhoto(key=p["file"], path=data / p["file"], author=p["user_id"]) for p in manifest["photos"]]
    b = ev.CrowdBuilding("box", "Box", lat, lon, 150.0, photos, outline_path=outline,
                         gt={"kind": "poses", "poses": ev.synthetic_gt_poses(manifest)})
    summary = ev.run_crowd(b, tmp_path / "run", reconstructor="colmap", stylize=True)
    m, style = summary["metrics"], summary["style"]
    published = [x for x in summary["models"] if x["published"]]
    assert published and len(style["records"]) == len(published)
    assert all(r["status"] == "ok" and r["qa"]["passed"] and Path(r["glb"]).is_file() for r in style["records"])
    assert m["style_models_stylized"] == len(published) and m["style_qa_failed"] == 0
    assert m["style_provenance_violations"] == 0
    assert m["style_models_eligible"] == sum(bool(x["georegistered"]) for x in published)
    assert m["style_registration_passed"] >= 1 and 0 < m["style_registration_pass_rate"] <= 1
    passing = [r for r in style["records"] if r["eligible"] and r["gate"]["passed"]]
    assert all(r["reference"]["verdict"] in ("right", "wrong") for r in passing)   # truth for every camera
    assert m["style_walls_coloured"] >= 1 and m["style_walls_total"] == 4
    assert all(r.get("diagnosis") for r in style["records"] if r["eligible"] and not r["gate"]["passed"])
    # the work dirs were kept (sparse model + image links), the databases were not
    work = Path(summary["data_dir"]) / "work"
    jobs = [d for d in work.iterdir() if d.is_dir()]
    assert jobs and all((d / "sparse" / "0").is_dir() and not (d / "database.db").exists() for d in jobs
                        if (d / "sparse" / "0").is_dir())
    # stylizing the kept run again (benchmark rescore --restylize) gives the same records, times and paths aside
    reference, ref_report, _ = ev.building_reference(b, {p.key: p.meta for p in photos if p.meta}, LocalENU(lat, lon))
    again = ev.restylize_run(tmp_path / "run", b, (reference, ref_report), tmp_path / "restyle")

    def same(records: list) -> list:
        return [{k: v for k, v in r.items() if k not in ("time_s", "timings_s", "glb", "summary_json")}
                for r in json.loads(json.dumps(records, default=ev._json_default))]

    assert same(again["records"]) == same(style["records"])
    assert all(Path(r["glb"]).is_relative_to(tmp_path / "restyle") for r in again["records"])
    # schema 7 against the generator's truth: the anchors' geographic error (rotations from the kept sparse models),
    # the outline the shown models support, their calibration; the rescore of the kept run measures the same
    assert m["geo_models_checked"] >= 1 and m["geo_yaw_error_median_deg"] < 10.0 and m["geo_tilt_error_median_deg"] < 3.0
    assert m["geo_camera_error_median_m"] < 5.0 and m["wrong_face_models_published"] == 0
    assert 0 < m["cov_local_supported_m_1p5"] <= m["cov_local_supported_m_5"] and m["cov_unverified_m"] == 0.0
    assert m["calib_yaw_n"] >= 1 and m["stale_overwrites"] == 0 and m["peak_rss_jobs_mb"] > m["baseline_rss_mb"]
    geo = ev.rescore_geo(tmp_path / "run", b, reference)
    assert geo["metrics"] == {k: m[k] for k in geo["metrics"]}


# ---------------------------------------------------------------- round 3: the verdict in 3-D, with the cameras


def _look(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """cam_from_enu rows (right, down, forward) of a level camera looking at target."""
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def _tilt(deg: float, about: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A rotation of `deg` about the east axis through `about` (raises the north side): (R, t) of p' = R p + t."""
    th = math.radians(deg)
    r = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(th), -math.sin(th)], [0.0, math.sin(th), math.cos(th)]])
    return r, about - r @ about


def test_a_tilted_model_is_wrong_although_every_horizontal_check_passes(ev):
    """Frauenkirche SW#14: 2.9 deg of tilt about its cameras put the photos 3-4 m too high on walls 80 m away,
    while its camera centres and wall points matched the reference horizontally. correction_3d takes the
    rotation from the cameras' orientations, which see the tilt."""
    xs = np.linspace(-20.0, 20.0, 8)
    ref_c = np.stack([xs, np.full(8, -86.0), np.full(8, 1.6)], 1)
    ref_r = np.array([_look(c, [0.0, -6.0, 8.0]) for c in ref_c])
    reference = {"cameras": {f"k{i}": {"xy": ref_c[i, :2], "xyz": ref_c[i], "rot": ref_r[i], "component": "c",
                                       "verified": True} for i in range(8)}, "components": {"c": {"verified": True}}}
    photo_key = {f"p{i}": f"k{i}" for i in range(8)}
    walls = np.stack([np.linspace(-10, 10, 60), np.full(60, -6.0), np.tile([4.0, 8.0, 12.0], 20)], 1)
    r, t = _tilt(2.9, ref_c.mean(axis=0))
    cams = {f"p{i}": {"enu": list(r @ ref_c[i] + t)} for i in range(8)}
    rots = {f"p{i}": ref_r[i] @ r.T for i in range(8)}
    tilted_walls = walls @ r.T + t
    chk = ev.reference_check(cams, photo_key, reference, None, np.median(tilted_walls[:, :2], axis=0), tilted_walls,
                             rotations=rots)
    c3 = chk["correction_3d_after"]
    assert c3["tilt_deg"] == pytest.approx(2.9, abs=0.05) and c3["orientations_used"] == 8
    assert c3["walls_p90_m"] == pytest.approx(80.0 * math.radians(2.9), abs=0.6)          # ~4 m
    assert chk["correction_after"]["walls_p90_m"] < 1.5 and chk["error_after_m"]["median"] < 1.5
    assert chk["verdict"] == "wrong" and chk["wrong_because"] == ["walls_3d"]
    assert not chk["placed_right_before"]
    # the same model level: right, and the 3-D correction agrees with the 2-D one
    level = ev.reference_check({k: {"enu": list(ref_c[i])} for i, k in enumerate(photo_key)}, photo_key, reference,
                               None, np.median(walls[:, :2], axis=0), walls,
                               rotations={f"p{i}": ref_r[i] for i in range(8)})
    assert level["verdict"] == "right" and level["correction_3d_after"]["walls_p90_m"] < 0.05
    # without orientations there is no 3-D correction and the verdict is horizontal only
    flat = ev.reference_check(cams, photo_key, reference, None, np.median(tilted_walls[:, :2], axis=0), tilted_walls)
    assert "correction_3d_after" not in flat and flat["verdict"] == "right"


def test_correction_3d_leaves_out_a_camera_whose_orientation_is_off(ev):
    rng = np.random.default_rng(2)
    ref_c = np.c_[rng.uniform(-30, 30, (9, 2)) + [0.0, -80.0], np.full(9, 1.6)]
    ref_r = np.array([_look(c, [0.0, 0.0, 6.0]) for c in ref_c])
    mod_r = ref_r.copy()
    mod_r[0] = _look(ref_c[0], [60.0, 0.0, 6.0])               # one camera's orientation 30+ deg off
    c3 = ev.correction_3d(mod_r, ref_c, ref_r, ref_c, np.array([[0.0, -6.0, 5.0], [5.0, -6.0, 9.0]]))
    assert c3["orientations_used"] == 8 and c3["rotation_deg"] < 0.01 and c3["walls_p90_m"] < 0.01


def test_the_verdict_limits_are_the_documented_ones(ev):
    """5 deg, 1.5 m over the walls (90th percentile, 3-D when known), 10 % scale, 1.5 m camera median (round-2
    mutation check: a 10 deg turn limit survived every test)."""
    ok = {"rotation_deg": 4.9, "at_walls_m": 0.2, "walls_p90_m": 1.5, "scale": 1.0999}
    assert ev.correction_is_right(ok, {"walls_p90_m": 1.5}, 1.5)
    assert not ev.correction_is_right(dict(ok, rotation_deg=5.1), None, 0.5)
    assert not ev.correction_is_right(dict(ok, rotation_deg=-6.0), None, 0.5)
    assert not ev.correction_is_right(ok, {"walls_p90_m": 1.51}, 0.5)
    assert not ev.correction_is_right(dict(ok, scale=0.89), None, 0.5)
    assert ev.wrong_because(dict(ok, rotation_deg=6.0, scale=1.2), {"walls_p90_m": 2.0}, 1.6) == [
        "turn", "walls_3d", "scale", "cameras"]
    assert (ev.STYLE_RIGHT_ROT_DEG, ev.STYLE_RIGHT_SHIFT_M, ev.STYLE_RIGHT_SCALE, ev.STYLE_RIGHT_CAMERA_M) == (
        5.0, 1.5, 0.10, 1.5)


def test_a_model_record_is_eligible_only_when_georegistered(ev, tmp_path):
    """Eligibility (georeg_level = full) is the model's own flag (round-2 mutation check: making every model
    eligible survived every test)."""
    base = {"id": "m1", "kind": "face", "sector": "N", "request": 3, "latest_published": True, "yaw_source": "gps",
            "num_registered": 8, "num_points": 900, "job_id": "nojob", "ply_path": str(tmp_path / "x.ply")}
    for flag in (False, True):
        rec = ev.stylize_model_record(dict(base, georegistered=flag), work_dir=tmp_path,
                                      scene_base={}, out_dir=tmp_path, reference=None, photo_key={})
        assert rec["eligible"] is flag and rec["status"].startswith("skipped")


def test_the_reference_carries_3d_centres_and_orientations(ev):
    gt = {"kind": "poses", "poses": {"a.jpg": {"rot": np.eye(3).tolist(), "center": [3.0, -4.0, 1.6]}}}
    ref = ev.outline_corrected_reference(gt, {}, np.zeros((4, 2)), [], LocalENU(0.0, 0.0))
    cam = ref["cameras"]["a.jpg"]
    np.testing.assert_allclose(cam["xyz"], [3.0, -4.0, 1.6])
    np.testing.assert_allclose(cam["rot"], np.eye(3))


def test_diagnose_gate_names_the_round_3_causes(ev):
    def diag(reason):
        rec = style_record(gate={"passed": False, "reason": reason}, snap_candidate={"rotation_deg": 1.0},
                           snap={"accepted": False, "reason": reason, "meta": {"search": {"window_deg": 9.0}}})
        return ev.diagnose_gate(rec, None)["cause"]

    assert diag("snap onto the footprint rejected: the model is tilted: its cameras' heights rise 2.4 deg") == "tilt"
    assert diag("snap onto the footprint rejected: 4 unflagged cameras: too few to check the model's tilt") == "tilt"
    assert diag("1 outline edge(s) carry wall points that no camera can see past the building (edge 0:2)") == "unseen_face"
    assert diag("snap onto the footprint rejected: the model carries no camera positions: ...") == "no_cameras"


def test_the_stylization_has_its_own_time_budget(ev, synthetic_dataset, tmp_path):
    """The upload time cap must not eat into the stylization (verifier r2: under load a Frauenkirche run passed
    1.15 x the soft cap during the reference build and lost every style record, 'style gate 0/0'); models the
    stylization's own budget does not reach count in style_incomplete."""
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import benchmark
    finally:
        sys.path.remove(str(SCRIPT.parent))
    outline = tmp_path / "box.geojson"
    outline.write_text(json.dumps(benchmark.box_outline_geojson(synthetic_dataset.lat, synthetic_dataset.lon, 20.0, 12.0,
                                                                "box")))
    photos = [ev.CrowdPhoto(key=p["file"], path=synthetic_dataset.path / p["file"], author=p["user_id"])
              for p in synthetic_dataset.manifest["photos"]]
    b = ev.CrowdBuilding("synthetic", "Synthetic", synthetic_dataset.lat, synthetic_dataset.lon, 150.0, photos,
                         outline_path=outline, upload_all=False)
    summary = ev.run_crowd(b, tmp_path / "run", reconstructor="fake", arrival="single", soft_time_cap_s=1e-6,
                           stylize=True)
    records = summary["style"]["records"]
    # the soft cap was long over when the stylization started; it still reached every model (the fake
    # reconstructor keeps no COLMAP work dir, so there is nothing to stylize, but nothing was cut either)
    assert records and all(r["status"] == "skipped (no kept work dir)" for r in records)
    assert summary["metrics"]["style_incomplete"] == 0


# ---------------------------------------------------------------- photo poses (photo_pose_* metrics)


def _pose(pid, src, p=None, yaw=None, sp=None, sy=None, reason=None):
    from crowd3d.photoposes import PhotoPose

    return PhotoPose(pid, src, reason, p=p, yaw=yaw, sp=sp, sy=sy)


def test_photo_pose_records_and_metrics(ev):
    rows = [{"key": k, "photo_id": k.upper(), "accepted": acc} for k, acc in
            [("a", True), ("b", True), ("c", True), ("d", True), ("e", True), ("f", False), ("g", True)]]
    poses = {
        "A": _pose("A", "sfm", [10.0, 0.0, 1.6], 270.0, 15.0, 20.0),     # right
        "B": _pose("B", "sfm", [30.0, 0.0, 1.6], 270.0, 15.0, 20.0),     # 20 m off: wrong
        "C": _pose("C", "exif", [0.0, 12.0, 1.6], 200.0, 24.0, 30.0),    # 20 deg off: wrong, inside its uncertainty
        "D": _pose("D", "gps", [0.0, -10.0, 1.6], reason="gps_without_heading"),
        "E": _pose("E", "exif", [-50.0, 0.0, 1.6], 90.0, 24.0, 30.0),    # no reference: cannot be checked
        "F": _pose("F", None, reason="rejected"),
        "G": _pose("G", "sfm", [3.0, 4.0, 1.6], 2.0, 15.0, 20.0),        # strict: 1 m / 2 deg
    }
    ref = {"a": {"xy": [16.0, 1.0], "heading_deg": 265.0}, "b": {"xy": [10.0, 0.0], "heading_deg": 270.0},
           "c": {"xy": [0.0, 10.0], "heading_deg": 180.0}, "d": {"xy": [0.0, -12.0], "heading_deg": 0.0},
           "f": {"xy": [5.0, 5.0], "heading_deg": 0.0}, "g": {"xy": [3.0, 3.0], "heading_deg": 0.0}}
    recs = ev.photo_pose_records(rows, poses, ref)
    by = {r["key"]: r for r in recs}
    assert by["a"]["correct"] and not by["a"]["strict"] and by["a"]["within_uncertainty"]
    assert by["b"]["correct"] is False and by["b"]["pos_err_m"] == 20.0
    assert by["c"]["correct"] is False and by["c"]["head_err_deg"] == 20.0 and by["c"]["within_uncertainty"]
    assert by["d"]["determined"] is False and by["d"]["pos_err_m"] == 2.0 and "correct" not in by["d"]
    assert by["e"]["gt"] is False and by["e"]["determined"] and "correct" not in by["e"]
    assert by["f"]["reason"] == "rejected" and by["f"]["gt"]
    assert by["g"]["strict"]
    m = ev.pose_metrics(recs, True)
    assert set(m) == set(ev.POSE_METRICS)
    assert (m["photo_pose_determined"], m["photo_pose_sfm"], m["photo_pose_exif"], m["photo_pose_gps_only"]) == (5, 3, 2, 1)
    assert m["photo_pose_determined_rate"] == round(5 / 7, 4)
    assert (m["photo_pose_gt"], m["photo_pose_gt_determined"], m["photo_pose_correct"], m["photo_pose_wrong"]) == (6, 4, 2, 2)
    assert m["photo_pose_correct_rate"] == round(2 / 6, 4) and m["photo_pose_correct_strict"] == 1
    # pass: determined and not shown wrong; e has no reference and passes, b and c fail
    assert m["photo_pose_pass"] == 3 and m["photo_pose_pass_rate"] == round(3 / 7, 4)
    assert (m["photo_pose_sfm_gt"], m["photo_pose_sfm_correct"], m["photo_pose_exif_gt"], m["photo_pose_exif_correct"]) == (3, 2, 1, 0)
    assert m["photo_pose_within_uncertainty"] == 3 and m["photo_pose_uncertainty_coverage"] == 0.75
    # per source: b lies outside its SfM range
    assert (m["photo_pose_sfm_within_uncertainty"], m["photo_pose_sfm_uncertainty_coverage"]) == (2, round(2 / 3, 4))
    assert (m["photo_pose_exif_within_uncertainty"], m["photo_pose_exif_uncertainty_coverage"]) == (1, 1.0)
    none = ev.pose_metrics(ev.photo_pose_records(rows, poses, None), False)
    assert none["photo_pose_gt"] is None and none["photo_pose_correct_rate"] is None
    assert none["photo_pose_pass"] == 5 and none["photo_pose_determined"] == 5
    s = ev.pose_summary(recs)
    assert s["by_source"] == {"sfm": 3, "exif": 2, "gps": 1, "none": 1}
    assert s["undetermined"] == {"gps_without_heading": 1, "rejected": 1}
    assert s["wrong_by_source"] == {"sfm": 1, "exif": 1}


def test_reference_poses_take_the_verified_cameras_and_survive_the_cache(ev):
    rot = look_at(np.array([0.0, -30.0, 1.6]), np.array([0.0, 0.0, 1.6]), pitch_deg=10.0)
    reference = {"components": {"0": {"verified": True}},
                 "cameras": {"a": {"xy": np.array([0.0, -30.0]), "xyz": np.array([0.0, -30.0, 1.6]), "rot": rot,
                                   "component": "0", "verified": True},
                             "b": {"xy": np.array([5.0, 5.0]), "xyz": np.array([5.0, 5.0, 1.6]), "rot": rot,
                                   "component": "1", "verified": False}},
                 "points": {"0": np.array([[1.0, 2.0, 3.25], [0.1, 0.2, 0.3]])}, "sigma_ref": dict(ev.REFERENCE_SIGMA)}
    refs = ev.reference_poses(reference)
    assert set(refs) == {"a"} and refs["a"]["heading_deg"] == pytest.approx(0.0, abs=1e-9)
    gt_poses = {"a": (rot @ np.diag([1.0, -1.0, -1.0]), np.array([1 / 3, 2.0, 0.1]), "0")}
    doc = json.loads(json.dumps(ev.reference_cache_doc(reference, {"components": reference["components"]}, gt_poses)))
    back, poses = ev.reference_from_doc(doc)
    # the cache holds every number exactly: a cached reference scores exactly as the one built from the models
    assert ev.reference_poses(back) == refs
    for key, c in reference["cameras"].items():
        for f in ("xy", "xyz", "rot"):
            assert np.array_equal(back["cameras"][key][f], c[f])
        assert (back["cameras"][key]["component"], back["cameras"][key]["verified"]) == (c["component"], c["verified"])
    assert back["components"] == reference["components"] and back["sigma_ref"] == ev.REFERENCE_SIGMA
    assert np.array_equal(back["points"]["0"], reference["points"]["0"])
    assert np.array_equal(poses["a"][0], gt_poses["a"][0]) and np.array_equal(poses["a"][1], gt_poses["a"][1])
    assert poses["a"][2] == "0"
    assert ev.reference_poses(None) == {}


def test_building_reference_is_cached_and_read_back_without_the_reference_models(ev, synthetic_dataset, tmp_path):
    poses = ev.synthetic_gt_poses(synthetic_dataset.manifest)
    cache = {"path": str(tmp_path / "refs" / "s.json"), "key": "k1", "about": {"slug": "s"}}
    b = ev.CrowdBuilding("s", "S", synthetic_dataset.lat, synthetic_dataset.lon, 150.0, [],
                         gt={"kind": "poses", "poses": poses}, reference_cache=cache)
    reference, report, gt_poses = ev.building_reference(b, {}, None)
    doc = json.loads(Path(cache["path"]).read_text())
    assert (doc["key"], doc["slug"], doc["reference_version"]) == ("k1", "s", ev.REFERENCE_VERSION)
    assert reference["sigma_ref"] == {"yaw_deg": 0.0, "scale": 0.0} and reference["points"] == {}   # generator truth
    assert "cached" not in report and set(gt_poses) == set(poses)
    # the reference models gone (no gt): the cache answers, with the same numbers
    gone = ev.CrowdBuilding("s", "S", b.lat, b.lon, 150.0, [], reference_cache=cache)
    again, report2, gt2 = ev.building_reference(gone, {}, None)
    assert report2["cached"] == cache["path"] and report2["components"] == report["components"]
    assert ev.reference_poses(again) == ev.reference_poses(reference)
    assert all(np.array_equal(gt2[k][0], gt_poses[k][0]) and np.array_equal(gt2[k][1], gt_poses[k][1])
               for k in gt_poses)
    # another key (another test set or reference version): not this reference
    assert ev.building_reference(ev.CrowdBuilding("s", "S", b.lat, b.lon, 150.0, [],
                                                  reference_cache={**cache, "key": "k2"}), {}, None) == (None, None, None)


def test_correction_3d_returns_the_similarity_it_found(ev):
    rng = np.random.default_rng(4)
    model_c = rng.uniform(-60.0, 60.0, (8, 3)) * np.array([1.0, 1.0, 0.05])
    model_r = [_look(c, np.zeros(3)) for c in model_c]
    yaw, s, t = math.radians(7.0), 1.04, np.array([3.0, -2.0, 0.5])
    q = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    ref_c = s * model_c @ q.T + t
    ref_r = [r @ q.T for r in model_r]                       # cam_from_enu in the reference frame
    walls = rng.uniform(-10.0, 10.0, (20, 3))
    c3 = ev.correction_3d(np.array(model_r), model_c, np.array(ref_r), ref_c, walls)
    assert c3["yaw_deg"] == pytest.approx(7.0, abs=1e-3) and c3["scale"] == pytest.approx(1.04, abs=1e-4)
    q_, t_ = np.array(c3["rotation"]), np.array(c3["translation_m"])
    assert np.allclose(c3["scale"] * walls @ q_.T + t_, s * walls @ q.T + t, atol=0.01)


def test_accepted_snaps_only(ev):
    style = {"records": [
        {"model_id": "m1", "status": "ok", "snap": {"accepted": True, "meta": {"rotation_deg": 1.0}}},
        {"model_id": "m2", "status": "ok", "snap": {"accepted": False, "meta": {"rotation_deg": 9.0}}},
        {"model_id": "m3", "status": "skipped (time cap)"}]}
    assert ev.accepted_snaps(style) == {"m1": {"rotation_deg": 1.0}}
    assert ev.accepted_snaps(None) == {}


def test_run_crowd_scores_photo_poses_and_rescore_agrees(ev, synthetic_dataset, tmp_path):
    poses = ev.synthetic_gt_poses(synthetic_dataset.manifest)
    photos = [ev.CrowdPhoto(key=p["file"], path=synthetic_dataset.path / p["file"], author=p["user_id"])
              for p in synthetic_dataset.manifest["photos"]]
    b = ev.CrowdBuilding("synthetic", "Synthetic", synthetic_dataset.lat, synthetic_dataset.lon, 150.0, photos,
                         gt={"kind": "poses", "poses": poses})
    summary = ev.run_crowd(b, tmp_path / "run", reconstructor="fake")
    m = summary["metrics"]
    good = len(synthetic_dataset.entries("good"))
    # the fake reconstructor puts every camera at its GPS fix, looking along its compass: all accepted photos are
    # posed by SfM, the rejected ones are failures
    assert m["photo_pose_sfm"] == good and m["photo_pose_determined"] == good and m["photo_pose_gps_only"] == 0
    assert m["photo_pose_gt"] == m["photos_uploaded"] and m["photo_pose_gt_determined"] == good
    assert m["photo_pose_correct"] == good                # synthetic GPS / compass noise is small
    assert m["photo_pose_determined_rate"] == round(good / m["photos_uploaded"], 4)
    assert len(summary["poses"]["records"]) == m["photos_uploaded"]
    assert summary["poses"]["summary"]["undetermined"] == {"rejected": m["photos_uploaded"] - good}
    # rescoring the recorded run (no SfM) gives the same numbers
    again = ev.rescore_poses(tmp_path / "run", ev.outline_corrected_reference(b.gt, {}, None, [], None))
    assert again["metrics"] == {k: m[k] for k in ev.POSE_METRICS}
    # ... and the schema-7 scores of what it left
    geo = ev.rescore_geo(tmp_path / "run", b, ev.outline_corrected_reference(b.gt, {}, None, [], None))
    assert geo["metrics"] == {k: m[k] for k in (*ev.GEO_METRICS, *ev.VERSION_METRICS, *ev.REQUEST_METRICS)}
    assert geo["geo"] == json.loads(json.dumps(summary["geo"]))


def test_photo_pose_testset_denominators_count_never_uploaded_photos(ev):
    """verify-r1: the per-scene rates left out test-set photos the harness time cap never uploaded (Frauenkirche
    73/88 determined instead of 73/111). The *_testset metrics count them as failures; the per-upload metrics are
    unchanged; the bound pair brackets the pass rate over every photo."""
    rows = [{"key": k, "photo_id": k.upper(), "accepted": True} for k in "abc"]
    poses = {"A": _pose("A", "sfm", [10.0, 0.0, 1.6], 270.0, 15.0, 20.0),      # right
             "B": _pose("B", "sfm", [30.0, 0.0, 1.6], 270.0, 15.0, 20.0),      # wrong
             "C": _pose("C", "exif", [-50.0, 0.0, 1.6], 90.0, 24.0, 30.0)}     # no reference
    ref = {"a": {"xy": [10.0, 1.0], "heading_deg": 268.0}, "b": {"xy": [10.0, 0.0], "heading_deg": 270.0},
           "d": {"xy": [0.0, 20.0], "heading_deg": 180.0}}
    planned = ["a", "b", "c", "d", "e"]          # d (with a reference) and e were never uploaded
    recs = ev.photo_pose_records(rows, poses, ref, planned)
    by = {r["key"]: r for r in recs}
    assert by["d"] == {"key": "d", "photo_id": None, "accepted": False, "src": None, "determined": False,
                       "reason": ev.R_NOT_UPLOADED, "gt": True, "uploaded": False}
    assert by["e"]["gt"] is False and by["a"]["uploaded"] is True
    m = ev.pose_metrics(recs, True)
    assert set(m) == set(ev.POSE_METRICS)
    plain = ev.pose_metrics(ev.photo_pose_records(rows, poses, ref), True)
    per_upload = [k for k in ev.POSE_METRICS if "testset" not in k and k != "photo_pose_not_uploaded"]
    assert {k: m[k] for k in per_upload} == {k: plain[k] for k in per_upload}
    assert (m["photo_pose_determined"], m["photo_pose_gt"], m["photo_pose_correct"]) == (3, 2, 1)
    assert m["photo_pose_determined_rate"] == 1.0 and m["photo_pose_correct_rate"] == 0.5
    assert (m["photo_pose_testset_photos"], m["photo_pose_not_uploaded"], m["photo_pose_gt_testset"]) == (5, 2, 3)
    assert m["photo_pose_determined_rate_testset"] == 0.6
    assert m["photo_pose_correct_rate_testset"] == round(1 / 3, 4)
    assert m["photo_pose_pass_rate_low_testset"] == 0.2        # only a is shown right
    assert m["photo_pose_pass_rate_high_testset"] == 0.4       # a, and c that cannot be checked
    # without planned keys (old callers) the test set is what was uploaded
    assert plain["photo_pose_testset_photos"] == 3 and plain["photo_pose_not_uploaded"] == 0
    assert plain["photo_pose_pass_rate_high_testset"] == plain["photo_pose_pass_rate"]
    s = ev.pose_summary(recs)
    assert s["not_uploaded"] == 2 and s["not_uploaded_gt"] == 1 and s["gt"] == 2
    assert s["undetermined"] == {} and s["by_source"] == {"sfm": 2, "exif": 1}
    # without a reference: the GT side is None, the test-set side still counts
    none = ev.pose_metrics(ev.photo_pose_records(rows, poses, None, planned), False)
    assert none["photo_pose_testset_photos"] == 5 and none["photo_pose_determined_rate_testset"] == 0.6
    assert none["photo_pose_gt_testset"] is None and none["photo_pose_pass_rate_low_testset"] is None


def test_a_time_capped_run_counts_the_photos_it_never_uploaded(ev, synthetic_dataset, tmp_path):
    poses = ev.synthetic_gt_poses(synthetic_dataset.manifest)
    photos = [ev.CrowdPhoto(key=p["file"], path=synthetic_dataset.path / p["file"], author=p["user_id"])
              for p in synthetic_dataset.manifest["photos"]]
    b = ev.CrowdBuilding("synthetic", "Synthetic", synthetic_dataset.lat, synthetic_dataset.lon, 150.0, photos,
                         gt={"kind": "poses", "poses": poses})
    summary = ev.run_crowd(b, tmp_path / "run", reconstructor="fake", soft_time_cap_s=0.0)
    m = summary["metrics"]
    planned = ev.planned_upload_keys(b)
    assert summary["truncated"] and m["photos_uploaded"] < len(planned)
    assert m["photo_pose_testset_photos"] == len(planned) == summary["funnel"]["listed"] - summary["funnel"]["no_gps_in_file"]
    assert m["photo_pose_not_uploaded"] == len(planned) - m["photos_uploaded"] > 0
    assert m["photo_pose_determined_rate_testset"] == round(m["photo_pose_determined"] / len(planned), 4)
    assert m["photo_pose_gt_testset"] > m["photo_pose_gt"]
    # the rescore of the recorded run, given the same planned photos, agrees
    again = ev.rescore_poses(tmp_path / "run", ev.outline_corrected_reference(b.gt, {}, None, [], None), planned)
    assert again["metrics"] == {k: m[k] for k in ev.POSE_METRICS}
    assert again["funnel"]["uploaded"] == m["photos_uploaded"]


def test_a_request_capped_run_uploads_the_prefix_and_records_the_cap(ev, synthetic_dataset, tmp_path):
    photos = [ev.CrowdPhoto(key=p["file"], path=synthetic_dataset.path / p["file"], author=p["user_id"])
              for p in synthetic_dataset.manifest["photos"]]
    b = ev.CrowdBuilding("synthetic", "Synthetic", synthetic_dataset.lat, synthetic_dataset.lon, 150.0, photos)
    full = ev.run_crowd(b, tmp_path / "full", reconstructor="fake")
    capped = ev.run_crowd(b, tmp_path / "capped", reconstructor="fake", max_requests=2)
    assert full["max_requests"] is None and capped["max_requests"] == 2
    assert full["requests_planned"] == capped["requests_planned"] == len(full["requests"]) > 2
    assert [r["label"] for r in capped["requests"]] == [r["label"] for r in full["requests"][:2]]
    assert capped["metrics"]["photos_uploaded"] == sum(r["photos"] for r in full["requests"][:2])
    assert not capped["truncated"] and full["cache"] is None               # the fake reconstructor uses no cache


def test_the_feature_cache_report_sums_every_job_and_deletes_the_cache(ev, tmp_path):
    from crowd3d.landmark_map import CacheStats, FeatureCache

    cache = FeatureCache(tmp_path / "maps" / "lm" / "cache.sqlite3")
    cache.record_use(CacheStats(0, 4, 0, 6), 0.5, "j1")
    cache.record_use(CacheStats(4, 7, 6, 21), 0.25, "j2")
    report = ev.feature_cache_report(cache.path)
    assert {k: report[k] for k in ("jobs", "features_hit", "features", "pairs_hit", "pairs", "seconds")} == {
        "jobs": 2, "features_hit": 4, "features": 11, "pairs_hit": 6, "pairs": 27, "seconds": 0.8}
    assert report["feature_hit_rate"] == round(4 / 11, 4) and report["pair_hit_rate"] == round(6 / 27, 4)
    assert report["features_cached"] == 0 and report["mb_per_photo"] is None and report["size_mb"] >= 0
    assert not cache.path.exists() and ev.feature_cache_report(cache.path) is None


# ---------------------------------------------------------------- schema 7: geographic error, coverage, cost


def _box_scene(shift=(0.0, 0.0, 0.0), yaw_deg: float = 0.0, scale: float = 1.0, bulge_north_m: float = 0.0):
    """A 40 m x 20 m box centred at the origin: its outline, dense true wall points (every 0.2 m, 3-12 m up), 8 true
    cameras on a 45 m ring looking at it, and a model of it = s Rz(yaw) p + shift of the truth (its north wall pushed
    out by bulge_north_m first: a shape error). Returns (outline, reference, model cameras, model rotations, model wall
    points, photo_key)."""
    corners = np.array([[-20.0, -10.0], [20.0, -10.0], [20.0, 10.0], [-20.0, 10.0]])
    walls = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        for t in np.linspace(0.0, 1.0, int(np.linalg.norm(b - a) / 0.2) + 1):
            for z in (3.0, 6.0, 9.0, 12.0):
                walls.append([*(a + t * (b - a)), z])
    walls = np.array(walls)
    shaped = walls.copy()
    shaped[np.isclose(shaped[:, 1], 10.0), 1] += bulge_north_m
    th = math.radians(yaw_deg)
    rz = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])

    def err(p: np.ndarray) -> np.ndarray:
        return scale * np.asarray(p, float) @ rz.T + np.asarray(shift, float)

    ref, cams, rots, photo_key = {}, {}, {}, {}
    for i, a in enumerate(np.radians(np.arange(0, 360, 45) + 10)):
        c = np.array([45.0 * math.cos(a), 45.0 * math.sin(a), 1.6])
        r = _look(c, np.array([0.0, 0.0, 6.0]))
        ref[f"k{i}"] = {"xy": c[:2], "xyz": c, "rot": r, "component": "truth", "verified": True}
        cams[f"p{i}"] = {"enu": err(c[None])[0].tolist(), "heading_deg": heading_from_en(*(rz @ r[2])[:2]), "flags": []}
        rots[f"p{i}"] = r @ rz.T
        photo_key[f"p{i}"] = f"k{i}"
    reference = {"cameras": ref, "components": {"truth": {"verified": True}}, "points": {},
                 "sigma_ref": {"yaw_deg": 0.0, "scale": 0.0}}
    return corners, reference, cams, rots, err(shaped), photo_key


def test_outline_arc_and_bins_on_a_square(ev):
    ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    d, closest, s = ev.outline_arc(np.array([[5.0, -2.0], [12.0, 5.0], [5.0, 9.0], [0.5, 5.0]]), ring, chunk=2)
    np.testing.assert_allclose(d, [2.0, 2.0, 1.0, 0.5])
    np.testing.assert_allclose(closest, [[5.0, 0.0], [10.0, 5.0], [5.0, 10.0], [0.0, 5.0]])
    np.testing.assert_allclose(s, [5.0, 15.0, 25.0, 35.0])        # arc length from vertex 0, counter-clockwise
    assert ev.outline_bins(ring) == (40, 1.0)
    assert ev.outline_bins(ring * 1.01)[0] == 40 and ev.outline_bins(ring * 1.01)[1] == pytest.approx(1.01)


def test_geo_check_recovers_a_known_georegistration_error(ev):
    """The georegistration turned the model 3 deg, scaled it 5 % and moved it (2, -1, 0.5) m: geo_check finds the
    similarity that undoes it, the cameras' geographic error and how far it moves the walls."""
    outline, reference, cams, rots, walls, photo_key = _box_scene(shift=(2.0, -1.0, 0.5), yaw_deg=3.0, scale=1.05)
    g = ev.geo_check(cams, rots, photo_key, reference, walls)
    assert g["n"] == 8 and g["orientations_used"] == 8
    assert g["yaw_deg"] == pytest.approx(-3.0, abs=0.01) and g["tilt_deg"] == pytest.approx(0.0, abs=0.01)
    assert g["scale"] == pytest.approx(1 / 1.05, abs=1e-3)
    truth = np.array([reference["cameras"][photo_key[p]]["xy"] for p in sorted(cams)])
    shown = np.array([cams[p]["enu"][:2] for p in sorted(cams)])
    assert g["camera_median_m"] == pytest.approx(float(np.median(np.linalg.norm(shown - truth, axis=1))), abs=0.01)
    s3 = g["sim3"]
    back = s3["scale"] * walls @ np.asarray(s3["rotation"]).T + np.asarray(s3["translation_m"])
    _, _, _, _, true_walls, _ = _box_scene()
    assert np.abs(back - true_walls).max() < 0.02
    assert g["walls_p90_m"] == pytest.approx(float(np.percentile(np.linalg.norm(walls - true_walls, axis=1), 90)),
                                             abs=0.02)
    # flagged cameras and unverified reference poses do not count; without orientations: the camera error only
    cams["p0"]["flags"] = ["far_from_gps"]
    reference["cameras"]["k1"]["verified"] = False
    assert ev.geo_check(cams, rots, photo_key, reference, walls)["n"] == 6
    plain = ev.geo_check(cams, {}, photo_key, reference, walls)
    assert plain["camera_median_m"] is not None and "yaw_deg" not in plain and "sim3" not in plain
    assert ev.geo_check({k: cams[k] for k in ("p2", "p3")}, rots, photo_key, reference, walls) == {"n": 2}


def test_coverage_bins_tell_the_building_local_frame_from_the_geographic_one(ev):
    """A model shifted 3 m with an exact shape is supported everywhere in the building-local frame and within 5 m but
    not 1.5 m geographically; a wall pushed out 3 m is covered but not supported locally at 1.5 m."""
    outline, reference, cams, rots, walls, photo_key = _box_scene(shift=(0.0, 3.0, 0.0))
    sim3 = ev.geo_check(cams, rots, photo_key, reference, walls)["sim3"]
    cov = ev.coverage_bins(walls, outline, sim3)
    assert cov["checked"] and len(cov["covered"]) == 120             # the whole 120 m perimeter, 1 m bins
    # per bin: the points within 1.5 m and within 5 m, building-local then geographic
    assert all(local[0] == local[1] >= ev.COV_MIN_POINTS and geo[0] == 0 and geo[1] == local[1]
               for local, geo in cov["covered"].values())
    # shown 3 m north: the east and west walls slid along themselves, the north and south ones lie 3 m off theirs,
    # within COV_NEAR_M: claimed (but for a few corner bins)
    assert 100 <= len(cov["claimed"]) <= 120
    assert all(cov["landed"][k] == local[1] for k, (local, _) in cov["covered"].items())   # all within 5 m
    assert ev.majority_bins(cov) == set(cov["covered"])
    assert ev.coverage_bins(walls, outline, None) == {"claimed": cov["claimed"], "covered": {}, "landed": {},
                                                      "checked": False}
    _, reference, cams, rots, bulged, photo_key = _box_scene(bulge_north_m=3.0)
    sim3 = ev.geo_check(cams, rots, photo_key, reference, bulged)["sim3"]
    cov = ev.coverage_bins(bulged, outline, sim3)
    off = sorted(k for k, e in cov["covered"].items() if e[0][0] < ev.COV_MIN_POINTS)
    assert 38 <= len(off) <= 42 and all(60 <= k < 100 for k in off)   # the north edge runs from 60 to 100 m
    assert all(cov["covered"][k][0][1] >= ev.COV_MIN_POINTS for k in off)         # within 5 m: covered
    # a colonnade: twice as many points seen 3 m behind the front plane do not outvote the ones on it
    _, reference, cams, rots, exact, photo_key = _box_scene()
    behind = exact[np.isclose(exact[:, 1], 10.0)] - [0.0, 3.0, 0.0]
    cov = ev.coverage_bins(np.vstack([exact, behind, behind]), outline,
                           ev.geo_check(cams, rots, photo_key, reference, exact)["sim3"])
    assert all(cov["covered"][k][0][0] >= ev.COV_MIN_POINTS for k in range(61, 99))   # the north edge
    # ... but there they are two thirds of the points landing on the bin: no majority on the wall (majority_bins)
    assert set(range(120)) - ev.majority_bins(cov) == set(range(63, 97))
    # density: the count rule supports a bin once COV_MIN_POINTS points are on the wall, whatever their share; the
    # share rule does not change when the same cloud is twice as dense
    north = exact[np.isclose(exact[:, 1], 10.0)]
    sim3 = ev.geo_check(cams, rots, photo_key, reference, exact)["sim3"]
    sparse, dense = (ev.coverage_bins(np.vstack([north[::step], *[north - [0.0, 3.0, 0.0]] * 4]), outline, sim3)
                     for step in (4, 2))
    mid = set(range(65, 95))
    assert not {k for k, e in sparse["covered"].items() if e[0][0] >= ev.COV_MIN_POINTS} & mid
    assert mid <= {k for k, e in dense["covered"].items() if e[0][0] >= ev.COV_MIN_POINTS}
    assert not (ev.majority_bins(sparse) | ev.majority_bins(dense)) & mid
    assert ev.coverage_bins(np.zeros((0, 3)), outline, sim3)["claimed"] == []


def test_geo_scores_pool_the_shown_models(ev, tmp_path, monkeypatch):
    """geo_scores on store records: the georegistered models' geographic error and calibration, the wrong-face
    counts, and the outline coverage of the models the page shows (latest per target)."""
    from crowd3d.reconstruct import write_ply
    from crowd3d.records import ModelRecord

    outline, reference, cams_a, rots_a, walls_a, photo_key = _box_scene(shift=(0.0, 3.0, 0.0))
    _, _, cams_c, rots_c, walls_c, _ = _box_scene(yaw_deg=40.0)
    rotations = {"ja": rots_a, "jc": rots_c}
    monkeypatch.setattr(ev, "model_enu_rotations", lambda job_dir, cams: rotations.get(job_dir.name, {}))

    def model(mid, sector, created, cams, walls, **meta):
        write_ply(tmp_path / f"{mid}.ply", [tuple(p) for p in walls], [(9, 9, 9)] * len(walls))
        return ModelRecord(mid, "lm", f"j{mid}", "face", sector, f"{mid}.ply", len(cams),
                           {"cameras": cams, "published": True, **meta}, created)

    far = {f"q{i}": {"enu": [50.0, 5.0 * i, 1.6], "heading_deg": 270.0, "flags": []} for i in range(4)}
    models = [model("c", 0, 1.0, cams_c, walls_c, georegistered=True, yaw_sigma_deg=5.0),    # superseded by a
              model("a", 0, 2.0, cams_a, walls_a, georegistered=True, yaw_sigma_deg=1.0, scale_sigma=0.01),
              model("b", 2, 3.0, far, walls_a[walls_a[:, 0] > 19.5], georegistered=True),  # no reference cameras
              model("x", 4, 4.0, far, walls_a, georegistered=True, published=False)]       # unpublished
    out = ev.geo_scores(models, tmp_path, outline, reference, photo_key)
    m = out["metrics"]
    assert set(m) == set(ev.GEO_METRICS)
    assert [r["model_id"] for r in out["models"]] == ["c", "a", "b"]
    assert m["geo_models_checked"] == 2 and m["geo_camera_error_median_m"] > 3.0
    assert m["geo_yaw_error_median_deg"] == pytest.approx(20.0, abs=0.05)        # 0 and 40
    assert m["wrong_face_models_published"] == 1                                 # c, turned 40 deg
    assert m["wrong_face_cameras_shown"] == 0          # the page shows a's poses (3 m off) for p0-p7, b's have no reference
    # calibration: a's yaw is right within its 1 deg, c's 40 deg are 8 of its 5-deg sigmas; c has no scale sigma
    assert (m["calib_yaw_n"], m["calib_yaw_within2"], m["calib_yaw_beyond3"]) == (2, 1, 1)
    assert m["calib_yaw_cover2"] == 0.5 and m["calib_scale_n"] == 1 and m["calib_sigma_missing"] == 1
    assert m["calib_xy_n"] is None and m["calib_tilt_cover2"] is None
    # coverage of the shown models a (checked) and b (no similarity: unverified where a does not claim)
    assert m["cov_local_supported_m_1p5"] == m["cov_local_supported_m_5"] == m["cov_geo_supported_m_5"] == 120.0
    assert m["cov_geo_supported_m_1p5"] == 0.0 and m["cov_geo_wrong_m_5"] == 0.0
    assert m["cov_local_wrong_m_1p5"] == 0.0 and m["cov_claimed_m"] >= 100.0 and m["cov_unverified_m"] >= 0.0
    a = next(r for r in out["models"] if r["model_id"] == "a")
    assert a["coverage"]["local_1.5_m"] == 120.0 and a["coverage"]["geo_1.5_m"] == 0.0
    assert a["geo"]["z_yaw"] == pytest.approx(0.0, abs=0.01) and "sim3" not in a["geo"]
    assert "coverage" not in out["models"][0]                                     # c is not shown at the end
    # a face shown turned round: its cameras look the wrong way
    turned = dict(models[1].meta, cameras={p: dict(c, heading_deg=c["heading_deg"] + 90.0) for p, c in cams_a.items()})
    models[1] = ModelRecord("a", "lm", "ja", "face", 0, "a.ply", 8, turned, 2.0)
    assert ev.geo_scores(models, tmp_path, outline, reference, photo_key)["metrics"]["wrong_face_cameras_shown"] == 8
    # without a reference only the claimed metres; without an outline no coverage
    bare = ev.geo_scores(models, tmp_path, outline, None, photo_key)["metrics"]
    assert bare["cov_claimed_m"] == bare["cov_unverified_m"] >= 100.0 and bare["geo_models_checked"] is None
    assert bare["cov_local_supported_m_1p5"] is None and bare["calib_yaw_n"] is None
    assert ev.geo_scores(models, tmp_path, None, reference, photo_key)["metrics"]["cov_claimed_m"] is None


def test_geo_scores_count_every_shown_models_wrong_walls(ev, tmp_path, monkeypatch):
    """The guards cov_*_wrong_* count a bin once, and not at all when another shown model supports it; the _sum
    companions add up every shown model's wrong walls (verifier, phase 0)."""
    from crowd3d.reconstruct import write_ply
    from crowd3d.records import ModelRecord

    outline, reference, cams, rots, exact, photo_key = _box_scene()
    bulged = _box_scene(bulge_north_m=3.0)[4]
    monkeypatch.setattr(ev, "model_enu_rotations", lambda job_dir, c: rots)

    def model(mid, sector, walls):
        write_ply(tmp_path / f"{mid}.ply", [tuple(p) for p in walls], [(9, 9, 9)] * len(walls))
        return ModelRecord(mid, "lm", f"j{mid}", "face", sector, f"{mid}.ply", len(cams),
                           {"cameras": cams, "published": True, "georegistered": True}, float(sector))

    out = ev.geo_scores([model("a", 0, exact), model("d", 4, bulged)], tmp_path, outline, reference, photo_key)
    m, d = out["metrics"], next(r["coverage"] for r in out["models"] if r["model_id"] == "d")
    assert m["cov_local_wrong_m_1p5"] == 0.0                      # a supports every bin of d's pushed-out wall
    assert 38.0 <= d["local_wrong_1.5_m"] <= 42.0 and m["cov_local_wrong_sum_m_1p5"] == d["local_wrong_1.5_m"]
    assert d["local_majority_1.5_m"] == d["local_1.5_m"] == 120.0 - d["local_wrong_1.5_m"]
    assert m["cov_local_majority_m_1p5"] == m["cov_local_supported_m_1p5"] == 120.0
    assert m["cov_geo_wrong_m_5"] == m["cov_geo_wrong_sum_m_5"] == d["geo_wrong_5_m"] == 0.0     # 3 m off: within 5 m
    alone = ev.geo_scores([model("d", 4, bulged)], tmp_path, outline, reference, photo_key)["metrics"]
    assert alone["cov_local_wrong_m_1p5"] == alone["cov_local_wrong_sum_m_1p5"] == d["local_wrong_1.5_m"]


def test_model_enu_rotations_turn_the_kept_model_into_enu(ev, tmp_path):
    """The golden reconstruction's sparse model and model JSON: the rotations turned into ENU look along the headings
    the georegistration recorded for each camera."""
    import shutil

    fixture = Path(__file__).resolve().parent / "fixtures" / "recon_golden"
    cameras = json.loads((fixture / "golden.json").read_text())["meta"]["cameras"]
    shutil.copytree(fixture / "sparse", tmp_path / "job" / "sparse" / "0")
    rots = ev.model_enu_rotations(tmp_path / "job", cameras)
    assert set(rots) == set(cameras)
    for pid, r in rots.items():
        assert np.linalg.det(r) == pytest.approx(1.0)
        assert abs(angle_diff_deg(heading_from_en(r[2, 0], r[2, 1]), cameras[pid]["heading_deg"])) < 0.05
    assert ev.model_enu_rotations(tmp_path / "none", cameras) == {}
    assert ev.model_enu_rotations(tmp_path / "job", {}) == {}


def test_version_metrics_count_a_stale_overwrite_and_the_drift(ev):
    from types import SimpleNamespace as NS

    photos = [NS(id=f"p{i}", uploaded_at=float(i)) for i in range(6)]
    jobs = [NS(id="j1", photo_ids=["p0", "p1", "p2"]), NS(id="j2", photo_ids=["p0", "p1", "p2", "p3"]),
            NS(id="j3", photo_ids=["p0", "p1"]), NS(id="j4", photo_ids=["p4", "p5"])]

    def cams(dx: float, flagged: str = "") -> dict:
        return {f"p{i}": {"enu": [10.0 * i + dx, 0.0, 1.6], "flags": ["x"] if f"p{i}" == flagged else []}
                for i in range(4)}

    models = [NS(id="m1", job_id="j1", kind="face", sector=0, created_at=1.0, meta={"cameras": cams(0.0)}),
              NS(id="m2", job_id="j2", kind="face", sector=0, created_at=2.0, meta={"cameras": cams(1.0, "p3")}),
              # an older input set (p0, p1) published over m2, which had p2 and p3 uploaded later: stale
              NS(id="m3", job_id="j3", kind="face", sector=0, created_at=3.0, meta={"cameras": cams(4.0)}),
              NS(id="m4", job_id="j4", kind="face", sector=1, created_at=4.0, meta={"cameras": {}}),
              NS(id="m5", job_id="j2", kind="face", sector=1, created_at=5.0, meta={"cameras": {}, "published": False})]
    v = ev.version_metrics(models, jobs, photos)
    assert v["stale_overwrites"] == 1
    assert v["version_drift_median_m"] == 2.0 and v["version_drift_p90_m"] == pytest.approx(2.8)    # pairs: 1 m, 3 m
    assert ev.version_metrics(models[:1], jobs, photos) == {"stale_overwrites": 0, "version_drift_median_m": None,
                                                            "version_drift_p90_m": None}


def test_request_metrics_read_the_log_and_leave_what_an_older_log_lacks_none(ev):
    log = [{"ingest_s": 0.5, "jobs_s": 10.0, "cpu_s": 30.0, "rss_before_mb": 200.0, "peak_rss_mb": 900.0,
            "jobs": [{"kind": "face"}]},
           {"ingest_s": 0.5, "jobs_s": 0.0, "cpu_s": 0.0, "rss_before_mb": 210.0, "peak_rss_mb": 210.0, "jobs": []},
           {"ingest_s": 1.0, "jobs_s": 30.0, "cpu_s": 70.0, "rss_before_mb": 400.0, "peak_rss_mb": 1500.0,
            "jobs": [{"kind": "full"}]}]
    rows = [{"time_to_pose_s": 10.5}, {"time_to_pose_s": 40.0}, {"time_to_pose_s": None}, {}]
    m = ev.request_metrics(log, rows)
    assert set(m) == set(ev.REQUEST_METRICS)
    assert m["jobs_cpu_s"] == 100.0 and (m["peak_rss_jobs_mb"], m["baseline_rss_mb"]) == (1500.0, 400.0)
    assert m["request_latency_p50_s"] == 20.8 and m["request_latency_p90_s"] == 28.9     # 10.5 and 31 s
    assert m["photo_time_to_pose_p50_s"] == 25.2 and m["photo_time_to_pose_p90_s"] == 37.0
    old = ev.request_metrics([{k: r[k] for k in ("ingest_s", "jobs_s", "jobs")} for r in log], [{}])
    assert old["request_latency_p90_s"] == 28.9
    assert all(old[k] is None for k in ev.REQUEST_METRICS if not k.startswith("request_latency"))


def test_peak_memory_and_cpu_time(ev, monkeypatch, tmp_path):
    assert ev.cpu_seconds() > 0
    if ev.reset_peak_rss():
        big = np.ones(40 * 2**20 // 8)       # 40 MB touched after the reset
        assert ev.peak_rss_mb() >= ev._status_mb("VmRSS") >= 40 and big[-1] == 1.0
    # where /proc does not tell: the process peak (ru_maxrss), and no reset
    monkeypatch.setattr(ev, "_status_mb", lambda field: None)
    assert ev.peak_rss_mb() > 0
    real_open = open

    def no_clear_refs(path, *a, **k):
        if str(path).endswith("clear_refs"):
            raise PermissionError(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", no_clear_refs)
    assert ev.reset_peak_rss() is False
    (tmp_path / "keep").mkdir()
    (tmp_path / "work").mkdir()
    (tmp_path / "keep" / "a").write_bytes(b"x" * 2**20)
    (tmp_path / "work" / "b").write_bytes(b"x" * 2**20)
    (tmp_path / "keep" / "link").symlink_to(tmp_path / "work" / "b")
    assert ev.dir_size_mb(tmp_path, ("work",)) == 1.0 and ev.dir_size_mb(tmp_path) == 2.0


def test_calib_z_one_and_two_axes(ev):
    assert ev.calib_z("yaw", -6.0, 3.0, 0.0) == 2.0
    assert ev.calib_z("scale", 0.1, 0.06, 0.08) == 1.0
    assert ev.calib_z("yaw", 1.0, None, 1.0) is None and ev.calib_z("yaw", 1.0, 0.0, 0.0) is None
    # 2-D: the radius holding 0.954 of a 2-D normal (2.486 sigma) is z = 2, the one holding 0.683 (1.515) is z = 1
    assert ev.calib_z("xy", [2.486, 0.0], 1.0, 0.0) == pytest.approx(2.0, abs=0.002)
    assert ev.calib_z("tilt", [0.0, 1.515], 0.6, 0.8) == pytest.approx(1.0, abs=0.002)
    assert ev.calib_z("xy", [0.0, 0.0], 1.0, 0.0) == 0.0
    assert ev.calib_z("xy", [30.0, 40.0], 1.0, 0.0) == 50.0       # past the float tail: the radius itself
    rng = np.random.default_rng(0)                               # 2-D normal errors cover like 1-D ones
    z = np.array([ev.calib_z("xy", list(e), 1.0, 0.0) for e in rng.normal(0, 1, (4000, 2))])
    assert [float(np.mean(z <= k)) for k in (1, 2, 3)] == pytest.approx([0.683, 0.954, 0.997], abs=0.015)


@pytest.mark.parametrize("scale", [-0.0002, 0.0])
def test_model_calibration_skips_the_scale_of_a_degenerate_similarity(ev, scale):
    """geo_check can return a similarity scale <= 0 (B0: mdx Aachen r1, -0.0002): no scale error and no scale z,
    the other components as usual (it raised a math domain error and lost the building run)."""
    from crowd3d import params
    from crowd3d.anchor import GeoAnchor, floored_sigma

    sim3 = {"scale": scale, "rotation": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0]}
    geo = {"yaw_deg": 3.0, "scale": scale}
    reference = {"sigma_ref": {"yaw_deg": 1.0, "scale": 0.02}, "components": {}}
    legacy = ev.model_calibration(geo, sim3, None, {"yaw_sigma_deg": 2.0, "scale_sigma": 0.05}, reference)
    assert set(legacy["calib"]) == {"yaw"} and legacy["calib"]["yaw"]["e"] == -3.0
    core = {"yaw_deg": 1.5, "log_scale": 0.02, "xy_m": 1.0, "tilt_deg": 0.3}
    cal = {"id": "c", "floors": {"yaw_deg": 0.0, "log_scale": 0.0, "xy_m": 0.0, "tilt_deg": 0.0}}
    anchor = GeoAnchor("job", "full", 1.0, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0,
                       [30.0, 0.0, 1.6], floored_sigma(core, 1.5, cal["floors"]), {}, {"sigma_core": core}, cal)
    with params.override(anchor_calibration_id="c", anchor_yaw_floor_deg=0.0, anchor_scale_floor=0.0,
                         anchor_xy_floor_m=0.0, anchor_tilt_floor_deg=0.0):
        out = ev.model_calibration(geo, sim3, anchor.to_meta(), {}, reference)
    assert "scale" not in out["calib"] and {"yaw", "tilt"} <= set(out["calib"])


def test_geo_scores_calibrate_every_component_from_the_anchor(ev, tmp_path, monkeypatch):
    """A model with a GeoAnchor: yaw, log-scale, east/north at its pivot and tilt, each against the anchor's sigma
    with the current floors; with rescore --regeo's anchors a model without a replayed one is left out."""
    from crowd3d import params
    from crowd3d.anchor import GeoAnchor, floored_sigma
    from crowd3d.reconstruct import write_ply
    from crowd3d.records import ModelRecord

    outline, reference, cams, rots, walls, photo_key = _box_scene(shift=(2.0, -1.0, 0.0), yaw_deg=4.0, scale=1.05)
    monkeypatch.setattr(ev, "model_enu_rotations", lambda job_dir, c: rots)
    write_ply(tmp_path / "a.ply", [tuple(p) for p in walls], [(9, 9, 9)] * len(walls))
    core = {"yaw_deg": 1.5, "log_scale": 0.02, "xy_m": 1.0, "tilt_deg": 0.3}
    cal = {"id": "made-with", "floors": {"yaw_deg": 0.0, "log_scale": 0.0, "xy_m": 0.0, "tilt_deg": 0.0}}
    anchor = GeoAnchor("job", "full", 1.0, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0,
                       [30.0, 0.0, 1.6], floored_sigma(core, 1.5, cal["floors"]), {}, {"sigma_core": core}, cal)
    meta = {"cameras": cams, "published": True, "georegistered": True, "yaw_sigma_deg": 9.0, "anchor": anchor.to_meta()}
    m = ModelRecord("a", "lm", "ja", "face", 0, "a.ply", len(cams), meta, 1.0)
    with params.override(anchor_yaw_floor_deg=0.0, anchor_xy_floor_m=2.0, anchor_calibration_id="current"):
        out = ev.geo_scores([m], tmp_path, outline, reference, photo_key)
    g = out["models"][0]["geo"]
    assert (g["sigma_from"], g["calibration_id"], g["yaw_sigma_deg"]) == ("anchor", "current", 9.0)
    c = g["calib"]
    assert c["yaw"]["e"] == pytest.approx(4.0, abs=0.02) and c["yaw"]["sigma"] == 1.5     # model minus reference
    assert c["scale"]["e"] == pytest.approx(math.log(1.05), abs=1e-3)
    th = math.radians(-4.0)                                  # the pivot's reference place: the model's error undone
    q = (np.array([30.0, 0.0]) - [2.0, -1.0]) / 1.05
    want = np.array([30.0, 0.0]) - np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) @ q
    assert c["xy"]["e"] == pytest.approx(want.tolist(), abs=0.01)
    assert (c["xy"]["core"], c["xy"]["ref"]) == (1.0, 0.0) and c["xy"]["sigma"] == pytest.approx(math.sqrt(5.0))
    assert c["xy"]["z"] == pytest.approx(ev.calib_z("xy", want.tolist(), math.sqrt(5.0), 0.0), abs=0.01)
    assert c["tilt"]["e"] == pytest.approx([0.0, 0.0], abs=0.01) and c["tilt"]["z"] == pytest.approx(0.0, abs=0.05)
    mt = out["metrics"]
    assert mt["calib_yaw_n"] == mt["calib_scale_n"] == mt["calib_xy_n"] == mt["calib_tilt_n"] == 1
    assert (mt["calib_yaw_within2"], mt["calib_tilt_within2"], mt["calib_sigma_missing"]) == (0, 1, 0)
    (row,) = out["calibration"]
    assert row["key"] == ev.model_key(cams, photo_key) and row["comps"] == c and row["sigma_from"] == "anchor"
    # rescore --regeo: the replayed anchors by model id; a model without one is left out of the calibration
    regeo = ev.geo_scores([m], tmp_path, outline, reference, photo_key, anchors={})
    assert regeo["models"][0]["geo"]["calib"] == {} and regeo["calibration"] == []
    assert regeo["metrics"]["calib_yaw_n"] == regeo["metrics"]["calib_xy_n"] == 0
    assert regeo["metrics"]["geo_models_checked"] == mt["geo_models_checked"]


def test_regeo_models_replays_each_published_models_georegistration(ev, tmp_path):
    """rescore --regeo: a published model's job kept its SfM model; the replay with the photos' rows reproduces the
    model JSON's cameras (faithful: the replayed anchor), a model JSON whose cameras the replay does not reproduce
    within REGEO_MAX_DIFF_M, or without a kept model, is left out."""
    import shutil
    from types import SimpleNamespace

    import pycolmap

    from crowd3d.geo import LocalENU
    from crowd3d.reconstruct import ReconPhoto, georegister_inputs, georegister_model
    from crowd3d.records import ModelRecord

    from crowd3d import params

    fixture = Path(__file__).parent / "fixtures" / "recon_golden"
    golden = json.loads((fixture / "golden.json").read_text(encoding="utf-8"))
    lm = SimpleNamespace(lat=31.2397, lon=121.4998)
    photos = [SimpleNamespace(id=p[0], lat=p[2], lon=p[3], heading_deg=p[4], gps_accuracy_m=p[5], focal_35mm=p[6])
              for p in golden["photos"]]
    sparse = tmp_path / "work" / "j1" / "sparse" / "0"
    shutil.copytree(fixture / "sparse", sparse)
    rec, enu = pycolmap.Reconstruction(str(sparse)), LocalENU(lm.lat, lm.lon)
    by_name = {f"{k:05d}_{p.id}.jpg": ReconPhoto(p.id, Path(), p.lat, p.lon, p.heading_deg, p.gps_accuracy_m)
               for k, p in enumerate(photos)}
    with params.override(compass_sigma_deg=30.0):          # a run with another setting: the replay must use it
        geo, cams = georegister_model(georegister_inputs(rec, by_name, {n: np.array(enu.to_enu(p.lat, p.lon)[:2])
                                                                         for n, p in by_name.items()}), 1.7)
    shutil.copytree(sparse, tmp_path / "work" / "j2" / "sparse" / "0")
    off = {pid: dict(c, enu=[c["enu"][0] + 0.05, *c["enu"][1:]]) for pid, c in cams.items()}
    models = [ModelRecord(mid, "lm", job, "face", 2, f"{mid}.ply", 4, {"published": pub, "cameras": c}, t)
              for mid, job, c, pub, t in (("good", "j1", cams, True, 1.0), ("moved", "j2", off, True, 2.0),
                                          ("lost", "j3", cams, True, 3.0), ("hidden", "j1", cams, False, 4.0))]
    summary = {"effective_params": {"params": {"compass_sigma_deg": 30.0, "anchor_xy_floor_m": 99.0},
                                    "settings": {"camera_height_m": 1.7}}}
    with params.override(anchor_xy_floor_m=2.5):
        anchors, report = ev.regeo_models(models, photos, lm, tmp_path, summary)
    assert set(anchors) == {"good", "moved", "lost"} and anchors["moved"] is None and anchors["lost"] is None
    assert (report["published"], report["replayed"], report["faithful"]) == (3, 2, 1)
    assert report["max_diff_m"] == pytest.approx(0.05, abs=1e-3)
    assert [x["model_id"] for x in report["left_out"]] == ["moved", "lost"]
    a = anchors["good"]
    # the run's own parameters, the current calibration (not the run's anchor floor)
    assert a["transform"]["translation"] == pytest.approx(geo.anchor.translation, abs=1e-9)
    assert a["calibration"]["floors"]["xy_m"] == 2.5
    assert a["sigma"]["east_m"] == pytest.approx(math.hypot(a["raw"]["sigma_core"]["xy_m"], 2.5), abs=1e-6)
    for other in ({"params": {}, "settings": {"camera_height_m": 1.7}},
                  {"params": {"compass_sigma_deg": 30.0}, "settings": {}}):       # not the run's parameters
        assert ev.regeo_models(models[:1], photos, lm, tmp_path, {"effective_params": other})[0]["good"] is None
