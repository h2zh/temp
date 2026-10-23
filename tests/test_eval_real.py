"""Pure helpers of scripts/eval_real_landmark.py, on synthetic numbers (no COLMAP, no real photos)."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from crowd3d.exif import build_exif_bytes, read_photo_meta
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


def test_simulate_tags_without_noise_round_trips_and_noise_has_the_requested_spread(ev):
    enu = LocalENU(43.773056, 11.254722)
    rng = np.random.default_rng(0)
    lat, lon, heading = ev.simulate_tags(40.0, -25.0, 359.0, enu, rng, 0.0, 0.0)
    e, n, _ = enu.to_enu(lat, lon)
    assert (e, n) == (pytest.approx(40.0, abs=1e-6), pytest.approx(-25.0, abs=1e-6))
    assert heading == pytest.approx(359.0)

    samples = [ev.simulate_tags(40.0, -25.0, 350.0, enu, rng, 5.0, 15.0) for _ in range(3000)]
    en = np.array([enu.to_enu(la, lo)[:2] for la, lo, _ in samples])
    dh = np.array([angle_diff_deg(h, 350.0) for _, _, h in samples])
    assert en.mean(axis=0) == pytest.approx([40.0, -25.0], abs=0.4)
    assert en.std(axis=0) == pytest.approx([5.0, 5.0], rel=0.06)
    assert dh.std() == pytest.approx(15.0, rel=0.06)
    assert all(0.0 <= h < 360.0 for _, _, h in samples)

    again = ev.simulate_tags(40.0, -25.0, 350.0, enu, np.random.default_rng(7), 5.0, 15.0)
    assert again == ev.simulate_tags(40.0, -25.0, 350.0, enu, np.random.default_rng(7), 5.0, 15.0)


def _jpeg(orientation: int | None, **exif) -> bytes:
    rng = np.random.default_rng(3)
    img = Image.fromarray((rng.random((48, 64, 3)) * 255).astype(np.uint8))
    blob = build_exif_bytes(make="PhoneCo", model="P1", **exif)
    if orientation is not None:
        import piexif

        ex = piexif.load(blob)
        ex["0th"][piexif.ImageIFD.Orientation] = orientation
        blob = piexif.dump(ex)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90, exif=blob)
    return buf.getvalue()


def test_tag_jpeg_writes_tags_and_keeps_pixels_and_other_exif(ev):
    data = _jpeg(6)
    out = ev.tag_jpeg(data, lat=43.7731, lon=11.2556, heading_deg=271.25)
    meta = read_photo_meta(out)
    assert (meta.lat, meta.lon) == (pytest.approx(43.7731, abs=1e-7), pytest.approx(11.2556, abs=1e-7))
    assert meta.heading_deg == pytest.approx(271.25) and meta.heading_ref == "T"
    assert meta.orientation == 6 and (meta.make, meta.model) == ("PhoneCo", "P1")
    before = np.asarray(Image.open(io.BytesIO(data)))
    after = np.asarray(Image.open(io.BytesIO(out)))
    np.testing.assert_array_equal(before, after)


def test_tag_jpeg_adds_only_the_missing_heading(ev):
    data = _jpeg(None, lat=-33.5, lon=151.25)
    meta = read_photo_meta(ev.tag_jpeg(data, heading_deg=10.0))
    assert (meta.lat, meta.lon, meta.heading_deg) == (pytest.approx(-33.5), pytest.approx(151.25), pytest.approx(10.0))


def test_tag_jpeg_handles_a_photo_without_exif(ev):
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(buf, "JPEG")
    meta = read_photo_meta(ev.tag_jpeg(buf.getvalue(), lat=1.0, lon=2.0))
    assert (meta.lat, meta.lon, meta.heading_deg) == (pytest.approx(1.0), pytest.approx(2.0), None)


def test_assign_users_is_balanced_and_seeded(ev):
    users = ev.assign_users(23, 5, np.random.default_rng(4))
    counts = np.bincount(users, minlength=5)
    assert counts.sum() == 23 and counts.max() - counts.min() <= 1
    assert users == ev.assign_users(23, 5, np.random.default_rng(4))
    assert users != ev.assign_users(23, 5, np.random.default_rng(5))


def test_sector_of_en_follows_the_coverage_convention(ev):
    assert [ev.sector_of_en(*p) for p in [(0, 10), (10, 10), (10, 0), (0, -10), (-10, 0), (-10, 10)]] == [0, 1, 2, 4, 6, 7]


def test_split_stages_puts_the_most_popular_sector_first(ev):
    def up(sector):
        return ev.Upload("x.jpg", b"", "user-1", None, None, None, "real", "real", exif_sector=sector)

    uploads = [up(2), up(1), up(2), up(1), up(None), up(3)]
    assert ev.split_stages(uploads) == 1   # tie between 1 and 2: lower index
    assert [u.stage for u in uploads] == [2, 1, 2, 1, 2, 2]
    assert ev.split_stages([up(None)]) is None


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


def test_footprint_pose_and_sensor_check(ev):
    from crowd3d.exif import PhotoMeta

    enu = LocalENU(43.773056, 11.254722)
    lat, lon, _ = enu.to_geodetic(0.0, 13.0)          # GPS 3 m north of the corrected camera
    gt = ev.GroundTruth(
        images=[ev.GtImage("a.jpg", np.eye(3), np.zeros(3), PhotoMeta(800, 600, lat, lon, heading_deg=185.0), 1.0),
                ev.GtImage("b.jpg", np.eye(3), np.zeros(3), PhotoMeta(800, 600), 1.0)],
        points=np.zeros((0, 3)), summary={})
    frame = ev.GtFrame(np.eye(4), {}, poses={n: {"enu": np.array([10.0, 0.0, 1.6]), "heading_deg": 270.0}
                                             for n in ("a.jpg", "b.jpg")})
    assert frame.footprint_pose("a.jpg") is None
    frame.footprint_fix = (90.0, np.zeros(2))           # turn the GT 90 degrees counter-clockwise
    xy, heading = frame.footprint_pose("a.jpg")
    np.testing.assert_allclose(xy, [0.0, 10.0], atol=1e-9)
    assert heading == pytest.approx(180.0)
    check = ev.sensor_check(gt, frame, enu)
    assert check["real_gps_error_m"]["n"] == 1 and check["real_gps_error_m"]["median"] == pytest.approx(3.0, abs=1e-3)
    assert check["real_compass_signed_median_deg"] == pytest.approx(5.0)


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
    snap, cand = st.snap_to_footprint(off, outline), ev.snap_candidate(off, outline)
    assert snap.accepted and cand["rotation_deg"] == pytest.approx(snap.rotation_deg, abs=0.1)
    assert cand["shift_m"] == pytest.approx(float(np.linalg.norm(snap.shift)), abs=0.1)
    # A model 30 deg off: beyond the default yaw window (+-20 deg) the snap is refused and reset to the identity,
    # the candidate keeps the fit.
    far = moved(pts, 30.0, (0.0, 0.0))
    snap = st.snap_to_footprint(far, outline)
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
    assert at_anchor["verdict"] == "right" and at_anchor["correction_after"]["at_walls_m"] < 1e-6
    assert over_walls["verdict"] == "wrong"
    assert over_walls["correction_after"]["walls_p90_m"] == pytest.approx(0.9 * 30 * math.radians(4.5), abs=0.1)
    assert over_walls["correction_before"]["walls_p90_m"] == over_walls["correction_after"]["walls_p90_m"]
    short = walls[np.abs(walls[:, 0]) <= 10.0]
    scaled = {k: {"enu": [*(1.04 * np.asarray(ref_xy[photo_key[k]])), 1.6]} for k in photo_key}
    small = ev.reference_check(scaled, photo_key, reference, None, anchor, short)
    assert small["verdict"] == "right" and small["correction_after"]["walls_p90_m"] < 0.5
    assert not ev.correction_is_right({"rotation_deg": 0.0, "at_walls_m": 0.0, "walls_p90_m": 0.0, "scale": 1.12})
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
    right = {"verdict": "right", "error_after_m": {"median": 1.5}}
    wrong = {"verdict": "wrong", "error_after_m": {"median": 14.0}}
    records = [
        style_record(id="a", reference=right),
        # a pass whose georegistration was already right: not "refused although placed right"
        style_record(id="a2", reference={"verdict": "right", "error_after_m": {"median": 0.5},
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
    assert m["style_rejected_placed_right"] == 1           # c: 1.4 m over the walls; c2: 2 m is misplaced
    assert m["style_gate_passed_not_eligible"] == 1        # d
    assert m["style_snap_residual_before_m"] == 4.0 and m["style_snap_residual_m"] == 0.3
    assert m["style_ref_camera_error_m"] == pytest.approx(1.5)
    assert m["style_ref_camera_error_worst_m"] == pytest.approx(14.0)
    assert ev.best_style_record(records)["model_id"] == "b"
    assert (m["style_walls_coloured"], m["style_walls_unlocked"], m["style_glb_kb_gz"]) == (5, 3, 40.0)
    assert m["style_time_s"] == 14.0 and m["style_qa_failed"] == 0
    empty = ev.style_metrics([])
    assert empty["style_registration_pass_rate"] is None and empty["style_models_eligible"] == 0


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
