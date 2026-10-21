"""Pure helpers of scripts/eval_real_landmark.py, on synthetic numbers (no COLMAP, no real photos)."""

from __future__ import annotations

import importlib.util
import io
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
