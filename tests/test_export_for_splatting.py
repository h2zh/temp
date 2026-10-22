import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "export_for_splatting.py"


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("export_for_splatting", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    a, b = math.radians(yaw_deg), math.radians(pitch_deg)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rx @ rz


def _write_model(root: Path, model: str, params: list[float], size=(64, 48)) -> tuple[Path, Path, dict]:
    """A two-image COLMAP text model whose observations are exact projections of four points."""
    sparse, images = root / "sparse", root / "images"
    sparse.mkdir(parents=True)
    images.mkdir()
    w, h = size
    f, cx, cy = params[0], params[1], params[2]
    points = {1: [0.2, -0.1, 4.0], 2: [-0.3, 0.2, 5.0], 3: [0.1, 0.3, 4.5], 4: [-0.2, -0.25, 3.5]}
    poses = {1: (_rot(0, 0), np.array([0.0, 0.0, 0.0])), 2: (_rot(8, -3), np.array([-0.4, 0.05, 0.1]))}
    obs: dict[int, list] = {1: [], 2: []}
    for iid, (r, t) in poses.items():
        for pid, x in points.items():
            xc = r @ np.array(x) + t
            obs[iid].append((f * xc[0] / xc[2] + cx, f * xc[1] / xc[2] + cy, pid))
    (sparse / "cameras.txt").write_text(f"1 {model} {w} {h} {' '.join(map(str, params))}\n")
    lines = []
    for iid, (r, t) in poses.items():
        q = _quat(r)
        lines.append(f"{iid} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 img{iid}.jpg")
        lines.append(" ".join(f"{u} {v} {pid}" for u, v, pid in obs[iid]))
        Image.fromarray(np.random.default_rng(iid).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(
            images / f"img{iid}.jpg")
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")
    pts = [f"{pid} {x[0]} {x[1]} {x[2]} 200 100 50 0.1 1 {k} 2 {k}" for k, (pid, x) in enumerate(points.items())]
    (sparse / "points3D.txt").write_text("\n".join(pts) + "\n")
    return sparse, images, {"poses": poses, "obs": obs}


def _quat(r: np.ndarray) -> list[float]:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    return [w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)]


def test_quaternion_matches_rotation(ex):
    for yaw, pitch in [(0, 0), (30, 10), (179, -40), (-120, 85)]:
        r = _rot(yaw, pitch)
        w, x, y, z = ex.quat_wxyz(r)
        back = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        np.testing.assert_allclose(back, r, atol=1e-9)
        assert w >= 0


def test_pinhole_export_scales_intrinsics_and_observations_and_keeps_poses(ex, tmp_path):
    sparse, images, truth = _write_model(tmp_path / "in", "SIMPLE_PINHOLE", [50.0, 32.0, 24.0])
    stats = ex.export(sparse, images, tmp_path / "out", max_side=32)
    assert stats["images"] == 2 and stats["points"] == 4
    assert stats["reprojection_error_px"] < 1e-3
    cams = (tmp_path / "out/sparse/0/cameras.txt").read_text().splitlines()[1].split()
    assert cams[1:4] == ["PINHOLE", "32", "24"]
    np.testing.assert_allclose([float(v) for v in cams[4:]], [25.0, 25.0, 16.0, 12.0], atol=1e-6)
    lines = (tmp_path / "out/sparse/0/images.txt").read_text().splitlines()[2:]
    for k in range(2):
        head, obs = lines[2 * k].split(), lines[2 * k + 1].split()
        iid = int(head[0])
        r, t = truth["poses"][iid]
        np.testing.assert_allclose([float(v) for v in head[1:5]], ex.quat_wxyz(r), atol=1e-8)
        np.testing.assert_allclose([float(v) for v in head[5:8]], t, atol=1e-6)
        got = np.array(obs, dtype=float).reshape(-1, 3)
        want = np.array(truth["obs"][iid])
        np.testing.assert_allclose(got[:, :2], want[:, :2] / 2, atol=1e-3)
    with Image.open(tmp_path / "out/images/img1.jpg") as im:
        assert im.size == (32, 24)
        assert not im.info.get("exif")


def test_distorted_cameras_are_undistorted_first(ex, tmp_path):
    sparse, images, _ = _write_model(tmp_path / "in", "SIMPLE_RADIAL", [50.0, 32.0, 24.0, 0.05])
    stats = ex.export(sparse, images, tmp_path / "out", max_side=1600)
    assert stats["images"] == 2
    assert (tmp_path / "out/sparse/0/cameras.txt").read_text().splitlines()[1].split()[1] == "PINHOLE"


def test_max_images_keeps_the_best_connected(ex, tmp_path):
    sparse, images, _ = _write_model(tmp_path / "in", "PINHOLE", [50.0, 50.0, 32.0, 24.0])
    stats = ex.export(sparse, images, tmp_path / "out", max_images=1)
    assert stats["images"] == 1
    assert stats["points"] == 0     # every track needs two of the kept images
