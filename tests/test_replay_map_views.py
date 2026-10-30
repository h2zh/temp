"""scripts/replay_map_views.py (stage 2a of the landmark map): the view cut and the go/no-go summary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rmv():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("replay_map_views", ROOT / "scripts" / "replay_map_views.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_a_view_keeps_its_sectors_cameras_and_the_points_two_of_them_observe(rmv):
    photo_ids = ["a", "b", "c", "d"]
    tracks = [[0, 1], [1, 2], [2, 3], [0, 0], [3], [0, 2, 3], []]
    images, points = rmv.cut_view(photo_ids, tracks, {"a", "b"})
    assert images.tolist() == [True, True, False, False]
    assert points.tolist() == [True, False, False, True, False, False, False]
    images, points = rmv.cut_view(photo_ids, tracks, {"b", "c", "d"})
    assert points.tolist() == [False, True, True, False, False, True, False]
    assert not rmv.cut_view(photo_ids, tracks, set())[1].any()


def test_view_inputs_keep_the_views_cameras_and_need_three(rmv):
    from crowd3d.reconstruct import GeoregisterInputs

    n = 5
    full = GeoregisterInputs(images=list(range(n)), photos=list("abcde"), rots=np.tile(np.eye(3), (n, 1, 1)),
                             centers=np.arange(3 * n, dtype=float).reshape(n, 3), focal=np.array([1, 1, 50, 1, 1.0]),
                             gps_en=np.zeros((n, 2)), accuracy=np.full(n, np.nan), use=np.ones(n, bool),
                             xyz=np.zeros((9, 3)))
    view = rmv.view_inputs(full, np.array([True, False, True, True, True]), np.ones((4, 3)))
    assert view.photos == list("acde") and view.centers.tolist() == full.centers[[0, 2, 3, 4]].tolist()
    assert view.use.tolist() == [True, False, True, True] and view.xyz.shape == (4, 3)   # focal 50: degenerate
    assert rmv.view_inputs(full, np.array([True, True, False, False, False]), np.ones((4, 3))) is None


def test_the_summary_and_the_stage_2b_rule(rmv):
    def model(published, gross=0, n=10, pos=1.0, rot=0.5, outline=3.0):
        return {"published": published, "gt": {"gross_shown": gross if published else 0, "n": n,
                                                "position_error_m": {"median": pos},
                                                "rotation_error_deg": {"median": rot}},
                "outline": {"rotation_deg": -outline}}

    rows = [{"face": model(True, gross=2, pos=2.0), "view": model(True, pos=1.0), "component": model(False),
             "final": True},
            {"face": model(True, pos=1.0), "view": model(False), "component": model(False), "final": False},
            {"face": None, "view": model(True, gross=1, pos=3.0), "component": model(True), "final": True}]
    s = rmv.summarize_rows(rows)
    assert {k: s[k]["facades_published"] for k in rmv.VARIANTS} == {"face": 2, "view": 2, "component": 1,
                                                                   "hybrid": 3}
    assert {k: s[k]["gt_gross_shown"] for k in rmv.VARIANTS} == {"face": 2, "view": 1, "component": 0, "hybrid": 1}
    assert s["face"]["gt_position_median_m"] == 1.5 and s["hybrid"]["gt_position_median_m"] == 1.0
    assert s["view"]["outline_rotation_abs_deg"] == 3.0
    assert rmv.verdict(s, "view") == {"checks": {"facades": True, "gross_shown": True, "gt_position_median_m": False,
                                                 "gt_rotation_median_deg": True}, "go": False}
    assert rmv.verdict(s, "hybrid")["go"] is True and rmv.verdict(s, "component")["go"] is False
    scoped = rmv.scopes(rows)
    assert scoped["all"]["pairs"] == 3 and scoped["final"]["pairs"] == 2
    assert scoped["final"]["face"]["facades_published"] == 1
