"""GeoAnchor: where a reconstruction's own (local) frame sits on the map, and how sure that is.

A reconstruction keeps its SfM frame. The anchor is a small, versioned record next to it: the similarity "ENU from
local" (p_enu = scale R p_local + translation, the pycolmap.Sim3d convention), a 1-sigma and a source for each
component, the uncalibrated ingredients those sigmas came from, and the calibration (floors) applied to them. Every
ENU quantity of a model is the anchor applied to its local value. reconstruct.georegister builds it; the model JSON
stores it as meta["anchor"] (schema ANCHOR_SCHEMA) next to the per-camera local poses (cameras[pid]["local"]). It only
records: no consumer reads it yet (the legacy fields yaw_sigma_deg, scale_sigma, georegistered keep their formulas).

Sigmas (per component, docs/ARCHITECTURE.md 3.5.9):
- yaw_deg, log_scale, east_m / north_m (at pivot_enu), tilt_deg (per horizontal axis): sqrt(core^2 + floor^2), the
  core from the GPS fit and the camera orientations (raw["sigma_core"]), the floor from Params (anchor_*_floor*);
- up_m: Params.anchor_z_sigma_m (the camera-height assumption; not calibratable against the references, which make
  the same assumption).
About pivot_enu (the weighted centre of the GPS-inlier cameras) the weighted least-squares fit leaves yaw, log-scale
and translation uncorrelated, so the six sigmas are the whole (diagonal) covariance; position_sigma_m carries them to
any point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import params
from .geo import heading_from_en

ANCHOR_SCHEMA = "crowd3d.anchor/1"
LEVELS = ("full", "prior_only", "none")      # ARCHITECTURE 3.5.9 georeg_level; georegistered == (level == "full")
FLOORED = {"yaw_deg": "anchor_yaw_floor_deg", "log_scale": "anchor_scale_floor", "xy_m": "anchor_xy_floor_m",
           "tilt_deg": "anchor_tilt_floor_deg"}   # raw["sigma_core"] component -> its Params floor


def quat_xyzw(r: np.ndarray) -> list[float]:
    """Unit quaternion (x, y, z, w; w >= 0) of a proper 3x3 rotation."""
    m = np.asarray(r, dtype=float)
    t = float(np.trace(m))
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 1e-300))
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    q = np.asarray(q) / np.linalg.norm(q)
    return (q if q[3] >= 0 else -q).tolist()


def quat_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    """3x3 rotation of a quaternion (x, y, z, w)."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def floors(p: params.Params | None = None) -> dict[str, float]:
    """The calibration floors of a parameter set (default: the current one), per raw["sigma_core"] component."""
    p = params.current() if p is None else p
    return {c: float(getattr(p, name)) for c, name in FLOORED.items()}


def calibration(p: params.Params | None = None) -> dict[str, Any]:
    p = params.current() if p is None else p
    return {"id": p.anchor_calibration_id, "floors": floors(p)}


def floored_sigma(core: dict[str, float], up_m: float, floor: dict[str, float]) -> dict[str, float]:
    """The anchor's sigma block: each core component with its floor in quadrature (up: as given)."""
    def f(c: str) -> float:
        return round(math.hypot(core[c], floor[c]), 6)

    xy = f("xy_m")
    return {"yaw_deg": f("yaw_deg"), "log_scale": f("log_scale"), "east_m": xy, "north_m": xy,
            "up_m": round(float(up_m), 6), "tilt_deg": f("tilt_deg")}


@dataclass(frozen=True)
class GeoAnchor:
    frame: str                          # whose local frame: "job" (this model's job, work/<job>/sparse/0), "fake"
    level: str                          # LEVELS
    scale: float                        # p_enu = scale R p_local + translation
    rotation_xyzw: list[float]
    translation: list[float]
    up_local: list[float]               # the parts of R: R = rotation_z(yaw_deg) @ rotation_to_z(up_local)
    yaw_deg: float
    pivot_enu: list[float]              # where yaw, log-scale and east/north are uncorrelated
    sigma: dict[str, float]             # yaw_deg, log_scale, east_m, north_m, up_m, tilt_deg (1 sigma)
    source: dict[str, str]              # per component: yaw, scale, xy, up, gravity
    raw: dict[str, Any] = field(default_factory=dict)          # uncalibrated ingredients ("sigma_core" among them)
    calibration: dict[str, Any] = field(default_factory=dict)  # {"id", "floors"} the sigma block was made with

    # ------------------------------------------------------------ geometry
    def rotation(self) -> np.ndarray:
        return quat_matrix(self.rotation_xyzw)

    def matrix(self) -> np.ndarray:
        """4x4 similarity local -> ENU."""
        t = np.eye(4)
        t[:3, :3] = self.scale * self.rotation()
        t[:3, 3] = self.translation
        return t

    def to_enu(self, points: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(points, dtype=float) @ self.rotation().T + np.asarray(self.translation)

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - np.asarray(self.translation)) @ self.rotation() / self.scale

    def camera_enu(self, q_local: list[float], c_local: list[float]) -> tuple[np.ndarray, np.ndarray, float]:
        """A camera's local pose (cam_from_local quaternion, centre) in ENU: (centre, cam_from_enu, heading_deg)."""
        cam_from_enu = quat_matrix(q_local) @ self.rotation().T
        view = cam_from_enu[2]
        return self.to_enu(np.asarray(c_local, dtype=float)[None])[0], cam_from_enu, heading_from_en(view[0], view[1])

    def position_sigma_m(self, p_enu: np.ndarray | list[float]) -> tuple[float, float]:
        """(horizontal, vertical) 1-sigma position error of an ENU point: h^2 = sigma_xy^2 + d^2 (sigma_yaw^2 +
        sigma_logscale^2), v^2 = sigma_up^2 + d^2 sigma_tilt^2, d its horizontal distance from pivot_enu (the error
        at the walls)."""
        s = self.sigma
        d = float(np.linalg.norm(np.asarray(p_enu, dtype=float)[:2] - np.asarray(self.pivot_enu[:2])))
        xy = max(s["east_m"], s["north_m"])
        h = math.sqrt(xy * xy + d * d * (math.radians(s["yaw_deg"]) ** 2 + s["log_scale"] ** 2))
        v = math.sqrt(s["up_m"] ** 2 + d * d * math.radians(s["tilt_deg"]) ** 2)
        return h, v

    # ------------------------------------------------------------ calibration
    def recalibrated(self, p: params.Params | None = None) -> GeoAnchor:
        """The same anchor with the floors of p (default: the current set) applied to raw["sigma_core"]; itself when
        it was made with them or has no core (e.g. the fake reconstructor's)."""
        cal = calibration(p)
        core = self.raw.get("sigma_core")
        if cal == self.calibration or not core:
            return self
        return replace(self, sigma=floored_sigma(core, self.sigma["up_m"], cal["floors"]), calibration=cal)

    # ------------------------------------------------------------ storage
    def to_meta(self) -> dict[str, Any]:
        return {"schema": ANCHOR_SCHEMA, "frame": self.frame, "level": self.level,
                "transform": {"scale": self.scale, "rotation_xyzw": list(self.rotation_xyzw),
                              "translation": list(self.translation)},
                "parts": {"up_local": list(self.up_local), "yaw_deg": self.yaw_deg},
                "pivot_enu": list(self.pivot_enu), "sigma": dict(self.sigma), "source": dict(self.source),
                "raw": dict(self.raw), "calibration": dict(self.calibration)}

    @classmethod
    def from_meta(cls, d: dict[str, Any]) -> GeoAnchor:
        if d.get("schema") != ANCHOR_SCHEMA:
            raise ValueError(f"not a {ANCHOR_SCHEMA} record: schema {d.get('schema')!r}")
        t, parts = d["transform"], d["parts"]
        return cls(d["frame"], d["level"], float(t["scale"]), list(t["rotation_xyzw"]), list(t["translation"]),
                   list(parts["up_local"]), float(parts["yaw_deg"]), list(d["pivot_enu"]), dict(d["sigma"]),
                   dict(d["source"]), dict(d.get("raw") or {}), dict(d.get("calibration") or {}))


def read_anchor(meta: dict[str, Any]) -> GeoAnchor | None:
    """The GeoAnchor of a model JSON; None for a model recorded before anchors existed (only its legacy fields)."""
    a = (meta or {}).get("anchor")
    return GeoAnchor.from_meta(a) if a else None
