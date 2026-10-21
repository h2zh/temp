"""Geodesy helpers: distances, bearings and a local East-North-Up (ENU) frame.

Conventions used across the project:
- Angles are degrees. Bearings and compass headings are clockwise from true
  north in [0, 360).
- ENU coordinates are metres: x = east, y = north, z = up.
"""

from __future__ import annotations

import math

EARTH_MEAN_RADIUS_M = 6371008.8
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def normalize_deg(angle: float) -> float:
    """Wrap an angle into [0, 360)."""
    a = math.fmod(angle, 360.0)
    if a < 0:
        a += 360.0
    # fmod can return 360.0 - tiny for tiny negatives; keep the half-open range.
    return 0.0 if a >= 360.0 else a


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference a - b, in (-180, 180]."""
    d = normalize_deg(a - b)
    return d - 360.0 if d > 180.0 else d


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_MEAN_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return normalize_deg(math.degrees(math.atan2(y, x)))


def heading_from_en(east: float, north: float) -> float:
    """Compass heading of a horizontal direction given by its east/north components."""
    return normalize_deg(math.degrees(math.atan2(east, north)))


def geodetic_to_ecef(lat: float, lon: float, h: float = 0.0) -> tuple[float, float, float]:
    la, lo = math.radians(lat), math.radians(lon)
    sin_la = math.sin(la)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_la * sin_la)
    x = (n + h) * math.cos(la) * math.cos(lo)
    y = (n + h) * math.cos(la) * math.sin(lo)
    z = (n * (1.0 - WGS84_E2) + h) * sin_la
    return x, y, z


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(8):
        sin_la = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_la * sin_la)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    return math.degrees(lat), math.degrees(lon), h


def local_bearing_distance(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """(bearing, horizontal distance in metres) from (lat0, lon0) to (lat, lon) in the local ENU frame.

    Use this, not bearing_deg/haversine_m, wherever the result must agree with
    ENU coordinates (sector assignment, reconstruction): the spherical formulas
    drift ~0.1 degrees from the ellipsoidal ENU azimuth.
    """
    e, n, _ = LocalENU(lat0, lon0).to_enu(lat, lon)
    return heading_from_en(e, n), math.hypot(e, n)


class LocalENU:
    """Tangent-plane frame anchored at (lat0, lon0, h0)."""

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0) -> None:
        self.lat0, self.lon0, self.h0 = lat0, lon0, h0
        self._origin = geodetic_to_ecef(lat0, lon0, h0)
        la, lo = math.radians(lat0), math.radians(lon0)
        sl, cl = math.sin(la), math.cos(la)
        so, co = math.sin(lo), math.cos(lo)
        # Rows are the east, north and up unit vectors expressed in ECEF.
        self._r = (
            (-so, co, 0.0),
            (-sl * co, -sl * so, cl),
            (cl * co, cl * so, sl),
        )

    def to_enu(self, lat: float, lon: float, h: float = 0.0) -> tuple[float, float, float]:
        x, y, z = geodetic_to_ecef(lat, lon, h)
        dx, dy, dz = x - self._origin[0], y - self._origin[1], z - self._origin[2]
        e, n, u = (r[0] * dx + r[1] * dy + r[2] * dz for r in self._r)
        return e, n, u

    def to_geodetic(self, e: float, n: float, u: float = 0.0) -> tuple[float, float, float]:
        r = self._r
        dx = r[0][0] * e + r[1][0] * n + r[2][0] * u
        dy = r[0][1] * e + r[1][1] * n + r[2][1] * u
        dz = r[0][2] * e + r[1][2] * n + r[2][2] * u
        return ecef_to_geodetic(self._origin[0] + dx, self._origin[1] + dy, self._origin[2] + dz)
