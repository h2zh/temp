"""Synthetic check-in photos of a procedurally textured building, with EXIF.

Used by tests and the demo in place of real crowd photos: we know the true
camera poses, so geo-registration and heading estimates can be checked.
"""

from __future__ import annotations

import argparse
import colorsys
import functools
import io
import json
import math
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .coverage import sector_for_bearing
from .exif import build_exif_bytes
from .geo import LocalENU, heading_from_en, local_bearing_distance, normalize_deg

DEMO_LAT = 31.2397
DEMO_LON = 121.4998
DEMO_NAME = "演示建筑（合成）"

FACADE_PX_PER_M = 40.0
GROUND_PX_PER_M = 20.0
GROUND_HALF_M = 64.0           # ground texture covers [-64, 64] m on both axes
CAMERA_HEIGHT_M = 1.6
CAMERA_RADIUS_M = (28.0, 40.0)
SECTOR_EDGE_MARGIN_DEG = 4.0
HEADING_JITTER_DEG = 5.0
PITCH_RANGE_DEG = (5.0, 12.0)
FOCAL_FACTOR = 1.2             # focal_px = FOCAL_FACTOR * max(w, h), COLMAP's default prior
FAR_DISTANCE_M = 500.0
BAD_SIZE = (320, 240)
MIN_DHASH_DISTANCE = 10        # validation calls <= 4 bits a duplicate; keep a wide margin

_SUN = np.array([0.354, -0.612, 0.707])   # azimuth 150 deg, elevation 45 deg
_AMBIENT = 0.72
_PHONES = [
    ("Apple", "iPhone 15"),
    ("Google", "Pixel 8"),
    ("samsung", "SM-S921B"),
    ("Xiaomi", "23127PN0CC"),
    ("HUAWEI", "ALN-AL00"),
]
_ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
)


@dataclass
class BuildingSpec:
    width_e_m: float = 20.0      # extent along east
    depth_n_m: float = 14.0      # extent along north
    height_m: float = 12.0
    seed: int = 7                # textures are deterministic per seed


# ---------------------------------------------------------------- textures


class _Texture:
    """Image draped over a plane, sampled trilinearly from a mip pyramid to avoid aliasing."""

    def __init__(self, img: np.ndarray, px_per_m: float) -> None:
        self.px_per_m = px_per_m
        self.levels = [img]
        while min(self.levels[-1].shape[:2]) >= 16:
            self.levels.append(cv2.pyrDown(self.levels[-1]))

    def _bilinear(self, level: int, u_m: np.ndarray, v_m: np.ndarray) -> np.ndarray:
        img = self.levels[level]
        h, w = img.shape[:2]
        s = self.px_per_m / (1 << level)
        x = u_m * s - 0.5
        y = v_m * s - 0.5
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0).astype(np.float32)[:, None]
        fy = (y - y0).astype(np.float32)[:, None]
        x0 = x0.astype(np.intp)
        y0 = y0.astype(np.intp)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
        bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
        return top * (1 - fy) + bottom * fy

    def sample(self, u_m: np.ndarray, v_m: np.ndarray, footprint_m: np.ndarray) -> np.ndarray:
        """RGB float32 (N, 3); footprint_m is the size of one image pixel on the surface."""
        lod = np.log2(np.maximum(footprint_m * self.px_per_m, 1.0))
        lod = np.minimum(lod, len(self.levels) - 1)
        lo = np.floor(lod).astype(np.intp)
        frac = (lod - lo).astype(np.float32)[:, None]
        out = np.empty((len(u_m), 3), np.float32)
        for level in range(len(self.levels)):
            m = lo == level
            if not m.any():
                continue
            c = self._bilinear(level, u_m[m], v_m[m])
            if level + 1 < len(self.levels):
                f = frac[m]
                c = c * (1 - f) + self._bilinear(level + 1, u_m[m], v_m[m]) * f
            out[m] = c
        return out


def _color(rng: np.random.Generator, sat: tuple[float, float], val: tuple[float, float]) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(*sat), rng.uniform(*val))
    return int(r * 255), int(g * 255), int(b * 255)


def _bright(rng: np.random.Generator) -> tuple[int, int, int]:
    return _color(rng, (0.7, 1.0), (0.75, 1.0))


def _dark(rng: np.random.Generator) -> tuple[int, int, int]:
    return _color(rng, (0.2, 0.9), (0.05, 0.3))


def _word(rng: np.random.Generator, lo: int, hi: int) -> str:
    return "".join(rng.choice(list(_ALNUM), size=int(rng.integers(lo, hi + 1))))


def _noise_field(h: int, w: int, rng: np.random.Generator, cell_px: list[int]) -> np.ndarray:
    """Smooth value noise in [-1, 1] summed over the given cell sizes."""
    acc = np.zeros((h, w), np.float32)
    for c in cell_px:
        small = rng.random((h // c + 2, w // c + 2), dtype=np.float32)
        acc += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    acc -= acc.mean()
    return acc / max(float(np.abs(acc).max()), 1e-6)


def _put_text(img: np.ndarray, text: str, org: tuple[int, int], height_px: float, color: tuple[int, int, int],
              rng: np.random.Generator, thickness: int | None = None) -> None:
    font = int(rng.choice(_FONTS))
    scale = height_px / 22.0
    th = thickness or max(1, int(round(height_px / 9)))
    cv2.putText(img, text, org, font, scale, color, th, cv2.LINE_AA)


def _poster(img: np.ndarray, x: int, y: int, w: int, h: int, rng: np.random.Generator) -> None:
    """Rectangle with unique text and shapes: a sign, poster or ad."""
    cv2.rectangle(img, (x, y), (x + w, y + h), _bright(rng) if rng.random() < 0.6 else _dark(rng), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), _dark(rng), max(2, w // 40))
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = int(x + rng.uniform(0.15, 0.85) * w), int(y + rng.uniform(0.15, 0.85) * h)
        r = int(rng.uniform(0.08, 0.3) * min(w, h))
        if rng.random() < 0.5:
            cv2.circle(img, (cx, cy), r, _color(rng, (0.5, 1), (0.2, 1)), -1, cv2.LINE_AA)
        else:
            pts = np.array([[cx + r * math.cos(a), cy + r * math.sin(a)]
                            for a in rng.uniform(0, 2 * math.pi, int(rng.integers(3, 6)))], np.int32)
            cv2.fillPoly(img, [pts], _color(rng, (0.5, 1), (0.2, 1)), cv2.LINE_AA)
    lines = int(rng.integers(1, 4))
    for i in range(lines):
        th = h / (lines + 1) * rng.uniform(0.5, 0.8)
        _put_text(img, _word(rng, 2, 7), (x + int(0.05 * w), y + int((i + 1) * h / (lines + 1) + th / 2)),
                  th, _dark(rng) if rng.random() < 0.5 else (255, 255, 255), rng)


def _graffiti(img: np.ndarray, x: int, y: int, size: int, rng: np.random.Generator) -> None:
    pts = np.cumsum(rng.normal(0, size / 4, (int(rng.integers(5, 10)), 2)), axis=0)
    pts = (pts - pts.mean(0) + [x, y]).astype(np.int32)
    th = int(rng.integers(5, 13))
    cv2.polylines(img, [pts], False, (15, 15, 15), th + 6, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, _bright(rng), th, cv2.LINE_AA)
    if rng.random() < 0.6:
        _put_text(img, _word(rng, 2, 4), (x - size // 2, y + size // 3), size * 0.6, _bright(rng), rng,
                  thickness=int(rng.integers(4, 9)))


def _masonry(h: int, w: int, rng: np.random.Generator, style: int, hue: float) -> np.ndarray:
    ppm = FACADE_PX_PER_M
    block_h, block_w = [((0.3, 0.45), (0.5, 1.2)),      # stone blocks
                        ((1.0, 1.6), (1.2, 2.4)),       # concrete panels
                        ((0.17, 0.24), (0.3, 0.6)),     # brick
                        ((0.35, 0.7), (0.4, 0.9))][style % 4]  # rubble
    base = np.array(colorsys.hsv_to_rgb(hue, rng.uniform(0.2, 0.5), rng.uniform(0.7, 0.9)), np.float64) * 255
    joint = tuple(int(c) for c in base * 0.4)
    img = np.zeros((h, w, 3), np.uint8)
    y = 0
    while y < h:
        bh = max(4, int(rng.uniform(*block_h) * ppm))
        x = -int(rng.uniform(0, block_w[1]) * ppm)
        while x < w:
            bw = max(4, int(rng.uniform(*block_w) * ppm))
            tone = np.clip(base * rng.uniform(0.6, 1.25) + rng.normal(0, 14, 3), 0, 255)
            cv2.rectangle(img, (x, y), (x + bw - 1, y + bh - 1), tone.tolist(), -1)
            cv2.rectangle(img, (x, y), (x + bw - 1, y + bh - 1), joint, 1 if style == 2 else 2)
            x += bw
        y += bh
    return img


def _window(img: np.ndarray, x: int, y: int, w: int, h: int, rng: np.random.Generator) -> None:
    frame = _color(rng, (0, 0.6), (0.1, 1.0))
    ft = int(rng.integers(3, 8))
    cv2.rectangle(img, (x - ft, y - ft), (x + w + ft, y + h + ft), frame, -1)
    top = np.array(_color(rng, (0.2, 0.7), (0.15, 0.6)), np.float32)
    bottom = np.array(_color(rng, (0.1, 0.6), (0.05, 0.4)), np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img[y:y + h, x:x + w] = (top * (1 - t) + bottom * t).astype(np.uint8)
    # A reflection streak at a random angle.
    sx = int(rng.uniform(0, w))
    streak = np.array([[sx, y], [sx + w // 4, y], [sx - w // 3, y + h], [sx - w // 3 - w // 4, y + h]], np.int32)
    streak[:, 0] = np.clip(streak[:, 0] + x, x, x + w)
    cv2.fillPoly(img, [streak], tuple(int(c) for c in np.minimum(top * 1.6 + 30, 255)))
    if rng.random() < 0.55:  # curtain or blind
        ch = int(h * rng.uniform(0.15, 0.7))
        cv2.rectangle(img, (x, y), (x + w, y + ch), _color(rng, (0.1, 0.9), (0.4, 1.0)), -1)
        if rng.random() < 0.5:
            for yy in range(y, y + ch, max(3, ch // int(rng.integers(3, 8)))):
                cv2.line(img, (x, yy), (x + w, yy), _dark(rng), 1)
    for _ in range(int(rng.integers(0, 3))):  # plants, lamps, stickers
        cv2.circle(img, (int(x + rng.uniform(0.1, 0.9) * w), int(y + rng.uniform(0.4, 0.95) * h)),
                   int(rng.uniform(0.06, 0.2) * w), _color(rng, (0.4, 1), (0.3, 1)), -1, cv2.LINE_AA)
    mullion = int(rng.integers(0, 3))
    mt = max(2, ft - 2)
    if mullion >= 1:
        mx = int(x + w * rng.uniform(0.4, 0.6))
        cv2.line(img, (mx, y), (mx, y + h), frame, mt)
    if mullion == 2:
        my = int(y + h * rng.uniform(0.25, 0.45))
        cv2.line(img, (x, my), (x + w, my), frame, mt)
    sill_h = int(rng.integers(4, 10))
    cv2.rectangle(img, (x - 2 * ft, y + h + ft), (x + w + 2 * ft, y + h + ft + sill_h),
                  _color(rng, (0, 0.3), (0.6, 1.0)), -1)


def _ac_unit(img: np.ndarray, x: int, y: int, rng: np.random.Generator) -> None:
    w, h = int(rng.integers(30, 44)), int(rng.integers(22, 30))
    cv2.rectangle(img, (x, y), (x + w, y + h), _color(rng, (0, 0.15), (0.7, 0.95)), -1)
    cv2.circle(img, (x + w // 3, y + h // 2), h // 3, (60, 60, 60), 2, cv2.LINE_AA)
    for xx in range(x + 2 * w // 3, x + w - 2, 4):
        cv2.line(img, (xx, y + 3), (xx, y + h - 3), (90, 90, 90), 1)


def _facade_texture(width_m: float, height_m: float, rng: np.random.Generator, style: int, hue: float) -> np.ndarray:
    """A unique facade: masonry, varied windows, a shop front, posters, graffiti and murals."""
    ppm = FACADE_PX_PER_M
    w, h = int(round(width_m * ppm)), int(round(height_m * ppm))
    img = _masonry(h, w, rng, style, hue)
    stains = _noise_field(h, w, rng, [7, 23, 61, 157])
    img = np.clip(img * (0.88 + 0.22 * stains)[..., None], 0, 255).astype(np.uint8)
    # Small one-off marks (patched render, plaques, vents, stickers): dense detail that is unique per spot.
    for _ in range(int(width_m * height_m * 2.5)):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        s = int(rng.uniform(0.1, 0.35) * ppm)
        col = _color(rng, (0, 0.7), (0.05, 1.0))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.4, 1.5))), col, -1)
        else:
            cv2.circle(img, (x, y), max(2, s // 2), col, -1, cv2.LINE_AA)

    floor_h = 3.2 * ppm
    floors = max(1, int(height_m * ppm // floor_h))
    cols = max(1, int(round(width_m / 2.8)))
    col_w = w / cols
    for f in range(1, floors):
        y_floor = h - (f + 1) * floor_h
        cv2.line(img, (0, int(y_floor + floor_h)), (w, int(y_floor + floor_h)), _dark(rng), int(rng.integers(3, 9)))
        for c in range(cols):
            ww = int(rng.uniform(0.9, 1.5) * ppm)
            wh = int(rng.uniform(1.2, 1.8) * ppm)
            cx = (c + 0.5) * col_w + rng.uniform(-0.3, 0.3) * ppm
            x = int(cx - ww / 2)
            y = int(y_floor + (floor_h - wh) * rng.uniform(0.3, 0.6))
            r = rng.random()
            if r < 0.12:
                _poster(img, x, y, ww, wh, rng)
            elif r < 0.17:
                continue  # blank wall where a window could be
            else:
                _window(img, x, y, ww, wh, rng)
            if rng.random() < 0.2:
                _ac_unit(img, int(cx + rng.uniform(-0.8, 0.3) * ppm), int(y + wh + 12), rng)

    # Ground floor: shop fronts with a sign band, and doors.
    band_y = int(h - floor_h + rng.uniform(0.1, 0.3) * ppm)
    band_h = int(rng.uniform(0.6, 0.9) * ppm)
    x = int(rng.uniform(0.2, 1.0) * ppm)
    while x < w - ppm:
        shop_w = int(rng.uniform(3.0, 7.0) * ppm)
        shop_w = min(shop_w, w - x - int(0.3 * ppm))
        if shop_w < 1.2 * ppm:
            break
        _poster(img, x, band_y, shop_w, band_h, rng)
        gy = band_y + band_h + int(0.2 * ppm)
        door_w = int(rng.uniform(1.0, 1.6) * ppm)
        door_x = x + int(rng.uniform(0, max(1, shop_w - door_w)))
        _window(img, x + 6, gy, shop_w - 12, h - gy - int(0.25 * ppm), rng)
        cv2.rectangle(img, (door_x, gy), (door_x + door_w, h - 1), _dark(rng), -1)
        cv2.rectangle(img, (door_x, gy), (door_x + door_w, h - 1), _bright(rng), 4)
        cv2.circle(img, (door_x + door_w - 10, (gy + h) // 2), 4, (220, 200, 90), -1)
        x += shop_w + int(rng.uniform(0.4, 1.5) * ppm)

    # Large one-off elements: murals, posters, graffiti.
    for _ in range(int(rng.integers(3, 6))):
        ax, ay = int(rng.uniform(0.6, 2.2) * ppm), int(rng.uniform(0.3, 1.0) * ppm)
        cv2.ellipse(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h - floor_h))), (ax, ay),
                    rng.uniform(0, 180), 0, 360, _bright(rng), -1, cv2.LINE_AA)
    for _ in range(int(rng.integers(3, 6))):
        pw, ph = int(rng.uniform(0.8, 2.0) * ppm), int(rng.uniform(0.6, 1.5) * ppm)
        _poster(img, int(rng.uniform(0, w - pw)), int(rng.uniform(0, h - ph)), pw, ph, rng)
    for _ in range(int(rng.integers(3, 7))):
        gx, gy = int(rng.uniform(0, w)), int(h - rng.uniform(0.4, 2.6) * ppm)
        _graffiti(img, gx, gy, int(rng.uniform(0.8, 1.8) * ppm), rng)
    _put_text(img, _word(rng, 3, 6), (int(rng.uniform(0.05, 0.4) * w), int(rng.uniform(0.15, 0.3) * h)),
              rng.uniform(1.0, 1.6) * ppm, _bright(rng), rng, thickness=int(rng.integers(6, 12)))

    # Dirt toward the ground and fine grain.
    grime = np.clip(1.0 - 0.25 * np.linspace(-1, 1, h) ** 8, 0.7, 1.0).astype(np.float32)
    grain = rng.normal(0, 5, (h, w, 1)).astype(np.float32)
    return np.clip(img * grime[:, None, None] + grain, 0, 255).astype(np.uint8)


def _ground_texture(rng: np.random.Generator) -> np.ndarray:
    """Paving with per-slab tones, a plaza ring, painted markings, manholes and stains (north up)."""
    ppm = GROUND_PX_PER_M
    n = int(2 * GROUND_HALF_M * ppm)
    slab = int(0.9 * ppm)
    k = rng.uniform(0.6, 1.3, (n // slab + 1, n // slab + 1)).astype(np.float32)
    k = cv2.resize(k, (k.shape[1] * slab, k.shape[0] * slab), interpolation=cv2.INTER_NEAREST)[:n, :n]
    joint = np.arange(n) % slab < 2
    k[joint, :] *= 0.55
    k[:, joint] *= 0.55
    k *= 0.85 + 0.25 * _noise_field(n, n, rng, [5, 19, 67, 211])
    img = cv2.merge([k * c for c in (150.0, 146.0, 138.0)])
    # Warmer paving on a plaza ring around the building.
    c0, rp = n // 2, int(16 * ppm)
    yy, xx = np.ogrid[-rp:rp, -rp:rp]
    plaza = img[c0 - rp:c0 + rp, c0 - rp:c0 + rp]
    plaza[xx * xx + yy * yy <= rp * rp] *= np.float32([180 / 150, 156 / 146, 132 / 138])
    img = np.clip(img, 0, 255, out=img).astype(np.uint8)

    def m(lo: float, hi: float) -> float:
        return rng.uniform(lo, hi) * ppm

    paint = [(240, 240, 235), (240, 200, 40), (220, 60, 50), (40, 120, 220)]
    for _ in range(2200):
        kind = rng.random()
        c = int(rng.uniform(0, n)), int(rng.uniform(0, n))
        col = paint[int(rng.integers(0, len(paint)))] if rng.random() < 0.7 else _bright(rng)
        if kind < 0.2:  # stains
            cv2.ellipse(img, c, (int(m(0.2, 1.5)), int(m(0.15, 0.9))), rng.uniform(0, 180), 0, 360,
                        _color(rng, (0, 0.4), (0.1, 0.35)), -1, cv2.LINE_AA)
        elif kind < 0.3:  # manholes and drains
            r = int(m(0.35, 0.7))
            cv2.circle(img, c, r, (70, 70, 75), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (30, 30, 30), 3, cv2.LINE_AA)
            for dy in range(-r + 4, r - 3, 5):
                cv2.line(img, (c[0] - r + 4, c[1] + dy), (c[0] + r - 4, c[1] + dy), (35, 35, 35), 2)
        elif kind < 0.5:  # painted lines and arrows
            a = rng.uniform(0, 2 * math.pi)
            ln = m(1.0, 4.0)
            end = (int(c[0] + ln * math.cos(a)), int(c[1] + ln * math.sin(a)))
            th = int(m(0.12, 0.3))
            if rng.random() < 0.5:
                cv2.arrowedLine(img, c, end, col, th, cv2.LINE_AA, tipLength=0.3)
            else:
                cv2.line(img, c, end, col, th, cv2.LINE_AA)
        elif kind < 0.75:  # painted text and numbers
            _put_text(img, _word(rng, 1, 4), c, m(0.6, 1.6), col, rng, thickness=int(m(0.1, 0.25)))
        elif kind < 0.85:  # painted shapes
            r = int(m(0.3, 1.2))
            if rng.random() < 0.5:
                cv2.circle(img, c, r, col, int(m(0.08, 0.25)), cv2.LINE_AA)
            else:
                cv2.rectangle(img, (c[0] - r, c[1] - r // 2), (c[0] + r, c[1] + r // 2), col, -1)
        else:  # gravel / leaf patches
            pts = np.asarray(c) + rng.normal(0, m(0.2, 0.6), (40, 2))
            for p in pts.astype(int):
                cv2.circle(img, (int(p[0]), int(p[1])), int(rng.integers(1, 4)),
                           _color(rng, (0.3, 0.9), (0.15, 0.8)), -1)
    return img


@dataclass
class _Scene:
    lo: np.ndarray
    hi: np.ndarray
    walls: list[tuple[_Texture, np.ndarray, float]]   # (texture, outward normal, shade) for N, E, S, W
    ground: _Texture
    roof_rgb: np.ndarray


@functools.lru_cache(maxsize=4)
def _scene(width_m: float, depth_m: float, height_m: float, seed: int) -> _Scene:
    rng = np.random.default_rng(seed)
    normals = [np.array(v, float) for v in ((0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0))]
    hue0 = rng.uniform(0, 1)
    walls = []
    for i, nrm in enumerate(normals):
        span = width_m if i % 2 == 0 else depth_m
        # Hues a quarter turn apart and different masonry keep facades from looking alike.
        hue = (hue0 + 0.25 * i + rng.uniform(-0.05, 0.05)) % 1
        tex = _Texture(_facade_texture(span, height_m, rng, style=(i + seed) % 4, hue=hue), FACADE_PX_PER_M)
        walls.append((tex, nrm, _AMBIENT + (1 - _AMBIENT) * max(0.0, float(nrm @ _SUN))))
    ground = _Texture(_ground_texture(rng), GROUND_PX_PER_M)
    return _Scene(
        lo=np.array([-width_m / 2, -depth_m / 2, 0.0]),
        hi=np.array([width_m / 2, depth_m / 2, height_m]),
        walls=walls,
        ground=ground,
        roof_rgb=np.float32([120, 115, 110]),
    )


# ---------------------------------------------------------------- rendering


def camera_axes(heading_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, right, up) unit vectors in ENU for a camera with zero roll."""
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    fwd = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    return fwd, right, np.cross(right, fwd)


def _ray_box(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slab test. Returns (t, face) with face = -1 for a miss, else 0..3 for walls N, E, S, W and 4 for the roof."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    t_in = np.minimum(t1, t2)
    t_out = np.maximum(t1, t2).min(axis=1)
    axis = t_in.argmax(axis=1)
    t = t_in.max(axis=1)
    hit = (t <= t_out) & (t > 1e-6)
    neg = d[np.arange(len(d)), axis] < 0
    # axis 0: -dx hits the east wall; axis 1: -dy hits the north wall; axis 2: -dz hits the roof.
    face = np.select([axis == 0, axis == 1], [np.where(neg, 1, 3), np.where(neg, 0, 2)], np.where(neg, 4, -1))
    return t, np.where(hit, face, -1)


def render_view(
    building: BuildingSpec,
    cam_enu: tuple[float, float, float],
    heading_deg: float,
    pitch_deg: float,
    size: tuple[int, int],
    focal_px: float,
) -> np.ndarray:
    """Ray-cast an HxWx3 uint8 RGB image of the building (centred at ENU origin), ground and sky."""
    w, h = size
    scene = _scene(float(building.width_e_m), float(building.depth_n_m), float(building.height_m), int(building.seed))
    o = np.asarray(cam_enu, dtype=np.float64)
    fwd, right, up = camera_axes(heading_deg, pitch_deg)
    xs = (np.arange(w) + 0.5 - w / 2) / focal_px
    ys = (np.arange(h) + 0.5 - h / 2) / focal_px
    d = fwd + xs[None, :, None] * right - ys[:, None, None] * up
    d = d.reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    # Featureless sky gradient.
    elev = np.clip(d[:, 2], 0, 1).astype(np.float32)[:, None]
    rgb = np.float32([205, 218, 232]) * (1 - elev) + np.float32([95, 140, 205]) * elev

    t, face = _ray_box(o, d, scene.lo, scene.hi)
    for k, (tex, nrm, shade) in enumerate(scene.walls):
        m = face == k
        if not m.any():
            continue
        p = o + t[m, None] * d[m]
        u = (scene.hi[0] - p[:, 0], p[:, 1] - scene.lo[1], p[:, 0] - scene.lo[0], scene.hi[1] - p[:, 1])[k]
        cos = np.maximum(np.abs(d[m] @ nrm), 0.15)
        rgb[m] = tex.sample(u, scene.hi[2] - p[:, 2], t[m] / focal_px / cos) * shade
    rgb[face == 4] = scene.roof_rgb

    g = (face < 0) & (d[:, 2] < -1e-9)
    if g.any():
        tg = -o[2] / d[g, 2]
        p = o + tg[:, None] * d[g]
        cos = np.maximum(-d[g, 2], 1e-3)
        col = scene.ground.sample(p[:, 0] + GROUND_HALF_M, GROUND_HALF_M - p[:, 1], tg / focal_px / np.sqrt(cos))
        _, shadow_face = _ray_box(p, np.broadcast_to(_SUN, p.shape), scene.lo, scene.hi)
        lit = np.where(shadow_face >= 0, 0.62, 0.92).astype(np.float32)
        rgb[g] = col * lit[:, None]

    img = cv2.GaussianBlur(rgb.reshape(h, w, 3), (0, 0), 0.6)
    seed = zlib.crc32(np.array([*cam_enu, heading_deg, pitch_deg, w, h, focal_px], np.float64).tobytes())
    img += np.random.default_rng(seed).normal(0, 1.5, img.shape).astype(np.float32)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- dataset


def focal_px_for(size: tuple[int, int]) -> float:
    return FOCAL_FACTOR * max(size)


def _pitch_for(building: BuildingSpec, cam: np.ndarray, size: tuple[int, int], rng: np.random.Generator) -> float:
    """Pitch up so the top of the building sits in the upper part of the frame, clipped to PITCH_RANGE_DEG."""
    half_vfov = math.degrees(math.atan(size[1] / 2 / focal_px_for(size)))
    corners = [(sx * building.width_e_m / 2, sy * building.depth_n_m / 2) for sx in (-1, 1) for sy in (-1, 1)]
    roof = max(math.degrees(math.atan2(building.height_m - cam[2], math.hypot(cam[0] - x, cam[1] - y)))
               for x, y in corners)
    return float(np.clip(roof - rng.uniform(0.45, 0.75) * half_vfov, *PITCH_RANGE_DEG))


def _sample_pose(building: BuildingSpec, bearing: float, size: tuple[int, int],
                 rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    r = rng.uniform(*CAMERA_RADIUS_M)
    b = math.radians(bearing)
    cam = np.array([r * math.sin(b), r * math.cos(b), CAMERA_HEIGHT_M])
    heading = normalize_deg(heading_from_en(-cam[0], -cam[1]) + rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG))
    return cam, heading, _pitch_for(building, cam, size, rng)


def _noisy_gps(enu: LocalENU, cam: np.ndarray, sector: int, num_sectors: int, noise_m: float,
               rng: np.random.Generator) -> tuple[float, float]:
    """GPS fix with gaussian noise, redrawn until it still falls in the camera's own sector."""
    for _ in range(100):
        de, dn = np.clip(rng.normal(0, noise_m, 2), -3 * noise_m, 3 * noise_m) if noise_m > 0 else (0.0, 0.0)
        lat, lon, _ = enu.to_geodetic(cam[0] + de, cam[1] + dn, 0.0)
        bearing, _ = local_bearing_distance(enu.lat0, enu.lon0, lat, lon)
        if sector_for_bearing(bearing, num_sectors) == sector:
            return lat, lon
    lat, lon, _ = enu.to_geodetic(cam[0], cam[1], 0.0)
    return lat, lon


def _dhash_bits(img: np.ndarray) -> np.ndarray:
    """64-bit dHash (imagehash convention) under two common resampling filters, shape (2, 64)."""
    gray = Image.fromarray(img).convert("L")
    rows = []
    for flt in (Image.Resampling.LANCZOS, Image.Resampling.BILINEAR):
        px = np.asarray(gray.resize((9, 8), flt), dtype=np.int16)
        rows.append((px[:, 1:] > px[:, :-1]).ravel())
    return np.stack(rows)


def _jpeg(img: np.ndarray, exif: bytes, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality, exif=exif)
    return buf.getvalue()


def generate_dataset(
    out_dir: Path,
    *,
    landmark_lat: float = DEMO_LAT,
    landmark_lon: float = DEMO_LON,
    building: BuildingSpec | None = None,
    num_sectors: int = 8,
    photos_per_sector: int = 4,
    sectors: list[int] | None = None,
    num_users: int = 5,
    size: tuple[int, int] = (800, 600),
    gps_noise_m: float = 2.0,
    heading_noise_deg: float = 8.0,
    include_bad: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Write JPEGs plus manifest.json into out_dir; return the manifest.

    manifest = {"landmark": {"lat", "lon", "name"}, "building": {...},
                "photos": [{"file", "user_id", "sector", "true_enu": [x,y,z],
                            "true_heading_deg", "exif_lat", "exif_lon",
                            "exif_heading_deg", "expected_valid": bool,
                            "expected_reason": str | None}]}

    Good photos come first, sector by sector; bad ones (include_bad) follow, the
    duplicate after its original. Entries also carry "kind" (good, blurry, far,
    facing_away, duplicate, no_gps, too_small), "true_pitch_deg", and for the
    duplicate "duplicate_of". "sector" is the sector of the EXIF position (equal
    to the true one: noisy GPS fixes are redrawn until they agree), None without GPS.
    Every photo except the duplicate is at least MIN_DHASH_DISTANCE bits from all
    earlier ones, so no good photo can be mistaken for a duplicate.
    """
    building = building or BuildingSpec()
    sectors = list(range(num_sectors)) if sectors is None else list(sectors)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    enu = LocalENU(landmark_lat, landmark_lon)
    width = 360.0 / num_sectors
    usable = width - 2 * SECTOR_EDGE_MARGIN_DEG
    start_time = datetime(2026, 10, 1, 9, 0, 0)
    photos: list[dict[str, Any]] = []
    hashes: list[np.ndarray] = []

    def shoot(sector: int, slot: int = 0, slots: int = 1, shot_size: tuple[int, int] = size,
              blur_sigma: float = 0.0) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Render a view from inside the sector, re-posing until it does not look like an earlier photo."""
        lo = sector * width - width / 2 + SECTOR_EDGE_MARGIN_DEG
        for _ in range(20):
            bearing = lo + (slot + rng.uniform(0.2, 0.8)) * usable / slots
            cam, heading, pitch = _sample_pose(building, bearing, shot_size, rng)
            img = render_view(building, tuple(cam), heading, pitch, shot_size, focal_px_for(shot_size))
            if blur_sigma:
                img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
            bits = _dhash_bits(img)
            if all((bits != h).sum(axis=1).min() >= MIN_DHASH_DISTANCE for h in hashes):
                break
        hashes.append(bits)
        return cam, heading, pitch, img

    def noisy_heading(heading: float) -> float:
        noise = float(np.clip(rng.normal(0, heading_noise_deg), -3 * heading_noise_deg, 3 * heading_noise_deg))
        return normalize_deg(heading + noise)

    def add(file: str, kind: str, sector: int | None, shot: tuple[np.ndarray, float, float, np.ndarray], *,
            gps: tuple[float, float] | None, exif_heading: float | None, expected_reason: str | None) -> None:
        cam, heading, pitch, img = shot
        user = len(photos) % num_users + 1
        make, model = _PHONES[(user - 1) % len(_PHONES)]
        exif = build_exif_bytes(
            lat=gps[0] if gps else None,
            lon=gps[1] if gps else None,
            heading_deg=exif_heading,
            focal_35mm=round(FOCAL_FACTOR * 36),
            captured_at=(start_time + timedelta(minutes=7 * len(photos))).strftime("%Y:%m:%d %H:%M:%S"),
            make=make,
            model=model,
        )
        (out_dir / file).write_bytes(_jpeg(img, exif))
        photos.append({
            "file": file,
            "user_id": f"user{user}",
            "sector": sector,
            "true_enu": [round(float(c), 4) for c in cam],
            "true_heading_deg": round(heading, 4),
            "true_pitch_deg": round(pitch, 4),
            "exif_lat": gps[0] if gps else None,
            "exif_lon": gps[1] if gps else None,
            "exif_heading_deg": None if exif_heading is None else round(exif_heading, 2),
            "expected_valid": expected_reason is None,
            "expected_reason": expected_reason,
            "kind": kind,
        })

    def gps_in(sector: int, cam: np.ndarray) -> tuple[float, float]:
        return _noisy_gps(enu, cam, sector, num_sectors, gps_noise_m, rng)

    for s in sectors:
        for k in range(photos_per_sector):
            shot = shoot(s, k, photos_per_sector)
            add(f"s{s}_{k:02d}.jpg", "good", s, shot, gps=gps_in(s, shot[0]),
                exif_heading=noisy_heading(shot[1]), expected_reason=None)

    if include_bad and sectors:
        bad = [sectors[i % len(sectors)] for i in range(6)]

        # Blurry: correct GPS and heading, heavy defocus and no sensor noise on top.
        shot = shoot(bad[0], blur_sigma=6.0)
        add("bad_blurry.jpg", "blurry", bad[0], shot, gps=gps_in(bad[0], shot[0]),
            exif_heading=noisy_heading(shot[1]), expected_reason="too_blurry")

        # Far away: GPS fix 500 m out along the same bearing, compass still toward the landmark.
        shot = shoot(bad[1])
        b = math.atan2(shot[0][0], shot[0][1])
        lat, lon, _ = enu.to_geodetic(FAR_DISTANCE_M * math.sin(b), FAR_DISTANCE_M * math.cos(b), 0.0)
        add("bad_far.jpg", "far", bad[1], shot, gps=(lat, lon),
            exif_heading=noisy_heading(shot[1]), expected_reason="too_far")

        # Facing away: the compass says the camera points away from the landmark.
        shot = shoot(bad[2])
        add("bad_facing_away.jpg", "facing_away", bad[2], shot, gps=gps_in(bad[2], shot[0]),
            exif_heading=normalize_deg(shot[1] + 180.0), expected_reason="facing_away")

        # Duplicate: another user re-uploads a re-encoded copy of a good photo, EXIF and all.
        original = next((p for p in photos if p["kind"] == "good" and p["sector"] == bad[3]), None)
        if original is not None:
            with Image.open(out_dir / original["file"]) as im:
                data = _jpeg(np.asarray(im.convert("RGB")), im.info["exif"], quality=80)
            (out_dir / "bad_duplicate.jpg").write_bytes(data)
            photos.append({**original, "file": "bad_duplicate.jpg", "user_id": f"user{len(photos) % num_users + 1}",
                           "kind": "duplicate", "expected_valid": False, "expected_reason": "duplicate",
                           "duplicate_of": original["file"]})

        # No GPS: a good image and compass heading, but no position.
        shot = shoot(bad[4])
        add("bad_no_gps.jpg", "no_gps", None, shot, gps=None,
            exif_heading=noisy_heading(shot[1]), expected_reason="no_gps")

        # Too small: a sharp, correctly tagged 320x240 photo.
        shot = shoot(bad[5], shot_size=BAD_SIZE)
        add("bad_too_small.jpg", "too_small", bad[5], shot, gps=gps_in(bad[5], shot[0]),
            exif_heading=noisy_heading(shot[1]), expected_reason="too_small")

    manifest = {
        "landmark": {"lat": landmark_lat, "lon": landmark_lon, "name": DEMO_NAME},
        "building": asdict(building),
        "num_sectors": num_sectors,
        "size": list(size),
        "focal_px": focal_px_for(size),
        "photos": photos,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _parse_size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m crowd3d.synthetic --out DIR [--per-sector N] [--sectors 0,1] [--size 800x600] ..."""
    ap = argparse.ArgumentParser(prog="python -m crowd3d.synthetic", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--per-sector", type=int, default=4, help="good photos per sector")
    ap.add_argument("--sectors", default=None, help="comma-separated sector indices (default: all)")
    ap.add_argument("--num-sectors", type=int, default=8)
    ap.add_argument("--size", type=_parse_size, default=(800, 600), help="WxH, e.g. 640x480")
    ap.add_argument("--users", type=int, default=5)
    ap.add_argument("--lat", type=float, default=DEMO_LAT)
    ap.add_argument("--lon", type=float, default=DEMO_LON)
    ap.add_argument("--gps-noise", type=float, default=2.0, help="metres (1 sigma)")
    ap.add_argument("--heading-noise", type=float, default=8.0, help="degrees (1 sigma)")
    ap.add_argument("--no-bad", action="store_true", help="skip the deliberately invalid photos")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    sectors = [int(s) for s in args.sectors.split(",") if s.strip()] if args.sectors else None
    manifest = generate_dataset(
        args.out,
        landmark_lat=args.lat,
        landmark_lon=args.lon,
        num_sectors=args.num_sectors,
        photos_per_sector=args.per_sector,
        sectors=sectors,
        num_users=args.users,
        size=args.size,
        gps_noise_m=args.gps_noise,
        heading_noise_deg=args.heading_noise,
        include_bad=not args.no_bad,
        seed=args.seed,
    )
    good = sum(p["expected_valid"] for p in manifest["photos"])
    print(f"wrote {len(manifest['photos'])} photos ({good} good) and manifest.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
