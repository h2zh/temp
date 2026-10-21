import io
import math

import pytest
from PIL import Image

from crowd3d import geo
from crowd3d.exif import UnreadableImage, build_exif_bytes, read_photo_meta


def test_normalize_and_diff():
    assert geo.normalize_deg(-10) == 350
    assert geo.normalize_deg(720) == 0
    assert geo.angle_diff_deg(10, 350) == 20
    assert geo.angle_diff_deg(350, 10) == -20
    assert geo.angle_diff_deg(180, 0) == 180


def test_bearing_cardinal_directions():
    lat, lon = 31.0, 121.0
    assert geo.bearing_deg(lat, lon, lat + 0.01, lon) == pytest.approx(0, abs=1e-6)
    assert geo.bearing_deg(lat, lon, lat, lon + 0.01) == pytest.approx(90, abs=0.01)
    assert geo.bearing_deg(lat, lon, lat - 0.01, lon) == pytest.approx(180, abs=1e-6)
    assert geo.bearing_deg(lat, lon, lat, lon - 0.01) == pytest.approx(270, abs=0.01)


def test_enu_roundtrip_and_distance():
    enu = geo.LocalENU(31.2397, 121.4998)
    lat, lon, h = enu.to_geodetic(30.0, -40.0, 1.6)
    e, n, u = enu.to_enu(lat, lon, h)
    assert (e, n, u) == pytest.approx((30.0, -40.0, 1.6), abs=1e-6)
    assert geo.haversine_m(31.2397, 121.4998, lat, lon) == pytest.approx(50.0, rel=2e-3)
    # Spherical bearing is only approximately the ellipsoidal ENU azimuth ...
    assert geo.bearing_deg(31.2397, 121.4998, lat, lon) == pytest.approx(
        geo.heading_from_en(30.0, -40.0), abs=0.3
    )
    # ... while the local helper agrees with ENU exactly.
    b, d = geo.local_bearing_distance(31.2397, 121.4998, lat, lon)
    assert b == pytest.approx(geo.heading_from_en(30.0, -40.0), abs=1e-6)
    assert d == pytest.approx(50.0, abs=1e-3)


def test_heading_from_en():
    assert geo.heading_from_en(0, 1) == 0
    assert geo.heading_from_en(1, 0) == 90
    assert geo.heading_from_en(-1, 0) == 270


def _jpeg(exif: bytes | None = None, size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, (120, 130, 140))
    img.save(buf, "JPEG", exif=exif) if exif else img.save(buf, "JPEG")
    return buf.getvalue()


def test_exif_roundtrip():
    blob = build_exif_bytes(
        lat=-33.8568, lon=151.2153, alt=12.5, heading_deg=271.25, focal_35mm=43,
        captured_at="2026:10:21 09:30:00",
    )
    meta = read_photo_meta(_jpeg(blob))
    assert (meta.width, meta.height) == (64, 48)
    assert meta.lat == pytest.approx(-33.8568, abs=1e-6)
    assert meta.lon == pytest.approx(151.2153, abs=1e-6)
    assert meta.alt == pytest.approx(12.5)
    assert meta.heading_deg == pytest.approx(271.25)
    assert meta.heading_ref == "T"
    assert meta.focal_35mm == 43
    assert meta.captured_at == "2026:10:21 09:30:00"
    assert meta.has_gps


def test_exif_missing_and_garbage():
    meta = read_photo_meta(_jpeg())
    assert not meta.has_gps and meta.heading_deg is None
    with pytest.raises(UnreadableImage):
        read_photo_meta(b"not an image")
    assert not math.isnan(meta.width)


def test_decompression_bomb_is_unreadable(monkeypatch):
    # Pillow raises DecompressionBombError (a plain Exception) for absurd pixel counts.
    data = _jpeg(size=(64, 64))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(UnreadableImage):
        read_photo_meta(data)
