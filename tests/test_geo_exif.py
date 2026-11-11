import io
import math
import time

import piexif
import pytest
from PIL import Image

import hostile_images as hostile   # (pytest puts tests/ on the path)
from crowd3d import exif as exif_module
from crowd3d import geo
from crowd3d.exif import (MAX_TEXT_CHARS, UnreadableImage, build_exif_bytes, capture_time, jpeg_decodes_reduced,
                          read_photo_meta)


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
    assert not meta.has_exif
    assert read_photo_meta(_jpeg(build_exif_bytes(make="Apple", model="iPhone 15"))).has_exif
    with pytest.raises(UnreadableImage):
        read_photo_meta(b"not an image")
    assert not math.isnan(meta.width)


def test_decompression_bomb_is_unreadable(monkeypatch):
    # Pillow raises DecompressionBombError (a plain Exception) for absurd pixel counts.
    data = _jpeg(size=(64, 64))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(UnreadableImage):
        read_photo_meta(data)


# Crafted EXIF text and JPEGs that do not decode reduced (D34)


def test_an_exif_text_tag_is_text_of_bounded_length_or_nothing():
    """exif._text: bytes or str, stripped, cut at MAX_TEXT_CHARS; anything else (piexif returns a text tag written as
    SHORT or RATIONAL as an int or a tuple) is no tag, not an AttributeError."""
    text = exif_module._text
    assert text(b"Canon\x00") == text("Canon ") == text(b"  Canon\x00\x00 ") == "Canon"
    for nothing in (None, b"", b"\x00\x00", "   ", 24930, 0, 1.5, (24930, 0), ((1, 2),), [b"x"], {"a": 1}, True):
        assert text(nothing) is None, nothing
    assert text(b"\xff\xfeab") == "ab"                               # undecodable bytes are dropped
    assert MAX_TEXT_CHARS == 64
    for long in (b"M" * 65, b"M" * 1_000_000, "M" * 1_000_000, "\u5854".encode() * 400_000):
        got = text(long)
        assert len(got) == 64 and got == (long.decode() if isinstance(long, bytes) else long)[:64]
    assert text(b"M" * 64) == "M" * 64 and text(b" " * 10 + b"M" * 70) == "M" * 64


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_a_text_tag_written_as_a_number_is_a_missing_tag_not_an_exception(fmt):
    """A text tag whose IFD entry declares SHORT (piexif then gives an int, or a tuple for two of them) made
    read_photo_meta raise AttributeError, and the upload a server_error (the review's finding). Now that tag alone is
    missing; the rest of the EXIF is read as before. The module's promise: reading never raises on malformed EXIF."""
    good = hostile.exif_with(capture_time=b"2024:05:06 07:08:09", lat=31.25, lon=121.5, make=b"ab")
    ifds = {"0th": piexif.ImageIFD.Make, "Exif": piexif.ExifIFD.DateTimeOriginal, "GPS": piexif.GPSIFD.GPSLatitudeRef}
    for ifd, tag in ifds.items():
        for count in (1, 2):
            bad = hostile.retyped(good, tag, count)
            assert isinstance(piexif.load(bad)[ifd][tag], int if count == 1 else tuple)      # what piexif hands over
            meta = read_photo_meta(hostile.with_exif(fmt, bad))
            assert meta.has_exif and meta.format == fmt and meta.lat == pytest.approx(31.25, abs=1e-6)
            assert meta.make == (None if ifd == "0th" else "ab")
            assert meta.captured_at == (None if ifd == "Exif" else "2024:05:06 07:08:09")
    ref = hostile.exif_with(lat=31.25, lon=121.5)
    loaded = piexif.load(ref)
    loaded["GPS"][piexif.GPSIFD.GPSImgDirectionRef] = b"T"
    loaded["GPS"][piexif.GPSIFD.GPSImgDirection] = (9000, 100)
    bad = hostile.retyped(piexif.dump(loaded), piexif.GPSIFD.GPSImgDirectionRef)
    meta = read_photo_meta(hostile.with_exif(fmt, bad))
    assert (meta.heading_deg, meta.heading_ref) == (90.0, None)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_a_megabyte_of_capture_time_is_read_as_64_characters(fmt):
    """Every format carries EXIF of any size (a JPEG in several segments, which Pillow joins): a 1,000,000-character
    DateTimeOriginal, make or model is read as its first MAX_TEXT_CHARS characters."""
    long = b"2024:05:06 07:08:09" + b"9" * (1_000_000 - 19)
    data = hostile.with_exif(fmt, hostile.exif_with(capture_time=long, lat=31.25, lon=121.5, make=b"M" * 1_000_000))
    with Image.open(io.BytesIO(data)) as im:
        assert len(piexif.load(im.info["exif"])["Exif"][piexif.ExifIFD.DateTimeOriginal]) == 1_000_000
    meta = read_photo_meta(data)
    assert meta.captured_at == long[:64].decode() and meta.make == "M" * 64 and meta.has_gps
    assert capture_time(meta.captured_at) is None


def test_capture_time_keeps_only_the_exif_form():
    """exif.capture_time: exactly "YYYY:MM:DD HH:MM:SS", 19 ASCII characters; anything else is None."""
    assert capture_time("2026:10:21 09:30:00") == "2026:10:21 09:30:00"
    assert capture_time("0000:00:00 00:00:00") == "0000:00:00 00:00:00"        # an unset camera clock: the form holds
    fullwidth = "\uff12\uff10\uff12\uff16:10:21 09:30:00"            # digits to str.isdigit and to a regex's \\d
    assert len(fullwidth) == 19 and fullwidth[:4].isdigit()
    for bad in (None, "", "2026:10:21 09:30:00 ", "2026:10:21 09:30:000", "2026:10:21 09:30:0", "2026-10-21 09:30:00",
                "2026:10:21T09:30:00", "2026:10:21 09:30:00+02:00", "2026:10:21 09:30:00\n", "    :  :     :  :  ",
                fullwidth, "\u0662\u0660\u0662\u0666:10:21 09:30:00", "x" * 1_000_000, b"2026:10:21 09:30:00",
                20261021, ("2026:10:21 09:30:00",)):
        assert capture_time(bad) is None, bad


def test_only_a_sequential_single_scan_jpeg_decodes_reduced():
    """exif.jpeg_decodes_reduced, PhotoMeta.decodes_reduced: what a camera writes (sequential, all components in one
    scan; grey, any subsampling, optimised tables, an MPO's first frame) decodes at reduced size. A progressive JPEG
    does not, nor does a sequential one with a scan per component, which Pillow does not report as progressive and
    libjpeg buffers in full all the same. Anything the scan of the segments cannot follow is answered False."""
    for save in ({}, {"subsampling": 0}, {"subsampling": 2}, {"optimize": True}, {"quality": 30}):
        data = hostile.jpeg(**save)
        assert jpeg_decodes_reduced(data) and read_photo_meta(data).decodes_reduced, save
    assert jpeg_decodes_reduced(hostile.jpeg(mode="L"))
    with_exif = hostile.with_exif("JPEG", hostile.exif_with(capture_time=b"x" * 200_000))    # APP1 segments first
    assert jpeg_decodes_reduced(with_exif) and read_photo_meta(with_exif).decodes_reduced

    progressive = hostile.jpeg(progressive=True)
    with Image.open(io.BytesIO(progressive)) as im:
        assert im.info.get("progressive") == 1
    assert not jpeg_decodes_reduced(progressive) and not read_photo_meta(progressive).decodes_reduced
    assert not jpeg_decodes_reduced(hostile.jpeg(mode="L", progressive=True))

    scans = hostile.non_interleaved_jpeg()
    with Image.open(io.BytesIO(scans)) as im:                       # a readable JPEG, and nothing in its info says so
        assert (im.format, im.size, im.mode) == ("JPEG", (640, 480), "RGB") and "progressive" not in im.info
        im.load()
    assert not jpeg_decodes_reduced(scans) and not read_photo_meta(scans).decodes_reduced

    buf = io.BytesIO()
    hostile.textured().save(buf, "MPO", save_all=True, append_images=[hostile.textured()])
    meta = read_photo_meta(buf.getvalue())
    assert (meta.format, meta.decodes_reduced) == ("MPO", True)
    buf = io.BytesIO()
    hostile.textured().save(buf, "MPO", save_all=True, append_images=[hostile.textured()], progressive=True)
    assert not read_photo_meta(buf.getvalue()).decodes_reduced

    for fmt in ("PNG", "WEBP"):                                     # no reduced decode in these formats
        buf = io.BytesIO()
        hostile.textured().save(buf, fmt)
        assert not jpeg_decodes_reduced(buf.getvalue()) and not read_photo_meta(buf.getvalue()).decodes_reduced

    good = hostile.jpeg()
    sof = good.index(b"\xff\xc0")
    sos = good.index(b"\xff\xda")
    assert good[sof + 9] == good[sos + 4] == 3                       # the frame's components, the first scan's
    crafted = {
        "empty": b"", "not a JPEG": b"\x89PNG\r\n", "SOI only": good[:2], "cut before the scan": good[:sos],
        "cut inside the frame header": good[:sof + 6],
        "junk before a marker": good[:2] + b"\x00" + good[2:],
        "an FF 00 pair before a marker": good[:2] + b"\xff\x00" + good[2:],
        "a marker without a length (RST0)": good[:2] + b"\xff\xd0" + good[2:],
        "a reserved marker": good[:2] + b"\xff\x02\x00\x02" + good[2:],
        "two frame headers": good[:sof] + good[sof:sof + 2 + int.from_bytes(good[sof + 2:sof + 4], "big")] + good[sof:],
        "a scan before the frame header": good[:2] + good[sos:],
        "a scan of fewer components": good[:sos + 4] + b"\x01" + good[sos + 5:],
        "a progressive frame header": good[:sof + 1] + b"\xc2" + good[sof + 2:],
        "a lossless frame header": good[:sof + 1] + b"\xc3" + good[sof + 2:],
        "an arithmetic-coded frame header": good[:sof + 1] + b"\xc9" + good[sof + 2:],
        "a segment length of 1": good[:4] + b"\x00\x01" + good[6:],
        "a segment length past the end": good[:4] + b"\xff\xff" + good[6:],
    }
    for what, data in crafted.items():
        assert jpeg_decodes_reduced(data) is False, what
    assert jpeg_decodes_reduced(good[:2] + b"\xff\xff\xff" + good[2:])       # fill bytes before a marker are legal
    comment = b"\xff\xfe\x00\x02"                                           # an empty COM segment
    assert jpeg_decodes_reduced(good[:2] + comment * 900 + good[2:])
    assert not jpeg_decodes_reduced(good[:2] + comment * 1000 + good[2:])     # the walk is bounded: not followed
    assert not jpeg_decodes_reduced(good[:2] + b"\xff" * 1000 + good[2:])
    start = time.perf_counter()
    assert not jpeg_decodes_reduced(good[:2] + b"\xff" * 5_000_000 + good[2:])
    assert not jpeg_decodes_reduced(good[:2] + comment * 1_000_000 + good[2:])
    assert time.perf_counter() - start < 0.5
    assert jpeg_decodes_reduced(good[:sof + 1] + b"\xc1" + good[sof + 2:])      # SOF1: sequential too
