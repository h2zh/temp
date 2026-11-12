import dataclasses
import io
import math
import random
import struct
import time
import tracemalloc
import zlib

import piexif
import piexif._load
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
    """exif._text: bytes or str, stripped, cut at MAX_TEXT_CHARS; anything else (a text tag written as SHORT or
    RATIONAL is read as an int or a tuple, as piexif read it) is no tag, not an AttributeError."""
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
    """Every format carries EXIF of any size (a JPEG in several segments, which Pillow joins). In a PNG or a WebP a
    1,000,000-character DateTimeOriginal, make or model is read as its first MAX_TEXT_CHARS characters. A JPEG
    carrying it is unreadable (review round 2: Pillow parses a JPEG's EXIF while it opens the file, so a JPEG's is
    held to what one segment takes), and the longest texts one segment takes are read as 64 characters too."""
    long = b"2024:05:06 07:08:09" + b"9" * (1_000_000 - 19)
    data = hostile.with_exif(fmt, hostile.exif_with(capture_time=long, lat=31.25, lon=121.5, make=b"M" * 1_000_000))
    with Image.open(io.BytesIO(data)) as im:
        assert len(piexif.load(im.info["exif"])["Exif"][piexif.ExifIFD.DateTimeOriginal]) == 1_000_000
    if fmt == "JPEG":
        with pytest.raises(UnreadableImage, match="EXIF over 65536 bytes in a JPEG"):
            read_photo_meta(data)
        long = long[:32_000]
        data = hostile.with_exif(fmt, hostile.exif_with(capture_time=long, lat=31.25, lon=121.5, make=b"M" * 32_000))
        assert [m for m, _, _ in hostile._header_segments(data)].count(0xE1) == 1 and len(data) > 64_000
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
    assert jpeg_decodes_reduced(hostile.with_exif("JPEG", hostile.exif_with(capture_time=b"x" * 200_000)))
    with_exif = hostile.with_exif("JPEG", hostile.exif_with(capture_time=b"x" * 60_000))     # an APP1 segment first
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


# The EXIF block is read by the module's own bounded reader, not by piexif (D34, review round 1)

READ_TAGS = {name: exif_module._READ_TAGS[name] for name in ("0th", "Exif", "GPS")}
POINTER_TAGS = {piexif.ImageIFD.ExifTag: "Exif", piexif.ImageIFD.GPSTag: "GPS",
                piexif.ExifIFD.InteroperabilityTag: "Interop"}
FOLLOWED = {("0th", "Exif"), ("0th", "GPS"), ("Exif", "Interop")}      # (directory, the directory its tag points to)
MB = 1024 * 1024


class TooCostly(BaseException):
    pass


class GuardedStruct:
    """struct, for piexif while a test uses it as the oracle: a format of one character per declared value (what
    piexif builds) is refused before it is compiled, so no generated block can cost the test its memory."""
    error = struct.error

    @staticmethod
    def unpack(fmt, data):
        if len(fmt) > 2000:
            raise TooCostly
        return struct.unpack(fmt, data)


def piexif_reading(blob):
    """What piexif.load made of an EXIF block, as far as read_photo_meta looks: None when it raised, else the tags of
    exif._READ_TAGS, a text cut at exif._MAX_TEXT_BYTES and a tuple of more than exif._MAX_NUMBERS numbers replaced
    by exif._TOO_MANY (the two bounds of exif._tiff_value)."""
    try:
        loaded = piexif.load(blob)
    except Exception:
        return None

    def bounded(value):
        if isinstance(value, bytes):
            return value[:exif_module._MAX_TEXT_BYTES]
        if isinstance(value, tuple) and len(value) > exif_module._MAX_NUMBERS:
            return exif_module._TOO_MANY
        return value

    return {name: {tag: bounded(v) for tag, v in loaded[name].items() if tag in tags}
            for name, tags in READ_TAGS.items()}


def canonical(reading):
    """A reading as text, so that two of them compare equal with a NaN in them (a FLOAT tag) and in any tag order."""
    return None if reading is None else repr({name: sorted(reading[name].items()) for name in sorted(reading)})


def random_exif_block(rng):
    """An EXIF blob of five directories (the first, Exif, GPS, interoperability, the thumbnail's), each with random
    entries: the tags read_photo_meta reads, the pointer tags, other tags piexif knows and unknown ones; any of
    TIFF's twelve types and a few that are none; counts of 0 to 64; values in the entry or behind an offset inside
    or outside the block; pointers of the wrong type, to the wrong place or negative; either byte order; sometimes
    cut short. The directories never overlap and every pointer that lies inside the block points at one, so no
    directory is read from random bytes: piexif stays affordable as the oracle."""
    endian = rng.choice("<>")
    names = ["0th", "Exif", "GPS", "Interop", "1st"]
    counts = {name: rng.choice([0, 1, 2, 3, 5, 8, 12]) for name in names}
    data_bytes = rng.choice([16, 80, 300, 900])
    where, at = {}, 8 + data_bytes
    for name in names:
        where[name] = at
        at += 2 + 12 * counts[name] + 4 + rng.choice([0, 0, 2, 10])
    size = at + rng.choice([0, 4, 40])
    tame = rng.random() < 0.7               # mostly well-formed: TIFF's own types, values inside the block
    negative = False

    def pointer_field(target):
        nonlocal negative
        kind = rng.choice([4] * 12 + [3, 9, 9] + ([4] if tame and rng.random() < 0.9 else [1, 11, 2, 5]))
        if kind == 1 and target > 255:
            kind = 4
        if kind == 9 and rng.random() < 0.5:
            target, negative = target - size, True          # the same place, counted from the block's end
        if rng.random() < (0.03 if tame else 0.15):
            target = rng.choice([size, size + 5, 2 ** 31 - 1])
            kind = 4
        count = rng.choice([1] * (60 if tame else 6) + [0, 2])      # (two numbers are a tuple, which is no pointer)
        if kind == 11:
            return kind, count, struct.pack(endian + "f", float(target))
        if kind in (2, 5):
            return kind, count, struct.pack(endian + "L", target)
        return kind, count, struct.pack(endian + {4: "L", 3: "H", 9: "l", 1: "B"}[kind], target).ljust(4, b"\0")

    body = bytearray(rng.randbytes(size))
    for name in names:
        known, read = sorted(exif_module._KNOWN_TAGS[name]), sorted(exif_module._READ_TAGS[name])
        entries = []
        for _ in range(counts[name]):
            pick = rng.random()
            tag = (rng.choice(read) if pick < 0.5 and read else rng.choice(known) if pick < 0.85
                   else rng.randrange(65536))
            kind = rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12] + ([] if tame else [0, 13, 200]))
            count = rng.choice([1, 1, 1, 2, 2, 3, 3, 4, 5] if tame else [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 20, 64])
            in_entry = count * {1: 1, 2: 1, 6: 1, 7: 1, 3: 2, 8: 2, 4: 4, 9: 4, 11: 4}.get(kind, 8) <= 4
            if (name, POINTER_TAGS.get(tag)) in FOLLOWED:       # always: a pointer never leads into random bytes
                kind, count, field = pointer_field(where[POINTER_TAGS[tag]])
            elif tame and name == "1st" and tag in exif_module._THUMBNAIL and rng.random() < 0.9:
                kind, count, field = 4, 1, struct.pack(endian + "L", rng.randrange(0, size))
            elif in_entry if tame else rng.random() < 0.5:
                field = rng.randbytes(4)
            elif tame:
                field = struct.pack(endian + "L", rng.randrange(0, max(1, size - 100)))
            else:
                field = struct.pack(endian + "L", rng.choice([rng.randrange(0, size), size - rng.randrange(0, 12),
                                                               size + rng.randrange(0, 9), 2 ** 32 - 1]))
            entries.append((tag, kind, count, field))
        following = 0
        if name == "0th":
            following = rng.choice([where["1st"]] * 6 + [0, 0, 0, size + 3])
        block = hostile.directory(entries, endian, following)
        body[where[name]:where[name] + len(block)] = block
    first = where["0th"] if rng.random() < 0.97 else rng.choice([size, size - 2, 2 ** 32 - 1])
    order = b"II*\x00" if endian == "<" else b"MM\x00*"
    tiff = order + struct.pack(endian + "L", first) + bytes(body[8:])
    if not negative and rng.random() < (0.03 if tame else 0.2):
        tiff = tiff[:rng.randrange(0, len(tiff))]
    return hostile.EXIF_HEADER + tiff


def changed_exif_block(rng, blob):
    """A well-formed EXIF blob as piexif writes it (big-endian) with one to three of its entries changed: declared
    as another type, with another count, or with another value field. The entries that point to a directory stay as
    they are, so every directory is still read from where it is."""
    tiff = bytearray(blob[6:])

    def entries(at):
        return [at + 2 + 12 * i for i in range(struct.unpack(">H", tiff[at:at + 2])[0])]

    first = entries(8)
    pointers = {struct.unpack(">H", tiff[e:e + 2])[0]: struct.unpack(">L", tiff[e + 8:e + 12])[0] for e in first}
    every = first + entries(pointers[piexif.ImageIFD.ExifTag]) + entries(pointers[piexif.ImageIFD.GPSTag])
    changeable = [e for e in every if struct.unpack(">H", tiff[e:e + 2])[0] not in POINTER_TAGS]
    for _ in range(rng.choice([1, 1, 2, 3])):
        e, what = rng.choice(changeable), rng.random()
        if what < 0.4:
            tiff[e + 2:e + 4] = struct.pack(">H", rng.randrange(0, 14))
        elif what < 0.7:
            tiff[e + 4:e + 8] = struct.pack(">L", rng.choice([0, 1, 2, 3, 4, 5, 9, 64]))
        elif what < 0.85:
            tiff[e + 8:e + 12] = rng.randbytes(4)
        else:
            tiff[e + 8:e + 12] = struct.pack(">L", rng.randrange(len(tiff) + 8))
    return hostile.EXIF_HEADER + bytes(tiff)


def test_the_reader_reads_every_exif_block_as_piexif_did(monkeypatch):
    """exif._read_exif took piexif.load's place (the review: piexif is not safe on an upload). No photo may read
    differently for it, so on 7,000 generated blocks (4,000 of random_exif_block, 3,000 of changed_exif_block on a
    block with every field set) it gives exactly what piexif.load gave: the same blocks unreadable, the same values
    for every tag read_photo_meta uses. And read_photo_meta, given piexif's whole output in the reader's place,
    answers the same PhotoMeta: the reader's two bounds change nothing after it."""
    monkeypatch.setattr(piexif._load, "struct", GuardedStruct)
    rng = random.Random(34)
    full = piexif.load(build_exif_bytes(lat=-31.25, lon=121.5, alt=-12.5, gps_accuracy_m=4.5, heading_deg=271.5,
                                        heading_ref="M", focal_mm=4.25, focal_35mm=26,
                                        captured_at="2024:05:06 07:08:09", make="Make", model="Model"))
    full["0th"][piexif.ImageIFD.Orientation] = 6
    full = piexif.dump(full)
    blocks = [changed_exif_block(rng, full) for _ in range(3000)] + [random_exif_block(rng) for _ in range(4000)]
    readable = with_values = 0
    for blob in blocks:
        want = piexif_reading(blob)
        assert canonical(exif_module._read_exif(blob)) == canonical(want), blob.hex()
        readable += want is not None
        with_values += bool(want) and sum(len(tags) for tags in want.values()) >= 2
    print(f"{readable} of {len(blocks)} blocks readable, {with_values} with two or more of the tags read")
    assert readable > 3500 and with_values > 2500, (readable, with_values)      # the comparison is not an empty one

    def as_before(blob):
        try:
            return piexif.load(blob)
        except Exception:
            return None

    tiny = Image.new("RGB", (8, 8))
    metas = {}
    for blob in blocks[:3500]:
        data = hostile.with_exif("PNG", blob, tiny)
        now = read_photo_meta(data)
        with monkeypatch.context() as patched:
            patched.setattr(exif_module, "_read_exif", as_before)
            before = read_photo_meta(data)
        assert repr(now) == repr(before), blob.hex()
        metas[repr(now)] = now
    whole = read_photo_meta(hostile.with_exif("PNG", full, tiny))
    assert whole.has_gps and None not in (whole.alt, whole.heading_deg, whole.focal_mm, whole.orientation, whole.make)
    fields = ("has_exif", "has_gps", "alt", "gps_accuracy_m", "heading_deg", "heading_ref", "focal_mm", "focal_35mm",
              "captured_at", "make", "model", "orientation")
    seen = {name: len({repr(getattr(meta, name)) for meta in metas.values()}) for name in fields}
    print(f"{len(metas)} different PhotoMeta; different values per field: {seen}")
    assert len(metas) > 300 and all(n >= 2 for n in seen.values()), (len(metas), seen)     # every field was compared


def test_a_pointer_that_is_no_number_leads_nowhere():
    """The Exif and GPS directories are found through pointer tags. One written as text, a float, a rational or two
    numbers is no offset: the block is unreadable, as it was for piexif (a TypeError there). It is not read as
    offset 0: this block starts with bytes that are a directory of one entry there (a capture time), and its real
    first directory, at the offset those same bytes spell, holds only the broken pointer."""
    fake = struct.pack(">HHHLL", 1, piexif.ExifIFD.DateTimeOriginal, hostile.TIFF_ASCII, 20, 64) + b"\x00" * 4
    first = struct.unpack(">L", fake[4:8])[0]                       # what the type and count read as: 131072
    assert first == 0x20000
    time_text = b"2001:02:03 04:05:06\x00"
    for tag, name in ((piexif.ImageIFD.ExifTag, "Exif"), (piexif.ImageIFD.GPSTag, "GPS")):
        for kind, count, field in ((11, 1, struct.pack(">f", 0.0)), (hostile.TIFF_ASCII, 2, b"\x00" * 4),
                                   (hostile.TIFF_RATIONAL, 1, struct.pack(">L", 64)),
                                   (hostile.TIFF_LONG, 2, struct.pack(">L", 64)), (hostile.TIFF_SHORT, 2, b"\x00" * 4)):
            tiff = (fake + b"\x00" * (64 - len(fake)) + time_text).ljust(first, b"\x00")
            tiff += hostile.directory([(tag, kind, count, field)])
            blob = hostile.EXIF_HEADER + tiff
            with pytest.raises(Exception):
                piexif.load(blob)
            assert exif_module._read_exif(blob) is None, (name, kind, count)
        # the same block with a pointer that is a number, 0: the directory at the start is read
        blob = hostile.EXIF_HEADER + (fake + b"\x00" * (64 - len(fake)) + time_text).ljust(first, b"\x00") + \
            hostile.directory([(tag, hostile.TIFF_LONG, 1, 0)])
        assert canonical(exif_module._read_exif(blob)) == canonical(piexif_reading(blob))
        if name == "Exif":
            assert exif_module._read_exif(blob)["Exif"] == {piexif.ExifIFD.DateTimeOriginal: time_text[:-1]}


def test_the_reader_reads_what_cameras_and_the_writer_produce():
    """The same comparison on well-formed EXIF: build_exif_bytes' output with every field set, in each byte order
    and with the header a WebP chunk may lack."""
    blob = build_exif_bytes(lat=-31.25, lon=121.5, alt=-12.5, gps_accuracy_m=4.5, heading_deg=271.5, heading_ref="M",
                            focal_mm=4.25, focal_35mm=26, captured_at="2024:05:06 07:08:09", make="Make", model="Model")
    loaded = piexif.load(blob)
    loaded["0th"][piexif.ImageIFD.Orientation] = 6
    loaded["Exif"][piexif.ExifIFD.MakerNote] = b"\x01" * 5000
    blob = piexif.dump(loaded)
    reading = exif_module._read_exif(blob)
    assert canonical(reading) == canonical(piexif_reading(blob))
    assert reading["0th"][piexif.ImageIFD.Orientation] == 6 and reading["Exif"][piexif.ExifIFD.FocalLength] == (17, 4)
    assert reading["GPS"][piexif.GPSIFD.GPSLatitudeRef] == b"S" and len(reading["GPS"]) == 9
    assert canonical(exif_module._read_exif(blob[6:])) == canonical(reading)        # a WebP chunk: the TIFF block alone
    meta = read_photo_meta(hostile.with_exif("WEBP", blob))
    assert (meta.lat, meta.lon, meta.alt) == (pytest.approx(-31.25), pytest.approx(121.5), pytest.approx(-12.5))
    assert (meta.orientation, meta.heading_ref, meta.make, meta.captured_at) == (6, "M", "Make", "2024:05:06 07:08:09")


def cost_of(call, *args):
    """(result or the exception raised, seconds, peak bytes of Python allocations) of one call. The seconds are
    processor time of this process, not time on the clock: the calls measured here read bytes they hold in memory
    and wait for nothing, so alone on a machine the two are the same number (0.287-0.295 s both ways for the
    slowest header flood), while on a busy one the clock counts the other processes too (1.27-1.49 s against
    0.31-0.32 s with eight busy loops on four cores; a parallel run of the suite, pytest -n 4, failed on that)."""
    tracemalloc.start()
    start = time.process_time()
    try:
        result = call(*args)
    except Exception as exc:
        result = exc
    seconds = time.process_time() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return result, seconds, peak


@pytest.mark.filterwarnings("ignore:Tag data larger than file:UserWarning")
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_a_declared_count_of_billions_costs_nothing(fmt):
    """The review's finding: an entry that declares 60,000,000 BYTEs made piexif.load build a struct format of one
    character per value: a 1.4 KB upload took 2 GB, and the struct module kept it. A count of 2**31 or 2**32 - 1 of
    any number type, in any format, is now read in milliseconds and kilobytes; the block counts as unreadable, as it
    did (its numbers lie outside the block)."""
    kinds = (hostile.TIFF_BYTE, hostile.TIFF_SHORT, hostile.TIFF_LONG, hostile.TIFF_RATIONAL, 6, 8, 9, 10, 11, 12)
    small = hostile.textured((64, 48))
    make = (piexif.ImageIFD.Make, 2, 5, 8)
    good = read_photo_meta(hostile.with_exif(fmt, hostile.exif_block([make], b"Acme\0"), small))
    assert good.has_exif and good.make == "Acme"
    for kind in kinds:
        for count in (60_000_000, 2 ** 31, 2 ** 32 - 1):
            for tag in (piexif.ImageIFD.Make, piexif.ImageIFD.XResolution):       # a tag that is read, one that is not
                blob = hostile.exif_block([make, (tag, kind, count, 8)], b"Acme\0")
                meta, seconds, peak = cost_of(read_photo_meta, hostile.with_exif(fmt, blob, small))
                assert not meta.has_exif and meta.make is None and meta.format == fmt, (kind, count, tag)
                assert seconds < 1.0 and peak < MB, (kind, count, tag, seconds, peak)


def test_numbers_that_are_really_there_are_not_all_read():
    """A count the block does hold (a megabyte of numbers behind one tag): piexif read them all into a tuple, at 33
    bytes of RAM a value. The reader takes at most exif._MAX_NUMBERS: such a tag is exif._TOO_MANY, no value to
    read_photo_meta, like the long tuple before it; the rest of the block is read."""
    filler = b"Acme\0\0\0\0" + b"\x00\x00\x00\x01" * (MB // 4)
    for kind, each in ((hostile.TIFF_BYTE, 1), (hostile.TIFF_SHORT, 2), (hostile.TIFF_LONG, 4),
                       (hostile.TIFF_RATIONAL, 8), (12, 8)):
        entries = [(piexif.ImageIFD.Make, 2, 5, 8), (piexif.ImageIFD.Model, kind, MB // each, 16),
                   (piexif.ImageIFD.Orientation, kind, MB // each, 16)]
        blob = hostile.exif_block(entries, filler)
        reading, seconds, peak = cost_of(exif_module._read_exif, blob)
        assert reading["0th"] == {piexif.ImageIFD.Make: b"Acme", piexif.ImageIFD.Model: exif_module._TOO_MANY,
                                  piexif.ImageIFD.Orientation: exif_module._TOO_MANY}
        assert seconds < 0.5 and peak < 2 * MB, (kind, seconds, peak)           # (the block's own megabyte, once)
        meta = read_photo_meta(hostile.with_exif("PNG", blob))
        assert (meta.has_exif, meta.make, meta.model, meta.orientation) == (True, "Acme", None, None)
    assert exif_module._MAX_NUMBERS == 8
    for count, value in ((8, tuple([(1, 1)] * 8)), (9, ())):
        blob = hostile.exif_block([(piexif.GPSIFD.GPSLatitude, hostile.TIFF_RATIONAL, count, 16)], filler)
        assert exif_module._tiff_directory(blob[6:], ">", 8 + len(filler), "GPS")[0] == {2: value}


@pytest.mark.filterwarnings("ignore:Tag data larger than file:UserWarning")
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_many_tags_at_one_value_do_not_multiply_it(fmt):
    """Counts the block really holds, many times over: every tag piexif knows in the first directory pointing at
    the same 2 MB value. piexif copied the value once per tag (187 tags: 370 MB as text; as rationals, 49 million
    tuples). The reader copies a text's first exif._MAX_TEXT_BYTES bytes for the tags it reads, and nothing else."""
    tags = sorted(tag for tag in piexif.TAGS["Image"] if tag not in POINTER_TAGS)
    assert len(tags) == 187 and piexif.ImageIFD.Make in tags
    filler = b"M" * (60_000 if fmt == "JPEG" else 2 * MB)       # (a JPEG's EXIF: what one segment takes)
    # what reading the file costs with no tag at all in the same block (Pillow reads the chunk, or a JPEG's segment)
    _, _, bare = cost_of(read_photo_meta, hostile.with_exif(fmt, hostile.exif_block([], filler)))
    assert bare < 16 * MB
    for kind, count in ((hostile.TIFF_ASCII, len(filler)), (hostile.TIFF_UNDEFINED, len(filler)),
                        (hostile.TIFF_RATIONAL, len(filler) // 8), (hostile.TIFF_SHORT, len(filler) // 2)):
        blob = hostile.exif_block([(tag, kind, count, 8) for tag in tags], filler)
        _, seconds, peak = cost_of(exif_module._read_exif, blob)
        assert seconds < 0.5 and peak < len(blob) + 64 * 1024, (kind, seconds, peak)     # the reader copies it once
        meta, seconds, peak = cost_of(read_photo_meta, hostile.with_exif(fmt, blob))
        text = kind in (hostile.TIFF_ASCII, hostile.TIFF_UNDEFINED)
        assert meta.has_exif and meta.make == ("M" * 64 if text else None)
        # the whole reading: for a JPEG, Pillow itself parses the first directory when it opens the file and copies
        # one value of the block's size (it skips the tags after it: "Tag data larger than file")
        assert seconds < 1.0 and peak < bare + (3 * len(filler) if fmt == "JPEG" else MB), (kind, seconds, peak, bare)


def test_a_directory_of_more_than_1024_entries_is_not_read():
    """Cameras write dozens of entries; a directory of over exif._MAX_IFD_ENTRIES is not walked (65,535 are possible,
    in each of five directories): the block is unreadable."""
    assert exif_module._MAX_IFD_ENTRIES == 1024
    make = (piexif.ImageIFD.Make, 2, 5, 8)
    filler_tag = 0xEA1C                                              # a tag piexif does not know
    for extra, readable in ((1023, True), (1024, False), (65534, False)):
        blob = hostile.exif_block([make] + [(filler_tag, 7, 1, b"\0\0\0\0")] * extra, b"Acme\0")
        reading, seconds, peak = cost_of(exif_module._read_exif, blob)
        assert (reading is not None) == readable and seconds < 0.5, (extra, seconds)
        assert read_photo_meta(hostile.with_exif("PNG", blob)).make == ("Acme" if readable else None)


def test_an_exif_chunk_is_never_opened_as_a_file(tmp_path):
    """A WebP's EXIF chunk reaches the reader as it is. piexif.load takes input that is neither TIFF nor JPEG nor
    WebP for a file name: a chunk holding a path made the server open that file and read its EXIF as the upload's
    (another photo's GPS fix, camera and time). And it read a JPEG or WebP file embedded in the chunk. Now a chunk
    that starts neither with "Exif" nor with a TIFF byte order is no EXIF at all."""
    victim = tmp_path / "victim.jpg"
    secret = hostile.exif_with(capture_time=b"2001:02:03 04:05:06", lat=12.25, lon=98.75, make=b"SecretCam")
    victim.write_bytes(hostile.with_exif("JPEG", secret))
    path = str(victim).encode()
    assert piexif.load(path)["0th"][piexif.ImageIFD.Make] == b"SecretCam"       # what piexif does with such bytes
    for chunk in (path, victim.read_bytes(), hostile.with_exif("WEBP", secret), b"/dev/zero", b"", b"\x00" * 64):
        if chunk:
            data = hostile.webp_with_chunk(chunk)
            with Image.open(io.BytesIO(data)) as im:
                assert im.info["exif"] == chunk
            meta = read_photo_meta(data)
            assert (meta.format, meta.has_exif, meta.has_gps, meta.make, meta.captured_at) == ("WEBP", False, False,
                                                                                               None, None)
        assert exif_module._read_exif(chunk) is None
    for chunk in (secret, secret[6:]):                              # with and without the header: read
        assert read_photo_meta(hostile.webp_with_chunk(chunk)).make == "SecretCam"


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_an_exif_block_over_4_mib_makes_the_image_unreadable(fmt):
    """Pillow's own EXIF parser runs on the block when the pixels are turned upright (about 0.2 s and 25 MB per MiB
    of numbers, measured), so a PNG's or WebP's block over exif.MAX_EXIF_BYTES is refused before anything parses
    it: the image is unreadable. A camera's block is at most 64 KiB (a JPEG's limit: MAX_JPEG_EXIF_BYTES, below)."""
    assert exif_module.MAX_EXIF_BYTES == 4 * MB
    make = (piexif.ImageIFD.Make, 2, 5, 8)
    header = 0 if fmt == "WEBP" else 6          # Pillow hands out a WebP's chunk as it is, a PNG's after "Exif\0\0"
    for extra, readable in ((0, True), (1, False)):
        blob = hostile.exif_block([make], b"Acme\0".ljust(4 * MB - header - 8 - 18 + extra, b"\0"))
        data = hostile.with_exif(fmt, blob)
        with Image.open(io.BytesIO(data)) as im:
            assert len(im.info["exif"]) == 4 * MB + extra
        if readable:
            assert read_photo_meta(data).make == "Acme"
        else:
            with pytest.raises(UnreadableImage, match="EXIF block over 4194304 bytes"):
                read_photo_meta(data)


# A header that holds more than a camera's is refused before Pillow opens the file (D34, review round 2)

FLOODS = hostile.header_floods()
JPEG_LIMITS = ("_JPEG_MAX_SEGMENTS", "_JPEG_MAX_SKIPPED", "MAX_JPEG_EXIF_BYTES", "_JPEG_MAX_TABLE_BYTES",
               "_JPEG_MAX_RESOURCE_BYTES", "_JPEG_MAX_FRAME_BYTES")


def pillow_must_not_open(monkeypatch):
    def opened(*args, **kwargs):
        raise AssertionError("Pillow opened the file")

    monkeypatch.setattr(exif_module.Image, "open", opened)


def refused(data) -> bool:
    try:
        exif_module._check_header(data)
    except UnreadableImage:
        return True
    return False


@pytest.mark.parametrize("name", sorted(FLOODS))
def test_a_header_flood_is_unreadable_before_pillow_sees_it(name, monkeypatch):
    """The review's finding, and the same thing wherever it turned up afterwards. Pillow reads a header in Python and
    keeps objects for what it finds, so 24 MB of empty JPEG segments took 1.4 GB and 9 s to open, 24 MB of fat frame
    headers 1.4 GB, 2,000,000 empty PNG chunks 0.5 GB and 7 s, a 69 KB PNG of compressed text 64 MiB (hostile.
    header_floods has the list; the numbers are in docs/decisions/D34-review-fixes.md). Each is unreadable now, and
    Pillow never gets the file: refusing it takes milliseconds and no memory to speak of."""
    data = FLOODS[name]()
    assert len(data) <= 25 * MB
    pillow_must_not_open(monkeypatch)
    result, seconds, peak = cost_of(read_photo_meta, data)
    assert isinstance(result, UnreadableImage), result
    assert seconds < 0.5 and peak < 256 * 1024, (seconds, peak)


def test_the_jpeg_walk_reads_each_marker_as_pillow_does():
    """exif._check_jpeg_header bounds what Pillow does when it opens a JPEG only as long as both read the header the
    same way. Pillow's marker table, marker by marker: a segment with a length where Pillow's handler reads one, a
    bare marker where Pillow has no handler, the end of the walk where Pillow knows no marker; and the frame, table
    and APP handlers are on the markers the limits count. A Pillow whose table changes fails here first."""
    from PIL import JpegImagePlugin as plugin

    for code in range(256):
        entry = plugin.MARKER.get(0xFF00 | code)
        handler = entry[2] if entry else None
        assert (code in exif_module._JPEG_SEGMENTS) == (handler is not None), hex(code)
        assert (code in exif_module._JPEG_BARE) == (entry is not None and handler is None), hex(code)
        assert (code in exif_module._JPEG_FRAMES) == (handler is plugin.SOF), hex(code)
        assert (code == exif_module._JPEG_DQT) == (handler is plugin.DQT), hex(code)
        assert (0xE0 <= code <= 0xEF) == (handler is plugin.APP), hex(code)
    assert plugin.MARKER[0xFF00 | exif_module._JPEG_SOS][0] == "SOS"
    assert (exif_module._JPEG_APP1, exif_module._JPEG_APP13) == (0xE1, 0xED)


class CountedReads(io.BytesIO):
    """A file that counts how often it is read."""

    def __init__(self, data):
        super().__init__(data)
        self.reads = 0

    def read(self, *args):
        self.reads += 1
        return super().read(*args)


def test_pillow_does_no_more_with_a_jpeg_header_than_the_walk_let_through(monkeypatch):
    """The same, on 4,000 generated headers and with small limits: some random pieces (segments of every kind, junk,
    fill bytes, stuffed zeros, bare markers, bytes that are no marker, lengths of 0 and 1, an early scan, a cut-off
    end), then 2,000 empty segments, then a real JPEG's own header and scan. Whenever the walk lets a file through,
    Pillow's work on it stays within the limits too: the times it reads the file, the segments it keeps, the EXIF
    it joins, the frame components it lists. A walk that lost step with Pillow anywhere (and stopped early, or took
    a payload for markers) would let Pillow into the 2,000 segments."""
    from PIL import JpegImagePlugin as plugin

    small = dict(zip(JPEG_LIMITS, (24, 64, 120, 260, 200, 36)))
    for name, value in small.items():
        monkeypatch.setattr(exif_module, name, value)
    tail = hostile.jpeg(size=(16, 16))[2:]                  # APP0, two DQT, SOF0, four DHT, SOS and the scan
    flood = hostile.segment(0xE5) * 2000
    rng = random.Random(34)
    others = sorted(exif_module._JPEG_SEGMENTS - exif_module._JPEG_FRAMES - {0xDA, 0xDB, 0xE1, 0xED})

    def piece():
        kind = rng.randrange(13)
        if kind == 0:
            return bytes(rng.randrange(256) for _ in range(rng.randrange(1, 12)))        # junk, 0xFF among it
        if kind == 1:
            return b"\xff" * rng.randrange(1, 8)
        if kind == 2:
            return b"\xff\x00" * rng.randrange(1, 4)
        if kind == 3:
            return bytes([0xFF, rng.choice(sorted(exif_module._JPEG_BARE))])
        if kind == 4:
            return hostile.segment(0xE1, hostile.EXIF_HEADER[:rng.choice((6, 6, 6, 5))] + bytes(rng.randrange(40)))
        if kind == 5:
            return hostile.segment(0xED, hostile.PHOTOSHOP + hostile.RESOURCE_BLOCK * rng.randrange(8))
        if kind == 6:
            return hostile.segment(0xDB, hostile.QUANT_TABLE * rng.randrange(3))
        if kind == 7:
            return hostile.frame_header(rng.choice(sorted(exif_module._JPEG_FRAMES)), rng.randrange(6, 30))
        if kind == 8:
            return bytes([0xFF, rng.randrange(1, 0xC0)]) if rng.random() < 0.3 else b""    # no marker: Pillow stops
        if kind == 9:
            return bytes([0xFF, rng.choice(others), 0, rng.randrange(2)])                # a length of 0 or 1
        if kind == 10:
            return hostile.segment(0xDA, bytes(6)) if rng.random() < 0.3 else b""        # an early scan: both stop
        return hostile.segment(rng.choice(others), bytes(rng.randrange(256) for _ in range(rng.randrange(20))))

    let_through = opened = kept_out = 0
    most_reads = 0
    for _ in range(4000):
        data = b"\xff\xd8" + b"".join(piece() for _ in range(rng.randrange(16))) + flood + tail
        if rng.random() < 0.1:
            data = data[:rng.randrange(2, 200)]
        if refused(data):
            kept_out += 1
            continue
        let_through += 1
        fp = CountedReads(data)
        try:
            im = plugin.JpegImageFile(fp)
        except Exception:                   # not a JPEG to Pillow, for a reason of its own
            im = None
        most_reads = max(most_reads, fp.reads)
        # per segment: its marker, its length, its payload, the byte after it; per byte passed over: at most two
        assert fp.reads <= 4 * small["_JPEG_MAX_SEGMENTS"] + 2 * small["_JPEG_MAX_SKIPPED"] + 8, (fp.reads, data[:80])
        if im is not None:
            opened += 1
            assert len(im.applist) <= small["_JPEG_MAX_SEGMENTS"]
            assert len(im.info.get("exif", b"")) <= small["MAX_JPEG_EXIF_BYTES"]
            assert len(im.layer) <= small["_JPEG_MAX_FRAME_BYTES"] // 3
    assert let_through > 400 and opened > 50 and kept_out > 2000, (let_through, opened, kept_out)
    assert most_reads > 60                                  # and the bound was not an empty one


def test_a_jpeg_header_at_each_limit_is_read_and_one_step_over_is_not():
    """The JPEG limits, each at its edge, on a real JPEG's header (APP0, two DQT of 65 bytes, a 15-byte frame
    header, four DHT, SOS: nine segments). Counted as Pillow does its work: segments up to and with the first scan's
    header, every byte passed over, "Exif" APP1 payloads whole, DQT payloads, Photoshop APP13 payloads, frame
    header payloads (DHP too)."""
    good = hostile.jpeg()
    own = [m for m, _, _ in hostile._header_segments(good)]
    assert own == [0xE0, 0xDB, 0xDB, 0xC0, 0xC4, 0xC4, 0xC4, 0xC4, 0xDA]
    limit = {name: getattr(exif_module, name) for name in JPEG_LIMITS}
    assert limit == dict(zip(JPEG_LIMITS, (1000, 65536, 65536, 65536, 1024 * 1024, 6 + 3 * 255)))

    def with_extra(extra, after=0):
        return hostile.jpeg_with_header(extra, good, after)

    seg = hostile.segment
    edges = {
        "segments": lambda n: with_extra(seg(0xFE) * (n - len(own))),
        "junk": lambda n: with_extra(b"\x00" * n, after=1),
        "fill bytes": lambda n: with_extra(b"\xff" * n),
        "stuffed zeros and bare markers": lambda n: with_extra(b"\xff\x00\xff\xd0" * (n // 2) + b"\xff\xd9" * (n % 2)),
        "EXIF in two segments": lambda n: with_extra(seg(0xE1, hostile.EXIF_HEADER + bytes(60000))
                                                     + seg(0xE1, hostile.EXIF_HEADER + bytes(n - 60012))),
        "tables": lambda n: with_extra(seg(0xDB, bytes(n - 130))),
        "resource blocks": lambda n: with_extra(seg(0xED, hostile.PHOTOSHOP + bytes(65000)) * 16
                                                + seg(0xED, hostile.PHOTOSHOP + bytes(n - 16 * 65014 - 14))),
        "frame headers": lambda n: with_extra(seg(0xDE, bytes(n - 15))),
    }
    which = {"segments": "_JPEG_MAX_SEGMENTS", "junk": "_JPEG_MAX_SKIPPED", "fill bytes": "_JPEG_MAX_SKIPPED",
             "stuffed zeros and bare markers": "_JPEG_MAX_SKIPPED", "EXIF in two segments": "MAX_JPEG_EXIF_BYTES",
             "tables": "_JPEG_MAX_TABLE_BYTES", "resource blocks": "_JPEG_MAX_RESOURCE_BYTES",
             "frame headers": "_JPEG_MAX_FRAME_BYTES"}
    for what, build in edges.items():
        n = limit[which[what]]
        assert not refused(build(n)), what
        assert refused(build(n + 1)), what
    # a segment whose length says 0 or 1 holds no bytes (Pillow reads none): the two length bytes are not junk
    lying = bytes([0xFF, 0xFE, 0, 0, 0xFF, 0xFE, 0, 1])
    at_limit = hostile.jpeg_with_header(lying, edges["junk"](limit["_JPEG_MAX_SKIPPED"]))
    assert not refused(at_limit) and read_photo_meta(at_limit).width == 640
    # what is not counted: other APP1 and APP13 payloads, a second scan's header, anything after the first scan
    assert not refused(with_extra(seg(0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + bytes(65000)) * 300))
    assert not refused(with_extra(seg(0xE1, b"Exif\x00x" + bytes(65000)) * 2))      # not Pillow's "Exif\0\0"
    assert not refused(with_extra(seg(0xE2, hostile.EXIF_HEADER + bytes(65000)) * 2))       # nor in APP1
    assert not refused(with_extra(seg(0xED, bytes(65000)) * 300))
    assert not refused(good + seg(0xFE) * 100_000 + b"\x00" * 100_000 + hostile.frame_header() * 3)
    # at the limits the file is read like the plain one (Pillow agrees it is a JPEG), except that a header with stray
    # bytes is not taken for one that decodes reduced (jpeg_decodes_reduced follows none: the pixel cap applies)
    plain = read_photo_meta(good)
    assert plain.decodes_reduced
    for what in ("segments", "junk", "fill bytes", "stuffed zeros and bare markers"):
        meta = read_photo_meta(edges[what](limit[which[what]]))
        assert dataclasses.replace(meta, decodes_reduced=True) == plain, what
        assert meta.decodes_reduced == (what == "segments"), what
    one_segment = hostile.with_exif("JPEG", hostile.exif_block([(piexif.ImageIFD.Make, 2, 5, 8)],
                                                               b"Acme\0".ljust(65533 - 6 - 8 - 18, b"\0")))
    assert [end - start for m, start, end in hostile._header_segments(one_segment) if m == 0xE1] == [65535 + 2]
    assert read_photo_meta(one_segment).make == "Acme"                  # the largest segment there is: read


@pytest.mark.filterwarnings("ignore:Metadata Warning:UserWarning")
def test_a_jpeg_whose_exif_holds_megabytes_of_numbers_is_refused_unparsed(monkeypatch):
    """The review's second finding. Pillow parses a JPEG's EXIF while it opens the file (for the resolution, every
    value of that tag), and it joins all "Exif" segments first: a 24.6 MB JPEG whose XResolution declared 3,000,000
    rationals took 14 s and 0.7 GB in read_photo_meta, before the EXIF size was looked at (the 4 MiB check came
    after the open), and one just under 4 MiB 1.2 s. A JPEG's EXIF is now measured before Pillow opens the file and
    held to exif.MAX_JPEG_EXIF_BYTES. The most that fits (8,000 rationals in one segment) is read in milliseconds."""
    assert exif_module.MAX_JPEG_EXIF_BYTES == 64 * 1024
    allowed = hostile.with_exif("JPEG", hostile.numbers_exif(65_400))
    assert [m for m, _, _ in hostile._header_segments(allowed)].count(0xE1) == 1
    meta, seconds, peak = cost_of(read_photo_meta, allowed)
    assert meta.has_exif and (meta.width, meta.height) == (640, 480)
    assert seconds < 0.5 and peak < 8 * MB, (seconds, peak)             # measured: 0.006 s (0.02 s traced), 1.4 MB
    pillow_must_not_open(monkeypatch)
    for nbytes in (24_000_000, 3_900_000, 70_000):
        data = hostile.with_exif("JPEG", hostile.numbers_exif(nbytes))
        result, seconds, peak = cost_of(read_photo_meta, data)
        assert isinstance(result, UnreadableImage) and "EXIF over 65536 bytes in a JPEG" in str(result), result
        assert seconds < 0.1 and peak < 64 * 1024, (nbytes, seconds, peak)


def test_png_and_webp_chunks_at_each_limit_are_read_and_one_over_is_not():
    """The PNG limits: 1,000 chunks that are not pixel data, 65,536 that are (IDAT, and an animation's fdAT), 16 that
    Pillow inflates (zTXt, iTXt, iCCP), counted to IEND, before and after the pixel data; EXIF over MAX_EXIF_BYTES in
    an eXIf chunk wherever it sits, or as a "Raw profile type exif" text chunk (two bytes of text for one of EXIF).
    A WebP: 1,000 chunks, and its EXIF chunk."""
    good = hostile.png()
    assert good[12:16] == b"IHDR" and good.count(b"IDAT") == 1 and good[37:41] == b"IDAT"       # IHDR, then the pixels
    chunk = hostile.png_chunk

    def texts(n):
        return b"".join(chunk(b"tEXt", b"k%d\0v" % i) for i in range(n))

    assert (exif_module._PNG_MAX_CHUNKS, exif_module._PNG_MAX_DATA_CHUNKS, exif_module._PNG_MAX_INFLATED_CHUNKS,
            exif_module._WEBP_MAX_CHUNKS) == (1000, 65536, 16, 1000)
    plain = read_photo_meta(good)
    for before in (True, False):                # before the pixel data, and after it
        def put(extra):
            return hostile.png_with_chunks(**{"before" if before else "after": extra}, data=good)

        assert read_photo_meta(put(texts(999))) == plain and refused(put(texts(1000)))      # (IHDR is the first)
        assert refused(put(chunk(b"prVt") * 1000)) and refused(put(chunk(b"fcTL", bytes(26)) * 1000))
        assert read_photo_meta(put(chunk(b"IDAT") * 65535)) == plain and refused(put(chunk(b"IDAT") * 65536))
        assert not refused(put(chunk(b"fdAT", bytes(4)) * 65535)) and refused(put(chunk(b"fdAT", bytes(4)) * 65536))
        for kind in (b"zTXt", b"iTXt", b"iCCP"):
            assert not refused(put(chunk(kind, b"k\0\0\0\0\0") * 16)) and refused(put(chunk(kind, b"k\0\0\0\0\0") * 17))
        assert not refused(put(chunk(b"tEXt", b"k\0v") * 17))                # plain text is not inflated
        most = exif_module.MAX_EXIF_BYTES - 6
        assert not refused(put(chunk(b"eXIf", bytes(most)))) and refused(put(chunk(b"eXIf", bytes(most + 1))))
        raw = b"Raw profile type exif\0"
        for kind in (b"tEXt", b"zTXt", b"iTXt"):
            assert not refused(put(chunk(kind, raw.ljust(2 * exif_module.MAX_EXIF_BYTES, b"0"))))
            assert refused(put(chunk(kind, raw.ljust(2 * exif_module.MAX_EXIF_BYTES + 1, b"0"))))
        assert not refused(put(chunk(b"tEXt", b"Comment\0".ljust(3 * exif_module.MAX_EXIF_BYTES, b"0"))))
    assert not refused(good + texts(3000))                                   # nothing after IEND counts
    # an EXIF chunk after the pixel data is not this module's EXIF (Pillow reads it only once it has decoded)
    after = hostile.png_with_chunks(after=chunk(b"eXIf", hostile.exif_with(lat=31.25, lon=121.5)[6:]), data=good)
    assert read_photo_meta(after) == plain and not plain.has_exif

    webp = hostile.webp_with_chunks(b"")
    own, at = 0, 12
    while at + 8 <= len(webp):
        size = int.from_bytes(webp[at + 4:at + 8], "little")
        own, at = own + 1, at + 8 + size + size % 2
    assert 3 <= own <= 5 and at == len(webp)
    junk = b"JUNK\x01\x00\x00\x00x\x00"                                       # one byte and its padding
    assert read_photo_meta(hostile.webp_with_chunks(junk * (1000 - own))) == read_photo_meta(webp)
    assert refused(hostile.webp_with_chunks(junk * (1001 - own)))
    assert refused(webp[:12] + b"EXIF" + (exif_module.MAX_EXIF_BYTES + 1).to_bytes(4, "little"))
    assert not refused(webp[:12] + b"EXIF" + exif_module.MAX_EXIF_BYTES.to_bytes(4, "little"))


@pytest.mark.filterwarnings("ignore:Metadata Warning:UserWarning")
def test_what_the_limits_let_through_costs_milliseconds():
    """The most a header may hold, all of it at once, read by Pillow in read_photo_meta: a JPEG of 990 empty
    segments, 64 KiB of junk, 64 KiB of EXIF numbers, 64 KiB of quantisation tables and 1 MiB of Photoshop resource
    blocks; a PNG of 990 text chunks, 65,000 empty pixel data chunks and 16 compressed text chunks of 1 MiB. (A
    file's bytes held in segments or chunks are not limited: Pillow keeps them as they are, once.)"""
    seg = hostile.segment
    blocks = seg(0xED, hostile.PHOTOSHOP + hostile.RESOURCE_BLOCK * 5459)
    header = (seg(0xE5) * 970 + seg(0xDB, hostile.QUANT_TABLE * 1000) + blocks * 15)
    data = hostile.jpeg_with_header(header, hostile.with_exif("JPEG", hostile.numbers_exif(65_400)))
    data = hostile.jpeg_with_header(b"\x00" * 65_000, data, after=1)

    def seconds_of(data):                       # untraced: tracing the allocations takes many times as long
        start = time.perf_counter()
        read_photo_meta(data)
        return time.perf_counter() - start

    assert not refused(data) and min(seconds_of(data) for _ in range(3)) < 0.5      # measured: 0.045 s
    meta, _, peak = cost_of(read_photo_meta, data)
    assert meta.has_exif and (meta.width, meta.height, meta.format) == (640, 480, "JPEG")
    assert peak < 8 * MB, peak                                                      # measured: 2.4 MB
    chunk = hostile.png_chunk
    inflating = b"".join(chunk(b"zTXt", b"k%d\0\0" % i + zlib.compress(b"A" * 1_048_000, 9)) for i in range(16))
    text = b"".join(chunk(b"tEXt", b"k%d\0v" % i) for i in range(970))
    data = hostile.png_with_chunks(text + inflating + chunk(b"IDAT") * 65_000)
    assert not refused(data) and min(seconds_of(data) for _ in range(3)) < 0.5      # measured: 0.09 s
    meta, _, peak = cost_of(read_photo_meta, data)
    assert (meta.width, meta.height, meta.format) == (640, 480, "PNG")
    assert peak < 40 * MB, peak                                                     # measured: 20 MB (16 MiB of text)


def test_the_metadata_is_read_with_one_open_of_the_image(monkeypatch):
    """read_photo_meta opened the image twice (once to verify it, once to read it) and held both at once: twice
    Pillow's header objects for every upload being read. One open: the attributes are read, then it is verified."""
    opens = []
    real_open = Image.open

    def counted(*args, **kwargs):
        opens.append(kwargs.get("formats"))
        return real_open(*args, **kwargs)

    exif = hostile.exif_with(capture_time=b"2024:05:06 07:08:09", lat=31.25, lon=121.5)
    files = {fmt: hostile.with_exif(fmt, exif) for fmt in ("JPEG", "PNG", "WEBP")}
    monkeypatch.setattr(exif_module.Image, "open", counted)
    for fmt, data in files.items():
        del opens[:]
        meta = read_photo_meta(data)
        assert opens == [exif_module.ALLOWED_FORMATS], (fmt, opens)
        assert (meta.format, meta.has_gps, meta.captured_at) == (fmt, True, "2024:05:06 07:08:09")
    # and it is still verified: a PNG whose pixel data fails its checksum is unreadable
    png = bytearray(hostile.png())
    png[png.index(b"IDAT") + 10] ^= 0xFF
    with pytest.raises(UnreadableImage):
        read_photo_meta(bytes(png))


def oriented(fmt, orientation=None, xmp=None, size=(64, 48)) -> bytes:
    """A small image with a dark corner (so every turn shows), carrying an orientation in its EXIF, or XMP."""
    img = hostile.textured(size, variant=1)
    save = {}
    if orientation is not None:
        save["exif"] = hostile.exif_block([(0x0112, hostile.TIFF_SHORT, 1, struct.pack(">HH", orientation, 0))])
    if xmp is not None and fmt == "PNG":
        from PIL import PngImagePlugin

        save["pnginfo"] = PngImagePlugin.PngInfo()
        save["pnginfo"].add_itxt("XML:com.adobe.xmp", xmp.decode())
    elif xmp is not None:
        save["xmp"] = xmp
    buf = io.BytesIO()
    img.save(buf, fmt, **save, **({"lossless": True} if fmt == "WEBP" else {}))
    return buf.getvalue()


@pytest.mark.filterwarnings("ignore:Metadata Warning:UserWarning")
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_upright_gives_the_pixels_pillows_exif_transpose_gives(fmt, monkeypatch):
    """exif.upright turns an image by its orientation exactly as ImageOps.exif_transpose does (every orientation,
    none, a value that is none, one that sits only in the XMP, a JPEG decoded at reduced size), so no photo's
    analysis copy or processed copy changed when the decodes took it instead. What it leaves out is Pillow writing
    the EXIF block out again, which parsed every value in it: with 3.9 MB of numbers and an orientation in a PNG or
    WebP that was 0.65 s and 100 MB for each decode of the upload; now the block is not parsed at all."""
    from PIL import ImageChops, ImageOps

    def same(data, draft=None):
        with Image.open(io.BytesIO(data)) as a, Image.open(io.BytesIO(data)) as b:
            if draft:
                a.draft("RGB", draft)
                b.draft("RGB", draft)
            ours, theirs = exif_module.upright(a).convert("RGB"), ImageOps.exif_transpose(b).convert("RGB")
            assert ours.size == theirs.size and ImageChops.difference(ours, theirs).getbbox() is None
            return ours.size

    sizes = {o: same(oriented(fmt, o)) for o in (None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 65535)}
    assert {o for o, size in sizes.items() if size == (48, 64)} == {5, 6, 7, 8}          # the ones that turn it
    turned = []
    for o in range(2, 9):                                                                # and each turns it its way
        with Image.open(io.BytesIO(oriented(fmt, o))) as im:
            turned.append(exif_module.upright(im).convert("L").resize((4, 4)).tobytes())
    assert len(set(turned)) == 7
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description tiff:Orientation="6"/></x:xmpmeta>'
    assert same(oriented(fmt, None, xmp)) == (48, 64)                # no EXIF orientation: the XMP's counts
    assert same(oriented(fmt, 1, xmp)) == (64, 48)                   # the EXIF's comes first
    if fmt == "JPEG":
        assert same(oriented(fmt, 6, size=(640, 480)), draft=(160, 120)) == (120, 160)       # decoded at 1/4 scale
    # the block's values are not parsed: megabytes of numbers and an orientation cost a decode nothing
    big = hostile.with_exif(fmt, hostile.numbers_exif(60_000 if fmt == "JPEG" else 3_900_000, orientation=6))
    monkeypatch.setattr(Image.Exif, "tobytes", lambda self, offset=8: pytest.fail("the EXIF block was written out"))
    with Image.open(io.BytesIO(big)) as im:
        start = time.perf_counter()
        assert exif_module.upright(im).size == (480, 640)
        assert time.perf_counter() - start < 0.3                     # measured: 0.002 s (it was 0.67 s)
