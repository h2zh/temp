import io
import math
import random
import struct
import time
import tracemalloc

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
    """(result or the exception raised, seconds, peak bytes of Python allocations) of one call."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = call(*args)
    except Exception as exc:
        result = exc
    seconds = time.perf_counter() - start
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
    filler = b"M" * (2 * MB)
    # what reading the file costs with no tag at all in the same 2 MB block (Pillow reads the chunk, or joins the
    # segments of a JPEG)
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


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_an_exif_block_over_4_mib_makes_the_image_unreadable(fmt):
    """Pillow's own EXIF parser runs on the block when the pixels are turned upright (about 0.2 s and 25 MB per MiB
    of numbers, measured), so a block over exif.MAX_EXIF_BYTES is refused before anything parses it: the image is
    unreadable. A camera's block is at most 64 KiB."""
    assert exif_module.MAX_EXIF_BYTES == 4 * MB
    make = (piexif.ImageIFD.Make, 2, 5, 8)
    header = 0 if fmt == "WEBP" else 6          # Pillow hands out a WebP's chunk as it is, the others after "Exif\0\0"
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
