"""Builders of crafted uploads for the ingest and API tests (D34): EXIF with an over-long or wrong-typed text tag in
each accepted format, and JPEGs that cannot be decoded at reduced size."""

from __future__ import annotations

import io

import piexif
from PIL import Image, ImageDraw

from crowd3d.exif import build_exif_bytes

EXIF_HEADER = b"Exif\x00\x00"
APP1_PAYLOAD = 65533 - len(EXIF_HEADER)     # what one JPEG segment holds after its 2-byte length and the EXIF header
TIFF_ASCII, TIFF_SHORT = 2, 3


def textured(size=(640, 480), mode="RGB", variant: int = 0) -> Image.Image:
    """An image sharp enough to pass the blur check (a grid of lines); each variant has a dark block in another
    place, so two variants are not duplicates of each other."""
    img = Image.new(mode, size, "white")
    d = ImageDraw.Draw(img)
    ink = "black" if mode != "L" else 0
    for x in range(0, size[0], 16):
        d.line([(x, 0), (x, size[1])], fill=ink)
    for y in range(0, size[1], 16):
        d.line([(0, y), (size[0], y)], fill=ink)
    if variant:
        w, h = size[0] // 3, size[1] // 3
        x, y = (variant % 3) * w, (variant // 3 % 3) * h
        d.rectangle([x, y, x + w, y + h], fill=ink)
    return img


def exif_with(*, capture_time: bytes | None = None, lat: float | None = None, lon: float | None = None,
              make: bytes = b"Crowd3D") -> bytes:
    """An EXIF blob (with its "Exif\\0\\0" header) holding a GPS fix, a camera make and DateTimeOriginal exactly as
    given: any bytes, any length (piexif does not check either)."""
    ex = piexif.load(build_exif_bytes(lat=lat, lon=lon, focal_35mm=28.0, make=None, model=None))
    ex["0th"][piexif.ImageIFD.Make] = make
    if capture_time is not None:
        ex["Exif"][piexif.ExifIFD.DateTimeOriginal] = capture_time
    return piexif.dump(ex)


def retyped(exif: bytes, tag: int, count: int = 1) -> bytes:
    """exif with the IFD entry of the text tag declared as count SHORT numbers instead of ASCII text (its value field
    untouched): piexif then returns an int (count 1) or a tuple of ints (count 2), not bytes."""
    assert exif.startswith(EXIF_HEADER) and exif[6:8] == b"MM"         # piexif writes big-endian TIFF
    entry = tag.to_bytes(2, "big") + TIFF_ASCII.to_bytes(2, "big")
    assert exif.count(entry) == 1 and count in (1, 2), exif.count(entry)
    at = exif.index(entry)
    return exif[:at] + tag.to_bytes(2, "big") + TIFF_SHORT.to_bytes(2, "big") + count.to_bytes(4, "big") + exif[at + 8:]


def with_exif(fmt: str, exif: bytes, img: Image.Image | None = None) -> bytes:
    """img (default: textured) saved as fmt ("JPEG", "PNG" or "WEBP") carrying exif, whatever its size: a PNG's and a
    WebP's EXIF chunk has no limit, and a JPEG takes it in as many APP1 segments as it needs (Pillow joins them when
    it reads the file; its own writer refuses more than one segment)."""
    img = img or textured()
    buf = io.BytesIO()
    if fmt != "JPEG":
        img.save(buf, fmt, exif=exif, **({"lossless": True} if fmt == "WEBP" else {}))
        return buf.getvalue()
    img.save(buf, "JPEG", quality=92)
    data = buf.getvalue()
    assert data[:2] == b"\xff\xd8" and exif.startswith(EXIF_HEADER)
    tiff = exif[len(EXIF_HEADER):]
    segments = b""
    for start in range(0, len(tiff), APP1_PAYLOAD):
        payload = EXIF_HEADER + tiff[start:start + APP1_PAYLOAD]
        segments += b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    return data[:2] + segments + data[2:]


def _header_segments(data: bytes) -> list[tuple[int, int, int]]:
    """[(marker, start, end)] of a JPEG's segments from after SOI up to and including the first scan's header."""
    out, pos = [], 2
    while True:
        assert data[pos] == 0xFF, hex(data[pos])
        marker = data[pos + 1]
        end = pos + 2 + int.from_bytes(data[pos + 2:pos + 4], "big")
        out.append((marker, pos, end))
        pos = end
        if marker == 0xDA:
            return out


def jpeg(size=(640, 480), mode="RGB", **save) -> bytes:
    buf = io.BytesIO()
    textured(size, mode).save(buf, "JPEG", **save)
    return buf.getvalue()


def non_interleaved_jpeg(size=(640, 480), levels=(120, 128, 136)) -> bytes:
    """A sequential (SOF0) three-component 4:4:4 JPEG that sends each component in a scan of its own, spliced from
    three flat grayscale JPEGs of that size (the scan of a grayscale JPEG is a valid one-component scan, and the three
    share the standard tables). Pillow opens it as a plain JPEG with no "progressive" in its info; libjpeg decodes it
    like a progressive one, with every coefficient of the full image in memory."""
    grays = []
    for level in levels:
        buf = io.BytesIO()
        Image.new("L", size, level).save(buf, "JPEG", quality=75)
        grays.append(buf.getvalue())
    first = grays[0]
    out = bytearray(b"\xff\xd8")
    for marker, start, end in _header_segments(first):
        if marker == 0xDA:
            break
        seg = first[start:end]
        if marker == 0xC0:      # the frame header: one component -> three, each 1x1 with quantisation table 0
            body = seg[4:9] + bytes([3]) + b"".join(bytes([cid, 0x11, 0]) for cid in (1, 2, 3))
            seg = b"\xff\xc0" + (len(body) + 2).to_bytes(2, "big") + body
        out += seg
    for cid, gray in zip((1, 2, 3), grays):
        _, start, end = _header_segments(gray)[-1]
        sos = bytearray(gray[start:end])
        assert sos[4] == 1 and gray[-2:] == b"\xff\xd9"
        sos[5] = cid            # the scan's component
        out += sos + gray[end:-2]
    return bytes(out + b"\xff\xd9")
