"""Builders of crafted uploads for the ingest and API tests (D34): EXIF with an over-long or wrong-typed text tag in
each accepted format, EXIF blocks built entry by entry (huge declared counts, many tags at one value), JPEGs that
cannot be decoded at reduced size, and headers that hold far more than a camera's (segments, chunks: header_floods)."""

from __future__ import annotations

import io
import struct
import zlib

import piexif
from PIL import Image, ImageDraw

from crowd3d.exif import build_exif_bytes

EXIF_HEADER = b"Exif\x00\x00"
APP1_PAYLOAD = 65533 - len(EXIF_HEADER)     # what one JPEG segment holds after its 2-byte length and the EXIF header
TIFF_BYTE, TIFF_ASCII, TIFF_SHORT, TIFF_LONG, TIFF_RATIONAL, TIFF_UNDEFINED = 1, 2, 3, 4, 5, 7


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


def directory(entries, endian: str = ">", following: int = 0) -> bytes:
    """An image file directory: entries of (tag, type, count, value), value the entry's 4-byte field as bytes or a
    number written as a LONG (an offset into the block, or the value itself), then the offset of the next one."""
    out = struct.pack(endian + "H", len(entries))
    for tag, kind, count, value in entries:
        field = value if isinstance(value, bytes) else struct.pack(endian + "L", value)
        assert len(field) == 4
        out += struct.pack(endian + "HHL", tag, kind, count) + field
    return out + struct.pack(endian + "L", following)


def exif_block(entries, filler: bytes = b"", endian: str = ">") -> bytes:
    """An EXIF blob (with its "Exif\0\0" header): the TIFF header, filler at offset 8, then the first directory with
    these entries (directory). An entry may declare any type and count and point anywhere, at the filler too."""
    order = b"MM\x00*" if endian == ">" else b"II*\x00"
    return EXIF_HEADER + order + struct.pack(endian + "L", 8 + len(filler)) + filler + directory(entries, endian)


def webp_with_chunk(chunk: bytes) -> bytes:
    """A lossless WebP whose EXIF chunk is exactly these bytes, whatever they are (Pillow's writer wants a TIFF
    block; the chunk of a valid file is swapped)."""
    data = with_exif("WEBP", exif_block([]))
    at = data.index(b"EXIF")
    old = struct.unpack("<I", data[at + 4:at + 8])[0]
    rest = data[at + 8 + old + old % 2:]
    out = data[:at] + b"EXIF" + struct.pack("<I", len(chunk)) + chunk + b"\x00" * (len(chunk) % 2) + rest
    return out[:4] + struct.pack("<I", len(out) - 8) + out[8:]


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


# Headers that hold more than a camera's (D34, review round 2)


def segment(marker: int, payload: bytes = b"") -> bytes:
    """One JPEG segment: 0xFF, the marker, the 2-byte length (itself included), the payload."""
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def jpeg_with_header(extra: bytes, data: bytes | None = None, after: int = 0) -> bytes:
    """A JPEG (default: jpeg()) with extra put in after its first `after` header segments (0: right after SOI)."""
    data = jpeg() if data is None else data
    at = _header_segments(data)[after - 1][2] if after else 2
    return data[:at] + extra + data[at:]


def png_chunk(kind: bytes, data: bytes = b"") -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(img: Image.Image | None = None, **save) -> bytes:
    buf = io.BytesIO()
    (img or textured()).save(buf, "PNG", **save)
    return buf.getvalue()


def png_with_chunks(before: bytes = b"", after: bytes = b"", data: bytes | None = None) -> bytes:
    """A PNG (default: png()) with chunks put in before its pixel data (right after IHDR) and after it (before IEND)."""
    data = png() if data is None else data
    assert data[12:16] == b"IHDR" and data[-8:-4] == b"IEND"
    return data[:33] + before + data[33:-12] + after + data[-12:]


def webp_with_chunks(extra: bytes) -> bytes:
    """A WebP in the extended container (a lossless one with an EXIF chunk) with these chunks appended."""
    out = with_exif("WEBP", exif_block([])) + extra
    return out[:4] + struct.pack("<I", len(out) - 8) + out[8:]


def numbers_exif(nbytes: int, orientation: int | None = None) -> bytes:
    """An EXIF blob (with its header) whose XResolution tag declares nbytes // 8 rationals, all of them in the block
    (72/1 each): the tag Pillow reads, value by value, while it opens a JPEG. With orientation, that tag too, which
    makes Pillow write the whole block out again when it turns the pixels upright."""
    filler = (struct.pack(">LL", 72, 1) * (nbytes // 8 + 1))[:nbytes]
    entries = [(0x011A, TIFF_RATIONAL, nbytes // 8, 8), (0x0128, TIFF_SHORT, 1, struct.pack(">HH", 2, 0))]
    if orientation:
        entries.insert(0, (0x0112, TIFF_SHORT, 1, struct.pack(">HH", orientation, 0)))
    return exif_block(entries, filler)


def frame_header(marker: int = 0xC0, size: int = 65533) -> bytes:
    """A frame header segment of size payload bytes: 8 bits, 480 x 640, 3 components, then as many more component
    triples as fit (Pillow keeps a tuple for each)."""
    body = bytes([8]) + (480).to_bytes(2, "big") + (640).to_bytes(2, "big") + bytes([3])
    return segment(marker, body + bytes([1, 0x11, 0]) * ((size - len(body)) // 3))


QUANT_TABLE = bytes([0]) + bytes(range(1, 65))                              # one 8-bit quantisation table
RESOURCE_BLOCK = b"8BIM" + (0x0404).to_bytes(2, "big") + bytes(6)           # an empty Photoshop resource block
PHOTOSHOP = b"Photoshop 3.0\x00"


def header_floods(megabytes: int = 24) -> dict[str, callable]:
    """name -> builder of an upload of about that many MB (under the 25 MiB limit by default) whose header holds far
    more than this project lets Pillow see (exif._check_header). What each cost to read at 24 MB before the limits,
    measured in a fresh process (seconds, peak memory over the file's own): docs/decisions/D34-review-fixes.md."""
    size = megabytes * 1_000_000
    fat = size // 65_537                                                    # segments of the largest size that fit
    return {
        # JPEG: before the first scan
        "jpeg: empty APP segments": lambda: jpeg_with_header(segment(0xE5) * (size // 4)),
        "jpeg: empty comments": lambda: jpeg_with_header(segment(0xFE) * (size // 4)),
        "jpeg: fat frame headers": lambda: jpeg_with_header(frame_header() * fat),
        "jpeg: fat hierarchical frame headers": lambda: jpeg_with_header(frame_header(0xDE) * fat),
        "jpeg: colour profile segments": lambda: jpeg_with_header(
            segment(0xE2, b"ICC_PROFILE\0\x01\x01") * (size // 18)),
        "jpeg: small EXIF segments": lambda: jpeg_with_header(segment(0xE1, EXIF_HEADER + b"x") * (size // 11)),
        "jpeg: EXIF with megabytes of numbers": lambda: with_exif("JPEG", numbers_exif(size)),
        "jpeg: quantisation tables": lambda: jpeg_with_header(segment(0xDB, QUANT_TABLE * 1008) * fat),
        "jpeg: Photoshop resource blocks": lambda: jpeg_with_header(
            segment(0xED, PHOTOSHOP + RESOURCE_BLOCK * 5459) * fat),
        "jpeg: junk between segments": lambda: jpeg_with_header(b"\x55" * size, after=1),
        "jpeg: fill bytes": lambda: jpeg_with_header(b"\xff" * size),
        "jpeg: stuffed zeros": lambda: jpeg_with_header(b"\xff\x00" * (size // 2)),
        "jpeg: restart markers": lambda: jpeg_with_header(b"\xff\xd0" * (size // 2)),
        "jpeg: empty segments and no scan": lambda: b"\xff\xd8" + segment(0xE5) * (size // 4),
        # PNG: before and after the pixel data
        "png: text chunks": lambda: png_with_chunks(b"".join(png_chunk(b"tEXt", b"%x\0" % i) for i in range(50_000))),
        "png: private chunks": lambda: png_with_chunks(png_chunk(b"prVt") * (size // 12)),
        "png: private chunks after the pixels": lambda: png_with_chunks(after=png_chunk(b"prVt") * (size // 12)),
        "png: empty pixel data chunks": lambda: png_with_chunks(png_chunk(b"IDAT") * (size // 12)),
        "png: compressed text of 64 MiB": lambda: png_with_chunks(b"".join(
            png_chunk(b"zTXt", b"k%d\0\0" % i + zlib.compress(b"A" * 1_048_000, 9)) for i in range(64))),
        "png: colour profiles of 1 MiB": lambda: png_with_chunks(
            png_chunk(b"iCCP", b"p\0\0" + zlib.compress(bytes(1_048_000), 9)) * 990),
        "png: EXIF after the pixels": lambda: png_with_chunks(
            after=png_chunk(b"eXIf", numbers_exif(size, orientation=6)[len(EXIF_HEADER):])),
        "png: EXIF as a raw profile": lambda: png_with_chunks(png_chunk(
            b"tEXt", b"Raw profile type exif\0\nexif\n%8d\n" % (size // 2)
            + numbers_exif(size // 2, orientation=6)[len(EXIF_HEADER):].hex().encode())),
        # WebP
        "webp: empty chunks": lambda: webp_with_chunks(b"JUNK\0\0\0\0" * (size // 8)),
    }
