"""Read and write the EXIF fields the pipeline relies on.

Reading never raises on malformed EXIF: a missing or broken tag just becomes
None. Only an undecodable image raises UnreadableImage; one whose header holds
more than a camera's (_check_header: segments, chunks, EXIF bytes) counts as
one, and is refused before Pillow opens it. The EXIF block is read by this
module's own bounded reader (_read_exif); piexif only writes (build_exif_bytes).
"""

from __future__ import annotations

import io
import re
import struct
from dataclasses import dataclass
from fractions import Fraction

import piexif
from PIL import (  # noqa: F401 (registers formats)
    ExifTags,
    Image,
    JpegImagePlugin,
    PngImagePlugin,
    UnidentifiedImageError,
    WebPImagePlugin,
)

# Formats the pipeline decodes. Pillow would otherwise accept anything it knows, including EPS,
# which it renders by running Ghostscript on the uploaded PostScript.
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
# MPO: phone JPEGs with extra frames, which Pillow opens through its JPEG plugin. Not every JPEG is cheap to decode:
# only a sequential one with all its components in one scan decodes at 1/2..1/8 scale (draft) without buffers for
# the full image (PhotoMeta.decodes_reduced, jpeg_decodes_reduced).
JPEG_FORMATS = ("JPEG", "MPO")
# An EXIF text tag (camera make and model, the capture time, the heading's reference) is read up to this many
# characters: real ones are far shorter, and a crafted one can be megabytes (a PNG or WebP EXIF chunk has no size
# limit, and Pillow joins a JPEG's EXIF segments).
MAX_TEXT_CHARS = 64
_MAX_TEXT_BYTES = 1024      # of a text tag's bytes, what is decoded at all (leading padding and multi-byte characters)
_CAPTURE_TIME = re.compile(r"[0-9]{4}:[0-9]{2}:[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")    # ASCII digits only
# The EXIF block of a PNG or WebP upload (D34): a chunk, which can be as big as the file. Over this the image counts
# as unreadable. This module's reader does not care about the block's size, and Pillow opens these two formats
# without parsing it, but Pillow reads the block's directory when the pixels are turned upright (upright, inside
# ingest's decode slot). A PNG's is measured before Pillow opens the file, wherever it sits (_check_png_chunks), a
# WebP's too (_check_webp_chunks).
MAX_EXIF_BYTES = 4 * 1024 * 1024
# The EXIF of a JPEG upload, all its "Exif" APP1 segments together (D34, review round 2): what one segment holds, the
# EXIF standard's own limit and what cameras write. Pillow joins further segments, and it parses a JPEG's EXIF while
# it opens the file (for the resolution tag, every value of it): a 24.6 MB JPEG of such segments took 14 s and
# 0.7 GB to open, one just under MAX_EXIF_BYTES 2.6 s per upload. So a JPEG's is measured before Pillow opens the
# file, and over this the image counts as unreadable (_check_jpeg_header).
MAX_JPEG_EXIF_BYTES = 64 * 1024
# _read_exif: an image file directory with more entries than this is not a camera's (theirs hold dozens) ...
_MAX_IFD_ENTRIES = 1024
# ... and of a tag's numbers at most this many are read: every tag read here holds one to three
_MAX_NUMBERS = 8
_TOO_MANY = ()                      # what stands for a tag of more numbers: as unusable as the long tuple it replaces
_TIFF_ASCII, _TIFF_UNDEFINED = 2, 7
_TIFF_RATIONALS = {5: "L", 10: "l"}             # RATIONAL, SRATIONAL: pairs of 32-bit numbers, always behind an offset
# The other number types: struct code, bytes per number, and the largest count whose numbers sit in the entry itself
# (BYTE, SBYTE, SHORT, SSHORT, LONG, SLONG, FLOAT; DOUBLE is always behind an offset)
_TIFF_NUMBERS = {1: ("B", 1, 4), 6: ("b", 1, 4), 3: ("H", 2, 2), 8: ("h", 2, 2), 4: ("L", 4, 1), 9: ("l", 4, 1),
                 11: ("f", 4, 1), 12: ("d", 8, -1)}
_THUMBNAIL = (piexif.ImageIFD.JPEGInterchangeFormat, piexif.ImageIFD.JPEGInterchangeFormatLength)
# The tags _read_exif returns, per directory: what read_photo_meta uses, and the pointers to the other directories
_READ_TAGS = {
    "0th": frozenset({piexif.ImageIFD.Make, piexif.ImageIFD.Model, piexif.ImageIFD.Orientation,
                      piexif.ImageIFD.ExifTag, piexif.ImageIFD.GPSTag}),
    "Exif": frozenset({piexif.ExifIFD.DateTimeOriginal, piexif.ExifIFD.FocalLength,
                       piexif.ExifIFD.FocalLengthIn35mmFilm, piexif.ExifIFD.InteroperabilityTag}),
    "GPS": frozenset({piexif.GPSIFD.GPSLatitudeRef, piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLongitudeRef,
                      piexif.GPSIFD.GPSLongitude, piexif.GPSIFD.GPSAltitudeRef, piexif.GPSIFD.GPSAltitude,
                      piexif.GPSIFD.GPSImgDirectionRef, piexif.GPSIFD.GPSImgDirection,
                      piexif.GPSIFD.GPSHPositioningError}),
    "Interop": frozenset(),
    "1st": frozenset(_THUMBNAIL),
}
# The tags whose entries must be well-formed for the block to count as readable, per directory: the ones piexif knows
# (a table only; nothing of piexif parses an upload)
_KNOWN_TAGS = {name: frozenset(piexif.TAGS[name]) for name in _READ_TAGS}
# JPEG segments a decoder skips by their length on the way to the first scan (APPn, COM, DQT, DHT, DRI, DAC, DNL)
_JPEG_SKIPPED = frozenset({*range(0xE0, 0xF0), 0xFE, 0xDB, 0xC4, 0xDD, 0xCC, 0xDC})
_JPEG_SEQUENTIAL = (0xC0, 0xC1)     # SOF0, SOF1: sequential DCT, Huffman coding
_JPEG_SOS = 0xDA
_JPEG_MAX_STEPS = 1000              # segments and fill bytes walked before the first scan (a camera's JPEG: dozens)

# What an upload's header may hold (D34, review round 2; _check_header). Pillow reads a header in Python and keeps an
# object or more for everything it finds, so a header can cost far more than its bytes: a 24 MB JPEG of empty
# segments took 1.4 GB and 9 s to open, one of fat frame headers 1.4 GB, a PNG of empty chunks 0.5 GB. Over one of
# these the file is unreadable, before Pillow sees it. A camera's file is nowhere near any of them.
# A JPEG up to its first scan, as Pillow walks it (JpegImageFile._open):
_JPEG_MAX_SEGMENTS = 1000           # segments (extended XMP, an ICC profile, a depth map: at most a few hundred)
_JPEG_MAX_SKIPPED = 64 * 1024       # bytes passed over one by one: junk between segments, fill bytes, bare markers
_JPEG_MAX_TABLE_BYTES = 64 * 1024   # quantisation tables (four of them take 0.5 KiB; Pillow unpacks each one)
_JPEG_MAX_RESOURCE_BYTES = 1024 * 1024  # Photoshop resource blocks in APP13 (Pillow walks them block by block)
_JPEG_MAX_FRAME_BYTES = 6 + 3 * 255     # frame headers: the largest there is, 255 components (a tuple kept for each)
_JPEG_BARE = frozenset({0xC8, *range(0xD0, 0xDA), *range(0xF0, 0xFE)})      # no length: JPG, RSTn, SOI, EOI, JPGn
_JPEG_SEGMENTS = frozenset(range(0xC0, 0xFF)) - _JPEG_BARE                  # a 2-byte length and that many bytes
_JPEG_FRAMES = frozenset({*range(0xC0, 0xD0), 0xDE}) - {0xC4, 0xC8, 0xCC}   # SOFn and DHP: read as a frame header
_JPEG_DQT, _JPEG_APP1, _JPEG_APP13 = 0xDB, 0xE1, 0xED
_EXIF_HEADER = b"Exif\x00\x00"
_PHOTOSHOP_HEADER = b"Photoshop 3.0\x00"
# A PNG up to IEND: chunks that are not pixel data (a photo's: a dozen), and pixel data chunks (8 KiB each is usual:
# 3,200 in 25 MiB; an encoder that flushes every row writes one per row)
_PNG_MAX_CHUNKS = 1000
_PNG_MAX_DATA_CHUNKS = 65_536
# ... and the chunks Pillow inflates (compressed text, international text, the colour profile; a photo's: two or
# three): each to as much as 1 MiB, Pillow's own limit per chunk, so 64 of them in a 69 KB file took 64 MiB
_PNG_MAX_INFLATED_CHUNKS = 16
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_DATA_CHUNKS = (b"IDAT", b"fdAT")
_PNG_TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt")
_PNG_INFLATED_CHUNKS = (b"zTXt", b"iTXt", b"iCCP")
_PNG_RAW_EXIF = b"Raw profile type exif\x00"    # the text chunk Pillow also reads EXIF from, as hexadecimal digits
_WEBP_MAX_CHUNKS = 1000             # a WebP's chunks (a photo's: five; an animation's: one per frame)
# EXIF orientation -> the transposition that turns the pixels upright (ImageOps.exif_transpose's own table)
_UPRIGHT = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90}


class UnreadableImage(ValueError):
    """The bytes are not an image Pillow can decode."""


@dataclass(frozen=True)
class PhotoMeta:
    width: int
    height: int
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    gps_accuracy_m: float | None = None  # GPSHPositioningError: horizontal error the phone reported
    heading_deg: float | None = None     # GPSImgDirection: where the camera was pointing
    heading_ref: str | None = None       # "T" (true north) or "M" (magnetic north)
    focal_mm: float | None = None
    focal_35mm: float | None = None
    captured_at: str | None = None       # EXIF DateTimeOriginal as written (capture_time checks its form)
    make: str | None = None
    model: str | None = None
    orientation: int | None = None
    format: str | None = None            # Pillow format name, one of ALLOWED_FORMATS
    has_exif: bool = False               # a readable EXIF block was present at all
    decodes_reduced: bool = False        # a JPEG that decodes at 1/2..1/8 scale cheaply (jpeg_decodes_reduced)

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None


def _rational_to_float(r) -> float | None:
    try:
        num, den = r
        return None if den == 0 else num / den
    except (TypeError, ValueError):
        return None


def _to_rational(value: float, max_den: int = 1_000_000) -> tuple[int, int]:
    f = Fraction(value).limit_denominator(max_den)
    return f.numerator, f.denominator


def _dms_to_deg(dms, ref) -> float | None:
    try:
        d, m, s = (_rational_to_float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    if d is None or m is None or s is None:
        return None
    deg = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if ref in ("S", "W"):
        deg = -deg
    return deg


def _deg_to_dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    value = abs(value)
    d = int(value)
    m_full = (value - d) * 60.0
    m = int(m_full)
    s = (m_full - m) * 60.0
    return (d, 1), (m, 1), (int(round(s * 10000)), 10000)


def _text(v) -> str | None:
    """An EXIF text tag: stripped, cut at MAX_TEXT_CHARS characters; None when missing, empty or not text. A tag
    arrives as the type its entry declares (_tiff_value), so a text tag written as SHORT or RATIONAL is an int or a
    tuple: a broken tag, not an error."""
    if isinstance(v, bytes):
        v = v[:_MAX_TEXT_BYTES].decode(errors="ignore")
    elif isinstance(v, str):
        v = v[:_MAX_TEXT_BYTES]
    else:
        return None
    v = v.strip("\x00 ").strip()[:MAX_TEXT_CHARS]
    return v or None


def capture_time(text: str | None) -> str | None:
    """text when it is exactly EXIF's date and time form, "YYYY:MM:DD HH:MM:SS" (19 ASCII characters), else None:
    the one form of a capture time the store keeps (ingest)."""
    return text if isinstance(text, str) and _CAPTURE_TIME.fullmatch(text) else None


def jpeg_decodes_reduced(data: bytes) -> bool:
    """Whether a JPEG's pixels can be decoded straight at 1/2..1/8 scale (Pillow's draft) without buffers for the full
    image: a sequential DCT image (SOF0 or SOF1) whose first scan holds every component. Any other JPEG costs memory
    by its pixel count whatever the scale asked for: a progressive one, and equally a sequential one that sends its
    components in separate scans (Pillow reports only the first as "progressive"), make libjpeg keep every coefficient
    of the full image (9000 x 9000: 290-525 MB against 104 MB), and a lossless one has no reduced decode. Read from
    the segments before the first scan as libjpeg reads them (each skipped by its length, 0xFF fill bytes between
    them); anything else there (junk, a second frame header, a marker without a length, more than _JPEG_MAX_STEPS
    segments and fill bytes) is answered False, the careful side: ingest then applies Settings.max_image_pixels."""
    if data[:2] != b"\xff\xd8":
        return False
    pos, end, components = 2, len(data), None
    for _ in range(_JPEG_MAX_STEPS):
        if pos + 4 > end or data[pos] != 0xFF:
            return False
        marker = data[pos + 1]
        if marker == 0xFF:          # a fill byte before the marker
            pos += 1
            continue
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        if length < 2:
            return False
        if marker == _JPEG_SOS:
            return components is not None and pos + 4 < end and data[pos + 4] == components
        if marker in _JPEG_SEQUENTIAL:
            if components is not None or length < 8 or pos + 9 >= end:
                return False
            components = data[pos + 9]      # after the length: precision (1), height (2), width (2), components (1)
        elif marker not in _JPEG_SKIPPED:
            return False
        pos += 2 + length
    return False


def _check_header(data: bytes) -> None:
    """Raises UnreadableImage for a file whose header holds more than a camera's would, before Pillow opens it (D34,
    review round 2; the limits and their reasons are with the constants above). read_photo_meta calls this first, and
    ingest decodes nothing of a file it refuses. The format goes by the file's first bytes, as Pillow's does; any
    other file is left to Pillow, which refuses it from those same bytes."""
    if data[:3] == b"\xff\xd8\xff":
        _check_jpeg_header(data)
    elif data[:8] == _PNG_SIGNATURE:
        _check_png_chunks(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        _check_webp_chunks(data)


def _check_jpeg_header(data: bytes) -> None:
    """A JPEG from its start to its first scan, walked the way Pillow walks it when it opens the file
    (JpegImageFile._open; tests/test_geo_exif.py holds the two together): a byte that is not 0xFF is passed over, and
    so are 0xFF fill bytes, 0xFF 0x00 and the markers without a length; every other marker from 0xC0 up is a segment
    of the length its next two bytes give; anything else ends the walk, as it ends Pillow's (not a JPEG). Raises
    UnreadableImage over one of the limits: _JPEG_MAX_SEGMENTS, _JPEG_MAX_SKIPPED, MAX_JPEG_EXIF_BYTES for the "Exif"
    APP1 segments (each counted whole, so many small ones cannot make Pillow's joining of them expensive either),
    _JPEG_MAX_TABLE_BYTES, _JPEG_MAX_RESOURCE_BYTES and _JPEG_MAX_FRAME_BYTES."""
    end, pos = len(data), 2                 # pos: the byte Pillow looks at (it starts on the 0xFF after SOI)
    segments = skipped = exif = tables = resources = frames = 0
    while pos + 1 < end:
        if data[pos] != 0xFF:               # junk: byte by byte to the next 0xFF
            found = data.find(b"\xff", pos + 1)
            skipped += (found if found >= 0 else end) - pos
            pos = found if found >= 0 else end
        elif data[pos + 1] == 0xFF:         # a fill byte: the second 0xFF is looked at next
            skipped += 1
            pos += 1
        elif data[pos + 1] == 0x00 or data[pos + 1] in _JPEG_BARE:
            skipped += 1
            pos += 2
        elif data[pos + 1] not in _JPEG_SEGMENTS:
            return
        else:
            marker, start = data[pos + 1], pos + 4
            size = max(int.from_bytes(data[pos + 2:start], "big") - 2, 0)       # (a length under 2: no bytes)
            segments += 1
            if marker == _JPEG_APP1 and data.startswith(_EXIF_HEADER, start, start + size):
                exif += size
            elif marker == _JPEG_APP13 and data.startswith(_PHOTOSHOP_HEADER, start, start + size):
                resources += size
            elif marker == _JPEG_DQT:
                tables += size
            elif marker in _JPEG_FRAMES:
                frames += size
            if exif > MAX_JPEG_EXIF_BYTES:
                raise UnreadableImage(f"EXIF over {MAX_JPEG_EXIF_BYTES} bytes in a JPEG")
            if tables > _JPEG_MAX_TABLE_BYTES or resources > _JPEG_MAX_RESOURCE_BYTES or frames > _JPEG_MAX_FRAME_BYTES:
                raise UnreadableImage("a JPEG header with more tables, resource blocks or frame components than "
                                      "any camera writes")
            if marker == _JPEG_SOS and segments <= _JPEG_MAX_SEGMENTS:
                return
            pos = start + size
        if segments > _JPEG_MAX_SEGMENTS or skipped > _JPEG_MAX_SKIPPED:
            raise UnreadableImage(f"a JPEG with more than {_JPEG_MAX_SEGMENTS} segments or {_JPEG_MAX_SKIPPED} "
                                  f"stray bytes before its first scan")


def _check_png_chunks(data: bytes) -> None:
    """A PNG's chunks up to IEND, where Pillow stops too (it reads the ones before the pixel data when it opens the
    file, checks every one in verify(), and reads the ones after the pixel data when it has decoded). Raises
    UnreadableImage for more than _PNG_MAX_CHUNKS chunks that are not pixel data, _PNG_MAX_DATA_CHUNKS that are, or
    _PNG_MAX_INFLATED_CHUNKS that Pillow inflates; and for EXIF over MAX_EXIF_BYTES in either place Pillow takes it
    from: an eXIf chunk, also one after the pixel data (which this module's reader never sees, but Pillow's does once
    the pixels are decoded), and a text chunk "Raw profile type exif" (hexadecimal digits: two bytes of text for one
    of EXIF)."""
    end, pos = len(data), len(_PNG_SIGNATURE)
    chunks = pixels = inflated = 0
    while pos + 8 <= end:
        length, kind = int.from_bytes(data[pos:pos + 4], "big"), data[pos + 4:pos + 8]
        if kind == b"IEND":
            return
        if kind in _PNG_DATA_CHUNKS:
            pixels += 1
        else:
            chunks += 1
            inflated += kind in _PNG_INFLATED_CHUNKS
            if kind == b"eXIf" and len(_EXIF_HEADER) + length > MAX_EXIF_BYTES:
                raise UnreadableImage(f"EXIF block over {MAX_EXIF_BYTES} bytes")
            if kind in _PNG_TEXT_CHUNKS and length > 2 * MAX_EXIF_BYTES and data.startswith(_PNG_RAW_EXIF, pos + 8):
                raise UnreadableImage(f"EXIF block over {MAX_EXIF_BYTES} bytes")
        if chunks > _PNG_MAX_CHUNKS or pixels > _PNG_MAX_DATA_CHUNKS or inflated > _PNG_MAX_INFLATED_CHUNKS:
            raise UnreadableImage(f"a PNG with more than {_PNG_MAX_CHUNKS} chunks, {_PNG_MAX_DATA_CHUNKS} of pixel "
                                  f"data or {_PNG_MAX_INFLATED_CHUNKS} compressed ones")
        pos += 12 + length


def _check_webp_chunks(data: bytes) -> None:
    """A WebP's chunks: raises UnreadableImage for more than _WEBP_MAX_CHUNKS (the decoder library keeps a record of
    each: 3,000,000 empty ones in 24 MB took 0.3 GB) and for an EXIF chunk over MAX_EXIF_BYTES."""
    end, pos, chunks = len(data), 12, 0
    while pos + 8 <= end:
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        chunks += 1
        if chunks > _WEBP_MAX_CHUNKS:
            raise UnreadableImage(f"a WebP with more than {_WEBP_MAX_CHUNKS} chunks")
        if data[pos:pos + 4] == b"EXIF" and size > MAX_EXIF_BYTES:
            raise UnreadableImage(f"EXIF block over {MAX_EXIF_BYTES} bytes")
        pos += 8 + size + (size & 1)


class _BrokenExif(Exception):
    """The EXIF block has an entry no value can be read from: the whole block counts as unreadable."""


def _tiff_value(tiff: bytes, endian: str, kind: int, count: int, field: bytes, keep: bool):
    """The value of one directory entry (type kind, count values, the entry's 4-byte value field), as piexif.load
    returned it: bytes for ASCII (without its last byte, the terminator) and UNDEFINED, an int or a float for one
    number, a tuple for several, a (numerator, denominator) pair for one rational and a tuple of pairs for several.
    With two bounds piexif did not have: of a text at most _MAX_TEXT_BYTES bytes are taken (_text reads no more),
    and a tag of more than _MAX_NUMBERS numbers is _TOO_MANY, whatever its count says (piexif built a struct format
    of one character per declared value: 33 bytes of RAM each, kept in the struct module's cache). keep False only
    checks the entry and returns None. Raises _BrokenExif or struct.error for what piexif.load raised on: a type
    that is none of TIFF's twelve, numbers that lie outside the block, an entry cut off by the block's end."""
    if kind in (_TIFF_ASCII, _TIFF_UNDEFINED):
        length = count - 1 if kind == _TIFF_ASCII else count
        if count <= 4:
            return field[0:length] if keep else None
        (at,) = struct.unpack(endian + "L", field)
        return tiff[at:at + min(length, _MAX_TEXT_BYTES)] if keep else None
    if kind in _TIFF_RATIONALS:
        (at,) = struct.unpack(endian + "L", field)
        pairs = max(count, 1)                       # one pair is read for a count of 0 as well
        if at + 8 * pairs > len(tiff):
            raise _BrokenExif
        if not keep:
            return None
        if count > _MAX_NUMBERS:
            return _TOO_MANY
        numbers = struct.unpack(endian + _TIFF_RATIONALS[kind] * (2 * pairs), tiff[at:at + 8 * pairs])
        values = tuple(zip(numbers[0::2], numbers[1::2]))
        return values if count > 1 else values[0]
    if kind not in _TIFF_NUMBERS:
        raise _BrokenExif
    code, size, inline = _TIFF_NUMBERS[kind]
    if count > inline:
        (at,) = struct.unpack(endian + "L", field)
        if count and at + size * count > len(tiff):
            raise _BrokenExif
        if not keep:
            return None
        if count > _MAX_NUMBERS:
            return _TOO_MANY
        values = struct.unpack(endian + code * count, tiff[at:at + size * count])
    else:       # in the entry: at most four numbers (a 4-byte type takes the whole field, so a count of 0 is broken)
        values = struct.unpack(endian + code * count, field if size == 4 else field[0:size * count])
    return values[0] if len(values) == 1 else values


def _tiff_directory(tiff: bytes, endian: str, pointer, name: str) -> tuple[dict, bytes]:
    """The tags of _READ_TAGS[name] in the image file directory at pointer, and the four bytes after its entries (in
    the first directory: the offset of the next). Every entry of a tag in _KNOWN_TAGS[name] is checked
    (_tiff_value), a later entry of a tag replaces an earlier one, and the slices are piexif.load's own, so the
    same blocks are readable and the same values come out. More than _MAX_IFD_ENTRIES entries: _BrokenExif."""
    if not isinstance(pointer, int):        # the pointer tag was written as text, a float or several numbers
        raise _BrokenExif
    (entries,) = struct.unpack(endian + "H", tiff[pointer:pointer + 2])
    if entries > _MAX_IFD_ENTRIES:
        raise _BrokenExif
    known, read, values = _KNOWN_TAGS[name], _READ_TAGS[name], {}
    for index in range(entries):
        at = pointer + 2 + 12 * index
        (tag,) = struct.unpack(endian + "H", tiff[at:at + 2])
        (kind,) = struct.unpack(endian + "H", tiff[at + 2:at + 4])
        (count,) = struct.unpack(endian + "L", tiff[at + 4:at + 8])
        if tag in known:
            value = _tiff_value(tiff, endian, kind, count, tiff[at + 8:at + 12], tag in read)
            if tag in read:
                values[tag] = value
    end = pointer + 2 + 12 * entries
    return values, tiff[end:end + 4]


def _read_exif(exif_bytes: bytes) -> dict[str, dict] | None:
    """The EXIF tags read_photo_meta uses, as {"0th": {tag: value}, "Exif": {...}, "GPS": {...}} (_READ_TAGS, values
    as _tiff_value gives them), or None for a block that is not readable. exif_bytes is what Pillow hands out: the
    TIFF block after an "Exif\0\0" header (a JPEG's segments, a PNG's chunk), or a WebP's chunk as it is.

    This replaces piexif.load (D34), which is not safe on an upload: it builds a struct format of one character per
    value an entry declares (a 1.4 KB file took 2 GB, and the struct module kept it), copies a value once per tag
    that points at it (187 tags at one 20 MB value: 3.8 GB as text, 10 GB and 85 s as rationals), and opens a chunk
    that is neither TIFF nor JPEG as a file path on the server. The work here is bounded by the number of entries
    (_MAX_IFD_ENTRIES in each of five directories) whatever the block declares, and nothing is opened.

    The answer is the one piexif.load gave wherever that was affordable, so no photo reads differently: the same
    values (_tiff_value), and the same blocks count as unreadable, namely one with a broken entry of a tag piexif
    knows in any of its five directories (the first, Exif, GPS, interoperability and the thumbnail's) or a
    thumbnail whose offset and length are not two numbers. Two differences: a directory of more than
    _MAX_IFD_ENTRIES entries is unreadable, and so is a WebP chunk that starts neither with "Exif" nor with a TIFF
    byte order (piexif read a JPEG or a WebP file embedded there, or took the chunk for a path)."""
    if exif_bytes[0:4] == b"Exif":
        tiff = exif_bytes[6:]
    elif exif_bytes[0:2] in (b"II", b"MM"):
        tiff = exif_bytes
    else:
        return None
    endian = "<" if tiff[0:2] == b"II" else ">"
    try:
        (pointer,) = struct.unpack(endian + "L", tiff[4:8])
        zeroth, next_directory = _tiff_directory(tiff, endian, pointer, "0th")
        exif_ifd, gps = {}, {}
        if piexif.ImageIFD.ExifTag in zeroth:
            exif_ifd, _ = _tiff_directory(tiff, endian, zeroth[piexif.ImageIFD.ExifTag], "Exif")
        if piexif.ImageIFD.GPSTag in zeroth:
            gps, _ = _tiff_directory(tiff, endian, zeroth[piexif.ImageIFD.GPSTag], "GPS")
        if piexif.ExifIFD.InteroperabilityTag in exif_ifd:
            _tiff_directory(tiff, endian, exif_ifd[piexif.ExifIFD.InteroperabilityTag], "Interop")
        if next_directory != b"\x00\x00\x00\x00":
            (pointer,) = struct.unpack(endian + "L", next_directory)
            first, _ = _tiff_directory(tiff, endian, pointer, "1st")
            if all(tag in first for tag in _THUMBNAIL) and not all(isinstance(first[tag], int) for tag in _THUMBNAIL):
                raise _BrokenExif
    except (struct.error, _BrokenExif):
        return None
    return {"0th": zeroth, "Exif": exif_ifd, "GPS": gps}


def upright(img: Image.Image) -> Image.Image:
    """img's pixels turned upright by its orientation tag: img itself when there is nothing to turn. The pixels are
    ImageOps.exif_transpose's, by the same steps: the orientation as Pillow reads it (the EXIF tag, else the one in
    the XMP), and the same transposition. What is left out is the rest of that function: it writes the turned
    image's EXIF block out again without the tag, which parses every value in the block (a PNG or WebP with 3.9 MB
    of numbers there and an orientation: 0.65 s and 100 MB for each decode, 16 such uploads kept a server's other
    requests waiting 0.3 s for 15 s), for an EXIF block nothing here ever reads."""
    img.load()
    method = _UPRIGHT.get(img.getexif().get(ExifTags.Base.Orientation, 1))
    return img if method is None else img.transpose(method)


def read_photo_meta(data: bytes) -> PhotoMeta:
    """What the pipeline reads of an upload without decoding a pixel: size, format and the EXIF tags it uses. Raises
    UnreadableImage for bytes Pillow cannot open as a JPEG, PNG or WebP or does not verify, and for a header or an
    EXIF block over this module's limits (_check_header, MAX_EXIF_BYTES). Pillow opens the file once: with the
    header within the limits that costs at most about 0.1 s, and memory by the file's size (a few times a large
    text chunk's; measured in tests/test_geo_exif.py). ingest still runs it inside a decode slot."""
    _check_header(data)
    try:
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as img:
            width, height = img.size
            fmt = img.format
            exif_bytes = img.info.get("exif")
            img.verify()                    # last: a verified image is of no further use
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImage(str(exc)) from exc
    if not isinstance(exif_bytes, bytes):
        exif_bytes = b""
    if len(exif_bytes) > MAX_EXIF_BYTES:
        raise UnreadableImage(f"EXIF block over {MAX_EXIF_BYTES} bytes")

    reduced = fmt in JPEG_FORMATS and jpeg_decodes_reduced(data)
    ex = _read_exif(exif_bytes) if exif_bytes else None
    if ex is None:
        return PhotoMeta(width=width, height=height, format=fmt, decodes_reduced=reduced)

    gps = ex.get("GPS") or {}
    exif_ifd = ex.get("Exif") or {}
    zeroth = ex.get("0th") or {}

    lat = lon = None
    if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
        lat = _dms_to_deg(gps[piexif.GPSIFD.GPSLatitude], gps.get(piexif.GPSIFD.GPSLatitudeRef))
        lon = _dms_to_deg(gps[piexif.GPSIFD.GPSLongitude], gps.get(piexif.GPSIFD.GPSLongitudeRef))
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            lat = lon = None

    alt = None
    if piexif.GPSIFD.GPSAltitude in gps:
        alt = _rational_to_float(gps[piexif.GPSIFD.GPSAltitude])
        if alt is not None and gps.get(piexif.GPSIFD.GPSAltitudeRef) == 1:
            alt = -alt

    accuracy = None
    if piexif.GPSIFD.GPSHPositioningError in gps:
        accuracy = _rational_to_float(gps[piexif.GPSIFD.GPSHPositioningError])
        if accuracy is not None and not 0 < accuracy < 10_000:
            accuracy = None

    heading = None
    if piexif.GPSIFD.GPSImgDirection in gps:
        heading = _rational_to_float(gps[piexif.GPSIFD.GPSImgDirection])
        if heading is not None:
            heading %= 360.0

    focal_mm = None
    if piexif.ExifIFD.FocalLength in exif_ifd:
        focal_mm = _rational_to_float(exif_ifd[piexif.ExifIFD.FocalLength])
    focal_35 = exif_ifd.get(piexif.ExifIFD.FocalLengthIn35mmFilm)
    focal_35 = float(focal_35) if isinstance(focal_35, int) and focal_35 > 0 else None

    orientation = zeroth.get(piexif.ImageIFD.Orientation)
    return PhotoMeta(
        width=width,
        height=height,
        lat=lat,
        lon=lon,
        alt=alt,
        gps_accuracy_m=accuracy if lat is not None else None,
        heading_deg=heading,
        heading_ref=_text(gps.get(piexif.GPSIFD.GPSImgDirectionRef)),
        focal_mm=focal_mm,
        focal_35mm=focal_35,
        captured_at=_text(exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)),
        make=_text(zeroth.get(piexif.ImageIFD.Make)),
        model=_text(zeroth.get(piexif.ImageIFD.Model)),
        orientation=orientation if isinstance(orientation, int) else None,
        format=fmt,
        has_exif=True,
        decodes_reduced=reduced,
    )


def build_exif_bytes(
    *,
    lat: float | None = None,
    lon: float | None = None,
    alt: float | None = None,
    gps_accuracy_m: float | None = None,
    heading_deg: float | None = None,
    heading_ref: str = "T",
    focal_mm: float | None = None,
    focal_35mm: float | None = None,
    captured_at: str | None = None,
    make: str | None = "Crowd3D",
    model: str | None = "SyntheticCam",
) -> bytes:
    """Serialise an EXIF blob for Image.save(..., exif=...)."""
    zeroth: dict = {}
    exif_ifd: dict = {}
    gps: dict = {}
    if make:
        zeroth[piexif.ImageIFD.Make] = make.encode()
    if model:
        zeroth[piexif.ImageIFD.Model] = model.encode()
    if lat is not None and lon is not None:
        gps[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
        gps[piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
        gps[piexif.GPSIFD.GPSLatitude] = _deg_to_dms(lat)
        gps[piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
        gps[piexif.GPSIFD.GPSLongitude] = _deg_to_dms(lon)
    if alt is not None:
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = _to_rational(abs(alt), 1000)
    if gps_accuracy_m is not None:
        gps[piexif.GPSIFD.GPSHPositioningError] = _to_rational(gps_accuracy_m, 100)
    if heading_deg is not None:
        gps[piexif.GPSIFD.GPSImgDirectionRef] = heading_ref.encode()
        gps[piexif.GPSIFD.GPSImgDirection] = _to_rational(heading_deg % 360.0, 100)
    if focal_mm is not None:
        exif_ifd[piexif.ExifIFD.FocalLength] = _to_rational(focal_mm, 100)
    if focal_35mm is not None:
        exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(focal_35mm))
    if captured_at:
        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = captured_at.encode()
    return piexif.dump({"0th": zeroth, "Exif": exif_ifd, "GPS": gps, "1st": {}, "thumbnail": None})
