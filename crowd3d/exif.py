"""Read and write the EXIF fields the pipeline relies on.

Reading never raises on malformed EXIF: a missing or broken tag just becomes
None. Only an undecodable image raises UnreadableImage (one whose EXIF block is
over MAX_EXIF_BYTES counts as one). The block is read by this module's own
bounded reader (_read_exif); piexif only writes (build_exif_bytes).
"""

from __future__ import annotations

import io
import re
import struct
from dataclasses import dataclass
from fractions import Fraction

import piexif
from PIL import (  # noqa: F401 (registers formats)
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
# The EXIF block of an upload (D34). A camera's is at most one JPEG segment (64 KiB); a PNG's or WebP's chunk and
# Pillow's joined JPEG segments can be as big as the file. Over this the image counts as unreadable before anything
# parses the block: this module's reader does not care about its size, but Pillow's own EXIF parser runs on it when
# the pixels are turned upright (ImageOps.exif_transpose), at about 0.2 s and 25 MB per MiB of numbers.
MAX_EXIF_BYTES = 4 * 1024 * 1024
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


def read_photo_meta(data: bytes) -> PhotoMeta:
    try:
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as img:
            img.verify()
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as img:
            width, height = img.size
            fmt = img.format
            exif_bytes = img.info.get("exif")
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
