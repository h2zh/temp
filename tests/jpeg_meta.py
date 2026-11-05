"""What a served JPEG carries besides its pixels (D32: the public photo has no metadata). Imported by the tests (pytest
puts tests/ on the path)."""

import io

from PIL import Image

SOI, EOI, SOS, APP0, COM = 0xD8, 0xD9, 0xDA, 0xE0, 0xFE
# SOI, APP0 (JFIF), DQT, SOF0 (baseline), SOF2 (progressive), DHT, DRI, SOS, EOI: nothing else may stand in a public
# photo (no other APPn: EXIF / XMP in APP1, ICC in APP2, Adobe in APP14; no COM comment)
PUBLIC_MARKERS = {SOI, APP0, 0xDB, 0xC0, 0xC2, 0xC4, 0xDD, SOS, EOI}
# text a metadata block would carry: EXIF, its GPS IFD, XMP, an ICC profile
METADATA_TEXT = (b"Exif", b"GPS", b"http://ns.adobe.com/xap", b"ICC_PROFILE")


def jpeg_segments(data: bytes) -> list[tuple[int, bytes]]:
    """Every marker of a JPEG file in order, SOI to EOI, with its segment's payload (b"" for SOI / EOI), skipping the
    entropy-coded data after each SOS (stuffed 0xFF00 bytes and RSTn markers). Raises AssertionError for anything
    that is not a well-formed JPEG ending exactly at its EOI (no trailing data)."""
    assert data[:2] == b"\xff\xd8", "no SOI"
    out, i = [(SOI, b"")], 2
    while True:
        assert i < len(data) and data[i] == 0xFF, f"no marker at {i}"
        while data[i] == 0xFF:          # fill bytes
            i += 1
        m = data[i]
        i += 1
        if m == EOI:
            out.append((EOI, b""))
            assert i == len(data), f"{len(data) - i} bytes after EOI"
            return out
        length = int.from_bytes(data[i:i + 2], "big")
        out.append((m, data[i + 2:i + length]))
        i += length
        if m == SOS:                    # entropy-coded data up to the next marker that is not stuffing or RSTn
            while True:
                j = data.index(b"\xff", i)
                k = j + 1
                while data[k] == 0xFF:
                    k += 1
                if data[k] == 0x00 or 0xD0 <= data[k] <= 0xD7:
                    i = k + 1
                    continue
                i = j
                break


def assert_metadata_free(data: bytes, secrets: tuple[bytes, ...] = ()) -> None:
    """A public photo (photoposes.public_photo): only the segments of PUBLIC_MARKERS, one APP0 and it is JFIF, no EXIF
    readable, none of METADATA_TEXT nor of the secrets in its segments. A text of 8 bytes or more is in none of its
    bytes either; a shorter one ("GPS", a nickname) may occur by chance in the entropy-coded data, where no reader
    looks for metadata."""
    segments = jpeg_segments(data)
    markers = [m for m, _ in segments]
    assert set(markers) <= PUBLIC_MARKERS, [hex(m) for m in markers]
    assert markers.count(APP0) == 1 and dict(segments)[APP0].startswith(b"JFIF\x00")
    headers = b"".join(payload for _, payload in segments)
    for text in (*METADATA_TEXT, *secrets):
        assert text not in headers, text
        assert len(text) < 8 or text not in data, text
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "JPEG" and not im.getexif()
