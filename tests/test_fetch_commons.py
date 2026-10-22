"""scripts/fetch_commons_landmark.py offline: saved Commons HTML, crafted JPEGs, a fake Wikimedia and a fake clock."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image

from crowd3d.exif import read_photo_meta
from crowd3d.geo import LocalENU

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_commons_landmark.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "commons"
BAPTISTERE = "File:Baptistère San Giovanni - Florence (IT52) - 2022-08-31 - 8.jpg"
LAT0, LON0 = 43.773139, 11.254944


@pytest.fixture(scope="module")
def fc():
    spec = importlib.util.spec_from_file_location("fetch_commons_landmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # dataclasses look their module up while the class is created
    spec.loader.exec_module(module)
    return module


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- crafted camera files


def _rational(x: float, den: int = 10000) -> tuple[int, int]:
    return int(round(x * den)), den


def _dms(x: float) -> tuple[tuple[int, int], ...]:
    x = abs(x)
    d = int(x)
    m = int((x - d) * 60)
    return (d, 1), (m, 1), _rational(((x - d) * 60 - m) * 60, 10000)


def camera_exif(lat: float | None, lon: float | None, *, heading: float | None = None, orientation: int = 1,
                size: tuple[int, int] = (3000, 2400), accuracy: float | None = 4.5) -> bytes:
    """EXIF like a phone writes it: GPS, compass, focal, MakerNote and an IFD1 preview."""
    gps: dict = {}
    if lat is not None and lon is not None:
        gps = {piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S", piexif.GPSIFD.GPSLatitude: _dms(lat),
               piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W", piexif.GPSIFD.GPSLongitude: _dms(lon),
               piexif.GPSIFD.GPSAltitude: (48409, 1000)}
        if accuracy is not None:
            gps[piexif.GPSIFD.GPSHPositioningError] = _rational(accuracy, 100)
    if heading is not None:
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b"T"
        gps[piexif.GPSIFD.GPSImgDirection] = _rational(heading, 100)
    preview = io.BytesIO()
    Image.new("RGB", (160, 120), (90, 120, 150)).save(preview, "JPEG")
    return piexif.dump({
        "0th": {piexif.ImageIFD.Make: b"Apple", piexif.ImageIFD.Model: b"iPhone 13 Pro",
                piexif.ImageIFD.Orientation: orientation, piexif.ImageIFD.ImageWidth: size[0],
                piexif.ImageIFD.ImageLength: size[1]},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2022:08:31 12:03:22", piexif.ExifIFD.FocalLength: (57, 10),
                 piexif.ExifIFD.FocalLengthIn35mmFilm: 26, piexif.ExifIFD.PixelXDimension: size[0],
                 piexif.ExifIFD.PixelYDimension: size[1], piexif.ExifIFD.MakerNote: bytes(range(256)) * 8},
        "GPS": gps,
        "1st": {piexif.ImageIFD.JPEGInterchangeFormat: 0, piexif.ImageIFD.JPEGInterchangeFormatLength: 0},
        "thumbnail": preview.getvalue(),
    })


def checkerboard(size: tuple[int, int], cell: int = 40, angle_deg: float = 0.0) -> Image.Image:
    """Sharp (passes the blur check) with a large-scale gradient whose direction tells images apart (dHash)."""
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    a = np.radians(angle_deg)
    ramp = (np.cos(a) * x / w + np.sin(a) * y / h) * 120
    board = ((x // cell) + (y // cell)) % 2 * 60 + 40 + ramp - ramp.min()
    g = np.clip(board, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([g, g, g], axis=-1))


def jpeg(size: tuple[int, int], exif: bytes | None = None, *, filler: list[int] | None = None,
         angle_deg: float = 0.0) -> bytes:
    """A JPEG of the given stored size; filler adds APP2 segments of those payload sizes right after SOI."""
    buf = io.BytesIO()
    img = checkerboard(size, angle_deg=angle_deg)
    img.save(buf, "JPEG", quality=70, **({"exif": exif} if exif else {}))
    data = buf.getvalue()
    for n in reversed(filler or []):
        data = data[:2] + b"\xff\xe2" + struct.pack(">H", n + 2) + bytes(n) + data[2:]
    return data


def offset_of_exif(data: bytes) -> int:
    return data.index(b"Exif\x00\x00") - 4


# ---------------------------------------------------------------- a fake Wikimedia


class FakeWikimedia:
    """Serves pages and files by URL, honours Range like upload.wikimedia.org, records every request."""

    def __init__(self, fc) -> None:
        self.fc = fc
        self.pages: dict[str, str] = {}
        self.files: dict[str, bytes] = {}
        self.scripted: dict[str, list[tuple[int, dict[str, str]]]] = {}   # url -> responses served first
        self.calls: list[tuple[float, str, dict[str, str]]] = []
        self.now = 1000.0

    # fake clock for PoliteClient
    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        assert s >= 0
        self.now += s

    def __call__(self, url: str, headers: dict[str, str], max_bytes: int | None):
        self.calls.append((self.now, url, dict(headers)))
        self.now += 0.05                                           # requests take a little time
        if self.scripted.get(url):
            status, h = self.scripted[url].pop(0)
            return status, h, b"error page"
        if url in self.pages:
            return 200, {"content-type": "text/html; charset=UTF-8"}, self.pages[url].encode()
        if url in self.files:
            data = self.files[url]
            rng = headers.get("Range")
            if rng:
                a, b = (int(x) for x in re.match(r"bytes=(\d+)-(\d+)", rng).groups())
                if a >= len(data):
                    return 416, {}, b""
                part = data[a:b + 1]
                return 206, {"content-range": f"bytes {a}-{a + len(part) - 1}/{len(data)}"}, part
            return 200, {"content-length": str(len(data))}, data
        return 404, {}, b"not found"

    def client(self, cache_dir: Path, **kw):
        return self.fc.PoliteClient(cache_dir, transport=self, clock=self.clock, sleep=self.sleep, **kw)


# ---------------------------------------------------------------- HTML parsers on real (trimmed) pages


def test_search_page_fixture(fc):
    items, total, next_offset = fc.parse_search_page(fixture("search_nearcoord.html"))
    assert [i.title for i in items] == ["File:Florence Duomo from Michelangelo hill.jpg",
                                        "File:Florence from the Duomo terrace (61408p).jpg",
                                        "File:Firenze, campanile di Giotto (07).jpg"]
    assert (items[0].width, items[0].height) == (3308, 2399)
    assert (items[2].width, items[2].height) == (5184, 3456)
    assert total == 1333 and next_offset == 100


def test_category_page_fixtures(fc):
    files, subcats, more = fc.parse_category_page(fixture("category_root.html"))
    assert files == [] and not more
    assert subcats[:2] == ["Historical images of the Baptistry (Florence)", "Plans of the Baptistry (Florence)"]
    assert "Exterior of the Baptistry (Florence)" in subcats and len(subcats) == 7

    files, subcats, more = fc.parse_category_page(fixture("category_exterior.html"))
    assert subcats == ["Exterior of the Baptistry (Florence) - Details", "Baptistry (Florence) at night",
                       "Exterior of the Baptistry (Florence) - Partial views"]
    assert len(files) == 3 and not more
    assert files[0] == fc.Listing("File:Florncie1352.jpg", 283, 270)


def test_category_page_reports_more_files_on_later_pages(fc):
    page = fixture("category_exterior.html").replace(
        "</ul></div>", '</ul><a href="/w/index.php?title=Category:X&amp;filefrom=Z" title="Category:X">next page</a>'
                       "</div>")
    assert fc.parse_category_page(page)[2]


def test_file_page_fixture(fc):
    page = fc.parse_file_page(fixture("file_page.html"))
    assert page.title == BAPTISTERE
    assert page.author == "Chabe01" and page.author_url == "https://commons.wikimedia.org/wiki/User:Chabe01"
    assert page.licences == [{"link": "https://creativecommons.org/licenses/by-sa/4.0", "short": "CC BY-SA 4.0",
                              "long": "Creative Commons Attribution-Share Alike 4.0", "link_req": "true",
                              "attr_req": "true"}]
    assert page.camera_location == {"lat": 43.773211, "lon": 11.255464, "heading_deg": 241.69}
    assert page.object_location is None
    assert page.categories == ["Exterior of the Baptistry (Florence)", "August 2022 in Florence"]
    assert (page.width, page.height, page.mime) == (4032, 3024, "image/jpeg")
    # The page's own "Original file" link is exactly the URL computed from the title's MD5.
    assert page.original_url == fc.upload_url(BAPTISTERE)
    assert page.metadata["exif-gpsimgdirection"] == "241.69180296691"
    lat, lon = fc.page_gps(page)
    assert lat == pytest.approx(43.773211, abs=1e-6) and lon == pytest.approx(11.255464, abs=1e-6)


def test_file_page_variants(fc):
    html = fixture("file_page.html")
    # A second licence template, an object location, no metadata table.
    html = html.replace('<h2 id="metadata">', '<div class="layouttemplate licensetpl"><span class="licensetpl&#95;short">'
                        'GFDL</span><span class="licensetpl&#95;attr">Photo: A. Person</span></div><h2 id="metadata">')
    html = html.replace("Camera location", "Object location")
    html = re.sub(r'<table id="mw&#95;metadata".*?</table>', "", html, flags=re.S)
    page = fc.parse_file_page(html)
    assert [lic["short"] for lic in page.licences] == ["CC BY-SA 4.0", "GFDL"]
    assert page.credit == "Photo: A. Person"
    assert page.camera_location is None and page.object_location["lat"] == 43.773211
    assert page.metadata == {} and fc.page_gps(page) is None


@pytest.mark.parametrize("text,expected", [
    ("43° 46′ 23.56″ N", 43.773211), ("11° 15′ 19.67″ E", 11.255464), ("33° 51′ 35.9″ S", -33.859972),
    ("0° 7′ 39″ W", -0.1275), ("48° N", 48.0), ("no coordinates", None),
])
def test_parse_dms(fc, text, expected):
    got = fc.parse_dms(text)
    assert got == pytest.approx(expected, abs=1e-6) if expected is not None else got is None


# ---------------------------------------------------------------- names and URLs


def test_titles_and_urls(fc):
    assert fc.normalize_title("File:baptist%C3%A8re_x.jpg") == "File:Baptistère x.jpg"
    assert fc.normalize_title("Image:a  b.JPG") == "File:A b.JPG"
    assert fc.normalize_title("c.jpg") == "File:C.jpg"
    assert fc.page_url(BAPTISTERE) == ("https://commons.wikimedia.org/wiki/File:Baptist%C3%A8re_San_Giovanni_-_"
                                       "Florence_(IT52)_-_2022-08-31_-_8.jpg")
    assert fc.upload_url(BAPTISTERE) == ("https://upload.wikimedia.org/wikipedia/commons/2/29/Baptist%C3%A8re_San_"
                                         "Giovanni_-_Florence_%28IT52%29_-_2022-08-31_-_8.jpg")
    assert fc.thumb_url(BAPTISTERE) == ("https://thumb.wikimedia.org/wikipedia/commons/thumb/2/29/Baptist%C3%A8re_"
                                        "San_Giovanni_-_Florence_%28IT52%29_-_2022-08-31_-_8.jpg/1920px-Baptist%C3%A8"
                                        "re_San_Giovanni_-_Florence_%28IT52%29_-_2022-08-31_-_8.jpg")
    long_title = "File:" + "x" * 200 + ".jpg"
    assert fc.thumb_url(long_title).endswith("/1920px-thumbnail.jpg")
    with pytest.raises(ValueError):
        fc.thumb_url(BAPTISTERE, 1600)          # not a standard width: Wikimedia answers 429
    name = fc.local_name(BAPTISTERE, ".jpg")
    assert re.fullmatch(r"[A-Za-z0-9._-]+\.jpg", name) and name != fc.local_name("File:Other.jpg", ".jpg")


def test_merge_candidates_dedupes_and_ranks_multi_source_first(fc):
    L = fc.Listing
    merged = fc.merge_candidates([
        ("category:A", [L("File:1.jpg", 10, 20), L("File:2.jpg")]),
        ("category:B", [L("File:2.jpg", 30, 40)]),
        ("titles:t.txt", [L("File:3.jpg"), L("File:2.jpg")]),
        ("search:s.html", [L("File:3.jpg"), L("File:4.jpg")]),
    ])
    assert [c.title for c in merged] == ["File:2.jpg", "File:3.jpg", "File:1.jpg", "File:4.jpg"]
    two = merged[0]
    assert two.sources == ["category:A", "category:B", "titles:t.txt"] and (two.width, two.height) == (30, 40)
    assert two.source_kinds == {"category", "titles"}


def test_check_shape(fc):
    fc.check_shape(None, None)
    fc.check_shape(4032, 3024)
    for (w, h), reason in (((8000, 1500), "panorama"), ((960, 640), None), ((600, 800), "too_small_original"),
                           ((639, 1200), "too_small_original")):
        if reason is None:
            fc.check_shape(w, h)
            continue
        with pytest.raises(fc.Skip) as e:
            fc.check_shape(w, h)
        assert e.value.reason == reason


# ---------------------------------------------------------------- EXIF bytes


def test_scan_jpeg_finds_exif_and_stored_size(fc):
    exif = camera_exif(LAT0, LON0, heading=90.0, orientation=6)
    data = jpeg((300, 200), exif)
    scan = fc.scan_jpeg(data[:4096])
    assert scan.need is None and scan.size == (300, 200)
    assert scan.app1 == exif                # Pillow writes the blob byte for byte
    scan = fc.scan_jpeg(jpeg((300, 200)))
    assert scan.app1 is None and scan.size == (300, 200)
    with pytest.raises(ValueError):
        fc.scan_jpeg(b"\x89PNG\r\n\x1a\n")


def test_scan_jpeg_asks_for_more_when_exif_crosses_the_prefix(fc):
    exif = camera_exif(LAT0, LON0)
    data = jpeg((300, 200), exif, filler=[65533, 64000])
    start = offset_of_exif(data)
    end = start + 2 + struct.unpack(">H", data[start + 2:start + 4])[0]
    assert start < fc.EXIF_PREFIX_BYTES < end
    scan = fc.scan_jpeg(data[:fc.EXIF_PREFIX_BYTES])
    assert scan.app1 is None and scan.need == end
    assert fc.scan_jpeg(data[:end]).app1 == exif


def test_read_original_exif_extends_the_range_once(fc, tmp_path):
    wm = FakeWikimedia(fc)
    title = "File:Big header.jpg"
    exif = camera_exif(LAT0, LON0)
    wm.files[fc.upload_url(title)] = jpeg((300, 200), exif, filler=[65533, 65533])   # EXIF starts after 128 KiB
    client = wm.client(tmp_path / "cache")
    scan, total = fc.read_original_exif(client, title, "jpeg")
    assert scan.app1 == exif and scan.size == (300, 200)
    assert total == len(wm.files[fc.upload_url(title)])
    ranges = [h["Range"] for _, _, h in wm.calls]
    assert ranges[0] == f"bytes=0-{fc.EXIF_PREFIX_BYTES - 1}" and ranges[1].startswith(f"bytes={fc.EXIF_PREFIX_BYTES}-")
    assert len(ranges) == 2


def test_scan_png_exif_chunk(fc):
    exif = camera_exif(LAT0, LON0, heading=12.0)
    buf = io.BytesIO()
    checkerboard((64, 48)).save(buf, "PNG", exif=exif)
    scan = fc.scan_png(buf.getvalue())
    assert scan.size == (64, 48) and scan.app1 == exif
    plain = io.BytesIO()
    checkerboard((64, 48)).save(plain, "PNG")
    assert fc.scan_png(plain.getvalue()).app1 is None


def test_meta_from_app1_is_the_product_reader(fc):
    exif = camera_exif(LAT0, LON0, heading=250.5, orientation=6)
    full = read_photo_meta(jpeg((300, 200), exif))
    meta = fc.meta_from_app1(exif)
    assert meta.lat == full.lat and meta.lon == full.lon and meta.heading_deg == 250.5
    assert (meta.focal_mm, meta.focal_35mm, meta.make, meta.model, meta.orientation) == (5.7, 26.0, "Apple",
                                                                                       "iPhone 13 Pro", 6)
    assert meta.gps_accuracy_m == 4.5 and meta.captured_at == "2022:08:31 12:03:22"
    assert fc.lost_fields(full, meta) == []


def test_patch_exif_changes_only_the_tagged_values(fc):
    exif = camera_exif(LAT0, LON0, heading=90.0, orientation=6, size=(4032, 3024))
    patched = fc.patch_exif(exif, width=1920, height=2560)
    assert len(patched) == len(exif)
    changed = [i for i, (a, b) in enumerate(zip(exif, patched, strict=True)) if a != b]
    assert 0 < len(changed) <= 5 * 4               # Orientation, ImageWidth/Length, PixelX/YDimension values
    d0, d1 = piexif.load(exif), piexif.load(patched)
    assert d1["0th"][piexif.ImageIFD.Orientation] == 1
    assert (d1["Exif"][piexif.ExifIFD.PixelXDimension], d1["Exif"][piexif.ExifIFD.PixelYDimension]) == (1920, 2560)
    assert (d1["0th"][piexif.ImageIFD.ImageWidth], d1["0th"][piexif.ImageIFD.ImageLength]) == (1920, 2560)
    assert d1["Exif"][piexif.ExifIFD.MakerNote] == d0["Exif"][piexif.ExifIFD.MakerNote]
    assert d1["GPS"] == d0["GPS"] and d1["thumbnail"] == d0["thumbnail"]
    kept = fc.patch_exif(exif, width=1920, height=2560, orientation=None)
    assert piexif.load(kept)["0th"][piexif.ImageIFD.Orientation] == 6
    with pytest.raises(ValueError):
        fc.patch_exif(b"Exif\x00\x00XX\x00*", width=1, height=1)


def test_graft_round_trip(fc):
    """Original EXIF into a Commons-style upright thumbnail: everything the product reads survives."""
    exif = camera_exif(LAT0 + 0.0003, LON0, heading=181.25, orientation=6, size=(3000, 2400))
    original = fc.meta_from_app1(exif)
    thumb = jpeg((1920, 2400), piexif.dump({"0th": {piexif.ImageIFD.Software: b"thumbor"}}))   # a stray EXIF
    grafted = fc.set_exif_segment(thumb, fc.patch_exif(exif, width=1920, height=2400))
    got = read_photo_meta(grafted)
    assert (got.width, got.height, got.orientation) == (1920, 2400, 1)
    assert fc.lost_fields(original, got) == []
    assert grafted.count(b"Exif\x00\x00") == 1 and grafted.count(b"thumbor") == 0
    assert grafted[2:4] == b"\xff\xe0"              # JFIF APP0 stays first, EXIF right behind it
    with Image.open(io.BytesIO(grafted)) as im:
        im.load()
        assert im.size == (1920, 2400)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(grafted))), np.asarray(Image.open(io.BytesIO(thumb))))


# ---------------------------------------------------------------- polite HTTP


def test_client_spaces_requests_and_caches_on_disk(fc, tmp_path):
    wm = FakeWikimedia(fc)
    wm.pages["https://x.test/a"] = "A"
    wm.pages["https://y.test/b"] = "B"
    client = wm.client(tmp_path)
    assert client.get("https://x.test/a").text() == "A"
    assert client.get("https://y.test/b").text() == "B"
    assert client.get("https://x.test/a").from_cache
    assert client.get("https://nowhere.test/c").status == 404       # 404 is cached too
    times = [t for t, _, _ in wm.calls]
    assert len(times) == 3 and all(b - a >= 1.0 for a, b in zip(times, times[1:]))   # any host: >= 1 s apart
    assert all(h["User-Agent"] == fc.USER_AGENT for _, _, h in wm.calls)
    again = wm.client(tmp_path)                                     # a rerun resumes from the cache
    assert again.get("https://y.test/b").from_cache and again.get("https://nowhere.test/c").status == 404
    assert len(wm.calls) == 3 and again.report()["network_requests"] == 0
    offline = fc.PoliteClient(tmp_path / "empty", offline=True)
    assert offline.get("https://x.test/a").status == fc.NETWORK_ERROR


def test_client_honours_retry_after_and_never_caches_429(fc, tmp_path):
    wm = FakeWikimedia(fc)
    url = "https://upload.test/f.jpg"
    wm.files[url] = b"0123456789"
    wm.scripted[url] = [(429, {"retry-after": "7"}), (503, {})]
    client = wm.client(tmp_path, base_backoff_s=3.0)
    resp = client.get(url, byte_range=(0, 3))
    assert resp.status == 206 and resp.body == b"0123"
    t = [c[0] for c in wm.calls]
    assert t[1] - t[0] >= 7.0                     # Retry-After honoured
    assert client.min_interval_s == fc.MIN_INTERVAL_S + fc.INTERVAL_STEP_S   # and slower from now on
    assert t[2] - t[1] >= 6.0                     # no Retry-After on the 503: backoff 3 s * 2
    report = client.report()
    assert report["http_429"] == 1 and report["http_5xx"] == 1 and report["server_ordered_wait_s"] == 13.0
    assert client.get(url, byte_range=(0, 3)).from_cache


def test_client_backs_off_exponentially_when_a_short_retry_after_keeps_failing(fc, tmp_path):
    """thumb.wikimedia.org answered 429 'Retry-After: 1' four times in a row (sagrada-familia, 2026-10-22):
    the first retry honours the 1 s, later retries of the same request wait base * 2^(n-1) instead."""
    wm = FakeWikimedia(fc)
    url = "https://thumb.test/t.jpg"
    wm.files[url] = b"pixels"
    wm.scripted[url] = [(429, {"retry-after": "1"})] * 3
    client = wm.client(tmp_path, base_backoff_s=5.0)
    assert client.get(url).status == 200
    t = [c[0] for c in wm.calls]
    gaps = [b - a for a, b in zip(t, t[1:])]
    assert len(t) == 4 and gaps[0] >= 1.0 and gaps[1] >= 5.0 and gaps[2] >= 10.0
    assert client.report()["server_ordered_wait_s"] == 16.0


def test_client_stops_when_asked_to_wait_too_long(fc, tmp_path):
    wm = FakeWikimedia(fc)
    url = "https://upload.test/g.jpg"
    wm.scripted[url] = [(429, {"retry-after": "600"})]
    client = wm.client(tmp_path, max_wait_s=300.0)
    with pytest.raises(fc.RateLimited):
        client.get(url)
    assert len(wm.calls) == 1 and client.report()["http_429"] == 1
    # The host stays blocked for the requested time: the next request to it waits first.
    wm.files[url] = b"ok"
    before = wm.now
    assert client.get(url).status == 200 and wm.calls[-1][0] - before >= 599.0


def test_retry_after_parsing(fc):
    assert fc.retry_after_s("120") == 120.0
    assert fc.retry_after_s("Wed, 21 Oct 2026 22:54:38 GMT", now=1792623218.0) == pytest.approx(60.0)
    assert fc.retry_after_s(None) is None and fc.retry_after_s("soon") is None


def test_urllib_transport_reads_only_the_requested_bytes_and_gunzips(fc, monkeypatch):
    import gzip as gz

    class FakeResp:
        def __init__(self, body: bytes, headers: dict[str, str]) -> None:
            self.status, self.headers, self._body, self.read_sizes = 200, headers, body, []

        def read(self, n: int | None = None) -> bytes:
            self.read_sizes.append(n)
            return self._body if n is None else self._body[:n]

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None:
            pass

    resp = FakeResp(b"x" * 1000, {"Content-Type": "image/jpeg"})
    monkeypatch.setattr(fc.urllib.request, "urlopen", lambda req, timeout: resp)
    status, h, body = fc.urllib_transport("https://u.test/a.jpg", {"Range": "bytes=0-99"}, 100)
    assert status == 200 and len(body) == 100 and resp.read_sizes == [100]   # server ignored Range: stop reading
    zipped = FakeResp(gz.compress(b"<html>ok</html>"), {"Content-Encoding": "gzip"})
    monkeypatch.setattr(fc.urllib.request, "urlopen", lambda req, timeout: zipped)
    assert fc.urllib_transport("https://c.test/wiki/X", {"Accept-Encoding": "gzip"}, None)[2] == b"<html>ok</html>"


# ---------------------------------------------------------------- end to end against a fake Commons


def _gallery(fc, items: list[tuple[str, int, int]]) -> str:
    """Category media section in the markup of the real fixture, with our own files."""
    real = fixture("category_exterior.html")
    box = re.search(r'<li class="gallerybox".*?</li>', real, re.S).group(0)
    boxes = []
    for title, w, h in items:
        name = title[5:].replace(" ", "_")
        b = box.replace("Florncie1352.jpg", name).replace('data-file-width="283"', f'data-file-width="{w}"')
        boxes.append(b.replace('data-file-height="270"', f'data-file-height="{h}"'))
    return '<div id="mw-category-media"><ul class="gallery mw-gallery-traditional">' + "".join(boxes) + "</ul></div>"


def _subcats(names: list[str]) -> str:
    links = "".join(f'<li><a href="/wiki/Category:{n.replace(" ", "_")}" title="Category:{n}">{n}</a></li>'
                    for n in names)
    return f'<div id="mw-subcategories"><ul>{links}</ul></div>'


def _file_page(fc, title: str, size: tuple[int, int], gps: tuple[float, float] | None) -> str:
    html = fixture("file_page.html").replace(BAPTISTERE[5:], title[5:])
    html = html.replace("4,032 × 3,024", f"{size[0]:,} × {size[1]:,}")
    if gps is None:
        html = re.sub(r'<tr class="exif-gps(latitude|longitude)">.*?</tr>', "", html)
    else:
        lat, lon = gps
        dms = [(int(abs(v)), int(abs(v) * 60 % 60), abs(v) * 3600 % 60) for v in (lat, lon)]
        html = html.replace("43° 46′ 23.56″ N", "%d° %d′ %.2f″ N" % dms[0])
        html = html.replace("11° 15′ 19.67″ E", "%d° %d′ %.2f″ E" % dms[1])
    return html


def _offset(east: float, north: float) -> tuple[float, float]:
    lat, lon, _ = LocalENU(LAT0, LON0).to_geodetic(east, north)
    return lat, lon


# title: (listing size (upright), stored size, orientation, page GPS, EXIF GPS, heading)
E2E_FILES = {
    "File:Good thumb.jpg": ((2400, 3000), (3000, 2400), 6, _offset(0, -40), _offset(0, -40), 0.0),
    "File:Good small.jpg": ((1600, 1200), (1600, 1200), 1, _offset(30, 0), _offset(30, 0), None),
    "File:No gps page.jpg": ((1600, 1200), (1600, 1200), 1, None, None, None),
    "File:Far away.jpg": ((1600, 1200), (1600, 1200), 1, _offset(2000, 0), _offset(2000, 0), None),
    "File:Pano.jpg": ((8000, 1500), (8000, 1500), 1, None, None, None),
    "File:Tiny.jpg": ((500, 400), (500, 400), 1, None, None, None),
    "File:Drawing.svg": ((1000, 1000), (1000, 1000), 1, None, None, None),
    "File:Lied page.jpg": ((1600, 1200), (1600, 1200), 1, _offset(10, 10), None, None),
    "File:Deleted.jpg": ((1600, 1200), (1600, 1200), 1, None, None, None),
}
NEVER_FETCHED = ("File:Pano.jpg", "File:Tiny.jpg", "File:Drawing.svg", "File:Deleted.jpg")


@pytest.fixture(scope="module")
def commons_assets(fc):
    """Pages and files of a fake Commons with one landmark category, built once per module."""
    pages: dict[str, str] = {}
    files: dict[str, bytes] = {}
    cat = lambda n: fc.page_url("Category:" + n)   # noqa: E731
    root = list(E2E_FILES)[:5] + ["File:Tiny.jpg", "File:Drawing.svg", "File:Deleted.jpg"]
    pages[cat("Test Landmark")] = "<html><body>" + _subcats(["Test Landmark at night", "Interior of Test Landmark"]) \
        + _gallery(fc, [(t, *E2E_FILES[t][0]) for t in root]) + "</body></html>"
    pages[cat("Test Landmark at night")] = "<html><body>" + _subcats(["Too deep"]) \
        + _gallery(fc, [(t, *E2E_FILES[t][0]) for t in ("File:Lied page.jpg", "File:Good small.jpg")]) \
        + "</body></html>"
    for title, (listed, stored, orientation, page_gps, exif_gps, heading) in E2E_FILES.items():
        if title in NEVER_FETCHED:
            continue
        pages[fc.page_url(title)] = _file_page(fc, title, listed, page_gps)
        exif = camera_exif(*(exif_gps or (None, None)), heading=heading, orientation=orientation, size=stored)
        files[fc.upload_url(title)] = jpeg(stored, exif, angle_deg=len(title) * 37.0)
    files[fc.thumb_url("File:Good thumb.jpg")] = jpeg((1920, 2400), angle_deg=180.0)
    files[fc.thumb_url("File:Good small.jpg", 1280)] = jpeg((1280, 960), angle_deg=90.0)
    return pages, files


@pytest.fixture
def fake_commons(fc, commons_assets):
    wm = FakeWikimedia(fc)
    wm.pages, wm.files = dict(commons_assets[0]), dict(commons_assets[1])
    return wm


def test_end_to_end(fc, fake_commons, tmp_path):
    wm = fake_commons
    out = tmp_path / "ten" / "test"
    client = wm.client(tmp_path / "ten" / "http-cache")
    rc = fc.main(["--name", "Test Landmark", "--lat", str(LAT0), "--lon", str(LON0), "--radius", "150",
                  "--category", "Test Landmark", "--exclude-category", "interior", "--out", str(out),
                  "--prescreen-audit", "1"], client=client)
    assert rc == 0
    stats = json.loads((out / "screening_stats.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    f = stats["funnel"]
    assert f["candidates"] == 9 and f["page_screen_passed"] == 3 and f["with_exif_gps_in_radius"] == 2
    assert f["kept"] == 2 and f["kept_with_heading"] == 1 and f["kept_with_focal"] == 2
    assert f["skip_reasons"] == {"unsupported_format": 1, "panorama": 1, "too_small_original": 1, "page_missing": 1,
                                 "page_no_gps": 1, "gps_outside_radius": 1, "no_gps": 1}
    assert stats["listing"]["categories"]["excluded"] == ["Interior of Test Landmark"]
    assert [v["category"] for v in stats["listing"]["categories"]["visited"]] == ["Test Landmark",
                                                                                 "Test Landmark at night"]
    assert stats["prescreen_audit"]["checked"] == 1 and stats["prescreen_audit"]["missed_gps"] == 0
    assert stats["aborted"] is None and stats["http"]["http_429"] == 0

    by_title = {p["title"]: p for p in manifest["photos"]}
    thumb = by_title["File:Good thumb.jpg"]
    assert thumb["pixels"]["source"] == "thumbnail_1920" and thumb["pixels"]["orientation_reset"]
    assert (thumb["pixels"]["width"], thumb["pixels"]["height"]) == (1920, 2400)
    assert (thumb["original"]["stored_width"], thumb["original"]["stored_height"]) == (3000, 2400)
    assert thumb["licence"] == "CC BY-SA 4.0" and thumb["author"] == "Chabe01"
    assert thumb["landmark_to_camera"]["sector_label"] == "S"
    assert thumb["landmark_to_camera"]["distance_m"] == pytest.approx(40.0, abs=0.01)
    assert thumb["landmark_to_camera"]["heading_offset_deg"] == pytest.approx(0.0, abs=0.01)
    assert thumb["exif"]["orientation"] == 6 and thumb["exif"]["gps_accuracy_m"] == 4.5
    small = by_title["File:Good small.jpg"]
    assert small["pixels"]["source"] == "original" and small["sources"] == ["category:Test Landmark",
                                                                          "category:Test Landmark at night"]
    assert small["validation"]["accepted"] and "no_heading" in small["validation"]["warnings"]

    delivered = read_photo_meta((out / thumb["pixels"]["file"]).read_bytes())
    assert delivered.orientation == 1 and delivered.heading_deg == 0.0
    assert delivered.lat == pytest.approx(thumb["exif"]["lat"], abs=1e-7)

    urls = [u for _, u, _ in wm.calls]
    assert not any("/w/" in u or "Special:" in u or "api.php" in u for u in urls)   # robots.txt-allowed paths only
    assert not any(fc.page_url(t) in urls for t in ("File:Pano.jpg", "File:Tiny.jpg", "File:Drawing.svg"))
    assert fc.page_url("Category:Too deep") not in urls and fc.page_url("Category:Interior of Test Landmark") not in urls
    times = [t for t, _, _ in wm.calls]
    assert all(b - a >= 1.0 for a, b in zip(times, times[1:]))
    assert stats["http"]["network_requests"] == len(wm.calls)
    outcomes = {r["title"]: r["outcome"] for r in json.loads((out / "candidates.json").read_text())}
    assert outcomes["File:Lied page.jpg"] == "no_gps" and outcomes["File:Good thumb.jpg"] == "kept"

    # Rerun: everything comes from the cache, same result.
    n = len(wm.calls)
    assert fc.main(["--name", "Test Landmark", "--lat", str(LAT0), "--lon", str(LON0), "--category", "Test Landmark",
                    "--exclude-category", "interior", "--out", str(out)],
                   client=wm.client(tmp_path / "ten" / "http-cache")) == 0
    assert len(wm.calls) == n
    assert json.loads((out / "screening_stats.json").read_text())["funnel"]["kept"] == 2


def test_end_to_end_stops_politely_on_rate_limit(fc, fake_commons, tmp_path):
    wm = fake_commons
    wm.scripted[fc.upload_url("File:Good thumb.jpg")] = [(429, {"retry-after": "600"})]
    out = tmp_path / "run"
    rc = fc.main(["--name", "T", "--lat", str(LAT0), "--lon", str(LON0), "--category", "Test Landmark",
                  "--category-depth", "0", "--out", str(out), "--max-wait", "300", "--exif-source", "original"],
                 client=wm.client(tmp_path / "cache", max_wait_s=300.0))
    assert rc == 1
    stats = json.loads((out / "screening_stats.json").read_text())
    assert "429" in stats["aborted"] and stats["funnel"]["page_screen_passed"] == 2   # phase 1 still reported
    assert stats["funnel"]["kept"] == 0 and stats["http"]["http_429"] == 1
    assert sum(1 for _, u, _ in wm.calls if "upload" in u) == 1                        # no retry hammering


def test_refuses_output_inside_the_repo(fc, tmp_path):
    with pytest.raises(SystemExit, match="inside the repository"):
        fc.main(["--name", "T", "--lat", "0", "--lon", "0", "--category", "X", "--out", str(SCRIPT.parent / "out")])
    with pytest.raises(SystemExit, match="candidate source"):
        fc.main(["--name", "T", "--lat", "0", "--lon", "0", "--out", str(tmp_path)])


def test_end_to_end_falls_back_to_page_exif_when_originals_are_refused(fc, fake_commons, tmp_path):
    wm = fake_commons
    wm.scripted[fc.upload_url("File:Good thumb.jpg")] = [(429, {"retry-after": "600"})]
    out = tmp_path / "run"
    rc = fc.main(["--name", "T", "--lat", str(LAT0), "--lon", str(LON0), "--category", "Test Landmark",
                  "--category-depth", "0", "--out", str(out), "--prescreen-audit", "3"],
                 client=wm.client(tmp_path / "cache"))
    assert rc == 0
    stats = json.loads((out / "screening_stats.json").read_text())
    photos = {p["title"]: p for p in json.loads((out / "manifest.json").read_text())["photos"]}
    assert stats["funnel"]["kept"] == 2 and "429" in stats["originals_fallback"]
    assert stats["funnel"]["kept_by_exif_source"] == {"commons_page_metadata": 2}
    assert stats["prescreen_audit"]["checked"] == 0                       # needs originals: not attempted
    small = photos["File:Good small.jpg"]
    assert small["pixels"]["source"] == "thumbnail_1280" and small["exif_source"] == "commons_page_metadata"
    # The page fixture's camera values, re-encoded: still what the product reads.
    got = read_photo_meta((out / small["pixels"]["file"]).read_bytes())
    assert (got.width, got.height, got.focal_mm, got.focal_35mm, got.make) == (1280, 960, 5.7, 26.0, "Apple")
    assert got.heading_deg == pytest.approx(241.69, abs=0.01) and got.captured_at == "2022:08:31 12:03:00"
    exp_lat, _ = _offset(30, 0)
    assert got.lat == pytest.approx(exp_lat, abs=0.01 / 3600)            # MediaWiki shows 0.01 arc-seconds
    assert sum(1 for _, u, _ in wm.calls if "upload.wikimedia.org" in u) == 1   # one refused original, no more


def test_exif_from_page_fixture(fc):
    page = fc.parse_file_page(fixture("file_page.html"))
    meta = fc.meta_from_app1(fc.exif_from_page(page))
    assert meta.lat == pytest.approx(43.773211, abs=1e-6) and meta.lon == pytest.approx(11.255464, abs=1e-6)
    assert meta.alt == pytest.approx(48.409) and meta.heading_ref == "T"
    assert meta.heading_deg == pytest.approx(241.69180296691, abs=0.01)
    assert (meta.focal_mm, meta.focal_35mm, meta.make, meta.model) == (5.7, 26.0, "Apple", "iPhone 13 Pro")
    assert meta.captured_at == "2022:08:31 12:03:00" and meta.gps_accuracy_m is None
    page.metadata.pop("exif-gpslatitude")
    assert fc.exif_from_page(page) is None


@pytest.mark.parametrize("text,expected", [
    ("12:03, 31 August 2022", "2022:08:31 12:03:00"), ("07:05:09, 1 March 2019", "2019:03:01 07:05:09"),
    ("31 August 2022", "2022:08:31 00:00:00"), ("unknown", None),
])
def test_page_datetime(fc, text, expected):
    assert fc.page_datetime(text) == expected


def test_pick_thumb_width(fc):
    assert fc.pick_thumb_width(4032, 3024) == 1920
    assert fc.pick_thumb_width(1920, 1080) == 1280           # never the original's own width (no upscaling)
    assert fc.pick_thumb_width(1600, 1200) == 1280
    assert fc.pick_thumb_width(1200, 1600) == 960
    assert fc.pick_thumb_width(1000, 750) == 960
    assert fc.pick_thumb_width(800, 700) is None             # the 500 px step is below the 640 px short side


def test_rerun_offline_uses_page_exif_for_originals_never_cached(fc, fake_commons, tmp_path):
    wm = fake_commons
    wm.scripted[fc.upload_url("File:Good thumb.jpg")] = [(429, {"retry-after": "600"})]
    args = ["--name", "T", "--lat", str(LAT0), "--lon", str(LON0), "--category", "Test Landmark",
            "--category-depth", "0"]
    assert fc.main([*args, "--out", str(tmp_path / "a")], client=wm.client(tmp_path / "cache")) == 0
    n = len(wm.calls)
    offline = wm.client(tmp_path / "cache", offline=True)
    assert fc.main([*args, "--out", str(tmp_path / "b")], client=offline) == 0
    assert len(wm.calls) == n                     # nothing fetched
    a, b = (json.loads((tmp_path / d / "manifest.json").read_text())["photos"] for d in ("a", "b"))
    assert [p["pixels"]["sha256"] for p in a] == [p["pixels"]["sha256"] for p in b] and len(b) == 2


def test_crop_with_the_same_exif_is_a_duplicate(fc, fake_commons, tmp_path):
    wm = fake_commons
    crop = "File:Good small crop.jpg"
    stored = (1500, 1100)
    wm.pages[fc.page_url(crop)] = _file_page(fc, crop, stored, _offset(30, 0))
    wm.files[fc.upload_url(crop)] = jpeg(stored, camera_exif(*_offset(30, 0), size=stored), angle_deg=300.0)
    titles = tmp_path / "titles.txt"
    titles.write_text("File:Good small.jpg\nFile:Good_small_crop.jpg\n", encoding="utf-8")
    out = tmp_path / "run"
    assert fc.main(["--name", "T", "--lat", str(LAT0), "--lon", str(LON0), "--titles", str(titles),
                    "--out", str(out)], client=wm.client(tmp_path / "cache")) == 0
    stats = json.loads((out / "screening_stats.json").read_text())
    assert stats["funnel"]["kept"] == 1 and stats["funnel"]["skip_reasons"] == {"duplicate_exif": 1}
