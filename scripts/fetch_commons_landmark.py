"""Build a real-photo test set for one landmark from Wikimedia Commons: only photos whose OWN EXIF has GPS.

    python scripts/fetch_commons_landmark.py --name "Baptistry (Florence)" --lat 43.773139 --lon 11.254944 \\
        --radius 150 --category "Baptistry (Florence)" --category-depth 3 \\
        --exclude-category "interior|plans|in art|historical|art from" --out <dir outside the repo>

Product rule: uploads must be camera originals with EXIF and GPS, so the test set contains only Commons
files whose own EXIF carries GPS latitude/longitude. Nothing is simulated; compass heading stays optional
and is only counted.

Pipeline (strictly sequential, every HTTP response cached under --cache-dir, so a rerun resumes and
--offline replays a run without the network):
1. Candidates. Commons category pages /wiki/Category:<name> (first page of each, + subcategories up to
   --category-depth, minus --exclude-category), optionally file titles (--titles, e.g. a MegaDepth-X
   scene's image_name column) and saved Special:Search result pages (--search-html, e.g. a
   nearcoord:<r>m,<lat>,<lon> search). De-duplicated; files named by several kinds of source first;
   at most --max-candidates.
2. Page screen (Commons HTML only). JPEG/PNG only, no panoramas (aspect > 3), short side >= 640 px; then
   the file description page /wiki/File:<name>: author, licences, Commons coordinates, categories and
   MediaWiki's table of the file's EXIF. Files whose table shows no GPS, or GPS beyond --keep-radius,
   are skipped here (--no-page-prescreen keeps them; --prescreen-audit N range-checks N of them).
3. EXIF. --exif-source original: a Range request reads the first 128 KiB of the original (extended once
   when the EXIF segment is longer) and crowd3d.exif.read_photo_meta, the product's own reader, parses
   it. --exif-source page: the EXIF values the description page shows are re-encoded with
   crowd3d.exif.build_exif_bytes (GPS rounded to 0.01 arc-second, time to the minute, MakerNote and
   GPSHPositioningError lost). auto (default): original, switching to page for the rest of the run
   once upload.wikimedia.org refuses originals with a long Retry-After. manifest.json says which.
4. Pixels. A standard-width Commons thumbnail (1920 px; smaller originals: the original itself when
   reachable, else the largest standard width below it) with the EXIF APP1 segment grafted in: the
   original's bytes unchanged except Orientation := 1 (Commons thumbnails are already rotated upright)
   and the pixel-dimension tags. This equals the server's own processing copy (it keeps the upload's
   EXIF and downscales to Settings.processing_max_side_px = 1600). read_photo_meta must read back the
   same GPS, heading, focal, time and camera from the result, else the file is dropped. A file whose
   EXIF GPS, time and heading repeat a kept file's (a crop or re-upload of the same shot) is dropped too.
5. Output in --out: photos/, manifest.json (title, URLs, author, licences, EXIF fields and their source,
   Commons coordinates, landmark -> camera bearing/distance/sector, the product validator's verdict),
   screening_stats.json (funnel, skip reasons, HTTP counts, 429s) and candidates.json (every fate).

Access (checked 2026-10-21 from this shared cloud IP): the MediaWiki Action API and the REST summary
answer 429 to anonymous requests; robots.txt disallows /w/ and /wiki/Special: (Special:Search,
Special:FilePath) for crawlers; upload.wikimedia.org refuses originals (429, Retry-After 600) and
thumbnails ("robot policy" 429), while thumb.wikimedia.org - the thumbnail host Commons' own pages
link - serves standard widths. So this tool requests only /wiki/Category:, /wiki/File: (CDN-cached,
no query string), thumb.wikimedia.org and upload.wikimedia.org originals.
Politeness: identifying User-Agent, one request at a time, >= 1 s between any two requests (each 429
widens that by 0.25 s, up to 3 s), Retry-After honoured on 429/5xx (exponential backoff when it is absent,
or on top of it when the same request is refused again), and a stop instead of a retry when a server asks
for a longer pause than --max-wait.

Licensing: every Commons file keeps its own licence (recorded in manifest.json). Photos and anything
derived from them must stay outside this repository: --out and --cache-dir are refused inside it
(except its gitignored data/ folder).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html.parser
import http.client
import io
import json
import logging
import math
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in (None, ""):
    # `python scripts/fetch_commons_landmark.py` from a checkout: make the crowd3d package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd3d.config import Settings  # noqa: E402
from crowd3d.coverage import sector_for_bearing, sector_label  # noqa: E402
from crowd3d.exif import PhotoMeta, UnreadableImage, build_exif_bytes, read_photo_meta  # noqa: E402
from crowd3d.geo import angle_diff_deg, local_bearing_distance  # noqa: E402
from crowd3d.records import Landmark  # noqa: E402
from crowd3d.validation import validate_photo  # noqa: E402

log = logging.getLogger("fetch_commons_landmark")

REPO_ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "Crowd3D-prototype/0.1 (https://github.com/h2zh/temp; research evaluation)"
COMMONS = "https://commons.wikimedia.org"
UPLOAD_BASE = "https://upload.wikimedia.org/wikipedia/commons"
# Commons' own pages link thumbnails on this host; upload.wikimedia.org/.../thumb/ answers 429 "robot policy"
# to this machine while thumb.wikimedia.org serves the same files.
THUMB_BASE = "https://thumb.wikimedia.org/wikipedia/commons/thumb"
# Wikimedia only serves these thumbnail widths to direct requests (mediawiki.org/wiki/Common_thumbnail_sizes).
STANDARD_THUMB_WIDTHS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)
THUMB_WIDTH = 1920
# MediaWiki names a thumbnail "<w>px-thumbnail.<ext>" when the file name is longer than this (WMF config).
THUMB_NAME_ABBREVIATION_BYTES = 160
EXIF_PREFIX_BYTES = 128 * 1024
MIN_SHORT_SIDE_PX = 640
MAX_ASPECT = 3.0                 # wider than this is a panorama
MIN_INTERVAL_S = 1.0
MAX_INTERVAL_S = 3.0             # each 429 widens the spacing by INTERVAL_STEP_S, up to this
INTERVAL_STEP_S = 0.25
EXIF_HEADER = b"Exif\x00\x00"
JPEG_EXTS = (".jpg", ".jpeg", ".jpe")
PNG_EXTS = (".png",)
# HTTP answers that will not change on a retry and may be cached; 429/5xx and network errors never are.
CACHEABLE_STATUSES = (200, 206, 400, 404, 410, 416)
RETRY_STATUSES = (429, 500, 502, 503, 504)
NETWORK_ERROR = -1


# ---------------------------------------------------------------- polite, cached HTTP


@dataclass
class Response:
    url: str
    status: int
    headers: dict[str, str]          # lower-case names
    body: bytes
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.status in (200, 206)

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class RateLimited(RuntimeError):
    """A server keeps refusing (429/5xx) or asks for a longer pause than we are willing to wait."""


# (url, request headers, max body bytes or None) -> (status, lower-case headers, body)
Transport = Callable[[str, dict[str, str], int | None], tuple[int, dict[str, str], bytes]]


def urllib_transport(url: str, headers: dict[str, str], max_bytes: int | None) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, hdrs = r.status, r.headers
            # A server that ignores Range sends the whole original; read only what was asked for.
            body = r.read(max_bytes) if max_bytes is not None else r.read()
    except urllib.error.HTTPError as e:
        status, hdrs, body = e.code, e.headers, e.read()
    h = {k.lower(): v for k, v in hdrs.items()}
    if h.get("content-encoding") == "gzip":
        body = gzip.decompress(body)
    return status, h, body


def retry_after_s(value: str | None, now: float | None = None) -> float | None:
    """Seconds from a Retry-After header (delta-seconds or an HTTP date), None when absent or unparsable."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, when - (time.time() if now is None else now))


def cache_key(url: str, byte_range: tuple[int, int] | None) -> str:
    raw = url if byte_range is None else f"{url}#bytes={byte_range[0]}-{byte_range[1]}"
    return hashlib.sha256(raw.encode()).hexdigest()


class PoliteClient:
    """Sequential GETs with a disk cache, >= min_interval_s between network requests and 429/5xx backoff."""

    def __init__(self, cache_dir: Path, *, transport: Transport = urllib_transport,
                 min_interval_s: float = MIN_INTERVAL_S, max_retries: int = 3, max_wait_s: float = 900.0,
                 max_total_wait_s: float = 1800.0, base_backoff_s: float = 5.0, offline: bool = False,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.cache_dir = cache_dir
        self.transport = transport
        self.min_interval_s = max(MIN_INTERVAL_S, min_interval_s)
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.max_total_wait_s = max_total_wait_s
        self.base_backoff_s = base_backoff_s
        self.offline = offline
        self.clock, self.sleep = clock, sleep
        self.stats: Counter[str] = Counter()
        self.by_host: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.waited_s = 0.0           # pauses ordered by servers (Retry-After/backoff), not the spacing
        self.events: list[dict[str, Any]] = []    # every 429/5xx/network error, for the report
        self._last: float | None = None
        self._blocked_until: dict[str, float] = {}

    # -- cache

    def _paths(self, key: str) -> tuple[Path, Path]:
        d = self.cache_dir / key[:2]
        return d / f"{key}.json", d / f"{key}.bin"

    def _load(self, key: str) -> Response | None:
        meta_p, body_p = self._paths(key)
        if not meta_p.exists() or not body_p.exists():
            return None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return Response(meta["url"], int(meta["status"]), meta["headers"], body_p.read_bytes(), True)
        except (OSError, ValueError, KeyError):
            return None

    def _store(self, key: str, resp: Response) -> None:
        meta_p, body_p = self._paths(key)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        for p, data in ((body_p, resp.body),
                        (meta_p, json.dumps({"url": resp.url, "status": resp.status, "headers": resp.headers,
                                             "fetched_at": time.time()}).encode())):
            tmp = p.with_suffix(p.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(p)            # body first, then meta: a half-written entry is never loaded

    # -- network

    def _wait_turn(self, host: str) -> None:
        now = self.clock()
        ready = max(self._blocked_until.get(host, 0.0),
                    (self._last + self.min_interval_s) if self._last is not None else 0.0)
        if ready > now:
            self.sleep(ready - now)

    def get(self, url: str, *, byte_range: tuple[int, int] | None = None, gzip_ok: bool = False,
            max_bytes: int | None = None, max_wait_s: float | None = None) -> Response:
        """GET through the cache. max_wait_s lowers the longest pause this call accepts before RateLimited."""
        key = cache_key(url, byte_range)
        cached = self._load(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        if self.offline:
            self.stats["offline_misses"] += 1
            return Response(url, NETWORK_ERROR, {}, b"")
        headers = {"User-Agent": USER_AGENT}
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
            max_bytes = byte_range[1] - byte_range[0] + 1
        elif gzip_ok:
            headers["Accept-Encoding"] = "gzip"
        host = urllib.parse.urlsplit(url).netloc
        limit = self.max_wait_s if max_wait_s is None else min(max_wait_s, self.max_wait_s)
        blocked = self._blocked_until.get(host, 0.0) - self.clock()
        if max_wait_s is not None and blocked > limit:
            # An earlier answer told us to stay away longer than this caller accepts (auto mode's
            # --original-wait): fail now instead of sleeping through the whole pause first.
            raise RateLimited(f"{host} asked for a pause until {blocked:.0f} s from now (limit {limit:.0f} s); "
                              f"not requesting {url}")
        attempt = 0
        while True:
            self._wait_turn(host)
            try:
                status, h, body = self.transport(url, headers, max_bytes)
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
                status, h, body = NETWORK_ERROR, {"x-error": str(exc)[:200]}, b""
            self._last = self.clock()
            self.stats["requests"] += 1
            self.stats["bytes"] += len(body)
            self.by_host[host] += 1
            self.statuses[str(status)] += 1
            if status not in RETRY_STATUSES and status != NETWORK_ERROR:
                resp = Response(url, status, h, body)
                if status in CACHEABLE_STATUSES:
                    self._store(key, resp)
                return resp
            told = retry_after_s(h.get("retry-after"))
            if told is None:
                wait = self.base_backoff_s * 2 ** attempt
            else:
                # Honour Retry-After; when the same request is refused again anyway (thumb.wikimedia.org kept
                # answering "Retry-After: 1"), back off exponentially on top of it.
                wait = told if attempt == 0 else max(told, self.base_backoff_s * 2 ** (attempt - 1))
            self.events.append({"url": url, "status": status, "retry_after_s": told, "attempt": attempt,
                                "error": h.get("x-error")})
            if status == 429:       # the server says we are too fast: slow down for the rest of the run
                self.min_interval_s = min(MAX_INTERVAL_S, self.min_interval_s + INTERVAL_STEP_S)
            log.warning("HTTP %s from %s (attempt %d, wait %.0f s)", status, host, attempt + 1, wait)
            self._blocked_until[host] = self.clock() + wait
            if attempt >= self.max_retries or wait > limit or self.waited_s + wait > self.max_total_wait_s:
                raise RateLimited(f"HTTP {status} from {host} for {url} (asked to wait {wait:.0f} s, attempt "
                                  f"{attempt + 1}); stopping instead of retrying")
            self.waited_s += wait
            attempt += 1

    def report(self) -> dict[str, Any]:
        return {"network_requests": self.stats["requests"], "cache_hits": self.stats["cache_hits"],
                "bytes_downloaded": self.stats["bytes"], "by_host": dict(self.by_host),
                "status_counts": dict(self.statuses), "http_429": self.statuses.get("429", 0),
                "http_5xx": sum(v for k, v in self.statuses.items() if k.startswith("5")),
                "server_ordered_wait_s": round(self.waited_s, 1), "final_min_interval_s": self.min_interval_s,
                "retry_events": self.events}


# ---------------------------------------------------------------- a minimal HTML tree (stdlib only)

VOID_TAGS = frozenset(("area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
                       "track", "wbr"))


class Node:
    __slots__ = ("attrs", "children", "parent", "tag")

    def __init__(self, tag: str, attrs: dict[str, str], parent: Node | None = None) -> None:
        self.tag, self.attrs, self.parent = tag, attrs, parent
        self.children: list[Node | str] = []

    @property
    def classes(self) -> list[str]:
        return self.attrs.get("class", "").split()

    def iter(self) -> Iterator[Node]:
        stack: list[Node] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(c for c in reversed(node.children) if isinstance(c, Node))

    def find_all(self, tag: str | None = None, cls: str | None = None, id: str | None = None) -> list[Node]:
        return [n for n in self.iter()
                if (tag is None or n.tag == tag) and (cls is None or cls in n.classes)
                and (id is None or n.attrs.get("id") == id)]

    def find(self, tag: str | None = None, cls: str | None = None, id: str | None = None) -> Node | None:
        found = self.find_all(tag, cls, id)
        return found[0] if found else None

    def text(self) -> str:
        parts: list[str] = []

        def walk(n: Node) -> None:
            for c in n.children:
                if isinstance(c, str):
                    parts.append(c)
                elif c.tag not in ("script", "style"):
                    walk(c)
        walk(self)
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    def next_element(self) -> Node | None:
        if self.parent is None:
            return None
        sibs = [c for c in self.parent.children if isinstance(c, Node)]
        i = sibs.index(self)
        return sibs[i + 1] if i + 1 < len(sibs) else None


class _TreeBuilder(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)   # decodes &#95; in class names and &amp; in hrefs
        self.root = Node("#root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, {k: v or "" for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.stack[-1].children.append(Node(tag, {k: v or "" for k, v in attrs}, self.stack[-1]))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def parse_html(text: str) -> Node:
    b = _TreeBuilder()
    b.feed(text)
    b.close()
    return b.root


# ---------------------------------------------------------------- Commons names and URLs


def normalize_title(title: str) -> str:
    """'File:Some name.jpg' as MediaWiki displays it (spaces, first letter of the name upper-case)."""
    t = urllib.parse.unquote(title).replace("_", " ").strip()
    if t.lower().startswith("file:"):
        t = t[5:]
    elif t.lower().startswith("image:"):
        t = t[6:]
    t = re.sub(r"\s+", " ", t).strip()
    return "File:" + (t[:1].upper() + t[1:])


def file_name(title: str) -> str:
    """Storage name of a file: no namespace, underscores for spaces."""
    return normalize_title(title)[5:].replace(" ", "_")


def _quote(name: str) -> str:
    return urllib.parse.quote(name, safe="")


# Characters MediaWiki leaves unescaped in /wiki/ URLs (wfUrlencode): requesting exactly the canonical URL
# hits the CDN cache.
WIKI_URL_SAFE = ";@$!*(),/~:"


def page_url(title: str) -> str:
    """Canonical /wiki/<Title> URL: the CDN-cached form the robot policy asks crawlers to use."""
    return f"{COMMONS}/wiki/{urllib.parse.quote(title.replace(' ', '_'), safe=WIKI_URL_SAFE)}"


def upload_url(title: str) -> str:
    name = file_name(title)
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"{UPLOAD_BASE}/{h[0]}/{h[:2]}/{_quote(name)}"


def thumb_url(title: str, width: int = THUMB_WIDTH) -> str:
    if width not in STANDARD_THUMB_WIDTHS:
        raise ValueError(f"{width} px is not a standard Wikimedia thumbnail width")
    name = file_name(title)
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    ext = Path(name).suffix
    thumb_name = f"thumbnail{ext}" if len(name.encode("utf-8")) > THUMB_NAME_ABBREVIATION_BYTES else name
    return f"{THUMB_BASE}/{h[0]}/{h[:2]}/{_quote(name)}/{width}px-{_quote(thumb_name)}"


def local_name(title: str, ext: str) -> str:
    name = file_name(title)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem).strip("-.")[:60] or "file"
    return f"{slug}-{hashlib.md5(name.encode('utf-8')).hexdigest()[:8]}{ext}"


# ---------------------------------------------------------------- page parsers


@dataclass
class Listing:
    title: str                        # normalized "File:..."
    width: int | None = None          # as Commons lists it (upright, i.e. after EXIF rotation)
    height: int | None = None


def _int_attr(node: Node | None, name: str) -> int | None:
    try:
        return int(node.attrs[name]) if node is not None else None
    except (KeyError, ValueError):
        return None


def parse_dims(text: str) -> tuple[int, int] | None:
    m = re.search(r"([\d,]+)\s*[×x]\s*([\d,]+)", text)
    if not m:
        return None
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


def parse_search_page(text: str) -> tuple[list[Listing], int | None, int | None]:
    """Special:Search result page -> (files, reported total, next offset or None)."""
    root = parse_html(text)
    items: list[Listing] = []
    for li in root.find_all("li", cls="mw-search-result"):
        head = li.find("div", cls="mw-search-result-heading")
        a = head.find("a") if head else None
        if a is None or not a.attrs.get("title", "").startswith("File:"):
            continue
        img = li.find("img")
        w, h = _int_attr(img, "data-file-width"), _int_attr(img, "data-file-height")
        if w is None:
            data = li.find("div", cls="mw-search-result-data")
            dims = parse_dims(data.text()) if data else None
            w, h = dims if dims else (None, None)
        items.append(Listing(normalize_title(a.attrs["title"]), w, h))
    total = None
    for n in root.iter():
        if "data-mw-num-results-total" in n.attrs:
            total = _int_attr(n, "data-mw-num-results-total")
            break
    next_offset = None
    for a in root.find_all("a"):
        if a.attrs.get("rel") == "next" and a.attrs.get("href"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(a.attrs["href"]).query)
            if q.get("offset"):
                next_offset = int(q["offset"][0])
            break
    return items, total, next_offset


def parse_category_page(text: str) -> tuple[list[Listing], list[str], bool]:
    """Category page -> (files on this page, subcategory names, whether more files follow on later pages)."""
    root = parse_html(text)
    subcats: list[str] = []
    sub = root.find(id="mw-subcategories")
    if sub is not None:
        for a in sub.find_all("a"):
            href = a.attrs.get("href", "")
            if href.startswith("/wiki/Category:"):
                name = a.attrs.get("title") or urllib.parse.unquote(href[len("/wiki/"):]).replace("_", " ")
                name = name[len("Category:"):]
                if name not in subcats:
                    subcats.append(name)
    files: list[Listing] = []
    more = False
    media = root.find(id="mw-category-media")
    if media is not None:
        for box in media.find_all("li", cls="gallerybox"):
            a = box.find("a", cls="galleryfilename") or box.find("a", cls="mw-file-description")
            if a is None:
                continue
            title = a.attrs.get("title") or urllib.parse.unquote(a.attrs.get("href", "")[len("/wiki/"):])
            if not title.startswith("File:"):
                continue
            img = box.find("img")
            files.append(Listing(normalize_title(title), _int_attr(img, "data-file-width"),
                                 _int_attr(img, "data-file-height")))
        more = any("filefrom=" in a.attrs.get("href", "") for a in media.find_all("a"))
    return files, subcats, more


@dataclass
class FilePage:
    title: str | None = None
    author: str | None = None
    author_url: str | None = None
    credit: str | None = None                 # attribution text a licence template asks for, when given
    licences: list[dict[str, str]] = field(default_factory=list)
    camera_location: dict[str, float] | None = None
    object_location: dict[str, float] | None = None
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)   # MediaWiki's rendering of the original's EXIF
    original_url: str | None = None
    width: int | None = None                  # upright size of the original
    height: int | None = None
    mime: str | None = None


def _absolute(href: str) -> str:
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return COMMONS + href
    return href


def _location(row: Node) -> dict[str, float] | None:
    geo = row.find("span", cls="geo")
    if geo is None:
        return None
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*[;,]\s*(-?\d+(?:\.\d+)?)", geo.text())
    if not m:
        return None
    loc: dict[str, float] = {"lat": float(m.group(1)), "lon": float(m.group(2))}
    for a in row.find_all("a"):
        h = re.search(r"heading:(-?\d+(?:\.\d+)?)", urllib.parse.unquote(a.attrs.get("href", "")))
        if h:
            loc["heading_deg"] = float(h.group(1)) % 360.0
            break
    return loc


def parse_file_page(text: str) -> FilePage:
    root = parse_html(text)
    page = FilePage()
    h1 = root.find("h1", id="firstHeading")
    if h1 is not None:
        page.title = normalize_title(h1.text())

    aut = root.find(id="fileinfotpl_aut")
    cell = aut.next_element() if aut is not None else None
    if cell is not None:
        page.author = cell.text()[:300] or None
        link = cell.find("a")
        if link is not None and link.attrs.get("href"):
            page.author_url = _absolute(link.attrs["href"])

    seen: set[tuple[tuple[str, str], ...]] = set()
    for box in root.find_all(cls="licensetpl"):
        lic: dict[str, str] = {}
        for span in box.find_all("span"):
            for c in span.classes:
                if c.startswith("licensetpl_") and c[len("licensetpl_"):] not in lic:
                    lic[c[len("licensetpl_"):]] = span.text()
        key = tuple(sorted(lic.items()))
        if lic.get("short") and key not in seen:
            seen.add(key)
            page.licences.append(lic)
    if not page.licences:
        page.licences = [{"short": t} for t in dict.fromkeys(
            span.text() for span in root.find_all("span", cls="licensetpl_short") if span.text())]
    for lic in page.licences:
        if lic.get("attr") and not page.credit:
            page.credit = lic["attr"]
        if lic.get("aut") and not page.author:
            page.author = lic["aut"]

    for th in root.find_all("th"):
        label = th.text().lower()
        if label.startswith(("camera location", "object location")) and th.parent is not None:
            loc = _location(th.parent)
            if label.startswith("camera"):
                page.camera_location = page.camera_location or loc
            else:
                page.object_location = page.object_location or loc

    cats = root.find(id="mw-normal-catlinks")
    if cats is not None:
        page.categories = [a.text() for li in cats.find_all("li") for a in li.find_all("a")[:1]]

    table = root.find(id="mw_metadata")
    if table is not None:
        for tr in table.find_all("tr"):
            key = next((c for c in tr.classes if c.startswith("exif-")), None)
            td = tr.find("td")
            if key and td is not None:
                page.metadata[key] = td.text()

    full = root.find(cls="fullMedia")
    if full is not None:
        for a in full.find_all("a"):
            href = a.attrs.get("href", "")
            if "upload.wikimedia.org" in href:
                page.original_url = _absolute(href.split("?")[0])
                break
        info = full.text()
        dims = parse_dims(info)
        if dims:
            page.width, page.height = dims
        mime = re.search(r"MIME type:\s*([\w.+-]+/[\w.+-]+)", info)
        page.mime = mime.group(1) if mime else None
    return page


_DMS = re.compile(r"(\d+(?:\.\d+)?)\s*°\s*(?:(\d+(?:\.\d+)?)\s*′)?\s*(?:(\d+(?:\.\d+)?)\s*″)?\s*([NSEW])")


def parse_dms(text: str) -> float | None:
    """'43° 46′ 23.56″ N' (MediaWiki's EXIF rendering) -> 43.773211; plain decimal degrees pass through."""
    m = _DMS.search(text or "")
    if not m:
        d = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*°?\s*([NSEW])?\s*", text or "")
        if not d:
            return None
        return -abs(float(d.group(1))) if d.group(2) in ("S", "W") else float(d.group(1))
    deg = float(m.group(1)) + float(m.group(2) or 0) / 60.0 + float(m.group(3) or 0) / 3600.0
    return -deg if m.group(4) in "SW" else deg


def page_gps(page: FilePage) -> tuple[float, float] | None:
    lat = parse_dms(page.metadata.get("exif-gpslatitude", ""))
    lon = parse_dms(page.metadata.get("exif-gpslongitude", ""))
    return (lat, lon) if lat is not None and lon is not None else None


MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"), 1)}


def _number(text: str | None) -> float | None:
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    return float(m.group(0).replace(",", "")) if m else None


def page_datetime(text: str | None) -> str | None:
    """'12:03, 31 August 2022' (MediaWiki's rendering, seconds dropped) -> '2022:08:31 12:03:00'."""
    m = re.search(r"(?:(\d{1,2}):(\d{2})(?::(\d{2}))?,\s*)?(\d{1,2}) ([A-Za-z]+) (\d{4})", text or "")
    if not m or m.group(5).lower() not in MONTHS:
        return None
    hh, mm, ss = (int(g) if g else 0 for g in m.group(1, 2, 3))
    return f"{int(m.group(6)):04d}:{MONTHS[m.group(5).lower()]:02d}:{int(m.group(4)):02d} {hh:02d}:{mm:02d}:{ss:02d}"


def exif_from_page(page: FilePage) -> bytes | None:
    """The EXIF fields the product reads, re-encoded from MediaWiki's rendering of the file's own EXIF.

    Fallback for when the original cannot be range-read. The values are the camera's, rounded as MediaWiki
    shows them: GPS to 0.01 arc-second (0.3 m), DateTimeOriginal to the minute; MakerNote and every other
    tag are lost. None when the page shows no parsable GPS.
    """
    gps = page_gps(page)
    if gps is None:
        return None
    md = page.metadata
    alt = _number(md.get("exif-gpsaltitude"))
    if alt is not None and "below" in md.get("exif-gpsaltitude", ""):
        alt = -abs(alt)
    accuracy = _number(md.get("exif-gpshpositioningerror"))
    return build_exif_bytes(
        lat=gps[0], lon=gps[1], alt=alt, gps_accuracy_m=accuracy if accuracy and accuracy > 0 else None,
        heading_deg=_number(md.get("exif-gpsimgdirection")),
        heading_ref="M" if md.get("exif-gpsimgdirectionref", "").lower().startswith("magnetic") else "T",
        focal_mm=_number(md.get("exif-focallength")), focal_35mm=_number(md.get("exif-focallengthin35mmfilm")),
        captured_at=page_datetime(md.get("exif-datetimeoriginal")),
        make=md.get("exif-make") or None, model=md.get("exif-model") or None)


# ---------------------------------------------------------------- EXIF bytes: scan, parse, graft

SOF_MARKERS = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))
MAX_SEGMENT = 65535 + 4


@dataclass
class ExifScan:
    app1: bytes | None = None                 # the EXIF payload, starting with b"Exif\0\0"
    size: tuple[int, int] | None = None       # stored (raw) pixel size from SOF / IHDR
    need: int | None = None                   # prefix length that would let the scan finish


def scan_jpeg(data: bytes) -> ExifScan:
    """Walk JPEG marker segments up to the first scan, collecting the EXIF APP1 payload and the SOF size."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    out = ExifScan()
    i = 2
    while True:
        if i + 4 > len(data):
            out.need = i + MAX_SEGMENT
            return out
        if data[i] != 0xFF:
            raise ValueError(f"corrupt JPEG marker at byte {i}")
        marker = data[i + 1]
        if marker == 0xFF:                    # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            i += 2
            continue
        if marker in (0xD9, 0xDA):            # end of image / start of scan: no more headers
            return out
        end = i + 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xE1 and out.app1 is None:
            if end > len(data):
                if len(data) < i + 10 or data[i + 4:i + 10] == EXIF_HEADER:
                    out.need = end
                    return out
            elif data[i + 4:i + 10] == EXIF_HEADER:
                out.app1 = data[i + 4:end]
        elif marker in SOF_MARKERS and i + 9 <= len(data):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            out.size = (w, h)
        if out.app1 is not None and out.size is not None:
            return out
        i = end


def scan_png(data: bytes) -> ExifScan:
    """Walk PNG chunks up to the first IDAT: IHDR size and an eXIf chunk (as an APP1-style payload)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    out = ExifScan()
    i = 8
    while True:
        if i + 8 > len(data):
            out.need = i + MAX_SEGMENT
            return out
        length, ctype = struct.unpack(">I4s", data[i:i + 8])
        end = i + 12 + length
        if ctype == b"IHDR" and i + 16 <= len(data):
            out.size = struct.unpack(">II", data[i + 8:i + 16])
        elif ctype == b"eXIf":
            if end > len(data):
                out.need = end
                return out
            out.app1 = EXIF_HEADER + data[i + 8:i + 8 + length]
            return out
        elif ctype in (b"IDAT", b"IEND"):
            return out
        i = end


def _tiny_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (16, 16), 128).save(buf, "JPEG")
    return buf.getvalue()


def meta_from_app1(app1: bytes) -> PhotoMeta:
    """Parse an EXIF payload with the product's own reader (read_photo_meta) by putting it into a tiny JPEG.

    width/height of the result are the tiny image's; callers replace them.
    """
    return read_photo_meta(set_exif_segment(_tiny_jpeg(), app1))


def set_exif_segment(jpeg: bytes, app1: bytes) -> bytes:
    """JPEG bytes with every EXIF APP1 segment replaced by one carrying app1 (after a JFIF APP0, if any)."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    if not app1.startswith(EXIF_HEADER) or len(app1) + 2 > 0xFFFF:
        raise ValueError("bad EXIF payload")
    kept: list[bytes] = []
    i = 2
    while True:
        if i + 4 > len(jpeg) or jpeg[i] != 0xFF:
            raise ValueError("truncated or corrupt JPEG header")
        marker = jpeg[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD9, 0xDA):
            break
        end = i + 2 + struct.unpack(">H", jpeg[i + 2:i + 4])[0]
        if not (marker == 0xE1 and jpeg[i + 4:i + 10] == EXIF_HEADER):
            kept.append(jpeg[i:end])
        i = end
    segment = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    at = 1 if kept and kept[0][:2] == b"\xff\xe0" else 0
    kept.insert(at, segment)
    return b"\xff\xd8" + b"".join(kept) + jpeg[i:]


def patch_exif(app1: bytes, *, width: int, height: int, orientation: int | None = 1) -> bytes:
    """Copy of an EXIF payload with Orientation and the pixel-dimension tags overwritten in place.

    Same length and every other byte as the camera wrote it (MakerNote offsets stay valid). Tags that are
    absent stay absent. orientation=None leaves Orientation alone.
    """
    if not app1.startswith(EXIF_HEADER):
        raise ValueError("bad EXIF payload")
    buf = bytearray(app1)
    base = len(EXIF_HEADER)
    order = bytes(buf[base:base + 2])
    if order not in (b"II", b"MM"):
        raise ValueError("bad TIFF byte order")
    e = "<" if order == b"II" else ">"

    def u16(off: int) -> int:
        return struct.unpack_from(e + "H", buf, base + off)[0]

    def u32(off: int) -> int:
        return struct.unpack_from(e + "I", buf, base + off)[0]

    def entries(ifd: int) -> Iterator[tuple[int, int, int, int]]:
        if ifd <= 0 or base + ifd + 2 > len(buf):
            raise ValueError("IFD offset out of range")
        n = u16(ifd)
        if base + ifd + 2 + 12 * n > len(buf):
            raise ValueError("IFD runs past the payload")
        for k in range(n):
            off = ifd + 2 + 12 * k
            tag, typ, count = struct.unpack_from(e + "HHI", buf, base + off)
            yield off, tag, typ, count

    def put(off: int, typ: int, count: int, value: int) -> None:
        if count != 1:
            return
        if typ == 3 and value <= 0xFFFF:
            struct.pack_into(e + "H", buf, base + off + 8, value)
        elif typ == 4:
            struct.pack_into(e + "I", buf, base + off + 8, value)

    if u16(2) != 42:
        raise ValueError("bad TIFF magic")
    exif_ifd = None
    for off, tag, typ, count in entries(u32(4)):
        if tag == 0x0112 and orientation is not None:
            put(off, typ, count, orientation)
        elif tag == 0x0100:
            put(off, typ, count, width)
        elif tag == 0x0101:
            put(off, typ, count, height)
        elif tag == 0x8769 and count == 1:
            exif_ifd = u32(off + 8)
    if exif_ifd:
        for off, tag, typ, count in entries(exif_ifd):
            if tag == 0xA002:
                put(off, typ, count, width)
            elif tag == 0xA003:
                put(off, typ, count, height)
    return bytes(buf)


SURVIVING_FIELDS = ("lat", "lon", "alt", "gps_accuracy_m", "heading_deg", "heading_ref", "focal_mm", "focal_35mm",
                    "captured_at", "make", "model")


def lost_fields(before: PhotoMeta, after: PhotoMeta) -> list[str]:
    """EXIF fields the product reads that differ between the original and the delivered file."""
    bad = []
    for name in SURVIVING_FIELDS:
        a, b = getattr(before, name), getattr(after, name)
        if isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9):
                bad.append(name)
        elif a != b:
            bad.append(name)
    return bad


# ---------------------------------------------------------------- candidates


@dataclass
class Candidate:
    title: str
    width: int | None = None
    height: int | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def source_kinds(self) -> set[str]:
        return {s.split(":", 1)[0] for s in self.sources}


def merge_candidates(groups: list[tuple[str, list[Listing]]]) -> list[Candidate]:
    """De-duplicate by title. Files named by more kinds of source (category, titles, search) come first."""
    by_title: dict[str, Candidate] = {}
    for source, listings in groups:
        for item in listings:
            c = by_title.setdefault(item.title, Candidate(item.title))
            if c.width is None and item.width is not None:
                c.width, c.height = item.width, item.height
            if source not in c.sources:
                c.sources.append(source)
    order = {t: i for i, t in enumerate(by_title)}
    return sorted(by_title.values(), key=lambda c: (-len(c.source_kinds), order[c.title]))


def walk_categories(client: PoliteClient, roots: list[str], depth: int, max_files: int,
                    exclude: re.Pattern[str] | None) -> tuple[list[tuple[str, list[Listing]]], dict[str, Any]]:
    """Breadth-first over categories (first page of each: later pages need robots-disallowed /w/ URLs)."""
    queue = [(r.removeprefix("Category:").replace("_", " "), 0) for r in roots]
    seen: set[str] = set()
    groups: list[tuple[str, list[Listing]]] = []
    report: dict[str, Any] = {"visited": [], "excluded": [], "truncated": [], "missing": []}
    n_files = 0
    while queue and n_files < max_files:
        name, d = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        resp = client.get(page_url("Category:" + name), gzip_ok=True)
        if not resp.ok:
            report["missing"].append({"category": name, "status": resp.status})
            continue
        files, subcats, more = parse_category_page(resp.text())
        report["visited"].append({"category": name, "depth": d, "files": len(files), "subcategories": len(subcats)})
        if more:
            report["truncated"].append(name)
        groups.append((f"category:{name}", files[:max_files - n_files]))
        n_files += len(files)
        if d < depth:
            for s in subcats:
                if exclude is not None and exclude.search(s):
                    report["excluded"].append(s)
                elif s not in seen:
                    queue.append((s, d + 1))
    return groups, report


def read_titles(path: Path) -> list[Listing]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(Listing(normalize_title(line)))
    return out


# ---------------------------------------------------------------- screening and pixels


@dataclass
class Config:
    name: str
    lat: float
    lon: float
    radius_m: float
    keep_radius_m: float
    out: Path
    max_photos: int | None = None
    page_prescreen: bool = True
    prescreen_audit: int = 0
    exif_source: str = "auto"       # "original" | "page" | "auto" (original, page once originals are refused)
    original_wait_s: float = 60.0   # auto: a longer Retry-After on an original switches to page EXIF
    settings: Settings = field(default_factory=Settings)


class Skip(Exception):
    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason, self.detail = reason, detail


@dataclass
class Screened:
    page: FilePage
    meta: PhotoMeta                 # from the original's EXIF, with the original's stored size
    app1: bytes
    kind: str                       # "jpeg" | "png"
    upright: tuple[int, int]        # size as displayed (after EXIF rotation)
    total_bytes: int | None         # size of the original file, when the server said
    exif_source: str = "original"   # or "commons_page_metadata" (see exif_from_page)


def file_kind(title: str) -> str | None:
    ext = Path(title).suffix.lower()
    return "jpeg" if ext in JPEG_EXTS else "png" if ext in PNG_EXTS else None


def check_shape(width: int | None, height: int | None) -> None:
    if not width or not height:
        return
    if max(width, height) / min(width, height) > MAX_ASPECT:
        raise Skip("panorama", f"{width}x{height}")
    if min(width, height) < MIN_SHORT_SIDE_PX:
        raise Skip("too_small_original", f"{width}x{height}")


def total_size(resp: Response) -> int | None:
    m = re.search(r"/(\d+)\s*$", resp.headers.get("content-range", ""))
    if m:
        return int(m.group(1))
    if resp.status == 200 and resp.headers.get("content-length", "").isdigit():
        return int(resp.headers["content-length"])
    return None


def read_original_exif(client: PoliteClient, title: str, kind: str,
                       max_wait_s: float | None = None) -> tuple[ExifScan, int | None]:
    url = upload_url(title)
    resp = client.get(url, byte_range=(0, EXIF_PREFIX_BYTES - 1), max_wait_s=max_wait_s)
    if not resp.ok:
        raise Skip("original_unavailable", f"HTTP {resp.status}")
    data = resp.body
    scan = scan_jpeg if kind == "jpeg" else scan_png
    try:
        result = scan(data)
        if result.need is not None and resp.status == 206 and len(data) == EXIF_PREFIX_BYTES:
            # One extension: up to the end of the EXIF segment (or one more maximal segment).
            more = client.get(url, byte_range=(len(data), result.need - 1), max_wait_s=max_wait_s)
            if more.status == 206:
                data += more.body
                result = scan(data)
    except (ValueError, struct.error) as exc:
        raise Skip("corrupt_header", str(exc)) from exc
    if result.need is not None and result.app1 is None:
        raise Skip("exif_beyond_prefix", f"needs {result.need} bytes")
    return result, total_size(resp)


def screen_page(client: PoliteClient, cand: Candidate, cfg: Config) -> tuple[str, FilePage]:
    """Phase 1, Commons HTML only: format, size and (prescreen) whether the page shows EXIF GPS nearby."""
    kind = file_kind(cand.title)
    if kind is None:
        raise Skip("unsupported_format", Path(cand.title).suffix.lower() or "(none)")
    check_shape(cand.width, cand.height)
    resp = client.get(page_url(cand.title), gzip_ok=True)
    if not resp.ok:
        raise Skip("page_missing", f"HTTP {resp.status}")
    page = parse_file_page(resp.text())
    if page.mime and page.mime not in ("image/jpeg", "image/png"):
        raise Skip("unsupported_format", page.mime)
    check_shape(page.width, page.height)
    if cfg.page_prescreen:
        if not page.metadata:
            raise Skip("page_no_exif")
        if "exif-gpslatitude" not in page.metadata or "exif-gpslongitude" not in page.metadata:
            raise Skip("page_no_gps")
        gps = page_gps(page)
        if gps is not None and local_bearing_distance(cfg.lat, cfg.lon, *gps)[1] > cfg.keep_radius_m:
            raise Skip("gps_outside_radius", "from the page's EXIF rendering")
    return kind, page


def screen_original(client: PoliteClient, cand: Candidate, kind: str, page: FilePage, cfg: Config) -> Screened:
    """Phase 2, first step: the original's own EXIF (Range request), parsed by the product's reader."""
    scan, total = read_original_exif(client, cand.title, kind, _original_wait(cfg))
    if scan.app1 is None:
        raise Skip("no_exif")
    try:
        meta = meta_from_app1(scan.app1)
    except UnreadableImage as exc:
        raise Skip("exif_unreadable", str(exc)) from exc
    if not meta.has_exif:
        raise Skip("exif_unreadable")
    if not meta.has_gps:
        raise Skip("no_gps")
    if local_bearing_distance(cfg.lat, cfg.lon, meta.lat, meta.lon)[1] > cfg.keep_radius_m:
        raise Skip("gps_outside_radius")

    stored = scan.size or (page.width, page.height)
    if page.width and page.height:
        upright = (page.width, page.height)
    elif stored[0] and meta.orientation in (5, 6, 7, 8):
        upright = (stored[1], stored[0])
    else:
        upright = stored
    if not upright[0] or not upright[1]:
        raise Skip("unknown_size")
    check_shape(*upright)
    meta = replace(meta, width=stored[0] or upright[0], height=stored[1] or upright[1],
                   format="PNG" if kind == "png" else "JPEG")
    return Screened(page, meta, scan.app1, kind, upright, total)


def _original_wait(cfg: Config) -> float | None:
    return cfg.original_wait_s if cfg.exif_source == "auto" else None


def screen_from_page(cand: Candidate, kind: str, page: FilePage, cfg: Config) -> Screened:
    """Phase 2 without the original: EXIF values from the description page (exif_from_page)."""
    app1 = exif_from_page(page)
    if app1 is None:
        raise Skip("page_gps_unparsed", page.metadata.get("exif-gpslatitude"))
    meta = meta_from_app1(app1)
    if local_bearing_distance(cfg.lat, cfg.lon, meta.lat, meta.lon)[1] > cfg.keep_radius_m:
        raise Skip("gps_outside_radius")
    if not page.width or not page.height:
        raise Skip("unknown_size")
    meta = replace(meta, width=page.width, height=page.height, format="PNG" if kind == "png" else "JPEG")
    return Screened(page, meta, app1, kind, (page.width, page.height), None, "commons_page_metadata")


def pick_thumb_width(width: int, height: int) -> int | None:
    """Largest standard thumbnail width up to THUMB_WIDTH, below the original's, with a short side >= 640 px."""
    for tw in sorted(STANDARD_THUMB_WIDTHS, reverse=True):
        if tw <= THUMB_WIDTH and tw < width and min(tw, tw * height / width) >= MIN_SHORT_SIDE_PX:
            return tw
    return None


def fetch_pixels(client: PoliteClient, cand: Candidate, s: Screened, cfg: Config,
                 originals_ok: bool = True) -> tuple[bytes, dict[str, Any]]:
    """The delivered file: the original when it is at most THUMB_WIDTH wide (and reachable), else a standard
    Commons thumbnail carrying the EXIF in s.app1."""
    info: dict[str, Any] = {"exif_source": s.exif_source}
    w, h = s.upright
    if originals_ok and (s.kind == "png" or w <= THUMB_WIDTH) and s.exif_source == "original" and not (
            s.total_bytes is not None and s.total_bytes > cfg.settings.max_upload_bytes):
        try:
            resp = client.get(upload_url(cand.title), max_wait_s=_original_wait(cfg))
        except RateLimited:
            if cfg.exif_source == "original":
                raise
            resp = None
        if resp is not None and resp.ok:
            info.update(source="original", url=upload_url(cand.title), orientation_reset=False)
            return resp.body, info
        info["original_refused"] = resp.status if resp is not None else 429
    if s.kind != "jpeg":
        raise Skip("pixels_unavailable", "PNG original not reachable")
    tw = pick_thumb_width(w, h)
    if tw is None:
        raise Skip("no_usable_thumbnail", f"{w}x{h}")
    url = thumb_url(cand.title, tw)
    resp = client.get(url)
    if not resp.ok or resp.body[:2] != b"\xff\xd8":
        raise Skip("thumbnail_refused", f"HTTP {resp.status}")
    with Image.open(io.BytesIO(resp.body)) as im:
        tw_px, th_px = im.size
    # Commons renders thumbnails upright. If one ever is not, keep the camera's Orientation tag.
    raw = s.meta.width / s.meta.height
    rotated = s.meta.orientation in (5, 6, 7, 8) and abs(tw_px / th_px - raw) < abs(tw_px / th_px - w / h)
    app1 = patch_exif(s.app1, width=tw_px, height=th_px, orientation=None if rotated else 1)
    info.update(source=f"thumbnail_{tw}", url=url, orientation_reset=not rotated)
    return set_exif_segment(resp.body, app1), info


# ---------------------------------------------------------------- the run


def landmark_geometry(cfg: Config, lat: float, lon: float, heading: float | None) -> dict[str, Any]:
    bearing, dist = local_bearing_distance(cfg.lat, cfg.lon, lat, lon)
    sector = sector_for_bearing(bearing, cfg.settings.num_sectors)
    out = {"bearing_deg": round(bearing, 2), "distance_m": round(dist, 2), "sector": sector,
           "sector_label": sector_label(sector, cfg.settings.num_sectors)}
    if heading is not None:
        # 0 = camera looks straight at the landmark centre (same convention as crowd3d.validation)
        out["heading_offset_deg"] = round(angle_diff_deg(heading, bearing + 180.0), 2)
    return out


def _round_meta(meta: PhotoMeta) -> dict[str, Any]:
    d = asdict(meta)
    for k in ("lat", "lon"):
        if d[k] is not None:
            d[k] = round(d[k], 7)
    for k in ("alt", "gps_accuracy_m", "heading_deg", "focal_mm", "focal_35mm"):
        if d[k] is not None:
            d[k] = round(d[k], 3)
    return d


def manifest_entry(cand: Candidate, s: Screened, got: PhotoMeta, data: bytes, pix: dict[str, Any], rel: Path,
                   verdict: Any, cfg: Config) -> dict[str, Any]:
    page = s.page
    cam = page.camera_location
    return {
        "title": cand.title,
        "file_name": cand.title[5:],
        "page_url": page_url(cand.title),
        "original_url": upload_url(cand.title),
        "pixels": {**pix, "file": rel.as_posix(), "width": got.width, "height": got.height,
                   "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
        "original": {"width": s.upright[0], "height": s.upright[1], "mime": page.mime, "bytes": s.total_bytes,
                     # pixel size as stored (before EXIF rotation): only known from the original's header
                     "stored_width": s.meta.width if s.exif_source == "original" else None,
                     "stored_height": s.meta.height if s.exif_source == "original" else None},
        "author": page.author,
        "author_url": page.author_url,
        "credit": page.credit,
        "licence": page.licences[0].get("short") if page.licences else None,
        "licences": page.licences,
        "exif": _round_meta(s.meta),
        "exif_source": s.exif_source,
        "page_coordinates": {"camera": cam, "object": page.object_location},
        "exif_to_page_camera_m": (round(local_bearing_distance(got.lat, got.lon, cam["lat"], cam["lon"])[1], 2)
                                  if cam else None),
        "landmark_to_camera": landmark_geometry(cfg, got.lat, got.lon, got.heading_deg),
        "categories": page.categories,
        "sources": cand.sources,
        "in_category": "category" in cand.source_kinds,
        "validation": {"accepted": verdict.accepted, "reasons": verdict.reasons, "warnings": verdict.warnings},
    }


def run(client: PoliteClient, candidates: list[Candidate], cfg: Config) -> dict[str, Any]:
    """Phase 1 (Commons pages) over every candidate, then phase 2 (media host) over the survivors.

    The phases are separate so that a media host that rate-limits us still leaves a page-level funnel,
    and a rerun resumes from the cache.
    """
    (cfg.out / "photos").mkdir(parents=True, exist_ok=True)
    landmark = Landmark(id="commons", name=cfg.name, lat=cfg.lat, lon=cfg.lon, radius_m=cfg.radius_m, created_at=0.0)
    skips: Counter[str] = Counter()
    rows: dict[str, dict[str, Any]] = {c.title: {"title": c.title, "sources": c.sources} for c in candidates}
    aborted = None

    def skip(cand: Candidate, sk: Skip, n: int, total: int) -> None:
        skips[sk.reason] += 1
        rows[cand.title].update(outcome=sk.reason, detail=sk.detail)
        log.info("[%d/%d] skip %-22s %s", n, total, sk.reason, cand.title)

    passed: list[tuple[Candidate, str, FilePage]] = []
    audit_pool: list[Candidate] = []
    for n, cand in enumerate(candidates, 1):
        try:
            kind, page = screen_page(client, cand, cfg)
            passed.append((cand, kind, page))
        except Skip as sk:
            skip(cand, sk, n, len(candidates))
            if sk.reason in ("page_no_gps", "page_no_exif"):
                audit_pool.append(cand)
        except RateLimited as exc:
            aborted = str(exc)
            break
    log.info("phase 1: %d of %d candidates pass the page screen", len(passed), len(candidates))

    kept: list[dict[str, Any]] = []
    hashes: list[str] = []
    fingerprints: dict[tuple[Any, ...], str] = {}
    with_gps = 0
    originals_ok = cfg.exif_source != "page"
    fallback = None
    for n, (cand, kind, page) in enumerate(passed if aborted is None else [], 1):
        if cfg.max_photos is not None and len(kept) >= cfg.max_photos:
            break
        try:
            s = None
            if originals_ok:
                try:
                    s = screen_original(client, cand, kind, page, cfg)
                except RateLimited as exc:
                    if cfg.exif_source == "original":
                        raise
                    originals_ok, fallback = False, str(exc)
                    log.warning("originals refused (%s): EXIF from the description pages from now on", exc)
                except Skip as sk:
                    # One original not served (or not cached in --offline): page EXIF for this file only.
                    if sk.reason != "original_unavailable" or cfg.exif_source == "original":
                        raise
            if s is None:
                s = screen_from_page(cand, kind, page, cfg)
            with_gps += 1
            # Crops and re-uploads of one shot carry its EXIF unchanged; dHash misses crops.
            fp = (round(s.meta.lat, 6), round(s.meta.lon, 6), s.meta.captured_at, s.meta.heading_deg)
            if s.meta.captured_at and fp in fingerprints:
                raise Skip("duplicate_exif", fingerprints[fp])
            data, pix = fetch_pixels(client, cand, s, cfg, originals_ok)
            try:
                got = read_photo_meta(data)
            except UnreadableImage as exc:
                raise Skip("pixels_unreadable", str(exc)) from exc
            lost = lost_fields(s.meta, got)
            if lost:
                raise Skip("exif_lost_in_graft", ",".join(lost))
        except Skip as sk:
            skip(cand, sk, n, len(passed))
            continue
        except RateLimited as exc:
            aborted = str(exc)
            break
        rel = Path("photos") / local_name(cand.title, ".png" if got.format == "PNG" else ".jpg")
        (cfg.out / rel).write_bytes(data)
        verdict = validate_photo(data, got, landmark, hashes, cfg.settings)
        if verdict.phash:
            hashes.append(verdict.phash)
        entry = manifest_entry(cand, s, got, data, pix, rel, verdict, cfg)
        kept.append(entry)
        fingerprints[fp] = cand.title
        rows[cand.title].update(outcome="kept", file=rel.as_posix())
        geo = entry["landmark_to_camera"]
        log.info("[%d/%d] KEEP %s (%.0f m, %s, heading %s)", n, len(passed), cand.title, geo["distance_m"],
                 geo["sector_label"], got.heading_deg)
    if aborted:
        log.error("%s", aborted)
    for row in rows.values():
        row.setdefault("outcome", "not_reached")
    audit = None
    if cfg.prescreen_audit and audit_pool and aborted is None:
        audit = (audit_prescreen(client, audit_pool[:cfg.prescreen_audit], _original_wait(cfg)) if originals_ok
                 else {"skipped": "originals refused", "checked": 0})
    return {"kept": kept, "skips": skips, "page_passed": len(passed), "with_gps": with_gps,
            "log": list(rows.values()), "aborted": aborted, "audit": audit, "originals_fallback": fallback}


def audit_prescreen(client: PoliteClient, pool: list[Candidate], max_wait_s: float | None = None) -> dict[str, Any]:
    """Range-check files the page prescreen dropped: does their own EXIF have GPS after all?"""
    rows = []
    for cand in pool:
        kind = file_kind(cand.title) or "jpeg"
        try:
            scan, _ = read_original_exif(client, cand.title, kind, max_wait_s)
            meta = meta_from_app1(scan.app1) if scan.app1 else None
            rows.append({"title": cand.title, "exif": meta is not None and meta.has_exif,
                         "gps": bool(meta and meta.has_gps)})
        except Skip as sk:
            rows.append({"title": cand.title, "error": sk.reason})
        except RateLimited as exc:
            rows.append({"title": cand.title, "error": f"aborted: {exc}"})
            break
    return {"checked": len(rows), "missed_gps": sum(1 for r in rows if r.get("gps")), "rows": rows}


def funnel(result: dict[str, Any], n_candidates: int) -> dict[str, Any]:
    kept = result["kept"]
    ex = [k["exif"] for k in kept]
    reasons: Counter[str] = Counter(r for k in kept for r in k["validation"]["reasons"])
    sectors: Counter[str] = Counter(k["landmark_to_camera"]["sector_label"] for k in kept)
    return {
        "candidates": n_candidates,
        "page_screen_passed": result["page_passed"],
        "with_exif_gps_in_radius": result["with_gps"],
        "kept": len(kept),
        "skip_reasons": dict(result["skips"].most_common()),
        "outcomes": dict(Counter(r["outcome"] for r in result["log"]).most_common()),
        "kept_with_heading": sum(1 for e in ex if e["heading_deg"] is not None),
        "kept_heading_true": sum(1 for e in ex if e["heading_deg"] is not None and e["heading_ref"] == "T"),
        "kept_heading_magnetic": sum(1 for e in ex if e["heading_deg"] is not None and e["heading_ref"] == "M"),
        "kept_with_focal": sum(1 for e in ex if e["focal_mm"] is not None or e["focal_35mm"] is not None),
        "kept_with_focal_35mm": sum(1 for e in ex if e["focal_35mm"] is not None),
        "kept_with_gps_accuracy": sum(1 for e in ex if e["gps_accuracy_m"] is not None),
        "kept_with_make_model": sum(1 for e in ex if e["make"] or e["model"]),
        "kept_in_category": sum(1 for k in kept if k["in_category"]),
        "kept_from_thumbnail": sum(1 for k in kept if k["pixels"]["source"].startswith("thumbnail")),
        "kept_by_exif_source": dict(Counter(k["exif_source"] for k in kept)),
        "kept_by_sector": dict(sorted(sectors.items())),
        "kept_capture_days": len({(e["captured_at"] or "?")[:10] for e in ex}),
        "product_validation_accepted": sum(1 for k in kept if k["validation"]["accepted"]),
        "product_validation_reasons": dict(reasons),
        "licences": dict(Counter(k["licence"] or "unknown" for k in kept).most_common()),
    }


# ---------------------------------------------------------------- CLI


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_outside_repo(path: Path, what: str) -> None:
    if _inside(path, REPO_ROOT) and not _inside(path, REPO_ROOT / "data"):
        raise SystemExit(f"{what} {path} is inside the repository: Commons photos and anything derived from them "
                         f"must stay outside it (or under its gitignored data/ folder)")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python scripts/fetch_commons_landmark.py", description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="landmark name (reports only)")
    ap.add_argument("--lat", type=float, required=True, help="landmark centre latitude")
    ap.add_argument("--lon", type=float, required=True, help="landmark centre longitude")
    ap.add_argument("--radius", type=float, default=150.0, help="check-in radius in metres (default 150)")
    ap.add_argument("--keep-radius", type=float, default=None,
                    help="keep photos whose EXIF GPS is at most this far from the centre (default --radius)")
    ap.add_argument("--category", action="append", default=[], help="Commons category (repeatable)")
    ap.add_argument("--category-depth", type=int, default=1, help="subcategory levels to follow (default 1)")
    ap.add_argument("--exclude-category", default=None,
                    help="regular expression (case-insensitive): subcategories matching it are not followed")
    ap.add_argument("--titles", type=Path, action="append", default=[],
                    help="text file with one Commons file title per line (repeatable)")
    ap.add_argument("--search-html", type=Path, action="append", default=[],
                    help="saved Special:Search result page to take titles from (repeatable; not fetched here)")
    ap.add_argument("--out", type=Path, required=True, help="output dir (outside the repo)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="HTTP cache (default <out>/../http-cache)")
    ap.add_argument("--max-candidates", type=int, default=300)
    ap.add_argument("--max-photos", type=int, default=None, help="stop after keeping this many")
    ap.add_argument("--no-page-prescreen", action="store_true",
                    help="range-read every candidate's EXIF instead of skipping files whose page shows no GPS")
    ap.add_argument("--prescreen-audit", type=int, default=0,
                    help="range-check up to N prescreen-rejected files to measure the prescreen's miss rate")
    ap.add_argument("--exif-source", choices=("auto", "original", "page"), default="auto",
                    help="original: range-read each original's EXIF (wait out rate limits, else stop); page: "
                         "re-encode the EXIF values the description page shows; auto (default): original, "
                         "switching to page for the rest of the run once originals are refused")
    ap.add_argument("--original-wait", type=float, default=60.0,
                    help="auto: longest Retry-After on an original to wait out before switching to page EXIF")
    ap.add_argument("--min-interval", type=float, default=MIN_INTERVAL_S, help="seconds between requests (>= 1)")
    ap.add_argument("--max-wait", type=float, default=900.0,
                    help="largest single Retry-After/backoff pause to honour; beyond it the run stops")
    ap.add_argument("--offline", action="store_true", help="use only the HTTP cache")
    return ap.parse_args(argv)


def gather(client: PoliteClient, args: argparse.Namespace) -> tuple[list[Candidate], dict[str, Any]]:
    exclude = re.compile(args.exclude_category, re.IGNORECASE) if args.exclude_category else None
    groups: list[tuple[str, list[Listing]]] = []
    report: dict[str, Any] = {}
    for p in args.titles:
        groups.append((f"titles:{p.name}", read_titles(p)))
    for p in args.search_html:
        items, total, _ = parse_search_page(p.read_text(encoding="utf-8"))
        groups.append((f"search:{p.name}", items))
        report.setdefault("search_pages", []).append({"file": str(p), "results": len(items), "reported_total": total})
    if args.category:
        cat_groups, report["categories"] = walk_categories(client, args.category, args.category_depth,
                                                           args.max_candidates, exclude)
        groups.extend(cat_groups)
    merged = merge_candidates(groups)
    report["unique_titles"] = len(merged)
    report["by_source_kinds"] = dict(Counter("+".join(sorted(c.source_kinds)) for c in merged))
    return merged[:args.max_candidates], report


def main(argv: list[str] | None = None, client: PoliteClient | None = None) -> int:
    """CLI entry point; tests pass a client with a fake transport and clock."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not (args.category or args.titles or args.search_html):
        raise SystemExit("give at least one candidate source: --category, --titles or --search-html")
    cache_dir = args.cache_dir or args.out.parent / "http-cache"
    check_outside_repo(args.out, "--out")
    check_outside_repo(cache_dir, "--cache-dir")
    args.out.mkdir(parents=True, exist_ok=True)
    client = client or PoliteClient(cache_dir, min_interval_s=args.min_interval, max_wait_s=args.max_wait,
                                    offline=args.offline)
    cfg = Config(name=args.name, lat=args.lat, lon=args.lon, radius_m=args.radius,
                 keep_radius_m=args.keep_radius if args.keep_radius is not None else args.radius, out=args.out,
                 max_photos=args.max_photos, page_prescreen=not args.no_page_prescreen,
                 prescreen_audit=args.prescreen_audit, exif_source=args.exif_source,
                 original_wait_s=args.original_wait)
    t0 = time.monotonic()
    aborted = None
    try:
        candidates, listing = gather(client, args)
    except RateLimited as exc:
        candidates, listing, aborted = [], {}, str(exc)
    t_list = time.monotonic() - t0
    result = (run(client, candidates, cfg) if candidates else
              {"kept": [], "skips": Counter(), "page_passed": 0, "with_gps": 0, "log": [],
               "aborted": aborted, "audit": None, "originals_fallback": None})
    landmark = {"name": args.name, "lat": args.lat, "lon": args.lon, "radius_m": args.radius,
                "keep_radius_m": cfg.keep_radius_m}
    stats = {
        "landmark": landmark,
        "params": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                   for k, v in vars(args).items()},
        "user_agent": USER_AGENT,
        "funnel": funnel(result, len(candidates)),
        "listing": listing,
        "prescreen_audit": result["audit"],
        "originals_fallback": result["originals_fallback"],
        "http": client.report(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "listing_elapsed_s": round(t_list, 1),
        "aborted": result["aborted"] or aborted,
    }
    manifest = {"landmark": landmark, "generated_by": "scripts/fetch_commons_landmark.py",
                "note": "Every photo keeps its Commons licence; credit the author as given. pixels.source "
                        "'original' is the camera file itself; 'thumbnail_<w>' is a Commons thumbnail carrying "
                        "the original's EXIF APP1 (Orientation reset, sizes updated) when exif_source is "
                        "'original', or the EXIF values the description page shows, re-encoded, when it is "
                        "'commons_page_metadata'.",
                "photos": result["kept"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "screening_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "candidates.json").write_text(json.dumps(result["log"], indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    f = stats["funnel"]
    print(f"candidates {f['candidates']} -> page screen {f['page_screen_passed']} -> EXIF GPS in radius "
          f"{f['with_exif_gps_in_radius']} -> kept {f['kept']} (heading {f['kept_with_heading']}, "
          f"focal {f['kept_with_focal']}); {stats['http']['network_requests']} requests, "
          f"{stats['http']['http_429']} x 429, {stats['elapsed_s']} s")
    if stats["aborted"]:
        print(f"ABORTED: {stats['aborted']}")
    print(f"wrote {args.out / 'manifest.json'} and {args.out / 'screening_stats.json'}")
    return 1 if stats["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
