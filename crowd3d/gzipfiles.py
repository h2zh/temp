"""Static files sent gzip-compressed to the clients that accept it (D34).

The site's own text files (the pages, scripts, styles and JSON of the web dir, the vendored three.js and Leaflet)
and the public model files (.glb, .ply) are several times smaller compressed: three.module.js 1,272,972 bytes ->
256,378, a 497 KB stylized GLB -> 48 KB (it is mostly JSON), an ASCII point cloud to about 40 %. Only the two
static mounts of crowd3d.api (WebFiles, ModelFiles) compress, and only files of the suffixes they name: no route's
JSON, nothing that can hold a token or anything else secret (a compressed response that mixes a secret with text an
attacker chooses leaks the secret through its length), and no format that is compressed already (photos, PNG, WebP).

A file is compressed once per version (its path, modification time and size) and the bytes are kept in a bounded
cache (GzipCache), so a request costs a dictionary lookup, not 70 ms of CPU for three.module.js. One compression
runs at a time; a file over GZIP_MAX_BYTES is sent as it is.

The compressed and the plain form are two representations of the file (RFC 9110 8.8.3.3): each has its own strong
ETag (the plain one's with "-gzip" before the closing quote), both carry Vary: Accept-Encoding, and a request's
If-None-Match is held against the ETag of the form that request would get. A request with a Range header always
gets the plain form (ranges of compressed bytes are of no use to a client that asked for part of the file), so a
range is never cut from the compressed bytes; the compressed form advertises no Accept-Ranges.
"""

from __future__ import annotations

import os
import re
import threading
import zlib
from collections import OrderedDict
from collections.abc import Iterable

import anyio
import anyio.to_thread
from anyio.lowlevel import RunVar
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response
from starlette.staticfiles import NotModifiedResponse, StaticFiles
from starlette.types import Receive, Scope, Send

# The best ratio: a file is compressed once per version, not per request. Measured on this machine (one core):
# three.module.js 68 ms (level 6: 29 ms, 0.6 % bigger), a 2.3 MB GLB 75 ms (level 6: 19 ms, 5 % bigger).
GZIP_LEVEL = 9
# Below this a file is sent as it is: it fits one packet either way (the themes' JSON files are under 1 KiB)
GZIP_MIN_BYTES = 1024
# Above this a file is sent as it is, so one compression is bounded: an ASCII point cloud of 8 MiB takes 0.7 s of
# one core, a GLB of 8 MiB 0.2 s. The largest model files the pipeline has written so far: a 3.0 MB GLB, a 0.8 MB
# point cloud.
GZIP_MAX_BYTES = 8 * 1024 * 1024
# The compressed bytes the cache keeps, least recently used out first. The whole site compressed is under 0.5 MB;
# a model file is 10-270 KB compressed.
GZIP_CACHE_MAX_BYTES = 32 * 1024 * 1024
GZIP_CHUNK_BYTES = 64 * 1024         # the pieces a compressed body is sent in (as FileResponse sends a file)
GZIP_LOOKUPS = 3                     # times a file is looked up again when it changed before it could be read
VARY = "Accept-Encoding"
_ETAG_SUFFIX = "-gzip"
# A coding's weight in Accept-Encoding (RFC 9110 12.4.2): 0 to 1 with at most three decimals
_QVALUE = re.compile(r"(?:0(?:\.[0-9]{0,3})?|1(?:\.0{0,3})?)")
_gzip_limiter: RunVar[anyio.CapacityLimiter] = RunVar("crowd3d_gzip_limiter")


def accepts_gzip(accept_encoding: Iterable[str]) -> bool:
    """Whether a request's Accept-Encoding header lines allow a gzip-coded response (RFC 9110 12.5.3): gzip (or its
    old name x-gzip) listed with a weight above 0, or "*" with a weight above 0 and gzip not listed. No header, an
    empty one, gzip;q=0 and a weight that is not a number of the allowed form all mean no: the plain form is always
    a correct answer."""
    star = False
    for line in accept_encoding:
        for item in line.split(","):
            coding, *params = (part.strip().lower() for part in item.split(";"))
            weight = 1.0
            for param in params:
                name, _, value = param.partition("=")
                if name.strip() == "q":
                    value = value.strip()
                    weight = float(value) if _QVALUE.fullmatch(value) else 0.0
            if coding in ("gzip", "x-gzip"):
                return weight > 0
            if coding == "*":
                star = weight > 0
    return star


def gzip_etag(etag: str) -> str:
    """The ETag of a file's compressed form, from the plain form's (FileResponse's quoted hash of the file's time
    and size): another strong tag, as the two forms are different bytes."""
    return etag[:-1] + _ETAG_SUFFIX + '"' if etag.endswith('"') else etag + _ETAG_SUFFIX


def gzip_bytes(data: bytes, level: int = GZIP_LEVEL) -> bytes:
    """data as one gzip member. The same bytes for the same data every time: zlib writes no time and no file name
    into the header."""
    compressor = zlib.compressobj(level, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _version(stat_result: os.stat_result) -> tuple[int, int]:
    return stat_result.st_mtime_ns, stat_result.st_size


class GzipCache:
    """The compressed bytes of files, one entry per path, valid for one version of the file (its modification time
    and size, what its ETag is made of): a changed file is compressed again and replaces its entry. At most max_bytes
    of compressed data in all; over that, the entries used longest ago go. Safe to use from several threads.

    compressions counts the files compressed, hits the answers from the cache (tests and measurements read them)."""

    def __init__(self, max_bytes: int = GZIP_CACHE_MAX_BYTES, level: int = GZIP_LEVEL) -> None:
        self.max_bytes = max_bytes
        self.level = level
        self.compressions = 0
        self.hits = 0
        self._entries: OrderedDict[str, tuple[tuple[int, int], bytes]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def stored_bytes(self) -> int:
        return self._bytes

    def get(self, path: str | os.PathLike[str], stat_result: os.stat_result) -> bytes | None:
        """The compressed bytes of the file at path in the version stat_result describes, if the cache holds them."""
        with self._lock:
            entry = self._entries.get(os.fspath(path))
            if entry is None or entry[0] != _version(stat_result):
                return None
            self._entries.move_to_end(os.fspath(path))
            self.hits += 1
            return entry[1]

    def compress(self, path: str | os.PathLike[str], stat_result: os.stat_result) -> bytes | None:
        """The compressed bytes of the file at path in the version stat_result describes, read and compressed now
        unless the cache holds them (a caller that waited its turn behind another one for the same file finds them
        there). None when the file is not that version any more: it changed or went between the lookup and the
        read, so what could be read is not what the lookup's ETag and Last-Modified describe. Reads at most the
        looked-up size (and one byte, to see a file that grew)."""
        cached = self.get(path, stat_result)
        if cached is not None:
            return cached
        version = _version(stat_result)
        try:
            with open(path, "rb") as f:
                before = os.fstat(f.fileno())
                data = f.read(stat_result.st_size + 1)
                after = os.fstat(f.fileno())
        except OSError:
            return None
        if not (_version(before) == version == _version(after) and len(data) == stat_result.st_size):
            return None
        body = gzip_bytes(data, self.level)
        with self._lock:
            self.compressions += 1
            if len(body) <= self.max_bytes:
                key = os.fspath(path)
                old = self._entries.pop(key, None)
                if old is not None:
                    self._bytes -= len(old[1])
                self._entries[key] = (version, body)
                self._bytes += len(body)
                while self._bytes > self.max_bytes:
                    _, (_, dropped) = self._entries.popitem(last=False)
                    self._bytes -= len(dropped)
        return body


def _limiter() -> anyio.CapacityLimiter:
    """One compression at a time (per event loop): a burst of first requests for different files keeps one core
    busy, not all of them, and requests for the same file wait for the first one's result instead of repeating its
    work. The waiting is done by the requests' tasks, not by worker threads the routes need."""
    try:
        return _gzip_limiter.get()
    except LookupError:
        limiter = anyio.CapacityLimiter(1)
        _gzip_limiter.set(limiter)
        return limiter


class GzipFileResponse(Response):
    """The compressed form of the file a FileResponse would send: its headers with Content-Encoding: gzip, the
    compressed form's own ETag and length, and no Accept-Ranges. The body comes from the cache (load, which the
    mount awaits before the response is sent), so the length is known and the answer needs no chunked transfer;
    it goes out in pieces of GZIP_CHUNK_BYTES."""

    def __init__(self, plain: FileResponse, cache: GzipCache) -> None:
        self.plain = plain
        self.cache = cache
        self.status_code = plain.status_code
        self.media_type = plain.media_type
        self.background = None
        self.body = b""
        plain_headers = plain.headers
        dropped = (b"content-length", b"accept-ranges", b"etag", b"content-encoding")
        self.raw_headers = [(name, value) for name, value in plain.raw_headers if name not in dropped]
        self.headers["etag"] = gzip_etag(plain_headers["etag"])
        self.headers["content-encoding"] = "gzip"

    async def load(self) -> bool:
        """Fetches the compressed bytes: from the cache, or by compressing the file in a worker thread, one file at
        a time. False when the file is no longer the version that was looked up (GzipCache.compress): this response
        cannot be sent, and the mount looks the file up again."""
        path, stat_result = self.plain.path, self.plain.stat_result
        assert stat_result is not None
        body = self.cache.get(path, stat_result)
        if body is None:
            body = await anyio.to_thread.run_sync(self.cache.compress, path, stat_result, limiter=_limiter())
            if body is None:
                return False
        self.body = body
        self.headers["content-length"] = str(len(body))
        return True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # What the request sent as its body is read before the answer begins (a GET's is empty: one message, there
        # at once). A body over the body limit is then refused whole (crowd3d.api BodySizeLimit: 413, always), and
        # none is left unread in the socket, where it cuts the answer short when the server closes the connection.
        # (FileResponse reads it while it sends the file, so there the answer is the 413 or the file, by which came
        # first.) A client that has left gets its answer sent all the same: the server drops it.
        while True:
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        if scope["method"].upper() == "HEAD":
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        body = self.body
        for start in range(0, max(len(body), 1), GZIP_CHUNK_BYTES):
            await send({"type": "http.response.body", "body": body[start:start + GZIP_CHUNK_BYTES],
                        "more_body": start + GZIP_CHUNK_BYTES < len(body)})


class GzipFiles(StaticFiles):
    """StaticFiles that sends the files of the suffixes in gzip_suffixes compressed to a client that accepts gzip.

    Which form a request gets is decided from the request and the file's suffix and size alone, so it is the same
    for a GET, a HEAD and a conditional request: the compressed form when the file is compressible (a suffix in
    gzip_suffixes, GZIP_MIN_BYTES to GZIP_MAX_BYTES long), the request's Accept-Encoding allows gzip (accepts_gzip)
    and it has no Range header; else the plain form, sent by Starlette's FileResponse as before (ranges and all).
    Every answer for a compressible file (200, 206, 304; either form) carries Vary: Accept-Encoding, so no cache
    hands the compressed bytes to a client that did not ask for them. The ETag is the chosen form's (gzip_etag), on
    the 200 and on the 304 alike, and If-None-Match is held against it: a client revalidates the form it has. A
    file of any other suffix is served exactly as StaticFiles serves it, whatever the request accepts."""

    gzip_suffixes: tuple[str, ...] = ()

    def __init__(self, *args: object, gzip_cache: GzipCache | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)        # type: ignore[arg-type]
        self.gzip_cache = gzip_cache if gzip_cache is not None else GzipCache()

    def compressible(self, full_path: str | os.PathLike[str], stat_result: os.stat_result) -> bool:
        return os.fspath(full_path).lower().endswith(self.gzip_suffixes) \
            and GZIP_MIN_BYTES <= stat_result.st_size <= GZIP_MAX_BYTES

    async def get_response(self, path: str, scope: Scope) -> Response:
        for _ in range(GZIP_LOOKUPS):
            response = await super().get_response(path, scope)
            if not isinstance(response, GzipFileResponse) or await response.load():
                return response
        # the file changed under every lookup (it is being rewritten): no whole version of it to send right now
        raise HTTPException(status_code=503, headers={"Retry-After": "1"})

    def file_response(self, full_path: str | os.PathLike[str], stat_result: os.stat_result, scope: Scope,
                      status_code: int = 200) -> Response:
        request_headers = Headers(scope=scope)
        response: Response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
        if status_code == 200 and self.compressible(full_path, stat_result):
            response.headers["vary"] = VARY
            if "range" not in request_headers and accepts_gzip(request_headers.getlist("accept-encoding")):
                response = GzipFileResponse(response, self.gzip_cache)      # type: ignore[arg-type]
        if self.is_not_modified(response.headers, request_headers):
            return NotModifiedResponse(response.headers)
        return response
