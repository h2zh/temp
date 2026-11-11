"""Static files sent gzip-compressed (crowd3d/gzipfiles.py, D34).

What these tests hold the server to: the site's text files and the public model files go compressed to a client
that accepts gzip, and the compressed body is the file byte for byte; the two forms of a file have their own ETags,
both carry Vary: Accept-Encoding, and a revalidation gets its 304 for the form the client holds (a page's 304 with
the page's Content-Security-Policy); a request for a range gets the plain bytes; HEAD answers as GET does; formats
that are compressed already, and everything that is not one of the two file mounts, are never compressed; a file is
compressed once per version, one at a time, in a worker thread (the event loop goes on answering), and kept in a
cache that holds no more than its bound, the application's own included. The requests go straight to the ASGI
application: a test client would decode a compressed body before the test saw it.
"""

from __future__ import annotations

import asyncio
import gzip
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from crowd3d import api as api_module
from crowd3d import csp, gzipfiles
from crowd3d.api import WEB_DIR, ModelFiles, WebFiles, create_app
from crowd3d.gzipfiles import GzipCache, accepts_gzip, gzip_bytes, gzip_etag

LM = "demo"
GZIP = [(b"accept-encoding", b"gzip")]
BROWSER = [(b"accept-encoding", b"gzip, deflate, br, zstd")]         # what Chromium sends
THREE = "/vendor/three/build/three.module.js"
VARY = ["Accept-Encoding"]
# headers that differ between the two forms of one file; everything else is the same on both
PER_FORM = ("content-length", "etag", "content-encoding", "accept-ranges")
TEXT = b"A line of text that compresses well, again and again.\n"


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def app(client):
    return client.app


def scope_of(path, headers, method="GET"):
    return {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"host", b"testserver"), *headers], "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80)}


def answer_of(sent):
    """(status, headers as a dict of lists, body) of the messages an application sent: one start, then the body."""
    start, *rest = sent
    assert start["type"] == "http.response.start" and all(m["type"] == "http.response.body" for m in rest), sent[:2]
    assert [m.get("more_body", False) for m in rest] == [True] * (len(rest) - 1) + [False]
    out = {}
    for name, value in start["headers"]:
        out.setdefault(name.decode().lower(), []).append(value.decode())
    return start["status"], out, b"".join(m.get("body", b"") for m in rest)


async def request(app, path, headers=(), method="GET", body=None):
    """One request to the ASGI application: (status, headers as a dict of lists, body). body: the request's body
    as its http.request messages (default: the one empty message of a request without a body); after them the
    client stays connected and silent."""
    messages = [{"type": "http.request", "body": b"", "more_body": False}] if body is None else body
    sent = []

    async def receive():
        if messages:
            return messages.pop(0)
        await anyio.sleep_forever()

    async def send(message):
        sent.append(message)

    with anyio.fail_after(120):
        await app(scope_of(path, headers, method), receive, send)
    return answer_of(sent)


def call(app, path, headers=(), method="GET", body=None):
    return anyio.run(request, app, path, headers, method, body)


def header(name, value):
    return [(name.encode(), value.encode())]


def gunzip(body):
    """The data of a gzip body that is exactly one member, with nothing after it."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    data = inflater.decompress(body)
    assert inflater.eof and inflater.unused_data == b""
    assert gzip.decompress(body) == data
    return data


def without(headers, names=PER_FORM):
    return {name: value for name, value in headers.items() if name not in names}


def model_file(app, name, data):
    folder = app.state.settings.models_dir / LM
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(data)
    return f"/files/models/{LM}/{name}"


# what the request's Accept-Encoding allows


@pytest.mark.parametrize("value, expected", [
    ("gzip", True), ("GZip", True), ("x-gzip", True), (" gzip ", True), ("gzip, deflate, br, zstd", True),
    ("deflate, gzip;q=0.5", True), ("gzip;q=1", True), ("gzip;q=1.000", True), ("gzip; q=0.001", True),
    ("gzip ; q = 0.3", True), ("br;q=1.0, gzip;q=0.8, *;q=0.1", True), ("*", True), ("*;q=0.5", True),
    ("identity, *;q=0.2", True), ("gzip;level=9", True),
    ("", False), ("identity", False), ("deflate, br", False), ("gzip;q=0", False), ("gzip;q=0.0", False),
    ("gzip;q=0.000", False), ("*;q=0", False), ("*, gzip;q=0", False), ("gzip;q=0, *", False),
    ("gzipped", False), ("not-gzip", False), ("gzip;q=", False), ("gzip;q=abc", False), ("gzip;q=2", False),
    ("gzip;q=1.5", False), ("gzip;q=0.1234", False), ("gzip;q=nan", False), ("gzip;q=inf", False),
    ("gzip;q=-1", False), ("gzip;q=1e3", False), ("deflate;q=0.5, identity;q=1", False),
])
def test_accept_encoding_is_read_with_its_weights(value, expected):
    """accepts_gzip: gzip (or x-gzip) with a weight above 0, or "*" above 0 with gzip not listed; a weight that is
    not a number of the allowed form (0 to 1, three decimals) means no, as does a header without gzip: the plain
    form is always a correct answer. (Starlette's own middleware looks for the letters "gzip" in the header, so
    gzip;q=0 gets a compressed answer from it.)"""
    assert accepts_gzip([value]) is expected


def test_accept_encoding_may_come_in_several_header_lines(app):
    assert accepts_gzip([]) is False
    assert accepts_gzip(["br", "gzip"]) is True and accepts_gzip(["gzip;q=0", "*"]) is False
    _, headers, _ = call(app, "/style.css", [(b"accept-encoding", b"br"), (b"accept-encoding", b"gzip")])
    assert headers["content-encoding"] == ["gzip"]
    for value in (b"", b"identity", b"gzip;q=0", b"br, deflate", b"*;q=0"):
        _, headers, body = call(app, "/style.css", [(b"accept-encoding", value)])
        assert "content-encoding" not in headers and body == (WEB_DIR / "style.css").read_bytes(), value


def test_the_compressed_bytes_of_the_same_data_are_always_the_same():
    """gzip_bytes writes one gzip member without a time or a file name in its header, so the same file version
    gives the same bytes in every process: what a strong ETag says. gzip_etag: the plain form's tag with -gzip
    inside the quotes."""
    data = TEXT * 500
    first = gzip_bytes(data)
    assert gzip_bytes(data) == first and gunzip(first) == data and len(first) < len(data) // 20
    assert first[:4] == b"\x1f\x8b\x08\x00" and first[4:8] == bytes(4)       # deflate, no flags, no time
    assert len(gzip_bytes(data, 9)) <= len(gzip_bytes(data, 1))
    assert gunzip(gzip_bytes(b"")) == b""
    assert gzip_etag('"0123abc"') == '"0123abc-gzip"' and gzip_etag('"0123abc"') != '"0123abc"'
    assert gzipfiles.GZIP_LEVEL == 9 and gzipfiles.GZIP_MIN_BYTES == 1024
    assert gzipfiles.GZIP_MAX_BYTES == 8 * 1024 * 1024 and gzipfiles.GZIP_CACHE_MAX_BYTES == 32 * 1024 * 1024


# the site's files


def test_every_text_file_of_the_site_goes_compressed_and_comes_back_byte_for_byte(app):
    """Every file of the site, asked for as a browser does and without Accept-Encoding. A text file of 1 KiB or
    more (WEB_GZIP_SUFFIXES) comes gzip-compressed to the browser: Content-Encoding gzip, the length of what is
    sent, its own ETag, Vary: Accept-Encoding, no Accept-Ranges, and the body decompresses to the file's bytes
    exactly; every other header is the plain answer's (the type, Last-Modified, Cache-Control no-cache, the
    Content-Security-Policy, the baseline headers), and HEAD answers with the same headers and no body. Without
    Accept-Encoding the same file comes as before, now with Vary. Every other file (the libraries' PNG images,
    which are compressed already; the small theme files; licences) is served exactly as before to both."""
    files = sorted(p for p in WEB_DIR.rglob("*") if p.is_file() and p.relative_to(WEB_DIR).parts[0] != "tests")
    assert api_module.WEB_GZIP_SUFFIXES == (".html", ".htm", ".js", ".mjs", ".css", ".json", ".svg")
    assert WebFiles.gzip_suffixes == api_module.WEB_GZIP_SUFFIXES
    compressed, sizes = [], {}
    for path in files:
        url = "/" + path.relative_to(WEB_DIR).as_posix()
        data = path.read_bytes()
        status, plain, body = call(app, url)
        assert status == 200 and body == data and "content-encoding" not in plain, url
        assert plain["content-length"] == [str(len(data))] and plain["accept-ranges"] == ["bytes"], url
        status, headers, body = call(app, url, BROWSER)
        assert status == 200, url
        if not (path.suffix in api_module.WEB_GZIP_SUFFIXES and len(data) >= 1024):
            assert headers == plain and body == data and "vary" not in headers, url
            continue
        compressed.append(url)
        sizes[url] = (len(data), len(body))
        assert headers["content-encoding"] == ["gzip"] and headers["vary"] == VARY == plain["vary"], url
        assert headers["content-length"] == [str(len(body))] and len(body) < len(data), url
        assert gunzip(body) == data, url
        (plain_tag,), (tag,) = plain["etag"], headers["etag"]
        assert plain_tag[0] == plain_tag[-1] == '"' and tag == plain_tag[:-1] + '-gzip"', (url, plain_tag, tag)
        assert "accept-ranges" not in headers and without(headers) == without(plain), url
        assert headers["cache-control"] == ["no-cache"] and len(headers["content-security-policy"]) == 1, url
        assert call(app, url, BROWSER, "HEAD") == (200, headers, b""), url
        assert call(app, url, method="HEAD") == (200, plain, b""), url
    # which files that is: the pages, the scripts and styles, the libraries; not their images, not the small JSON
    assert {".html", ".js", ".css"} <= {Path(url).suffix for url in compressed} <= set(api_module.WEB_GZIP_SUFFIXES)
    assert {"/index.html", "/landmark.html", "/stylized.html", "/terms.html", "/style.css", "/api.js", "/boot.js",
            THREE, "/vendor/three/examples/jsm/loaders/GLTFLoader.js", "/vendor/leaflet/leaflet.js",
            "/vendor/leaflet/leaflet.css"} <= set(compressed)
    assert len(compressed) >= 30 and len(files) - len(compressed) >= 12
    assert not [p for p in files if p.suffix == ".png" and "/" + p.relative_to(WEB_DIR).as_posix() in compressed]
    whole, sent = sizes[THREE]
    assert whole > 1_200_000 and sent < 0.22 * whole          # 1,272,972 -> 256,378 bytes
    assert sum(b for _, b in sizes.values()) < 0.3 * sum(a for a, _ in sizes.values())
    # the directory index is index.html in either form
    status, headers, body = call(app, "/", BROWSER)
    assert status == 200 and headers["content-encoding"] == ["gzip"]
    assert gunzip(body) == (WEB_DIR / "index.html").read_bytes()
    assert headers["content-security-policy"] == [csp.page_policy((WEB_DIR / "index.html").read_text("utf-8"))]
    # the node tests stay unserved, whatever the request accepts
    status, headers, _ = call(app, "/tests/api.test.mjs", BROWSER)
    assert status == 404 and "content-encoding" not in headers and "vary" not in headers


def test_only_the_named_kinds_of_file_are_ever_compressed(tmp_path, app):
    """The suffix decides, not the content: a file of text that would compress to a fiftieth is sent as it is
    under any name that is not one of the mount's suffixes. So a photo, a PNG, a WebP, a font, an archive (formats
    that are compressed already) and anything unknown is never compressed, whatever the client accepts. The model
    mount serves and compresses .ply and .glb, nothing else."""
    text = TEXT * 200
    never = ["a.jpg", "a.jpeg", "a.JPG", "a.png", "a.webp", "a.gif", "a.avif", "a.ico", "a.woff", "a.woff2", "a.zip",
             "a.gz", "a.br", "a.mp4", "a.pdf", "a.wasm", "a.bin", "a.txt", "a.md", "a.map", "a.xml", "a.csv",
             "a.sqlite3", "noext", "a.js.bak", "a.htmlx", "a.json5", "js"]
    always = ["a.html", "a.htm", "a.js", "a.mjs", "a.css", "a.json", "a.svg", "A.JS", "b.Html", "c.min.js"]
    for name in never + always:
        (tmp_path / name).write_bytes(text)
    site = WebFiles(directory=tmp_path)
    for name in never:
        for accept in ((), GZIP, BROWSER):
            status, headers, body = call(site, "/" + name, accept)
            assert status == 200 and body == text, name
            assert "content-encoding" not in headers and "vary" not in headers and headers["etag"], name
            assert headers["content-length"] == [str(len(text))] and headers["accept-ranges"] == ["bytes"], name
    for name in always:
        status, headers, body = call(site, "/" + name, BROWSER)
        assert status == 200 and headers["content-encoding"] == ["gzip"] and gunzip(body) == text, name
        assert len(body) < len(text) // 20 and headers["vary"] == VARY, name
    assert site.gzip_cache.compressions == len(always) == len(site.gzip_cache)

    assert ModelFiles.gzip_suffixes == ModelFiles.SUFFIXES == (".ply", ".glb")
    served = {"m.ply": b"ply\nformat ascii 1.0\n" + b"1.2345 -6.7890 0.1234 200 190 180\n" * 400,
              "m.glb": b"glTF" + json.dumps({"nodes": [{"extras": {"part": i}} for i in range(400)]}).encode()}
    for name, data in served.items():
        url = model_file(app, name, data)
        status, plain, body = call(app, url)
        assert status == 200 and body == data and "content-encoding" not in plain and plain["vary"] == VARY
        status, headers, body = call(app, url, BROWSER)
        assert status == 200 and headers["content-encoding"] == ["gzip"] and gunzip(body) == data, name
        assert headers["content-length"] == [str(len(body))] and len(body) < len(data) // 3, name
        assert headers["etag"] == [gzip_etag(plain["etag"][0])] and "accept-ranges" not in headers, name
        assert without(headers) == without(plain) and "cache-control" not in headers, name
        assert headers["content-security-policy"] == [csp.STRICT_POLICY], name
        assert call(app, url, BROWSER, "HEAD") == (200, headers, b""), name
    for name in ("m.json", "m.style.json", "m.txt"):          # the sidecars: not served, compressed or not
        model_file(app, name, text)
        status, headers, _ = call(app, f"/files/models/{LM}/{name}", BROWSER)
        assert status == 404 and "content-encoding" not in headers and "vary" not in headers, name


def test_no_route_of_the_api_is_compressed(client, app):
    """Only the two file mounts compress. No route's JSON does, whatever the client accepts: a response that mixes
    a secret (a contributor token, a manage token) with text the requester chose would leak the secret through its
    compressed length, and these responses are small. (tests/test_api.py holds every kind of response, a served
    photo among them, to the same.)"""
    register = json.dumps({"nickname": "gz", "terms": api_module.TERMS_CURRENT}).encode()
    accept = {"accept-encoding": "gzip, deflate, br, zstd"}
    got = [client.get("/api/health", headers=accept), client.get("/api/landmarks", headers=accept),
           client.get("/openapi.json", headers=accept), client.get("/api/landmarks/nope", headers=accept),
           client.post("/api/landmarks", json={}, headers=accept),
           client.post("/api/contributors", content=register, headers=accept | {"content-type": "application/json"}),
           client.post("/api/contributors/photos", json={"contributor_token": "0" * 64}, headers=accept)]
    assert [r.status_code for r in got] == [200, 200, 200, 404, 422, 201, 422]
    assert len(got[2].content) > 5000 and len(got[5].json()["contributor_token"]) == 64
    for r in got:
        assert "content-encoding" not in r.headers and "vary" not in r.headers, r.url
        assert r.headers["content-length"] == str(len(r.content)), r.url
    assert app.state.gzip_cache.compressions == 0 and len(app.state.gzip_cache) == 0


# revalidation


def test_a_revalidation_gets_its_304_for_the_form_the_client_holds(app):
    """If-None-Match is held against the ETag of the form the request would get: a browser that holds the
    compressed form sends that form's tag with its Accept-Encoding and gets a 304; a client without Accept-Encoding
    gets one for the plain tag; a tag of the other form is not a match, and the answer is the whole file in the
    form asked for. The 304 carries the form's ETag, Vary, and what the 200 carried besides (Cache-Control of the
    site's files; a page's own Content-Security-Policy, without which the stored page could load nothing). No
    revalidation compresses anything."""
    glb = b"glTF" + json.dumps({"nodes": list(range(3000))}).encode()
    urls = {"/": csp.page_policy((WEB_DIR / "index.html").read_text("utf-8")),
            "/stylized.html": csp.page_policy((WEB_DIR / "stylized.html").read_text("utf-8")),
            "/api.js": csp.STRICT_POLICY, THREE: csp.STRICT_POLICY, model_file(app, "m.glb", glb): csp.STRICT_POLICY}
    cache = app.state.gzip_cache
    for url, policy in urls.items():
        status, plain, plain_body = call(app, url)
        (plain_tag,), (modified,) = plain["etag"], plain["last-modified"]
        tag = gzip_etag(plain_tag)
        # before anything was compressed: the 304 needs the tag only
        before = cache.compressions
        status, headers, body = call(app, url, GZIP + header("if-none-match", tag))
        assert (status, body, headers["etag"]) == (304, b"", [tag]) and cache.compressions == before, url
        status, packed, packed_body = call(app, url, GZIP)
        assert status == 200 and packed["etag"] == [tag] and cache.compressions == before + 1, url
        other = '"0123456789abcdef0123456789abcdef"'
        cases = [
            (GZIP + header("if-none-match", tag), 304, tag),
            (header("if-none-match", plain_tag), 304, plain_tag),
            (BROWSER + header("if-none-match", "W/" + tag), 304, tag),
            (header("if-none-match", "W/" + plain_tag), 304, plain_tag),
            (GZIP + header("if-none-match", f"{other}, {tag}"), 304, tag),
            (GZIP + header("if-none-match", "*"), 304, tag),
            (header("if-none-match", "*"), 304, plain_tag),
            (GZIP + header("if-modified-since", modified), 304, tag),
            (header("if-modified-since", modified), 304, plain_tag),
            (GZIP + header("if-none-match", tag) + header("range", "bytes=0-9"), 200, plain_tag),
            # the other form's tag, another file's tag, an older date: the whole file, in the form asked for
            (GZIP + header("if-none-match", plain_tag), 200, tag),
            (header("if-none-match", tag), 200, plain_tag),
            (GZIP + header("if-none-match", other), 200, tag),
            (GZIP + header("if-none-match", other) + header("if-modified-since", modified), 200, tag),
            (GZIP + header("if-modified-since", "Thu, 01 Jan 2004 00:00:00 GMT"), 200, tag),
            (header("accept-encoding", "gzip;q=0") + header("if-none-match", tag), 200, plain_tag),
        ]
        for sent, expected, expected_tag in cases:
            status, headers, body = call(app, url, sent)
            if (b"range", b"bytes=0-9") in sent:      # a range is cut from the plain form: its tag is the plain one
                assert status == 206 and headers["etag"] == [plain_tag] and body == plain_body[:10], (url, sent)
                continue
            assert status == expected and headers["etag"] == [expected_tag] and headers["vary"] == VARY, (url, sent)
            assert headers["content-security-policy"] == [policy], (url, sent)
            assert headers.get("cache-control") == plain.get("cache-control"), (url, sent)
            assert headers["x-content-type-options"] == ["nosniff"], (url, sent)
            if status == 304:
                assert body == b"" and "content-encoding" not in headers and "content-length" not in headers
            elif expected_tag == tag:
                assert headers == packed and body == packed_body, (url, sent)
            else:
                assert headers == plain and body == plain_body, (url, sent)
        assert cache.compressions == before + 1, url
    # a file that is not compressible has one form, one tag and no Vary, as before
    icon = "/vendor/leaflet/images/marker-icon.png"
    _, plain, _ = call(app, icon)
    status, headers, _ = call(app, icon, BROWSER + header("if-none-match", plain["etag"][0]))
    assert status == 304 and headers["etag"] == plain["etag"] and "vary" not in headers
    assert call(app, icon, BROWSER + header("if-none-match", gzip_etag(plain["etag"][0])))[0] == 200


# ranges


def test_a_request_for_a_range_gets_the_plain_bytes(app):
    """A range is always cut from the plain form, whatever the request accepts: a client that asks for part of a
    file wants those bytes of the file. The 206 carries the plain form's ETag and Vary, never a Content-Encoding;
    If-Range with the plain tag gives the range, with the compressed form's tag the whole plain file (it names
    other bytes). The compressed form does not offer ranges (no Accept-Ranges). No range request compresses."""
    glb = b"glTF" + bytes(range(256)) * 600 + json.dumps({"nodes": list(range(20000))}).encode()
    ply = b"ply\n" + b"0.5 0.25 0.125 1 2 3\n" * 8000
    files = {model_file(app, "m.glb", glb): glb, THREE: (WEB_DIR / THREE[1:]).read_bytes(),
             model_file(app, "m.ply", ply): ply}
    cache = app.state.gzip_cache
    for url, data in files.items():
        size, before = len(data), cache.compressions
        _, plain, _ = call(app, url)
        (plain_tag,) = plain["etag"]
        ranges = {"bytes=0-99": (0, 100), "bytes=100-299": (100, 300), f"bytes={size - 50}-": (size - 50, size),
                  "bytes=-70": (size - 70, size), "bytes=0-": (0, size), f"bytes=5-{size * 2}": (5, size),
                  "bytes=70000-140000": (70000, 140001)}
        for accept in ((), GZIP, BROWSER):
            for spec, (start, end) in ranges.items():
                status, headers, body = call(app, url, [*accept, *header("range", spec)])
                assert status == 206 and body == data[start:end], (url, spec)
                assert headers["content-range"] == [f"bytes {start}-{end - 1}/{size}"], (url, spec)
                assert headers["content-length"] == [str(end - start)] and headers["etag"] == [plain_tag], (url, spec)
                assert "content-encoding" not in headers and headers["vary"] == VARY, (url, spec)
                assert headers["accept-ranges"] == ["bytes"], (url, spec)
                assert without(headers, ("content-length", "content-range")) == without(plain, ("content-length",))
            # two ranges: one multipart answer of the plain bytes
            status, headers, body = call(app, url, [*accept, *header("range", "bytes=0-9, 50-59")])
            assert status == 206 and headers["content-type"][0].startswith("multipart/byteranges"), url
            assert "content-encoding" not in headers and data[0:10] in body and data[50:60] in body, url
            assert headers["content-length"] == [str(len(body))] and headers["etag"] == [plain_tag], url
            # past the end: 416; the head of a range: its headers
            status, headers, body = call(app, url, [*accept, *header("range", f"bytes={size}-")])
            assert status == 416 and headers["content-range"] == [f"bytes */{size}"], url
            assert "content-encoding" not in headers, url
            status, headers, body = call(app, url, [*accept, *header("range", "bytes=0-99")], "HEAD")
            assert status == 206 and body == b"" and headers["content-length"] == ["100"], url
            assert "content-encoding" not in headers, url
            # If-Range: the plain tag is this form's; the compressed form's tag is not, so the whole file
            sent = [*accept, *header("range", "bytes=0-99")]
            status, headers, body = call(app, url, sent + header("if-range", plain_tag))
            assert status == 206 and body == data[:100], url
            status, headers, body = call(app, url, sent + header("if-range", gzip_etag(plain_tag)))
            assert status == 200 and body == data and headers == plain, url
        assert cache.compressions == before
        status, headers, _ = call(app, url, GZIP)
        assert status == 200 and headers["content-encoding"] == ["gzip"] and "accept-ranges" not in headers
        assert plain["accept-ranges"] == ["bytes"] and cache.compressions == before + 1


# the cache


def stat_of(path):
    return os.stat(path)


def test_the_cache_compresses_a_file_once_per_version(tmp_path):
    """GzipCache: the compressed bytes of a file are made once and kept while the file stays the same version (its
    modification time and size); a changed file is compressed again and takes the old entry's place. A file that is
    no longer the version that was looked up (changed, grown, shrunk or gone before it was read) gives None and
    leaves the cache as it was: what could be read is not what the lookup's ETag describes."""
    cache = GzipCache()
    path = tmp_path / "a.js"
    path.write_bytes(TEXT * 100)
    first = stat_of(path)
    assert cache.get(path, first) is None and (cache.compressions, cache.hits, len(cache)) == (0, 0, 0)
    body = cache.compress(path, first)
    assert gunzip(body) == TEXT * 100 and (cache.compressions, cache.hits, len(cache)) == (1, 0, 1)
    assert cache.stored_bytes == len(body)
    # at the cache's level: GZIP_LEVEL unless it was built with another
    assert cache.level == gzipfiles.GZIP_LEVEL and body == gzip_bytes(TEXT * 100, gzipfiles.GZIP_LEVEL)
    quick = GzipCache(level=1).compress(path, first)
    assert quick == gzip_bytes(TEXT * 100, 1) != body and gunzip(quick) == TEXT * 100
    assert cache.get(path, first) is body and cache.compress(path, first) is body
    assert cache.get(str(path), first) is body
    assert (cache.compressions, cache.hits) == (1, 3)
    # another size: another version, which replaces the entry
    path.write_bytes(TEXT * 150)
    second = stat_of(path)
    assert cache.get(path, second) is None
    changed = cache.compress(path, second)
    assert gunzip(changed) == TEXT * 150 and (cache.compressions, len(cache)) == (2, 1)
    assert cache.stored_bytes == len(changed) and cache.get(path, first) is None
    # the same size at another time: another version too
    path.write_bytes((TEXT * 150).upper())
    os.utime(path, ns=(second.st_atime_ns, second.st_mtime_ns + 1))
    third = stat_of(path)
    assert third.st_size == second.st_size and cache.get(path, third) is None
    assert gunzip(cache.compress(path, third)) == (TEXT * 150).upper() and (cache.compressions, len(cache)) == (3, 1)
    # and another size at the same time
    path.write_bytes((TEXT * 151).upper())
    os.utime(path, ns=(third.st_atime_ns, third.st_mtime_ns))
    fourth = stat_of(path)
    assert fourth.st_mtime_ns == third.st_mtime_ns and fourth.st_size != third.st_size
    assert cache.get(path, fourth) is None
    assert gunzip(cache.compress(path, fourth)) == (TEXT * 151).upper() and (cache.compressions, len(cache)) == (4, 1)
    # a lookup that is out of date when the file is read: nothing compressed, nothing kept, the entry untouched
    kept = cache.get(path, fourth)
    for rewrite in (TEXT * 152, TEXT * 149, b""):
        path.write_bytes(rewrite)
        assert cache.compress(path, second) is None
    path.write_bytes((TEXT * 150).lower())                   # the old size again, at a later time
    os.utime(path, ns=(second.st_atime_ns, second.st_mtime_ns + 2_000_000_000))
    assert cache.compress(path, second) is None and cache.get(path, fourth) is kept
    path.unlink()
    assert cache.compress(path, second) is None
    assert cache.compress(tmp_path, stat_of(tmp_path)) is None           # a directory cannot be read
    assert cache.compressions == 4 and len(cache) == 1 and cache.stored_bytes == len(kept)


def test_the_cache_is_bounded_and_drops_what_was_used_longest_ago(tmp_path):
    """At most max_bytes of compressed data: over that, the entries used longest ago go, and a file whose
    compressed bytes alone are over the bound is compressed for its request and not kept."""
    files = {}
    for name in "abcde":
        path = tmp_path / f"{name}.js"
        path.write_bytes(os.urandom(1000) + name.encode() * 24)       # compresses to about 1,030 bytes
        files[name] = (path, stat_of(path))
    size = {name: len(gzip_bytes(path.read_bytes())) for name, (path, _) in files.items()}
    assert all(1000 < n < 1100 for n in size.values())
    cache = GzipCache(max_bytes=size["a"] + size["b"] + size["c"] + 10)
    for name in "abc":
        cache.compress(*files[name])
    assert len(cache) == 3 and cache.stored_bytes == size["a"] + size["b"] + size["c"]
    assert cache.get(*files["a"]) is not None                          # a is the most recently used now
    cache.compress(*files["d"])                                        # so b goes
    assert cache.get(*files["b"]) is None and all(cache.get(*files[n]) is not None for n in "acd")
    assert len(cache) == 3 and cache.stored_bytes == size["a"] + size["c"] + size["d"] <= cache.max_bytes
    cache.compress(*files["e"])                                        # a was used before c and d just now: a goes
    assert cache.get(*files["a"]) is None and len(cache) == 3 and cache.stored_bytes <= cache.max_bytes
    assert cache.compressions == 5
    big = tmp_path / "big.js"
    big.write_bytes(os.urandom(5000))
    body = cache.compress(big, stat_of(big))
    assert gunzip(body) == big.read_bytes() and cache.get(big, stat_of(big)) is None
    assert len(cache) == 3 and cache.stored_bytes <= cache.max_bytes and cache.compressions == 6
    tiny = GzipCache(max_bytes=0)
    assert gunzip(tiny.compress(*files["a"])) == files["a"][0].read_bytes() and len(tiny) == 0 == tiny.stored_bytes


def test_a_file_is_compressed_once_however_often_it_is_asked_for_and_again_when_it_changes(app):
    """The mounts and the cache together: 20 requests for a file compress it once; requests that only ask (HEAD, a
    revalidation, a range, a client that does not accept gzip) do not add to that; the file rewritten is served in
    its new version with a new ETag at the next request, and the old ETag no longer gets a 304."""
    cache = app.state.gzip_cache
    old, new = TEXT * 3000, b"glTF" + TEXT.upper() * 2500
    url = model_file(app, "m.glb", old)
    path = app.state.settings.models_dir / LM / "m.glb"
    answers = [call(app, url, BROWSER) for _ in range(20)]
    assert all(answer == answers[0] for answer in answers) and gunzip(answers[0][2]) == old
    assert (cache.compressions, len(cache)) == (1, 1) and cache.hits >= 19
    (tag,) = answers[0][1]["etag"]
    for sent, method in (((), "GET"), (BROWSER, "HEAD"), (BROWSER + header("if-none-match", tag), "GET"),
                         (BROWSER + header("range", "bytes=0-9"), "GET"), (header("accept-encoding", "br"), "GET")):
        assert call(app, url, sent, method)[0] in (200, 206, 304)
    assert cache.compressions == 1
    before = stat_of(path)
    path.write_bytes(new)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 3_000_000_000))
    status, headers, body = call(app, url, BROWSER + header("if-none-match", tag))
    assert status == 200 and gunzip(body) == new and headers["etag"] != [tag]
    assert headers["etag"] == [gzip_etag(call(app, url)[1]["etag"][0])]
    assert (cache.compressions, len(cache), cache.stored_bytes) == (2, 1, len(body))
    assert call(app, url, BROWSER + header("if-none-match", headers["etag"][0]))[0] == 304
    # the site's files share the cache: one bound for the process
    call(app, "/style.css", BROWSER)
    assert (cache.compressions, len(cache)) == (3, 2)
    assert {route.app.gzip_cache for route in app.routes if getattr(route, "name", "") in ("web", "models")} == {cache}


def test_the_applications_cache_holds_no_more_than_its_bound(app):
    """The cache the application serves from has the bound GZIP_CACHE_MAX_BYTES: create_app builds it as GzipCache()
    does, and that default is the constant. Held to it with files as they are served: five model files of 7 MiB
    that do not compress make 35 MiB of compressed bytes, more than the 32 MiB the cache may hold. After every
    request the cache holds no more than its bound; the fifth file pushes out the first (the one used longest
    ago), which is compressed again when it is asked for again, and only that one."""
    cache = app.state.gzip_cache
    assert cache.max_bytes == gzipfiles.GZIP_CACHE_MAX_BYTES == GzipCache().max_bytes
    size = 7 * 1024 * 1024
    assert size <= gzipfiles.GZIP_MAX_BYTES and 4 * size < cache.max_bytes < 5 * size
    urls, sent = [], []
    try:
        for n in range(5):
            data = os.urandom(size)
            urls.append(model_file(app, f"noise{n}.glb", data))
            status, headers, body = call(app, urls[-1], GZIP)
            assert status == 200 and headers["content-encoding"] == ["gzip"] and gunzip(body) == data, n
            sent.append(len(body))
            assert len(body) > size and cache.stored_bytes <= cache.max_bytes, n
            assert (cache.compressions, len(cache)) == (n + 1, min(n + 1, 4)), n
        assert cache.stored_bytes == sum(sent[1:]) > 4 * size
        hits = cache.hits
        for url in urls[1:]:                                   # these four are still there
            assert call(app, url, GZIP, "HEAD")[1]["content-length"] == [str(sent[urls.index(url)])]
        assert (cache.compressions, cache.hits) == (5, hits + 4)
        status, headers, _ = call(app, urls[0], GZIP, "HEAD")  # the first one went: compressed again, the second goes
        assert status == 200 and headers["content-length"] == [str(sent[0])]
        assert (cache.compressions, len(cache)) == (6, 4)
        assert cache.stored_bytes == sent[0] + sum(sent[2:]) <= cache.max_bytes
    finally:
        for url in urls:                                       # 35 MB that no later test needs
            (app.state.settings.models_dir / LM / Path(url).name).unlink(missing_ok=True)


def test_files_outside_the_size_bounds_are_sent_as_they_are(app, monkeypatch):
    """Below GZIP_MIN_BYTES a file fits a packet either way; above GZIP_MAX_BYTES one compression would take too
    long (the bound keeps a first request's CPU bounded). Both are served exactly as before, without Vary: no
    request gets another form of them. The bounds themselves are inside."""
    monkeypatch.setattr(gzipfiles, "GZIP_MAX_BYTES", 6000)
    sizes = {1023: False, 1024: True, 1025: True, 6000: True, 6001: False, 20000: False}
    for size, compressed in sizes.items():
        data = (TEXT * 400)[:size]
        url = model_file(app, f"m{size}.ply", data)
        _, plain, _ = call(app, url)
        status, headers, body = call(app, url, BROWSER)
        assert status == 200 and ("content-encoding" in headers) is compressed, size
        if compressed:
            assert gunzip(body) == data and headers["vary"] == VARY == plain["vary"], size
        else:
            assert body == data and headers == plain and "vary" not in headers, size
    assert app.state.gzip_cache.compressions == 3
    theme = "/themes/default.json"                 # the site's small JSON files: under 1 KiB, sent as they are
    assert (WEB_DIR / theme[1:]).stat().st_size < 1024
    status, headers, body = call(app, theme, BROWSER)
    assert status == 200 and "content-encoding" not in headers and body == (WEB_DIR / theme[1:]).read_bytes()


def test_a_file_that_changes_between_the_lookup_and_the_read_is_looked_up_again(app, monkeypatch):
    """The compressed form's ETag, Last-Modified and length are of the version that was looked up. A file replaced
    before it could be read is looked up again, so the answer is one version throughout; one that is gone by then
    is a 404; one that changes under every lookup (it is being written) is a 503 with Retry-After, not a torn file
    under a validator."""
    cache = app.state.gzip_cache
    old, new = TEXT * 2000, TEXT.upper() * 2500
    url = model_file(app, "m.glb", old)
    path = app.state.settings.models_dir / LM / "m.glb"
    compress = GzipCache.compress
    calls = []

    def replaced_before_the_read(self, file, stat_result):
        calls.append(stat_result.st_size)
        if len(calls) == 1:
            path.write_bytes(new)
        return compress(self, file, stat_result)

    monkeypatch.setattr(GzipCache, "compress", replaced_before_the_read)
    status, headers, body = call(app, url, BROWSER)
    assert status == 200 and gunzip(body) == new and calls == [len(old), len(new)]
    assert headers["etag"] == [gzip_etag(call(app, url)[1]["etag"][0])] and cache.compressions == 1
    assert headers["content-length"] == [str(len(body))]

    def gone_before_the_read(self, file, stat_result):
        path.unlink()
        return compress(self, file, stat_result)

    path.write_bytes(old + b"again")
    monkeypatch.setattr(GzipCache, "compress", gone_before_the_read)
    status, headers, body = call(app, url, BROWSER)
    assert status == 404 and json.loads(body) == {"detail": "Not Found"} and "content-encoding" not in headers

    def changing_all_the_time(self, file, stat_result):
        calls.append("changing")
        path.write_bytes(old + b"x" * len(calls))
        return compress(self, file, stat_result)

    path.write_bytes(old)
    del calls[:]
    monkeypatch.setattr(GzipCache, "compress", changing_all_the_time)
    status, headers, body = call(app, url, BROWSER)
    assert status == 503 and headers["retry-after"] == ["1"] and "content-encoding" not in headers
    assert calls == ["changing"] * gzipfiles.GZIP_LOOKUPS == ["changing"] * 3
    assert headers["x-content-type-options"] == ["nosniff"] and cache.compressions == 1
    monkeypatch.setattr(GzipCache, "compress", compress)
    status, headers, body = call(app, url, BROWSER)                    # the file at rest again: served
    assert status == 200 and gunzip(body) == path.read_bytes()


def test_one_compression_runs_at_a_time_and_waiting_requests_share_its_result(app, monkeypatch):
    """A burst of first requests: 12 for one file and 4 for three others, all at once. Each file is compressed
    once (the requests that waited find the first one's result), and never two files at the same time, so a burst
    keeps one core busy with compressing, not all of them.

    And none of it in the event loop: every compression runs in a worker thread, and while it runs the loop
    answers a request for a file that is in the cache already (asked from inside the compression, which waits for
    the answer: a loop that was blocked, by the compression itself or by a lock held across it, could not give
    it). The compression holds the one token of the compressions' own limiter, so the requests behind it wait as
    tasks for that token, not in worker threads of the pool the routes use."""
    cache = app.state.gzip_cache
    files = {model_file(app, f"m{i}.glb", b"glTF" + (b"%d " % i) * 40000): 12 if i == 0 else 4 for i in range(4)}
    assert call(app, "/style.css", GZIP)[0] == 200             # in the cache before the burst
    assert cache.compressions == 1
    running, most, lock = [0], [0], threading.Lock()
    loop, threads, tokens, meanwhile = {}, [], [], []
    compress = gzipfiles.gzip_bytes

    async def during():
        limiter = gzipfiles._limiter()
        tokens.append((limiter.total_tokens, limiter.borrowed_tokens))
        return await request(app, "/style.css", GZIP)

    def counted(data, level=9):
        threads.append(threading.get_ident())
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        try:
            if threads[-1] != loop["thread"]:                  # (in the loop's own thread the wait would never end)
                meanwhile.append(asyncio.run_coroutine_threadsafe(during(), loop["loop"]).result(timeout=30))
            time.sleep(0.05)
            return compress(data, level)
        finally:
            with lock:
                running[0] -= 1

    monkeypatch.setattr(gzipfiles, "gzip_bytes", counted)
    answers = {}

    async def burst():
        loop.update(loop=asyncio.get_running_loop(), thread=threading.get_ident())

        async def one(url, n):
            answers[url, n] = await request(app, url, BROWSER)

        async with anyio.create_task_group() as group:
            for url, count in files.items():
                for n in range(count):
                    group.start_soon(one, url, n)

    anyio.run(burst)
    assert len(answers) == 24 and cache.compressions == 1 + 4 and most[0] == 1
    assert len(threads) == 4 and loop["thread"] not in threads
    assert tokens == [(1, 1)] * 4
    assert [(status, headers["content-encoding"]) for status, headers, _ in meanwhile] == [(200, ["gzip"])] * 4
    assert all(gunzip(body) == (WEB_DIR / "style.css").read_bytes() for _, _, body in meanwhile)
    for (url, _), (status, headers, body) in answers.items():
        assert status == 200 and headers["content-encoding"] == ["gzip"], url
        assert gunzip(body) == (app.state.settings.models_dir / LM / Path(url).name).read_bytes(), url


def test_the_compressed_form_is_sent_in_pieces(app, monkeypatch):
    """The compressed body goes out in pieces of 64 KiB, as a plain file does, under one Content-Length; the last
    piece, and only that one, says nothing follows, also when the body is a whole number of pieces."""
    data = os.urandom(300_000)                       # does not compress: the body is a little over 300,000 bytes
    url = model_file(app, "noise.glb", data)
    sent = []

    async def main():
        messages = [{"type": "http.request", "body": b"", "more_body": False}]

        async def receive():
            return messages.pop(0)                   # asked once: the request's (empty) body, then not again

        async def send(message):
            sent.append(message)

        await app(scope_of(url, GZIP), receive, send)

    anyio.run(main)
    pieces = [m["body"] for m in sent[1:]]
    status, headers, body = answer_of(sent)
    assert status == 200 and gunzip(body) == data and headers["content-length"] == [str(len(body))]
    assert [len(p) for p in pieces[:-1]] == [gzipfiles.GZIP_CHUNK_BYTES] * 4 and 0 < len(pieces[-1]) <= 64 * 1024
    assert gzipfiles.GZIP_CHUNK_BYTES == 64 * 1024
    for size in (len(body), len(body) // 2 + len(body) % 2, len(body) + 1, 1000):    # one piece, two, one, many
        monkeypatch.setattr(gzipfiles, "GZIP_CHUNK_BYTES", size)
        assert call(app, url, GZIP) == (status, headers, body), size            # (answer_of checks the pieces' ends)


def test_a_request_body_is_read_before_the_compressed_form_is_sent(app):
    """A GET may carry a body of its own. The compressed form reads it before it answers: within the body limit it
    is dropped and the file is sent whole; over the limit the answer is the limit's 413, always (the plain form
    reads the body while it sends the file, so there it is the 413 or the file: tests/test_api.py). Left unread, a
    body cut the compressed answer short on a real server when the connection closed. A client that leaves instead
    of sending its body ends the request without an error."""
    small = api_module.SMALL_BODY_MAX_BYTES
    url = model_file(app, "m.glb", b"glTF" + TEXT * 20000)
    data = (app.state.settings.models_dir / LM / "m.glb").read_bytes()
    chunked = [(b"transfer-encoding", b"chunked")]

    def pieces(*sizes):
        return [{"type": "http.request", "body": b"a" * size, "more_body": True} for size in sizes] + \
            [{"type": "http.request", "body": b"", "more_body": False}]

    for path, expected in ((url, data), (THREE, (WEB_DIR / THREE[1:]).read_bytes())):
        for sizes in ((1,), (small // 2, small // 2), (small,), ()):
            body = pieces(*sizes)
            status, headers, answer = call(app, path, GZIP + chunked, body=body)
            assert status == 200 and headers["content-encoding"] == ["gzip"] and gunzip(answer) == expected, sizes
            assert not body, (path, sizes)                           # read to its end before the answer
        for sizes in ((small + 1,), (small, 1), (1024, small), (small // 2,) * 3):
            status, headers, answer = call(app, path, GZIP + chunked, body=pieces(*sizes))
            assert status == 413 and json.loads(answer) == {"detail": f"request body over {small} bytes"}, sizes
            assert "content-encoding" not in headers and headers["x-content-type-options"] == ["nosniff"]
        declared = GZIP + header("content-length", str(small + 1))
        assert call(app, path, declared, body=pieces(small + 1))[0] == 413
        for leaving in ([{"type": "http.disconnect"}], pieces(10)[:1] + [{"type": "http.disconnect"}]):
            status, headers, answer = call(app, path, GZIP + chunked, body=leaving)
            assert status == 200 and gunzip(answer) == expected and not leaving
        status, headers, answer = call(app, path, GZIP + chunked, "HEAD", body=pieces(100))
        assert status == 200 and answer == b"" and headers["content-encoding"] == ["gzip"]


# a real server


def raw_get(port, path, headers, method="GET"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        connection.close()


def test_a_real_server_sends_both_forms_whole(tmp_path):
    """On a real server process (uvicorn checks every answer's length against its Content-Length, and frames the
    connection by it): the compressed form of the biggest script and of a model file arrive whole and decompress
    to the file; two requests on one connection stay in step; HEAD, a range, a 304 and a client that does not
    accept gzip get what the tests above say; a GET that carries a chunked body gets the compressed file whole when
    the body is within the body limit and the 413 when it is over; the log stays free of errors."""
    model = b"glTF" + json.dumps({"nodes": [{"extras": {"n": i, "kind": "wall"}} for i in range(30000)]}).encode()
    script = (WEB_DIR / THREE[1:]).read_bytes()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    (tmp_path / "data" / "models" / LM).mkdir(parents=True)
    (tmp_path / "data" / "models" / LM / "m.glb").write_bytes(model)
    env = {k: v for k, v in os.environ.items() if k != "MALLOC_MMAP_THRESHOLD_"} | {"CROWD3D_STRUCTURE_CATALOG": ""}
    with open(tmp_path / "server.log", "wb") as log:
        server = subprocess.Popen([sys.executable, "-m", "crowd3d.api", "--port", str(port), "--data-dir",
                                   str(tmp_path / "data"), "--reconstructor", "fake"],
                                  cwd=Path(api_module.__file__).resolve().parent.parent, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                assert raw_get(port, "/api/health", {})[0] == 200
                break
            except OSError:
                assert server.poll() is None and time.monotonic() < deadline, (tmp_path / "server.log").read_text()
                time.sleep(0.1)
        accept = {"Accept-Encoding": "gzip, deflate, br, zstd"}
        for path, data in ((THREE, script), (f"/files/models/{LM}/m.glb", model), ("/", None)):
            data = (WEB_DIR / "index.html").read_bytes() if data is None else data
            status, plain, body = raw_get(port, path, {})
            assert status == 200 and body == data and "content-encoding" not in plain, path
            assert plain["content-length"] == str(len(data)) and plain["vary"] == "Accept-Encoding", path
            status, headers, body = raw_get(port, path, accept)
            assert status == 200 and headers["content-encoding"] == "gzip" and gunzip(body) == data, path
            assert headers["content-length"] == str(len(body)) and "transfer-encoding" not in headers, path
            assert headers["etag"] == gzip_etag(plain["etag"]) and headers["vary"] == "Accept-Encoding", path
            assert len(body) < len(data) // 2, path
            status, head, nothing = raw_get(port, path, accept, "HEAD")
            assert status == 200 and nothing == b"" and head["content-length"] == headers["content-length"], path
            assert head["content-encoding"] == "gzip" and head["etag"] == headers["etag"], path
            status, again, nothing = raw_get(port, path, accept | {"If-None-Match": headers["etag"]})
            assert status == 304 and nothing == b"" and again["etag"] == headers["etag"], path
            assert again["vary"] == "Accept-Encoding" and "content-encoding" not in again, path
            status, part, piece = raw_get(port, path, accept | {"Range": "bytes=10-109"})
            assert status == 206 and piece == data[10:110] and "content-encoding" not in part, path
            assert part["etag"] == plain["etag"] and part["content-range"] == f"bytes 10-109/{len(data)}", path
            # two requests on one connection: the second is read from where the first one's length says it ends
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            try:
                for _ in range(2):
                    connection.request("GET", path, headers=accept)
                    response = connection.getresponse()
                    assert response.status == 200 and gunzip(response.read()) == data, path
                connection.request("GET", "/api/health")
                response = connection.getresponse()
                assert json.loads(response.read())["ok"] is True and response.getheader("content-encoding") is None
            finally:
                connection.close()
        # a GET with a chunked body: within the body limit the compressed file arrives whole, over it the 413 does;
        # either way the connection stays in step for the next request
        for chunks, expected in ((1, 200), (2, 200), (3, 413)):
            for _ in range(4):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
                try:
                    connection.request("GET", THREE, body=iter([b"a" * (32 * 1024)] * chunks), encode_chunked=True,
                                       headers=accept | {"Transfer-Encoding": "chunked"})
                    response = connection.getresponse()
                    content = response.read()
                    assert response.status == expected, (chunks, response.status)
                    if expected == 200:
                        assert response.getheader("content-encoding") == "gzip" and gunzip(content) == script
                    else:
                        assert json.loads(content) == {"detail": "request body over 65536 bytes"}
                    connection.request("GET", "/api/health")
                    assert json.loads(connection.getresponse().read())["ok"] is True
                finally:
                    connection.close()
        assert raw_get(port, "/api/health", {})[0] == 200
        time.sleep(0.2)
        log = (tmp_path / "server.log").read_text()
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
    assert log.count(f"GET {THREE} HTTP/1.1") == 18, log[-2000:]
    assert not any(word in log for word in ("ERROR", "Exception", "Traceback")), log[-4000:]
