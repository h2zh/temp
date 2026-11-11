"""The Content-Security-Policy (crowd3d/csp.py, D34) and the libraries the site serves itself (web/vendor).

What these tests hold the site to: every page is served with a policy that allows its own inline blocks by hash
and nothing else inline; no page and no page script loads a script or a style from another origin; no page script
writes an inline style or evaluates text as code, which the policy would refuse in a browser (a real-browser run is
in docs/decisions/D34-review-fixes.md); and web/vendor holds exactly the files scripts/vendor_web_libs.py pins.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crowd3d import api as api_module
from crowd3d import csp
from crowd3d.api import WEB_DIR, create_app

ROOT = Path(__file__).resolve().parent.parent
VENDOR = WEB_DIR / "vendor"
PAGES = sorted(p for p in WEB_DIR.rglob("*.html") if p.relative_to(WEB_DIR).parts[0] not in api_module.WEB_NOT_SERVED)
# the site's own scripts (not the libraries, not the node tests)
SCRIPTS = sorted(WEB_DIR.glob("*.js"))
IMPORT_MAP = {"three": "./vendor/three/build/three.module.js", "three/addons/": "./vendor/three/examples/jsm/"}


@pytest.fixture(scope="module")
def vendor_tool():
    spec = importlib.util.spec_from_file_location("vendor_web_libs", ROOT / "scripts" / "vendor_web_libs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def sha256_source(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"


def directives(policy: str) -> dict[str, list[str]]:
    """{directive: its sources} of a policy; every directive once."""
    parts = [part.split() for part in policy.split(";")]
    assert all(parts) and len({part[0] for part in parts}) == len(parts), policy
    return {part[0]: part[1:] for part in parts}


# The inline blocks of a page, found another way than csp.inline_blocks finds them (a regular expression over the
# text, not an HTML parser), so the two check each other.
_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.S | re.I)
_STYLE = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.S | re.I)
_TAG = re.compile(r"<[a-zA-Z][^>]*>")
_INLINE_ATTRIBUTE = re.compile(r"""\s(on[a-z]+|style)\s*=|=\s*["']?\s*javascript:""", re.I)


def inline_scripts(html: str) -> list[str]:
    return [body for attrs, body in _SCRIPT.findall(html) if not re.search(r"\bsrc\s*=", attrs)]


def module_specs(js: str) -> list[str]:
    """The module specifiers a script imports: import ... from '...', export ... from '...', import '...' and
    import('...')."""
    specs = re.findall(r"""^\s*(?:import|export)\b[^;'"`]*?\bfrom\s*['"]([^'"\n]+)['"]""", js, re.M)
    specs += re.findall(r"""^\s*import\s*['"]([^'"\n]+)['"]""", js, re.M)
    specs += re.findall(r"""\bimport\(\s*['"]([^'"\n]+)['"]\s*\)""", js)
    return specs


def without_blocks(html: str) -> str:
    return _STYLE.sub("", _SCRIPT.sub("", re.sub(r"<!--.*?-->", "", html, flags=re.S)))


# ---------------------------------------------------------------- csp.py

def test_inline_blocks_reads_scripts_styles_and_inline_attributes():
    html = ('<!doctype html><html><head>\n'
            '<script src="boot.js"></script>\n'
            '<script type="importmap">\n  {"imports": {"a": "./a.js"}}\n</script>\n'
            '<script>if (a < b) document.title = "</div>";</script>\n'
            '<style>\n  body { color: red; }\n</style>\n'
            '<SCRIPT TYPE="module">import "./x.js";</SCRIPT>\n'
            '</head><body><p class="x" data-on="1">text <b>bold</b></p></body></html>')
    found = csp.inline_blocks(html)
    assert found.scripts == ['\n  {"imports": {"a": "./a.js"}}\n', 'if (a < b) document.title = "</div>";',
                             'import "./x.js";']
    assert found.styles == ["\n  body { color: red; }\n"]
    assert found.attributes == []
    assert found.scripts == inline_scripts(html)                         # the tests' own reading agrees

    # a browser reads CR LF and CR as LF before it hashes a block: a checkout with CR LF line ends keeps working
    assert csp.inline_blocks(html.replace("\n", "\r\n")) == found
    assert csp.inline_blocks(html.replace("\n", "\r")) == found

    # a parser may hand a block over in pieces (older versions of html.parser split a script at each "</"): they
    # are one text
    parser = csp._InlineParser()
    parser.handle_data("before")
    parser.handle_starttag("script", [("type", "module")])
    for piece in ("if (a ", "</ b) ", "c();"):
        parser.handle_data(piece)
    parser.handle_endtag("script")
    parser.handle_data("after")
    assert (parser.found.scripts, parser.found.styles) == (["if (a </ b) c();"], [])

    # what no hash can allow is reported, by tag and attribute
    bad = ('<body onload="x()" style="margin:0"><a href="javascript:alert(1)">a</a><a href=" JaVa\nScript:x">b</a>'
           '<img src="x.png" ONERROR="y()"><form action="javascript:z()"></form><p style>p</p>'
           '<a href="#javascript:">fine</a><p data-style="x" title="onclick=1">fine</p></body>')
    assert csp.inline_blocks(bad).attributes == [("body", "onload"), ("body", "style"), ("a", "href"), ("a", "href"),
                                                 ("img", "onerror"), ("form", "action"), ("p", "style")]


def test_source_hash_is_the_sha256_of_the_utf8_text():
    # RFC 4648 / CSP 3: base64 with padding of the SHA-256 digest; the empty text and a known vector
    assert csp.source_hash("") == "'sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU='"
    assert csp.source_hash("abc") == "'sha256-ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0='"
    assert csp.source_hash("众拍") == sha256_source("众拍") != csp.source_hash("众拍 ")


def test_page_policy_allows_the_pages_own_files_its_inline_blocks_by_hash_and_the_tiles():
    html = "<script>a()</script><script src=x.js></script><style>b{}</style><script>a()</script><script>c()</script>"
    got = directives(csp.page_policy(html))
    assert got == {
        "default-src": ["'self'"],
        "script-src": ["'self'", csp.source_hash("a()"), csp.source_hash("c()")],       # each hash once
        "style-src": ["'self'", csp.source_hash("b{}")],
        "img-src": ["'self'", "data:", "blob:", "https://tile.openstreetmap.org"],
        "connect-src": ["'self'"],
        "object-src": ["'none'"],
        "base-uri": ["'none'"],
        "form-action": ["'self'"],
        "frame-ancestors": ["'none'"],
    }
    assert directives(csp.page_policy("<p>no inline block</p>"))["script-src"] == ["'self'"]
    assert directives(csp.page_policy("<p>no inline block</p>"))["style-src"] == ["'self'"]
    assert csp.TILE_ORIGIN == "https://tile.openstreetmap.org"
    assert f"'{csp.TILE_ORIGIN}/{{z}}/{{x}}/{{y}}.png'" in (WEB_DIR / "index.js").read_text(encoding="utf-8")
    # everything else: nothing loads or runs; inline styles only, for the browser's own image and text viewers
    assert directives(csp.STRICT_POLICY) == {"default-src": ["'none'"], "style-src": ["'unsafe-inline'"],
                                             "base-uri": ["'none'"], "form-action": ["'none'"],
                                             "frame-ancestors": ["'none'"]}


def test_page_policies_follow_the_file(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<script>one()</script>", encoding="utf-8")
    policies = csp.PagePolicies()
    first = policies.of(page, page.stat())
    assert csp.source_hash("one()") in first and policies.of(str(page), page.stat()) is first      # kept, not re-read
    stat = page.stat()
    page.write_text("<script>two()</script>", encoding="utf-8")
    os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = policies.of(page, page.stat())
    assert csp.source_hash("two()") in second and csp.source_hash("one()") not in second
    page.write_text("<script>three()!</script>", encoding="utf-8")                    # another size, the same time
    os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert csp.source_hash("three()!") in policies.of(page, page.stat())
    page.write_bytes(b"<script>\xff()</script>")                                         # not UTF-8: no error
    assert "script-src 'self' 'sha256-" in policies.of(page, page.stat())


# ---------------------------------------------------------------- the pages as served

def test_every_page_is_served_with_a_policy_that_covers_its_inline_blocks(client):
    """Each page of the site, as the server sends it: one Content-Security-Policy, which names the sha256 of every
    inline <script> and <style> the page holds (found here by a regular expression, by the server with an HTML
    parser) and nothing looser: no 'unsafe-inline', no 'unsafe-eval', no wildcard, no other host than the map
    tiles for images. And the page holds nothing a hash cannot allow: no event handler attribute, no style
    attribute, no javascript: URL, no <base>, and no script, style or image from another origin."""
    assert {p.name for p in PAGES} >= {"index.html", "landmark.html", "stylized.html", "terms.html"}
    hashed = {"script-src": 0, "style-src": 0}
    for page in PAGES:
        url = "/" + page.relative_to(WEB_DIR).as_posix()
        r = client.get(url)
        html = r.content.decode("utf-8")
        assert r.status_code == 200 and html == page.read_text(encoding="utf-8"), url
        assert r.headers.get_list("content-security-policy") == [csp.page_policy(html)], url
        got = directives(r.headers["content-security-policy"])
        assert list(got) == ["default-src", "script-src", "style-src", "img-src", "connect-src", "object-src",
                             "base-uri", "form-action", "frame-ancestors"], url

        scripts, styles = inline_scripts(html), _STYLE.findall(html)
        assert got["script-src"] == ["'self'", *dict.fromkeys(map(sha256_source, scripts))], url
        assert got["style-src"] == ["'self'", *dict.fromkeys(map(sha256_source, styles))], url
        hashed["script-src"] += len(got["script-src"]) - 1
        hashed["style-src"] += len(got["style-src"]) - 1
        assert "\r" not in html, url             # (the hashes above are of the file's text as it is)

        sources = [source for name in got for source in got[name]]
        assert not any(source.startswith("'unsafe-") or source in ("*", "https:", "http:", "'strict-dynamic'")
                       for source in sources), url
        assert [source for source in sources if "//" in source] == ["https://tile.openstreetmap.org"], url
        assert (got["default-src"], got["connect-src"], got["form-action"]) == (["'self'"],) * 3, url
        assert (got["object-src"], got["base-uri"], got["frame-ancestors"]) == (["'none'"],) * 3, url
        assert got["img-src"] == ["'self'", "data:", "blob:", "https://tile.openstreetmap.org"], url

        # nothing inline but the hashed blocks
        assert csp.inline_blocks(html).attributes == [], url
        markup = without_blocks(html)
        assert [tag for tag in _TAG.findall(markup) if _INLINE_ATTRIBUTE.search(tag)] == [], url
        assert not re.search(r"<base\b", markup, re.I), url
        # every script, style sheet, image and frame comes from the site (a relative URL) or is a data: URL
        loaded = re.findall(r"""<(?:script|link|img|iframe|source|video|audio|embed|object)\b[^>]*?\s(?:src|href|data)\s*=\s*["']([^"']*)""",
                            re.sub(r"<!--.*?-->", "", html, flags=re.S), re.I)
        assert len([u for u in loaded if u.endswith(".js")]) >= 2, url          # (boot.js and the page's module)
        assert loaded and all(u.startswith("data:image/") or (re.fullmatch(r"[\w./-]+", u) and not u.startswith("//"))
                              for u in loaded), (url, loaded)

        # a 304 carries the page's policy too: a browser replaces a stored page's headers with the 304's, and the
        # strict policy of every other response would leave the stored page unable to load anything
        again = client.get(url, headers={"if-none-match": r.headers["etag"]})
        assert again.status_code == 304
        assert again.headers.get_list("content-security-policy") == r.headers.get_list("content-security-policy")
    # the pages' inline blocks today: the two import maps and the clay viewer's style block
    assert hashed == {"script-src": 2, "style-src": 1}
    r = client.get("/")
    assert r.headers["content-security-policy"] == client.get("/index.html").headers["content-security-policy"]


def test_a_page_edited_while_the_server_runs_is_served_with_the_hashes_of_what_it_holds(settings, tmp_path, monkeypatch):
    """The hashes are computed from the HTML file, and again when the file changes: an edited import map or style
    block cannot leave a page whose policy refuses it (a constant in the code would)."""
    web = tmp_path / "web"
    web.mkdir()
    page = web / "page.html"
    page.write_text('<!doctype html><script type="importmap">{"imports": {}}</script><style>a{}</style>',
                    encoding="utf-8")
    (web / "other.js").write_text("export {};", encoding="utf-8")
    monkeypatch.setattr(api_module, "WEB_DIR", web)
    with TestClient(create_app(settings)) as client:
        first = client.get("/page.html")
        got = directives(first.headers["content-security-policy"])
        assert got["script-src"] == ["'self'", sha256_source('{"imports": {}}')]
        assert got["style-src"] == ["'self'", sha256_source("a{}")]
        assert client.get("/other.js").headers["content-security-policy"] == csp.STRICT_POLICY

        stat = page.stat()
        page.write_text('<!doctype html><script type="importmap">{"imports": {"three": "./t.js"}}</script>',
                        encoding="utf-8")
        os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        edited = client.get("/page.html", headers={"if-none-match": first.headers["etag"]})
        assert edited.status_code == 200                                  # (the ETag follows the file too)
        got = directives(edited.headers["content-security-policy"])
        assert got["script-src"] == ["'self'", sha256_source('{"imports": {"three": "./t.js"}}')]
        assert got["style-src"] == ["'self'"]
        # a page added while the server runs; a page is known by its suffix, in either case
        (web / "new.html").write_text("<script>x()</script>", encoding="utf-8")
        assert sha256_source("x()") in client.get("/new.html").headers["content-security-policy"]
        (web / "OLD.HTM").write_text("<script>y()</script>", encoding="utf-8")
        assert sha256_source("y()") in client.get("/OLD.HTM").headers["content-security-policy"]
        (web / "page.html.txt").write_text("<script>z()</script>", encoding="utf-8")
        assert client.get("/page.html.txt").headers["content-security-policy"] == csp.STRICT_POLICY


def test_no_page_script_does_what_the_policy_refuses():
    """The site's own scripts, read as text: none names another origin to load from, none writes an inline style
    (a style attribute as text, a <style> element) or evaluates text as code. A browser refuses each of those under
    the pages' policy, silently for a style (the progress bars showed full, the photo layer's panels lost their
    rules): styles are set through element.style, rules are added with web/ui.js adoptCss."""
    assert len(SCRIPTS) >= 18 and WEB_DIR / "ui.js" in SCRIPTS
    refused = {
        "a style attribute in markup": re.compile(r"""style\s*=\s*\\?["'`$]"""),
        "a style attribute set as text": re.compile(r"""setAttribute\(\s*['"`]style['"`]|\.cssText\b"""),
        # (a CSS declaration as text; Intl's {style: 'percent'} is not one)
        "a style given to el() as text": re.compile(r"""\bstyle:\s*['"`][^'"`\n]*:"""),
        "a <style> element": re.compile(r"""createElement(?:NS)?\([^)]*['"`]style['"`]\)|<style\b"""),
        "an event handler attribute in markup": re.compile(r"""\son[a-z]+\s*=\s*\\?["'`]"""),
        "text evaluated as code": re.compile(r"""\beval\s*\(|new\s+Function\b|set(?:Timeout|Interval)\(\s*['"`]|document\.write|javascript:"""),
        "a worker or a frame": re.compile(r"""new\s+(?:Shared)?Worker\b|createElement\(\s*['"`](?:iframe|object|embed)['"`]"""),
    }
    # The URLs the scripts may name: the map tiles (images) and OpenStreetMap's copyright page (a link); the SVG and
    # XHTML namespaces and http://localhost (the default origin of two URL parsers) are not requests.
    allowed = {"https://tile.openstreetmap.org/{z}/{x}/{y}.png", "https://www.openstreetmap.org/copyright",
               "http://www.w3.org/2000/svg", "http://localhost"}
    for script in SCRIPTS:
        text = script.read_text(encoding="utf-8")
        # the code without its comments (a line comment starts the line or follows a space: "https://" is none)
        code = re.sub(r"(?m)(^|\s)//.*$", r"\1", re.sub(r"/\*.*?\*/", "", text, flags=re.S))
        for what, pattern in refused.items():
            assert pattern.findall(code) == [], (script.name, what)
        urls = set(re.findall(r"""https?://[^\s'"`<>)\\]+""", text))
        if script.name == "strings.js":          # the wording may cite a page (the licence, the OSM copyright)
            assert not any("cdn" in u or "unpkg" in u or "jsdelivr" in u for u in urls), script.name
        else:
            assert urls <= allowed, (script.name, sorted(urls - allowed))
    # the two ways a page script styles what it builds
    ui = (WEB_DIR / "ui.js").read_text(encoding="utf-8")
    assert "else if (key === 'style') Object.assign(node.style, value);" in ui
    assert "doc.adoptedStyleSheets = [...doc.adoptedStyleSheets, sheet];" in ui
    for name in ("photoposes.js", "compare.js"):
        assert re.search(r"\badoptCss\((?:doc|document), CSS\b", (WEB_DIR / name).read_text(encoding="utf-8")), name


# ---------------------------------------------------------------- the libraries the site serves itself

def test_the_pages_load_three_and_leaflet_from_the_site():
    """The import maps of the two pages with a 3-D viewer point at web/vendor/three, every three.js module the page
    scripts import is there, and the home page's script loads Leaflet from web/vendor/leaflet."""
    maps = {}
    for page in PAGES:
        for attrs, body in _SCRIPT.findall(page.read_text(encoding="utf-8")):
            if re.search(r"""type\s*=\s*["']importmap["']""", attrs):
                assert page.name not in maps
                maps[page.name] = json.loads(body)
    assert maps == {"landmark.html": {"imports": IMPORT_MAP}, "stylized.html": {"imports": IMPORT_MAP}}

    imported = set()
    for script in SCRIPTS:
        specs = module_specs(script.read_text(encoding="utf-8"))
        assert all(spec.startswith(("./", "three")) for spec in specs), (script.name, specs)
        imported |= {spec for spec in specs if not spec.startswith(".")}
    assert module_specs("import a from './a.js';\nimport {\n  b,\n} from 'three';\nexport * from \"./c.js\";\n"
                        "import './d.js';\nconst e = await import('three/addons/e.js');") == [
        "./a.js", "three", "./c.js", "./d.js", "three/addons/e.js"]
    assert imported == {"three", "three/addons/controls/OrbitControls.js", "three/addons/lines/LineMaterial.js",
                        "three/addons/lines/LineSegments2.js", "three/addons/lines/LineSegmentsGeometry.js",
                        "three/addons/loaders/GLTFLoader.js", "three/addons/loaders/PLYLoader.js",
                        "three/addons/renderers/CSS2DRenderer.js", "three/addons/utils/BufferGeometryUtils.js"}
    for spec in imported:
        target = IMPORT_MAP["three"] if spec == "three" else IMPORT_MAP["three/addons/"] + spec.removeprefix("three/addons/")
        assert (WEB_DIR / target).is_file(), spec

    index = (WEB_DIR / "index.js").read_text(encoding="utf-8")
    assert "css: { href: 'vendor/leaflet/leaflet.css' }" in index and "js: { src: 'vendor/leaflet/leaflet.js' }" in index
    assert "integrity" not in index and "crossorigin" not in index
    for name in ("leaflet.css", "leaflet.js"):
        assert (VENDOR / "leaflet" / name).is_file()


def test_the_vendored_libraries_import_and_load_only_what_is_in_the_folder():
    """Every import in web/vendor/three is the bare 'three' (the import map) or a relative path to a file that is
    there; leaflet.css names only images that are there; no library file names another origin to load from."""
    three = VENDOR / "three"
    modules = sorted(three.rglob("*.js"))
    assert len(modules) == 9
    for module in modules:
        specs = module_specs(module.read_text(encoding="utf-8"))
        if module.name != "three.module.js":
            assert "three" in specs, module
        for spec in specs:
            assert spec == "three" or (spec.startswith(".") and (module.parent / spec).resolve().is_file()), (module, spec)
            assert spec == "three" or three.resolve() in (module.parent / spec).resolve().parents, (module, spec)
    css = (VENDOR / "leaflet" / "leaflet.css").read_text(encoding="utf-8")
    urls = re.findall(r"url\(([^)]*)\)", css)
    assert set(urls) == {"#default#VML", "images/layers.png", "images/layers-2x.png", "images/marker-icon.png"}
    for url in urls:
        assert url.startswith("#") or (VENDOR / "leaflet" / url).is_file(), url
    leaflet = (VENDOR / "leaflet" / "leaflet.js").read_text(encoding="utf-8")
    for name in ("marker-icon.png", "marker-icon-2x.png", "marker-shadow.png"):      # the default marker's images
        assert name in leaflet and (VENDOR / "leaflet" / "images" / name).is_file()
    # and nothing pulls in more code or styles another way
    assert "@import" not in css and module_specs(leaflet) == []
    for path in [*modules, VENDOR / "leaflet" / "leaflet.js"]:
        assert not re.search(r"\bimportScripts\b|new\s+Worker\b", path.read_text(encoding="utf-8")), path


def test_web_vendor_holds_exactly_the_pinned_files(vendor_tool):
    """web/vendor is what scripts/vendor_web_libs.py wrote: the files PACKAGES lists and no other, each with the
    sha256 SHA256SUMS records (nobody edited a library), each package with its LICENSE; the README names the
    pinned versions and tarball hashes."""
    assert vendor_tool.check() == []
    sums = vendor_tool.read_sums((VENDOR / "SHA256SUMS").read_text(encoding="ascii"))
    on_disk = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*") if p.is_file()}
    assert on_disk == set(sums) | {"SHA256SUMS", "README.md"}
    assert len(sums) == 18 and {"three/LICENSE", "leaflet/LICENSE", "three/build/three.module.js",
                                "leaflet/leaflet.js", "leaflet/leaflet.css"} <= set(sums)
    for path, digest in sums.items():
        assert hashlib.sha256((VENDOR / path).read_bytes()).hexdigest() == digest, path
    # the two files everything else hangs on, pinned here as well (three r160, Leaflet 1.9.4)
    assert sums["three/build/three.module.js"] == "76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495"
    assert sums["leaflet/leaflet.js"] == "db49d009c841f5ca34a888c96511ae936fd9f5533e90d8b2c4d57596f4e5641a"
    assert "REVISION = '160'" in (VENDOR / "three" / "build" / "three.module.js").read_text(encoding="utf-8")
    assert 'version="1.9.4"' in (VENDOR / "leaflet" / "leaflet.js").read_text(encoding="utf-8")
    assert "MIT License" in (VENDOR / "three" / "LICENSE").read_text(encoding="utf-8")
    assert "BSD 2-Clause License" in (VENDOR / "leaflet" / "LICENSE").read_text(encoding="utf-8")

    # byte for byte as the tarballs hold them: Leaflet ships two files with CR LF line ends, which git must not
    # rewrite on anyone's machine (.gitattributes)
    assert b"\r\n" in (VENDOR / "leaflet" / "leaflet.css").read_bytes()
    assert "web/vendor/** -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()

    readme = (VENDOR / "README.md").read_text(encoding="utf-8")
    assert [(p.name, p.version) for p in vendor_tool.PACKAGES] == [("three", "0.160.0"), ("leaflet", "1.9.4")]
    for package in vendor_tool.PACKAGES:
        assert re.fullmatch(r"sha512-[A-Za-z0-9+/]{86}==", package.integrity)
        for fact in (package.version, package.integrity, package.url):
            assert fact in readme, (package.name, fact)
        assert package.url == f"https://registry.npmjs.org/{package.name}/-/{package.name}-{package.version}.tgz"


def test_vendoring_refuses_a_tarball_that_is_not_the_pinned_one(vendor_tool, tmp_path, monkeypatch):
    def tarball(files: dict[str, bytes]) -> bytes:
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return out.getvalue()

    good = tarball({"package/LICENSE": b"licence", "package/dist/lib.js": b"lib();", "package/dist/extra.js": b"no"})
    package = vendor_tool.Package(name="lib", version="1.0.0", integrity=vendor_tool.integrity_of(good),
                                  files={"package/LICENSE": "LICENSE", "package/dist/lib.js": "lib.js"})
    assert vendor_tool.integrity_of(b"") == ("sha512-z4PhNX7vuL3xVChQ1m2AB9Yg5AULVxXcg/SpIdNs6c5H0NE8XYXysP+DGNKHfuwv"
                                             "Y7kxvUdBeoGlODJ6+SfaPg==")
    # only the listed files, under the package's folder; the archive's other members are left
    assert vendor_tool.unpack(package, good) == {"lib/LICENSE": b"licence", "lib/lib.js": b"lib();"}
    assert (package.tarball, package.url) == ("lib-1.0.0.tgz", "https://registry.npmjs.org/lib/-/lib-1.0.0.tgz")

    tampered = tarball({"package/LICENSE": b"licence", "package/dist/lib.js": b"evil();"})
    with pytest.raises(vendor_tool.VendorError, match="integrity"):
        vendor_tool.unpack(package, tampered)
    with pytest.raises(vendor_tool.VendorError, match="integrity"):
        vendor_tool.unpack(package, good + b"\0")
    missing = tarball({"package/LICENSE": b"licence"})
    lacking = vendor_tool.Package(name="lib", version="1.0.0", integrity=vendor_tool.integrity_of(missing),
                                  files=package.files)
    with pytest.raises(vendor_tool.VendorError, match="no file package/dist/lib.js"):
        vendor_tool.unpack(lacking, missing)

    # write_vendor and check, on a folder of their own: an edited, a missing and an extra file are each reported
    monkeypatch.setattr(vendor_tool, "PACKAGES", (package,))
    vendor = tmp_path / "vendor"
    (vendor / "lib" / "old").mkdir(parents=True)
    (vendor / "lib" / "old" / "stale.js").write_bytes(b"stale")
    with pytest.raises(vendor_tool.VendorError):
        vendor_tool.write_vendor({"lib": tampered}, vendor)
    assert (vendor / "lib" / "old" / "stale.js").is_file()              # refused before anything was written
    vendor_tool.write_vendor({"lib": good}, vendor)
    assert sorted(p.relative_to(vendor).as_posix() for p in vendor.rglob("*") if p.is_file()) == [
        "SHA256SUMS", "lib/LICENSE", "lib/lib.js"]
    assert (vendor / "SHA256SUMS").read_text() == (
        f"{hashlib.sha256(b'licence').hexdigest()} *lib/LICENSE\n{hashlib.sha256(b'lib();').hexdigest()} *lib/lib.js\n")
    assert vendor_tool.check(vendor) == []
    (vendor / "README.md").write_text("notes")                           # the hand-written README is no library file
    assert vendor_tool.check(vendor) == []
    (vendor / "lib" / "lib.js").write_bytes(b"lib(); edited();")
    (vendor / "lib" / "LICENSE").unlink()
    (vendor / "lib" / "more.js").write_bytes(b"more")
    assert vendor_tool.check(vendor) == ["lib/LICENSE: in SHA256SUMS but not in the folder",
                                         "lib/more.js: in the folder but not in SHA256SUMS",
                                         "lib/lib.js: does not match SHA256SUMS"]
    (vendor / "SHA256SUMS").unlink()
    assert vendor_tool.check(vendor) == [f"no {vendor / 'SHA256SUMS'}"]
