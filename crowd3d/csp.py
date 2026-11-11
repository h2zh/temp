"""The Content-Security-Policy of every response (D34).

The site's pages run only what the site itself serves: scripts, styles and fonts from the same origin, requests
to the same origin, images from the same origin, data: and blob: URLs and the map tile host. A script that got
into a page some other way (a tampered library, an injection found later) cannot load code from elsewhere or send
the contributor and withdrawal tokens the pages keep in localStorage to another origin through a request, an image,
a form or a frame. Everything that is not a page (the API's JSON, a served photo, a model file, a script opened
as a document, an error) may load and run nothing at all.

A page's inline <script> and <style> blocks are allowed by their sha256 (the pages' import maps, the clay viewer's
style block); the hashes are computed from the HTML file, so a page that is edited keeps working without a constant
to update. Nothing else inline is allowed: no event handler attribute, no style attribute, no javascript: URL, no
<style> element made by a script (tests/test_csp.py holds the pages to that; web/ui.js adoptCss is how a page
script adds rules).
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

# The home page's map tiles (web/index.js): images only
TILE_ORIGIN = "https://tile.openstreetmap.org"
HEADER = "Content-Security-Policy"
# Every response that is not one of the site's pages: nothing may be loaded, run, framed, submitted or re-based.
# Inline styles are allowed because a browser shows an image or a text file opened on its own (a served photo, a
# licence) in a document it styles inline: without them Chromium shows the photo unscaled in a corner and logs three
# refusals. A style can load nothing here (default-src 'none' covers images and fonts).
STRICT_POLICY = ("default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
                 "frame-ancestors 'none'")
_HASHED = ("script", "style")       # elements whose inline text a policy names by hash


@dataclass
class InlineBlocks:
    """What a page holds inline. scripts, styles: the text of each <script> without src and each <style>, as a
    browser hashes it (CR LF and CR read as LF). attributes: (tag, attribute) of every event handler and style
    attribute and every javascript: URL, which no hash can allow."""

    scripts: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    attributes: list[tuple[str, str]] = field(default_factory=list)


class _InlineParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found = InlineBlocks()
        self._open: str | None = None       # the hashed element being read
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            url = "".join((value or "").split()).lower()         # browsers drop whitespace inside a URL's scheme
            if name.startswith("on") or name == "style" or url.startswith("javascript:"):
                self.found.attributes.append((tag, name))
        if tag in _HASHED and not (tag == "script" and any(name == "src" for name, _ in attrs)):
            self._open, self._text = tag, []

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._open:
            (self.found.scripts if tag == "script" else self.found.styles).append("".join(self._text))
            self._open = None


def inline_blocks(html: str) -> InlineBlocks:
    """The inline scripts, styles and attributes of an HTML document (InlineBlocks)."""
    parser = _InlineParser()
    parser.feed(html.replace("\r\n", "\n").replace("\r", "\n"))
    parser.close()
    return parser.found


def source_hash(text: str) -> str:
    """The CSP source expression of an inline block: 'sha256-<base64 of the digest of its UTF-8 text>'."""
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


def page_policy(html: str) -> str:
    """The policy of one of the site's pages, from its HTML. Scripts and styles: the site's own files and this
    page's inline blocks by hash, nothing else (no 'unsafe-inline', no 'unsafe-eval'). Images: the site, data:
    (the pages' icon), blob: and the map tiles. Requests (fetch): the site. No plugin, no <base>, forms only to
    the site, and no page in a frame anywhere (with X-Frame-Options: DENY for browsers that do not know
    frame-ancestors). Fonts, media, workers and frames fall back to default-src: the site."""
    found = inline_blocks(html)
    scripts = " ".join(["'self'", *dict.fromkeys(map(source_hash, found.scripts))])
    styles = " ".join(["'self'", *dict.fromkeys(map(source_hash, found.styles))])
    return "; ".join((
        "default-src 'self'",
        f"script-src {scripts}",
        f"style-src {styles}",
        f"img-src 'self' data: blob: {TILE_ORIGIN}",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ))


class PagePolicies:
    """page_policy of each HTML file asked for, read from the file the first time and again whenever the file has
    changed (its size or modification time), so an edited page is served with the hashes of what it holds now."""

    def __init__(self) -> None:
        self._known: dict[str, tuple[tuple[int, int], str]] = {}

    def of(self, path: str | os.PathLike[str], stat_result: os.stat_result) -> str:
        key = (stat_result.st_mtime_ns, stat_result.st_size)
        known = self._known.get(os.fspath(path))
        if known is None or known[0] != key:
            html = Path(path).read_bytes().decode("utf-8", errors="replace")
            known = self._known[os.fspath(path)] = (key, page_policy(html))
        return known[1]
