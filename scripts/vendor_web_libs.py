"""Maintainers: put the web UI's third-party libraries into web/vendor (three.js and Leaflet, from their npm tarballs).

    python scripts/vendor_web_libs.py                    # download the pinned tarballs from the npm registry
    python scripts/vendor_web_libs.py --tarballs DIR     # or take them from DIR (made by: npm pack <name>@<version>)
    python scripts/vendor_web_libs.py --check            # only compare web/vendor with web/vendor/SHA256SUMS

The pages load no script and no style from another origin (D34: the server's Content-Security-Policy allows none,
crowd3d/csp.py), so the libraries are served by the site itself. This script is the only way files get into
web/vendor: nothing there is edited by hand.

PACKAGES pins each package to a version and to the sha512 of its tarball: the registry's dist.integrity
(npm view <name>@<version> dist.integrity). A tarball with any other hash is refused. Only the listed files are
taken: for three.js the module build and exactly the examples/jsm files the pages import with the files those
import (tests/test_csp.py checks that every import in web/vendor/three resolves there), for Leaflet the script, its
stylesheet and the images the stylesheet names, and each package's LICENSE. web/vendor/SHA256SUMS then lists every
file (sha256sum -c reads it); tests/test_csp.py compares web/vendor with it and with PACKAGES.

To update a library: change its version and integrity here (and its file list, if the pages import other files),
run this script, update web/vendor/README.md, run the tests and look at the pages in a browser.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "web" / "vendor"
SUMS_NAME = "SHA256SUMS"
README_NAME = "README.md"                 # written by hand; not a library file, so not in SHA256SUMS
REGISTRY = "https://registry.npmjs.org"
MAX_TARBALL_BYTES = 64 * 1024 * 1024      # three 0.160.0 is 9.4 MB


@dataclass(frozen=True)
class Package:
    name: str
    version: str
    integrity: str                 # "sha512-<base64>": the registry's dist.integrity of the tarball
    files: dict[str, str]          # path in the tarball -> path under web/vendor/<name>/

    @property
    def tarball(self) -> str:
        return f"{self.name}-{self.version}.tgz"

    @property
    def url(self) -> str:
        return f"{REGISTRY}/{self.name}/-/{self.tarball}"


def _same(prefix: str, names: tuple[str, ...]) -> dict[str, str]:
    """names under package/<prefix> in the tarball, kept at the same path under the vendor folder."""
    return {f"package/{prefix}{name}": f"{prefix}{name}" for name in names}


PACKAGES = (
    Package(
        name="three",
        version="0.160.0",
        integrity="sha512-DLU8lc0zNIPkM7rH5/e1Ks1Z8tWCGRq6g8mPowdDJpw1CFBJMU7UoJjC6PefXW7z//SSl0b2+GCw14LB+uDhng==",
        # The package's own layout, so the addons' relative imports and the pages' import map
        # ("three/addons/" -> vendor/three/examples/jsm/) both resolve. web/viewer.js imports PLYLoader and
        # OrbitControls; web/stylized.js GLTFLoader (which imports BufferGeometryUtils), OrbitControls,
        # BufferGeometryUtils, LineSegments2 (which imports the other two lines files), LineSegmentsGeometry,
        # LineMaterial and CSS2DRenderer.
        files={
            "package/LICENSE": "LICENSE",
            **_same("build/", ("three.module.js",)),
            **_same("examples/jsm/", (
                "controls/OrbitControls.js",
                "lines/LineMaterial.js",
                "lines/LineSegments2.js",
                "lines/LineSegmentsGeometry.js",
                "loaders/GLTFLoader.js",
                "loaders/PLYLoader.js",
                "renderers/CSS2DRenderer.js",
                "utils/BufferGeometryUtils.js",
            )),
        },
    ),
    Package(
        name="leaflet",
        version="1.9.4",
        integrity="sha512-nxS1ynzJOmOlHp+iL3FyWqK89GtNL8U8rvlMOsQdTTssxZwCXh8N2NB3GDQOL+YR3XnWyZAxwQixURb+FA74PA==",
        # dist/ flattened: leaflet.css names its images as images/<name>, next to itself
        files={
            "package/LICENSE": "LICENSE",
            "package/dist/leaflet.js": "leaflet.js",
            "package/dist/leaflet.css": "leaflet.css",
            **{f"package/dist/images/{name}": f"images/{name}" for name in (
                "layers.png", "layers-2x.png", "marker-icon.png", "marker-icon-2x.png", "marker-shadow.png")},
        },
    ),
)


class VendorError(Exception):
    pass


def integrity_of(data: bytes) -> str:
    """The npm integrity string of a tarball's bytes: "sha512-" and the base64 of the digest."""
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def unpack(package: Package, tarball: bytes) -> dict[str, bytes]:
    """The package's listed files out of its tarball, by their paths under web/vendor: {"three/LICENSE": bytes}.
    Refuses a tarball whose sha512 is not the pinned one, and one that lacks a listed file. Nothing is extracted
    to disk and no path of the archive is used as a path: members are looked up by the listed names only."""
    got = integrity_of(tarball)
    if got != package.integrity:
        raise VendorError(f"{package.tarball}: integrity {got}, expected {package.integrity}")
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        for source, dest in package.files.items():
            if source not in members:
                raise VendorError(f"{package.tarball}: no file {source}")
            out[f"{package.name}/{dest}"] = tar.extractfile(members[source]).read()
    return out


def sums_text(files: dict[str, bytes]) -> str:
    """SHA256SUMS of files, as sha256sum writes it for binary files: "<hex> *<path>" lines, sorted by path."""
    return "".join(f"{hashlib.sha256(files[path]).hexdigest()} *{path}\n" for path in sorted(files))


def read_sums(text: str) -> dict[str, str]:
    """{path: sha256 hex} of a SHA256SUMS text ("<hex> *<path>" lines, or sha256sum's text-mode "<hex>  <path>")."""
    sums = {}
    for line in text.splitlines():
        digest, _, path = line.partition(" ")
        sums[path.lstrip(" *")] = digest
    return sums


def vendored_files(vendor_dir: Path = VENDOR_DIR) -> dict[str, bytes]:
    """Every library file in the vendor folder, by its path under it (not SHA256SUMS and the README)."""
    return {path.relative_to(vendor_dir).as_posix(): path.read_bytes()
            for path in sorted(vendor_dir.rglob("*"))
            if path.is_file() and path.relative_to(vendor_dir).as_posix() not in (SUMS_NAME, README_NAME)}


def check(vendor_dir: Path = VENDOR_DIR) -> list[str]:
    """What is wrong with the vendor folder, one line each (none: it holds exactly the files of SHA256SUMS, which
    are exactly the files PACKAGES lists)."""
    sums_path = vendor_dir / SUMS_NAME
    if not sums_path.is_file():
        return [f"no {sums_path}"]
    sums = read_sums(sums_path.read_text(encoding="ascii"))
    files = vendored_files(vendor_dir)
    listed = {f"{package.name}/{dest}" for package in PACKAGES for dest in package.files.values()}
    problems = [f"{path}: in {SUMS_NAME} but not in the folder" for path in sorted(set(sums) - set(files))]
    problems += [f"{path}: in the folder but not in {SUMS_NAME}" for path in sorted(set(files) - set(sums))]
    problems += [f"{path}: does not match {SUMS_NAME}" for path in sorted(set(files) & set(sums))
                 if hashlib.sha256(files[path]).hexdigest() != sums[path]]
    problems += [f"{path}: in {SUMS_NAME} but not listed in PACKAGES" for path in sorted(set(sums) - listed)]
    problems += [f"{path}: listed in PACKAGES but not in {SUMS_NAME}" for path in sorted(listed - set(sums))]
    return problems


def write_vendor(tarballs: dict[str, bytes], vendor_dir: Path = VENDOR_DIR) -> dict[str, bytes]:
    """Replaces each package's folder under vendor_dir with the listed files of its tarball and writes SHA256SUMS.
    tarballs: {package name: tarball bytes}. Every tarball is checked and unpacked before anything is written."""
    files: dict[str, bytes] = {}
    for package in PACKAGES:
        files.update(unpack(package, tarballs[package.name]))
    for package in PACKAGES:
        folder = vendor_dir / package.name
        if folder.exists():
            for path in sorted(folder.rglob("*"), reverse=True):      # files before the folders that hold them
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
    for path, data in files.items():
        target = vendor_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (vendor_dir / SUMS_NAME).write_text(sums_text(files), encoding="ascii")
    return files


def fetch(package: Package, tarballs_dir: Path | None) -> bytes:
    """The package's tarball: from tarballs_dir when given, else from the npm registry (HTTPS, certificates
    verified as Python does by default; the proxy and CA settings of the environment apply)."""
    if tarballs_dir is not None:
        return (tarballs_dir / package.tarball).read_bytes()
    with urllib.request.urlopen(package.url, timeout=120) as response:      # (a fixed https URL)
        data = response.read(MAX_TARBALL_BYTES + 1)
    if len(data) > MAX_TARBALL_BYTES:
        raise VendorError(f"{package.url}: over {MAX_TARBALL_BYTES} bytes")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tarballs", type=Path, metavar="DIR",
                        help="take <name>-<version>.tgz from DIR instead of downloading them")
    parser.add_argument("--check", action="store_true", help="only compare web/vendor with SHA256SUMS and PACKAGES")
    args = parser.parse_args(argv)
    if not args.check:
        try:
            files = write_vendor({package.name: fetch(package, args.tarballs) for package in PACKAGES})
        except (VendorError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for package in PACKAGES:
            size = sum(len(data) for path, data in files.items() if path.startswith(package.name + "/"))
            print(f"{package.name} {package.version}: {len(package.files)} files, {size} bytes")
    problems = check()
    for line in problems:
        print(f"error: {line}", file=sys.stderr)
    if not problems:
        print(f"web/vendor matches {SUMS_NAME}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
