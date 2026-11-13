"""The structure catalogue (Settings.structure_catalog, eval/landmarks.json): a landmark's entry, and its OSM parts
snapshot and main outline as a StructureSource. Standard library only.

The server asks the catalogue on every upload with GPS (subject.binding) and whenever it plans a stylize job
(jobs.structure_slug). These few lines used to live in crowd3d.structure_pipeline, whose import loads OpenCV and the
whole clay-model family into the server process (D34 part C); they were moved here unchanged, and
crowd3d.structure_pipeline imports every name back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "eval" / "landmarks.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class StructureSource:
    """A landmark's OSM parts snapshot and main outline, and the ENU frame (the scene's: the landmark's centre)."""
    slug: str
    osm_parts: Path
    osm_parts_sha256: str
    outline: Path
    lat: float
    lon: float

    def to_meta(self) -> dict[str, Any]:
        def rel(p: Path) -> str:
            p = Path(p).resolve()
            return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
        return {"slug": self.slug, "osm_parts": rel(self.osm_parts), "osm_parts_sha256": self.osm_parts_sha256,
                "outline": rel(self.outline), "outline_sha256": _sha256(self.outline), "lat": self.lat, "lon": self.lon}


def catalog_entry(catalog: Path, slug: str) -> dict[str, Any] | None:
    """The catalogue's entry of this slug (a list of entries, or {"landmarks": [...]}); None when it has none."""
    data = json.loads(Path(catalog).read_text(encoding="utf-8"))
    entries = data["landmarks"] if isinstance(data, dict) else data
    return next((e for e in entries if e.get("slug") == slug), None)


def find_source(slug: str, lat: float, lon: float, catalog: Path = CATALOG, osm_parts: Path | None = None,
                outline: Path | None = None) -> StructureSource | None:
    """The landmark's snapshot (the catalogue's osm_parts block, its sha256 checked; osm_parts: another file) and
    outline (the catalogue's osm file unless given), in the ENU frame at (lat, lon). None when the landmark has no
    snapshot or no outline. The binding is by slug only."""
    entry = catalog_entry(catalog, slug) or {}
    root = Path(catalog).resolve().parent.parent
    if osm_parts is None:
        block = entry.get("osm_parts")
        if not block:
            return None
        osm_parts = root / block["file"]
        sha = _sha256(osm_parts) if osm_parts.is_file() else None
        if sha != block["sha256"]:
            what = "missing" if sha is None else "not the snapshot the catalogue lists"
            raise ValueError(f"{slug}: {osm_parts} is {what}")
    else:
        sha = _sha256(osm_parts)
    if outline is None:
        file = (entry.get("osm") or {}).get("file")
        if not file:
            return None
        outline = root / file
    return StructureSource(slug, Path(osm_parts), sha, Path(outline), float(lat), float(lon))
