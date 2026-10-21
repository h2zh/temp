"""Shared fixtures: fast Settings and a small synthetic photo set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crowd3d.config import Settings
from crowd3d.synthetic import generate_dataset

DATASET_SIZE = (640, 480)
DATASET_SECTORS = [0, 1]
DATASET_PER_SECTOR = 4


@dataclass
class SyntheticDataset:
    """Photos from generate_dataset: per sector DATASET_PER_SECTOR good ones, then one of each bad kind."""

    path: Path
    manifest: dict[str, Any]

    @property
    def lat(self) -> float:
        return self.manifest["landmark"]["lat"]

    @property
    def lon(self) -> float:
        return self.manifest["landmark"]["lon"]

    def entries(self, kind: str | None = None, sector: int | None = None) -> list[dict[str, Any]]:
        """Manifest entries in upload order, optionally filtered by kind ("good", "blurry", ...) and sector."""
        return [
            p
            for p in self.manifest["photos"]
            if (kind is None or p["kind"] == kind) and (sector is None or p["sector"] == sector)
        ]

    def entry(self, kind: str) -> dict[str, Any]:
        (found,) = self.entries(kind)
        return found

    def read(self, entry: dict[str, Any]) -> bytes:
        return (self.path / entry["file"]).read_bytes()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fresh data dir, fake reconstructor, jobs run inline inside the request."""
    return Settings(data_dir=tmp_path / "data", reconstructor="fake", inline_jobs=True)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> SyntheticDataset:
    out = tmp_path_factory.mktemp("synthetic_dataset")
    manifest = generate_dataset(
        out,
        sectors=DATASET_SECTORS,
        photos_per_sector=DATASET_PER_SECTOR,
        num_users=3,
        size=DATASET_SIZE,
        seed=0,
    )
    return SyntheticDataset(out, manifest)
