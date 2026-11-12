"""Shared fixtures: fast Settings and a small synthetic photo set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crowd3d import api as api_module
from crowd3d import ingest as ingest_module
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


@pytest.fixture(scope="session", autouse=True)
def server_starts_leave_the_allocator_alone():
    """A server pins glibc's mmap threshold for the process it runs in when it starts (the lifespan of
    api.create_app calls ingest.pin_malloc_threshold with Settings.malloc_mmap_threshold_bytes), once and for good.
    The `settings` fixture below switches that off, but code that makes its own Settings does not: the evaluation
    harness (run_crowd, rescore) and the end-to-end fixtures started their servers with the default, so every test
    that ran after one of them in the same process ran on a pinned allocator, and which tests those were depended
    on the order of the run (and, under pytest-xdist, on the worker). For the whole session the server's start-up
    call does nothing in the test process. (For the session, not per test: a module's fixtures start their servers
    before a test's own fixtures are set up.) What the call does to a process is tested where it cannot leak:
    tests/test_ingest.py (a C library that records its calls) and the real server processes of tests/test_api.py."""
    real = api_module.pin_malloc_threshold
    api_module.pin_malloc_threshold = lambda threshold_bytes: False
    try:
        yield
    finally:
        api_module.pin_malloc_threshold = real


@pytest.fixture(autouse=True)
def the_test_process_keeps_its_allocator():
    """A test that pins the allocator of the test process some other way fails itself, here, instead of changing
    what the tests after it run on."""
    yield
    assert ingest_module._malloc_pinned_at is None and ingest_module._trim_libc is None, \
        "this test left the allocator of the test process pinned (ingest.pin_malloc_threshold)"


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fresh data dir, fake reconstructor, jobs run inline inside the request. The C allocator of the test process is
    left alone (a server pins glibc's mmap threshold when it starts: Settings.malloc_mmap_threshold_bytes)."""
    return Settings(data_dir=tmp_path / "data", reconstructor="fake", inline_jobs=True, malloc_mmap_threshold_bytes=0)


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
