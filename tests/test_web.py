"""Runs the web UI's JavaScript unit tests (node --test) as part of the pytest suite."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WEB_TESTS = Path(__file__).resolve().parent.parent / "web" / "tests"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_web_unit_tests():
    files = sorted(str(p) for p in WEB_TESTS.glob("*.test.mjs"))
    assert files, f"no *.test.mjs files in {WEB_TESTS}"
    proc = subprocess.run(["node", "--test", *files], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
