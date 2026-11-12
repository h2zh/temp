"""A real server process for the tests that need one: uvicorn's own checks of an answer, its worker threads, the
memory of the process (tests/test_api.py, tests/test_gzipfiles.py; pytest puts tests/ on the path).

The server picks its own port. The tests used to ask the kernel for a free port, close the socket and hand the
number to the server; between the two another process can be given the same port, and two test processes running
at once (pytest -n 4) can be handed the same one, after which one test talks to the other's server. With --port 0
the server binds a port nobody else holds, and uvicorn's log says which.
"""

from __future__ import annotations

import http.client
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from crowd3d import api as api_module

LISTENING = re.compile(r"Uvicorn running on http://127\.0\.0\.1:(\d+)")      # uvicorn's start-up line


@contextmanager
def server_process(tmp_path: Path) -> Iterator[tuple[int, subprocess.Popen]]:
    """python -m crowd3d.api on a port of its own, the fake reconstructor, tmp_path/data as its data directory
    (the caller may have put files there) and tmp_path/server.log as its log: (port, process) once it answers.
    No structure catalogue, and glibc's own switch for the malloc threshold is not passed on (the server's
    setting is what the tests measure). Stopped when the block ends."""
    env = {k: v for k, v in os.environ.items() if k != "MALLOC_MMAP_THRESHOLD_"} | {"CROWD3D_STRUCTURE_CATALOG": ""}
    log_path = tmp_path / "server.log"
    with open(log_path, "wb") as log:
        server = subprocess.Popen([sys.executable, "-m", "crowd3d.api", "--port", "0", "--data-dir",
                                   str(tmp_path / "data"), "--reconstructor", "fake"],
                                  cwd=Path(api_module.__file__).resolve().parent.parent, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while True:
            listening = LISTENING.search(log_path.read_text(errors="replace"))
            if listening:
                break
            assert server.poll() is None and time.monotonic() < deadline, log_path.read_text(errors="replace")
            time.sleep(0.1)
        port = int(listening[1])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        try:
            connection.request("GET", "/api/health")
            assert connection.getresponse().status == 200
        finally:
            connection.close()
        yield port, server
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
