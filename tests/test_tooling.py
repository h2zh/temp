"""The project's tooling, checked mechanically (D34 part C, docs/decisions/D34-part-c-engineering.md).

The requirements files cover what the code imports: until D34 the one requirements.txt lacked pyarrow (nine fast
tests errored in a fresh environment) and left starlette, pydantic and anyio, which crowd3d imports by name, to
whatever FastAPI pulled in. Now every third-party import of crowd3d/, scripts/ and tests/ must be pinned in the
requirements file of its layer, the pins must be the versions installed, the lock must agree with the pins, and the
linter (ruff, configured in pyproject.toml) must have nothing to report.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# which requirements file covers the third-party imports of which directory; each file includes the one before it
LAYERS = (("crowd3d", "requirements.txt"), ("scripts", "requirements-eval.txt"), ("tests", "requirements-dev.txt"))
EVERYTHING = "requirements-dev.txt"
LOCK = "requirements-lock.txt"
INSTALL = f"pip install -r {EVERYTHING} -c {LOCK}"

# pinned although no file of its layer imports it, and why it is needed all the same
NOT_IMPORTED = {
    "python-multipart": "FastAPI reads the upload form with it: without it create_app raises a RuntimeError",
    "pytest-xdist": "pytest's -n option, the parallel run of the fast suite",
    "ruff": "the linter: test_ruff_has_nothing_to_report below and the CI run it",
}
# the calls that import a module given by name
IMPORTING_CALLS = {"__import__", "import_module", "importorskip"}

PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.!+_-]+)")


def normalized(name: str) -> str:
    """A distribution's name as the package index compares it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_lines(name: str) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in (ROOT / name).read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line]


def own_pins(name: str) -> dict[str, str]:
    """The name==version lines of one requirements file. Every line must be one, or the include of another file."""
    found: dict[str, str] = {}
    for line in requirement_lines(name):
        if line.startswith("-r "):
            continue
        match = PIN.fullmatch(line)
        assert match, f"{name}: {line!r} is not an exact pin (name==version)"
        assert normalized(match[1]) not in found, f"{name}: {match[1]} is pinned twice"
        found[normalized(match[1])] = match[2]
    return found


def pins(name: str) -> dict[str, str]:
    """The pins of a requirements file and of the files it includes (-r), which must not pin a package twice."""
    found = own_pins(name)
    for line in requirement_lines(name):
        if line.startswith("-r "):
            included = pins(line[3:].strip())
            assert not set(included) & set(found), f"{name}: {sorted(set(included) & set(found))} pinned twice"
            found |= included
    return found


def own_modules() -> set[str]:
    """The top-level names that are this repository's own code: the package, the scripts, the test helpers."""
    return {"crowd3d"} | {path.stem for folder in ("scripts", "tests") for path in (ROOT / folder).glob("*.py")}


def third_party_imports(directory: str) -> dict[str, list[str]]:
    """Top-level module -> where it is imported ("file:line"), for the Python files of a directory: import statements
    anywhere in a file (inside functions too), and __import__ / import_module / importorskip of a literal name.
    The standard library and the repository's own modules are left out."""
    skipped = set(sys.stdlib_module_names) | own_modules()
    found: dict[str, list[str]] = {}
    for path in sorted((ROOT / directory).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if called in IMPORTING_CALLS and isinstance(node.args[0].value, str):
                    names = [node.args[0].value]
            for name in names:
                top = name.split(".")[0]
                if top not in skipped:
                    found.setdefault(top, []).append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def test_the_import_scan_reads_every_way_a_module_is_imported(tmp_path, monkeypatch):
    """third_party_imports on a folder of its own: plain, dotted, from-imports, imports inside a function, and the
    three calls that take a module's name; not the standard library, the repository's modules, relative imports,
    or a name that is computed."""
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text(
        "import os, alpha\nimport beta.gamma as bg\nfrom delta.x import y\nfrom . import sibling\nfrom .sub import z\n"
        "from crowd3d import geo\nimport benchmark\nimport conftest\n"
        "def f():\n    import epsilon\n    return __import__('zeta').array\n", encoding="utf-8")
    (tmp_path / "pkg" / "sub" / "b.py").write_text(
        "import importlib\nimport pytest\nm = importlib.import_module('eta.theta')\n"
        "p = pytest.importorskip('iota')\nq = pytest.importorskip('crowd3d.stylize')\n"
        "name = 'kappa'\nr = importlib.import_module(name)\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "own_modules", lambda: {"crowd3d", "benchmark", "conftest"})
    found = third_party_imports("pkg")
    assert found == {"alpha": ["pkg/a.py:1"], "beta": ["pkg/a.py:2"], "delta": ["pkg/a.py:3"],
                     "epsilon": ["pkg/a.py:10"], "zeta": ["pkg/a.py:11"], "pytest": ["pkg/sub/b.py:2"],
                     "eta": ["pkg/sub/b.py:3"], "iota": ["pkg/sub/b.py:4"]}


def test_the_requirements_files_are_exact_pins_one_layer_on_the_other(tmp_path, monkeypatch):
    """Each file is name==version lines plus the include of the layer below; a range, a bare name or a package
    pinned in two layers is refused (pip would take whichever it met first)."""
    assert not any(line.startswith("-r ") for line in requirement_lines("requirements.txt"))
    assert requirement_lines("requirements-eval.txt")[0] == "-r requirements.txt"
    assert requirement_lines("requirements-dev.txt")[0] == "-r requirements-eval.txt"
    service, evaluation, everything = (pins(name) for _, name in LAYERS)
    assert set(service) < set(evaluation) < set(everything)
    assert set(NOT_IMPORTED) <= set(everything)                 # (the table above names nothing that is gone)

    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    (tmp_path / "base.txt").write_text("# a comment\nAlpha_Beta==1.0   # why\n\n", encoding="utf-8")
    (tmp_path / "top.txt").write_text("-r base.txt\ngamma==2.0.post1\n", encoding="utf-8")
    assert pins("top.txt") == {"alpha-beta": "1.0", "gamma": "2.0.post1"}
    for bad in ("gamma>=2.0\n", "gamma\n", "gamma==2.0; python_version<'3.12'\n", "gamma==2.0\nGamma==2.1\n",
                "-r base.txt\nalpha.beta==1.1\n"):
        (tmp_path / "bad.txt").write_text(bad, encoding="utf-8")
        with pytest.raises(AssertionError):
            pins("bad.txt")


@pytest.mark.parametrize(("directory", "requirements"), LAYERS)
def test_every_third_party_import_is_pinned_in_the_requirements_file_of_its_layer(directory, requirements):
    """What crowd3d/ imports is pinned in requirements.txt, what scripts/ imports in requirements-eval.txt (or the
    file it includes), what tests/ imports in requirements-dev.txt; and a file pins nothing its layer does not
    import, except the packages NOT_IMPORTED gives a reason for."""
    listed = pins(requirements)
    distributions = importlib.metadata.packages_distributions()
    imported, missing = set(), {}
    for module, where in sorted(third_party_imports(directory).items()):
        provided_by = {normalized(name) for name in distributions.get(module, ())}
        assert provided_by, (f"{module} (imported at {where[0]}) is not installed here, so the package that "
                             f"provides it is unknown: pin it in {requirements}, then {INSTALL}")
        imported |= provided_by
        if not provided_by & set(listed):
            missing[module] = (sorted(provided_by), where[0])
    assert not missing, f"imported in {directory}/ but not pinned in {requirements}: {missing}"
    unused = set(own_pins(requirements)) - imported - set(NOT_IMPORTED)
    assert not unused, f"pinned in {requirements} but imported nowhere in {directory}/: {sorted(unused)}"


@pytest.mark.parametrize("package", sorted(pins(EVERYTHING)))
def test_the_installed_version_is_the_pinned_one(package):
    """The suite runs on the versions the requirements files name. (A package that is not installed is a skip here:
    the tests that need it say so themselves.)"""
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{package} is not installed ({INSTALL})")
    assert installed == pins(EVERYTHING)[package], f"{package}: {INSTALL}"


def test_the_lock_file_holds_every_pin_at_its_version():
    """requirements-lock.txt is the pip freeze of the clean environment the suites last passed in: exact pins only
    (it is used as a constraints file), the directly required packages at the versions the requirements files pin,
    and the packages those pull in besides."""
    locked, direct = own_pins(LOCK), pins(EVERYTHING)
    assert {package: locked.get(package) for package in direct} == direct
    assert len(locked) > len(direct)


def test_ruff_has_nothing_to_report():
    """ruff check with the rules of pyproject.toml ([tool.ruff]) over the whole repository. The CI runs the same
    command as a step of its own."""
    if importlib.util.find_spec("ruff") is None:
        pytest.skip(f"ruff is not installed ({INSTALL})")
    done = subprocess.run([sys.executable, "-m", "ruff", "check", "--no-cache", "."], cwd=ROOT, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
