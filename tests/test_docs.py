"""The documentation, held to the repository and to the code (D34 part C step 2,
docs/decisions/D34-part-c-engineering.md).

Until D34 the README was one file of 1,191 lines; its reference parts now live in docs/ (API.md, CONFIG.md,
TESTING.md, docs/decisions/). A reference nothing checks drifts, so these tests fail when:

- a relative link or a #anchor of a Markdown file leads nowhere;
- docs/API.md lists a route the application does not serve, or leaves one out (method and path), or the
  application's route table holds an entry of a kind the reference has no row for (a WebSocket route, say);
- the tables of docs/CONFIG.md (every field of Settings and Params: default, how to set it) or the code map of
  docs/ARCHITECTURE.md (every module: its layer, what it imports) are not what scripts/gen_docs.py writes from the
  code, or a row's hand-written cell is empty. The code is the source; `python scripts/gen_docs.py` rewrites the
  generated cells;
- the README grows back, loses the link to a document, or a document gives a path of a development session's
  scratch directory.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from starlette.endpoints import HTTPEndpoint
from starlette.routing import BaseRoute, Host, Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles

from crowd3d.api import WEB_DIR, create_app
from crowd3d.config import Settings

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
README = ROOT / "README.md"
API_DOC = DOCS / "API.md"
CONFIG_DOC = DOCS / "CONFIG.md"
DECISIONS = DOCS / "decisions"
DECISIONS_INDEX = DECISIONS / "README.md"

README_MAX_LINES = 320
# directories that hold no document of the repository: tool state, installed packages, a server's data
NOT_OURS = {".git", ".venv", ".claude", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules", "data"}
# the one place that may spell out a session's scratch directory: the note that says what `<scratch>` stands for
SCRATCH_NOTE = DECISIONS_INDEX
SCRATCH_MARKS = ("/tmp/claude-0", "scratchpad")


@pytest.fixture(scope="module")
def gen_docs():
    spec = importlib.util.spec_from_file_location("gen_docs", ROOT / "scripts" / "gen_docs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def markdown_files() -> list[Path]:
    """Every .md file of the repository: in a git checkout the tracked ones and the new ones git does not ignore;
    elsewhere (an export of the tree) every .md file outside the folders that are no part of it."""
    listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.md"],
                            cwd=ROOT, capture_output=True, check=False) if (ROOT / ".git").exists() else None
    if listed is not None and listed.returncode == 0 and listed.stdout:
        return sorted(ROOT / name for name in listed.stdout.decode().split("\0") if name and (ROOT / name).is_file())
    found = []
    for folder, subfolders, names in os.walk(ROOT):
        subfolders[:] = [name for name in subfolders if name not in NOT_OURS]
        found += [Path(folder) / name for name in names if name.endswith(".md")]
    return sorted(found)


def name_of(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# ------------------------------------------------------------------------------------------------------------
# links and anchors
# ------------------------------------------------------------------------------------------------------------

FENCE = re.compile(r"\s*(`{3,}|~{3,})")
HEADING = re.compile(r" {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
INLINE_LINK = re.compile(r"\]\(\s*<?([^()\s>]+(?:\([^()\s]*\)[^()\s>]*)*)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.M)
ATTRIBUTE = re.compile(r"<(?:img|a|source|video)\b[^>]*?\b(?:src|href)=\"([^\"]+)\"")
NAMED_ANCHOR = re.compile(r"<a\b[^>]*?\b(?:name|id)=\"([^\"]+)\"")
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def prose_lines(text: str) -> list[str]:
    """The lines of a Markdown text with every fenced code block blanked (its lines stay, empty)."""
    out, fence = [], None
    for line in text.split("\n"):
        mark = FENCE.match(line)
        if fence is None and mark:
            fence = mark[1][0] * 3
            out.append("")
        elif fence is not None:
            if mark and mark[1].startswith(fence):
                fence = None
            out.append("")
        else:
            out.append(line)
    return out


def without_code_spans(line: str) -> str:
    return re.sub(r"(`+)(?:(?!\1).)+\1", "", line)


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: its text without the inline markup, lower-cased, every character dropped
    that is not a letter, a mark, a decimal digit, an underscore-like connector, a hyphen or a space, and spaces
    turned into hyphens."""
    parts = re.split(r"(`+)((?:(?!\1).)+)\1", heading)       # text, backticks, a code span's content, text, ...
    for i in range(0, len(parts), 3):                         # outside code spans:
        text = re.sub(r"!?\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^)]*\)", r"\1", parts[i])     # a link's or image's text
        text = re.sub(r"<[^>]+>", "", text)                                              # no HTML tags
        parts[i] = re.sub(r"(\*\*|\*|~~)(?=\S)(.+?)(?<=\S)\1", r"\2", text)             # no emphasis marks
    text = "".join(part for i, part in enumerate(parts) if i % 3 != 1)
    kept = (c for c in text.strip().lower()
            if c in "- " or unicodedata.category(c)[0] in "LM" or unicodedata.category(c) in ("Nd", "Pc"))
    return "".join(kept).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """Every anchor a Markdown text offers: one per heading (a repeated heading gets -1, -2, ...) and the named
    ones (<a name="...">)."""
    found: set[str] = set()
    seen: dict[str, int] = {}
    for line in prose_lines(text):
        heading = HEADING.match(line)
        if heading:
            base = slug(heading[2])
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
        found.update(NAMED_ANCHOR.findall(line))
    return found


def link_targets(text: str) -> list[str]:
    """The targets of the links and images of a Markdown text (inline links, reference definitions, src and href
    of HTML tags), outside code blocks and code spans; targets with a scheme (https:, mailto:) are left out."""
    prose = "\n".join(without_code_spans(line) for line in prose_lines(text))
    targets = INLINE_LINK.findall(prose) + REFERENCE.findall(prose) + ATTRIBUTE.findall(prose)
    return [target for target in targets if not SCHEME.match(target)]


def broken_links(path: Path, text: str, anchors_of: dict[Path, set[str]]) -> list[str]:
    found = []
    for target in link_targets(text):
        file_part, _, fragment = target.partition("#")
        dest = (path.parent / unquote(file_part)).resolve() if file_part else path
        if not dest.is_relative_to(ROOT):
            found.append(f"{name_of(path)}: {target} leaves the repository")
        elif not dest.exists():
            found.append(f"{name_of(path)}: {target} does not exist")
        elif fragment:
            if dest.suffix != ".md":
                found.append(f"{name_of(path)}: {target} has an anchor, but {name_of(dest)} is not a Markdown file")
                continue
            if dest not in anchors_of:
                anchors_of[dest] = anchors(dest.read_text(encoding="utf-8"))
            if unquote(fragment) not in anchors_of[dest]:
                found.append(f"{name_of(path)}: {name_of(dest)} has no heading for #{unquote(fragment)}")
    return found


def test_every_relative_link_and_anchor_of_every_markdown_file_resolves():
    files = markdown_files()
    assert {README, API_DOC, CONFIG_DOC, DECISIONS_INDEX, ROOT / "web" / "vendor" / "README.md"} <= set(files)
    anchors_of: dict[Path, set[str]] = {}
    checked = 0
    broken = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        checked += len(link_targets(text))
        broken += broken_links(path, text, anchors_of)
    assert not broken, "\n".join(broken)
    assert checked > 200          # the README and the indexes alone hold that many: the scan is not empty


def test_the_link_check_reads_markdown_the_way_github_renders_it(tmp_path):
    """The check's own rules, on a made-up page: what counts as a link, and which anchor a heading gets."""
    assert slug("6.13 D34 的验证（地理配准怎么权衡 GPS 和罗盘：离线重放）") == "613-d34-的验证地理配准怎么权衡-gps-和罗盘离线重放"
    assert slug("`Settings`（`crowd3d/config.py`）") == "settingscrowd3dconfigpy"
    assert slug("**Bold** and [a link](x.md) ½ ① snake_case") == "bold-and-a-link---snake_case"
    assert slug("A <b>tag</b> and `<code>/…` kept") == "a-tag-and-code-kept"
    page = "\n".join([
        "# Title", "## Twice", "## Twice", "```", "## not a heading", "[not a link](missing.md)", "```",
        "`[nor this](missing.md)` but [this](other.md#twice-1) and ![image](pic.png \"title\")",
        "[external](https://example.org/a_(b)) <img src=\"shots/a.png\" width=\"3\">", "[ref]: later.md",
    ])
    assert anchors(page) == {"title", "twice", "twice-1"}
    assert link_targets(page) == ["other.md#twice-1", "pic.png", "later.md", "shots/a.png"]


# ------------------------------------------------------------------------------------------------------------
# docs/API.md and the application's routes
# ------------------------------------------------------------------------------------------------------------

def table_rows(text: str, first_cells: tuple[str, ...]) -> list[list[str]]:
    """The rows of every table of a Markdown text whose header starts with these cells."""
    rows, inside = [], False
    for line in prose_lines(text):
        if not line.lstrip().startswith("|"):
            inside = False
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip()[1:-1])]
        if tuple(cells[:len(first_cells)]) == first_cells:
            inside = True
        elif inside and not set("".join(cells)) <= set("-: "):
            rows.append(cells)
    return rows


def documented_routes(text: str) -> set[tuple[str, str]]:
    """(method, path) of every row of the route tables of docs/API.md. A cell may name several paths; a query
    string is no part of the path, and neither is the name of a path parameter."""
    found = set()
    for method, paths, *_ in table_rows(text, ("方法", "路径")):
        assert re.fullmatch(r"[A-Z]+", method), f"docs/API.md: {method!r} is not a method"
        names = re.findall(r"`([^`]+)`", paths)
        assert names, f"docs/API.md: no path in {paths!r}"
        found |= {(method, plain_path(name)) for name in names}
    return found


def plain_path(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path.split("?")[0])


@dataclass
class RouteTable:
    """What walking an application's route table finds, sorted by kind."""
    api: set[tuple[str, str]] = field(default_factory=set)          # (method, path) of every API route
    plain: set[tuple[str, str]] = field(default_factory=set)        # (method, path) of every plain Starlette route
    folders: list[tuple[str, Path]] = field(default_factory=list)   # (path, directory) of every mounted folder
    unknown: list[str] = field(default_factory=list)                # every entry of any other kind, described


def route_table(router, found: RouteTable | None = None) -> RouteTable:
    """Every entry of a router's route table: its API routes, whether or not they are in the OpenAPI schema (the
    upload route sits in a router of its own: an included router is walked the same way), its plain Starlette
    routes and the folders of files it mounts. Whatever else it holds is described in `unknown`: a WebSocket
    route, a Host, a Mount of anything but a folder of files (a sub-application, a group of routes), a subclass of
    Route or Mount this walk does not know, and a route the router keeps outside `routes` (FastAPI's frontend()
    files are held that way). The reference has no kind of row for those, so nothing here could compare them."""
    found = RouteTable() if found is None else found
    for route in router.routes:
        if getattr(route, "original_router", None) is not None:         # a router included in this one
            route_table(route.original_router, found)
        elif isinstance(route, APIRoute):
            found.api |= {(method, plain_path(route.path)) for method in route.methods}
        elif type(route) is Route:
            found.plain |= {(method, route.path) for method in (route.methods or {"ANY"}) - {"HEAD"}}
        elif type(route) is Mount and isinstance(route.app, StaticFiles) and route.app.directory is not None:
            found.folders.append((route.path, Path(route.app.directory)))
        else:
            found.unknown.append(kind_of(route))
    apart = {id(route): route for name, held in vars(router).items() if name != "routes"
             for route in (held if isinstance(held, list | tuple) else [held]) if isinstance(route, BaseRoute)}
    found.unknown += [f"{kind_of(route)} (not in the router's routes)" for route in apart.values()]
    return found


def kind_of(route) -> str:
    """The class of an entry of the route table, with its path or host when it has one, and for a Mount what it
    mounts."""
    where = getattr(route, "path", getattr(route, "host", None))
    mounted = f" of {type(route.app).__name__}" if isinstance(route, Mount) else ""
    return type(route).__name__ + ("" if where is None else f" {where!r}") + mounted


def test_the_api_reference_lists_every_route_the_application_serves_and_no_other(settings):
    """Method and path of every row of docs/API.md against the application's route table: its API routes (as its
    own OpenAPI description lists them, and as walking the table finds them: no route is hidden from one of the
    two), the other routes it registers (/openapi.json), and the two folders it serves files from. The table may
    hold nothing else: an entry of a kind the reference has no row for (a WebSocket route, a Host, a mounted
    sub-application, ...) fails here by name, until docs/API.md describes it and route_table() knows the kind.
    The interactive /docs and /redoc pages are not served (tests/test_api.py), and the reference says so. What is
    not in the route table is not compared: a middleware or an exception handler that answers a request itself."""
    app = create_app(settings)
    table = route_table(app.router)
    assert not table.unknown, (f"the application's route table holds {table.unknown}: docs/API.md has no kind of "
                               "row for that, and this test cannot compare it. Describe it there and teach "
                               "route_table() the kind")
    api = {(method.upper(), plain_path(path)) for path, item in app.openapi()["paths"].items() for method in item}
    assert api == table.api and ("POST", "/api/photos") in api
    other = table.plain
    assert other == {("GET", "/openapi.json")}
    assert sorted(table.folders) == [("", Path(WEB_DIR)), ("/files/models", Path(settings.models_dir))]

    documented = documented_routes(API_DOC.read_text(encoding="utf-8"))
    in_api = {(method, path) for method, path in documented if path.startswith("/api/")}
    assert in_api == api, f"only in docs/API.md: {sorted(in_api - api)}; not in docs/API.md: {sorted(api - in_api)}"
    assert other <= documented, f"not in docs/API.md: {sorted(other - documented)}"

    files = documented - in_api - other
    assert {method for method, _ in files} == {"GET"}
    model_files = {path for _, path in files if path.startswith("/files/models/")}
    pages = {path for _, path in files} - model_files
    assert model_files, "docs/API.md has no row for the model files under /files/models"
    assert pages >= {"/", "/landmark.html", "/terms.html", "/stylized.html"}
    for page in pages:                      # a documented page is a file the web folder really has
        assert (Path(WEB_DIR) / (page.lstrip("/") or "index.html")).is_file(), f"docs/API.md: no web file for {page}"


def test_the_route_walk_names_every_entry_it_cannot_hold_to_the_reference(tmp_path):
    """The walk's own rules, on a made-up application. API routes, plain routes and mounted folders are sorted by
    kind, in an included router as well (a plain route that answers every method is listed as ANY). Every other
    entry is named, wherever it sits: a WebSocket route (FastAPI's and Starlette's), a Host, a subclass of Route,
    a sub-application and a group of routes behind a Mount, and the files of frontend(), which FastAPI keeps
    outside the router's `routes`."""
    def endpoint(request):
        raise NotImplementedError

    async def socket(websocket):
        raise NotImplementedError

    class OwnRoute(Route):
        pass

    app = FastAPI(docs_url=None, redoc_url=None)
    inner = APIRouter()
    app.add_api_route("/api/a", endpoint, methods=["GET", "DELETE"])
    inner.add_api_route("/api/b/{name}", endpoint, methods=["POST"], include_in_schema=False)
    app.include_router(inner)
    app.mount("/files", StaticFiles(directory=tmp_path))
    known = RouteTable(api={("GET", "/api/a"), ("DELETE", "/api/a"), ("POST", "/api/b/{}")},
                       plain={("GET", "/openapi.json")}, folders=[("/files", tmp_path)])
    assert route_table(app.router) == known

    inner.add_route("/plain", endpoint, methods=["PUT"])                # known kinds, in the included router
    inner.routes.append(Route("/every-method", HTTPEndpoint))           # a class as endpoint: no list of methods
    inner.mount("/more", StaticFiles(directory=tmp_path / "more", check_dir=False))
    known.plain |= {("PUT", "/plain"), ("ANY", "/every-method")}
    known.folders.insert(0, ("/more", tmp_path / "more"))               # in the table's order: `inner` comes first
    assert route_table(app.router) == known

    app.add_api_websocket_route("/api/ws", socket)
    inner.add_api_websocket_route("/api/inner-ws", socket)
    app.router.routes += [WebSocketRoute("/ws", socket), OwnRoute("/own", endpoint), Host("admin.example.org", inner),
                          Mount("/group", routes=[Route("/hidden", endpoint)])]
    app.mount("/sub", FastAPI())
    assert app.router.routes == app.routes and len(app.routes) == 10
    inner.frontend("/inner-app", directory=tmp_path, check_dir=False)
    app.frontend("/app", directory=tmp_path, check_dir=False)
    assert len(app.routes) == 10                                        # frontend() added nothing to `routes`
    known.unknown = ["APIWebSocketRoute '/api/inner-ws'", "_FrontendRouteGroup (not in the router's routes)",
                     "APIWebSocketRoute '/api/ws'", "WebSocketRoute '/ws'", "OwnRoute '/own'",
                     "Host 'admin.example.org'", "Mount '/group' of Router", "Mount '/sub' of FastAPI",
                     "_FrontendRouteGroup (not in the router's routes)"]
    assert route_table(app.router) == known


def test_the_route_table_reader_takes_method_and_path_from_each_row():
    text = "\n".join([
        "| 方法 | 路径 | 说明 |", "|---|---|---|",
        "| GET | `/api/landmarks/{id}/poses` | a `code` span \\| an escaped bar |",
        "| GET | `/`、`/landmark.html?id=…` | two pages |",
        "", "| 字段 | 默认值 |", "|---|---|", "| `x` | 1 |",
        "```", "| POST | `/api/not-a-row` | in a code block |", "```",
    ])
    assert documented_routes(text) == {("GET", "/api/landmarks/{}/poses"), ("GET", "/"), ("GET", "/landmark.html")}


# ------------------------------------------------------------------------------------------------------------
# the generated tables: docs/CONFIG.md and the code map of docs/ARCHITECTURE.md
# ------------------------------------------------------------------------------------------------------------

def test_the_config_reference_and_the_code_map_are_what_the_code_gives(gen_docs):
    """Every field of Settings and Params with its default and the ways to set it, every module with its layer and
    its imports: the documents equal what scripts/gen_docs.py writes from the code, and no row lacks its
    hand-written cell (what the field means, what the module is for)."""
    assert gen_docs.problems() == []
    documents = {name_of(path): [table.name for table in tables] for path, tables in gen_docs.documents().items()}
    assert documents == {"docs/CONFIG.md": ["settings", "params"],
                         "docs/ARCHITECTURE.md": ["codemap-crowd3d", "codemap-scripts", "codemap-cycles"]}
    assert len(gen_docs.settings_rows()) == len(Settings.__dataclass_fields__) >= 45
    assert {row[0] for row in gen_docs.codemap_rows("crowd3d")} == {f"`crowd3d/{path.name}`"
                                                                   for path in (ROOT / "crowd3d").glob("*.py")}


def test_a_changed_default_a_new_field_and_a_removed_field_change_the_config_table(gen_docs):
    """What the check above would see: a default that changed in the code, a field the document does not know,
    and a field the code no longer has."""
    @dataclass
    class Changed(Settings):
        num_sectors: int = 9
        brand_new: bool = False

    (table,) = [table for table in gen_docs.TABLES if table.name == "settings"]
    text = CONFIG_DOC.read_text(encoding="utf-8")
    assert gen_docs.render(text, (table,)) == text
    assert "| `num_sectors` | `8` | 基准配置 |" in text and "brand_new" not in text

    changed = gen_docs.render(text, (table,), {"settings": gen_docs.settings_rows(Changed)})
    assert changed != text
    assert "| `num_sectors` | `9` | 基准配置 |" in changed and "| `num_sectors` | `8` |" not in changed
    assert "| `brand_new` | `False` | — |" in changed           # not tunable by a benchmark config, no variable
    assert gen_docs.unwritten(changed, (table,)) == ["settings: `brand_new`"]
    assert gen_docs.unwritten(text, (table,)) == []

    without = [row for row in gen_docs.settings_rows() if row[0] != "`num_sectors`"]
    assert "`num_sectors` | `8`" not in gen_docs.render(text, (table,), {"settings": without})


def test_the_settings_table_says_how_each_field_can_be_set(gen_docs, monkeypatch):
    """The environment variables and command line options come from the source (the default factories of
    crowd3d.config, the replace() calls of crowd3d.api.main), the benchmark's whitelist from crowd3d.params; and
    the defaults shown are those of an environment with none of the variables set."""
    assert gen_docs.settings_env() == {"data_dir": ["CROWD3D_DATA_DIR"], "reconstructor": ["CROWD3D_RECONSTRUCTOR"],
                                       "keep_work_dirs": ["CROWD3D_KEEP_WORK_DIRS"],
                                       "structure_catalog": ["CROWD3D_STRUCTURE_CATALOG"]}
    assert gen_docs.settings_cli() == {"data_dir": "--data-dir", "reconstructor": "--reconstructor"}
    monkeypatch.setenv("CROWD3D_DATA_DIR", "/somewhere/else")
    monkeypatch.setenv("CROWD3D_RECONSTRUCTOR", "fake")
    monkeypatch.setenv("CROWD3D_KEEP_WORK_DIRS", "1")
    monkeypatch.setenv("CROWD3D_STRUCTURE_CATALOG", "")
    rows = {row[0]: row[1:] for row in gen_docs.settings_rows()}
    assert rows["`data_dir`"] == ("`data`", "`$CROWD3D_DATA_DIR`、`--data-dir`")
    assert rows["`reconstructor`"] == ("`'colmap'`", "`$CROWD3D_RECONSTRUCTOR`、`--reconstructor`")
    assert rows["`keep_work_dirs`"] == ("`False`", "`$CROWD3D_KEEP_WORK_DIRS`")
    assert rows["`structure_catalog`"] == ("`eval/landmarks.json`", "`$CROWD3D_STRUCTURE_CATALOG`")
    assert rows["`max_upload_bytes`"] == ("`26_214_400（25 MiB）`", "基准配置")
    assert rows["`max_concurrent_decodes`"] == ("`2`", "—")
    assert os.environ["CROWD3D_DATA_DIR"] == "/somewhere/else"       # the variables are put back


def test_the_code_map_reads_imports_layers_and_cycles_from_the_source(gen_docs, tmp_path):
    """The rules of the code map on a made-up repository: an import at the top of a module is a dependency at
    import time, one inside a function is listed apart, one under TYPE_CHECKING is none; `from crowd3d import x`
    and `from . import x` fetch the module x, not the package; a script names another script by its bare name;
    a module's layer is one above the highest module it imports at import time; modules that import each other
    through a function are a cycle."""
    package, scripts = tmp_path / "crowd3d", tmp_path / "scripts"
    package.mkdir()
    scripts.mkdir()
    (package / "__init__.py").write_text('"""package"""\n')
    (package / "base.py").write_text("import os\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
                                     "    from .top import Thing\n")
    (package / "mid.py").write_text("from . import base\nfrom .base import x\n\ndef late():\n    from . import top\n")
    (package / "top.py").write_text("from crowd3d import mid\nimport crowd3d.base\n\nclass A:\n"
                                    "    def f(self):\n        import crowd3d\n")
    (scripts / "tool.py").write_text("from crowd3d.top import A\n")
    (scripts / "runner.py").write_text("import tool\nimport numpy\n\ndef main():\n    from crowd3d import base\n")
    graph = gen_docs.imports(tmp_path)
    assert graph == {
        "crowd3d/__init__.py": (set(), set()),
        "crowd3d/base.py": (set(), set()),
        "crowd3d/mid.py": ({"crowd3d/base.py"}, {"crowd3d/top.py"}),
        "crowd3d/top.py": ({"crowd3d/mid.py", "crowd3d/base.py"}, {"crowd3d/__init__.py"}),
        "scripts/runner.py": ({"scripts/tool.py"}, {"crowd3d/base.py"}),
        "scripts/tool.py": ({"crowd3d/top.py"}, set()),
    }
    assert gen_docs.layers(graph) == {"crowd3d/__init__.py": 0, "crowd3d/base.py": 0, "crowd3d/mid.py": 1,
                                      "crowd3d/top.py": 2, "scripts/tool.py": 3, "scripts/runner.py": 4}
    assert gen_docs.cycles(graph) == [["crowd3d/mid.py", "crowd3d/top.py"]]
    assert gen_docs.codemap_rows("crowd3d", tmp_path) == [
        ("`crowd3d/__init__.py`", "0", "—", "—"), ("`crowd3d/base.py`", "0", "—", "—"),
        ("`crowd3d/mid.py`", "1", "`base`", "`top`"), ("`crowd3d/top.py`", "2", "`base`、`mid`", "`crowd3d`")]
    assert gen_docs.codemap_rows("scripts", tmp_path) == [
        ("`scripts/tool.py`", "3", "`top`", "—"), ("`scripts/runner.py`", "4", "`scripts/tool`", "`base`")]
    assert gen_docs.cycle_rows(tmp_path) == [("`mid`、`top`", "`mid` → `top`")]

    (package / "base.py").write_text("from . import top\n")          # now a cycle at import time
    with pytest.raises(ValueError, match="import each other at import time"):
        gen_docs.layers(gen_docs.imports(tmp_path))


def test_the_only_module_nothing_imports_is_the_one_the_documents_say(gen_docs):
    """The code map (docs/ARCHITECTURE.md), docs/STATUS.md and docs/decisions/README.md say that nothing in the
    repository calls crowd3d/tilt.py (D25-T): only its unit tests do, and the driver that measured the tilt of the
    tower of Pisa was never committed. An entry point for it, or another module left without a caller, makes those
    sentences wrong: this test fails then, and they are to be brought up to date with it."""
    graph = gen_docs.imports()
    imported = set().union(*(at_import | in_function for at_import, in_function in graph.values()))
    uncalled = {name for name in graph if name.startswith("crowd3d/")} - imported
    assert uncalled == {"crowd3d/__init__.py", "crowd3d/tilt.py"}
    for document in (DOCS / "ARCHITECTURE.md", DOCS / "STATUS.md", DECISIONS_INDEX):
        assert "tilt.py" in document.read_text(encoding="utf-8")


def test_a_generated_table_keeps_the_hand_written_cells_and_follows_the_code(gen_docs, tmp_path):
    """The round trip on a small document: rows come out in the code's order with the cells the code gives; the
    hand-written last cell of each row stays (an escaped bar in it too); a new key gets an empty cell, which is
    reported; a missing marker is an error, not a silent pass."""
    table = gen_docs.Table("demo", tmp_path / "doc.md", ("键", "值", "说明"), lambda: [("`a`", "1"), ("`b`", "2")])
    begin, end = gen_docs.markers("demo")
    text = f"before\n{begin}\n| 键 | 值 | 说明 |\n|---|---|---|\n| `b` | 9 | second \\| kept |\n| `gone` | 0 | lost |\n{end}\nafter\n"
    rendered = gen_docs.render(text, (table,))
    assert rendered == (f"before\n{begin}\n| 键 | 值 | 说明 |\n|---|---|---|\n| `a` | 1 | |\n"
                        f"| `b` | 2 | second \\| kept |\n{end}\nafter\n")
    assert gen_docs.unwritten(rendered, (table,)) == ["demo: `a`"]
    assert gen_docs.render(rendered, (table,)) == rendered
    with pytest.raises(ValueError, match="exactly one"):
        gen_docs.render("no markers here", (table,))


# ------------------------------------------------------------------------------------------------------------
# the layout: a short README, an index of the decisions, no scratch paths
# ------------------------------------------------------------------------------------------------------------

def test_the_readme_stays_short_and_links_every_document():
    """The README is the way in, not the reference: at most README_MAX_LINES lines, no section per decision (those
    are files under docs/decisions/), and a link to every document under docs/ and to the decisions' index."""
    text = README.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert len(lines) <= README_MAX_LINES, f"README.md has {len(lines)} lines"
    assert not [line for line in prose_lines(text) if re.match(r"#+ *D\d+\b", line)]
    linked = {(ROOT / unquote(target.partition("#")[0])).resolve() for target in link_targets(text)}
    documents = set(DOCS.glob("*.md")) | {DECISIONS_INDEX}
    assert len(documents) >= 16
    assert not documents - linked, f"README.md links no {sorted(name_of(path) for path in documents - linked)}"


def test_the_decisions_index_links_every_decision_record():
    text = DECISIONS_INDEX.read_text(encoding="utf-8")
    linked = {(DECISIONS / unquote(target.partition("#")[0])).resolve() for target in link_targets(text)}
    records = set(DECISIONS.glob("*.md")) - {DECISIONS_INDEX}
    assert len(records) >= 6
    assert not records - linked, f"the index links no {sorted(path.name for path in records - linked)}"


def test_no_document_gives_a_session_scratch_path():
    """Numbers in the documents come with pointers into the scratch directory of the development session that
    produced them. Those directories are not in the repository and cannot be rebuilt from it, so a pointer is
    written `<scratch>/...`, and one note (docs/decisions/README.md) says what that stands for."""
    assert "<scratch>" in SCRATCH_NOTE.read_text(encoding="utf-8")
    found = [f"{name_of(path)}: {mark}" for path in markdown_files() if path != SCRATCH_NOTE
             for mark in SCRATCH_MARKS if mark in path.read_text(encoding="utf-8")]
    assert not found, "\n".join(found)
