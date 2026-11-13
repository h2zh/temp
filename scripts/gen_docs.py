"""Keeps the tables of the documentation that restate the code equal to the code.

    python scripts/gen_docs.py            # rewrite the generated cells of the documents in place
    python scripts/gen_docs.py --check    # change nothing; exit 1 and show what differs

Which way round: the code is the source, the documents follow. Three kinds of table are generated:

    docs/CONFIG.md        `settings`  every field of crowd3d.config.Settings: default, how it can be set
                          `params`    every field of crowd3d.params.Params: default
    docs/ARCHITECTURE.md  `codemap-crowd3d`, `codemap-scripts`  every module of crowd3d/ and scripts/: its layer
                          and the repository modules it imports
                          `codemap-cycles`  the groups of modules that import each other through imports made
                          inside functions

In each table the first column is the key (a field, a file) and the last column is written by hand (what the field
means, what the module is for). The script reads the document, rebuilds every other cell from the code, and puts
the hand-written cell back unchanged; rows come out in the code's order. A key the document does not have yet gets
an empty last cell, a key the code no longer has loses its row. tests/test_docs.py fails when a document is not
what this script writes, or when a hand-written cell is empty, so a field, a default or an import cannot change
without the document.

A table sits between two marker lines in its document:

    <!-- gen_docs:begin settings -->
    | ... |
    <!-- gen_docs:end settings -->
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import os
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:      # run as a script: the crowd3d package is not installed
    sys.path.insert(0, str(REPO_ROOT))

from crowd3d import params as params_module  # noqa: E402  (REPO_ROOT, assigned above, puts the package on the path)
from crowd3d.config import Settings  # noqa: E402
from crowd3d.params import Params  # noqa: E402

CONFIG_DOC = REPO_ROOT / "docs" / "CONFIG.md"
CODEMAP_DOC = REPO_ROOT / "docs" / "ARCHITECTURE.md"
CODE_DIRS = ("crowd3d", "scripts")
ENV_PREFIX = "CROWD3D_"
MIB = 1024 * 1024
NOTHING = "—"

Row = tuple[str, ...]           # the generated cells of one table row; the first one is the row's key


# ---------------------------------------------------------------------------------------------------------------
# Settings and Params
# ---------------------------------------------------------------------------------------------------------------

def literal(value: Any) -> str:
    """A default as the table shows it: the Python literal, a path relative to the repository when it lies inside
    it. Integers from 10,000 up are written with underscores, and a whole number of MiB says so."""
    if isinstance(value, Path):
        inside = value.is_absolute() and value.is_relative_to(REPO_ROOT)
        return (value.relative_to(REPO_ROOT) if inside else value).as_posix()
    if isinstance(value, bool) or not isinstance(value, int):
        return repr(value)
    text = f"{value:_}" if abs(value) >= 10_000 else str(value)
    return f"{text}（{value // MIB} MiB）" if value >= MIB and value % MIB == 0 else text


def code(text: str) -> str:
    return f"`{text}`"


def _class_fields(source: Path, class_name: str) -> dict[str, ast.expr | None]:
    """field name -> the expression of its default in the class body (None: no default)."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    (cls,) = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    return {node.target.id: node.value for node in cls.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def settings_env(source: Path | None = None, class_name: str = "Settings") -> dict[str, list[str]]:
    """field -> the environment variables its default reads, found in the source: the CROWD3D_* string constants
    in the field's default expression and in the module-level functions that expression names (default_factory)."""
    source = source or Path(sys.modules[Settings.__module__].__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def constants(node: ast.AST) -> list[str]:
        return [n.value for n in ast.walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith(ENV_PREFIX)]

    found: dict[str, list[str]] = {}
    for name, default in _class_fields(source, class_name).items():
        if default is None:
            continue
        names = constants(default)
        for node in ast.walk(default):
            if isinstance(node, ast.Name) and node.id in functions:
                names += constants(functions[node.id])
        if names:
            found[name] = sorted(set(names))
    return found


def settings_cli(source: Path | None = None) -> dict[str, str]:
    """field -> the option of `python -m crowd3d.api` that sets it, found in the source of crowd3d.api.main: the
    keywords of its replace(settings, field=args.option) calls."""
    source = source or REPO_ROOT / "crowd3d" / "api.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    (main,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    found: dict[str, str] = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "replace":
            for keyword in node.keywords:
                value = keyword.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "args":
                    found[keyword.arg] = "--" + value.attr.replace("_", "-")
    return found


@contextmanager
def _without_env(names: set[str]) -> Iterator[None]:
    """The defaults the table shows are the ones with none of the settings' environment variables set."""
    saved = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def settings_rows(cls: type = Settings, tunable: tuple[str, ...] | None = None) -> list[Row]:
    """Per field of Settings: the field, its default, and the ways to set it other than Settings(...) in code: its
    environment variable, the server's command line option, a benchmark configuration (eval/configs/*.json)."""
    env = settings_env()
    cli = settings_cli()
    tunable = params_module.SETTINGS_NAMES if tunable is None else tunable
    with _without_env({name for names in env.values() for name in names}):
        defaults = cls()
    rows = []
    for field in dataclasses.fields(cls):
        ways = [code("$" + name) for name in env.get(field.name, [])]
        if field.name in cli:
            ways.append(code(cli[field.name]))
        if field.name in tunable:
            ways.append("基准配置")
        rows.append((code(field.name), code(literal(getattr(defaults, field.name))), "、".join(ways) or NOTHING))
    return rows


def params_rows(cls: type = Params) -> list[Row]:
    """Per field of Params: the field and its default (every one is set the same way, see the document)."""
    defaults = cls()
    return [(code(field.name), code(literal(getattr(defaults, field.name)))) for field in dataclasses.fields(cls)]


# ---------------------------------------------------------------------------------------------------------------
# The code map: which module imports which
# ---------------------------------------------------------------------------------------------------------------

def modules(root: Path = REPO_ROOT) -> dict[str, Path]:
    """Every .py file of crowd3d/ and scripts/ by the path the code map shows ("crowd3d/geo.py")."""
    return {path.relative_to(root).as_posix(): path for folder in CODE_DIRS for path in sorted((root / folder).glob("*.py"))}


def _imported(importer: str, node: ast.stmt, known: dict[str, Path]) -> list[str]:
    """The repository modules one import statement fetches. `from crowd3d import geo` and `from . import geo`
    fetch crowd3d/geo.py, not the package; a script names another script by its bare name (they share a folder)."""
    folder = importer.split("/")[0]

    def module_file(dotted: str) -> str | None:
        parts = dotted.split(".")
        if folder == "scripts" and len(parts) == 1 and f"scripts/{parts[0]}.py" in known:
            return f"scripts/{parts[0]}.py"
        if parts[0] != "crowd3d":
            return None
        path = "crowd3d/__init__.py" if len(parts) == 1 else f"crowd3d/{parts[1]}.py" if len(parts) == 2 else None
        return path if path in known else None

    if isinstance(node, ast.Import):
        return [found for alias in node.names if (found := module_file(alias.name))]
    if node.level:
        if folder != "crowd3d" or node.level != 1:
            return []
        base = "crowd3d" + (f".{node.module}" if node.module else "")
    else:
        base = node.module or ""
    fetched = []
    from_base = False
    for alias in node.names:
        submodule = module_file(f"{base}.{alias.name}")
        if submodule:
            fetched.append(submodule)
        else:
            from_base = True
    if from_base and (found := module_file(base)):
        fetched.append(found)
    return fetched


def _module_imports(name: str, path: Path, known: dict[str, Path]) -> tuple[set[str], set[str]]:
    """The repository modules one file imports: (when it is imported, only inside a function)."""
    at_import: set[str] = set()
    in_function: set[str] = set()

    def visit(node: ast.AST, inside_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                for other in child.orelse:          # the body never runs; an else branch does
                    visit(other, inside_function)
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                (in_function if inside_function else at_import).update(_imported(name, child, known))
            visit(child, inside_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))

    visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), False)
    at_import.discard(name)
    return at_import, in_function - at_import - {name}


def imports(root: Path = REPO_ROOT) -> dict[str, tuple[set[str], set[str]]]:
    """module -> (the repository modules it imports when it is imported, the ones it imports only inside a
    function). An import under `if TYPE_CHECKING:` is neither: it never runs. An import in a class body or under
    a try or an if at the top of the module runs when the module is imported."""
    known = modules(root)
    return {name: _module_imports(name, path, known) for name, path in known.items()}


def layers(graph: dict[str, tuple[set[str], set[str]]]) -> dict[str, int]:
    """module -> its layer: 0 imports no repository module when it is imported, any other one is one above the
    highest module it imports then. Imports made inside functions do not count (they are how the remaining
    cycles are written); a cycle among the imports at import time is an error."""
    level: dict[str, int] = {}
    visiting: list[str] = []

    def of(name: str) -> int:
        if name not in level:
            if name in visiting:
                raise ValueError("modules import each other at import time: " + " -> ".join([*visiting, name]))
            visiting.append(name)
            level[name] = 1 + max((of(dep) for dep in graph[name][0]), default=-1)
            visiting.pop()
        return level[name]

    for name in graph:
        of(name)
    return level


def _shown(importer: str, names: set[str]) -> str:
    """Imported modules as the code map lists them: a crowd3d module by its bare name, a script as scripts/name."""
    def short(name: str) -> str:
        return name.removesuffix(".py").removeprefix("crowd3d/") if name != "crowd3d/__init__.py" else "crowd3d"
    return "、".join(code(short(name)) for name in sorted(names, key=short)) or NOTHING


def codemap_rows(folder: str, root: Path = REPO_ROOT) -> list[Row]:
    """Per module of one folder, lowest layer first: the file, its layer, what it imports at import time, and what
    it imports only inside functions."""
    graph = imports(root)
    level = layers(graph)
    names = sorted((name for name in graph if name.startswith(folder + "/")), key=lambda name: (level[name], name))
    return [(code(name), str(level[name]), _shown(name, graph[name][0]), _shown(name, graph[name][1])) for name in names]


def cycles(graph: dict[str, tuple[set[str], set[str]]]) -> list[list[str]]:
    """The groups of modules that import each other once the imports made inside functions count too (the
    strongly connected components with more than one module), each sorted, the groups sorted."""
    edges = {name: top | late for name, (top, late) in graph.items()}
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    found: list[list[str]] = []

    def visit(name: str) -> None:           # Tarjan's algorithm
        index[name] = low[name] = len(index)
        stack.append(name)
        for dep in sorted(edges[name]):
            if dep not in index:
                visit(dep)
                low[name] = min(low[name], low[dep])
            elif dep in stack:
                low[name] = min(low[name], index[dep])
        if low[name] == index[name]:
            group = []
            while True:
                group.append(stack.pop())
                if group[-1] == name:
                    break
            if len(group) > 1:
                found.append(sorted(group))

    for name in sorted(edges):
        if name not in index:
            visit(name)
    return sorted(found)


def cycle_rows(root: Path = REPO_ROOT) -> list[Row]:
    """Per group of modules that import each other: the group, and the imports inside functions that close it."""
    graph = imports(root)
    rows = []
    for group in cycles(graph):
        late = [f"{_shown(name, {name})} → {_shown(name, {dep})}"
                for name in group for dep in sorted(graph[name][1]) if dep in group]
        rows.append(("、".join(_shown(name, {name}) for name in group), "；".join(late)))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# The tables in the documents
# ---------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Table:
    name: str                           # the marker's name
    document: Path
    header: Row                         # every column's title; the last column is the hand-written one
    rows: Callable[[], list[Row]]       # the generated cells of every row, in order


TABLES = (
    Table("settings", CONFIG_DOC, ("字段", "默认值", "除了在代码里，还能怎么设", "含义"), settings_rows),
    Table("params", CONFIG_DOC, ("字段", "默认值", "含义"), params_rows),
    Table("codemap-crowd3d", CODEMAP_DOC, ("文件", "层", "导入时就依赖的模块", "只在函数里导入的", "职责"),
          lambda: codemap_rows("crowd3d")),
    Table("codemap-scripts", CODEMAP_DOC, ("文件", "层", "导入时就依赖的模块", "只在函数里导入的", "职责"),
          lambda: codemap_rows("scripts")),
    Table("codemap-cycles", CODEMAP_DOC, ("互相导入的一组模块", "把它们连成环的函数内导入", "说明"), cycle_rows),
)

_CELL_BREAK = re.compile(r"(?<!\\)\|")


def markers(name: str) -> tuple[str, str]:
    return f"<!-- gen_docs:begin {name} -->", f"<!-- gen_docs:end {name} -->"


def split_row(line: str) -> list[str]:
    """The cells of one table line (a `\\|` inside a cell is not a cell break)."""
    line = line.strip()
    if not (line.startswith("|") and line.endswith("|") and len(line) >= 2):
        raise ValueError(f"not a table row: {line[:80]!r}")
    return [cell.strip() for cell in _CELL_BREAK.split(line[1:-1])]


def block_span(text: str, name: str) -> tuple[int, int]:
    """Where the lines between a table's two markers start and end in the document's text."""
    begin, end = markers(name)
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"expected exactly one {begin!r} and one {end!r}")
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    if stop < start:
        raise ValueError(f"{end!r} comes before {begin!r}")
    return start, stop


def written_cells(block: str) -> dict[str, str]:
    """key -> the hand-written last cell of each row of a table as the document has it."""
    lines = [line for line in block.splitlines() if line.strip()]
    cells: dict[str, str] = {}
    for line in lines[2:]:                      # after the header and the |---| line
        row = split_row(line)
        if row[0] in cells:
            raise ValueError(f"two rows for {row[0]}")
        cells[row[0]] = row[-1]
    return cells


def render_block(table: Table, block: str, rows: list[Row] | None = None) -> str:
    """The table as the code gives it, with the hand-written cells of `block` (empty for a key it does not have)."""
    written = written_cells(block)
    rows = table.rows() if rows is None else rows
    lines = ["| " + " | ".join(table.header) + " |", "|" + "---|" * len(table.header)]
    lines += ["| " + " | ".join([*row, written.get(row[0], "")]).rstrip() + " |" for row in rows]
    return "\n" + "\n".join(lines) + "\n"


def render(text: str, tables: tuple[Table, ...], rows: dict[str, list[Row]] | None = None) -> str:
    """A document's text with each of its tables rebuilt from the code (rows: other rows for a table, by name)."""
    for table in tables:
        start, stop = block_span(text, table.name)
        text = text[:start] + render_block(table, text[start:stop], (rows or {}).get(table.name)) + text[stop:]
    return text


def unwritten(text: str, tables: tuple[Table, ...]) -> list[str]:
    """The rows whose hand-written cell is empty: "table: key"."""
    missing = []
    for table in tables:
        start, stop = block_span(text, table.name)
        missing += [f"{table.name}: {key}" for key, cell in written_cells(text[start:stop]).items() if not cell]
    return missing


def documents() -> dict[Path, tuple[Table, ...]]:
    found: dict[Path, tuple[Table, ...]] = {}
    for table in TABLES:
        found[table.document] = (*found.get(table.document, ()), table)
    return found


def problems() -> list[str]:
    """What is wrong with the documents as they stand: one line per finding, with a diff where a table differs
    from what the code gives. Empty when everything is in step."""
    found = []
    for path, tables in documents().items():
        name = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        try:
            expected = render(text, tables)
        except ValueError as error:             # a marker is missing or doubled, or a row is not a table row
            found.append(f"{name}: {error}")
            continue
        if expected != text:
            diff = difflib.unified_diff(text.splitlines(), expected.splitlines(), f"{name} (as committed)",
                                        f"{name} (from the code)", lineterm="", n=0)
            found.append(f"{name} is not what the code gives; run `python scripts/gen_docs.py`:\n" + "\n".join(diff))
        found += [f"{name}: the hand-written cell is empty for {row}" for row in unwritten(expected, tables)]
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild the generated tables of docs/CONFIG.md and docs/ARCHITECTURE.md.")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 when a document differs")
    args = ap.parse_args(argv)
    if args.check:
        found = problems()
        print("\n".join(found) if found else "the generated tables are what the code gives")
        return 1 if found else 0
    for path, tables in documents().items():
        text = path.read_text(encoding="utf-8")
        expected = render(text, tables)
        if expected != text:
            path.write_text(expected, encoding="utf-8")
            print(f"rewrote {path.relative_to(REPO_ROOT)}")
        for row in unwritten(expected, tables):
            print(f"{path.relative_to(REPO_ROOT)}: write the last cell of {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
