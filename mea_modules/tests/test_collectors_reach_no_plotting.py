"""What the collectors compute never reaches what the figures draw.

A pipeline fingerprints a capsule's diagnostics by the code they import: every
module `collect_segment_diagnostics`, `collect_concat_diagnostics`, the cache
and the readable JSON reach, directly or through each other. If a plotting
module is among them, editing a figure (a colour, a legend) changes the
fingerprint of numbers it never touched, and every cache is recomputed for
nothing. So the compute side imports only compute modules; the renderers
import the compute side, never the other way round.

This walks the imports statically (module-level and function-local alike, as
the fingerprint does) from the compute entry points, inside the package.
"""

import ast
from pathlib import Path

import mea_modules

PACKAGE_ROOT = Path(mea_modules.__file__).parent
ENTRY_POINTS = (
    "mea_modules.diagnostics.collect",
    "mea_modules.diagnostics.collect_concat",
    "mea_modules.diagnostics.cache",
    "mea_modules.diagnostics.readable",
    "mea_modules.diagnostics.buffer",
)
_DRAWING_PREFIXES = ("plot_", "draw_", "shade_", "_plot", "_draw", "_shade")
_FIGURE_MODULES = {"mea_modules.diagnostics.figure_text", "mea_modules.diagnostics.figure_style"}


def _source(module):
    relative = Path(*module.split(".")[1:])
    for candidate in (PACKAGE_ROOT / relative.with_suffix(".py"), PACKAGE_ROOT / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _resolve(module, node):
    """The absolute module an ImportFrom names (the package-relative part)."""
    if node.level:
        base = module.split(".")
        is_package = _source(module) is not None and _source(module).name == "__init__.py"
        base = base if is_package else base[:-1]
        base = base[: len(base) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))
    return node.module or ""


def _defining(target, name, depth=0):
    """The module that defines `name` imported from `target`: a submodule of
    that name, the module itself, or -- for a package re-exporting it -- the
    submodule its ``__init__`` imports it from (as the fingerprint resolves it)."""
    sub = f"{target}.{name}"
    if _source(sub):
        return sub
    source = _source(target)
    if source is None or source.name != "__init__.py" or depth > 5:
        return target
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ImportFrom) and any(
                (alias.asname or alias.name) == name for alias in node.names):
            origin = next(alias.name for alias in node.names if (alias.asname or alias.name) == name)
            return _defining(_resolve(target, node), origin, depth + 1)
    return target


def _imports(module):
    """Every in-package module `module` imports, each name resolved to the
    submodule that defines it when it comes from a package."""
    tree = ast.parse(_source(module).read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = _resolve(module, node)
            if not target.startswith("mea_modules"):
                continue
            for alias in node.names:
                found.add(_defining(target, alias.name))
        elif isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names if alias.name.startswith("mea_modules")}
    return {name for name in found if _source(name)}


def _draws(module):
    """True for a module that draws: it imports matplotlib, defines a drawing
    function, or is one of the figure text/style modules."""
    if module in _FIGURE_MODULES:
        return True
    tree = ast.parse(_source(module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            if any(name.split(".")[0] == "matplotlib" for name in names):
                return True
    return any(isinstance(node, ast.FunctionDef) and node.name.startswith(_DRAWING_PREFIXES)
               for node in tree.body)


def _closure():
    seen, stack, via = set(), list(ENTRY_POINTS), {}
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        for child in _imports(module):
            if child not in seen and child != module:
                via.setdefault(child, module)
                stack.append(child)
    return seen, via


def test_the_compute_side_reaches_no_plotting_module():
    reached, via = _closure()
    drawing = sorted(name for name in reached if _draws(name))
    assert not drawing, "the compute side reaches plotting modules: " + ", ".join(
        f"{name} (via {via.get(name)})" for name in drawing)


def test_the_walk_sees_function_local_imports():
    """The fingerprint counts an import inside a function; so does this walk."""
    reached, _ = _closure()
    assert "mea_modules.quality.detection" in reached
