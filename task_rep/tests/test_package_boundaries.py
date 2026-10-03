"""Runner packages are parallel: neither imports the other, and the shared
core imports neither. Deleting one runner package must not break the other."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNERS = ("pipeline", "agentic")


def _imported_packages(package: str) -> set:
    found = set()
    for path in (ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("package", ["contextual", *RUNNERS])
def test_no_cross_runner_imports(package):
    if not (ROOT / package).exists():
        pytest.skip(f"{package}/ not present")
    forbidden = set(RUNNERS) - {package}
    assert not _imported_packages(package) & forbidden


@pytest.mark.parametrize("package", ["contextual", *RUNNERS])
def test_tests_do_not_import_other_runners(package):
    tests = ROOT / "tests" / package
    if not tests.exists():
        pytest.skip(f"tests/{package}/ not present")
    imported = set()
    for path in tests.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    assert not imported & (set(RUNNERS) - {package})
