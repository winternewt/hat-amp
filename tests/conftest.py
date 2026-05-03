"""Pytest helpers for optional third-party reference comparisons."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import httpx
import pytest

REFERENCE_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "aaryashBharadwaj/Aperiodic-Monotile-Percolation/Version_3_Final"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register opt-in reference-test flag."""
    parser.addoption(
        "--run-reference",
        action="store_true",
        default=False,
        help="run tests that download third-party reference scripts",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip reference tests unless the flag or env var is enabled."""
    run_reference = bool(config.getoption("--run-reference")) or (
        os.environ.get("HAT_AMP_RUN_REFERENCE") == "1"
    )
    if run_reference:
        return

    skip_reference = pytest.mark.skip(
        reason="requires --run-reference or HAT_AMP_RUN_REFERENCE=1"
    )
    for item in items:
        if "reference" in item.keywords:
            item.add_marker(skip_reference)


@pytest.fixture(scope="session")
def reference_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the session temp directory used for reference downloads."""
    return tmp_path_factory.mktemp("third_party_reference")


@pytest.fixture(scope="session")
def reference_file(reference_data_dir: Path) -> Callable[[str], Path]:
    """Download one reference file from GitHub raw content."""

    def fetch(relative_path: str) -> Path:
        output = reference_data_dir / relative_path
        if output.exists():
            return output

        output.parent.mkdir(parents=True, exist_ok=True)
        url = f"{REFERENCE_BASE_URL}/{relative_path}"
        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            pytest.skip(f"download failed for {relative_path}: {exc}")
        output.write_text(response.text, encoding="utf-8")
        return output

    return fetch


@pytest.fixture(scope="session")
def reference_module(reference_file: Callable[[str], Path]) -> Callable[[str], ModuleType]:
    """Load a downloaded reference script as a Python module."""

    def load(relative_path: str) -> ModuleType:
        module_path = reference_file(relative_path)
        module_name = f"third_party_{module_path.stem}"
        module_dir = str(module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
