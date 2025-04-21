"""Global test fixtures and configuration."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cleanup_docs_dir() -> None:
    """Auto-use fixture to clean up any docs directories created by tests.

    This ensures we don't leave behind test files in the actual project directory.
    """
    # Setup - nothing to do
    yield

    # Cleanup
    project_root = Path.cwd()
    docs_paths = [project_root / "docs", project_root / "documentation", project_root / "custom_docs_dir"]

    for path in docs_paths:
        if path.exists() and path.is_dir():
            # Don't delete the directory if it's part of the original repo structure
            # Instead, just delete any files that were created during tests
            for item in path.iterdir():
                # Skip .gitkeep and other special files
                if item.name.startswith("."):
                    continue

                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except (PermissionError, OSError):
                    pass
