"""Tests for markdown documentation generation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from codemap.generators.markdown_generator import MarkdownGenerator


@pytest.fixture
def mock_repo_root(tmp_path: Path) -> Path:
    """Create a temporary repository root for testing."""
    # Create a .git directory to mark this as the repository root
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def basic_config() -> dict[str, list[str] | int]:
    """Provide a basic configuration for testing."""
    return {
        "token_limit": 1000,
        "include_patterns": ["*.py"],
        "exclude_patterns": ["__pycache__", "*.pyc"],
        "sections": ["overview", "dependencies", "details"],
    }


@pytest.fixture
def generator(mock_repo_root: Path, basic_config: dict[str, list[str] | int]) -> MarkdownGenerator:
    """Create a MarkdownGenerator instance for testing."""
    # Patch ConfigLoader to ensure it doesn't read .codemap.yml
    with patch("codemap.utils.config_loader.ConfigLoader") as mock_loader:
        mock_loader.return_value.config = basic_config
        yield MarkdownGenerator(mock_repo_root, basic_config)


def test_generator_initialization(generator: MarkdownGenerator) -> None:
    """Test MarkdownGenerator initialization."""
    assert generator.repo_root is not None
    assert generator.config is not None


def test_generate_documentation_empty(generator: MarkdownGenerator) -> None:
    """Test documentation generation with empty file set."""
    doc = generator.generate_documentation({})
    assert isinstance(doc, str)
    assert "# Code Documentation" in doc


def test_generate_documentation_with_files(generator: MarkdownGenerator, mock_repo_root: Path) -> None:
    """Test documentation generation with mock files."""
    # Create some test files
    (mock_repo_root / "src").mkdir()
    (mock_repo_root / "src" / "main.py").write_text("# Sample content")
    (mock_repo_root / "src" / "utils.py").write_text("# Utility functions")
    (mock_repo_root / "__pycache__").mkdir()
    (mock_repo_root / "__pycache__" / "main.cpython-39.pyc").write_text("ignored")

    mock_files = {
        mock_repo_root / "src" / "main.py": {
            "imports": ["os", "sys"],
            "classes": ["MainClass"],
            "functions": ["main"],
            "docstring": "Main module docstring",
            "content": "# Sample content",
        },
        mock_repo_root / "src" / "utils.py": {
            "imports": ["typing"],
            "classes": ["UtilClass"],
            "functions": ["helper"],
            "docstring": "Utilities module",
            "content": "# Utility functions",
        },
    }

    doc = generator.generate_documentation(mock_files)

    # Check for expected sections
    assert "# Code Documentation" in doc
    assert "## Overview" in doc
    assert "## Dependencies" in doc
    assert "## Details" in doc

    # Check for file content
    assert "main.py" in doc
    assert "utils.py" in doc
    assert "MainClass" in doc
    assert "UtilClass" in doc

    # Check that excluded files are not in the tree
    assert "__pycache__" not in doc
    assert "main.cpython-39.pyc" not in doc


def test_generate_documentation_with_custom_sections(mock_repo_root: Path) -> None:
    """Test documentation generation with custom sections configuration."""
    custom_config = {
        "token_limit": 1000,
        "include_patterns": ["*.py"],
        "exclude_patterns": [],
        "sections": ["custom_section"],
    }

    generator = MarkdownGenerator(mock_repo_root, custom_config)
    doc = generator.generate_documentation({})

    assert "## Custom Section" in doc
    assert "## Overview" not in doc


def test_file_sorting(generator: MarkdownGenerator, mock_repo_root: Path) -> None:
    """Test that files are properly sorted in the documentation."""
    # Create test files
    (mock_repo_root / "z.py").write_text("")
    (mock_repo_root / "a.py").write_text("")
    (mock_repo_root / "m.py").write_text("")

    mock_files = {
        mock_repo_root / "z.py": {"importance_score": 0.5},
        mock_repo_root / "a.py": {"importance_score": 0.8},
        mock_repo_root / "m.py": {"importance_score": 0.3},
    }

    doc = generator.generate_documentation(mock_files)

    # Find the Details section
    details_section = doc[doc.find("## Details") :]

    # Check that files appear in order of importance score in the Details section
    a_pos = details_section.find("a.py")
    z_pos = details_section.find("z.py")
    m_pos = details_section.find("m.py")

    assert a_pos >= 0 and z_pos >= 0 and m_pos >= 0, "All files should be present in Details section"
    assert a_pos < z_pos < m_pos, "Files should be ordered by importance score (highest to lowest)"


def test_markdown_escaping(generator: MarkdownGenerator, mock_repo_root: Path) -> None:
    """Test proper escaping of markdown special characters.

    This test verifies that:
    1. Inline formatting characters (* _ `) are escaped in docstrings
    2. Code content inside code blocks is not escaped
    3. Headings and other structural elements are not escaped
    """
    # Create test file
    test_file = mock_repo_root / "test.py"
    test_file.write_text("# Code with *special* _characters_")

    mock_files = {
        test_file: {
            "docstring": "Contains * and _ and ` characters",
            "content": "# Code with *special* _characters_",
            "classes": ["Test_Class"],  # Underscore should not be escaped in headings
            "functions": ["test_func"],  # Underscore should not be escaped in headings
        },
    }

    doc = generator.generate_documentation(mock_files)

    # Verify that inline formatting characters are escaped in docstrings
    assert "Contains \\* and \\_ and \\` characters" in doc

    # Verify that code content is not escaped (should be inside code block)
    assert "# Code with *special* _characters_" in doc

    # Verify that headings are not escaped
    assert "#### Test_Class" in doc
    assert "#### test_func" in doc
