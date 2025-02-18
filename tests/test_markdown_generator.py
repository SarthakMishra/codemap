"""Tests for markdown documentation generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from codemap.generators.markdown_generator import MarkdownGenerator


@pytest.fixture
def mock_repo_root(tmp_path: Path) -> Path:
    """Create a temporary repository root for testing."""
    return tmp_path


@pytest.fixture
def basic_config() -> dict[str, list[str] | int]:
    """Provide a basic configuration for testing."""
    return {
        "token_limit": 1000,
        "include_patterns": ["*.py"],
        "exclude_patterns": [],
        "sections": ["overview", "dependencies", "details"],
    }


@pytest.fixture
def generator(mock_repo_root: Path, basic_config: dict[str, list[str] | int]) -> MarkdownGenerator:
    """Create a MarkdownGenerator instance for testing."""
    return MarkdownGenerator(mock_repo_root, basic_config)


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
    mock_files = {
        mock_repo_root / "main.py": {
            "imports": ["os", "sys"],
            "classes": ["MainClass"],
            "functions": ["main"],
            "docstring": "Main module docstring",
            "content": "# Sample content",
        },
        mock_repo_root / "utils.py": {
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
    mock_files = {
        mock_repo_root / "z.py": {"importance_score": 0.5},
        mock_repo_root / "a.py": {"importance_score": 0.8},
        mock_repo_root / "m.py": {"importance_score": 0.3},
    }

    doc = generator.generate_documentation(mock_files)

    # Check that files appear in order of importance score
    a_pos = doc.find("a.py")
    z_pos = doc.find("z.py")
    m_pos = doc.find("m.py")

    assert a_pos < z_pos < m_pos


def test_markdown_escaping(generator: MarkdownGenerator, mock_repo_root: Path) -> None:
    """Test proper escaping of markdown special characters."""
    mock_files = {
        mock_repo_root / "test.py": {
            "docstring": "Contains * and _ and # characters",
            "content": "# Code with *special* _characters_",
        },
    }

    doc = generator.generate_documentation(mock_files)

    # Verify special characters are properly escaped
    assert r"Contains \* and \_ and \# characters" in doc
