"""Tests for the CLI functionality."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import yaml
from typer.testing import CliRunner

from codemap.cli import _format_output_path, app
from codemap.config import DEFAULT_CONFIG

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for testing."""
    yield tmp_path
    # Cleanup
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture
def mock_code_parser() -> Mock:
    """Create a mock CodeParser instance."""
    with patch("codemap.cli.CodeParser") as mock:
        parser_instance = Mock()
        parser_instance.should_parse.return_value = True
        parser_instance.parse_file.return_value = {
            "imports": [],
            "classes": [],
            "functions": [],
            "docstring": "Test docstring",
        }
        parser_instance.parsers = {"python": Mock(), "javascript": Mock()}
        mock.return_value = parser_instance
        yield mock


@patch("codemap.cli.CodeParser")
def test_init_command(mock_parser: Mock, temp_dir: Path) -> None:
    """Test the init command creates necessary files."""
    # Setup mock
    parser_instance = Mock()
    parser_instance.should_parse.return_value = True
    parser_instance.parsers = {"python": Mock(), "javascript": Mock()}
    mock_parser.return_value = parser_instance

    result = runner.invoke(app, ["init", str(temp_dir)])
    assert result.exit_code == 0

    # Check if files were created
    config_file = temp_dir / ".codemap.yml"
    cache_dir = temp_dir / ".codemap_cache"
    assert config_file.exists()
    assert cache_dir.exists()
    assert (cache_dir / ".gitignore").exists()
    assert (cache_dir / "info.json").exists()


@patch("codemap.cli.CodeParser")
def test_init_command_with_existing_files(mock_parser: Mock, temp_dir: Path) -> None:
    """Test init command handles existing files correctly."""
    # Setup mock
    parser_instance = Mock()
    parser_instance.should_parse.return_value = True
    parser_instance.parsers = {"python": Mock(), "javascript": Mock()}
    mock_parser.return_value = parser_instance

    # Create initial files
    runner.invoke(app, ["init", str(temp_dir)])

    # Try to init again without force
    result = runner.invoke(app, ["init", str(temp_dir)])
    assert result.exit_code == 1
    assert "CodeMap files already exist" in result.stdout

    # Try with force flag
    result = runner.invoke(app, ["init", "-f", str(temp_dir)])
    assert result.exit_code == 0


@patch("codemap.cli.CodeParser")
@patch("codemap.cli.DependencyGraph")
@patch("codemap.cli.MarkdownGenerator")
def test_generate_command(
    mock_generator: Mock,
    mock_graph: Mock,
    mock_parser: Mock,
    temp_dir: Path,
) -> None:
    """Test the generate command with mocked components."""
    # Setup mocks
    parser_instance = Mock()
    parser_instance.should_parse.return_value = True
    parser_instance.parse_file.return_value = {
        "imports": [],
        "classes": [],
        "functions": [],
        "docstring": "Test docstring",
    }
    mock_parser.return_value = parser_instance

    mock_graph_instance = mock_graph.return_value
    mock_graph_instance.get_important_files.return_value = []

    mock_generator_instance = mock_generator.return_value
    mock_generator_instance.generate_documentation.return_value = "# Test Documentation"

    # Run generate command
    output_file = temp_dir / "docs.md"
    result = runner.invoke(app, ["generate", str(temp_dir), "-o", str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Documentation generated successfully" in result.stdout


@pytest.mark.usefixtures("temp_dir")
@patch("codemap.cli.CodeParser")
@patch("codemap.cli.DependencyGraph")
def test_generate_command_with_config(
    mock_dependency_graph_class: Mock,
    mock_codeparser_class: Mock,
    temp_dir: Path,
) -> None:
    """Test generate command with custom config file."""
    # Create a real CodeParser and simply override its should_parse method
    from codemap.analyzer.tree_parser import CodeParser

    real_parser = CodeParser()

    # Only parse Python files
    real_parser.should_parse = lambda p: p.suffix.lower() == ".py"

    # Patch so that anywhere the CLI does CodeParser(), it returns our real_parser
    mock_codeparser_class.return_value = real_parser

    # Mock the dependency graph
    mock_graph = Mock()
    mock_graph.get_important_files.return_value = []  # No important files for simplicity
    mock_dependency_graph_class.return_value = mock_graph

    # Create a test config file
    config_file = temp_dir / "test_config.yml"
    config_file.write_text("token_limit: 1000\n")

    # Create a dummy Python file to parse
    (temp_dir / "test.py").write_text("print('hello')")

    # Mock scipy dependency
    scipy_mock = Mock()
    scipy_mock.sparse = Mock()
    scipy_mock.sparse.csr_matrix = Mock()
    csr_mock = Mock()
    csr_mock.__getitem__ = lambda _, __: Mock()
    scipy_mock.sparse.csr_matrix.return_value = csr_mock
    with patch.dict("sys.modules", {"scipy": scipy_mock}):
        result = runner.invoke(
            app,
            ["generate", str(temp_dir), "--config", str(config_file), "--map-tokens", "2000"],
        )
        assert result.exit_code == 0


@patch("codemap.cli.CodeParser")
def test_generate_command_with_invalid_path(mock_parser: Mock) -> None:
    """Test generate command with non-existent path."""
    # Setup mock
    parser_instance = Mock()
    parser_instance.should_parse.return_value = True
    mock_parser.return_value = parser_instance

    result = runner.invoke(app, ["generate", "/nonexistent/path"])
    assert result.exit_code == 2  # Changed from 1 to 2 to match typer's behavior
    assert "does not exist" in result.stdout


def test_format_output_path_with_custom_path(temp_dir: Path) -> None:
    """Test output path formatting when a custom path is provided."""
    custom_path = temp_dir / "custom" / "docs.md"
    result = _format_output_path(temp_dir, custom_path, DEFAULT_CONFIG)
    assert result == custom_path


def test_format_output_path_creates_directory(temp_dir: Path) -> None:
    """Test that output path formatting creates missing directories."""
    config = {
        "output": {
            "directory": "nested/docs/dir",
            "filename_format": "doc.md",
        },
    }
    result = _format_output_path(temp_dir, None, config)
    assert result.parent.exists()
    assert result.parent == temp_dir / "nested" / "docs" / "dir"


def test_format_output_path_with_timestamp(temp_dir: Path) -> None:
    """Test output path formatting with timestamp."""
    current_time = datetime.now(tz=timezone.utc)
    config = {
        "output": {
            "directory": "docs",
            "filename_format": "{base}.{timestamp}.md",
            "timestamp_format": "%Y%m%d",
        },
    }

    with patch("codemap.cli.datetime") as mock_datetime:
        mock_datetime.now.return_value = current_time
        result = _format_output_path(temp_dir, None, config)
        expected_name = f"documentation.{current_time.strftime('%Y%m%d')}.md"
        assert result.name == expected_name


def test_format_output_path_with_root_directory(temp_dir: Path) -> None:
    """Test output path formatting when in root directory."""
    config = {
        "output": {
            "directory": "docs",
            "filename_format": "{base}.{directory}.{timestamp}.md",
        },
    }
    result = _format_output_path(temp_dir, None, config)
    assert temp_dir.name in result.name


def test_generate_command_creates_output_directory(temp_dir: Path) -> None:
    """Test generate command creates output directory if missing."""
    # Setup
    from codemap.analyzer.tree_parser import CodeParser

    real_parser = CodeParser()
    real_parser.should_parse = lambda p: p.suffix.lower() == ".py"

    with patch("codemap.cli.CodeParser") as mock_codeparser_class, patch(
        "codemap.cli.DependencyGraph",
    ) as mock_graph_class:
        # Setup mocks
        mock_codeparser_class.return_value = real_parser
        mock_graph = Mock()
        mock_graph.get_important_files.return_value = []
        mock_graph_class.return_value = mock_graph

        # Create test files
        config = {
            "token_limit": 1000,
            "output": {
                "directory": "deeply/nested/docs",
                "filename_format": "doc.md",
            },
        }
        config_file = temp_dir / "test_config.yml"
        config_file.write_text(yaml.dump(config))
        (temp_dir / "test.py").write_text("print('hello')")

        # Run command
        result = runner.invoke(
            app,
            ["generate", str(temp_dir), "--config", str(config_file)],
        )

        # Verify
        assert result.exit_code == 0
        output_dir = temp_dir / "deeply" / "nested" / "docs"
        assert output_dir.exists()
        assert list(output_dir.glob("*.md"))


def test_generate_command_with_missing_parent_directory(temp_dir: Path) -> None:
    """Test generate command with output path in non-existent directory."""
    missing_dir = temp_dir / "nonexistent"
    output_path = missing_dir / "doc.md"

    result = runner.invoke(
        app,
        ["generate", str(temp_dir), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert missing_dir.exists()
    assert output_path.exists()
