"""Tests for the CLI functionality."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
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
def sample_repo(tmp_path: Path) -> Path:
    """Create a copy of the sample repository for testing."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_repo"
    repo_path = tmp_path / "sample_repo"
    shutil.copytree(fixtures_path, repo_path)
    return repo_path


@pytest.fixture
def mock_code_parser() -> Mock:
    """Create a mock CodeParser instance."""
    with patch("codemap.cli.CodeParser") as mock:
        parser_instance = Mock()
        parser_instance.should_parse.return_value = True
        parser_instance.parse_file.return_value = {
            "imports": [],
            "classes": [],
            "references": [],
            "bases": {},
            "attributes": {},
            "docstring": "Test docstring",
        }
        parser_instance.parsers = {"python": Mock()}
        mock.return_value = parser_instance
        yield mock


def test_init_command(temp_dir: Path) -> None:
    """Test the init command creates necessary files."""
    result = runner.invoke(app, ["init", str(temp_dir)])
    assert result.exit_code == 0
    assert (temp_dir / ".codemap.yml").exists()
    assert (temp_dir / "documentation").exists()


def test_init_command_with_existing_files(temp_dir: Path) -> None:
    """Test init command handles existing files correctly."""
    # Create initial files
    runner.invoke(app, ["init", str(temp_dir)])

    # Try to init again without force
    result = runner.invoke(app, ["init", str(temp_dir)])
    assert result.exit_code == 1
    assert "CodeMap files already exist" in result.stdout

    # Try with force flag
    result = runner.invoke(app, ["init", "-f", str(temp_dir)])
    assert result.exit_code == 0
    assert "CodeMap initialized successfully" in result.stdout


def test_generate_command(sample_repo: Path) -> None:
    """Test the generate command with real files."""
    # Initialize CodeMap in the sample repo
    runner.invoke(app, ["init", str(sample_repo)])

    # Run generate command
    output_file = sample_repo / "docs.md"
    result = runner.invoke(app, ["generate", str(sample_repo), "-o", str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text()


def test_generate_command_with_config(sample_repo: Path) -> None:
    """Test generate command with custom config file."""
    # Create a test config file
    config_file = sample_repo / "test_config.yml"
    config = {
        "token_limit": 1000,
        "exclude_patterns": ["__pycache__", "*.pyc"],
        "output": {
            "directory": "docs",
            "filename_format": "documentation.md",
        },
    }
    config_file.write_text(yaml.dump(config))

    # Initialize CodeMap in the sample repo
    runner.invoke(app, ["init", str(sample_repo)])

    # Run generate command with config
    result = runner.invoke(
        app,
        ["generate", str(sample_repo), "--config", str(config_file), "--map-tokens", "2000"],
    )
    assert result.exit_code == 0


def test_generate_command_with_invalid_path() -> None:
    """Test generate command with non-existent path."""
    result = runner.invoke(app, ["generate", "/nonexistent/path"])
    assert result.exit_code == 2  # Changed from 1 to 2 to match typer's behavior
    assert "does not exist" in result.stdout


def test_generate_command_creates_output_directory(sample_repo: Path) -> None:
    """Test generate command creates output directory if missing."""
    # Initialize CodeMap in the sample repo
    runner.invoke(app, ["init", str(sample_repo)])

    # Create a nested output path
    output_dir = sample_repo / "nested" / "docs"
    output_file = output_dir / "documentation.md"

    # Run generate command
    result = runner.invoke(app, ["generate", str(sample_repo), "-o", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()


def test_generate_command_with_missing_parent_directory(sample_repo: Path) -> None:
    """Test generate command fails gracefully with invalid output directory."""
    # Initialize CodeMap in the sample repo
    runner.invoke(app, ["init", str(sample_repo)])

    # Try to generate to a path with non-existent parent
    output_file = Path("/nonexistent/path/docs.md")
    result = runner.invoke(app, ["generate", str(sample_repo), "-o", str(output_file)])
    assert result.exit_code == 2
    assert "File system error" in result.stdout


def test_format_output_path() -> None:
    """Test output path formatting."""
    from codemap.cli import _format_output_path

    repo_root = Path("/test/repo")
    config = {
        "output": {
            "directory": "docs",
            "filename_format": "{base}.{directory}.{timestamp}.md",
            "timestamp_format": "%Y%m%d",
        },
    }

    # Test with custom output path
    custom_path = Path("custom/path.md")
    assert _format_output_path(repo_root, custom_path, config) == custom_path

    # Test with config-based path
    result = _format_output_path(repo_root, None, config)
    assert result.parent == repo_root / "docs"
    assert result.suffix == ".md"
    assert "documentation" in result.name
    assert "repo" in result.name


def test_format_output_path_with_custom_path(sample_repo: Path) -> None:
    """Test output path formatting when a custom path is provided."""
    custom_path = sample_repo / "custom" / "docs.md"
    result = _format_output_path(sample_repo, custom_path, DEFAULT_CONFIG)
    assert result == custom_path


def test_format_output_path_creates_directory(sample_repo: Path) -> None:
    """Test that output path formatting creates missing directories."""
    config = {
        "output": {
            "directory": "nested/docs/dir",
            "filename_format": "doc.md",
        },
    }
    result = _format_output_path(sample_repo, None, config)
    assert result.parent.exists()
    assert result.parent == sample_repo / "nested" / "docs" / "dir"


def test_format_output_path_with_timestamp(sample_repo: Path) -> None:
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
        result = _format_output_path(sample_repo, None, config)
        expected_name = f"documentation.{current_time.strftime('%Y%m%d')}.md"
        assert result.name == expected_name


def test_format_output_path_with_root_directory(sample_repo: Path) -> None:
    """Test output path formatting when in root directory."""
    config = {
        "output": {
            "directory": "docs",
            "filename_format": "{base}.{directory}.{timestamp}.md",
        },
    }
    result = _format_output_path(sample_repo, None, config)
    assert sample_repo.name in result.name
