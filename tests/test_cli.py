"""Tests for the CLI functionality."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import IO, Generator, TypeVar
from unittest.mock import Mock, patch

import pytest
import yaml
from typer.testing import CliRunner

import codemap.cli_app
from codemap.cli.main import _get_output_path
from codemap.config import DEFAULT_CONFIG

app = codemap.cli_app.app

runner = CliRunner()
T = TypeVar("T")  # Generic type for return value of Path.open


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
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
def mock_code_parser() -> Generator[Mock, None, None]:
    """Create a mock CodeParser instance."""
    with patch("codemap.cli.CodeParser") as mock:
        parser_instance = Mock()
        parser_instance.should_parse.return_value = True
        parser_instance.parse_file.return_value = {
            "imports": [],
            "classes": [],
            "references": [],
            "content": "Test content",
            "language": "python",
        }
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
        "use_gitignore": False,
        "output_dir": str(sample_repo / "test_docs"),  # Use a path within sample_repo
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

    # Verify output dir was created in the correct location
    assert (sample_repo / "test_docs").exists()


def test_generate_command_with_invalid_path() -> None:
    """Test generate command with non-existent path."""
    result = runner.invoke(app, ["generate", "/nonexistent/path"])
    assert result.exit_code == 2  # Typer's behavior for non-existent path
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
    assert result.exit_code != 0  # Should fail


def test_get_output_path(temp_dir: Path) -> None:
    """Test output path generation."""
    repo_root = temp_dir
    config = {
        "output_dir": "docs",
    }

    # Test with custom output path
    custom_path = temp_dir / "custom/path.md"
    assert _get_output_path(repo_root, custom_path, config) == custom_path

    # Test with config-based path - mock mkdir to avoid permission issues
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        result = _get_output_path(repo_root, None, config)
        assert result.parent == repo_root / "docs"
        assert result.suffix == ".md"
        assert "documentation_" in result.name  # Timestamp format
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_get_output_path_with_custom_path(sample_repo: Path) -> None:
    """Test output path generation when a custom path is provided."""
    custom_path = sample_repo / "custom" / "docs.md"
    result = _get_output_path(sample_repo, custom_path, DEFAULT_CONFIG)
    assert result == custom_path


def test_get_output_path_creates_directory(sample_repo: Path) -> None:
    """Test that output path generation creates missing directories."""
    config = {
        "output_dir": "nested/docs/dir",
    }
    result = _get_output_path(sample_repo, None, config)
    assert result.parent.exists()
    assert result.parent == sample_repo / "nested" / "docs" / "dir"


def test_get_output_path_with_timestamp(sample_repo: Path) -> None:
    """Test output path generation with timestamp."""
    current_time = datetime.now(tz=timezone.utc)
    config = {
        "output_dir": str(sample_repo / "test_docs"),  # Use a path within sample_repo
    }

    with patch("codemap.cli.main.datetime") as mock_datetime:
        mock_datetime.now.return_value = current_time
        mock_datetime.timezone = timezone  # Make sure timezone is accessible
        result = _get_output_path(sample_repo, None, config)
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
        expected_name = f"documentation_{formatted_time}.md"
        assert result.name == expected_name


def test_generate_tree_command(sample_repo: Path) -> None:
    """Test the tree generation command."""
    # Create some files for the tree generation
    (sample_repo / "src" / "main.py").parent.mkdir(exist_ok=True, parents=True)
    (sample_repo / "src" / "main.py").write_text("# Main file")
    (sample_repo / "src" / "utils" / "helper.py").parent.mkdir(exist_ok=True, parents=True)
    (sample_repo / "src" / "utils" / "helper.py").write_text("# Helper file")

    # Run tree command
    result = runner.invoke(app, ["generate", "--tree", str(sample_repo)])

    assert result.exit_code == 0
    # Check that tree was generated and contains our directories
    assert "src" in result.stdout
    assert "main.py" in result.stdout
    assert "utils" in result.stdout
    assert "helper.py" in result.stdout


def test_generate_tree_command_with_output(sample_repo: Path) -> None:
    """Test the tree generation command with output to file."""
    # Create some files for the tree generation
    (sample_repo / "src" / "main.py").parent.mkdir(exist_ok=True, parents=True)
    (sample_repo / "src" / "main.py").write_text("# Main file")

    # Output file
    output_file = sample_repo / "tree.txt"

    # Run tree command with output
    result = runner.invoke(app, ["generate", "--tree", str(sample_repo), "-o", str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()
    tree_content = output_file.read_text()
    # Check tree content
    assert "src" in tree_content
    assert "main.py" in tree_content


def test_respect_output_dir_from_config(sample_repo: Path) -> None:
    """Test that generate command respects output_dir from config."""
    # Create a config file with custom output_dir
    config_file = sample_repo / ".codemap.yml"
    custom_output_dir = "custom_docs_dir"
    config_content = {
        "token_limit": 10000,
        "use_gitignore": True,
        "output_dir": custom_output_dir,
    }
    config_file.write_text(yaml.dump(config_content))

    # Create a subdirectory to analyze
    subdir = sample_repo / "src"
    subdir.mkdir(exist_ok=True, parents=True)
    (subdir / "test.py").write_text("# Test file")

    # Track paths where write_text is called
    written_paths = []

    def mock_write_text(self: Path, _content: str) -> None:
        """Mock write_text to track where it's called.

        Args:
            self: The path object
            _content: Content to write (unused in this mock)
        """
        written_paths.append(str(self))

    # Create a custom Path class with modified exists and is_file methods
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_open = Path.open

    def custom_exists(self: Path) -> bool:
        """Custom Path.exists implementation for testing."""
        if self.name == ".codemap.yml":
            return True
        return original_exists(self)

    def custom_is_file(self: Path) -> bool:
        """Custom Path.is_file implementation for testing."""
        if self.name == ".codemap.yml":
            return True
        return original_is_file(self)

    def custom_open(self: Path, mode: str = "r", *_args, **_kwargs) -> IO[str]:
        """Custom Path.open implementation for testing.

        Returns:
            A file-like object (StringIO for config files, regular file otherwise)
        """
        if ".codemap.yml" in str(self):
            return StringIO(yaml.dump(config_content))
        # Use the original Path.open to avoid infinite recursion
        return original_open(self, mode, *_args, **_kwargs)

    # Apply the patches
    with (
        patch.object(Path, "write_text", mock_write_text),
        patch.object(Path, "exists", custom_exists),
        patch.object(Path, "is_file", custom_is_file),
        patch.object(Path, "open", custom_open),
    ):
        # Run the command
        result = runner.invoke(app, ["generate", str(subdir)])
        assert result.exit_code == 0

    # The output path should include custom_docs_dir
    assert len(written_paths) > 0, "No paths were written to"
    assert any(custom_output_dir in path for path in written_paths), (
        f"Custom output dir '{custom_output_dir}' not found in paths: {written_paths}"
    )
