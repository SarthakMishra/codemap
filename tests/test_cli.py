"""Tests for the CLI functionality."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from typer.testing import CliRunner

from codemap.analyzer.tree_parser import CodeParser
from codemap.cli import _get_output_path, app
from codemap.config import DEFAULT_CONFIG

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
        "output_dir": "docs",
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


def test_get_output_path() -> None:
    """Test output path generation."""
    repo_root = Path("/test/repo")
    config = {
        "output_dir": "docs",
    }

    # Test with custom output path
    custom_path = Path("custom/path.md")
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
        "output_dir": "docs",
    }

    with patch("codemap.cli.datetime") as mock_datetime:
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
    config = {
        "token_limit": 10000,
        "use_gitignore": True,
        "output_dir": custom_output_dir,
    }
    config_file.write_text(yaml.dump(config))

    # Create a subdirectory to analyze
    subdir = sample_repo / "src"
    subdir.mkdir(exist_ok=True, parents=True)
    (subdir / "test.py").write_text("# Test file")

    # Track paths where write_text is called
    written_paths = []

    def mock_write_text(self, content):
        written_paths.append(str(self))

    def mock_mkdir(self, parents=False, exist_ok=False):
        return None

    with patch("codemap.cli.Path.cwd", return_value=sample_repo):
        # Mock file operations to avoid actual filesystem access
        with patch.object(Path, "write_text", mock_write_text):
            with patch.object(Path, "mkdir", mock_mkdir):
                # Ensure exists returns True for any path
                with patch.object(Path, "exists", return_value=True):
                    with patch.object(Path, "is_file", return_value=False):
                        # Ensure is_absolute returns False for the expected paths
                        with patch.object(Path, "is_absolute", return_value=False):
                            # Run the command
                            result = runner.invoke(app, ["generate", str(subdir)])

    # We don't need to check the exit code as we've mocked all file operations
    # and just want to verify the paths used

    # The output path should include custom_docs_dir
    expected_path_fragment = f"{custom_output_dir}/documentation_"

    # Check if any written path contains the expected output directory
    path_contains_custom_dir = any(expected_path_fragment in path for path in written_paths)
    assert path_contains_custom_dir, f"No paths containing '{expected_path_fragment}' in {written_paths}"


def test_pygments_language_detection(sample_repo: Path) -> None:
    """Test that Pygments is used for language detection in CodeParser."""
    # Create files with different extensions
    (sample_repo / "script.py").write_text("def main(): pass")
    (sample_repo / "index.html").write_text("<html><body>Hello</body></html>")
    (sample_repo / "style.css").write_text("body { color: #333; }")

    # Use mock to track calls to Pygments
    with patch("codemap.analyzer.tree_parser.get_lexer_for_filename") as mock_get_lexer:
        with patch("codemap.analyzer.tree_parser.guess_lexer"):
            # Setup the mock to return lexers with different names
            py_lexer = Mock()
            py_lexer.name = "Python"
            html_lexer = Mock()
            html_lexer.name = "HTML"
            css_lexer = Mock()
            css_lexer.name = "CSS"

            # Return different lexers for different files
            def side_effect(filename):
                if filename.endswith(".py"):
                    return py_lexer
                if filename.endswith(".html"):
                    return html_lexer
                if filename.endswith(".css"):
                    return css_lexer
                raise ValueError(f"Unknown file: {filename}")

            mock_get_lexer.side_effect = side_effect

            # Create parser and parse files
            parser = CodeParser({})

            # Test Python file
            py_info = parser.parse_file(sample_repo / "script.py")
            assert py_info["language"] == "python"

            # Test HTML file
            html_info = parser.parse_file(sample_repo / "index.html")
            assert html_info["language"] == "html"

            # Test CSS file
            css_info = parser.parse_file(sample_repo / "style.css")
            assert css_info["language"] == "css"

            # Verify Pygments was called for each file
            assert mock_get_lexer.call_count == 6  # 3 calls in should_parse and 3 in parse_file
