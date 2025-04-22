"""Tests for the file filtering utilities."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from pygments.util import ClassNotFound

from codemap.utils.file_filters import FileFilter


class TestFileFilter:
    """Test cases for the FileFilter class."""

    def test_init_no_config(self) -> None:
        """Test initialization with no config."""
        file_filter = FileFilter()
        assert file_filter.config == {}
        assert file_filter.gitignore_patterns == []

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = {"use_gitignore": True}
        with patch.object(FileFilter, "_load_gitignore") as mock_load:
            file_filter = FileFilter(config)
            assert file_filter.config == config
            mock_load.assert_called_once()

    def test_load_gitignore(self) -> None:
        """Test loading gitignore patterns."""
        gitignore_content = """
# Comment
*.pyc
__pycache__/
.venv/
"""
        m = mock_open(read_data=gitignore_content)

        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.open", m):
            file_filter = FileFilter({"use_gitignore": True})
            assert len(file_filter.gitignore_patterns) == 3
            assert "*.pyc" in file_filter.gitignore_patterns
            assert "__pycache__/" in file_filter.gitignore_patterns
            assert ".venv/" in file_filter.gitignore_patterns

    def test_matches_pattern_directory_pattern(self) -> None:
        """Test matching directory patterns."""
        file_filter = FileFilter()
        assert file_filter.matches_pattern(Path("src/app/__pycache__/file.py"), "__pycache__/")
        assert file_filter.matches_pattern(Path("src/__pycache__"), "__pycache__/")

    def test_matches_pattern_dot_pattern(self) -> None:
        """Test matching dot patterns."""
        file_filter = FileFilter()
        assert file_filter.matches_pattern(Path("src/.venv/lib"), ".venv")
        assert file_filter.matches_pattern(Path(".git/config"), ".git")
        assert file_filter.matches_pattern(Path("src/.env"), ".env")

    def test_matches_pattern_with_path_separator(self) -> None:
        """Test matching patterns with directory separators."""
        file_filter = FileFilter()
        assert file_filter.matches_pattern(Path("src/temp/file.txt"), "**/temp/*")
        assert file_filter.matches_pattern(Path("src/module/test.py"), "src/module/*.py")

    def test_matches_pattern_simple_pattern(self) -> None:
        """Test matching simple patterns."""
        file_filter = FileFilter()
        assert file_filter.matches_pattern(Path("src/file.py"), "*.py")
        assert file_filter.matches_pattern(Path("src/module/test.pyc"), "*.pyc")

    def test_should_parse_default_excluded(self) -> None:
        """Test should_parse with default excluded directories."""
        file_filter = FileFilter()
        assert not file_filter.should_parse(Path("src/__pycache__/file.py"))
        assert not file_filter.should_parse(Path(".git/config"))
        assert not file_filter.should_parse(Path(".env"))
        assert not file_filter.should_parse(Path(".venv/lib/file.py"))
        assert not file_filter.should_parse(Path("venv/bin/python"))
        assert not file_filter.should_parse(Path("build/artifact.txt"))
        assert not file_filter.should_parse(Path("dist/package.tar.gz"))

    def test_should_parse_gitignore_patterns(self) -> None:
        """Test should_parse with gitignore patterns."""
        file_filter = FileFilter({"use_gitignore": True})
        file_filter.gitignore_patterns = ["*.log", "temp/", "secret.txt"]

        assert not file_filter.should_parse(Path("app.log"))
        assert not file_filter.should_parse(Path("logs/server.log"))
        assert not file_filter.should_parse(Path("src/temp/cache.json"))
        assert not file_filter.should_parse(Path("secret.txt"))

        # These should not be excluded by the patterns
        with patch("pygments.lexers.get_lexer_for_filename") as mock_get_lexer:
            mock_get_lexer.return_value = MagicMock()
            assert file_filter.should_parse(Path("src/app.py"))
            assert file_filter.should_parse(Path("src/templates/index.html"))

    def test_should_parse_no_lexer(self) -> None:
        """Test should_parse when no lexer is found."""
        file_filter = FileFilter()

        # Mock get_lexer_for_filename to raise ClassNotFound
        with patch("pygments.lexers.get_lexer_for_filename", side_effect=ClassNotFound):
            assert not file_filter.should_parse(Path("unknown.xyz"))

    def test_should_parse_valid_file(self) -> None:
        """Test should_parse with a valid file."""
        file_filter = FileFilter()

        # Mock get_lexer_for_filename to return a lexer
        with patch("pygments.lexers.get_lexer_for_filename") as mock_get_lexer:
            mock_get_lexer.return_value = MagicMock()
            assert file_filter.should_parse(Path("src/app.py"))
            assert file_filter.should_parse(Path("src/components/Button.tsx"))
