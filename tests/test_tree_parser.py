"""Tests for the tree_parser module."""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound

from codemap.analyzer.tree_parser import CodeParser
from codemap.utils.file_filters import FileFilter


@pytest.fixture
def parser() -> CodeParser:
	"""Create a CodeParser instance for testing."""
	return CodeParser({"use_gitignore": False})


@pytest.fixture
def file_filter() -> FileFilter:
	"""Create a FileFilter instance for testing."""
	return FileFilter({"use_gitignore": False})


@pytest.fixture
def file_filter_with_gitignore() -> FileFilter:
	"""Create a FileFilter instance with gitignore enabled."""
	return FileFilter({"use_gitignore": True})


@pytest.fixture
def temp_gitignore(tmp_path: Path) -> Path:
	"""Create a temporary .gitignore file."""
	gitignore_path = tmp_path / ".gitignore"
	gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo
"""
	gitignore_path.write_text(gitignore_content)
	return gitignore_path


@pytest.mark.unit
@pytest.mark.analyzer
class TestCodeParserInitialization:
	"""Tests for CodeParser initialization."""

	def test_init_default_config(self) -> None:
		"""Test initialization with default config."""
		# Act
		parser = CodeParser()

		# Assert
		assert parser.config == {}
		assert hasattr(parser, "file_filter")
		assert isinstance(parser.file_filter, FileFilter)

	def test_init_with_config(self) -> None:
		"""Test initialization with custom config."""
		# Arrange
		config = {"use_gitignore": True, "token_limit": 4000}

		# Act
		parser = CodeParser(config)

		# Assert
		assert parser.config == config
		assert hasattr(parser, "file_filter")
		assert isinstance(parser.file_filter, FileFilter)


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.filesystem
class TestFileFilter:
	"""Tests for the FileFilter class."""

	def test_load_gitignore(self, tmp_path: Path) -> None:
		"""Test loading patterns from .gitignore file."""
		# Arrange
		# Create a temporary .gitignore file in the tmp_path
		gitignore_path = tmp_path / ".gitignore"
		gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo
"""
		gitignore_path.write_text(gitignore_content)

		# Change to the directory with the .gitignore file
		old_cwd = Path.cwd()
		os.chdir(tmp_path)

		try:
			# Act
			filter_obj = FileFilter({"use_gitignore": True})

			# Assert
			assert len(filter_obj.gitignore_patterns) > 0
			assert "__pycache__/" in filter_obj.gitignore_patterns
			assert "*.py[cod]" in filter_obj.gitignore_patterns
			assert ".venv" in filter_obj.gitignore_patterns
		finally:
			os.chdir(old_cwd)

	@pytest.mark.parametrize(
		("file_path", "pattern", "expected"),
		[
			(Path("test.py"), "*.py", True),
			(Path("test.txt"), "*.py", False),
			(Path("venv/lib/test.py"), "venv", True),
			(Path("project/venv/test.py"), "venv", True),
		],
	)
	def test_matches_pattern_simple(
		self, file_filter: FileFilter, file_path: Path, pattern: str, expected: bool
	) -> None:
		"""Test matching simple patterns."""
		# Act & Assert
		assert file_filter.matches_pattern(file_path, pattern) == expected

	@pytest.mark.parametrize(
		("file_path", "pattern", "expected"),
		[
			(Path(".venv/lib/test.py"), ".venv", True),
			(Path("project/.venv/test.py"), ".venv", True),
			(Path(".gitignore"), ".gitignore", True),
			(Path("node_modules/package/file.js"), "node_modules/", True),
			(Path("src/node_modules/file.js"), "node_modules/", True),
			(Path("venv/lib/test.py"), ".venv", True),
			(Path("src/config/.env"), ".env", True),
			(Path("src/.env/config"), ".env", True),
			(Path("src/config.txt"), "*.txt", True),
		],
	)
	def test_matches_pattern_dot_files(
		self, file_filter: FileFilter, file_path: Path, pattern: str, expected: bool
	) -> None:
		"""Test matching dot files and directories."""
		# Act & Assert
		assert file_filter.matches_pattern(file_path, pattern) == expected

	@pytest.mark.parametrize(
		("file_path", "pattern", "expected"),
		[
			(Path("src/temp/file.py"), "src/temp/*", True),
			(Path("project/src/temp/file.py"), "**/temp/*", True),
			(Path("src/other/file.py"), "src/temp/*", False),
		],
	)
	def test_matches_pattern_with_directory_separators(
		self, file_filter: FileFilter, file_path: Path, pattern: str, expected: bool
	) -> None:
		"""Test matching patterns with directory separators."""
		# Act & Assert
		assert file_filter.matches_pattern(file_path, pattern) == expected

	@pytest.mark.parametrize(
		"file_path",
		[
			Path("__pycache__/module.py"),
			Path("src/__pycache__/module.py"),
			Path(".git/config"),
			Path(".venv/bin/python"),
			Path("venv/bin/python"),
			Path("build/lib/module.py"),
			Path("dist/package.tar.gz"),
		],
	)
	def test_should_parse_excluded_directories(self, file_filter: FileFilter, file_path: Path) -> None:
		"""Test should_parse with default excluded directories."""
		# Act & Assert
		assert not file_filter.should_parse(file_path)

	def test_should_parse_with_gitignore(self, file_filter_with_gitignore: FileFilter) -> None:
		"""Test should_parse with gitignore patterns."""
		# Arrange
		# Mock the _matches_pattern method to simulate gitignore patterns
		with patch.object(file_filter_with_gitignore, "_matches_pattern") as mock_matches:
			# Set up the mock to return True for a specific pattern
			mock_matches.return_value = True

			# Act & Assert
			# The file should not be parsed if it matches a gitignore pattern
			assert not file_filter_with_gitignore.should_parse(Path("ignored_file.py"))

			# Verify the mock was called with the correct arguments
			mock_matches.assert_called()

	def test_should_parse_lexer_not_found(self, file_filter: FileFilter) -> None:
		"""Test should_parse when lexer is not found."""
		# Arrange
		# Mock get_lexer_for_filename to raise ClassNotFound
		with patch("codemap.utils.file_filters.get_lexer_for_filename") as mock_get_lexer:
			mock_get_lexer.side_effect = ClassNotFound("No lexer found")

			# Act & Assert
			# The file should not be parsed if no lexer is found
			assert not file_filter.should_parse(Path("unknown_extension.xyz"))

	def test_should_parse_valid_file(self, file_filter: FileFilter) -> None:
		"""Test should_parse with a valid file."""
		# Arrange
		# Mock get_lexer_for_filename to return a valid lexer
		with patch("codemap.utils.file_filters.get_lexer_for_filename") as mock_get_lexer:
			mock_get_lexer.return_value = get_lexer_for_filename("test.py")

			# Act & Assert
			# The file should be parsed if it has a valid extension and is not excluded
			assert file_filter.should_parse(Path("src/module.py"))


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.filesystem
class TestCodeParserFileParsing:
	"""Tests for CodeParser file parsing functionality."""

	def test_parse_file_python(self, parser: CodeParser, tmp_path: Path) -> None:
		"""Test parsing a Python file."""
		# Arrange
		python_content = """
import os
from pathlib import Path
import sys

class TestClass:
    def __init__(self):
        self.value = 42

    def test_method(self):
        return self.value

class AnotherClass:
    pass
"""
		# Create a test Python file
		test_file = tmp_path / "test.py"
		test_file.write_text(python_content)

		# Use the file_filter to determine if the file should be parsed
		with patch.object(parser.file_filter, "should_parse", return_value=True):
			# Act
			file_info = parser.parse_file(test_file)

			# Assert
			assert file_info["language"] == "python"
			assert file_info["content"] == python_content
			assert set(file_info["imports"]) == {"os", "pathlib", "sys"}
			assert set(file_info["classes"]) == {"TestClass", "AnotherClass"}

	def test_parse_file_non_python(self, parser: CodeParser, tmp_path: Path) -> None:
		"""Test parsing a non-Python file."""
		# Arrange
		js_content = """
function testFunction() {
    return 42;
}

class TestClass {
    constructor() {
        this.value = 42;
    }
}
"""
		# Create a test JS file
		test_file = tmp_path / "test.js"
		test_file.write_text(js_content)

		# Use the file_filter to determine if the file should be parsed
		with patch.object(parser.file_filter, "should_parse", return_value=True):
			# Act
			file_info = parser.parse_file(test_file)

			# Assert
			assert file_info["language"] == "javascript"
			assert file_info["content"] == js_content
			assert file_info["imports"] == []
			assert file_info["classes"] == []

	def test_parse_file_unknown_extension(self, parser: CodeParser, tmp_path: Path) -> None:
		"""Test parsing a file with unknown extension but guessable content."""
		# Arrange
		html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Hello World</h1>
</body>
</html>
"""
		# Create a file with unknown extension but HTML content
		test_file = tmp_path / "test.unknown"
		test_file.write_text(html_content)

		# Mock should_parse to return True to force parsing and mock get_lexer_for_filename
		with (
			patch.object(parser.file_filter, "should_parse", return_value=True),
			patch("codemap.analyzer.tree_parser.get_lexer_for_filename") as mock_get_lexer,
		):
			mock_get_lexer.side_effect = ClassNotFound("No lexer found")

			# Act
			file_info = parser.parse_file(test_file)

			# Assert
			assert file_info["language"] == "html"
			assert file_info["content"] == html_content

	def test_parse_file_unguessable_content(self, parser: CodeParser, tmp_path: Path) -> None:
		"""Test parsing a file with content that can't be guessed."""
		# Arrange
		# Create a file with ambiguous content that can't be easily guessed
		test_file = tmp_path / "test.unknown"
		test_file.write_text("This is just plain text with no specific format.")

		# Mock should_parse to return True to force parsing
		with (
			patch.object(parser.file_filter, "should_parse", return_value=True),
			patch("codemap.analyzer.tree_parser.get_lexer_for_filename") as mock_get_lexer,
			patch("codemap.analyzer.tree_parser.guess_lexer") as mock_guess_lexer,
		):
			mock_get_lexer.side_effect = ClassNotFound("No lexer found")
			mock_guess_lexer.side_effect = ClassNotFound("No lexer found")

			# Act
			file_info = parser.parse_file(test_file)

			# Assert
			assert file_info["language"] == "unknown"
			assert file_info["content"] == "This is just plain text with no specific format."

	def test_parse_file_unicode_error(self, parser: CodeParser) -> None:
		"""Test parsing a file with Unicode decode error."""
		# Arrange
		# Patch both should_parse and file open operation
		with (
			patch.object(parser.file_filter, "should_parse", return_value=True),
			patch("builtins.open", mock_open()) as mock_file,
		):
			mock_file.side_effect = UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")

			# Act
			file_info = parser.parse_file(Path("test.py"))

			# Assert
			assert file_info["language"] == "unknown"
			assert file_info["content"] == ""
			assert file_info["imports"] == []
			assert file_info["classes"] == []

	def test_parse_file_os_error(self, parser: CodeParser) -> None:
		"""Test parsing a file with OS error."""
		# Arrange
		# Patch both should_parse and file open operation
		with (
			patch.object(parser.file_filter, "should_parse", return_value=True),
			patch("builtins.open", mock_open()) as mock_file,
		):
			mock_file.side_effect = OSError("Permission denied")

			# Act
			file_info = parser.parse_file(Path("test.py"))

			# Assert
			assert file_info["language"] == "unknown"
			assert file_info["content"] == ""
			assert file_info["imports"] == []
			assert file_info["classes"] == []

	def test_parse_file_should_not_parse(self, parser: CodeParser) -> None:
		"""Test parsing a file that should not be parsed."""
		# Arrange
		# Mock should_parse to return False
		with patch.object(parser.file_filter, "should_parse", return_value=False):
			# Act
			file_info = parser.parse_file(Path("__pycache__/module.py"))

			# Assert
			assert file_info["language"] == "unknown"
			assert file_info["content"] == ""
			assert file_info["imports"] == []
			assert file_info["classes"] == []
