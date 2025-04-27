"""Tests for the analyzer/processor.py module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.progress import Progress

from codemap.analyzer.processor import DocumentationProcessor
from codemap.analyzer.tree_parser import CodeParser
from tests.base import FileSystemTestBase


@pytest.fixture
def mock_file_filter() -> Mock:
	"""Mock file filter for testing."""
	mock_filter = Mock()
	mock_filter.should_parse.return_value = True
	return mock_filter


@pytest.fixture
def mock_parser(mock_file_filter: Mock) -> Mock:
	"""Mock code parser for testing."""
	mock = Mock(spec=CodeParser)
	mock.file_filter = mock_file_filter
	mock.parse_file.return_value = {"imports": [], "classes": [], "content": "test content"}
	return mock


@pytest.fixture
def processor(mock_parser: Mock) -> DocumentationProcessor:
	"""Create a DocumentationProcessor with mocked parser."""
	return DocumentationProcessor(parser=mock_parser)


@pytest.mark.unit
@pytest.mark.analyzer
class TestDocumentationProcessorInitialization:
	"""Test initialization of DocumentationProcessor."""

	def test_init_with_defaults(self, mock_parser: Mock) -> None:
		"""Test initialization with default values."""
		processor = DocumentationProcessor(parser=mock_parser)
		assert processor.parser == mock_parser
		assert processor.token_limit == 10000
		assert processor.total_tokens == 0

	def test_init_with_custom_token_limit(self, mock_parser: Mock) -> None:
		"""Test initialization with custom token limit."""
		processor = DocumentationProcessor(parser=mock_parser, token_limit=5000)
		assert processor.token_limit == 5000


@pytest.mark.unit
@pytest.mark.analyzer
class TestProcessFile:
	"""Test process_file method of DocumentationProcessor."""

	def test_process_file_success(self, processor: DocumentationProcessor, mock_parser: Mock) -> None:
		"""Test successful file processing."""
		with patch("codemap.analyzer.processor.count_tokens", return_value=100):
			file_path = Path("test_file.py")
			file_info, new_total = processor.process_file(file_path)

			# Check results
			assert file_info is not None
			assert new_total == 100
			assert processor.total_tokens == 100

			# Verify parser was called correctly
			mock_parser.parse_file.assert_called_once_with(file_path)

	def test_process_file_should_not_parse(self, processor: DocumentationProcessor, mock_file_filter: Mock) -> None:
		"""Test file processing when filter says not to parse."""
		mock_file_filter.should_parse.return_value = False

		file_path = Path("test_file.py")
		file_info, new_total = processor.process_file(file_path)

		# Check results
		assert file_info is None
		assert new_total == 0
		assert processor.total_tokens == 0

	def test_process_file_token_limit_reached(self, processor: DocumentationProcessor) -> None:
		"""Test file processing when token limit is reached."""
		processor.token_limit = 50
		processor.total_tokens = 40

		with patch("codemap.analyzer.processor.count_tokens", return_value=20):
			file_path = Path("test_file.py")
			file_info, new_total = processor.process_file(file_path)

			# Check results - should not process file
			assert file_info is None
			assert new_total == 40  # Unchanged
			assert processor.total_tokens == 40  # Unchanged

	def test_process_file_with_progress(self, processor: DocumentationProcessor) -> None:
		"""Test file processing with progress bar."""
		with patch("codemap.analyzer.processor.count_tokens", return_value=100):
			file_path = Path("test_file.py")

			progress = Progress()
			progress.add_task("Testing", total=None)
			progress.update = MagicMock()  # Mock update method

			file_info, new_total = processor.process_file(file_path, progress)

			# Check results
			assert file_info is not None
			assert new_total == 100

			# Verify progress bar was updated
			progress.update.assert_called_once()

	def test_process_file_with_exception(self, processor: DocumentationProcessor) -> None:
		"""Test file processing with file read exception."""
		with (
			patch(
				"codemap.analyzer.processor.count_tokens",
				side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "test error"),
			),
			patch.object(logging.getLogger("codemap.analyzer.processor"), "warning") as mock_warning,
		):
			file_path = Path("test_file.py")
			file_info, new_total = processor.process_file(file_path)

			# Check results
			assert file_info is None
			assert new_total == 0
			assert processor.total_tokens == 0

			# Verify warning was logged
			mock_warning.assert_called_once()

	def test_process_file_os_error(self, processor: DocumentationProcessor) -> None:
		"""Test file processing with OSError."""
		with (
			patch("codemap.analyzer.processor.count_tokens", side_effect=OSError("test error")),
			patch.object(logging.getLogger("codemap.analyzer.processor"), "warning") as mock_warning,
		):
			file_path = Path("test_file.py")
			file_info, new_total = processor.process_file(file_path)

			# Check results
			assert file_info is None
			assert new_total == 0

			# Verify warning was logged
			mock_warning.assert_called_once()


@pytest.mark.unit
@pytest.mark.analyzer
class TestProcessDirectory:
	"""Test process_directory method of DocumentationProcessor."""

	def test_process_directory_empty(self, processor: DocumentationProcessor) -> None:
		"""Test processing an empty directory."""
		with patch("os.walk", return_value=[]):
			result = processor.process_directory(Path("/test/dir"))
			assert result == {}

	def test_process_directory(self, processor: DocumentationProcessor) -> None:
		"""Test processing a directory with files."""
		# Mock os.walk to return some files
		mock_walk_data = [
			("/test/dir", ["subdir"], ["file1.py", "file2.py"]),
			("/test/dir/subdir", [], ["file3.py"]),
		]

		with patch("os.walk", return_value=mock_walk_data):
			# Mock process_file to return some data for each file
			def mock_process_file(file_path: Path, _: object = None) -> tuple[dict[str, str], int]:
				return {"content": f"Content of {file_path.name}"}, 100

			processor.process_file = MagicMock(side_effect=mock_process_file)

			result = processor.process_directory(Path("/test/dir"))

			# Check results
			assert len(result) == 3
			assert Path("/test/dir/file1.py") in result
			assert Path("/test/dir/file2.py") in result
			assert Path("/test/dir/subdir/file3.py") in result

			# Verify process_file was called for each file
			assert processor.process_file.call_count == 3

	def test_process_directory_token_limit_reached(self, processor: DocumentationProcessor) -> None:
		"""Test processing a directory when token limit is reached."""
		# Mock os.walk to return some files
		mock_walk_data = [
			("/test/dir", ["subdir"], ["file1.py", "file2.py"]),
			("/test/dir/subdir", [], ["file3.py"]),
		]

		# Set up processor to reach token limit after first file
		processor.token_limit = 100

		# Instead of using side_effect with a list, create a custom function
		# that returns different values based on which file is processed
		def mock_process_file_fn(file_path: Path, _: object = None) -> tuple[dict[str, str] | None, int]:
			if str(file_path).endswith("file1.py"):
				# First file is processed successfully
				return {"content": "Content of file1.py"}, 100
			# Other files reach token limit
			return None, 100

		with patch("os.walk", return_value=mock_walk_data):
			processor.process_file = MagicMock(side_effect=mock_process_file_fn)

			result = processor.process_directory(Path("/test/dir"))

			# Check results - should only have processed first file
			assert len(result) == 1
			assert Path("/test/dir/file1.py") in result

			# Verify process_file was called at least once
			assert processor.process_file.call_count >= 1

	def test_process_directory_with_progress(self, processor: DocumentationProcessor) -> None:
		"""Test processing a directory with progress tracking."""
		# Mock os.walk to return some files
		mock_walk_data = [
			("/test/dir", [], ["file1.py"]),
		]

		# We need to mock the entire Progress context manager
		progress_mock = MagicMock()
		progress_mock.task_ids = ["task_id"]

		# Use context manager to avoid calling actual Progress methods
		with (
			patch("os.walk", return_value=mock_walk_data),
			patch("codemap.analyzer.processor.Progress") as mock_progress_cls,
		):
			# Set up the context manager to return our mock
			mock_progress_cls.return_value.__enter__.return_value = progress_mock

			# Mock process_file to avoid real file operations
			processor.process_file = MagicMock(return_value=({"content": "test"}, 100))

			# Call the method being tested
			processor.process_directory(Path("/test/dir"))

			# Verify progress task was added at least once
			assert progress_mock.add_task.called
			# Verify process_file was called with the progress mock
			assert processor.process_file.called

			# Assert that file path was correctly passed to process_file
			file_arg = processor.process_file.call_args[0][0]
			assert str(file_arg).endswith("file1.py")


@pytest.mark.unit
@pytest.mark.analyzer
class TestProcess:
	"""Test process method of DocumentationProcessor."""

	def test_process_file_path(self, processor: DocumentationProcessor) -> None:
		"""Test processing when target is a file."""
		target_path = Path("/test/file.py")

		# Instead of patching is_file, mock the processor's process_file/process_directory methods
		processor.process_file = MagicMock(return_value=({"content": "test"}, 100))
		processor.process_directory = MagicMock()

		# Use a simpler approach with MagicMock for the path
		with patch.object(Path, "is_file") as mock_is_file:
			# Set up the mock to return True for our test path
			mock_is_file.return_value = True

			result = processor.process(target_path)

			# Check results
			assert len(result) == 1
			assert target_path in result

			# Verify process_file was called and process_directory was not
			processor.process_file.assert_called_once_with(target_path)
			processor.process_directory.assert_not_called()

	def test_process_directory_path(self, processor: DocumentationProcessor) -> None:
		"""Test processing when target is a directory."""
		target_path = Path("/test/dir")

		# Instead of patching is_file, mock the processor's process_file/process_directory methods
		processor.process_file = MagicMock()
		expected_result = {Path("/test/dir/file.py"): {"content": "test"}}
		processor.process_directory = MagicMock(return_value=expected_result)

		# Use a simpler approach with MagicMock for the path
		with patch.object(Path, "is_file") as mock_is_file:
			# Set up the mock to return False for our test path (it's a directory)
			mock_is_file.return_value = False

			result = processor.process(target_path)

			# Check results
			assert result == expected_result

			# Verify process_directory was called and process_file was not
			processor.process_directory.assert_called_once_with(target_path)
			processor.process_file.assert_not_called()


@pytest.mark.integration
@pytest.mark.analyzer
@pytest.mark.fs
class TestDocumentationProcessorWithFiles(FileSystemTestBase):
	"""Integration tests for DocumentationProcessor with actual files."""

	def test_process_real_files(self, mock_parser: Mock) -> None:
		"""Test processing real files from a test directory."""
		# Create test directory with files
		self.create_test_file("test_file1.py", "def test_function():\n    return True")
		self.create_test_file("test_file2.py", "class TestClass:\n    def method(self):\n        pass")
		self.create_test_file("subdir/test_file3.py", "import os\n\ndef another_function():\n    return os.getcwd()")

		# Create processor with real parser
		processor = DocumentationProcessor(parser=mock_parser)

		# Make count_tokens return a realistic value based on content length
		with patch("codemap.analyzer.processor.count_tokens", side_effect=lambda p: len(p.read_text().split())):
			# Process the test directory
			result = processor.process(self.temp_dir)

			# Check results
			assert len(result) == 3
			assert mock_parser.parse_file.call_count == 3

			# Check that token count was updated
			assert processor.total_tokens > 0

	def test_token_limit_enforcement(self, mock_parser: Mock) -> None:
		"""Test that token limit is properly enforced."""
		# Create test files
		self.create_test_file("test_file1.py", "def test_function():\n    return True")
		self.create_test_file("test_file2.py", "class TestClass:\n    def method(self):\n        pass")

		# Create processor with very low token limit
		processor = DocumentationProcessor(parser=mock_parser, token_limit=3)

		# Make count_tokens return a consistent value for each file
		with patch("codemap.analyzer.processor.count_tokens", return_value=2):
			# Process the test directory
			result = processor.process(self.temp_dir)

			# Should have processed only the first file before hitting limit
			assert len(result) == 1
			assert processor.total_tokens == 2
			assert mock_parser.parse_file.call_count == 1

	def test_process_with_infinite_token_limit(self, mock_parser: Mock) -> None:
		"""Test processing with infinite token limit (0)."""
		# Create test files
		self.create_test_file("test_file1.py", "def test_function():\n    return True")
		self.create_test_file("test_file2.py", "class TestClass:\n    def method(self):\n        pass")

		# Create processor with infinite token limit (0)
		processor = DocumentationProcessor(parser=mock_parser, token_limit=0)

		# Make count_tokens return a high value
		with patch("codemap.analyzer.processor.count_tokens", return_value=10000):
			# Process the test directory
			result = processor.process(self.temp_dir)

			# Should have processed all files regardless of token count
			assert len(result) == 2
			assert processor.total_tokens == 20000
			assert mock_parser.parse_file.call_count == 2
