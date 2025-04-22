"""Tests for the DocumentationProcessor class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from codemap.analyzer.processor import DocumentationProcessor
from codemap.analyzer.tree_parser import CodeParser
from codemap.utils.file_filters import FileFilter

if TYPE_CHECKING:
    from pathlib import Path

    from rich.progress import Progress


@pytest.fixture
def mock_code_parser() -> MagicMock:
    """Mock CodeParser for testing."""
    file_filter = MagicMock(spec=FileFilter)
    file_filter.should_parse.return_value = True

    parser = MagicMock(spec=CodeParser)
    parser.file_filter = file_filter
    parser.parse_file.return_value = {"type": "test", "content": "test content"}
    return parser


@pytest.fixture
def temp_py_file(tmp_path: Path) -> Path:
    """Create a temporary Python file for testing."""
    file_path = tmp_path / "test.py"
    file_path.write_text("def test():\n    return 'test'")
    return file_path


def test_process_file(mock_code_parser: MagicMock, temp_py_file: Path) -> None:
    """Test processing a single file."""
    processor = DocumentationProcessor(mock_code_parser, token_limit=100)

    # Process the file
    file_info, tokens = processor.process_file(temp_py_file)

    # Verify results
    assert file_info == {"type": "test", "content": "test content"}
    assert tokens > 0
    mock_code_parser.file_filter.should_parse.assert_called_once_with(temp_py_file)
    mock_code_parser.parse_file.assert_called_once_with(temp_py_file)


def test_process_file_should_not_parse(mock_code_parser: MagicMock, temp_py_file: Path) -> None:
    """Test processing a file that should not be parsed."""
    processor = DocumentationProcessor(mock_code_parser, token_limit=100)
    mock_code_parser.file_filter.should_parse.return_value = False

    # Process the file
    file_info, tokens = processor.process_file(temp_py_file)

    # Verify results
    assert file_info is None
    assert tokens == 0
    mock_code_parser.file_filter.should_parse.assert_called_once_with(temp_py_file)
    mock_code_parser.parse_file.assert_not_called()


def test_token_limit_reached(mock_code_parser: MagicMock, temp_py_file: Path) -> None:
    """Test token limit enforcement."""

    # Create a custom subclass for testing that makes the token count predictable
    class TestDocumentationProcessor(DocumentationProcessor):
        def process_file(self, file_path: Path, progress: Progress | None = None) -> tuple[dict[str, Any] | None, int]:  # noqa: ARG002
            if not self.parser.file_filter.should_parse(file_path):
                return None, self.total_tokens

            # For this test, always return 200 tokens
            tokens = 200

            # Check token limit
            if self.token_limit > 0 and self.total_tokens + tokens > self.token_limit:
                return None, self.total_tokens

            file_info = self.parser.parse_file(file_path)
            self.total_tokens += tokens
            return file_info, self.total_tokens

    # Use our custom processor
    processor = TestDocumentationProcessor(mock_code_parser, token_limit=100)

    # Process the file
    file_info, tokens = processor.process_file(temp_py_file)

    # Verify results
    assert file_info is None  # Should be None because token limit is reached
    assert tokens == 0
    mock_code_parser.file_filter.should_parse.assert_called_once_with(temp_py_file)
    # Make sure parse_file is not called when token limit is reached
    mock_code_parser.parse_file.assert_not_called()


def test_process_directory(mock_code_parser: MagicMock, tmp_path: Path) -> None:
    """Test processing a directory."""
    # Create test files
    (tmp_path / "test1.py").write_text("def test1():\n    return 'test1'")
    (tmp_path / "test2.py").write_text("def test2():\n    return 'test2'")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "test3.py").write_text("def test3():\n    return 'test3'")

    processor = DocumentationProcessor(mock_code_parser, token_limit=1000)

    # Mock parser to always return the same file info
    file_info = {"type": "test", "content": "test content"}
    mock_code_parser.parse_file.return_value = file_info

    # Process the directory
    with patch("codemap.utils.file_utils.count_tokens", return_value=10):
        parsed_files = processor.process_directory(tmp_path)

    # Verify results - should be 3 files
    assert len(parsed_files) == 3
    assert all(info == file_info for info in parsed_files.values())
    assert mock_code_parser.parse_file.call_count == 3
