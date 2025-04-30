"""Tests for file utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codemap.gen.utils import determine_output_path
from codemap.utils.file_utils import count_tokens
from tests.base import FileSystemTestBase


@pytest.mark.unit
@pytest.mark.fs
class TestFileUtils(FileSystemTestBase):
	"""Test cases for file utility functions."""

	def test_count_tokens(self) -> None:
		"""Test token counting functionality."""
		# Create a test file
		test_file = self.create_test_file("test.txt", "This is a test file with some tokens")

		# Count tokens
		token_count = count_tokens(test_file)

		# The test file has 8 tokens
		assert token_count == 8

	def test_count_tokens_error(self) -> None:
		"""Test token counting with a non-existent file."""
		non_existent_file = self.temp_dir / "non_existent.txt"

		# Count tokens (should return 0 for non-existent file)
		token_count = count_tokens(non_existent_file)

		assert token_count == 0

	@patch("pathlib.Path.mkdir")
	@patch("datetime.datetime")
	def test_get_output_path_with_explicit_path(self, mock_datetime: MagicMock, mock_mkdir: MagicMock) -> None:
		"""Test get_output_path with an explicit output path."""
		# Set up test data
		repo_root = Path("/repo")
		output_path = Path("/output/doc.md")
		config = {}

		# Call function
		result = determine_output_path(repo_root, output_path, config)

		# Verify result
		assert result == output_path
		# mkdir should not be called for explicit paths
		mock_mkdir.assert_not_called()
		# datetime should not be used for explicit paths
		assert not mock_datetime.called


@pytest.mark.unit
@pytest.mark.fs
class TestOutputPath(FileSystemTestBase):
	"""Test cases for output path generation."""

	@patch("datetime.datetime")
	def test_get_output_path_with_config(self, mock_datetime: MagicMock) -> None:
		"""Test get_output_path using config and no explicit path."""
		# Set up mock datetime
		mock_now = mock_datetime.now.return_value
		mock_now.strftime.return_value = "20240601_123456"

		# Set up test data
		config = {"output_dir": "docs"}

		# Call function
		result = determine_output_path(self.temp_dir, None, config)

		# Verify result
		expected_path = self.temp_dir / "docs" / "documentation_20240601_123456.md"
		assert result == expected_path
		# datetime.now should be called
		mock_datetime.now.assert_called_once()
		mock_now.strftime.assert_called_once_with("%Y%m%d_%H%M%S")

	def test_get_output_path_absolute_dir(self) -> None:
		"""Test get_output_path with absolute directory in config."""
		# Set up test data
		repo_root = Path("/repo")
		output_path = None
		docs_dir = self.temp_dir / "my_docs"
		docs_dir.mkdir(exist_ok=True)
		config = {"output_dir": str(docs_dir)}

		# Call function
		with patch("datetime.datetime") as mock_datetime:
			mock_now = mock_datetime.now.return_value
			mock_now.strftime.return_value = "20240601_123456"

			result = determine_output_path(repo_root, output_path, config)

		# Verify result
		expected_path = docs_dir / "documentation_20240601_123456.md"
		assert result == expected_path
