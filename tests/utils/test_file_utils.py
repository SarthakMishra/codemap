"""Tests for file utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from codemap.utils.file_utils import count_tokens, ensure_directory_exists
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

	def test_ensure_directory_exists_success(self, tmp_path: Path) -> None:
		"""Test ensuring a directory exists with success."""
		# Directory that doesn't exist yet
		test_dir = tmp_path / "new_dir"
		ensure_directory_exists(test_dir)
		assert test_dir.exists()
		assert test_dir.is_dir()

		# Directory that already exists
		ensure_directory_exists(test_dir)  # Should not raise an exception
		assert test_dir.exists()

	def test_ensure_directory_exists_permission_error(self) -> None:
		"""Test ensuring a directory exists with permission error."""
		with patch("pathlib.Path.mkdir") as mock_mkdir:
			mock_mkdir.side_effect = PermissionError("Permission denied")

			with pytest.raises(PermissionError, match="Permission denied"):
				ensure_directory_exists(Path("/invalid/path"))

	def test_ensure_directory_exists_not_a_directory(self, tmp_path: Path) -> None:
		"""Test ensuring a directory exists when path exists but is not a directory."""
		# Create a file
		test_file = tmp_path / "file.txt"
		test_file.touch()

		# Try to ensure directory exists for a file path
		with pytest.raises(NotADirectoryError):
			ensure_directory_exists(test_file)
