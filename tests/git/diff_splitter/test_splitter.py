"""Tests for the diff splitter implementation."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from codemap.config import ConfigLoader
from codemap.git.diff_splitter.schemas import DiffChunk
from codemap.git.diff_splitter.splitter import DiffSplitter
from codemap.git.utils import GitDiff
from tests.base import GitTestBase


@pytest.mark.unit
@pytest.mark.git
class TestDiffSplitter(GitTestBase):
	"""Tests for the DiffSplitter class."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Create a mock repo path
		self.repo_root = Path("/mock/repo")

		# Create mock ConfigLoader with proper config structure
		self.mock_config = Mock(spec=ConfigLoader)
		self.mock_config.get = MagicMock()
		self.mock_config.get.repo_root = self.repo_root
		self.mock_config.get.commit = MagicMock()
		self.mock_config.get.commit.diff_splitter = MagicMock()

		# Set required diff_splitter config values
		self.mock_config.get.commit.diff_splitter.similarity_threshold = 0.6
		self.mock_config.get.commit.diff_splitter.directory_similarity_threshold = 0.3
		self.mock_config.get.commit.diff_splitter.min_chunks_for_consolidation = 2
		self.mock_config.get.commit.diff_splitter.max_chunks_before_consolidation = 20
		self.mock_config.get.commit.diff_splitter.max_file_size_for_llm = 50000
		self.mock_config.get.commit.diff_splitter.max_log_diff_size = 1000

		# Create a splitter instance
		self.splitter = DiffSplitter(config_loader=self.mock_config)

		# Create a sample diff for testing
		self.sample_diff = GitDiff(
			files=["file1.py", "file2.py"],
			content="""diff --git a/file1.py b/file1.py
index 1234567..abcdefg 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@ def existing_function():
    pass
diff --git a/file2.py b/file2.py
index 2345678..bcdefgh 100645
--- a/file2.py
+++ b/file2.py
@@ -5,3 +5,6 @@ def old_function():
    pass""",
			is_staged=False,
		)

	@pytest.mark.asyncio
	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.SemanticSplitStrategy")
	async def test_split_diff_basic(
		self,
		mock_semantic_strategy_cls,
		mock_is_test,
		mock_filter_files,
	) -> None:
		"""Test basic functionality of split_diff method."""
		# Arrange
		mock_is_test.return_value = False
		mock_filter_files.return_value = (["file1.py", "file2.py"], [])

		# Setup mock strategy
		mock_strategy = AsyncMock()
		mock_semantic_strategy_cls.return_value = mock_strategy

		expected_chunks = [
			DiffChunk(
				files=["file1.py"],
				content="file1 content",
				description="Changes in file1.py",
			),
			DiffChunk(
				files=["file2.py"],
				content="file2 content",
				description="Changes in file2.py",
			),
		]
		mock_strategy.split.return_value = expected_chunks

		# Act
		result_chunks, large_files = await self.splitter.split_diff(self.sample_diff)

		# Assert
		assert result_chunks == expected_chunks
		assert large_files == []
		mock_filter_files.assert_called_once()
		mock_semantic_strategy_cls.assert_called_once()
		mock_strategy.split.assert_called_once()

	@pytest.mark.asyncio
	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.SemanticSplitStrategy")
	async def test_split_diff_large_content(self, mock_semantic_strategy_cls, mock_is_test, mock_filter_files) -> None:
		"""Test split_diff with large diff content."""
		# Arrange
		mock_is_test.return_value = False
		mock_filter_files.return_value = (["file1.py", "file2.py"], [])

		# Configure mock strategy
		mock_strategy = AsyncMock()
		mock_semantic_strategy_cls.return_value = mock_strategy

		# Create expected chunks
		expected_chunks = [
			DiffChunk(
				files=["file1.py"],
				content="file1 content",
				description="Changes in file1.py",
			),
			DiffChunk(
				files=["file2.py"],
				content="file2 content",
				description="Changes in file2.py",
			),
		]
		mock_strategy.split.return_value = expected_chunks

		# Make sample diff content large
		large_diff_content = "a" * (self.splitter.max_file_size_for_llm + 10)
		large_sample_diff = GitDiff(
			files=["file1.py", "file2.py"],
			content=large_diff_content,
			is_staged=False,
		)

		# Act
		result_chunks, large_files = await self.splitter.split_diff(large_sample_diff)

		# Assert
		assert result_chunks == expected_chunks
		assert large_files == []
		mock_filter_files.assert_called_once()
		mock_strategy.split.assert_called_once()

	@pytest.mark.asyncio
	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.SemanticSplitStrategy")
	@patch("codemap.git.diff_splitter.splitter.logger")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	async def test_split_diff_semantic_error_fallback(
		self, mock_is_test, mock_logger, mock_semantic_strategy_cls, mock_filter_files
	) -> None:
		"""Test that split_diff falls back to basic file chunks when semantic splitting fails."""
		# Arrange
		mock_is_test.return_value = False
		# Mock filter_valid_files to return the files unchanged
		mock_filter_files.return_value = (["file1.py", "file2.py"], [])

		# Mock the semantic strategy to raise an exception
		mock_strategy = AsyncMock()
		mock_semantic_strategy_cls.return_value = mock_strategy
		mock_strategy.split.side_effect = Exception("Semantic splitting failed")

		# Create expected fallback chunks
		expected_fallback_chunks = [
			DiffChunk(files=["file1.py"], content="basic content", description="Basic description")
		]

		# Use a custom diff with valid content
		test_diff = self.sample_diff

		# Mock the fallback method to return our expected chunks
		with patch.object(
			self.splitter, "_create_basic_file_chunk", return_value=expected_fallback_chunks
		) as mock_fallback:
			# Act
			result_chunks, large_files = await self.splitter.split_diff(test_diff)

			# Assert - result should match our mocked fallback chunks
			assert result_chunks == expected_fallback_chunks
			assert large_files == []

			# Verify the correct sequence of calls
			mock_filter_files.assert_called_once()
			mock_strategy.split.assert_called_once()
			mock_fallback.assert_called_once()
			mock_logger.exception.assert_called_once()

	@pytest.mark.asyncio
	async def test_split_diff_with_untracked_files(self, tmp_path: Path) -> None:
		"""Test splitting a diff with untracked files."""
		# Create a new DiffSplitter with a valid config
		mock_config = Mock(spec=ConfigLoader)
		mock_config.get = MagicMock()
		mock_config.get.repo_root = tmp_path
		mock_config.get.commit = MagicMock()
		mock_config.get.commit.diff_splitter = MagicMock()
		mock_config.get.commit.diff_splitter.max_log_diff_size = 1000

		splitter = DiffSplitter(config_loader=mock_config)

		# Create a diff with untracked files
		untracked_diff = GitDiff(
			files=["new_file.py", "another.txt"],
			content="This is content that's not in valid diff format",
			is_staged=False,
			is_untracked=True,
		)

		# Test the special untracked files handling
		chunks, filtered_files = await splitter.split_diff(untracked_diff)

		# Should create one chunk per file
		assert len(chunks) == 2
		assert len(filtered_files) == 0

		# Each chunk should represent one file
		file_names = [chunk.files[0] for chunk in chunks]
		assert "new_file.py" in file_names
		assert "another.txt" in file_names

		# Chunks should be marked as new files
		for chunk in chunks:
			assert chunk.description is not None
			assert chunk.description.startswith("New file:")
