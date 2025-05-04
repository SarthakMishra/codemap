"""Tests for the diff splitter implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codemap.git.diff_splitter.schemas import DiffChunk
from codemap.git.diff_splitter.splitter import DiffSplitter
from codemap.git.utils import GitDiff
from tests.base import GitTestBase
from tests.conftest import skip_git_tests


@pytest.mark.unit
@pytest.mark.git
@skip_git_tests
class TestDiffSplitter(GitTestBase):
	"""Tests for the DiffSplitter class."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Create a mock repo path
		self.repo_root = Path("/mock/repo")

		# Create a splitter instance
		self.splitter = DiffSplitter(self.repo_root)

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

	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter.are_sentence_transformers_available")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter.is_model_available")
	@patch("codemap.git.diff_splitter.splitter.loading_spinner")
	def test_split_diff_basic(
		self,
		_mock_spinner,
		mock_is_model_avail,
		mock_are_st_avail,
		mock_is_test,
		mock_filter_files,
	) -> None:
		"""Test basic functionality of split_diff method."""
		# Arrange
		mock_is_test.return_value = False
		mock_are_st_avail.return_value = True
		mock_is_model_avail.return_value = True
		mock_filter_files.return_value = (["file1.py", "file2.py"], [])

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

		# Mock _split_semantic to return our expected chunks
		with patch.object(self.splitter, "_split_semantic", return_value=expected_chunks):
			# Act
			result_chunks, large_files = self.splitter.split_diff(self.sample_diff)

			# Assert
			assert result_chunks == expected_chunks
			assert large_files == []
			mock_filter_files.assert_called_once_with(["file1.py", "file2.py"], False)  # noqa: FBT003
			mock_are_st_avail.assert_called()
			mock_is_model_avail.assert_called()

	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	def test_split_diff_empty_files(self, mock_is_test, mock_filter_files) -> None:
		"""Test split_diff with empty files list."""
		# Arrange
		mock_is_test.return_value = False
		empty_diff = GitDiff(files=[], content="", is_staged=False)

		# Act
		result_chunks, large_files = self.splitter.split_diff(empty_diff)

		# Assert
		assert result_chunks == []
		assert large_files == []
		mock_filter_files.assert_not_called()

	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter.are_sentence_transformers_available")
	def test_split_diff_no_sentence_transformers(self, mock_are_st_avail, mock_is_test, mock_filter_files) -> None:
		"""Test split_diff when sentence_transformers is not available."""
		# Arrange
		mock_is_test.return_value = False
		mock_are_st_avail.return_value = False  # Set to False to trigger the ValueError
		mock_filter_files.return_value = (["file1.py"], [])

		# Act & Assert
		with pytest.raises(ValueError, match="Semantic splitting is not available"):
			self.splitter.split_diff(self.sample_diff)

	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter.is_model_available")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter._check_model_availability")
	@patch("codemap.git.diff_splitter.splitter.loading_spinner")
	def test_split_diff_model_not_available(
		self,
		_mock_spinner,
		mock_check_model,
		mock_is_model_avail,
		mock_is_test,
		mock_filter_files,
	) -> None:
		"""Test split_diff when embedding model is not available."""
		# Arrange
		mock_is_test.return_value = False
		mock_is_model_avail.return_value = False  # Simulate model not being available
		mock_check_model.return_value = False  # Ensure check also returns False
		mock_filter_files.return_value = (["file1.py"], [])

		# Act & Assert
		with pytest.raises(ValueError, match="Semantic splitting failed: embedding model could not be loaded"):
			self.splitter.split_diff(self.sample_diff)

	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.re")  # Mock re for large content check
	def test_split_diff_large_content(
		self, mock_re: MagicMock, mock_is_test: MagicMock, mock_filter_files: MagicMock
	) -> None:
		"""Test split_diff with large diff content (should still process files)."""
		# Arrange
		mock_is_test.return_value = False
		mock_filter_files.return_value = (["file1.py", "file2.py"], [])  # filter_valid_files still returns valid files
		mock_re.findall.return_value = [
			("a/file1.py", "file1.py"),
			("a/file2.py", "file2.py"),
		]  # Mock finding files in large diff

		# Create a splitter instance inside patches
		with (
			patch(
				"codemap.git.diff_splitter.splitter.DiffSplitter.are_sentence_transformers_available",
				return_value=True,
			),
			patch("codemap.git.diff_splitter.splitter.DiffSplitter.is_model_available", return_value=True),
			patch("codemap.git.diff_splitter.splitter.loading_spinner"),
			patch.object(
				DiffSplitter, "_split_semantic", return_value=[]
			) as mock_split_semantic,  # Mock the semantic split
		):
			splitter = DiffSplitter(self.repo_root)
			# Make sample diff content large
			large_sample_diff = GitDiff(
				files=["file1.py", "file2.py"],  # Original files list
				content="a" * (splitter.max_file_size_for_llm + 10),  # Large content
				is_staged=False,
			)

			# Act
			result_chunks, large_files = splitter.split_diff(large_sample_diff)

			# Assert
			# Even though content was large, files were extracted and passed to filter_valid_files
			# _split_semantic was called with the modified diff (empty content, extracted files)
			mock_filter_files.assert_called_once_with(["file1.py", "file2.py"], False)  # noqa: FBT003
			mock_split_semantic.assert_called_once()
			assert result_chunks == []  # As mocked
			assert large_files == []  # Should always be empty

	@patch("codemap.git.diff_splitter.splitter.SemanticSplitStrategy")
	def test_split_semantic_success(self, mock_semantic_strategy_cls) -> None:
		"""Test the _split_semantic method with successful semantic splitting."""
		# Arrange
		mock_strategy = MagicMock()
		mock_semantic_strategy_cls.return_value = mock_strategy

		expected_chunks = [
			DiffChunk(files=["file1.py"], content="chunk1", description="Description 1"),
			DiffChunk(files=["file2.py"], content="chunk2", description="Description 2"),
		]
		mock_strategy.split.return_value = expected_chunks

		# Act
		result = self.splitter._split_semantic(self.sample_diff)

		# Assert
		assert result == expected_chunks
		mock_semantic_strategy_cls.assert_called_once()
		mock_strategy.split.assert_called_once_with(self.sample_diff)

	@patch("codemap.git.diff_splitter.splitter.FileSplitStrategy")  # Mock fallback strategy
	def test_split_semantic_st_unavailable_fallback(self, mock_file_strategy_cls) -> None:
		"""Test _split_semantic fallback when sentence transformers are unavailable."""
		# Arrange
		mock_file_strategy = MagicMock()
		mock_file_strategy_cls.return_value = mock_file_strategy
		# Define what FileSplitStrategy would *actually* return for the sample diff
		# It creates one chunk per file mentioned in the diff header
		expected_fallback_chunks = [
			DiffChunk(
				files=["file1.py"],
				# Content should ONLY be the diff for file1.py
				content=(
					"diff --git a/file1.py b/file1.py\n"
					"index 1234567..abcdefg 100644\n"
					"--- a/file1.py\n"
					"+++ b/file1.py\n"
					"@@ -10,7 +10,7 @@ def existing_function():\n"
					"    pass\n"
				),
				# Description uses determine_commit_type + create_chunk_description
				description="chore: update file1.py",
			),
			DiffChunk(
				files=["file2.py"],
				# Content should ONLY be the diff for file2.py
				content=(
					"diff --git a/file2.py b/file2.py\n"
					"index 2345678..bcdefgh 100645\n"
					"--- a/file2.py\n"
					"+++ b/file2.py\n"
					"@@ -5,3 +5,6 @@ def old_function():\n"
					"    pass\n"
				),
				description="chore: update file2.py",
			),
		]
		mock_file_strategy.split.return_value = expected_fallback_chunks

		# Simulate sentence transformers being unavailable
		with patch.object(self.splitter, "are_sentence_transformers_available", return_value=False):
			# Act
			# We call _split_semantic, which should internally call the fallback _create_basic_file_chunk,
			# which instantiates and calls FileSplitStrategy.split (which we mocked)
			result = self.splitter._split_semantic(self.sample_diff)

		# Assert
		# The result should be what our mocked FileSplitStrategy.split returned
		assert result == expected_fallback_chunks
		mock_file_strategy_cls.assert_called_once()  # Verify fallback class was instantiated
		mock_file_strategy.split.assert_called_once_with(self.sample_diff)
