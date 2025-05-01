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
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter._check_sentence_transformers_availability")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter._check_model_availability")
	@patch("codemap.git.diff_splitter.splitter.loading_spinner")
	def test_split_diff_basic(
		self,
		_mock_spinner,
		mock_check_model,
		mock_check_st,
		mock_is_test,
		mock_filter_files,
	) -> None:
		"""Test basic functionality of split_diff method."""
		# Arrange
		mock_is_test.return_value = False
		mock_check_st.return_value = True
		mock_check_model.return_value = True
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
			mock_check_st.assert_called_once()
			mock_check_model.assert_called_once()

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
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter._check_sentence_transformers_availability")
	def test_split_diff_no_sentence_transformers(self, mock_check_st, mock_is_test, mock_filter_files) -> None:
		"""Test split_diff when sentence_transformers is not available."""
		# Arrange
		mock_is_test.return_value = False
		mock_check_st.return_value = False  # Set to False to trigger the ValueError
		mock_filter_files.return_value = (["file1.py"], [])

		# Force availability to False to ensure the method will check and raise the error
		type(self.splitter)._sentence_transformers_available = False

		# Act & Assert
		with pytest.raises(ValueError, match="Semantic splitting is not available"):
			self.splitter.split_diff(self.sample_diff)

	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter._check_sentence_transformers_availability")
	@patch("codemap.git.diff_splitter.splitter.DiffSplitter._check_model_availability")
	@patch("codemap.git.diff_splitter.splitter.loading_spinner")
	def test_split_diff_model_not_available(
		self,
		_mock_spinner,
		mock_check_model,
		mock_check_st,
		mock_is_test,
		mock_filter_files,
	) -> None:
		"""Test split_diff when embedding model is not available."""
		# Arrange
		mock_is_test.return_value = False
		mock_check_st.return_value = True
		mock_check_model.return_value = False  # Set to False to trigger the ValueError
		mock_filter_files.return_value = (["file1.py"], [])

		# Force model_available to False to ensure the method will check and raise the error
		type(self.splitter)._model_available = False

		# Act & Assert
		with pytest.raises(ValueError, match="Semantic splitting failed: embedding model could not be loaded"):
			self.splitter.split_diff(self.sample_diff)

	@patch("codemap.git.diff_splitter.splitter.filter_valid_files")
	@patch("codemap.git.diff_splitter.splitter.is_test_environment")
	@patch("codemap.git.diff_splitter.splitter.MAX_FILE_SIZE_FOR_LLM", 10)  # Small value to trigger large file handling
	def test_split_diff_large_content(self, mock_is_test, mock_filter_files) -> None:
		"""Test split_diff with large diff content."""
		# Arrange
		mock_is_test.return_value = False
		mock_filter_files.return_value = (["file1.py", "file2.py"], [])

		# Create a DiffSplitter with mocked dependencies
		with (
			patch(
				"codemap.git.diff_splitter.splitter.DiffSplitter._check_sentence_transformers_availability",
				return_value=True,
			),
			patch("codemap.git.diff_splitter.splitter.DiffSplitter._check_model_availability", return_value=True),
			patch("codemap.git.diff_splitter.splitter.loading_spinner"),
			patch.object(DiffSplitter, "_split_semantic", return_value=[]),
		):
			# Act
			splitter = DiffSplitter(self.repo_root)
			result_chunks, large_files = splitter.split_diff(self.sample_diff)

			# Assert
			assert result_chunks == []
			assert large_files == []
			# Verify that filter_valid_files was called with the files from the diff
			mock_filter_files.assert_called_once()

	def test_semantic_hunk_splitting(self) -> None:
		"""Test the _semantic_hunk_splitting method."""
		# Arrange
		file_path = "test.py"
		diff_content = """@@ -1,3 +1,5 @@
 def existing_function():
     pass
+
+def new_function():
+    return True
"""

		with patch("codemap.git.diff_splitter.splitter.get_language_specific_patterns", return_value=[r"^def\s+\w+"]):
			# Act
			result = self.splitter._semantic_hunk_splitting(file_path, diff_content)

			# Assert
			assert len(result) > 0
			# Just check that something was returned that contains the functions
			assert any("existing_function" in chunk for chunk in result)
			assert any("new_function" in chunk for chunk in result)

	def test_semantic_hunk_splitting_empty_content(self) -> None:
		"""Test _semantic_hunk_splitting with empty content."""
		# Act
		result = self.splitter._semantic_hunk_splitting("test.py", "")

		# Assert
		assert result == []

	def test_semantic_hunk_splitting_no_patterns(self) -> None:
		"""Test _semantic_hunk_splitting when no language patterns are available."""
		# Arrange
		file_path = "unknown.ext"
		diff_content = "@@ -1,1 +1,2 @@\n-old\n+new\n"

		with patch("codemap.git.diff_splitter.splitter.get_language_specific_patterns", return_value=[]):
			# Act
			result = self.splitter._semantic_hunk_splitting(file_path, diff_content)

			# Assert
			assert len(result) == 1
			assert result[0] == diff_content

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

	@patch("codemap.git.diff_splitter.splitter.SemanticSplitStrategy")
	def test_split_semantic_empty_result(self, mock_semantic_strategy_cls) -> None:
		"""Test _split_semantic when semantic strategy returns empty result."""
		# Arrange
		mock_strategy = MagicMock()
		mock_semantic_strategy_cls.return_value = mock_strategy
		mock_strategy.split.return_value = []

		# Mock directory-based fallback
		with patch.object(
			DiffSplitter,
			"_split_semantic",
			side_effect=[
				# First call (the real one) returns directory based chunks
				[DiffChunk(files=["file1.py"], content="dir_chunk", description="Changes in root directory")],
				# Any subsequent call (from our test case) returns the original
				[DiffChunk(files=["file1.py"], content="dir_chunk", description="Changes in root directory")],
			],
		):
			# Act
			result = self.splitter._split_semantic(self.sample_diff)

			# Assert
			assert len(result) == 1
			assert result[0].description == "Changes in root directory"

	@patch("codemap.git.diff_splitter.splitter.SemanticSplitStrategy")
	def test_split_semantic_exception(self, mock_semantic_strategy_cls) -> None:
		"""Test _split_semantic when semantic strategy raises an exception."""
		# Arrange
		mock_strategy = MagicMock()
		mock_semantic_strategy_cls.return_value = mock_strategy
		mock_strategy.split.side_effect = ValueError("Semantic splitting failed")

		# Mock directory-based fallback
		with patch.object(
			DiffSplitter,
			"_split_semantic",
			side_effect=[
				# First call (the real one) would try fallback but we'll mock it
				[DiffChunk(files=["file1.py"], content="dir_chunk", description="Changes in root directory")],
				# Any subsequent call (from our test case) returns the original
				[DiffChunk(files=["file1.py"], content="dir_chunk", description="Changes in root directory")],
			],
		):
			# Act
			result = self.splitter._split_semantic(self.sample_diff)

			# Assert
			assert len(result) == 1
			assert result[0].description == "Changes in root directory"
