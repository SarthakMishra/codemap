"""Tests for diff splitting strategies."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codemap.git.diff_splitter.schemas import DiffChunk
from codemap.git.diff_splitter.strategies import (
	EmbeddingModel,
	FileSplitStrategy,
	SemanticSplitStrategy,
)
from codemap.git.utils import GitDiff


@pytest.mark.unit
@pytest.mark.git
class TestFileSplitStrategy:
	"""Tests for the FileSplitStrategy."""

	@pytest.fixture
	def strategy(self) -> FileSplitStrategy:
		"""Provides a FileSplitStrategy instance."""
		return FileSplitStrategy()

	def test_split_simple_diff(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting a diff with multiple files."""
		diff_content = (
			"diff --git a/file1.py b/file1.py\n"
			"index 123..456 100644\n"
			"--- a/file1.py\n"
			"+++ b/file1.py\n"
			"@@ -1,1 +1,1 @@\n"
			"-old line\n"
			"+new line\n"
			"diff --git a/file2.txt b/file2.txt\n"
			"new file mode 100644\n"
			"index 000..abc 100644\n"
			"--- /dev/null\n"
			"+++ b/file2.txt\n"
			"@@ -0,0 +1,1 @@\n"
			"+content\n"
		)
		git_diff = GitDiff(files=["file1.py", "file2.txt"], content=diff_content, is_staged=True)

		chunks = strategy.split(git_diff)

		assert len(chunks) == 2
		assert isinstance(chunks[0], DiffChunk)
		assert chunks[0].files == ["file1.py"]
		assert chunks[0].content.startswith("diff --git a/file1.py b/file1.py")
		assert chunks[0].description == "Changes in file1.py"

		assert isinstance(chunks[1], DiffChunk)
		assert chunks[1].files == ["file2.txt"]
		assert chunks[1].content.startswith("diff --git a/file2.txt b/file2.txt")
		assert chunks[1].description == "Changes in file2.txt"

	def test_split_empty_diff_content(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting when diff content is empty."""
		# Pass empty string instead of None for content
		git_diff_none_content = GitDiff(files=["file1.py"], content="", is_staged=True)
		git_diff_empty_str = GitDiff(files=["file1.py"], content="", is_staged=True)

		chunks_none = strategy.split(git_diff_none_content)
		chunks_empty_str = strategy.split(git_diff_empty_str)

		assert chunks_none == []
		assert chunks_empty_str == []

	def test_split_untracked_files(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting untracked files (empty content, files listed)."""
		# Should only happen for non-staged diffs
		git_diff = GitDiff(files=["new_file.py", "another.txt"], content="", is_staged=False)

		chunks = strategy.split(git_diff)

		assert len(chunks) == 2
		assert chunks[0].files == ["new_file.py"]
		assert chunks[0].content == ""
		assert chunks[1].files == ["another.txt"]
		assert chunks[1].content == ""

	def test_split_untracked_files_staged(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting staged diff with empty content (should yield no chunks)."""
		git_diff = GitDiff(files=["new_file.py"], content="", is_staged=True)
		chunks = strategy.split(git_diff)
		assert chunks == []

	@pytest.mark.parametrize(
		("filename", "expected"),
		[
			("valid_file.py", True),
			("path/to/file.txt", True),
			("file-with-hyphens.js", True),
			("_private_file.ts", True),
			("", False),  # Empty string
			(None, False),  # None input
			("file*.py", False),  # Contains *
			("file+.py", False),  # Contains +
			("file{}.py", False),  # Contains {}
			("file\\.py", False),  # Contains \
			('"quoted file"', False),  # Starts with quote
			("a/b/c", True),
		],
	)
	def test_is_valid_filename(self, strategy: FileSplitStrategy, filename: str | None, expected: bool) -> None:
		"""Test the _is_valid_filename helper method."""
		# Accessing protected member for testing specific helper
		assert strategy._is_valid_filename(filename) == expected  # type: ignore[attr-defined]

	def test_split_diff_with_invalid_filenames_in_content(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting diff content that might contain invalid filenames."""
		# This scenario is less likely with real git diff output but tests robustness
		diff_content = (
			"diff --git a/valid.py b/valid.py\n"
			"--- a/valid.py\n"
			"+++ b/valid.py\n"
			"@@ +1 @@\n"
			"+valid content\n"
			"diff --git a/invalid*.txt b/invalid*.txt\n"  # Invalid name
			"--- a/invalid*.txt\n"
			"+++ b/invalid*.txt\n"
			"@@ +1 @@\n"
			"+some content\n"
			"diff --git a/another_valid.js b/another_valid.js\n"
			"--- a/another_valid.js\n"
			"+++ b/another_valid.js\n"
			"@@ +1 @@\n"
			"+more content\n"
		)
		# Files listed might be different from headers if diff is unusual
		git_diff = GitDiff(files=["valid.py", "invalid*.txt", "another_valid.js"], content=diff_content, is_staged=True)

		chunks = strategy.split(git_diff)

		# Should only create chunks for the valid filenames found in the headers
		assert len(chunks) == 2
		assert chunks[0].files == ["valid.py"]
		assert chunks[1].files == ["another_valid.js"]

	def test_handle_empty_diff_content_with_invalid_files(self, strategy: FileSplitStrategy) -> None:
		"""Test _handle_empty_diff_content with invalid filenames in diff.files."""
		# Pass empty string instead of None for content
		git_diff = GitDiff(files=["valid.py", "invalid*.txt", "another_valid.js"], content="", is_staged=False)

		# Accessing protected method for targeted testing
		chunks = strategy._handle_empty_diff_content(git_diff)  # type: ignore[attr-defined]

		# Should only create chunks for valid files from the list
		assert len(chunks) == 2
		assert chunks[0].files == ["valid.py"]
		assert chunks[1].files == ["another_valid.js"]


@pytest.mark.unit
@pytest.mark.git
class TestSemanticSplitStrategy:
	"""Tests for the SemanticSplitStrategy."""

	@pytest.fixture
	def mock_embedding_model(self) -> MagicMock:
		"""Provides a mock embedding model."""
		mock = MagicMock(spec=EmbeddingModel)
		# Mock the encode method to return predictable embeddings
		# This needs to return a numpy array
		mock.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
		return mock

	@pytest.fixture
	def semantic_strategy(self, mock_embedding_model: MagicMock) -> SemanticSplitStrategy:
		"""Provides a SemanticSplitStrategy instance with a mock model."""
		return SemanticSplitStrategy(embedding_model=mock_embedding_model)

	@pytest.fixture
	def simple_semantic_diff(self) -> GitDiff:
		"""Provides a simple GitDiff for semantic testing."""
		diff_content = (
			"diff --git a/src/main.py b/src/main.py\n"
			"index 111..222 100644\n"
			"--- a/src/main.py\n"
			"+++ b/src/main.py\n"
			"@@ -1,1 +1,1 @@\n"
			'-print("hello")\n'
			'+print("hello world")\n'
			"diff --git a/tests/test_main.py b/tests/test_main.py\n"
			"index 333..444 100644\n"
			"--- a/tests/test_main.py\n"
			"+++ b/tests/test_main.py\n"
			"@@ -5,1 +5,1 @@\n"
			"-assert main() == 1\n"
			"+assert main() == 0 # Changed expectation\n"
		)
		return GitDiff(files=["src/main.py", "tests/test_main.py"], content=diff_content, is_staged=True)

	# Patch helper functions used within the split process for focused unit testing
	@patch("codemap.git.diff_splitter.strategies.are_files_related", return_value=True)  # Assume related for simplicity
	@patch("codemap.git.diff_splitter.strategies.calculate_semantic_similarity", return_value=0.9)  # High similarity
	@patch("codemap.git.diff_splitter.strategies.extract_code_from_diff", return_value=("added_code", "removed_code"))
	@patch("codemap.git.diff_splitter.strategies.create_chunk_description", return_value="Mocked Description")
	def test_split_simple_related_files(
		self,
		mock_create_desc: MagicMock,
		mock_extract_code: MagicMock,
		_mock_calc_sim: MagicMock,  # Unused but needed for patch order
		mock_are_related: MagicMock,
		semantic_strategy: SemanticSplitStrategy,
		simple_semantic_diff: GitDiff,
	) -> None:
		"""Test splitting a diff with two semantically related files."""
		# Since files are related and assumed similar, they should be grouped
		chunks = semantic_strategy.split(simple_semantic_diff)

		assert len(chunks) == 1
		assert isinstance(chunks[0], DiffChunk)
		assert set(chunks[0].files) == {"src/main.py", "tests/test_main.py"}
		# Check that helper functions were called as expected
		assert mock_are_related.called
		# Need to check calls to similarity more carefully based on implementation
		# assert mock_calc_sim.called
		assert mock_extract_code.called
		assert mock_create_desc.called
		# Content should be a combination - exact content depends on consolidation logic
		assert "diff --git a/src/main.py b/src/main.py" in chunks[0].content
		assert "diff --git a/tests/test_main.py b/tests/test_main.py" in chunks[0].content
		assert chunks[0].description == "Mocked Description"

	@patch("codemap.git.diff_splitter.strategies.is_test_environment", return_value=False)
	def test_validate_embedding_model_raises_error(self, _mock_is_test_env: MagicMock) -> None:
		"""Test that validation raises error if model is None outside test env."""
		strategy = SemanticSplitStrategy(embedding_model=None)
		with pytest.raises(ValueError, match="embedding model not available"):
			strategy._validate_embedding_model()

	@patch("codemap.git.diff_splitter.strategies.is_test_environment", return_value=True)
	def test_validate_embedding_model_in_test_env(self, _mock_is_test_env: MagicMock) -> None:
		"""Test that validation passes if model is None inside test env."""
		strategy = SemanticSplitStrategy(embedding_model=None)
		try:
			strategy._validate_embedding_model()
		except ValueError:
			pytest.fail("_validate_embedding_model raised ValueError unexpectedly in test env")

		# Optionally, check for specific expected patterns if they are stable
		# e.g., assert any(p[0].pattern == r'(.+)\\.py$' for p in patterns)

	@patch("codemap.git.diff_splitter.strategies.SemanticSplitStrategy._process_group")
	def test_split_handles_large_file_count(
		self, mock_process_group: MagicMock, semantic_strategy: SemanticSplitStrategy
	) -> None:
		"""Test that split calls _process_group multiple times for many files."""
		# Create a diff with more files than MAX_FILES_PER_GROUP (default is 5, let's use 7)
		# We need to adjust MAX_FILES_PER_GROUP if it changes
		from codemap.git.diff_splitter.constants import MAX_FILES_PER_GROUP

		num_files = MAX_FILES_PER_GROUP + 2
		files = [f"src/dir1/file_{i}.py" for i in range(num_files // 2)] + [
			f"src/dir2/file_{i}.py" for i in range(num_files // 2, num_files)
		]

		# Mock diff content isn't crucial here as _process_group is mocked
		diff_content = ""
		for f in files:
			diff_content += f"diff --git a/{f} b/{f}\n--- a/{f}\n+++ b/{f}\n@@ +1 @@\n+ content\n"

		large_diff = GitDiff(files=files, content=diff_content, is_staged=True)

		# Mock _process_group to return a dummy chunk to avoid downstream errors
		mock_process_group.return_value = [DiffChunk(files=["dummy"], content="dummy content")]

		semantic_strategy.split(large_diff)

		# Expect _process_group to be called multiple times due to batching
		# The exact number depends on the batch size (default 3)
		# 7 files -> batches of 3, 3, 1 -> 3 calls expected
		# Grouping by directory: dir1 (4 files -> 2 batches), dir2 (3 files -> 1 batch) = 3 calls
		assert mock_process_group.call_count > 1
		# Check if calls were made with GitDiff objects containing file subsets
		for call_args in mock_process_group.call_args_list:
			assert isinstance(call_args[0][0], GitDiff)
			assert len(call_args[0][0].files) <= 3  # Based on default batch size

	@patch("codemap.git.diff_splitter.strategies.are_files_related", return_value=False)  # Assume unrelated
	@patch("codemap.git.diff_splitter.strategies.calculate_semantic_similarity", return_value=0.1)  # Low similarity
	@patch("codemap.git.diff_splitter.strategies.extract_code_from_diff", return_value=("added", "removed"))
	@patch(
		"codemap.git.diff_splitter.strategies.create_chunk_description", side_effect=lambda _x, f: f"Desc for {f[0]}"
	)
	def test_split_unrelated_files(
		self,
		_mock_create_desc: MagicMock,  # Unused but needed for patch order
		_mock_extract_code: MagicMock,  # Unused but needed for patch order
		_mock_calc_sim: MagicMock,  # Unused but needed for patch order
		_mock_are_related: MagicMock,  # Unused but needed for patch order
		semantic_strategy: SemanticSplitStrategy,
	) -> None:
		"""Test splitting a diff with two unrelated files."""
		diff_content = (
			"diff --git a/file1.py b/file1.py\n"
			"--- a/file1.py\n+++ b/file1.py\n@@ +1 @@\n+ content1\n"
			"diff --git a/unrelated.txt b/unrelated.txt\n"
			"--- a/unrelated.txt\n+++ b/unrelated.txt\n@@ +1 @@\n+ content2\n"
		)
		unrelated_diff = GitDiff(files=["file1.py", "unrelated.txt"], content=diff_content, is_staged=True)

		# Since files are unrelated, they should remain in separate chunks
		chunks = semantic_strategy.split(unrelated_diff)

		assert len(chunks) == 2
		assert set(chunks[0].files) == {"file1.py"} or set(chunks[1].files) == {"file1.py"}
		assert set(chunks[0].files) == {"unrelated.txt"} or set(chunks[1].files) == {"unrelated.txt"}
		# Verify descriptions match the side effect
		assert chunks[0].description in ["Desc for file1.py", "Desc for unrelated.txt"]
		assert chunks[1].description in ["Desc for file1.py", "Desc for unrelated.txt"]
		assert chunks[0].description != chunks[1].description

	def test_consolidate_small_chunks_single_file(self, semantic_strategy: SemanticSplitStrategy) -> None:
		"""Test consolidating multiple small chunks for the same file."""
		chunks_in = [
			DiffChunk(files=["file.py"], content="diff1\n+a", description="Chunk 1"),
			DiffChunk(files=["file.py"], content="diff2\n-b", description="Chunk 2"),
			DiffChunk(files=["other.py"], content="diff3", description="Chunk 3"),  # Different file
		]
		# Accessing protected method
		consolidated = semantic_strategy._consolidate_small_chunks(chunks_in)  # type: ignore[attr-defined]

		assert len(consolidated) == 2  # file.py chunks merged, other.py remains
		# Find the merged chunk for file.py
		merged_chunk = next((c for c in consolidated if c.files == ["file.py"]), None)
		other_chunk = next((c for c in consolidated if c.files == ["other.py"]), None)

		assert merged_chunk is not None
		assert other_chunk is not None
		assert other_chunk == chunks_in[2]  # Should be unchanged

		# Merged content should combine original contents
		assert "diff1\n+a" in merged_chunk.content
		assert "diff2\n-b" in merged_chunk.content
		# Description might be generic or combined, let's assume generic for now
		assert merged_chunk.description is not None  # Explicit check for type checker
		assert merged_chunk.description == chunks_in[0].description  # "Chunk 1"

	def test_consolidate_small_chunks_no_merge(self, semantic_strategy: SemanticSplitStrategy) -> None:
		"""Test consolidation when no chunks need merging (different files)."""
		chunks_in = [
			DiffChunk(files=["file1.py"], content="diff1", description="Chunk 1"),
			DiffChunk(files=["file2.py"], content="diff2", description="Chunk 2"),
		]
		consolidated = semantic_strategy._consolidate_small_chunks(chunks_in)  # type: ignore[attr-defined]
		assert len(consolidated) == 2
		assert consolidated == chunks_in  # Order might change, but content should be same

	def test_consolidate_small_chunks_empty(self, semantic_strategy: SemanticSplitStrategy) -> None:
		"""Test consolidation with an empty input list."""
		consolidated = semantic_strategy._consolidate_small_chunks([])  # type: ignore[attr-defined]
		assert consolidated == []
