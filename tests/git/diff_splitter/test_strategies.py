"""Tests for diff splitting strategies."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from codemap.config import ConfigLoader
from codemap.git.diff_splitter.schemas import DiffChunk
from codemap.git.diff_splitter.strategies import (
	FileSplitStrategy,
	SemanticSplitStrategy,
)
from codemap.git.semantic_grouping.embedder import DiffEmbedder
from codemap.git.utils import GitDiff


@pytest.mark.unit
@pytest.mark.git
class TestFileSplitStrategy:
	"""Tests for the FileSplitStrategy."""

	@pytest.fixture
	def strategy(self) -> FileSplitStrategy:
		"""Provides a FileSplitStrategy instance."""
		return FileSplitStrategy()

	@pytest.mark.asyncio
	async def test_split_simple_diff(self, strategy: FileSplitStrategy) -> None:
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

		chunks = await strategy.split(git_diff)

		assert len(chunks) == 2
		assert isinstance(chunks[0], DiffChunk)
		assert chunks[0].files == ["file1.py"]
		assert chunks[0].content.startswith("diff --git a/file1.py b/file1.py")
		assert chunks[0].description == "Changes in file1.py"

		assert isinstance(chunks[1], DiffChunk)
		assert chunks[1].files == ["file2.txt"]
		assert chunks[1].content.startswith("diff --git a/file2.txt b/file2.txt")
		assert chunks[1].description == "Changes in file2.txt"

	@pytest.mark.asyncio
	async def test_split_empty_diff_content(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting when diff content is empty."""
		# Pass empty string instead of None for content
		git_diff_none_content = GitDiff(files=["file1.py"], content="", is_staged=True)
		git_diff_empty_str = GitDiff(files=["file1.py"], content="", is_staged=True)

		chunks_none = await strategy.split(git_diff_none_content)
		chunks_empty_str = await strategy.split(git_diff_empty_str)

		assert chunks_none == []
		assert chunks_empty_str == []

	@pytest.mark.asyncio
	async def test_split_untracked_files(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting untracked files (empty content, files listed)."""
		# Should only happen for non-staged diffs
		git_diff = GitDiff(files=["new_file.py", "another.txt"], content="", is_staged=False, is_untracked=True)

		chunks = await strategy.split(git_diff)

		assert len(chunks) == 2
		assert chunks[0].files == ["new_file.py"]
		assert chunks[0].content == ""
		assert chunks[0].description == "New file: new_file.py"
		assert chunks[1].files == ["another.txt"]
		assert chunks[1].content == ""
		assert chunks[1].description == "New file: another.txt"

	@pytest.mark.asyncio
	async def test_split_untracked_files_staged(self, strategy: FileSplitStrategy) -> None:
		"""Test splitting staged diff with empty content (should yield no chunks)."""
		git_diff = GitDiff(files=["new_file.py"], content="", is_staged=True, is_untracked=False)
		chunks = await strategy.split(git_diff)
		assert chunks == []

	@pytest.mark.parametrize(
		("filename", "expected"),
		[
			("valid_file.py", True),
			("path/to/file.txt", True),
			("file-with-hyphens.js", True),
			("_private_file.ts", True),
			("", False),  # Empty string
			("file*.py", False),  # Contains *
			("file+.py", False),  # Contains +
			("file{}.py", False),  # Contains {}
			("file\\.py", False),  # Contains \
			('"quoted file"', False),  # Starts with quote
			("a/b/c", True),
		],
	)
	def test_is_valid_filename(self, strategy: FileSplitStrategy, filename: str, expected: bool) -> None:
		"""Test the _is_valid_filename helper method."""
		# Accessing protected member for testing specific helper
		assert strategy._is_valid_filename(filename) == expected

	@pytest.mark.asyncio
	async def test_split_diff_with_invalid_filenames_in_content(self, strategy: FileSplitStrategy) -> None:
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

		chunks = await strategy.split(git_diff)

		# Should only create chunks for the valid filenames found in the headers
		assert len(chunks) == 2
		assert chunks[0].files == ["valid.py"]
		assert chunks[1].files == ["another_valid.js"]

	def test_handle_empty_diff_content_with_invalid_files(self, strategy: FileSplitStrategy) -> None:
		"""Test _handle_empty_diff_content with invalid filenames in diff.files."""
		# Pass empty string instead of None for content
		git_diff = GitDiff(files=["valid.py", "invalid*.txt", "another_valid.js"], content="", is_staged=False)

		# Accessing protected method for targeted testing
		chunks = strategy._handle_empty_diff_content(git_diff)

		# Should only create chunks for valid files from the list
		assert len(chunks) == 2
		assert chunks[0].files == ["valid.py"]
		assert chunks[1].files == ["another_valid.js"]


@pytest.mark.unit
@pytest.mark.git
class TestSemanticSplitStrategy:
	"""Tests for the SemanticSplitStrategy."""

	@pytest.fixture
	def mock_embedder(self) -> AsyncMock:
		"""Provides a mock DiffEmbedder."""
		mock = AsyncMock(spec=DiffEmbedder)
		# Mock the encode method to return predictable embeddings
		mock.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
		return mock

	@pytest.fixture
	def mock_config(self) -> Mock:
		"""Provides a mock ConfigLoader with required settings."""
		mock_config = Mock(spec=ConfigLoader)
		mock_config.get = MagicMock()

		# Mock required configuration values
		mock_config.get.commit = MagicMock()
		mock_config.get.commit.diff_splitter = MagicMock()
		mock_config.get.commit.diff_splitter.similarity_threshold = 0.6
		mock_config.get.commit.diff_splitter.directory_similarity_threshold = 0.3
		mock_config.get.commit.diff_splitter.min_chunks_for_consolidation = 2
		mock_config.get.commit.diff_splitter.max_chunks_before_consolidation = 20
		mock_config.get.commit.diff_splitter.max_file_size_for_llm = 50000
		mock_config.get.commit.diff_splitter.max_log_diff_size = 1000
		mock_config.get.commit.diff_splitter.file_move_similarity_threshold = 0.85
		mock_config.get.commit.diff_splitter.default_code_extensions = [
			"js",
			"jsx",
			"ts",
			"tsx",
			"py",
			"java",
			"c",
			"cpp",
		]

		mock_config.get.repo_root = Path("/mock/repo")

		return mock_config

	@pytest.fixture
	def semantic_strategy(self, mock_config: Mock) -> SemanticSplitStrategy:
		"""Provides a SemanticSplitStrategy instance with a mock config."""
		with patch("codemap.git.diff_splitter.strategies.DiffEmbedder") as mock_embedder_cls:
			mock_embedder = AsyncMock()
			mock_embedder_cls.return_value = mock_embedder

			# Mock the encode method
			mock_embedder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

			return SemanticSplitStrategy(config_loader=mock_config)

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

	@pytest.mark.asyncio
	@patch("codemap.git.diff_splitter.strategies.are_files_related", return_value=True)  # Assume related for simplicity
	@patch("codemap.git.diff_splitter.strategies.calculate_semantic_similarity", return_value=0.9)  # High similarity
	@patch("codemap.git.diff_splitter.strategies.create_chunk_description", return_value="Mocked Description")
	async def test_split_simple_related_files(
		self,
		_mock_create_desc: MagicMock,
		_mock_calc_sim: MagicMock,  # Unused but needed for patch order
		_mock_are_related: MagicMock,
		semantic_strategy: SemanticSplitStrategy,
		simple_semantic_diff: GitDiff,
	) -> None:
		"""Test splitting a diff with two semantically related files."""
		# Since files are related and assumed similar, they should be grouped
		chunks = await semantic_strategy.split(simple_semantic_diff)

		# With the refactored implementation, we now get one chunk per file
		# instead of consolidated chunks as before
		assert len(chunks) == 2
		assert isinstance(chunks[0], DiffChunk)
		assert isinstance(chunks[1], DiffChunk)

		# Verify that each file is in a separate chunk
		file_names = {chunk.files[0] for chunk in chunks}
		assert file_names == {"src/main.py", "tests/test_main.py"}

		# Verify content from each file is present
		for chunk in chunks:
			if chunk.files[0] == "src/main.py":
				assert 'print("hello world")' in chunk.content
			elif chunk.files[0] == "tests/test_main.py":
				assert "assert main() == 0" in chunk.content

		# Each chunk should have a description
		for chunk in chunks:
			assert chunk.description is not None

	@pytest.mark.asyncio
	@patch("codemap.git.diff_splitter.strategies.are_files_related", return_value=False)  # Assume unrelated
	@patch("codemap.git.diff_splitter.strategies.calculate_semantic_similarity", return_value=0.1)  # Low similarity
	@patch(
		"codemap.git.diff_splitter.strategies.create_chunk_description", side_effect=lambda _x, f: f"Desc for {f[0]}"
	)
	async def test_split_unrelated_files(
		self,
		_mock_create_desc: MagicMock,  # Unused but needed for patch order
		_mock_calc_sim: MagicMock,  # Unused but needed for patch order
		_mock_are_related: MagicMock,  # Unused but needed for patch order
		semantic_strategy: SemanticSplitStrategy,
	) -> None:
		"""Test splitting a diff with semantically unrelated files."""
		# Create a diff with files that shouldn't be semantically related
		unrelated_diff_content = (
			"diff --git a/src/backend/api.py b/src/backend/api.py\n"
			"index 111..222 100644\n"
			"--- a/src/backend/api.py\n"
			"+++ b/src/backend/api.py\n"
			"@@ -1,1 +1,1 @@\n"
			"-def get_data(): return {}\n"
			"+def get_data(): return {'status': 'ok'}\n"
			"diff --git a/src/frontend/styles.css b/src/frontend/styles.css\n"
			"index 333..444 100644\n"
			"--- a/src/frontend/styles.css\n"
			"+++ b/src/frontend/styles.css\n"
			"@@ -5,1 +5,1 @@\n"
			"-body { color: red; }\n"
			"+body { color: blue; }\n"
		)
		unrelated_diff = GitDiff(
			files=["src/backend/api.py", "src/frontend/styles.css"],
			content=unrelated_diff_content,
			is_staged=True,
		)

		chunks = await semantic_strategy.split(unrelated_diff)

		# Should get a chunk for each file since they're not similar
		assert len(chunks) == 2
		files_in_chunks = {file for chunk in chunks for file in chunk.files}
		assert files_in_chunks == {"src/backend/api.py", "src/frontend/styles.css"}

	@pytest.mark.asyncio
	async def test_consolidate_small_chunks_single_file(self, semantic_strategy: SemanticSplitStrategy) -> None:
		"""Test consolidation of small chunks."""
		# Set up test chunks - three single-file chunks
		chunks = [
			DiffChunk(
				files=["file1.py"],
				content="diff --git a/file1.py b/file1.py\n...",
				description="Changes in file1.py",
			),
			DiffChunk(
				files=["file2.py"],
				content="diff --git a/file2.py b/file2.py\n...",
				description="Changes in file2.py",
			),
			DiffChunk(
				files=["file3.py"],
				content="diff --git a/file3.py b/file3.py\n...",
				description="Changes in file3.py",
			),
		]

		# Mock the _should_merge_chunks method to return True only for file1 and file2
		with patch.object(
			semantic_strategy,
			"_should_merge_chunks",
			side_effect=lambda c1, c2: "file3.py" not in c1.files and "file3.py" not in c2.files,
		):
			result = await semantic_strategy._consolidate_small_chunks(chunks)

		# Should consolidate file1.py and file2.py, but leave file3.py separate
		assert len(result) == 2

		# Find which chunk has merged files
		merged_chunk = next((c for c in result if len(c.files) > 1), None)
		single_chunk = next((c for c in result if len(c.files) == 1), None)

		assert merged_chunk is not None
		assert single_chunk is not None

		# Verify merged chunk has file1 and file2
		assert set(merged_chunk.files) == {"file1.py", "file2.py"}

		# Verify single chunk has file3
		assert single_chunk.files == ["file3.py"]

		# Verify content is present
		assert "diff --git a/file1.py b/file1.py" in merged_chunk.content
		assert "diff --git a/file2.py b/file2.py" in merged_chunk.content
		assert "diff --git a/file3.py b/file3.py" in single_chunk.content

	@pytest.mark.asyncio
	async def test_consolidate_small_chunks_no_merge(self, semantic_strategy: SemanticSplitStrategy) -> None:
		"""Test that chunks are not consolidated when not required."""
		# Set up test chunks - mock two chunks that should not be consolidated
		chunks = [
			DiffChunk(
				files=["bigfile1.py"],
				content="diff --git a/bigfile1.py b/bigfile1.py\n" + "x" * 1000,
				description="Large chunk 1",
			),
			DiffChunk(
				files=["bigfile2.py"],
				content="diff --git a/bigfile2.py b/bigfile2.py\n" + "y" * 1000,
				description="Large chunk 2",
			),
		]

		# Mock the _should_merge_chunks method to always return False
		with patch.object(semantic_strategy, "_should_merge_chunks", return_value=False):
			result = await semantic_strategy._consolidate_small_chunks(chunks)

		# Should not be consolidated
		assert len(result) == 2
		assert result[0].files == ["bigfile1.py"]
		assert result[1].files == ["bigfile2.py"]

	@pytest.mark.asyncio
	async def test_consolidate_small_chunks_empty(self, semantic_strategy: SemanticSplitStrategy) -> None:
		"""Test consolidating empty list of chunks."""
		result = await semantic_strategy._consolidate_small_chunks([])
		assert result == []
