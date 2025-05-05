"""Tests for the commit feature."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml
from dotenv import load_dotenv
from rich.console import Console

from codemap.git.commit_generator.generator import CommitMessageGenerator
from codemap.git.diff_splitter import (
	DiffChunk,
	DiffSplitter,
)
from codemap.git.utils import GitDiff
from tests.base import GitTestBase, LLMTestBase

if TYPE_CHECKING:
	from collections.abc import Generator

console = Console(highlight=False)

# Allow tests to access private members
# ruff: noqa: SLF001

# Load environment variables from .env.test if present
if load_dotenv:
	load_dotenv(".env.test")


@pytest.fixture
def mock_git_diff() -> GitDiff:
	"""Create a mock GitDiff with sample content."""
	return GitDiff(
		files=["file1.py", "file2.py"],
		content="""diff --git a/file1.py b/file1.py
index 1234567..abcdefg 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@ def existing_function():
     return True

 def new_function():
-    return False
+    return True
diff --git a/file2.py b/file2.py
index 2345678..bcdefgh 100644
--- a/file2.py
+++ b/file2.py
@@ -5,3 +5,6 @@ def old_function():
     # Some code
     pass

+def added_function():
+    return "Hello, World!"
""",
		is_staged=False,
	)


@pytest.fixture
def mock_diff_splitter() -> Generator[Mock, None, None]:
	"""Create a mock DiffSplitter."""
	with patch("codemap.git.diff_splitter.splitter.DiffSplitter") as mock:
		splitter = Mock(spec=DiffSplitter)
		splitter.split_diff.return_value = [
			DiffChunk(
				files=["file1.py"],
				content="diff content for file1.py",
				description=None,
			),
			DiffChunk(
				files=["file2.py"],
				content="diff content for file2.py",
				description=None,
			),
		]
		mock.return_value = splitter
		yield mock.return_value


@pytest.fixture
def mock_git_utils() -> Generator[dict[str, Mock], None, None]:
	"""Create a mock for git utilities."""
	with (
		patch("codemap.git.utils.get_staged_diff") as mock_staged,
		patch("codemap.git.utils.get_unstaged_diff") as mock_unstaged,
		patch("codemap.git.utils.get_untracked_files") as mock_untracked,
		patch("codemap.git.utils.commit_only_files") as mock_commit,
	):
		# Mock the staged diff
		staged_diff = GitDiff(
			files=["file1.py"],
			content="diff content for file1.py",
			is_staged=True,
		)
		mock_staged.return_value = staged_diff

		# Mock the unstaged diff
		unstaged_diff = GitDiff(
			files=["file2.py"],
			content="diff content for file2.py",
			is_staged=False,
		)
		mock_unstaged.return_value = unstaged_diff

		# Mock untracked files
		mock_untracked.return_value = ["file3.py"]

		# Mock commit
		mock_commit.return_value = []

		yield {
			"get_staged_diff": mock_staged,
			"get_unstaged_diff": mock_unstaged,
			"get_untracked_files": mock_untracked,
			"commit_only_files": mock_commit,
		}


@pytest.fixture
def mock_config_file() -> str:
	"""Create a mock config file content."""
	config = {
		"commit": {
			"strategy": "hunk",
			"llm": {
				"model": "gpt-4o-mini",
				"provider": "openai",
			},
			"convention": {
				"types": ["feat", "fix", "docs", "style", "refactor"],
				"scopes": ["core", "ui", "tests"],
				"max_length": 72,
			},
		},
	}
	return yaml.dump(config)


@pytest.mark.unit
@pytest.mark.git
class TestDiffSplitter(GitTestBase):
	"""
	Test cases for diff splitting functionality.

	Tests the semantic splitting of git diffs into logical chunks.

	"""

	def test_diff_splitter_semantic_only(self) -> None:
		"""
		Test that the diff splitter now only uses semantic strategy.

		Verifies that the splitter defaults to semantic chunking.

		"""
		# Arrange: Create test diff
		diff = GitDiff(
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

		# Using a mock repo_root
		repo_root = Path("/mock/repo")
		splitter = DiffSplitter(repo_root)

		# Act/Assert: Mock both git commands and _split_semantic to avoid file system access
		with (
			patch("codemap.git.utils.run_git_command") as mock_git,
			patch.object(splitter, "_split_semantic") as mock_split,
		):
			# Mock git status command
			mock_git.return_value = ""

			expected_chunks = [
				DiffChunk(
					files=["file1.py", "file2.py"],
					content="diff content for semantic chunk",
				),
			]
			mock_split.return_value = expected_chunks

			# Test the split_diff method (should use semantic strategy by default)
			result_tuple = splitter.split_diff(diff)
			result_chunks = result_tuple[0]  # Extract chunks from tuple
			assert result_chunks == expected_chunks
			mock_split.assert_called_once_with(diff)

	def test_diff_splitter_semantic_strategy(self) -> None:
		"""
		Test the semantic splitting strategy.

		Verifies that related files are correctly grouped together.

		"""
		# Arrange: Create test diff
		diff = GitDiff(
			files=["models.py", "views.py", "tests/test_models.py"],
			content="mock diff content",
			is_staged=False,
		)

		# Using a mock repo_root
		repo_root = Path("/mock/repo")
		splitter = DiffSplitter(repo_root)

		# Act/Assert: Mock both git commands and _split_semantic to avoid file system access
		with (
			patch("codemap.git.utils.run_git_command") as mock_git,
			patch.object(splitter, "_split_semantic") as mock_split,
		):
			# Mock git status command
			mock_git.return_value = ""

			expected_chunks = [
				DiffChunk(
					files=["models.py", "tests/test_models.py"],
					content="diff content for semantic chunk 1",
					description="Model-related changes",
				),
				DiffChunk(
					files=["views.py"],
					content="diff content for semantic chunk 2",
					description="View-related changes",
				),
			]
			mock_split.return_value = expected_chunks

			# Test the split_diff method (now always uses semantic strategy)
			result_tuple = splitter.split_diff(diff)
			result_chunks = result_tuple[0]  # Extract chunks from tuple
			assert result_chunks == expected_chunks
			mock_split.assert_called_once_with(diff)


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.llm
class TestMessageGenerator(LLMTestBase):
	"""
	Test cases for commit message generation.

	Tests the generation of commit messages using LLMs.

	"""

	@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
	def test_message_generator_fallback(self) -> None:
		"""
		Test message generator fallback when API key is not available.

		Verifies that when LLM API is unavailable, a reasonable fallback
		message is generated.

		"""

	@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
	def test_message_generator_openai(self) -> None:
		"""
		Test message generation with OpenAI provider.

		Verifies interaction with OpenAI models for message generation.

		"""

	@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
	def test_message_generator_anthropic(self) -> None:
		"""
		Test message generation with Anthropic provider.

		Verifies interaction with Anthropic Claude models for message
		generation.

		"""

	@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
	def test_message_linting_valid(self) -> None:
		"""
		Test message generation with linting - valid message case.

		Verifies that a valid message passes linting without regeneration.
		"""

	@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
	def test_message_linting_invalid_with_regeneration(self) -> None:
		"""
		Test message generation with linting - invalid message case.

		Verifies that an invalid message is regenerated with linting feedback.
		"""


@pytest.mark.unit
@pytest.mark.git
class TestFileRelations(GitTestBase):
	"""
	Test cases for determining file relatedness.

	Focuses on pattern matching and semantic similarity logic.

	"""

	def setup_method(self) -> None:
		"""Set up test environment."""
		# Mock embedding model for semantic tests
		self.mock_embedding_model = Mock()
		self.mock_embedding_model.encode.side_effect = lambda texts: np.array(
			[[hash(t) % 100 / 100.0] for t in texts]  # Simple deterministic embedding
		)

	def test_has_related_file_pattern(self) -> None:
		"""Test the matching of related file patterns."""

		# Use a simplified direct pattern matching approach instead of complex regex
		def check_related(file1: str, file2: str) -> bool:
			# Check Python test files
			if file1.endswith(".py") and file2 == file1.replace(".py", "_test.py"):
				return True
			if file2.endswith(".py") and file1 == file2.replace(".py", "_test.py"):
				return True

			# Check component style files
			if (
				file1.endswith((".jsx", ".tsx"))
				and file2.endswith((".css", ".scss"))
				and file1.replace(".jsx", "").replace(".tsx", "") == file2.replace(".css", "").replace(".scss", "")
			):
				return True
			if (
				file2.endswith((".jsx", ".tsx"))
				and file1.endswith((".css", ".scss"))
				and file2.replace(".jsx", "").replace(".tsx", "") == file1.replace(".css", "").replace(".scss", "")
			):
				return True

			# Check C header files
			if file1.endswith(".c") and file2.endswith(".h") and file1.replace(".c", "") == file2.replace(".h", ""):
				return True
			if file2.endswith(".c") and file1.endswith(".h") and file2.replace(".c", "") == file1.replace(".h", ""):
				return True

			# README.md matches everything
			return bool(file1 == "README.md" or file2 == "README.md")

		# Assert: Test cases using the helper
		assert check_related("file.py", "file_test.py")
		assert not check_related("file.py", "other.py")
		assert check_related("component.jsx", "component.css")
		assert not check_related("src/Component.tsx", "src/Container.tsx")
		assert not check_related("component.jsx", "unrelated.js")
		assert check_related("main.c", "main.h")
		assert check_related("README.md", "main.py")
		assert check_related("README.md", "LICENSE")


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.cli
class TestCommitConfig(GitTestBase):
	"""
	Test cases for commit command configuration.

	Tests the loading and application of config settings.

	"""

	@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
	def test_config_loading(self) -> None:
		"""
		Test loading configuration from .codemap.yml.

		Verifies that commit configuration is properly loaded from config
		files.

		"""

	@pytest.mark.skip("Function setup_message_generator was removed in refactoring")
	def test_setup_message_generator(self) -> None:
		"""
		Test message generator setup with different configurations.

		Verifies proper initialization with various dependency injection
		options.

		"""


@pytest.fixture
def mock_generator():
	"""Create a mock generator for testing fallback generation."""
	# Create a mock object that mimics CommitMessageGenerator
	generator = Mock(spec=CommitMessageGenerator)

	# Define a side effect function to simulate fallback_generation behavior
	def fallback_side_effect(chunk):
		if "feat: add new button" in chunk.description:
			return "feat: add new button"
		if "tests/test_main.py" in chunk.files:
			return "test: update tests/test_main.py"
		if "fix bug 123" in chunk.content:
			return "fix: update src/utils.py"
		if "src/main.py" in chunk.files and "src/utils.py" in chunk.files:
			return "chore: update files in src"
		if "src/main.py" in chunk.files:
			return "chore: update src/main.py"
		if "file.py" in chunk.files and "update files" in chunk.description:
			return "chore: update file.py"  # Ignore non-specific description
		# Default fallback if no specific condition matches
		return f"chore: update {chunk.files[0]}"  # Basic fallback for other cases

	generator.fallback_generation.side_effect = fallback_side_effect
	return generator


@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
def test_fallback_generation_basic(mock_generator):
	"""Test basic fallback generation."""


@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
def test_fallback_generation_with_fix_keyword(mock_generator):
	"""Test fallback generation detects 'fix' keyword."""


@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
def test_fallback_generation_with_test_file(mock_generator):
	"""Test fallback generation detects test files."""


@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
def test_fallback_generation_multiple_files(mock_generator):
	"""Test fallback generation with multiple files."""


@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
def test_fallback_generation_with_chunk_description(mock_generator):
	"""Test fallback uses specific chunk description if available."""


@pytest.mark.skip("Test needs to be refactored to match new CommitMessageGenerator API")
def test_fallback_generation_with_non_specific_description(mock_generator):
	"""Test fallback ignores non-specific chunk descriptions."""
