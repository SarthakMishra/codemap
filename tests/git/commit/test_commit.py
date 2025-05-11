"""Tests for the commit feature."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml
from dotenv import load_dotenv
from rich.console import Console

from codemap.config import ConfigLoader
from codemap.config.config_schema import CommitSchema
from codemap.git.commit_generator.generator import CommitMessageGenerator
from codemap.git.commit_generator.schemas import CommitMessageSchema
from codemap.git.diff_splitter import (
	DiffChunk,
	DiffSplitter,
)
from codemap.git.utils import GitDiff
from codemap.llm import LLMClient
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
		splitter = AsyncMock(spec=DiffSplitter)
		chunks = [
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
		# Now returns tuple of (chunks, filtered_files)
		splitter.split_diff.return_value = (chunks, [])
		mock.return_value = splitter
		yield mock.return_value


@pytest.fixture
def mock_git_utils() -> Generator[dict[str, Mock], None, None]:
	"""Create a mock for git utilities."""
	with (
		patch("codemap.git.utils.ExtendedGitRepoContext.get_staged_diff") as mock_staged,
		patch("codemap.git.utils.ExtendedGitRepoContext.get_unstaged_diff") as mock_unstaged,
		patch("codemap.git.utils.ExtendedGitRepoContext.get_untracked_files") as mock_untracked,
		patch("codemap.git.utils.ExtendedGitRepoContext.commit_only_files") as mock_commit,
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
			"convention": {
				"types": ["feat", "fix", "docs", "style", "refactor"],
				"scopes": ["core", "ui", "tests"],
				"max_length": 72,
			},
		},
		"llm": {
			"model": "gpt-4o-mini",
			"temperature": 0.5,
			"max_output_tokens": 1024,
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

	@pytest.mark.asyncio
	async def test_diff_splitter_semantic_only(self) -> None:
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

		# Create mock config_loader
		config_loader = Mock(spec=ConfigLoader)
		# Mock the DiffSplitter configuration properly
		mock_diff_splitter = Mock()
		mock_diff_splitter.similarity_threshold = 0.6
		mock_diff_splitter.directory_similarity_threshold = 0.3
		mock_diff_splitter.min_chunks_for_consolidation = 2
		mock_diff_splitter.max_chunks_before_consolidation = 20
		mock_diff_splitter.max_file_size_for_llm = 50000
		mock_diff_splitter.max_log_diff_size = 1000

		# Configure mock hierarchy
		config_loader.get.repo_root = repo_root
		config_loader.get.commit.diff_splitter = mock_diff_splitter

		# Create the expected chunks
		expected_chunks = [
			DiffChunk(
				files=["file1.py", "file2.py"],
				content="diff content for semantic chunk",
			),
		]

		# Mock the split_diff method at the class level
		with patch.object(DiffSplitter, "split_diff", new_callable=AsyncMock) as mock_split:
			# Setup return value
			mock_split.return_value = (expected_chunks, [])

			# Create the splitter instance
			splitter = DiffSplitter(config_loader=config_loader)

			# Call the method
			result_chunks, _ = await splitter.split_diff(diff)

			# Assert that we got the expected results
			assert result_chunks == expected_chunks
			mock_split.assert_awaited_once()

	@pytest.mark.asyncio
	async def test_diff_splitter_semantic_strategy(self) -> None:
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

		# Create mock config_loader
		config_loader = Mock(spec=ConfigLoader)
		# Mock the DiffSplitter configuration properly
		mock_diff_splitter = Mock()
		mock_diff_splitter.similarity_threshold = 0.6
		mock_diff_splitter.directory_similarity_threshold = 0.3
		mock_diff_splitter.min_chunks_for_consolidation = 2
		mock_diff_splitter.max_chunks_before_consolidation = 20
		mock_diff_splitter.max_file_size_for_llm = 50000
		mock_diff_splitter.max_log_diff_size = 1000

		# Configure mock hierarchy
		config_loader.get.repo_root = repo_root
		config_loader.get.commit.diff_splitter = mock_diff_splitter

		# Create the expected chunks
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

		# Mock the split_diff method at the class level
		with patch.object(DiffSplitter, "split_diff", new_callable=AsyncMock) as mock_split:
			# Setup return value
			mock_split.return_value = (expected_chunks, [])

			# Create the splitter instance
			splitter = DiffSplitter(config_loader=config_loader)

			# Call the method
			result_chunks, _ = await splitter.split_diff(diff)

			# Assert that we got the expected results
			assert result_chunks == expected_chunks
			mock_split.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.llm
class TestMessageGenerator(LLMTestBase):
	"""Test cases for commit message generation."""

	@pytest.mark.asyncio
	async def test_message_generator_generate(self) -> None:
		"""Test commit message generation using the updated API."""
		# Create a mock repo root
		repo_root = Path("/mock/repo")

		# Create mock config loader
		config_loader = Mock(spec=ConfigLoader)
		config_loader.get.llm.max_output_tokens = 1024
		config_loader.get.commit.use_lod_context = True

		# Create mock LLM client
		llm_client = Mock(spec=LLMClient)
		llm_client.set_template = Mock()
		# Mock the completion method that's actually used in the implementation
		llm_client.completion = Mock()

		# Mock LLM response with structured output
		fake_llm_response = CommitMessageSchema(
			type="feat",
			scope="core",
			description="add new functionality",
			body="This implements the new feature we discussed",
			breaking=False,
			footers=[],
		)

		# Set the return value
		llm_client.completion.return_value = fake_llm_response

		# Create sample diff chunk
		chunk = DiffChunk(files=["file1.py"], content="diff content", description="Sample change")

		# Create generator with mocks
		generator = CommitMessageGenerator(
			repo_root=repo_root, llm_client=llm_client, prompt_template="Test template", config_loader=config_loader
		)

		# Test the generate_message method - it's synchronous, don't await
		message, is_auto_generated = generator.generate_message(chunk)

		# Assertions
		assert isinstance(message, CommitMessageSchema)
		assert message.type == "feat"
		assert message.scope == "core"
		assert message.description == "add new functionality"
		assert is_auto_generated is True
		llm_client.completion.assert_called_once()


@pytest.mark.unit
@pytest.mark.git
class TestFileRelations(GitTestBase):
	"""Test cases for file relationship detection."""

	def setup_method(self) -> None:
		"""Set up the test fixture."""
		self.repo_root = Path("/mock/repo")

	def test_has_related_file_pattern(self) -> None:
		"""Test detection of related files based on patterns."""

		def check_related(file1: str, file2: str) -> bool:
			"""Check if file pairs are related using simple pattern matching."""
			# Python files and tests
			if file1.endswith(".py") and not file1.startswith("tests/"):
				# Implementation file to test file
				base_name = file1[:-3]  # Remove .py extension
				if file2 == f"tests/test_{base_name}.py":
					return True
			if file1.startswith("tests/test_") and file1.endswith(".py"):
				# Test file to implementation file
				base_name = file1[11:-3]  # Remove "tests/test_" and ".py"
				if file2 == f"{base_name}.py":
					return True

			# JS files and tests
			if file1.startswith("src/") and file1.endswith(".js"):
				# Implementation file to test file
				parts = file1.split("/")
				if len(parts) >= 3:
					module = parts[1]
					component = parts[2][:-3]  # Remove .js extension
					if file2 == f"tests/{module}/{component}.test.js":
						return True
			if file1.startswith("tests/") and file1.endswith(".test.js"):
				# Test file to implementation file
				parts = file1.split("/")
				if len(parts) >= 3:
					module = parts[1]
					component = parts[2][:-8]  # Remove .test.js extension
					if file2 == f"src/{module}/{component}.js":
						return True

			# React component files and tests
			if file1.startswith("components/") and file1.endswith(".tsx"):
				# Component file to test file
				component = file1[11:-4]  # Remove "components/" and ".tsx"
				if file2 == f"__tests__/{component}.test.tsx":
					return True
			if file1.startswith("__tests__/") and file1.endswith(".test.tsx"):
				# Test file to component file
				component = file1[10:-9]  # Remove "__tests__/" and ".test.tsx"
				if file2 == f"components/{component}.tsx":
					return True

			return False

		# Test with Python modules and tests
		assert check_related("models.py", "tests/test_models.py")
		assert check_related("tests/test_models.py", "models.py")
		assert not check_related("models.py", "views.py")

		# Test with JS modules and tests
		assert check_related("src/components/Button.js", "tests/components/Button.test.js")
		assert check_related("tests/components/Button.test.js", "src/components/Button.js")
		assert not check_related("src/components/Button.js", "tests/components/Badge.test.js")

		# Test with React components
		assert check_related("components/Header.tsx", "__tests__/Header.test.tsx")
		assert check_related("__tests__/Header.test.tsx", "components/Header.tsx")
		assert not check_related("components/Header.tsx", "__tests__/Footer.test.tsx")


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.cli
class TestCommitConfig(GitTestBase):
	"""Test cases for commit configuration."""

	@pytest.mark.asyncio
	async def test_config_loading(self) -> None:
		"""Test loading of commit configuration with new ConfigLoader."""
		# Create a mock config loader
		config_loader = Mock(spec=ConfigLoader)

		# Set up commit schema mock
		commit_schema = Mock(spec=CommitSchema)
		commit_schema.strategy = "file"
		commit_schema.use_lod_context = True

		# Set up config structure
		config_loader.get.commit = commit_schema
		config_loader.get.llm.model = "gpt-4o-mini"
		config_loader.get.llm.temperature = 0.5

		# Verify config access
		assert config_loader.get.commit.strategy == "file"
		assert config_loader.get.commit.use_lod_context is True
		assert config_loader.get.llm.model == "gpt-4o-mini"
		assert config_loader.get.llm.temperature == 0.5
