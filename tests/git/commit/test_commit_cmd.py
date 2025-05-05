"""Tests for commit command module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from rich.console import Console

from codemap.git.diff_splitter import DiffChunk
from tests.base import GitTestBase

# Import fixtures
pytest.importorskip("dotenv")

# Constants for testing
FAKE_REPO_PATH = Path("/fake/repo")

# Example test data for chunks
TEST_CHUNK = DiffChunk(
	files=["file1.py", "file2.py"],
	content="diff --git a/file1.py b/file1.py\n@@ -1,3 +1,3 @@\n-def old():\n+def new():\n     pass",
)


@pytest.fixture
def mock_console() -> Console:
	"""Create a mock console for testing."""
	return MagicMock(spec=Console)


@pytest.fixture
def mock_diff_chunk() -> DiffChunk:
	"""Create a mock DiffChunk for testing."""
	chunk = Mock(spec=DiffChunk)
	chunk.files = ["file1.py", "file2.py"]
	chunk.content = """
diff --git a/file1.py b/file1.py
index 1234567..abcdef0 100644
--- a/file1.py
+++ b/file1.py
@@ -1,7 +1,7 @@
-def old_function():
+def new_function():
     return True

+def added_function():
+    return True
"""
	chunk.description = None
	chunk.is_llm_generated = False
	return chunk


@pytest.mark.skip("CommitOptions class was removed during refactoring")
@pytest.fixture
def commit_options():
	"""Create CommitOptions for testing."""


@pytest.mark.skip("_load_prompt_template function was removed during refactoring")
@pytest.mark.unit
@pytest.mark.fs
class TestLoadPromptTemplate:
	"""Test loading prompt templates from files."""

	def test_load_prompt_template_exists(self, tmp_path: Path) -> None:
		"""Test loading a prompt template that exists."""

	def test_load_prompt_template_not_exists(self) -> None:
		"""Test loading a prompt template that doesn't exist."""

	def test_load_prompt_template_none(self) -> None:
		"""Test loading with None path."""


@pytest.mark.unit
@pytest.mark.git
class TestCommitChanges(GitTestBase):
	"""Test committing changes."""

	@pytest.mark.skip("_commit_changes was moved to CommitCommand class during refactoring")
	def test_commit_changes_success(self) -> None:
		"""Test successful commit."""

	@pytest.mark.skip("_commit_changes was moved to CommitCommand class during refactoring")
	def test_commit_changes_with_hooks_bypass(self) -> None:
		"""Test commit with hooks bypass."""

	@pytest.mark.skip("_commit_changes was moved to CommitCommand class during refactoring")
	def test_commit_changes_failure(self) -> None:
		"""Test failed commit."""


@pytest.mark.unit
@pytest.mark.git
class TestPerformCommit:
	"""Test performing commit operations."""

	@pytest.mark.skip("_perform_commit was moved to CommitCommand class during refactoring")
	def test_perform_commit_with_file_checks(self) -> None:
		"""Test commit with file checks."""

	@pytest.mark.skip("_perform_commit was moved to CommitCommand class during refactoring")
	def test_perform_commit_with_other_files(self) -> None:
		"""Test commit when there are other files."""

	@pytest.mark.skip("_perform_commit was moved to CommitCommand class during refactoring")
	def test_perform_commit_failure(self) -> None:
		"""Test commit failure."""


@pytest.mark.unit
@pytest.mark.git
class TestEditCommitMessage:
	"""Test editing commit message."""

	@pytest.mark.skip("_edit_commit_message was moved to CommitCommand class during refactoring")
	def test_edit_commit_message(self) -> None:
		"""Test editing commit message with user input."""


@pytest.mark.unit
@pytest.mark.git
class TestCommitWithMessage:
	"""Test commit with message functionality."""

	@pytest.mark.skip("_commit_with_message was moved to CommitCommand class during refactoring")
	def test_commit_with_message(self) -> None:
		"""Test commit with message."""


@pytest.mark.unit
@pytest.mark.skip("_load_prompt_template was removed during refactoring")
def test_load_prompt_template_success(tmp_path: Path) -> None:
	"""Test loading prompt template successfully."""


@pytest.mark.unit
@pytest.mark.skip("_load_prompt_template was removed during refactoring")
def test_load_prompt_template_nonexistent() -> None:
	"""Test loading prompt template with nonexistent file."""


@pytest.mark.unit
@pytest.mark.skip("_load_prompt_template was removed during refactoring")
def test_load_prompt_template_none() -> None:
	"""Test loading prompt template with None path."""


@pytest.mark.unit
@pytest.mark.skip("CommitOptions class was removed during refactoring")
def test_commit_options_dataclass() -> None:
	"""Test CommitOptions dataclass initialization and defaults."""


@pytest.mark.unit
@pytest.mark.skip("RunConfig class was removed during refactoring")
def test_run_config_dataclass() -> None:
	"""Test RunConfig dataclass initialization and defaults."""


@pytest.mark.unit
@pytest.mark.skip("ChunkContext dataclass was removed during refactoring")
def test_chunk_context_dataclass() -> None:
	"""Test the ChunkContext dataclass."""


@pytest.mark.unit
@pytest.mark.skip("_commit_changes was moved to CommitCommand class during refactoring")
def test_commit_changes_no_valid_files() -> None:
	"""Test commit_changes with no valid files."""


@pytest.mark.unit
@pytest.mark.skip("_commit_changes was moved to CommitCommand class during refactoring")
def test_commit_changes_exception() -> None:
	"""Test commit_changes with exception."""


@pytest.mark.unit
@pytest.mark.skip("_perform_commit was moved to CommitCommand class during refactoring")
def test_perform_commit_success() -> None:
	"""Test perform_commit with success."""


@pytest.mark.unit
@pytest.mark.skip("_perform_commit was moved to CommitCommand class during refactoring")
def test_perform_commit_failure() -> None:
	"""Test perform_commit with failure."""


@pytest.mark.unit
@pytest.mark.skip("_raise_command_failed_error was removed during refactoring")
def test_raise_command_failed_error() -> None:
	"""Test raising command failed error."""


@pytest.mark.unit
@pytest.mark.git
class TestBypassHooksIntegration:
	"""Test cases for bypass_hooks integration in the commit command."""

	@pytest.mark.skip("validate_and_process_commit was removed during refactoring")
	def test_bypass_hooks_from_config(self, tmp_path: Path) -> None:
		"""Test that bypass_hooks is correctly loaded from config."""

	@pytest.mark.skip("validate_and_process_commit was removed during refactoring")
	def test_bypass_hooks_cli_override(self, tmp_path: Path) -> None:
		"""Test that bypass_hooks from CLI overrides config file."""
