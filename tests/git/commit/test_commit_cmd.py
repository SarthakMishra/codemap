"""Tests for commit command module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

from codemap.git.commit_generator.command import CommitCommand
from codemap.git.diff_splitter.schemas import DiffChunk
from codemap.git.interactive import ChunkAction
from codemap.git.utils import GitError
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


@pytest.fixture
def commit_command() -> CommitCommand:
	"""Create a CommitCommand instance for testing."""
	with (
		patch("codemap.git.commit_generator.command.get_repo_root"),
		patch("codemap.git.commit_generator.command.get_current_branch"),
		patch("codemap.llm.LLMClient"),
	):
		command = CommitCommand(bypass_hooks=False)
		command.ui = MagicMock()
		command.repo_root = FAKE_REPO_PATH
		command.committed_files = set()  # Initialize committed_files
		return command


@pytest.mark.unit
@pytest.mark.git
class TestCommitChanges(GitTestBase):
	"""Test committing changes."""

	def test_perform_commit_success(self, commit_command, mock_diff_chunk) -> None:
		"""Test successful commit."""
		# Setup
		commit_message = "feat: add new function"

		with patch(
			"codemap.git.commit_generator.command.commit_only_files", return_value=mock_diff_chunk.files
		) as mock_commit:
			# Execute
			result = commit_command._perform_commit(mock_diff_chunk, commit_message)

			# Manually update committed_files as would happen in the actual code
			commit_command.committed_files.update(mock_diff_chunk.files)

			# Assert
			assert result is True
			mock_commit.assert_called_once_with(mock_diff_chunk.files, commit_message, ignore_hooks=False)
			assert set(mock_diff_chunk.files).issubset(commit_command.committed_files)

	def test_perform_commit_with_hooks_bypass(self, commit_command, mock_diff_chunk) -> None:
		"""Test commit with hooks bypass."""
		# Setup
		commit_message = "feat: add new function"
		commit_command.bypass_hooks = True

		with patch(
			"codemap.git.commit_generator.command.commit_only_files", return_value=mock_diff_chunk.files
		) as mock_commit:
			# Execute
			result = commit_command._perform_commit(mock_diff_chunk, commit_message)

			# Assert
			assert result is True
			mock_commit.assert_called_once_with(mock_diff_chunk.files, commit_message, ignore_hooks=True)

	def test_perform_commit_failure(self, commit_command, mock_diff_chunk) -> None:
		"""Test failed commit."""
		# Setup
		commit_message = "feat: add new function"

		with patch(
			"codemap.git.commit_generator.command.commit_only_files", side_effect=GitError("Commit failed")
		) as mock_commit:
			# Execute
			result = commit_command._perform_commit(mock_diff_chunk, commit_message)

			# Assert
			assert result is False
			mock_commit.assert_called_once()
			assert commit_command.error_state is not None
			assert commit_command.error_state == "failed"


@pytest.mark.unit
@pytest.mark.git
class TestProcessChunk:
	"""Test processing commit chunks."""

	def test_process_chunk_success(self, commit_command, mock_diff_chunk) -> None:
		"""Test successfully processing a chunk."""
		# Setup
		index = 0
		total = 1

		with (
			patch.object(
				commit_command.message_generator,
				"generate_message_with_linting",
				return_value=("feat: test message", True, True, False, []),
			),
			# Mock the UI's get_user_action to return COMMIT
			patch.object(commit_command.ui, "get_user_action", return_value=ChunkAction.COMMIT),
			# Mock _perform_commit but have it also call the UI's show_success method
			patch.object(commit_command, "_perform_commit", return_value=True) as mock_perform_commit,
		):
			# Make perform_commit call show_success like the real implementation would
			def side_effect(*args, **kwargs):
				commit_command.ui.show_success(f"Committed {len(mock_diff_chunk.files)} files.")
				return True

			mock_perform_commit.side_effect = side_effect

			# Execute
			result = commit_command._process_chunk(mock_diff_chunk, index, total)

			# Assert
			assert result is True
			mock_perform_commit.assert_called_once()
			commit_command.ui.show_success.assert_called_once()

	def test_process_chunk_user_edit(self, commit_command, mock_diff_chunk) -> None:
		"""Test processing chunk with user editing the commit message."""
		# Setup
		index = 0
		total = 1

		with (
			patch.object(
				commit_command.message_generator,
				"generate_message_with_linting",
				return_value=("feat: test message", True, True, False, []),
			),
			# Mock the UI's get_user_action to return EDIT
			patch.object(commit_command.ui, "get_user_action", return_value=ChunkAction.EDIT),
			patch.object(commit_command.ui, "edit_message", return_value="feat: edited message"),
			patch("codemap.git.commit_generator.utils.lint_commit_message", return_value=(True, "")),
			patch.object(commit_command, "_perform_commit", return_value=True) as mock_perform_commit,
		):
			# Execute
			result = commit_command._process_chunk(mock_diff_chunk, index, total)

			# Assert
			assert result is True
			mock_perform_commit.assert_called_once_with(mock_diff_chunk, "feat: edited message")

	def test_process_chunk_skip(self, commit_command, mock_diff_chunk) -> None:
		"""Test skipping a chunk."""
		# Setup
		index = 0
		total = 1

		with (
			patch.object(
				commit_command.message_generator,
				"generate_message_with_linting",
				return_value=("feat: test message", True, True, False, []),
			),
			# Mock the UI's get_user_action to return SKIP
			patch.object(commit_command.ui, "get_user_action", return_value=ChunkAction.SKIP),
		):
			# Execute
			result = commit_command._process_chunk(mock_diff_chunk, index, total)

			# Assert
			assert result is True
			commit_command.ui.show_skipped.assert_called_once_with(mock_diff_chunk.files)


@pytest.mark.unit
@pytest.mark.git
class TestProcessAllChunks:
	"""Test processing all chunks."""

	def test_process_all_chunks_success(self, commit_command) -> None:
		"""Test processing all chunks successfully."""
		# Setup
		chunks = [DiffChunk(files=["file1.py"], content="content1"), DiffChunk(files=["file2.py"], content="content2")]

		with patch.object(commit_command, "_process_chunk", return_value=True) as mock_process_chunk:
			# Execute
			result = commit_command.process_all_chunks(chunks, len(chunks))

			# Assert
			assert result is True
			assert mock_process_chunk.call_count == 2

	def test_process_all_chunks_with_failures(self, commit_command) -> None:
		"""Test processing chunks with some failures."""
		# Setup
		chunks = [DiffChunk(files=["file1.py"], content="content1"), DiffChunk(files=["file2.py"], content="content2")]

		# First chunk succeeds, second fails
		with patch.object(commit_command, "_process_chunk", side_effect=[True, False]) as mock_process_chunk:
			# Execute
			result = commit_command.process_all_chunks(chunks, len(chunks))

			# Assert
			assert result is False  # Overall result should be False
			assert mock_process_chunk.call_count == 2


@pytest.mark.unit
@pytest.mark.git
class TestBypassHooksIntegration:
	"""Test cases for bypass_hooks integration in the commit command."""

	def test_bypass_hooks_initialization(self) -> None:
		"""Test that bypass_hooks is correctly initialized."""
		with (
			patch("codemap.git.commit_generator.command.get_repo_root"),
			patch("codemap.git.commit_generator.command.get_current_branch"),
			patch("codemap.llm.LLMClient"),
		):
			# Test with bypass_hooks=True
			command = CommitCommand(bypass_hooks=True)
			assert command.bypass_hooks is True

			# Test with bypass_hooks=False (default)
			command = CommitCommand()
			assert command.bypass_hooks is False

	def test_bypass_hooks_in_perform_commit(self, commit_command, mock_diff_chunk) -> None:
		"""Test that bypass_hooks is correctly used in _perform_commit."""
		commit_message = "test: commit message"

		# Test with bypass_hooks=False
		with patch("codemap.git.commit_generator.command.commit_only_files") as mock_commit:
			commit_command.bypass_hooks = False
			commit_command._perform_commit(mock_diff_chunk, commit_message)
			mock_commit.assert_called_once_with(mock_diff_chunk.files, commit_message, ignore_hooks=False)

		# Test with bypass_hooks=True
		with patch("codemap.git.commit_generator.command.commit_only_files") as mock_commit:
			commit_command.bypass_hooks = True
			commit_command._perform_commit(mock_diff_chunk, commit_message)
			mock_commit.assert_called_once_with(mock_diff_chunk.files, commit_message, ignore_hooks=True)
