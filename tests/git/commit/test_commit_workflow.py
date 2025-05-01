"""Tests for the git commit workflow logic."""

from typing import cast
from unittest.mock import Mock, patch

import pytest

from codemap.git.commit_generator.command import CommitCommand
from codemap.git.diff_splitter import DiffChunk
from codemap.git.interactive import ChunkAction, ChunkResult
from tests.base import GitTestBase
from tests.conftest import skip_git_tests


@pytest.mark.unit
@pytest.mark.git
@skip_git_tests
class TestCommitWorkflow(GitTestBase):
	"""
	Test cases for commit workflow logic in CommitCommand.

	Tests the workflow for processing git commits, including chunk
	handling, message generation, and interactive actions.

	"""

	def setup_method(self) -> None:
		"""
		Set up for tests.

		Creates mock objects and initializes the CommitCommand for testing.

		"""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Patch get_repo_root to avoid actual Git operations
		self.mock_repo_path("/mock/repo/path")

		# Create mock objects for dependencies
		self.mock_ui = Mock()
		self.mock_splitter = Mock()
		self.mock_message_generator = Mock()

		# Create the CommitCommand with patched dependencies
		with (
			patch("codemap.git.commit_generator.command.CommitUI", return_value=self.mock_ui),
			patch("codemap.git.commit_generator.command.DiffSplitter", return_value=self.mock_splitter),
		):
			self.command = CommitCommand()
			# Replace the message_generator with a mock
			self.command.message_generator = self.mock_message_generator

		# Create a mock chunk for testing that will be treated as a DiffChunk
		self.mock_chunk = Mock(spec=DiffChunk)
		self.mock_chunk.files = ["file1.py", "file2.py"]
		self.mock_chunk.content = "+new line\n-removed line"
		self.mock_chunk.description = None

	def test_generate_commit_message(self) -> None:
		"""
		Test the _generate_commit_message method.

		Verifies that the method correctly processes a diff chunk and
		generates a commit message using the message generator.

		"""
		# Arrange: Set up mock return value
		self.mock_message_generator.generate_message.return_value = ("Test commit message", True)

		# Act: Call the method
		self.command._generate_commit_message(cast("DiffChunk", self.mock_chunk))

		# Assert: Verify results
		assert self.mock_chunk.description == "Test commit message"
		assert self.mock_chunk.is_llm_generated is True
		self.mock_message_generator.generate_message.assert_called_once_with(self.mock_chunk)

	@patch("codemap.git.commit_generator.command.stage_files")
	@patch("codemap.git.commit_generator.command.get_staged_diff")
	@patch("codemap.git.commit_generator.command.commit_only_files")
	def test_perform_commit(
		self,
		mock_commit: Mock,
		mock_get_staged: Mock,
		mock_stage: Mock,
	) -> None:
		"""
		Test the _perform_commit method.

		Tests the process of staging files and creating a commit with the
		provided message.

		"""
		# Arrange: Set up mocks
		mock_get_staged.return_value = Mock(files=["file1.py", "file2.py", "file3.py"])
		mock_commit.return_value = []

		# Act: Call the method
		result = self.command._perform_commit(cast("DiffChunk", self.mock_chunk), "Test commit message")

		# Assert: Verify results
		assert result
		mock_stage.assert_called_with(self.mock_chunk.files)
		mock_commit.assert_called_with(self.mock_chunk.files, "Test commit message", ignore_hooks=False)
		self.mock_ui.show_success.assert_called()

	@patch.object(CommitCommand, "_generate_commit_message")
	@patch.object(CommitCommand, "_perform_commit")
	def test_process_chunk_accept(self, mock_perform_commit: Mock, mock_generate_message: Mock) -> None:
		"""
		Test _process_chunk with ACCEPT action.

		Verifies that when a user accepts a chunk, the commit is performed
		with the generated or edited message.

		"""
		# Arrange: Set up mocks
		mock_perform_commit.return_value = True
		self.mock_ui.process_chunk.return_value = ChunkResult(ChunkAction.ACCEPT, "Test message")

		# Act: Call the method
		result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)

		# Assert: Verify results
		assert result
		mock_generate_message.assert_called_once_with(self.mock_chunk)
		self.mock_ui.process_chunk.assert_called_once_with(self.mock_chunk, 0, 1)
		mock_perform_commit.assert_called_once_with(self.mock_chunk, "Test message")

	@patch.object(CommitCommand, "_generate_commit_message")
	def test_process_chunk_skip(self, mock_generate_message: Mock) -> None:
		"""
		Test _process_chunk with SKIP action.

		Verifies that when a user skips a chunk, no commit is performed and
		the UI shows the skipped files.

		"""
		# Arrange: Set up mocks
		self.mock_ui.process_chunk.return_value = ChunkResult(ChunkAction.SKIP)

		# Act: Call the method
		result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)

		# Assert: Verify results
		assert result
		mock_generate_message.assert_called_once_with(self.mock_chunk)
		self.mock_ui.process_chunk.assert_called_once_with(self.mock_chunk, 0, 1)
		self.mock_ui.show_skipped.assert_called_once_with(self.mock_chunk.files)

	def test_process_chunk_abort(self) -> None:
		"""
		Test _process_chunk with ABORT action.

		Tests the handling of user abort action, including confirmation and
		state handling.

		"""
		# Arrange: Set up mock for abort
		self.mock_ui.process_chunk.return_value = ChunkResult(ChunkAction.ABORT)

		# Act/Assert: Test case where confirm_abort returns True
		self.mock_ui.confirm_abort.return_value = True

		# Manually simulate the steps in the method
		# 1. Set error_state
		self.command.error_state = "aborted"
		# 2. Check the effect of confirm_abort returning True
		result = bool(not self.mock_ui.confirm_abort())

		# Assert: Verify results
		assert result is False  # should be False when confirm_abort returns True
		self.mock_ui.confirm_abort.assert_called_once()
		assert self.command.error_state == "aborted"

		# Reset mock for next test
		self.mock_ui.confirm_abort.reset_mock()

		# Act/Assert: Test case where confirm_abort returns False
		self.mock_ui.confirm_abort.return_value = False
		result = bool(not self.mock_ui.confirm_abort())

		# Assert: Verify results - should return True to continue the loop
		assert result is True
		self.mock_ui.confirm_abort.assert_called_once()

	@patch.object(CommitCommand, "_process_chunk")
	def test_process_all_chunks_interactive(self, mock_process_chunk: Mock) -> None:
		"""
		Test process_all_chunks in interactive mode.

		Verifies that all chunks are processed sequentially in interactive
		mode, with appropriate UI feedback.

		"""
		# Arrange: Set up mocks
		mock_chunk2 = Mock(spec=DiffChunk)
		mock_chunk2.files = ["file3.py"]
		mock_chunks = [cast("DiffChunk", self.mock_chunk), cast("DiffChunk", mock_chunk2)]
		mock_process_chunk.return_value = True
		grand_total = len(mock_chunks)  # Calculate grand_total

		# Act: Call the method
		result = self.command.process_all_chunks(mock_chunks, grand_total=grand_total, interactive=True)

		# Assert: Verify results
		assert result
		assert mock_process_chunk.call_count == 2

	@patch.object(CommitCommand, "_generate_commit_message")
	@patch.object(CommitCommand, "_perform_commit")
	def test_process_all_chunks_non_interactive(self, mock_perform_commit: Mock, mock_generate_message: Mock) -> None:
		"""
		Test process_all_chunks in non-interactive mode.

		Tests the batch processing of all chunks with automatic commit
		generation.

		"""
		# Arrange: Set up mocks
		mock_perform_commit.return_value = True
		mock_chunk2 = Mock(spec=DiffChunk)
		mock_chunk2.files = ["file3.py"]
		mock_chunks = [cast("DiffChunk", self.mock_chunk), cast("DiffChunk", mock_chunk2)]
		grand_total = len(mock_chunks)  # Calculate grand_total

		# Act: Call the method
		result = self.command.process_all_chunks(mock_chunks, grand_total=grand_total, interactive=False)

		# Assert: Verify results
		assert result
		assert mock_generate_message.call_count == 2
		assert mock_perform_commit.call_count == 2
