"""Tests for the git commit workflow logic."""

import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest

from codemap.git.commit_generator.command import CommitCommand
from codemap.git.commit_generator.generator import CommitMessageGenerator
from codemap.git.diff_splitter.schemas import DiffChunk
from codemap.git.interactive import ChunkAction, CommitUI
from tests.conftest import skip_git_tests


@pytest.mark.unit
@pytest.mark.git
@skip_git_tests
class TestCommitWorkflow(unittest.TestCase):
	"""
	Test cases for commit workflow logic in CommitCommand.

	Tests the workflow for processing git commits, including chunk
	handling, message generation, and interactive actions.

	"""

	def setUp(self) -> None:
		"""Set up mocks for each test method."""
		self.mock_repo_root = Path("/mock/repo")
		self.mock_generator = Mock(spec=CommitMessageGenerator)
		# Configure the mock generator's method to return a 4-tuple
		self.mock_generator.generate_message_with_linting.return_value = (
			"mock commit message",  # message
			True,  # used_llm
			True,  # passed_linting
			[],  # lint_messages
		)
		self.mock_ui = Mock(spec=CommitUI)
		self.mock_chunk = Mock(spec=DiffChunk)
		self.mock_chunk.files = ["file1.py", "file2.py"]
		self.mock_chunk.content = "mock diff content"
		self.mock_chunk.description = None
		self.mock_chunk.is_llm_generated = False

		# Patch the setup_message_generator to return our mock
		# Note: setup_message_generator is in commit_cmd, not command
		patcher = patch("codemap.cli.commit_cmd.setup_message_generator", return_value=self.mock_generator)
		self.addCleanup(patcher.stop)
		patcher.start()

		# Instantiate the command object AFTER patching setup_message_generator
		self.command = CommitCommand()
		# Replace the generator instance created by __init__ with our fully mocked one
		self.command.message_generator = self.mock_generator
		self.command.ui = self.mock_ui  # Inject the mock CommitUI

	def test_generate_commit_message(self) -> None:
		"""
		Test _generate_commit_message method.

		Verifies that the method correctly processes a diff chunk and
		generates a commit message using the message generator.

		"""
		# Arrange: Set up mock return value
		self.mock_generator.generate_message.return_value = ("Test commit message", True)

		# Act: Call the method
		self.command._generate_commit_message(cast("DiffChunk", self.mock_chunk))

		# Assert: Verify results
		assert self.mock_chunk.description == "Test commit message"
		assert self.mock_chunk.is_llm_generated is True
		self.mock_generator.generate_message.assert_called_once_with(self.mock_chunk)

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

	@patch.object(CommitCommand, "_perform_commit")
	def test_process_chunk_accept(self, mock_perform_commit: Mock) -> None:
		"""
		Test _process_chunk with COMMIT action.

		Verifies commit is performed when user accepts.

		"""
		# Arrange: Mock UI to return COMMIT action
		self.mock_ui.get_user_action.return_value = ChunkAction.COMMIT
		mock_perform_commit.return_value = True  # Simulate successful commit
		# Ensure generate_message_with_linting returns a valid message
		self.mock_generator.generate_message_with_linting.return_value = ("Valid Commit Message", True, True, [])

		# Act: Call the method
		result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)

		# Assert: Verify results
		assert result is True  # Should return True on successful commit
		self.mock_generator.generate_message_with_linting.assert_called_once_with(self.mock_chunk)
		self.mock_ui.display_chunk.assert_called_once_with(self.mock_chunk, 0, 1)
		self.mock_ui.get_user_action.assert_called_once()
		mock_perform_commit.assert_called_once_with(self.mock_chunk, "Valid Commit Message")

	def test_process_chunk_skip(self) -> None:
		"""
		Test _process_chunk with SKIP action.

		Verifies skip action is handled correctly.

		"""
		# Arrange: Mock UI to return SKIP action
		self.mock_ui.get_user_action.return_value = ChunkAction.SKIP
		# Ensure generate_message_with_linting returns a valid message first
		self.mock_generator.generate_message_with_linting.return_value = ("Valid Commit Message", True, True, [])

		# Act: Call the method
		result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)

		# Assert: Verify results
		assert result is True  # Should return True to continue processing
		self.mock_generator.generate_message_with_linting.assert_called_once_with(self.mock_chunk)
		self.mock_ui.display_chunk.assert_called_once_with(self.mock_chunk, 0, 1)
		self.mock_ui.get_user_action.assert_called_once()
		self.mock_ui.show_skipped.assert_called_once_with(self.mock_chunk.files)

	# Test test_process_chunk_abort remains largely the same but check state
	def test_process_chunk_abort(self) -> None:
		"""Test _process_chunk with EXIT action."""
		# Arrange: Mock UI to return EXIT action
		self.mock_ui.get_user_action.return_value = ChunkAction.EXIT
		self.mock_generator.generate_message_with_linting.return_value = (
			"Valid Commit Message",
			True,
			True,
			[],  # Assume generation was fine
		)

		# Act & Assert: Confirm exit leads to False result and aborted state
		self.mock_ui.confirm_exit.return_value = True
		result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)
		assert result is False
		assert self.command.error_state == "aborted"
		self.mock_ui.confirm_exit.assert_called_once()

		# Reset mocks and state
		self.mock_ui.reset_mock()
		self.mock_generator.reset_mock()
		self.command.error_state = None

		# Arrange again for cancel exit
		self.mock_ui.get_user_action.return_value = ChunkAction.EXIT
		self.mock_generator.generate_message_with_linting.return_value = ("Valid Commit Message", True, True, [])

		# Act & Assert: Cancel exit continues loop (need to break it for test)
		self.mock_ui.confirm_exit.return_value = False
		# To prevent infinite loop in test, make get_user_action return SKIP second time
		self.mock_ui.get_user_action.side_effect = [ChunkAction.EXIT, ChunkAction.SKIP]
		result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)
		assert result is True  # Should eventually return True after skipping
		assert self.mock_ui.confirm_exit.call_count == 1  # Called once
		assert self.mock_ui.get_user_action.call_count == 2  # Called twice

	@patch.object(CommitCommand, "_process_chunk")
	def test_process_all_chunks_interactive(self, mock_process_chunk: Mock) -> None:
		"""Test process_all_chunks in interactive mode."""
		# Arrange
		mock_chunk2 = Mock(spec=DiffChunk)
		mock_chunk2.files = ["file3.py"]
		mock_chunks = [cast("DiffChunk", self.mock_chunk), cast("DiffChunk", mock_chunk2)]
		mock_process_chunk.return_value = True
		grand_total = len(mock_chunks)

		# Act
		result = self.command.process_all_chunks(mock_chunks, grand_total=grand_total, interactive=True)

		# Assert
		assert result is True
		assert mock_process_chunk.call_count == 2

	@patch.object(CommitCommand, "_perform_commit")
	# Patch the generate_message_with_linting method on the *mocked* instance
	# No need for @patch.object decorator here, configure the mock directly
	def test_process_all_chunks_non_interactive(self, mock_perform_commit: Mock) -> None:
		"""Test process_all_chunks in non-interactive mode."""
		# Arrange
		mock_perform_commit.return_value = True
		# Mock the return value for generate_message_with_linting on the instance
		self.mock_generator.generate_message_with_linting.return_value = ("Non-interactive message", True, True, [])

		mock_chunk2 = Mock(spec=DiffChunk)
		mock_chunk2.files = ["file3.py"]
		mock_chunks = [cast("DiffChunk", self.mock_chunk), cast("DiffChunk", mock_chunk2)]
		grand_total = len(mock_chunks)

		# Act: Call the method
		result = self.command.process_all_chunks(mock_chunks, grand_total=grand_total, interactive=False)

		# Assert: Verify results
		assert result is True
		# Check that the mock generator's method was called twice
		assert self.mock_generator.generate_message_with_linting.call_count == 2
		assert mock_perform_commit.call_count == 2
