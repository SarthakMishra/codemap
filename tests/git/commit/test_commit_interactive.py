"""Tests for the interactive commit UI."""

from unittest.mock import Mock, patch

import pytest

from codemap.git.interactive import ChunkAction, CommitUI
from tests.base import GitTestBase


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.interactive
class TestCommitUI(GitTestBase):
	"""
	Test cases for the CommitUI class.

	Tests the user interface components for interactive commit
	functionality.

	"""

	def setup_method(self) -> None:
		"""Set up test environment for each test method."""
		self.ui = CommitUI()

		# Create a mock chunk for testing
		self.mock_chunk = Mock()
		self.mock_chunk.files = ["file1.py", "file2.py"]
		self.mock_chunk.content = "+new line\n-removed line"
		self.mock_chunk.description = "Test commit message"
		self.mock_chunk.is_llm_generated = True

	@patch("rich.console.Console.print")
	def test_display_chunk(self, mock_print: Mock) -> None:
		"""
		Test that display_chunk correctly formats chunk for display.

		Verifies that the Console.print method is called to display the chunk
		information.

		"""
		# Act: Display the chunk
		self.ui.display_chunk(self.mock_chunk, 0, 1)

		# Assert: Console.print was called at least once
		mock_print.assert_called()

	@patch("rich.console.Console.print")
	def test_display_message(self, mock_print: Mock) -> None:
		"""
		Test that display_message shows a commit message panel.

		Checks that the commit message is properly displayed using Rich's
		formatting capabilities.

		"""
		# Act: Display a commit message
		self.ui.display_message("Test message", is_llm_generated=True)

		# Assert: Console.print was called
		mock_print.assert_called()

	@patch("questionary.select")
	def test_get_user_action(self, mock_select: Mock) -> None:
		"""
		Test that get_user_action returns correct ChunkAction.

		Tests the conversion from user selection string to the appropriate
		ChunkAction enum value.

		"""
		# Arrange: Set up the mock to return a specific action
		mock_select.return_value.ask.return_value = "Commit with this message"

		# Act: Get user action
		action = self.ui.get_user_action()

		# Assert: Correct action was returned
		assert action == ChunkAction.ACCEPT

	@patch("rich.prompt.Prompt.ask")
	def test_edit_message(self, mock_ask: Mock) -> None:
		"""
		Test that edit_message returns the edited message.

		Verifies that user input is correctly captured and returned.

		"""
		# Arrange: Mock user input
		mock_ask.return_value = "Edited message"

		# Act: Edit a message
		result = self.ui.edit_message("Original message")

		# Assert: Edited message is returned
		assert result == "Edited message"

	@patch.object(CommitUI, "display_chunk")
	@patch.object(CommitUI, "get_user_action")
	@patch.object(CommitUI, "edit_message")
	def test_process_chunk_accept(self, mock_edit: Mock, mock_action: Mock, mock_display: Mock) -> None:
		"""
		Test that process_chunk returns correct ChunkResult for ACCEPT action.

		Verifies the behavior when a user accepts a commit chunk without
		editing.

		"""
		# Arrange: Set up mocks
		mock_action.return_value = ChunkAction.ACCEPT

		# Act: Process the chunk
		result = self.ui.process_chunk(self.mock_chunk, 0, 1)

		# Assert: Result matches expectations
		assert result.action == ChunkAction.ACCEPT
		assert result.message == "Test commit message"
		mock_display.assert_called_once()
		mock_edit.assert_not_called()

	@patch.object(CommitUI, "display_chunk")
	@patch.object(CommitUI, "get_user_action")
	@patch.object(CommitUI, "edit_message")
	def test_process_chunk_edit(self, mock_edit: Mock, mock_action: Mock, mock_display: Mock) -> None:
		"""
		Test that process_chunk returns correct ChunkResult for EDIT action.

		Verifies the behavior when a user chooses to edit the commit message.

		"""
		# Arrange: Set up mocks
		mock_action.return_value = ChunkAction.EDIT
		mock_edit.return_value = "Edited message"

		# Act: Process the chunk
		result = self.ui.process_chunk(self.mock_chunk, 0, 1)

		# Assert: Result matches expectations
		assert result.action == ChunkAction.ACCEPT
		assert result.message == "Edited message"
		mock_display.assert_called_once()
		mock_edit.assert_called_once()

	@patch.object(CommitUI, "display_chunk")
	@patch.object(CommitUI, "get_user_action")
	def test_process_chunk_other_actions(self, mock_action: Mock, mock_display: Mock) -> None:
		"""Test that process_chunk returns correct ChunkResult for other actions."""
		# Test cases
		test_cases = [ChunkAction.SKIP, ChunkAction.ABORT, ChunkAction.REGENERATE]

		for action in test_cases:
			mock_action.return_value = action
			result = self.ui.process_chunk(self.mock_chunk, 0, 1)

			# Verify display_chunk was called
			mock_display.assert_called_once_with(self.mock_chunk, 0, 1)
			mock_display.reset_mock()  # Reset for next iteration

			assert result.action == action
			assert result.message is None
