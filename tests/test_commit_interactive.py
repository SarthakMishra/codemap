"""Tests for the interactive commit UI."""

import unittest
from unittest.mock import Mock, patch

from codemap.git.commit.interactive import ChunkAction, CommitUI


class TestCommitUI(unittest.TestCase):
    """Test cases for the CommitUI class."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.ui = CommitUI()

        # Create a mock chunk
        self.mock_chunk = Mock()
        self.mock_chunk.files = ["file1.py", "file2.py"]
        self.mock_chunk.content = "+new line\n-removed line"
        self.mock_chunk.description = "Test commit message"
        self.mock_chunk.is_llm_generated = True

    @patch("rich.console.Console.print")
    def test_display_chunk(self, mock_print: Mock) -> None:
        """Test that display_chunk correctly formats chunk for display."""
        self.ui.display_chunk(self.mock_chunk, 0, 1)
        # Assert that console.print was called at least once
        mock_print.assert_called()

    @patch("rich.console.Console.print")
    def test_display_message(self, mock_print: Mock) -> None:
        """Test that display_message shows a commit message panel."""
        self.ui.display_message("Test message", is_llm_generated=True)
        mock_print.assert_called()

    @patch("questionary.select")
    def test_get_user_action(self, mock_select: Mock) -> None:
        """Test that get_user_action returns correct ChunkAction."""
        mock_select.return_value.ask.return_value = "Commit with this message"
        action = self.ui.get_user_action()
        assert action == ChunkAction.ACCEPT

    @patch("rich.prompt.Prompt.ask")
    def test_edit_message(self, mock_ask: Mock) -> None:
        """Test that edit_message returns the edited message."""
        mock_ask.return_value = "Edited message"
        result = self.ui.edit_message("Original message")
        assert result == "Edited message"

    @patch.object(CommitUI, "display_chunk")
    @patch.object(CommitUI, "get_user_action")
    @patch.object(CommitUI, "edit_message")
    def test_process_chunk_accept(self, mock_edit: Mock, mock_action: Mock, mock_display: Mock) -> None:
        """Test that process_chunk returns correct ChunkResult for ACCEPT action."""
        mock_action.return_value = ChunkAction.ACCEPT
        result = self.ui.process_chunk(self.mock_chunk, 0, 1)

        assert result.action == ChunkAction.ACCEPT
        assert result.message == "Test commit message"
        mock_display.assert_called_once()
        mock_edit.assert_not_called()

    @patch.object(CommitUI, "display_chunk")
    @patch.object(CommitUI, "get_user_action")
    @patch.object(CommitUI, "edit_message")
    def test_process_chunk_edit(self, mock_edit: Mock, mock_action: Mock, mock_display: Mock) -> None:
        """Test that process_chunk returns correct ChunkResult for EDIT action."""
        mock_action.return_value = ChunkAction.EDIT
        mock_edit.return_value = "Edited message"

        result = self.ui.process_chunk(self.mock_chunk, 0, 1)

        assert result.action == ChunkAction.ACCEPT
        assert result.message == "Edited message"
        mock_display.assert_called_once()
        mock_edit.assert_called_once()

    @patch.object(CommitUI, "display_chunk")
    @patch.object(CommitUI, "get_user_action")
    def test_process_chunk_other_actions(self, mock_action: Mock, mock_display: Mock) -> None:  # noqa: ARG002
        """Test that process_chunk returns correct ChunkResult for other actions."""
        test_cases = [ChunkAction.SKIP, ChunkAction.ABORT, ChunkAction.REGENERATE]

        for action in test_cases:
            mock_action.return_value = action
            result = self.ui.process_chunk(self.mock_chunk, 0, 1)

            assert result.action == action
            assert result.message is None
