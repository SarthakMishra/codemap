"""Tests for CLI utility functions."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from codemap.utils.cli_utils import (
	SpinnerState,
	console,
	progress_indicator,
)
from tests.base import CLITestBase


# Helper to reset SpinnerState singleton for isolated tests
@pytest.fixture(autouse=True)
def reset_spinner_state_singleton():
	"""Reset the SpinnerState singleton for isolated tests."""
	SpinnerState._instance = None
	# Yield control to the test
	yield
	# Clean up after the test by resetting again
	SpinnerState._instance = None


@pytest.mark.unit
@pytest.mark.cli
class TestCliUtils(CLITestBase):
	"""Test cases for CLI utility functions."""

	def test_spinner_state_singleton(self) -> None:
		"""Test that SpinnerState behaves as a singleton."""
		spinner1 = SpinnerState()
		spinner2 = SpinnerState()
		assert spinner1 is spinner2

		# Test stack behavior through the singleton instance
		assert not spinner1.spinner_message_stack
		spinner1.start_new_spinner("Test Message")
		assert len(spinner1.spinner_message_stack) == 1
		assert spinner2.spinner_message_stack[0] == "Test Message"
		spinner2.stop_current_spinner_and_resume_parent()
		assert not spinner1.spinner_message_stack

	def test_progress_indicator_in_test_environment(self) -> None:
		"""Test progress indicator behavior in test environment."""
		with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_name"}):
			with progress_indicator("Testing...") as advance:
				advance(None, 1, None)  # (description, completed, total)
			spinner_state = SpinnerState()
			assert not spinner_state.spinner_message_stack
			assert spinner_state.active_rich_status_cm is None

	def test_progress_indicator_in_ci_environment(self) -> None:
		"""Test progress indicator behavior in CI environment."""
		with patch.dict(os.environ, {"CI": "true"}):
			with progress_indicator("Testing...") as advance:
				advance(None, 1, None)
			spinner_state = SpinnerState()
			assert not spinner_state.spinner_message_stack
			assert spinner_state.active_rich_status_cm is None

	@patch.object(console, "status")
	def test_progress_indicator_spinner_style_single(self, mock_console_status: MagicMock) -> None:
		"""Test progress indicator with a single spinner."""
		mock_status_instance = MagicMock()
		mock_console_status.return_value = mock_status_instance

		with patch.dict(os.environ, {}, clear=True), progress_indicator("Working...", style="spinner") as advance:
			advance(None, None, None)  # No-op for spinner
			spinner_state = SpinnerState()
			assert spinner_state.spinner_message_stack == ["Working..."]
			assert spinner_state.active_rich_status_cm is mock_status_instance
			mock_console_status.assert_called_once_with("Working...")
			mock_status_instance.__enter__.assert_called_once()

		# After context exit
		spinner_state = SpinnerState()
		assert not spinner_state.spinner_message_stack
		assert spinner_state.active_rich_status_cm is None
		mock_status_instance.__exit__.assert_called_once()

	@patch.object(console, "status")
	def test_progress_indicator_nested_spinners_tree_display(self, mock_console_status: MagicMock) -> None:
		"""Test nested spinners display as a tree and update correctly."""
		mock_status_instance = MagicMock()
		mock_console_status.return_value = mock_status_instance

		with patch.dict(os.environ, {}, clear=True), progress_indicator("Parent Task", style="spinner"):
			# Initial call for parent
			mock_console_status.assert_called_with("Parent Task")
			mock_status_instance.update.assert_not_called()  # No updates yet for single spinner
			mock_status_instance.reset_mock()  # Reset for update calls

			with progress_indicator("Child Task 1", style="spinner"):
				# Child 1 starts, status should be updated with tree
				# Expected tree: Parent Task
				#                └─ ▸ Child Task 1
				expected_tree_child1 = "Parent Task\n[green]└─ [/green]▸ Child Task 1"
				mock_status_instance.update.assert_called_with(expected_tree_child1)
				mock_status_instance.update.reset_mock()

				with progress_indicator("Grandchild Task", style="spinner"):
					# Grandchild starts
					# Expected tree: Parent Task
					#                └─ ▸ Child Task 1
					#                   └─ ▸ Grandchild Task
					expected_tree_grandchild = (
						"Parent Task\n[green]└─ [/green]▸ Child Task 1\n   [green]└─ [/green]▸ Grandchild Task"
					)
					mock_status_instance.update.assert_called_with(expected_tree_grandchild)
					mock_status_instance.update.reset_mock()

				# Grandchild ends, should revert to Child 1 tree
				mock_status_instance.update.assert_called_with(expected_tree_child1)
				mock_status_instance.update.reset_mock()

			# Child 1 ends, should revert to Parent Task only
			mock_status_instance.update.assert_called_with("Parent Task")

		# All spinners exited
		spinner_state = SpinnerState()
		assert not spinner_state.spinner_message_stack
		assert spinner_state.active_rich_status_cm is None

	@patch("codemap.utils.cli_utils.Progress")
	@patch.object(SpinnerState, "temporarily_halt_visual_spinner")
	@patch.object(SpinnerState, "resume_visual_spinner_if_needed")
	@patch.object(console, "status")  # To control spinner creation
	def test_progress_bar_halts_and_resumes_spinner(
		self,
		mock_console_status: MagicMock,
		mock_resume_spinner: MagicMock,
		mock_halt_spinner: MagicMock,
		_mock_progress_cls: MagicMock,
	) -> None:
		"""Test that a progress bar halts an active spinner and resumes it after."""
		mock_status_instance = MagicMock()
		mock_console_status.return_value = mock_status_instance

		with patch.dict(os.environ, {}, clear=True):
			# Start a spinner first
			with progress_indicator("Outer Spinner", style="spinner"):
				assert SpinnerState().spinner_message_stack == ["Outer Spinner"]
				mock_console_status.assert_called_with("Outer Spinner")  # Initial creation
				mock_halt_spinner.assert_not_called()
				mock_resume_spinner.assert_not_called()

				# Now start a progress bar within the spinner's context
				with progress_indicator("Progressing...", style="progress", total=10) as advance:
					advance(None, 5, 10)  # Update progress bar
					# Check spinner was halted
					mock_halt_spinner.assert_called_once()
					# Spinner stack should still exist, but visual spinner (active_rich_status_cm) should be None during halt
					assert SpinnerState().spinner_message_stack == ["Outer Spinner"]
					# active_rich_status_cm is made None by _stop_active_status_cm called by halt
					# This is an internal detail, difficult to assert directly without more mocks or checks

				# After progress bar finishes, spinner should be resumed
				mock_resume_spinner.assert_called_once()
				# We don't need to check console.status call count, as resume_visual_spinner_if_needed is mocked
				# and we've verified it was called, which is sufficient

			# After all contexts exit, spinner stack should be empty
			assert not SpinnerState().spinner_message_stack

	@patch.dict(os.environ, {}, clear=True)
	@patch("codemap.utils.cli_utils.Progress")
	def test_progress_indicator_progress_style_no_spinner_active(self, mock_progress_cls: MagicMock) -> None:
		"""Test progress indicator with progress style when no spinner is active."""
		with progress_indicator("Processing...", style="progress", total=10) as advance:
			advance(None, 2, 10)  # (description, completed, total)

		mock_progress_cls.assert_called_once()
		# Ensure no spinner interactions happened
		spinner_state = SpinnerState()
		assert not spinner_state.spinner_message_stack
		assert spinner_state.active_rich_status_cm is None
