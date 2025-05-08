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


@pytest.mark.unit
@pytest.mark.cli
class TestCliUtils(CLITestBase):
	"""Test cases for CLI utility functions."""

	def test_spinner_state_singleton(self) -> None:
		"""Test that SpinnerState behaves as a singleton."""
		# Create first instance
		spinner1 = SpinnerState()
		spinner1.is_active = True

		# Create second instance - should be same object
		spinner2 = SpinnerState()

		# Both should be the same instance
		assert spinner1 is spinner2
		assert spinner2.is_active is True

		# Change value on second instance
		spinner2.is_active = False

		# First instance should reflect the change
		assert spinner1.is_active is False

	def test_progress_indicator_in_test_environment(self) -> None:
		"""Test progress indicator behavior in test environment."""
		# PYTEST_CURRENT_TEST is set in pytest environment
		with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_name"}):
			with progress_indicator("Testing...") as advance:
				# Call the advance function to ensure it doesn't raise errors
				advance(1)
			# Verify spinner state is not changed
			assert not SpinnerState().is_active

	def test_progress_indicator_in_ci_environment(self) -> None:
		"""Test progress indicator behavior in CI environment."""
		# CI environment variable is set
		with patch.dict(os.environ, {"CI": "true"}):
			with progress_indicator("Testing...") as advance:
				# Call the advance function to ensure it doesn't raise errors
				advance(1)
			# Verify spinner state is not changed
			assert not SpinnerState().is_active

	def test_progress_indicator_active_spinner(self) -> None:
		"""Test progress indicator behavior when spinner is already active."""
		# Set spinner as active
		spinner_state = SpinnerState()
		spinner_state.is_active = True

		# Should not create new spinner when style is spinner
		with patch.object(console, "status") as mock_status:
			with progress_indicator("Testing...", style="spinner") as advance:
				advance(1)
			# Verify console.status was not called
			mock_status.assert_not_called()

		# Restore state
		spinner_state.is_active = False

	def test_progress_indicator_spinner_style(self) -> None:
		"""Test progress indicator with spinner style."""
		# Create clean environment (no PYTEST_CURRENT_TEST, no CI)
		with (
			patch.dict(os.environ, {}, clear=True),
			patch.object(console, "status", return_value=MagicMock()) as mock_status,
		):
			with progress_indicator("Working...", style="spinner") as advance:
				# Verify spinner is active
				assert SpinnerState().is_active
				advance(1)  # Should do nothing for spinner style

			# Verify console.status was called
			mock_status.assert_called_once_with("Working...")
			# Verify spinner is inactive after context exit
			assert not SpinnerState().is_active

	@patch.dict(os.environ, {}, clear=True)
	@patch("codemap.utils.cli_utils.Progress")
	def test_progress_indicator_progress_style(self, mock_progress_cls: MagicMock) -> None:
		"""Test progress indicator with progress style."""
		# We'll just verify the progress indicator is created with the right style
		# and that our advance function works without error
		with progress_indicator("Processing...", style="progress", total=10) as advance:
			advance(2)  # Advance by 2 units

		# Just verify Progress class was instantiated
		mock_progress_cls.assert_called_once()

	def test_progress_indicator_step_style(self) -> None:
		"""Test progress indicator with step style."""
		with patch.dict(os.environ, {}, clear=True), patch.object(console, "print") as mock_print:
			with progress_indicator("Stepping...", style="step", total=5) as advance:
				advance(2)  # Advance by 2 steps
				advance(1)  # Advance by 1 more step
				# Complete the steps
				advance(2)  # This brings the total to 5/5

			# Verify initial, progress, and completion prints
			assert mock_print.call_count >= 4
			mock_print.assert_any_call("Stepping... [0/5]")
			mock_print.assert_any_call("Stepping... [2/5]")
			mock_print.assert_any_call("Stepping... [3/5]")
			mock_print.assert_any_call("Stepping... [5/5]")
			mock_print.assert_any_call("Stepping... [green]Complete![/green]")
