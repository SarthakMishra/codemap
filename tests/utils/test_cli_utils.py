"""Tests for CLI utility functions."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.progress import Progress

from codemap.utils.cli_utils import (
	SpinnerState,
	console,
	create_spinner_progress,
	ensure_directory_exists,
	loading_spinner,
	setup_logging,
)
from tests.base import CLITestBase


@pytest.mark.unit
@pytest.mark.cli
class TestCliUtils(CLITestBase):
	"""Test cases for CLI utility functions."""

	def test_setup_logging_verbose(self) -> None:
		"""Test logging setup with verbose mode enabled."""
		with patch("logging.basicConfig") as mock_logging:
			setup_logging(is_verbose=True)
			mock_logging.assert_called_once()
			assert mock_logging.call_args[1]["level"] == "DEBUG"

	def test_setup_logging_non_verbose(self) -> None:
		"""Test logging setup with verbose mode disabled."""
		with patch("logging.basicConfig") as mock_logging, patch.dict(os.environ, {}, clear=True):
			setup_logging(is_verbose=False)
			mock_logging.assert_called_once()
			assert mock_logging.call_args[1]["level"] == "ERROR"

	def test_setup_logging_env_variables(self) -> None:
		"""Test logging setup with environment variables."""
		# Test with verbose=False and LOG_LEVEL environment variable set
		# Note: The LOG_LEVEL is now ignored when is_verbose=False as we always use ERROR
		with patch("logging.basicConfig") as mock_logging, patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}, clear=True):
			setup_logging(is_verbose=False)
			mock_logging.assert_called_once()
			assert mock_logging.call_args[1]["level"] == "ERROR"

	def test_setup_logging_specific_loggers(self) -> None:
		"""Test that specific loggers are configured correctly."""
		with patch("logging.basicConfig"), patch("logging.getLogger") as mock_get_logger:
			mock_logger = mock_get_logger.return_value
			setup_logging(is_verbose=False)
			# Verify that getLogger was called for multiple loggers and they were set to ERROR
			assert mock_get_logger.call_count > 0
			assert mock_logger.setLevel.call_count > 0

	def test_create_spinner_progress(self) -> None:
		"""Test creation of spinner progress bar."""
		progress = create_spinner_progress()
		assert isinstance(progress, Progress)
		assert len(progress.columns) == 2  # Should have SpinnerColumn and TextColumn

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

	def test_loading_spinner_in_test_environment(self) -> None:
		"""Test loading spinner behavior in test environment."""
		# PYTEST_CURRENT_TEST is set in pytest environment
		with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_name"}), loading_spinner("Testing..."):
			# Verify spinner state is not changed
			assert not SpinnerState().is_active

	def test_loading_spinner_in_ci_environment(self) -> None:
		"""Test loading spinner behavior in CI environment."""
		# CI environment variable is set
		with patch.dict(os.environ, {"CI": "true"}), loading_spinner("Testing..."):
			# Verify spinner state is not changed
			assert not SpinnerState().is_active

	def test_loading_spinner_active_spinner(self) -> None:
		"""Test loading spinner behavior when spinner is already active."""
		# Set spinner as active
		spinner_state = SpinnerState()
		spinner_state.is_active = True

		# Should not create new spinner
		with patch.object(console, "status") as mock_status, loading_spinner("Testing..."):
			# Verify console.status was not called
			mock_status.assert_not_called()

		# Restore state
		spinner_state.is_active = False

	def test_loading_spinner_standard_usage(self) -> None:
		"""Test standard usage of loading spinner."""
		# Create clean environment (no PYTEST_CURRENT_TEST, no CI)
		with (
			patch.dict(os.environ, {}, clear=True),
			patch.object(console, "status", return_value=MagicMock()) as mock_status,
			loading_spinner("Working..."),
		):
			# Verify spinner is active
			assert SpinnerState().is_active
			# Verify console.status was called
			mock_status.assert_called_once_with("Working...")

		# Verify spinner is inactive after context exit
		assert not SpinnerState().is_active


@pytest.mark.unit
@pytest.mark.fs
class TestDirectoryUtils(CLITestBase):
	"""Test cases for directory utility functions."""

	def test_ensure_directory_exists_success(self, tmp_path: Path) -> None:
		"""Test ensuring a directory exists with success."""
		# Directory that doesn't exist yet
		test_dir = tmp_path / "new_dir"
		ensure_directory_exists(test_dir)
		assert test_dir.exists()
		assert test_dir.is_dir()

		# Directory that already exists
		ensure_directory_exists(test_dir)  # Should not raise an exception
		assert test_dir.exists()

	def test_ensure_directory_exists_permission_error(self) -> None:
		"""Test ensuring a directory exists with permission error."""
		with patch("pathlib.Path.mkdir") as mock_mkdir:
			mock_mkdir.side_effect = PermissionError("Permission denied")

			with patch("codemap.utils.cli_utils.console") as mock_console:
				with pytest.raises(PermissionError):
					ensure_directory_exists(Path("/invalid/path"))

				# Verify error is printed
				mock_console.print.assert_called_once()
				assert "Unable to create directory" in mock_console.print.call_args[0][0]
