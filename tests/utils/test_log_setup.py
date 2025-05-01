"""Tests for logging setup utility functions."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch

import pytest
from rich.text import Text

from codemap.utils.log_setup import (
	display_error_summary,
	display_warning_summary,
	log_environment_info,
	setup_logging,
)


@pytest.mark.unit
class TestLogSetup:
	"""Test cases for logging setup functions."""

	@patch("codemap.utils.log_setup.logging.getLogger")
	@patch("codemap.utils.log_setup.RichHandler")
	def test_setup_logging_defaults(self, mock_rich_handler: MagicMock, mock_get_logger: MagicMock) -> None:
		"""Test setup_logging with default arguments (INFO level, console logging)."""
		mock_root_logger = MagicMock()
		mock_get_logger.return_value = mock_root_logger
		mock_root_logger.handlers = []  # Start with no handlers

		setup_logging()

		mock_get_logger.assert_called_once_with()
		mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
		mock_rich_handler.assert_called_once_with(
			level=logging.INFO, rich_tracebacks=True, show_time=True, show_path=False
		)
		mock_root_logger.addHandler.assert_called_once_with(mock_rich_handler.return_value)

	@patch("codemap.utils.log_setup.logging.getLogger")
	@patch("codemap.utils.log_setup.RichHandler")
	def test_setup_logging_verbose(self, mock_rich_handler: MagicMock, mock_get_logger: MagicMock) -> None:
		"""Test setup_logging with verbose=True (DEBUG level)."""
		mock_root_logger = MagicMock()
		mock_get_logger.return_value = mock_root_logger
		mock_root_logger.handlers = []

		setup_logging(is_verbose=True)

		mock_get_logger.assert_called_once_with()
		mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)
		mock_rich_handler.assert_called_once_with(
			level=logging.DEBUG, rich_tracebacks=True, show_time=True, show_path=True
		)
		mock_root_logger.addHandler.assert_called_once_with(mock_rich_handler.return_value)

	@patch("codemap.utils.log_setup.logging.getLogger")
	@patch("codemap.utils.log_setup.RichHandler")
	def test_setup_logging_no_console(self, mock_rich_handler: MagicMock, mock_get_logger: MagicMock) -> None:
		"""Test setup_logging with log_to_console=False."""
		mock_root_logger = MagicMock()
		mock_get_logger.return_value = mock_root_logger
		mock_root_logger.handlers = []

		setup_logging(log_to_console=False)

		mock_get_logger.assert_called_once_with()
		mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
		mock_rich_handler.assert_not_called()
		mock_root_logger.addHandler.assert_not_called()

	@patch("codemap.utils.log_setup.logging.getLogger")
	def test_setup_logging_clears_handlers(self, mock_get_logger: MagicMock) -> None:
		"""Test that setup_logging removes existing handlers."""
		mock_root_logger = MagicMock()
		mock_handler1 = MagicMock()
		mock_handler2 = MagicMock()
		mock_root_logger.handlers = [mock_handler1, mock_handler2]
		mock_get_logger.return_value = mock_root_logger

		with patch("codemap.utils.log_setup.RichHandler"):  # Mock RichHandler to avoid side effects
			setup_logging()

		# Check handlers were removed
		assert mock_root_logger.removeHandler.call_count == 2
		mock_root_logger.removeHandler.assert_has_calls([call(mock_handler1), call(mock_handler2)], any_order=True)
		# Check new handler was added (assuming console logging is default)
		assert mock_root_logger.addHandler.call_count == 1

	@patch("platform.python_version")
	@patch("platform.platform")
	@patch("codemap.__version__", "1.2.3")
	@patch("codemap.utils.log_setup.logging.getLogger")
	def test_log_environment_info_success(
		self, mock_get_logger: MagicMock, mock_platform_platform: MagicMock, mock_platform_python_version: MagicMock
	) -> None:
		"""Test log_environment_info logs correct info on success."""
		mock_logger = MagicMock()
		mock_get_logger.return_value = mock_logger
		# Configure the return values of the patched platform functions
		mock_platform_python_version.return_value = "3.10.0"
		mock_platform_platform.return_value = "Linux-Test"

		log_environment_info()

		mock_get_logger.assert_called_once_with("codemap.utils.log_setup")
		mock_logger.info.assert_has_calls(
			[
				call("CodeMap version: %s", "1.2.3"),
				call("Python version: %s", "3.10.0"),
				call("Platform: %s", "Linux-Test"),
			]
		)
		# Check that the patched platform functions were called
		mock_platform_python_version.assert_called_once()
		mock_platform_platform.assert_called_once()

	@patch("platform.python_version", side_effect=RuntimeError("Mock platform error"))
	@patch("codemap.utils.log_setup.logging.getLogger")
	def test_log_environment_info_exception(
		self,
		mock_get_logger: MagicMock,
		_mock_python_version: MagicMock,  # Prefixed unused arg
	) -> None:
		"""Test log_environment_info handles exceptions during info gathering."""
		# The side_effect on python_version should trigger the except block
		mock_logger = MagicMock()
		mock_get_logger.return_value = mock_logger

		log_environment_info()

		mock_get_logger.assert_called_once_with("codemap.utils.log_setup")
		mock_logger.exception.assert_called_once()
		args, _ = mock_logger.exception.call_args
		# Check only the message string, as exception info is handled separately
		assert args[0] == "Error logging environment info:"

	@patch("codemap.utils.log_setup.console")
	@patch("codemap.utils.log_setup.Rule")
	def test_display_error_summary(self, mock_rule: MagicMock, mock_console: MagicMock) -> None:
		"""Test display_error_summary prints correctly formatted output."""
		error_message = "Something went wrong!"
		expected_title = Text("Error Summary", style="bold red")
		mock_rule_instance = MagicMock()
		mock_rule.return_value = mock_rule_instance

		display_error_summary(error_message)

		mock_rule.assert_has_calls(
			[
				call(expected_title, style="red"),
				call(style="red"),
			]
		)
		mock_console.print.assert_has_calls(
			[
				call(),
				call(mock_rule_instance),  # First rule with title
				call(f"\n{error_message}\n"),
				call(mock_rule_instance),  # Second rule without title
				call(),
			]
		)

	@patch("codemap.utils.log_setup.console")
	@patch("codemap.utils.log_setup.Rule")
	def test_display_warning_summary(self, mock_rule: MagicMock, mock_console: MagicMock) -> None:
		"""Test display_warning_summary prints correctly formatted output."""
		warning_message = "This might be an issue."
		expected_title = Text("Warning Summary", style="bold yellow")
		mock_rule_instance = MagicMock()
		mock_rule.return_value = mock_rule_instance

		display_warning_summary(warning_message)

		mock_rule.assert_has_calls(
			[
				call(expected_title, style="yellow"),
				call(style="yellow"),
			]
		)
		mock_console.print.assert_has_calls(
			[
				call(),
				call(mock_rule_instance),  # First rule with title
				call(f"\n{warning_message}\n"),
				call(mock_rule_instance),  # Second rule without title
				call(),
			]
		)
