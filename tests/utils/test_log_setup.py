"""Tests for logging setup utility functions."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from codemap.utils.log_setup import setup_logging


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
		mock_root_logger.setLevel.assert_called_once_with(logging.ERROR)
		mock_rich_handler.assert_called_once_with(
			level=logging.ERROR, rich_tracebacks=True, show_time=True, show_path=False
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
		mock_root_logger.setLevel.assert_called_once_with(logging.ERROR)
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

	@patch("codemap.utils.log_setup.logging.getLogger")
	@patch("codemap.utils.log_setup.logging.FileHandler")
	def test_setup_logging_with_file(self, mock_file_handler: MagicMock, mock_get_logger: MagicMock) -> None:
		"""Test setup_logging with log_file_path provided."""
		mock_root_logger = MagicMock()
		mock_get_logger.return_value = mock_root_logger
		mock_root_logger.handlers = []
		log_file_path = Path("/tmp/test.log")

		with patch("codemap.utils.log_setup.Path") as mock_path:
			mock_path_instance = MagicMock()
			mock_path_instance.parent = MagicMock()
			mock_path.return_value = mock_path_instance

			setup_logging(log_file_path=log_file_path)

		mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
		mock_file_handler.assert_called_once()
		mock_root_logger.addHandler.assert_called_with(mock_file_handler.return_value)
		assert mock_root_logger.addHandler.call_count == 2  # Once for console, once for file
