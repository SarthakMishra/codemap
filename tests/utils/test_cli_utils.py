"""Tests for CLI utility functions."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.progress import Progress

from codemap.utils.cli_utils import (
    create_spinner_progress,
    ensure_directory_exists,
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
