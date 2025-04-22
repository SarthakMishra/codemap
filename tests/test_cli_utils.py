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


def test_setup_logging() -> None:
    """Test logging setup with different verbosity levels."""
    # Test with verbose=True
    with patch("logging.basicConfig") as mock_logging:
        setup_logging(is_verbose=True)
        mock_logging.assert_called_once()
        assert mock_logging.call_args[1]["level"] == "DEBUG"

    # Test with verbose=False - should now be ERROR level by default
    with patch("logging.basicConfig") as mock_logging, patch.dict(os.environ, {}, clear=True):
        setup_logging(is_verbose=False)
        mock_logging.assert_called_once()
        assert mock_logging.call_args[1]["level"] == "ERROR"

    # Test with verbose=False and LOG_LEVEL environment variable set
    # Note: The LOG_LEVEL is now ignored when is_verbose=False as we always use ERROR
    with patch("logging.basicConfig") as mock_logging, patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}, clear=True):
        setup_logging(is_verbose=False)
        mock_logging.assert_called_once()
        assert mock_logging.call_args[1]["level"] == "ERROR"

    # Add test for specific loggers being set to ERROR level when not verbose
    with patch("logging.basicConfig"), patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        setup_logging(is_verbose=False)
        # Verify that getLogger was called for multiple loggers and they were set to ERROR
        assert mock_get_logger.call_count > 0
        assert mock_logger.setLevel.call_count > 0


def test_create_spinner_progress() -> None:
    """Test creation of spinner progress bar."""
    progress = create_spinner_progress()
    assert isinstance(progress, Progress)
    assert len(progress.columns) == 2  # Should have SpinnerColumn and TextColumn


def test_ensure_directory_exists_success(tmp_path: Path) -> None:
    """Test ensuring a directory exists with success."""
    # Directory that doesn't exist yet
    test_dir = tmp_path / "new_dir"
    ensure_directory_exists(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()

    # Directory that already exists
    ensure_directory_exists(test_dir)  # Should not raise an exception
    assert test_dir.exists()


def test_ensure_directory_exists_permission_error() -> None:
    """Test ensuring a directory exists with permission error."""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        mock_mkdir.side_effect = PermissionError("Permission denied")

        with patch("codemap.utils.cli_utils.console") as mock_console:
            with pytest.raises(PermissionError):
                ensure_directory_exists(Path("/invalid/path"))

            # Verify error is printed
            mock_console.print.assert_called_once()
            assert "Unable to create directory" in mock_console.print.call_args[0][0]
