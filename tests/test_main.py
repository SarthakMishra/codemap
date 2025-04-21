"""Tests for the __main__ module."""

import runpy
from unittest.mock import patch


def test_main_module():
    """Test the __main__ module."""
    with patch("codemap.cli_entry.app") as mock_app:
        # Use runpy to run the module as __main__
        runpy.run_module("codemap.__main__", run_name="__main__")
        
        # Verify app was called
        mock_app.assert_called_once()


def test_main_import():
    """Test importing the __main__ module."""
    with patch("codemap.cli_entry.app") as mock_app:
        # Import the module normally (should not call app)
        import codemap.__main__
        
        # Verify app was not called
        mock_app.assert_not_called()