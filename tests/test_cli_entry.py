"""Tests for the cli_entry module."""

from unittest.mock import patch


def test_main_function() -> None:
    """Test the main function."""
    with patch("codemap.cli_entry.app") as mock_app:
        mock_app.return_value = 0

        # Import the module to get the main function
        from codemap.cli_entry import main

        # Call the main function
        result = main()

        # Check the result
        assert result == 0
        mock_app.assert_called_once()


def test_module_import() -> None:
    """Test that the module can be imported without errors."""
    # Import the module
    import codemap.cli_entry

    # Check that the module has the expected attributes
    assert hasattr(codemap.cli_entry, "app")
    assert hasattr(codemap.cli_entry, "main")

    # Check that app is a Typer instance
    assert str(type(codemap.cli_entry.app)).endswith("typer.main.Typer'>")
