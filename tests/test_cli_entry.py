"""Tests for the cli_app module."""

from unittest.mock import patch


def test_main_function() -> None:
    """Test the main function."""
    with patch("codemap.cli_app.app") as mock_app:
        mock_app.return_value = 0

        # Import the module to get the main function
        from codemap.cli_app import main

        # Call the main function
        result = main()

        # Check the result
        assert result == 0
        mock_app.assert_called_once()


def test_module_import() -> None:
    """Test that the module can be imported without errors."""
    # Import the module
    import codemap.cli_app

    # Check that the module has the expected attributes
    assert hasattr(codemap.cli_app, "app")
    assert hasattr(codemap.cli_app, "main")

    # Check that app is a Typer instance
    assert str(type(codemap.cli_app.app)).endswith("typer.main.Typer'>")
