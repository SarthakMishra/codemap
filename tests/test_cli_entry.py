"""Tests for the cli_app module."""

from unittest.mock import patch

import pytest

from tests.base import CLITestBase


@pytest.mark.cli
@pytest.mark.unit
class TestCliEntry(CLITestBase):
	"""Test cases for the CLI entry points."""

	def test_main_function(self) -> None:
		"""
		Test the main function.

		Tests that the main function in cli_app calls the app function and
		returns its result.

		"""
		with patch("codemap.cli_app.app") as mock_app:
			mock_app.return_value = 0

			# Import the module to get the main function
			from codemap.cli_app import main

			# Act: Call the main function
			result = main()

			# Assert: Check the result
			assert result == 0
			mock_app.assert_called_once()

	def test_module_import(self) -> None:
		"""
		Test that the module can be imported without errors.

		Verifies that the module exports the expected attributes and that app
		is a Typer instance.

		"""
		# Arrange/Act: Import the module
		import codemap.cli_app

		# Assert: Check that the module has the expected attributes
		assert hasattr(codemap.cli_app, "app")
		assert hasattr(codemap.cli_app, "main")

		# Check that app is a Typer instance
		assert str(type(codemap.cli_app.app)).endswith("typer.main.Typer'>")

	def test_cli_invoke(self) -> None:
		"""
		Test that the CLI can be invoked.

		A smoke test to ensure the CLI can be invoked with basic parameters.
		Validates that the help option works and returns a successful exit
		code.

		"""
		# Act: Invoke the CLI with help option
		result = self.invoke_command(["--help"])

		# Assert: Check the result
		assert result.exit_code == 0
		assert "Usage:" in result.stdout
