"""Tests for the __main__ module."""

import importlib
import runpy
from unittest.mock import patch

import pytest


@pytest.mark.unit
@pytest.mark.core
class TestMainModule:
	"""Test cases for the __main__ module functionality."""

	def test_main_module_execution(self) -> None:
		"""Test that running the module as __main__ calls the app."""
		with patch("codemap.cli.app") as mock_app:
			# Use runpy to run the module as __main__
			runpy.run_module("codemap.__main__", run_name="__main__")

			# Verify app was called exactly once
			mock_app.assert_called_once()

	def test_main_import_no_execution(self) -> None:
		"""Test that importing the module doesn't execute the app function."""
		with patch("codemap.cli.app") as mock_app:
			# Import the module normally
			try:
				# Use importlib to ensure fresh import
				importlib.import_module("codemap.__main__")
			except ImportError as e:
				pytest.fail(f"Failed to import __main__ module: {e}")

			# Verify app was not called
			mock_app.assert_not_called()
