"""Tests for commit command CLI."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
	from collections.abc import Iterator


@pytest.fixture
def mock_semantic_commit_impl() -> Iterator[AsyncMock]:
	"""Fixture to mock the _semantic_commit_command_impl function."""
	with patch("codemap.cli.commit_cmd._semantic_commit_command_impl") as mock_impl:
		# Use AsyncMock since it's an async function
		mock_impl.return_value = None
		yield mock_impl


@pytest.fixture
def mock_git_utils() -> Iterator[dict[str, MagicMock]]:
	"""Fixture to mock git utility functions."""
	mocks = {}

	# Setup common mocks
	with (
		patch("codemap.git.utils.validate_repo_path") as mock_validate,
		patch("codemap.utils.cli_utils.exit_with_error") as mock_exit_with_error,
	):
		mocks["validate"] = mock_validate
		mocks["exit_with_error"] = mock_exit_with_error

		yield mocks


@pytest.mark.unit
@pytest.mark.cli
class TestCommitCommandModule:
	"""Test the commit command module structure."""

	def test_commit_command_structure(self) -> None:
		"""Test that the commit command module has the expected structure."""
		# Import the module
		commit_cmd = importlib.import_module("codemap.cli.commit_cmd")

		# Check that the key functions exist
		assert hasattr(commit_cmd, "register_command"), "register_command function is missing"
		assert hasattr(commit_cmd, "_semantic_commit_command_impl"), "Implementation function is missing"

		# Check that the command annotations are defined
		assert hasattr(commit_cmd, "NonInteractiveFlag"), "NonInteractiveFlag annotation is missing"
		assert hasattr(commit_cmd, "BypassHooksFlag"), "BypassHooksFlag annotation is missing"

	def test_semantic_commit_command_impl_signature(self) -> None:
		"""Test the signature of the _semantic_commit_command_impl function."""
		# Ensure the function has the correct signature
		from inspect import iscoroutinefunction, signature

		from codemap.cli.commit_cmd import _semantic_commit_command_impl

		# Check if it's an async function directly
		assert iscoroutinefunction(_semantic_commit_command_impl), "Should be an async function"

		sig = signature(_semantic_commit_command_impl)

		# Check parameter names and defaults
		parameters = sig.parameters
		assert "non_interactive" in parameters, "non_interactive parameter is missing"
		assert "bypass_hooks" in parameters, "bypass_hooks parameter is missing"
		assert "pathspecs" in parameters, "pathspecs parameter is missing"
		assert parameters["pathspecs"].default is None, "pathspecs should default to None"
