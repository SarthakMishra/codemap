"""Tests for PR command CLI."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

if TYPE_CHECKING:
	from collections.abc import Iterator


@pytest.fixture
def mock_pr_command_impl() -> Iterator[AsyncMock]:
	"""Fixture to mock the _pr_command_impl function."""
	with patch("codemap.cli.pr_cmd._pr_command_impl") as mock_impl:
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
class TestPRCommandModule:
	"""Test the PR command module structure."""

	def test_pr_command_structure(self) -> None:
		"""Test that the PR command module has the expected structure."""
		# Import the module
		pr_cmd = importlib.import_module("codemap.cli.pr_cmd")

		# Check that the key functions exist
		assert hasattr(pr_cmd, "register_command"), "register_command function is missing"
		assert hasattr(pr_cmd, "_pr_command_impl"), "Implementation function is missing"
		assert hasattr(pr_cmd, "PRAction"), "PRAction enum is missing"

		# Check that the command annotations are defined
		assert hasattr(pr_cmd, "ActionArg"), "ActionArg annotation is missing"
		assert hasattr(pr_cmd, "BranchNameOpt"), "BranchNameOpt annotation is missing"
		assert hasattr(pr_cmd, "BaseBranchOpt"), "BaseBranchOpt annotation is missing"
		assert hasattr(pr_cmd, "NonInteractiveOpt"), "NonInteractiveOpt annotation is missing"
		assert hasattr(pr_cmd, "BypassHooksFlag"), "BypassHooksFlag annotation is missing"

	def test_pr_command_impl_signature(self) -> None:
		"""Test the signature of the _pr_command_impl function."""
		# Ensure the function has the correct signature
		from inspect import iscoroutinefunction, signature

		from codemap.cli.pr_cmd import _pr_command_impl

		# Check if it's an async function directly
		assert iscoroutinefunction(_pr_command_impl), "Should be an async function"

		sig = signature(_pr_command_impl)

		# Check parameter names and defaults
		parameters = sig.parameters
		assert "action" in parameters, "action parameter is missing"
		assert "branch_name" in parameters, "branch_name parameter is missing"
		assert "branch_type" in parameters, "branch_type parameter is missing"
		assert "base_branch" in parameters, "base_branch parameter is missing"
		assert "non_interactive" in parameters, "non_interactive parameter is missing"
		assert "bypass_hooks" in parameters, "bypass_hooks parameter is missing"

	def test_validate_workflow_strategy(self) -> None:
		"""Test the validate_workflow_strategy function."""
		from codemap.cli.pr_cmd import validate_workflow_strategy

		# Test valid strategies
		assert validate_workflow_strategy("github-flow") == "github-flow"
		assert validate_workflow_strategy("gitflow") == "gitflow"
		assert validate_workflow_strategy("trunk-based") == "trunk-based"

		# Test None is allowed
		assert validate_workflow_strategy(None) is None

		# Test invalid strategy raises error
		with pytest.raises(typer.BadParameter, match="Invalid workflow strategy"):
			validate_workflow_strategy("invalid-strategy")
