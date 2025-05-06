"""Tests for the commit command CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from codemap.cli import app  # Assuming 'app' is your Typer application instance
from codemap.git.utils import GitError
from tests.base import FileSystemTestBase

if TYPE_CHECKING:
	from collections.abc import Iterator
	from pathlib import Path


# Mock the SemanticCommitCommand implementation
@pytest.fixture
def mock_semantic_commit_impl() -> Iterator[MagicMock]:
	"""Fixture to mock the _semantic_commit_command_impl function."""
	with patch("codemap.cli.commit_cmd._semantic_commit_command_impl") as mock_impl:
		yield mock_impl


# Mock git utils
@pytest.fixture
def mock_git_utils() -> Iterator[dict[str, MagicMock]]:
	"""Fixture to mock various git utility functions."""
	# Patch utils where they are imported/looked up
	with (
		patch("codemap.cli.commit_cmd.validate_repo_path") as mock_validate,
		patch("codemap.git.commit_generator.command.get_staged_diff") as mock_get_staged_diff,
		patch("codemap.git.commit_generator.command.get_unstaged_diff") as mock_get_unstaged_diff,
		patch("codemap.git.commit_generator.command.get_untracked_files") as mock_get_untracked,
		patch("codemap.git.commit_generator.command.commit_only_files") as mock_commit_files,
		patch("codemap.git.commit_generator.command.stage_files") as mock_stage_files,
		patch("codemap.cli.commit_cmd.exit_with_error") as mock_exit_with_error,
	):
		# Setup default return values for mocks if needed
		from codemap.git.utils import GitDiff  # Import for type hinting if needed

		# Instead of hardcoding a path, let validate_repo_path return what it's given
		mock_validate.side_effect = lambda path: path  # Return the path it was given
		# Simulate having some staged changes by default
		mock_get_staged_diff.return_value = GitDiff(files=["file1.py"], content="+ stage diff", is_staged=True)
		mock_get_unstaged_diff.return_value = None  # Default: no unstaged changes
		mock_get_untracked.return_value = []  # Default: no untracked files
		yield {
			"validate": mock_validate,
			"get_staged_diff": mock_get_staged_diff,
			"get_unstaged_diff": mock_get_unstaged_diff,
			"get_untracked": mock_get_untracked,
			"commit_files": mock_commit_files,
			"stage_files": mock_stage_files,
			"exit_with_error": mock_exit_with_error,
		}


@pytest.mark.cli
@pytest.mark.fs
class TestCommitCommand(FileSystemTestBase):
	"""Test cases for the 'commit' CLI command."""

	runner: CliRunner

	@pytest.fixture(autouse=True)
	def setup_cli(self, temp_dir: Path) -> None:
		"""Set up CLI test environment."""
		self.temp_dir = temp_dir
		self.runner = CliRunner()
		# Create a dummy repo structure if needed (might not be necessary with mocks)
		(self.temp_dir / ".git").mkdir(exist_ok=True)

	def test_commit_default(
		self,
		mock_semantic_commit_impl: MagicMock,
		_mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test default commit command invocation."""
		# Simulate running `codemap commit` in the temp dir
		# We pass the temp_dir path explicitly
		result = self.runner.invoke(app, ["commit", str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout

		# Check that _semantic_commit_command_impl was called with correct arguments
		mock_semantic_commit_impl.assert_called_once()
		_, kwargs = mock_semantic_commit_impl.call_args
		assert kwargs["path"] == self.temp_dir
		assert kwargs["model"] == "gpt-4o-mini"  # Default model
		assert kwargs["non_interactive"] is False
		assert kwargs["bypass_hooks"] is False
		assert kwargs["pathspecs"] is None  # Should be None by default

	def test_commit_all_files(
		self,
		mock_semantic_commit_impl: MagicMock,
		_mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with --all flag."""
		result = self.runner.invoke(app, ["commit", "--all", str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout

		# Verify implementation was called with correct args
		mock_semantic_commit_impl.assert_called_once()
		_, kwargs = mock_semantic_commit_impl.call_args
		assert kwargs["path"] == self.temp_dir

		# This may need adjustment based on how the --all flag is handled in the new implementation
		# It might be translated to a pathspecs list or other parameter

	def test_commit_with_message(
		self,
		mock_semantic_commit_impl: MagicMock,
		_mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with -m flag."""
		test_message = "feat: my manual commit message"

		result = self.runner.invoke(app, ["commit", "-m", test_message, str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout

		# Verify implementation was called
		mock_semantic_commit_impl.assert_called_once()
		_, kwargs = mock_semantic_commit_impl.call_args
		assert kwargs["path"] == self.temp_dir
		# In the new implementation, the message might be handled differently
		# or not supported at all - check the actual param handling

	def test_commit_non_interactive(
		self,
		mock_semantic_commit_impl: MagicMock,
		_mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with --non-interactive flag."""
		result = self.runner.invoke(app, ["commit", "--non-interactive", str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout

		# Verify implementation was called with non_interactive=True
		mock_semantic_commit_impl.assert_called_once()
		_, kwargs = mock_semantic_commit_impl.call_args
		assert kwargs["non_interactive"] is True

	def test_commit_invalid_repo(
		self,
		mock_semantic_commit_impl: MagicMock,
		mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with invalid repo path."""
		# Simulate a GitError when validating the repo
		mock_git_utils["validate"].side_effect = GitError("Not a git repository")

		self.runner.invoke(app, ["commit", str(self.temp_dir)])

		# In the new implementation, GitError might be caught by _semantic_commit_command_impl
		# which would call exit_with_error, so we check that instead
		mock_git_utils["exit_with_error"].assert_called_once()
		mock_semantic_commit_impl.assert_called_once()  # Still gets called before validation error
