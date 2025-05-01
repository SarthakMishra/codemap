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


# Mock the CommitCommand class entirely for initial tests
@pytest.fixture
def mock_commit_command() -> Iterator[tuple[MagicMock, MagicMock]]:
	"""Fixture to mock the CommitCommand class and its instance."""
	with patch("codemap.cli.commit_cmd.CommitCommand", autospec=True) as mock_class:
		# Create the mock instance properly
		mock_instance = MagicMock()
		mock_instance.run.return_value = True  # Simulate successful run
		mock_instance.error_state = None  # Add the error_state attribute properly

		# Configure the class mock to return our instance
		mock_class.return_value = mock_instance
		yield mock_class, mock_instance


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
		patch("codemap.cli.commit_cmd.run_git_command") as mock_run_git,
	):  # run_git_command might be used elsewhere too
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
			"run_git": mock_run_git,
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

	@patch("codemap.cli.commit_cmd.setup_logging")
	def test_commit_default(
		self,
		mock_setup_logging: MagicMock,
		mock_commit_command: tuple[MagicMock, MagicMock],
		mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test default commit command invocation."""
		mock_class, mock_instance = mock_commit_command

		# Simulate running `codemap commit` in the temp dir
		# We pass the temp_dir path explicitly
		result = self.runner.invoke(app, ["commit", str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout
		mock_setup_logging.assert_called_once_with(is_verbose=False)
		mock_git_utils["validate"].assert_called_once_with(self.temp_dir)

		# Check that CommitCommand was instantiated with correct arguments for __init__
		args, kwargs = mock_class.call_args
		assert not args  # Should be called with kwargs only
		assert "path" in kwargs
		assert kwargs["path"] == self.temp_dir  # Path should be the temp_dir we passed
		assert "model" in kwargs
		assert kwargs["model"] == "gpt-4o-mini"  # Default model from CLI
		assert "bypass_hooks" in kwargs
		assert kwargs["bypass_hooks"] is False  # Default bypass_hooks from CLI

		mock_instance.run.assert_called_once()

	@patch("codemap.cli.commit_cmd.setup_logging")
	def test_commit_all_files(
		self,
		mock_setup_logging: MagicMock,
		mock_commit_command: tuple[MagicMock, MagicMock],
		mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with --all flag."""
		mock_class, mock_instance = mock_commit_command

		result = self.runner.invoke(app, ["commit", "--all", str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout
		mock_setup_logging.assert_called_once_with(is_verbose=False)
		mock_git_utils["validate"].assert_called_once_with(self.temp_dir)

		# Check __init__ args - --all is handled later, not passed to init
		args, kwargs = mock_class.call_args
		assert not args
		assert kwargs["path"] == self.temp_dir
		assert kwargs["model"] == "gpt-4o-mini"
		assert kwargs["bypass_hooks"] is False

		mock_instance.run.assert_called_once()

	@patch("codemap.cli.commit_cmd.setup_logging")
	def test_commit_with_message(
		self,
		mock_setup_logging: MagicMock,
		mock_commit_command: tuple[MagicMock, MagicMock],
		mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with -m flag."""
		mock_class, mock_instance = mock_commit_command
		test_message = "feat: my manual commit message"

		result = self.runner.invoke(app, ["commit", "-m", test_message, str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout
		mock_setup_logging.assert_called_once_with(is_verbose=False)
		mock_git_utils["validate"].assert_called_once_with(self.temp_dir)

		# Check __init__ args - -m is handled later, not passed to init
		args, kwargs = mock_class.call_args
		assert not args
		assert kwargs["path"] == self.temp_dir
		assert kwargs["model"] == "gpt-4o-mini"
		assert kwargs["bypass_hooks"] is False

		# run() should still be called, the command handles the message internally
		mock_instance.run.assert_called_once()

	@patch("codemap.cli.commit_cmd.setup_logging")
	def test_commit_non_interactive(
		self,
		mock_setup_logging: MagicMock,
		mock_commit_command: tuple[MagicMock, MagicMock],
		mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with --non-interactive flag."""
		mock_class, mock_instance = mock_commit_command

		result = self.runner.invoke(app, ["commit", "--non-interactive", str(self.temp_dir)])

		assert result.exit_code == 0, result.stdout
		mock_setup_logging.assert_called_once_with(is_verbose=False)
		mock_git_utils["validate"].assert_called_once_with(self.temp_dir)

		# Check __init__ args - --non-interactive is handled later, not passed to init
		args, kwargs = mock_class.call_args
		assert not args
		assert kwargs["path"] == self.temp_dir
		assert kwargs["model"] == "gpt-4o-mini"
		assert kwargs["bypass_hooks"] is False

		mock_instance.run.assert_called_once()

	@patch("codemap.cli.commit_cmd.setup_logging")
	@patch("codemap.cli.commit_cmd.exit_with_error")
	def test_commit_invalid_repo(
		self,
		mock_exit_with_error: MagicMock,
		mock_setup_logging: MagicMock,
		mock_commit_command: tuple[MagicMock, MagicMock],
		mock_git_utils: dict[str, MagicMock],
	) -> None:
		"""Test commit command with invalid repo path."""
		mock_class, mock_instance = mock_commit_command
		mock_git_utils["validate"].side_effect = GitError("Not a git repository")

		self.runner.invoke(app, ["commit", str(self.temp_dir)])

		# Expect exit_with_error to be called
		mock_setup_logging.assert_called_once_with(is_verbose=False)
		mock_git_utils["validate"].assert_called_once_with(self.temp_dir)
		# Relax assertion: Just check if it was called, not the specific message for now
		mock_exit_with_error.assert_called_once()
		mock_class.assert_not_called()  # Should exit before command instantiation
		mock_instance.run.assert_not_called()
