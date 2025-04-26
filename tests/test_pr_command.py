"""Tests for the PR command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

import codemap.cli_app
from codemap.utils.git_utils import GitDiff
from codemap.utils.pr_utils import PullRequest

if TYPE_CHECKING:
	from collections.abc import Generator

app = codemap.cli_app.app


@pytest.fixture
def mock_branch_operations() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock git branch operations."""
	with (
		patch("codemap.utils.pr_utils.get_current_branch") as mock_get_current_branch,
		patch("codemap.utils.pr_utils.get_default_branch") as mock_get_default_branch,
		patch("codemap.utils.pr_utils.branch_exists") as mock_branch_exists,
		patch("codemap.utils.pr_utils.create_branch") as mock_create_branch,
		patch("codemap.utils.pr_utils.checkout_branch") as mock_checkout_branch,
		patch("codemap.utils.pr_utils.push_branch") as mock_push_branch,
	):
		mock_get_current_branch.return_value = "feature-branch"
		mock_get_default_branch.return_value = "main"
		mock_branch_exists.return_value = False

		yield {
			"get_current_branch": mock_get_current_branch,
			"get_default_branch": mock_get_default_branch,
			"branch_exists": mock_branch_exists,
			"create_branch": mock_create_branch,
			"checkout_branch": mock_checkout_branch,
			"push_branch": mock_push_branch,
		}


@pytest.fixture
def mock_pr_operations() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock PR operations."""
	with (
		patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commit_messages,
		patch("codemap.cli.pr_cmd.create_pull_request") as mock_create_pull_request,
		patch("codemap.cli.pr_cmd.get_existing_pr") as mock_get_existing_pr,
		patch("codemap.cli.pr_cmd.update_pull_request") as mock_update_pull_request,
	):
		# Mock commit messages
		mock_get_commit_messages.return_value = ["feat: Add new feature", "fix: Fix bug"]

		# Mock PR creation
		mock_pr = PullRequest(
			branch="feature-branch",
			title="Add new feature",
			description="## Changes\n\n### Features\n\n- Add new feature\n\n### Fixes\n\n- Fix bug\n\n",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)
		mock_create_pull_request.return_value = mock_pr
		mock_update_pull_request.return_value = mock_pr
		mock_get_existing_pr.return_value = None

		yield {
			"get_commit_messages": mock_get_commit_messages,
			"create_pull_request": mock_create_pull_request,
			"get_existing_pr": mock_get_existing_pr,
			"update_pull_request": mock_update_pull_request,
		}


@pytest.fixture
def mock_git_diff_operations() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock git diff operations."""
	with (
		patch("codemap.cli.pr_cmd.validate_repo_path") as mock_validate_repo_path,
		patch("codemap.cli.pr_cmd.get_staged_diff") as mock_get_staged_diff,
		patch("codemap.cli.pr_cmd.get_unstaged_diff") as mock_get_unstaged_diff,
		patch("codemap.cli.pr_cmd.get_untracked_files") as mock_get_untracked_files,
	):
		mock_validate_repo_path.return_value = Path("/fake/repo")

		# Mock git utilities
		mock_staged_diff = GitDiff(
			files=["file1.py"],
			content="diff content for file1.py",
			is_staged=True,
		)
		mock_get_staged_diff.return_value = mock_staged_diff

		mock_unstaged_diff = GitDiff(
			files=["file2.py"],
			content="diff content for file2.py",
			is_staged=False,
		)
		mock_get_unstaged_diff.return_value = mock_unstaged_diff

		mock_get_untracked_files.return_value = ["file3.py"]

		yield {
			"validate_repo_path": mock_validate_repo_path,
			"get_staged_diff": mock_get_staged_diff,
			"get_unstaged_diff": mock_get_unstaged_diff,
			"get_untracked_files": mock_get_untracked_files,
		}


@pytest.fixture
def mock_diff_processing() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock diff processing operations."""
	with (
		patch("codemap.cli.pr_cmd.DiffSplitter") as mock_diff_splitter,
		patch("codemap.utils.llm_utils.create_universal_generator") as mock_create_universal_generator,
		patch("codemap.cli.pr_cmd.CommitCommand.process_all_chunks") as mock_process_all_chunks,
	):
		# Mock DiffSplitter
		mock_splitter = MagicMock()
		mock_chunk = MagicMock()
		mock_chunk.files = ["file1.py"]
		mock_splitter.split_diff.return_value = [mock_chunk]
		mock_diff_splitter.return_value = mock_splitter

		# Mock message generator
		mock_generator = MagicMock()
		mock_generator.generate_message.return_value = ("feat: Add new feature", True)
		mock_create_universal_generator.return_value = mock_generator

		# Mock process_all_chunks
		mock_process_all_chunks.return_value = 0

		yield {
			"diff_splitter": mock_diff_splitter,
			"create_universal_generator": mock_create_universal_generator,
			"process_all_chunks": mock_process_all_chunks,
		}


@pytest.fixture
def mock_user_input() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock user input operations."""
	with (
		patch("questionary.confirm") as mock_confirm,
		patch("questionary.text") as mock_text,
		patch("questionary.select") as mock_select,
	):
		# Mock questionary
		mock_confirm.return_value.ask.return_value = True
		mock_text.return_value.ask.return_value = "feature-branch"
		mock_select.return_value.ask.return_value = "commit"

		yield {
			"confirm": mock_confirm,
			"text": mock_text,
			"select": mock_select,
		}


@pytest.fixture
def mock_llm_config() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock LLM configuration loading."""
	with patch("codemap.cli.pr_cmd._load_llm_config") as mock_load_config:
		# Return a default configuration for testing
		mock_load_config.return_value = {
			"model": "gpt-3.5-turbo",
			"api_key": "test-api-key",
			"api_base": "https://api.example.com",
		}

		yield {
			"load_llm_config": mock_load_config,
		}


@pytest.fixture
def mock_subprocess() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock subprocess run calls."""
	with patch("codemap.cli.pr_cmd.subprocess.run") as mock_run:
		# Create a mock result for GitHub CLI commands
		mock_result = MagicMock()
		mock_result.returncode = 0
		mock_result.stdout = json.dumps(
			{
				"number": 42,
				"title": "Test PR",
				"body": "Test description",
				"headRefName": "feature-branch",
				"url": "https://github.com/user/repo/pull/42",
			}
		)

		# Set the return value of the mock
		mock_run.return_value = mock_result

		yield {"subprocess_run": mock_run}


@pytest.fixture
def mock_git_utils(
	mock_branch_operations: dict[str, MagicMock],
	mock_pr_operations: dict[str, MagicMock],
	mock_git_diff_operations: dict[str, MagicMock],
	mock_diff_processing: dict[str, MagicMock],
	mock_user_input: dict[str, MagicMock],
	mock_llm_config: dict[str, MagicMock],
	mock_subprocess: dict[str, MagicMock],
) -> dict[str, Any]:
	"""Combine all mock fixtures into one dictionary for convenience."""
	return {
		**mock_branch_operations,
		**mock_pr_operations,
		**mock_git_diff_operations,
		**mock_diff_processing,
		**mock_user_input,
		**mock_llm_config,
		**mock_subprocess,
	}


@pytest.fixture
def mock_exit_with_error() -> Generator[MagicMock, None, None]:
	"""Mock the _exit_with_error function."""
	with patch("codemap.cli.pr_cmd._exit_with_error") as mock_exit:
		# Instead of exiting, just capture the error message for debugging
		mock_exit.side_effect = lambda _msg, *_args, **_kwargs: None
		yield mock_exit


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.cli
@pytest.mark.skip(reason="PR command integration test needs deeper mocking strategy")
def test_pr_create_command(mock_git_utils: dict[str, Any], mock_exit_with_error: MagicMock) -> None:
	"""Test the PR create command."""
	# Test skipped - requires deeper mocking strategy


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.cli
@pytest.mark.skip(reason="PR command integration test needs deeper mocking strategy")
def test_pr_update_command(mock_git_utils: dict[str, Any], mock_exit_with_error: MagicMock) -> None:
	"""Test the PR update command."""
	# Test skipped - requires deeper mocking strategy
