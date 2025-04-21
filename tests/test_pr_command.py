"""Tests for the PR command."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from codemap.cli.main import app
from codemap.utils.pr_utils import PullRequest


@pytest.fixture
def mock_git_utils() -> dict[str, Any]:
    """Mock git utilities."""
    with patch("codemap.cli.pr.get_current_branch") as mock_get_current_branch, patch(
        "codemap.cli.pr.get_default_branch"
    ) as mock_get_default_branch, patch("codemap.cli.pr.branch_exists") as mock_branch_exists, patch(
        "codemap.cli.pr.create_branch"
    ) as mock_create_branch, patch("codemap.cli.pr.checkout_branch") as mock_checkout_branch, patch(
        "codemap.cli.pr.push_branch"
    ) as mock_push_branch, patch("codemap.cli.pr.get_commit_messages") as mock_get_commit_messages, patch(
        "codemap.cli.pr.create_pull_request"
    ) as mock_create_pull_request, patch("codemap.cli.pr.get_existing_pr") as mock_get_existing_pr, patch(
        "codemap.cli.pr.update_pull_request"
    ) as mock_update_pull_request, patch("codemap.cli.pr.validate_repo_path") as mock_validate_repo_path, patch(
        "codemap.cli.pr.GitWrapper"
    ) as mock_git_wrapper, patch("codemap.cli.pr.DiffSplitter") as mock_diff_splitter, patch(
        "codemap.cli.pr.setup_message_generator"
    ) as mock_setup_message_generator, patch("codemap.cli.pr.process_all_chunks") as mock_process_all_chunks, patch(
        "questionary.confirm"
    ) as mock_confirm, patch("questionary.text") as mock_text, patch("questionary.select") as mock_select:
        # Set up mock returns
        mock_get_current_branch.return_value = "feature-branch"
        mock_get_default_branch.return_value = "main"
        mock_branch_exists.return_value = False
        mock_validate_repo_path.return_value = Path("/fake/repo")

        # Mock GitWrapper
        mock_git = MagicMock()
        mock_diff = MagicMock()
        mock_diff.files = ["file1.py", "file2.py"]
        mock_git.get_uncommitted_changes.return_value = mock_diff
        mock_git_wrapper.return_value = mock_git

        # Mock DiffSplitter
        mock_splitter = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.files = ["file1.py"]
        mock_splitter.split_diff.return_value = [mock_chunk]
        mock_diff_splitter.return_value = mock_splitter

        # Mock message generator
        mock_generator = MagicMock()
        mock_generator.generate_message.return_value = ("feat: Add new feature", True)
        mock_setup_message_generator.return_value = mock_generator

        # Mock process_all_chunks
        mock_process_all_chunks.return_value = 0

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

        # Mock questionary
        mock_confirm.return_value.ask.return_value = True
        mock_text.return_value.ask.return_value = "feature-branch"
        mock_select.return_value.ask.return_value = "commit"

        yield {
            "get_current_branch": mock_get_current_branch,
            "get_default_branch": mock_get_default_branch,
            "branch_exists": mock_branch_exists,
            "create_branch": mock_create_branch,
            "checkout_branch": mock_checkout_branch,
            "push_branch": mock_push_branch,
            "get_commit_messages": mock_get_commit_messages,
            "create_pull_request": mock_create_pull_request,
            "get_existing_pr": mock_get_existing_pr,
            "update_pull_request": mock_update_pull_request,
            "validate_repo_path": mock_validate_repo_path,
            "git_wrapper": mock_git_wrapper,
            "diff_splitter": mock_diff_splitter,
            "setup_message_generator": mock_setup_message_generator,
            "process_all_chunks": mock_process_all_chunks,
            "confirm": mock_confirm,
            "text": mock_text,
            "select": mock_select,
        }


def test_pr_create_command() -> None:
    """Test the PR create command."""
    runner = CliRunner()

    with patch("codemap.cli.main.pr_create") as mock_create:
        mock_create.return_value = None  # Simulate successful execution

        # Run the command
        result = runner.invoke(
            app,
            ["pr", "create", "--branch", "feature-branch", "--non-interactive"],
        )

        # Check that the command was called
        assert result.exit_code == 0


def test_pr_update_command() -> None:
    """Test the PR update command."""
    runner = CliRunner()

    with patch("codemap.cli.main.pr_update") as mock_update:
        mock_update.return_value = None  # Simulate successful execution

        # Run the command
        result = runner.invoke(
            app,
            ["pr", "update", "1", "--non-interactive"],
        )

        # Check that the command was called
        assert result.exit_code == 0
