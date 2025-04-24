"""Tests for PR utilities."""

from __future__ import annotations

import json
import subprocess
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from codemap.utils.git_utils import GitError
from codemap.utils.pr_utils import (
    branch_exists,
    checkout_branch,
    create_branch,
    create_pull_request,
    generate_pr_description_from_commits,
    generate_pr_title_from_commits,
    get_commit_messages,
    get_current_branch,
    get_default_branch,
    get_existing_pr,
    push_branch,
    suggest_branch_name,
    update_pull_request,
)


@pytest.fixture
def mock_run_git_command() -> Generator[MagicMock, None, None]:
    """Mock the run_git_command function."""
    with patch("codemap.utils.pr_utils.run_git_command") as mock:
        yield mock


@pytest.fixture
def mock_subprocess_run() -> Generator[MagicMock, None, None]:
    """Mock subprocess.run."""
    with patch("subprocess.run") as mock:
        yield mock


@pytest.fixture
def mock_git_version_check(mock_subprocess_run: MagicMock) -> MagicMock:
    """Set up mock for gh CLI version check that passes."""
    mock_version_process = MagicMock()
    mock_version_process.returncode = 0
    mock_subprocess_run.return_value = mock_version_process
    return mock_subprocess_run


class TestBranchOperations:
    """Tests for branch-related operations."""

    def test_get_current_branch(self, mock_run_git_command: MagicMock) -> None:
        """Test get_current_branch function."""
        mock_run_git_command.return_value = "feature-branch\n"

        result = get_current_branch()

        mock_run_git_command.assert_called_once_with(["git", "branch", "--show-current"])
        assert result == "feature-branch"

    def test_get_current_branch_error(self, mock_run_git_command: MagicMock) -> None:
        """Test get_current_branch function when git command fails."""
        mock_run_git_command.side_effect = GitError("Command failed")

        with pytest.raises(GitError, match="Failed to get current branch"):
            get_current_branch()

    def test_get_default_branch_from_remote(self, mock_run_git_command: MagicMock) -> None:
        """Test get_default_branch function when remote info is available."""
        mock_run_git_command.side_effect = [
            "  Remote branch:\n    HEAD branch: main\n    Remote branches:",
            "origin/main\norigin/feature-branch",
        ]

        result = get_default_branch()

        assert result == "main"
        mock_run_git_command.assert_called_with(["git", "remote", "show", "origin"])

    def test_get_default_branch_fallback_to_main(self, mock_run_git_command: MagicMock) -> None:
        """Test get_default_branch function when falling back to main."""
        mock_run_git_command.side_effect = ["Remote info without HEAD branch", "origin/main\norigin/feature-branch"]

        result = get_default_branch()

        assert result == "main"
        assert mock_run_git_command.call_count == 2

    def test_get_default_branch_fallback_to_master(self, mock_run_git_command: MagicMock) -> None:
        """Test get_default_branch function when falling back to master."""
        mock_run_git_command.side_effect = ["Remote info without HEAD branch", "origin/master\norigin/feature-branch"]

        result = get_default_branch()

        assert result == "master"
        assert mock_run_git_command.call_count == 2

    def test_create_branch(self, mock_run_git_command: MagicMock) -> None:
        """Test create_branch function."""
        create_branch("feature-branch")

        mock_run_git_command.assert_called_once_with(["git", "checkout", "-b", "feature-branch"])

    def test_create_branch_error(self, mock_run_git_command: MagicMock) -> None:
        """Test create_branch function when git command fails."""
        mock_run_git_command.side_effect = GitError("Command failed")

        with pytest.raises(GitError, match="Failed to create branch: feature-branch"):
            create_branch("feature-branch")

    def test_checkout_branch(self, mock_run_git_command: MagicMock) -> None:
        """Test checkout_branch function."""
        checkout_branch("feature-branch")

        mock_run_git_command.assert_called_once_with(["git", "checkout", "feature-branch"])

    def test_checkout_branch_error(self, mock_run_git_command: MagicMock) -> None:
        """Test checkout_branch function when git command fails."""
        mock_run_git_command.side_effect = GitError("Command failed")

        with pytest.raises(GitError, match="Failed to checkout branch: feature-branch"):
            checkout_branch("feature-branch")

    def test_branch_exists_local(self, mock_run_git_command: MagicMock) -> None:
        """Test branch_exists function for local branch."""
        mock_run_git_command.return_value = "feature-branch"

        result = branch_exists("feature-branch")

        assert result is True
        mock_run_git_command.assert_called_once_with(["git", "branch", "--list", "feature-branch"])

    def test_branch_exists_remote(self, mock_run_git_command: MagicMock) -> None:
        """Test branch_exists function for remote branch."""
        mock_run_git_command.side_effect = ["", "origin/feature-branch"]

        result = branch_exists("feature-branch")

        assert result is True
        assert mock_run_git_command.call_count == 2

    def test_branch_not_exists(self, mock_run_git_command: MagicMock) -> None:
        """Test branch_exists function when branch doesn't exist."""
        mock_run_git_command.side_effect = ["", ""]

        result = branch_exists("feature-branch")

        assert result is False
        assert mock_run_git_command.call_count == 2

    def test_branch_exists_git_error(self, mock_run_git_command: MagicMock) -> None:
        """Test branch_exists function when git command fails."""
        mock_run_git_command.side_effect = GitError("Command failed")

        result = branch_exists("feature-branch")

        assert result is False

    def test_push_branch(self, mock_run_git_command: MagicMock) -> None:
        """Test push_branch function."""
        push_branch("feature-branch")

        mock_run_git_command.assert_called_once_with(["git", "push", "-u", "origin", "feature-branch"])

    def test_push_branch_force(self, mock_run_git_command: MagicMock) -> None:
        """Test push_branch function with force option."""
        push_branch("feature-branch", force=True)

        mock_run_git_command.assert_called_once_with(["git", "push", "--force", "-u", "origin", "feature-branch"])

    def test_push_branch_error(self, mock_run_git_command: MagicMock) -> None:
        """Test push_branch function when git command fails."""
        mock_run_git_command.side_effect = GitError("Command failed")

        with pytest.raises(GitError, match="Failed to push branch: feature-branch"):
            push_branch("feature-branch")


class TestCommitOperations:
    """Tests for commit-related operations."""

    def test_get_commit_messages(self, mock_run_git_command: MagicMock) -> None:
        """Test get_commit_messages function."""
        mock_run_git_command.return_value = "feat: Add feature\nfix: Fix bug"

        result = get_commit_messages("main", "feature-branch")

        mock_run_git_command.assert_called_once_with(["git", "log", "main..feature-branch", "--pretty=format:%s"])
        assert result == ["feat: Add feature", "fix: Fix bug"]

    def test_get_commit_messages_empty(self, mock_run_git_command: MagicMock) -> None:
        """Test get_commit_messages function with empty result."""
        mock_run_git_command.return_value = ""

        result = get_commit_messages("main", "feature-branch")

        assert result == []

    def test_get_commit_messages_error(self, mock_run_git_command: MagicMock) -> None:
        """Test get_commit_messages function when git command fails."""
        mock_run_git_command.side_effect = GitError("Command failed")

        with pytest.raises(GitError, match="Failed to get commit messages between main and feature-branch"):
            get_commit_messages("main", "feature-branch")

    def test_generate_pr_title_from_commits_empty(self) -> None:
        """Test generate_pr_title_from_commits function with empty commits."""
        result = generate_pr_title_from_commits([])

        assert result == "Update branch"

    def test_generate_pr_title_from_commits_feat(self) -> None:
        """Test generate_pr_title_from_commits function with feature commit."""
        result = generate_pr_title_from_commits(["feat: Add new feature"])

        assert result == "Feature: Add new feature"

    def test_generate_pr_title_from_commits_fix(self) -> None:
        """Test generate_pr_title_from_commits function with fix commit."""
        result = generate_pr_title_from_commits(["fix: Fix critical bug"])

        assert result == "Fix: Fix critical bug"

    def test_generate_pr_title_from_commits_with_scope(self) -> None:
        """Test generate_pr_title_from_commits function with scoped commit."""
        result = generate_pr_title_from_commits(["feat(api): Add new endpoint"])

        assert result == "Feature: Add new endpoint"

    def test_generate_pr_title_from_commits_fallback(self) -> None:
        """Test generate_pr_title_from_commits function with non-conventional commit."""
        result = generate_pr_title_from_commits(["Update documentation"])

        assert result == "Update documentation"

    def test_generate_pr_description_from_commits_empty(self) -> None:
        """Test generate_pr_description_from_commits function with empty commits."""
        result = generate_pr_description_from_commits([])

        assert result == "No changes"

    def test_generate_pr_description_from_commits(self) -> None:
        """Test generate_pr_description_from_commits function."""
        commits = [
            "feat: Add new feature",
            "fix: Fix critical bug",
            "docs: Update README",
            "refactor: Improve code structure",
            "perf: Optimize database queries",
            "chore: Update dependencies",
        ]

        result = generate_pr_description_from_commits(commits)

        # Check that the description contains all the expected sections
        assert "## What type of PR is this?" in result
        assert "- [x] Feature" in result
        assert "- [x] Bug Fix" in result
        assert "- [x] Documentation Update" in result
        assert "- [x] Refactor" in result
        assert "- [x] Optimization" in result
        assert "### Features" in result
        assert "- Add new feature" in result
        assert "### Fixes" in result
        assert "- Fix critical bug" in result
        assert "### Documentation" in result
        assert "- Update README" in result
        assert "### Refactors" in result
        assert "- Improve code structure" in result
        assert "### Optimizations" in result
        assert "- Optimize database queries" in result
        assert "### Other" in result
        assert "- Update dependencies" in result

    def test_suggest_branch_name(self) -> None:
        """Test suggest_branch_name function."""
        commits = ["feat(api): Add new endpoint"]

        result = suggest_branch_name(commits)

        assert result.startswith("feat-api-add-new-endpoint")

    def test_suggest_branch_name_empty(self) -> None:
        """Test suggest_branch_name function with empty commits."""
        with patch("codemap.utils.pr_utils.get_timestamp") as mock_get_timestamp:
            mock_get_timestamp.return_value = "20250421-123456"

            result = suggest_branch_name([])

            assert result == "update-20250421-123456"

    def test_suggest_branch_name_non_conventional(self) -> None:
        """Test suggest_branch_name function with non-conventional commit."""
        result = suggest_branch_name(["Update documentation and fix typos"])

        assert result == "update-update-documentation-and"


class TestPullRequestOperations:
    """Tests for pull request operations."""

    def test_create_pull_request(self, mock_git_version_check: MagicMock) -> None:
        """Test create_pull_request function."""
        # Set up mock for gh pr create
        mock_create_process = MagicMock()
        mock_create_process.stdout = "https://github.com/user/repo/pull/123"
        mock_create_process.returncode = 0

        mock_git_version_check.side_effect = [mock_git_version_check.return_value, mock_create_process]

        result = create_pull_request("main", "feature-branch", "PR Title", "PR Description")

        assert mock_git_version_check.call_count == 2

        # Check the gh pr create command
        create_call_args = mock_git_version_check.call_args_list[1][0][0]
        assert create_call_args[0:3] == ["gh", "pr", "create"]
        assert "--base" in create_call_args
        assert "main" in create_call_args
        assert "--head" in create_call_args
        assert "feature-branch" in create_call_args

        # Check the result
        assert result.branch == "feature-branch"
        assert result.title == "PR Title"
        assert result.description == "PR Description"
        assert result.url == "https://github.com/user/repo/pull/123"
        assert result.number == 123

    def test_create_pull_request_gh_not_installed(self, mock_subprocess_run: MagicMock) -> None:
        """Test create_pull_request function when gh CLI is not installed."""
        # Mock for gh --version check that fails
        mock_subprocess_run.side_effect = FileNotFoundError("No such file or directory: 'gh'")

        with pytest.raises(GitError, match="GitHub CLI \\(gh\\) is not installed or not in PATH"):
            create_pull_request("main", "feature-branch", "PR Title", "PR Description")

    def test_create_pull_request_error(self, mock_git_version_check: MagicMock) -> None:
        """Test create_pull_request function when gh command fails."""
        # Set up mock for gh pr create that fails
        mock_git_version_check.side_effect = [
            mock_git_version_check.return_value,
            subprocess.CalledProcessError(1, "gh pr create", stderr="Error creating PR"),
        ]

        with pytest.raises(GitError, match="Failed to create PR: Error creating PR"):
            create_pull_request("main", "feature-branch", "PR Title", "PR Description")

    def test_update_pull_request(self, mock_git_version_check: MagicMock) -> None:
        """Test update_pull_request function."""
        # Set up mocks for gh commands
        mock_edit_process = MagicMock()
        mock_edit_process.returncode = 0

        mock_view_process = MagicMock()
        mock_view_process.returncode = 0
        mock_view_process.stdout = "https://github.com/user/repo/pull/123"

        # Set up the side_effect to return different mock processes for each call
        mock_git_version_check.side_effect = [mock_git_version_check.return_value, mock_edit_process, mock_view_process]

        with patch("codemap.utils.pr_utils.get_current_branch") as mock_get_current_branch:
            mock_get_current_branch.return_value = "feature-branch"

            result = update_pull_request(123, "Updated Title", "Updated Description")

            # Verify the subprocess.run calls
            assert mock_git_version_check.call_count == 3

            # First call should be to check gh version
            version_call_args = mock_git_version_check.call_args_list[0][0][0]
            assert version_call_args[0:2] == ["gh", "--version"]

            # Second call should be to edit the PR
            edit_call_args = mock_git_version_check.call_args_list[1][0][0]
            assert edit_call_args[0:3] == ["gh", "pr", "edit"]
            assert edit_call_args[3] == "123"

            # Third call should be to get the PR URL
            view_call_args = mock_git_version_check.call_args_list[2][0][0]
            assert view_call_args[0:3] == ["gh", "pr", "view"]

            # Check the result
            assert result.branch == "feature-branch"
            assert result.title == "Updated Title"
            assert result.description == "Updated Description"
            assert result.url == "https://github.com/user/repo/pull/123"
            assert result.number == 123

    def test_update_pull_request_gh_not_installed(self, mock_subprocess_run: MagicMock) -> None:
        """Test update_pull_request function when gh CLI is not installed."""
        # Mock for gh --version check that fails
        mock_subprocess_run.side_effect = FileNotFoundError("No such file or directory: 'gh'")

        with pytest.raises(GitError, match="GitHub CLI \\(gh\\) is not installed or not in PATH"):
            update_pull_request(123, "Updated Title", "Updated Description")

    def test_update_pull_request_error(self, mock_git_version_check: MagicMock) -> None:
        """Test update_pull_request function when gh command fails."""
        # Mock for gh pr edit that fails
        mock_git_version_check.side_effect = [
            mock_git_version_check.return_value,
            subprocess.CalledProcessError(1, "gh pr edit", stderr="Error updating PR"),
        ]

        with patch("codemap.utils.pr_utils.get_current_branch") as mock_get_current_branch:
            mock_get_current_branch.return_value = "feature-branch"

            with pytest.raises(GitError, match="Failed to update PR: Error updating PR"):
                update_pull_request(123, "Updated Title", "Updated Description")

    def test_get_existing_pr(self, mock_git_version_check: MagicMock) -> None:
        """Test get_existing_pr function."""
        # Set up mock for gh pr list
        mock_list_process = MagicMock()
        mock_list_process.stdout = json.dumps(
            {
                "number": 123,
                "title": "PR Title",
                "body": "PR Description",
                "url": "https://github.com/user/repo/pull/123",
            }
        )
        mock_list_process.returncode = 0

        mock_git_version_check.side_effect = [mock_git_version_check.return_value, mock_list_process]

        result = get_existing_pr("feature-branch")

        assert mock_git_version_check.call_count == 2

        # Check the gh pr list command
        list_call_args = mock_git_version_check.call_args_list[1][0][0]
        assert list_call_args[0:3] == ["gh", "pr", "list"]
        assert "--head" in list_call_args
        assert "feature-branch" in list_call_args

        # Check the result
        assert result is not None
        assert result.branch == "feature-branch"
        assert result.title == "PR Title"
        assert result.description == "PR Description"
        assert result.url == "https://github.com/user/repo/pull/123"
        assert result.number == 123

    def test_get_existing_pr_not_found(self, mock_git_version_check: MagicMock) -> None:
        """Test get_existing_pr function when PR doesn't exist."""
        # Set up mock for gh pr list with no results
        mock_list_process = MagicMock()
        mock_list_process.stdout = ""
        mock_list_process.returncode = 1

        mock_git_version_check.side_effect = [mock_git_version_check.return_value, mock_list_process]

        result = get_existing_pr("feature-branch")

        assert result is None
