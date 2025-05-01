"""Tests for PR generator utility functions."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from codemap.git.pr_generator.schemas import PullRequest
from codemap.git.pr_generator.utils import (
	PRCreationError,
	checkout_branch,
	create_branch,
	create_pull_request,
	detect_branch_type,
	generate_pr_content_from_template,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	get_branch_relation,
	get_commit_messages,
	get_current_branch,
	get_existing_pr,
	list_branches,
	push_branch,
	suggest_branch_name,
	update_pull_request,
)
from codemap.git.utils import GitError
from codemap.llm import LLMClient
from tests.base import GitTestBase


@pytest.mark.unit
@pytest.mark.git
class TestPRUtilsBranchManagement(GitTestBase):
	"""Tests for PR generator branch management utility functions."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		self._patchers = []

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_get_current_branch(self, mock_run_git) -> None:
		"""Test getting the current branch name."""
		# Arrange
		mock_run_git.return_value = "feature-branch\n"

		# Act
		result = get_current_branch()

		# Assert
		assert result == "feature-branch"
		mock_run_git.assert_called_once_with(["git", "branch", "--show-current"])

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_get_current_branch_error(self, mock_run_git) -> None:
		"""Test error handling when getting the current branch fails."""
		# Arrange
		mock_run_git.side_effect = GitError("Command failed")

		# Act and Assert
		with pytest.raises(GitError, match="Failed to get current branch"):
			get_current_branch()

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_create_branch(self, mock_run_git) -> None:
		"""Test creating a new branch."""
		# Act
		create_branch("feature-branch")

		# Assert
		mock_run_git.assert_called_once_with(["git", "checkout", "-b", "feature-branch"])

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_create_branch_error(self, mock_run_git) -> None:
		"""Test error handling when creating a branch fails."""
		# Arrange
		mock_run_git.side_effect = GitError("Command failed")

		# Act and Assert
		with pytest.raises(GitError, match="Failed to create branch: feature-branch"):
			create_branch("feature-branch")

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_checkout_branch(self, mock_run_git) -> None:
		"""Test checking out an existing branch."""
		# Act
		checkout_branch("feature-branch")

		# Assert
		mock_run_git.assert_called_once_with(["git", "checkout", "feature-branch"])

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_checkout_branch_error(self, mock_run_git) -> None:
		"""Test error handling when checking out a branch fails."""
		# Arrange
		mock_run_git.side_effect = GitError("Command failed")

		# Act and Assert
		with pytest.raises(GitError, match="Failed to checkout branch: feature-branch"):
			checkout_branch("feature-branch")

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_push_branch(self, mock_run_git) -> None:
		"""Test pushing a branch to remote."""
		# Act
		push_branch("feature-branch")

		# Assert
		mock_run_git.assert_called_once_with(["git", "push", "-u", "origin", "feature-branch"])

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_push_branch_force(self, mock_run_git) -> None:
		"""Test force pushing a branch to remote."""
		# Act
		push_branch("feature-branch", force=True)

		# Assert
		mock_run_git.assert_called_once_with(["git", "push", "--force", "-u", "origin", "feature-branch"])

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_push_branch_error(self, mock_run_git) -> None:
		"""Test error handling when pushing a branch fails."""
		# Arrange
		mock_run_git.side_effect = GitError("Command failed")

		# Act and Assert
		with pytest.raises(GitError, match="Failed to push branch: feature-branch"):
			push_branch("feature-branch")

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_list_branches(self, mock_run_git) -> None:
		"""Test listing branches."""
		# Arrange
		mock_run_git.side_effect = [
			"  master\n* feature-branch\n  develop\n",  # Local branches
			"  origin/master\n  origin/feature-branch\n  origin/develop\n",  # Remote branches
		]

		# Act
		result = list_branches()

		# Assert
		assert set(result) == {"master", "feature-branch", "develop"}
		assert mock_run_git.call_count == 2


@pytest.mark.unit
@pytest.mark.git
class TestPRUtilsCommitOperations(GitTestBase):
	"""Tests for PR generator commit-related utility functions."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		self._patchers = []

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_get_commit_messages(self, mock_run_git) -> None:
		"""Test getting commit messages between branches."""
		# Arrange
		mock_run_git.return_value = "feat: Add feature\nfix: Fix bug\n"

		# Act
		result = get_commit_messages("main", "feature")

		# Assert
		assert result == ["feat: Add feature", "fix: Fix bug"]
		mock_run_git.assert_called_once_with(["git", "log", "main..feature", "--pretty=format:%s"])

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_get_commit_messages_empty(self, mock_run_git) -> None:
		"""Test getting commit messages with empty result."""
		# Arrange
		mock_run_git.return_value = ""

		# Act
		result = get_commit_messages("main", "feature")

		# Assert
		assert result == []
		mock_run_git.assert_called_once_with(["git", "log", "main..feature", "--pretty=format:%s"])

	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_get_commit_messages_error(self, mock_run_git) -> None:
		"""Test error handling when getting commit messages fails."""
		# Arrange
		mock_run_git.side_effect = GitError("Command failed")

		# Act and Assert
		with pytest.raises(GitError, match="Failed to get commit messages between main and feature"):
			get_commit_messages("main", "feature")

	def test_generate_pr_title_from_commits_empty(self) -> None:
		"""Test generating PR title with empty commits."""
		# Act
		result = generate_pr_title_from_commits([])

		# Assert
		assert result == "Update branch"

	def test_generate_pr_title_from_commits_conventional(self) -> None:
		"""Test generating PR title from conventional commits."""
		# Arrange
		commits = ["feat: Add authentication", "fix: Fix login issue"]

		# Act
		result = generate_pr_title_from_commits(commits)

		# Assert
		assert result == "Feature: Add authentication"
		# It should use the first commit and identify it as a feature

	def test_generate_pr_title_from_commits_non_conventional(self) -> None:
		"""Test generating PR title from non-conventional commits."""
		# Arrange
		commits = ["Add authentication", "Fix login issue"]

		# Act
		result = generate_pr_title_from_commits(commits)

		# Assert
		assert result == "Add authentication"
		# It should use the first commit as the title

	def test_generate_pr_title_from_commits_multiple(self) -> None:
		"""Test generating PR title from multiple commits with different types."""
		# Arrange
		commits = ["docs: Update readme", "feat: Add new feature", "fix: Bug fix"]

		# Act
		result = generate_pr_title_from_commits(commits)

		# Assert
		assert result == "Docs: Update readme"
		# It should use the first commit, which is a docs commit

	@patch("codemap.llm.create_client")
	def test_generate_pr_title_with_llm(self, mock_create_client) -> None:
		"""Test generating PR title with LLM."""
		# Arrange
		commits = ["feat: Add user authentication", "fix: Fix login form validation"]

		# Mock the LLMClient
		mock_client = MagicMock(spec=LLMClient)
		mock_client.generate_text.return_value = "Add user authentication feature"
		mock_create_client.return_value = mock_client

		# Act
		result = generate_pr_title_with_llm(commits)

		# Assert
		assert result == "Add user authentication feature"
		mock_create_client.assert_called_once()
		mock_client.generate_text.assert_called_once()
		# Ensure the provided commits are included in the prompt
		assert "user authentication" in mock_client.generate_text.call_args[1]["prompt"]

	def test_generate_pr_title_with_llm_empty_commits(self) -> None:
		"""Test generating PR title with LLM and empty commits."""
		# Act
		result = generate_pr_title_with_llm([])

		# Assert
		assert result == "Update branch"
		# Should return default title without calling LLM

	def test_generate_pr_description_from_commits_empty(self) -> None:
		"""Test generating PR description with empty commits."""
		# Act
		result = generate_pr_description_from_commits([])

		# Assert
		assert result == "No changes"

	def test_generate_pr_description_from_commits(self) -> None:
		"""Test generating PR description from commits."""
		# Arrange
		commits = [
			"feat: Add user authentication",
			"fix: Fix login form validation",
			"docs: Update API documentation",
		]

		# Act
		result = generate_pr_description_from_commits(commits)

		# Assert
		# Check that the result contains key elements that should be in the generated description
		assert "## What type of PR is this?" in result  # Updated header check
		assert "Add user authentication" in result
		assert "Fix login form validation" in result
		assert "Update API documentation" in result

		# Check for type checkboxes (since we have feat, fix, and docs commits)
		assert "- [x] Feature" in result
		assert "- [x] Bug Fix" in result
		assert "- [x] Documentation" in result

	@patch("codemap.llm.create_client")
	def test_generate_pr_description_with_llm(self, mock_create_client) -> None:
		"""Test generating PR description with LLM."""
		# Arrange
		commits = ["feat: Add user authentication", "fix: Fix login form validation"]

		# Mock the LLM client
		mock_client = MagicMock(spec=LLMClient)
		mock_client.generate_text.return_value = "This PR adds user authentication and fixes login form validation."
		mock_create_client.return_value = mock_client

		# Act
		result = generate_pr_description_with_llm(commits, mock_client)

		# Assert
		assert result == "This PR adds user authentication and fixes login form validation."
		mock_client.generate_text.assert_called_once()
		# Make sure the commits are included in the prompt
		assert "user authentication" in mock_client.generate_text.call_args[1]["prompt"]
		assert "login form validation" in mock_client.generate_text.call_args[1]["prompt"]

	def test_generate_pr_description_with_llm_empty_commits(self) -> None:
		"""Test generating PR description with LLM and empty commits."""
		# Act
		result = generate_pr_description_with_llm([])

		# Assert
		assert result == "No changes"
		# Should return default description without calling LLM


@pytest.mark.unit
@pytest.mark.git
class TestPRUtilsPullRequestOperations(GitTestBase):
	"""Tests for PR generator PR-related utility functions."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		self._patchers = []

	@patch("subprocess.run")
	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_create_pull_request(self, mock_run_git, mock_subprocess_run) -> None:
		"""Test creating a pull request."""
		# Arrange - Mock subprocess.run to return a successful result with a URL
		mock_process = MagicMock()
		mock_process.stdout = "https://github.com/user/repo/pull/1"
		mock_process.returncode = 0
		mock_subprocess_run.return_value = mock_process

		# Mock checking for gh CLI
		mock_run_git.return_value = "gh version 2.0.0"

		# Act
		result = create_pull_request("main", "feature", "Add feature", "Description")

		# Assert
		assert isinstance(result, PullRequest)
		assert result.branch == "feature"
		assert result.title == "Add feature"
		assert result.description == "Description"
		assert result.url == "https://github.com/user/repo/pull/1"
		assert result.number == 1

		# Verify gh CLI command was constructed correctly
		assert mock_subprocess_run.call_count == 2  # Expect two calls (version check + create)
		args = mock_subprocess_run.call_args[0][0]  # Check the last call's args
		assert args[0:3] == ["gh", "pr", "create"]
		assert args[3:5] == ["--base", "main"]
		assert args[5:7] == ["--head", "feature"]
		assert args[7:9] == ["--title", "Add feature"]
		assert args[9:11] == ["--body", "Description"]

	@patch("subprocess.run")
	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_create_pull_request_error(self, mock_run_git, mock_subprocess_run) -> None:
		"""Test error handling when creating a pull request fails."""
		# Arrange
		# Mock the gh --version check to succeed
		# Mock the gh pr create call to fail
		mock_subprocess_run.side_effect = [
			MagicMock(returncode=0),  # Simulate successful gh --version check
			subprocess.CalledProcessError(1, ["gh", "pr", "create"], stderr="Error: Failed to create PR"),
		]

		# Mock checking for gh CLI (this mock might be redundant now but keep for safety)
		mock_run_git.return_value = "gh version 2.0.0"

		# Act and Assert
		with pytest.raises(PRCreationError, match="Failed to create PR"):
			create_pull_request("main", "feature", "Add feature", "Description")

	@patch("subprocess.run")
	@patch("codemap.git.pr_generator.utils.get_current_branch")
	def test_update_pull_request(self, mock_get_current, mock_subprocess_run) -> None:
		"""Test updating a pull request."""
		# Arrange
		mock_get_current.return_value = "feature"

		# Mock subprocess.run calls
		# 1. gh --version check (succeeds)
		# 2. gh pr edit call (succeeds)
		# 3. gh pr view call (succeeds, returns URL)
		mock_edit_process = MagicMock(returncode=0)
		mock_view_process = MagicMock(stdout="https://github.com/user/repo/pull/1\n", returncode=0)
		mock_subprocess_run.side_effect = [
			MagicMock(returncode=0),  # gh --version
			mock_edit_process,  # gh pr edit
			mock_view_process,  # gh pr view ... url
		]

		# Act
		result = update_pull_request(1, "Updated title", "Updated description")

		# Assert
		assert isinstance(result, PullRequest)
		assert result.title == "Updated title"
		assert result.description == "Updated description"
		assert result.url == "https://github.com/user/repo/pull/1"  # Check against stripped URL
		assert result.number == 1

		# Verify gh CLI commands were constructed correctly
		assert mock_subprocess_run.call_count == 3
		edit_call_args = mock_subprocess_run.call_args_list[1][0][0]  # Second call is edit
		assert edit_call_args[0:3] == ["gh", "pr", "edit"]
		assert "1" in edit_call_args
		assert "--title" in edit_call_args
		assert "Updated title" in edit_call_args
		assert "--body" in edit_call_args
		assert "Updated description" in edit_call_args

		view_call_args = mock_subprocess_run.call_args_list[2][0][0]  # Third call is view
		assert view_call_args[0:3] == ["gh", "pr", "view"]
		assert "1" in view_call_args
		assert "--json" in view_call_args
		assert "url" in view_call_args

	@patch("subprocess.run")
	@patch("codemap.git.pr_generator.utils.get_current_branch")
	def test_update_pull_request_error(self, mock_get_current, mock_subprocess_run) -> None:
		"""Test error handling when updating a pull request fails."""
		# Arrange
		mock_get_current.return_value = "feature"

		# First call to check if gh CLI is installed succeeds
		mock_subprocess_run.side_effect = [
			MagicMock(),  # First call to check gh CLI succeeds
			# Second call to update PR raises error
			subprocess.CalledProcessError(1, ["gh", "pr", "edit"], stderr="Error: PR not found"),
		]

		# Act and Assert
		with pytest.raises(PRCreationError, match="Failed to update PR"):
			update_pull_request(1, "Updated title", "Updated description")

	@patch("subprocess.run")
	def test_get_existing_pr(self, mock_subprocess_run) -> None:
		"""Test getting an existing PR."""
		# Arrange
		mock_process = MagicMock()
		pr_data = {  # Should be a dictionary, not a list
			"number": 1,
			"title": "Feature PR",
			"body": "PR description",
			"url": "https://github.com/user/repo/pull/1",
		}
		mock_process.stdout = json.dumps(pr_data)  # Simulate output after jq .[0]
		mock_process.returncode = 0

		# Mock the gh --version check as well
		mock_subprocess_run.side_effect = [
			MagicMock(returncode=0),  # gh --version check
			mock_process,  # gh pr list call
		]

		# Act
		result = get_existing_pr("feature")

		# Assert
		assert result is not None
		assert result.number == 1
		assert result.title == "Feature PR"
		assert result.description == "PR description"
		assert result.branch == "feature"

		# Verify gh CLI command was constructed correctly
		assert mock_subprocess_run.call_count == 2  # version check + list
		args = mock_subprocess_run.call_args_list[1][0][0]  # Check the second call
		assert args[0:3] == ["gh", "pr", "list"]
		assert "--head" in args
		assert "feature" in args
		assert "--json" in args

	@patch("subprocess.run")
	def test_get_existing_pr_not_found(self, mock_subprocess_run) -> None:
		"""Test getting a non-existent PR."""
		# Arrange - Mock subprocess.run to raise CalledProcessError on the list call
		mock_list_process = MagicMock(stdout="null", returncode=0)  # Simulate jq .[0] on empty list
		mock_subprocess_run.side_effect = [
			MagicMock(returncode=0),  # gh --version
			mock_list_process,
		]

		# Act
		result = get_existing_pr("feature")

		# Assert
		assert result is None

		# Verify gh CLI command was called
		assert mock_subprocess_run.call_count == 2

	@patch("subprocess.run")
	def test_get_existing_pr_error(self, mock_subprocess_run) -> None:
		"""Test error handling when getting an existing PR fails."""
		# Arrange - Mock subprocess.run to raise CalledProcessError on the list call
		mock_subprocess_run.side_effect = [
			MagicMock(returncode=0),  # gh --version check succeeds
			subprocess.CalledProcessError(1, ["gh", "pr", "list"], stderr="Error: Authentication failed"),
		]

		# Act
		result = get_existing_pr("feature")  # Should catch the error and return None

		# Assert
		assert result is None  # Function should return None on error, not raise GitError
		assert mock_subprocess_run.call_count == 2


@pytest.mark.unit
@pytest.mark.git
class TestPRUtilsMiscOperations(GitTestBase):
	"""Tests for miscellaneous PR generator utility functions."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		self._patchers = []

	@patch("codemap.git.pr_generator.utils.create_strategy")
	def test_generate_pr_content_from_template(self, mock_create_strategy) -> None:
		"""Test generating PR content from a template."""
		# Arrange
		mock_strategy = MagicMock()
		mock_strategy.detect_branch_type.return_value = "feature"  # Mock branch detection
		# Mock get_pr_templates to return actual template strings
		mock_strategy.get_pr_templates.return_value = {
			"title": "Feature: {description}",
			"description": "This PR implements: {description}\nBranch: {branch_name}",
		}
		mock_create_strategy.return_value = mock_strategy

		# Act
		result = generate_pr_content_from_template("feature/auth", "Add user authentication", "github-flow")

		# Assert
		assert result["title"] == "Feature: Add user authentication"  # Check formatted title
		assert "This PR implements: Add user authentication" in result["description"]
		assert "Branch: feature/auth" in result["description"]
		mock_create_strategy.assert_called_once_with("github-flow")
		mock_strategy.detect_branch_type.assert_called_once_with("feature/auth")
		mock_strategy.get_pr_templates.assert_called_once_with("feature")  # Check correct branch type used

	@patch("codemap.git.pr_generator.utils.create_strategy")
	def test_suggest_branch_name(self, mock_create_strategy) -> None:
		"""Test suggesting a branch name based on description."""
		# Arrange
		mock_strategy = MagicMock()
		mock_strategy.suggest_branch_name.return_value = "feature/auth"
		mock_create_strategy.return_value = mock_strategy

		# Act
		result = suggest_branch_name("Add user authentication", "github-flow")

		# Assert
		assert result == "feature/auth"
		mock_create_strategy.assert_called_once_with("github-flow")
		# Check that suggest_branch_name was called with determined type and cleaned message
		mock_strategy.suggest_branch_name.assert_called_once_with("feature", "Add user authentication")

	@patch("codemap.git.pr_generator.utils.branch_exists")
	@patch("codemap.git.pr_generator.utils.run_git_command")
	def test_get_branch_relation(self, mock_run_git, mock_branch_exists) -> None:
		"""Test getting the relationship between branches."""
		# Arrange
		# Mock branch_exists to return True for both branches
		mock_branch_exists.return_value = True

		# Mock run_git_command side effects for the different git commands
		# 1. git merge-base --is-ancestor feature main (fails -> GitError)
		# 2. git merge-base --is-ancestor main feature (succeeds -> return value doesn't matter)
		# 3. git rev-list --count feature..main (returns "0\n")
		mock_run_git.side_effect = [
			GitError("is-ancestor failed"),  # First is_ancestor call
			"",  # Second is_ancestor call (success)
			"0\n",  # rev-list count call
		]

		# Act
		is_ancestor, commit_count = get_branch_relation("feature", "main")

		# Assert based on the function's return value (is_ancestor, count)
		assert is_ancestor is False  # 'feature' is not ancestor of 'main'
		assert commit_count == 0  # Commits in 'main' not in 'feature'

		# Verify calls
		assert mock_branch_exists.call_count == 2  # Called for local=True for both branches
		assert mock_run_git.call_count == 3
		calls = mock_run_git.call_args_list
		assert "--is-ancestor" in calls[0][0][0]
		assert "feature" in calls[0][0][0]
		assert "main" in calls[0][0][0]
		assert "--is-ancestor" in calls[1][0][0]
		assert "main" in calls[1][0][0]
		assert "feature" in calls[1][0][0]
		assert "rev-list" in calls[2][0][0]
		assert "--count" in calls[2][0][0]
		assert "feature..main" in calls[2][0][0]

	@patch("codemap.git.pr_generator.utils.create_strategy")
	def test_detect_branch_type(self, mock_create_strategy) -> None:
		"""Test detecting branch type from name."""
		# Arrange
		mock_strategy = MagicMock()
		mock_strategy.detect_branch_type.return_value = "feature"
		mock_create_strategy.return_value = mock_strategy

		# Act
		result = detect_branch_type("feature/auth", "github-flow")

		# Assert
		assert result == "feature"
		mock_create_strategy.assert_called_once_with("github-flow")
		mock_strategy.detect_branch_type.assert_called_once_with("feature/auth")
