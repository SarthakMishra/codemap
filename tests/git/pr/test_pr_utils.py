"""Tests for PR generator utility functions."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from pygit2 import Commit
from pygit2 import GitError as Pygit2GitError
from pygit2.enums import SortMode
from pygit2.repository import Repository

from codemap.config import ConfigLoader
from codemap.git.pr_generator.pr_git_utils import PRGitUtils
from codemap.git.pr_generator.schemas import PullRequest
from codemap.git.pr_generator.utils import (
	PRCreationError,
	create_pull_request,
	detect_branch_type,
	generate_pr_content_from_template,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	get_existing_pr,
	suggest_branch_name,
	update_pull_request,
)
from codemap.git.utils import GitError
from codemap.llm.client import LLMClient
from tests.base import GitTestBase


@pytest.mark.unit
@pytest.mark.git
class TestPRUtilsBranchManagement(GitTestBase):
	"""Tests for PR generator branch management utility functions."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		super().setup_method()  # type: ignore[misc] # pylint: disable=no-member
		self.pgu = PRGitUtils()
		self.pgu.repo = MagicMock(spec=Repository)
		self.repo = self.pgu.repo
		self._patchers = []

	def test_get_current_branch(self) -> None:
		"""Test getting the current branch name using PRGitUtils."""
		# Arrange
		mock_head_attribute = MagicMock()
		mock_head_attribute.shorthand = "feature-branch"
		self.repo.configure_mock(head_is_detached=False, head=mock_head_attribute)

		# Act
		result = self.pgu.get_current_branch()

		# Assert
		assert result == "feature-branch"

	def test_get_current_branch_detached_head(self) -> None:
		"""Test getting current branch when HEAD is detached."""
		# Arrange
		mock_head_attribute = MagicMock()
		mock_revparse_result = MagicMock()
		mock_peeled_commit = MagicMock(spec=Commit)
		mock_peeled_commit.short_id = "abcdef0"
		mock_revparse_result.peel.return_value = mock_peeled_commit

		self.repo.configure_mock(
			head_is_detached=True,
			head=mock_head_attribute,
			revparse_single=MagicMock(return_value=mock_revparse_result),
		)

		# Act
		result = self.pgu.get_current_branch()

		# Assert
		assert result == "abcdef0"

	def test_get_current_branch_error(self) -> None:
		"""Test error handling when getting the current branch fails via PRGitUtils."""
		# Arrange
		self.repo.configure_mock(head=MagicMock(side_effect=Pygit2GitError("Pygit2 error")))

		# Act and Assert
		with pytest.raises(GitError, match="Failed to get current branch"):
			self.pgu.get_current_branch()

	def test_create_branch(self) -> None:
		"""Test creating a new branch using PRGitUtils."""
		# Arrange
		mock_head_peel = MagicMock(spec=Commit)
		mock_head_peel.id = "mock_head_commit_id"

		mock_head_ref_for_peel = MagicMock()
		mock_head_ref_for_peel.peel.return_value = mock_head_peel

		mock_create_branch_method = MagicMock()
		mock_lookup_ref_method = MagicMock()
		mock_checkout_method = MagicMock()

		self.repo.configure_mock(
			head_is_unborn=False,
			head=mock_head_ref_for_peel,
			create_branch=mock_create_branch_method,
			lookup_reference=mock_lookup_ref_method,
			checkout=mock_checkout_method,
		)
		mock_checkout_ref_obj = MagicMock()
		mock_lookup_ref_method.return_value = mock_checkout_ref_obj

		# Act
		self.pgu.create_branch("feature-branch")

		# Assert
		mock_create_branch_method.assert_called_once_with("feature-branch", mock_head_peel)
		mock_lookup_ref_method.assert_called_with("refs/heads/feature-branch")
		mock_checkout_method.assert_called_once_with(mock_checkout_ref_obj)

	def test_create_branch_from_reference(self) -> None:
		"""Test creating a new branch from a specific reference."""
		# Arrange
		mock_ref_commit = MagicMock(spec=Commit)
		mock_ref_commit.id = "ref_commit_id"
		mock_revparse_obj = MagicMock()
		mock_revparse_obj.peel.return_value = mock_ref_commit

		mock_revparse_single_method = MagicMock(return_value=mock_revparse_obj)
		mock_create_branch_method = MagicMock()
		mock_lookup_ref_method = MagicMock()
		mock_checkout_method = MagicMock()

		self.repo.configure_mock(
			revparse_single=mock_revparse_single_method,
			create_branch=mock_create_branch_method,
			lookup_reference=mock_lookup_ref_method,
			checkout=mock_checkout_method,
		)
		mock_checkout_ref_obj = MagicMock()
		mock_lookup_ref_method.return_value = mock_checkout_ref_obj

		# Act
		self.pgu.create_branch("new-feature", from_reference="main")

		# Assert
		mock_revparse_single_method.assert_called_once_with("main")
		mock_create_branch_method.assert_called_once_with("new-feature", mock_ref_commit)
		mock_lookup_ref_method.assert_called_with("refs/heads/new-feature")
		mock_checkout_method.assert_called_once_with(mock_checkout_ref_obj)

	def test_create_branch_error(self) -> None:
		"""Test error handling when creating a branch fails using PRGitUtils."""
		# Arrange
		mock_head_ref_for_peel = MagicMock()
		mock_head_ref_for_peel.peel.side_effect = Pygit2GitError("Pygit2 error")
		self.repo.configure_mock(head_is_unborn=False, head=mock_head_ref_for_peel)

		# Act and Assert
		with pytest.raises(GitError, match="Failed to create branch 'feature-branch' using pygit2"):
			self.pgu.create_branch("feature-branch")

	def test_checkout_branch(self) -> None:
		"""Test checking out an existing branch using PRGitUtils."""
		# Arrange
		mock_branch_ref_obj = MagicMock()
		mock_lookup_ref_method = MagicMock(return_value=mock_branch_ref_obj)
		mock_checkout_method = MagicMock()

		mock_head_attribute = MagicMock()
		mock_head_attribute.shorthand = "feature-branch"

		self.repo.configure_mock(
			lookup_reference=mock_lookup_ref_method,
			checkout=mock_checkout_method,
			head_is_detached=False,
			head=mock_head_attribute,
		)

		# Act
		self.pgu.checkout_branch("feature-branch")

		# Assert
		mock_lookup_ref_method.assert_called_once_with("refs/heads/feature-branch")
		mock_checkout_method.assert_called_once_with(mock_branch_ref_obj)

	def test_checkout_branch_error(self) -> None:
		"""Test error handling when checking out a branch fails using PRGitUtils."""
		# Arrange
		self.repo.configure_mock(lookup_reference=MagicMock(side_effect=Pygit2GitError("Pygit2 error")))

		# Act and Assert
		with pytest.raises(GitError, match="Failed to checkout branch 'feature-branch' using pygit2"):
			self.pgu.checkout_branch("feature-branch")

	def test_push_branch(self) -> None:
		"""Test pushing a branch to remote using PRGitUtils."""
		# Arrange
		mock_remote_obj = MagicMock()
		mock_remote_obj.push = MagicMock()

		# Explicitly mock the remotes collection and its __getitem__
		mock_remotes_collection = MagicMock()
		mock_remotes_collection.__getitem__.return_value = mock_remote_obj
		self.repo.remotes = mock_remotes_collection  # Assign the fully mocked collection

		# Act
		self.pgu.push_branch("feature-branch")

		# Assert
		mock_remotes_collection.__getitem__.assert_called_once_with("origin")
		mock_remote_obj.push.assert_called_once_with(["refs/heads/feature-branch:refs/heads/feature-branch"])

	def test_push_branch_force(self) -> None:
		"""Test force pushing a branch to remote using PRGitUtils."""
		# Arrange
		mock_remote_obj = MagicMock()
		mock_remote_obj.push = MagicMock()

		mock_remotes_collection = MagicMock()
		mock_remotes_collection.__getitem__.return_value = mock_remote_obj
		self.repo.remotes = mock_remotes_collection

		# Act
		self.pgu.push_branch("feature-branch", force=True)

		# Assert
		mock_remotes_collection.__getitem__.assert_called_once_with("origin")
		mock_remote_obj.push.assert_called_once_with(["+refs/heads/feature-branch:refs/heads/feature-branch"])

	def test_push_branch_error(self) -> None:
		"""Test error handling when pushing a branch fails using PRGitUtils."""
		# Arrange
		mock_remote_obj = MagicMock()
		mock_remote_obj.push.side_effect = Pygit2GitError("Pygit2 error")

		mock_remotes_collection = MagicMock()
		mock_remotes_collection.__getitem__.return_value = mock_remote_obj
		self.repo.remotes = mock_remotes_collection

		# Act and Assert
		with pytest.raises(GitError, match="Failed to push branch 'feature-branch' to remote 'origin' using pygit2"):
			self.pgu.push_branch("feature-branch")


@pytest.mark.unit
@pytest.mark.git
class TestPRUtilsCommitOperations(GitTestBase):
	"""Tests for PR generator commit-related utility functions."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		super().setup_method()  # type: ignore[misc] # pylint: disable=no-member
		self.pgu = PRGitUtils()
		self.pgu.repo = MagicMock(spec=Repository)
		self.repo = self.pgu.repo
		self._patchers = []

	def test_get_commit_messages(self) -> None:
		"""Test getting commit messages between branches using PRGitUtils."""
		# Arrange
		mock_commit1 = MagicMock(spec=Commit)
		mock_commit1.message = "feat: Add feature\nMore details."
		mock_commit2 = MagicMock(spec=Commit)
		mock_commit2.message = "fix: Fix bug\nMore details."

		mock_walk_iter = [mock_commit1, mock_commit2]
		mock_walker = MagicMock()
		mock_walker.__iter__.return_value = mock_walk_iter
		# Ensure 'hide' method exists and is callable on the walker mock
		mock_walker.hide = MagicMock()
		self.repo.walk.return_value = mock_walker

		# Mock revparse_single and peel for base_oid and head_oid resolution
		mock_base_rev_obj = MagicMock()
		mock_base_peeled_commit = MagicMock(spec=Commit, id="base_id")
		mock_base_rev_obj.peel.return_value = mock_base_peeled_commit

		mock_head_rev_obj = MagicMock()
		mock_head_peeled_commit = MagicMock(spec=Commit, id="head_id")
		mock_head_rev_obj.peel.return_value = mock_head_peeled_commit

		self.repo.revparse_single.side_effect = [mock_base_rev_obj, mock_head_rev_obj]

		# Act
		result = self.pgu.get_commit_messages("main", "feature")

		# Assert
		assert result == ["feat: Add feature", "fix: Fix bug"]
		self.repo.revparse_single.assert_any_call("main")
		self.repo.revparse_single.assert_any_call("feature")
		self.repo.walk.assert_called_once_with(mock_head_peeled_commit.id, SortMode.TOPOLOGICAL)
		mock_walker.hide.assert_called_once_with(mock_base_peeled_commit.id)

	def test_get_commit_messages_empty(self) -> None:
		"""Test getting commit messages with empty result using PRGitUtils."""
		# Arrange
		mock_walk_iter = []  # No commits
		mock_walker = MagicMock()
		mock_walker.__iter__.return_value = mock_walk_iter
		mock_walker.hide = MagicMock()
		self.repo.walk.return_value = mock_walker

		# Mock revparse_single and peel
		mock_base_rev_obj = MagicMock()
		mock_base_peeled_commit = MagicMock(spec=Commit, id="base_id")
		mock_base_rev_obj.peel.return_value = mock_base_peeled_commit
		mock_head_rev_obj = MagicMock()
		mock_head_peeled_commit = MagicMock(spec=Commit, id="head_id")
		mock_head_rev_obj.peel.return_value = mock_head_peeled_commit
		self.repo.revparse_single.side_effect = [mock_base_rev_obj, mock_head_rev_obj]

		# Act
		result = self.pgu.get_commit_messages("main", "feature")

		# Assert
		assert result == []

	def test_get_commit_messages_error(self) -> None:
		"""Test error handling when getting commit messages fails using PRGitUtils."""
		# Arrange
		self.repo.revparse_single.side_effect = Pygit2GitError("Resolution failed")

		# Act and Assert
		with pytest.raises(GitError, match="Failed to get commit messages between 'main' and 'feature' using pygit2"):
			self.pgu.get_commit_messages("main", "feature")

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

	@patch("codemap.llm.client.LLMClient")
	@patch("codemap.config.ConfigLoader")
	def test_generate_pr_title_with_llm(self, mock_config_loader, mock_llm_client_cls) -> None:
		"""Test generating PR title with LLM."""
		# Arrange
		commits = ["feat: Add user authentication", "fix: Fix login form validation"]

		# Mock the LLMClient
		mock_config = MagicMock(spec=ConfigLoader)
		mock_config_loader.return_value = mock_config

		mock_client = MagicMock(spec=LLMClient)
		mock_client.completion.return_value = "Add user authentication feature"
		mock_llm_client_cls.return_value = mock_client

		# Act
		result = generate_pr_title_with_llm(commits, mock_client)

		# Assert
		assert result == "Add user authentication feature"
		mock_client.completion.assert_called_once()
		# Ensure the provided commits are included in the messages
		assert any("user authentication" in str(arg) for arg in mock_client.completion.call_args[1]["messages"])

	def test_generate_pr_title_with_llm_empty_commits(self) -> None:
		"""Test generating PR title with LLM and empty commits."""
		# Create a mock LLMClient
		mock_client = MagicMock(spec=LLMClient)

		# Act
		result = generate_pr_title_with_llm([], mock_client)

		# Assert
		assert result == "Update branch"
		# Should return default title without calling LLM
		mock_client.completion.assert_not_called()

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

	@patch("codemap.llm.client.LLMClient")
	@patch("codemap.config.ConfigLoader")
	def test_generate_pr_description_with_llm(self, mock_config_loader, mock_llm_client_cls) -> None:
		"""Test generating PR description with LLM."""
		# Arrange
		commits = ["feat: Add user authentication", "fix: Fix login form validation"]

		# Mock the LLM client
		mock_config = MagicMock(spec=ConfigLoader)
		mock_config_loader.return_value = mock_config

		mock_client = MagicMock(spec=LLMClient)
		mock_client.completion.return_value = "This PR adds user authentication and fixes login form validation."
		mock_llm_client_cls.return_value = mock_client

		# Act
		result = generate_pr_description_with_llm(commits, mock_client)

		# Assert
		assert result == "This PR adds user authentication and fixes login form validation."
		mock_client.completion.assert_called_once()
		# Make sure the commits are included in the messages
		assert any("user authentication" in str(arg) for arg in mock_client.completion.call_args[1]["messages"])
		assert any("login form validation" in str(arg) for arg in mock_client.completion.call_args[1]["messages"])

	def test_generate_pr_description_with_llm_empty_commits(self) -> None:
		"""Test generating PR description with LLM and empty commits."""
		# Create a mock LLMClient
		mock_client = MagicMock(spec=LLMClient)

		# Act
		result = generate_pr_description_with_llm([], mock_client)

		# Assert
		assert result == "No changes"
		# Should return default description without calling LLM
		mock_client.completion.assert_not_called()


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
		super().setup_method()  # type: ignore[misc] # pylint: disable=no-member
		self.pgu = PRGitUtils()
		self.pgu.repo = MagicMock(spec=Repository)
		self.repo = self.pgu.repo
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

	def test_get_branch_relation(self) -> None:
		"""Test getting the relationship between branches using PRGitUtils."""
		# Arrange
		mock_branch_commit_peeled = MagicMock(spec=Commit, id="branch_oid")
		mock_branch_rev_obj = MagicMock()
		mock_branch_rev_obj.peel.return_value = mock_branch_commit_peeled

		mock_target_commit_peeled = MagicMock(spec=Commit, id="target_oid")
		mock_target_rev_obj = MagicMock()
		mock_target_rev_obj.peel.return_value = mock_target_commit_peeled

		self.repo.revparse_single.side_effect = [mock_branch_rev_obj, mock_target_rev_obj]
		self.repo.descendant_of.return_value = (
			False  # feature is not ancestor of main (main is not descendant of feature)
		)
		self.repo.ahead_behind.return_value = (0, 5)  # main is 5 commits ahead of feature (feature is 5 behind main)
		# (ahead, behind) relative to (target_oid, branch_oid)
		# so (target_ahead, branch_ahead) = (target_ahead, target_behind_other_way)
		# PRGitUtils: ahead, _ = self.repo.ahead_behind(target_oid, branch_oid)
		# Here, target is "main", branch is "feature"
		# So, ahead is commits in main not in feature.
		# If we want count = commits in target not in branch, and target is main, branch is feature
		# then we want first element of ahead_behind(main_oid, feature_oid)
		# if the call in code is get_branch_relation("feature", "main")
		# branch_ref_name = "feature", target_branch_ref_name = "main"
		# branch_oid from "feature", target_oid from "main"
		# descendant_of(main_oid, feature_oid) -> False (correct for feature not ancestor of main)
		# ahead_behind(main_oid, feature_oid) -> (commits in main not in feature, commits in feature not in main)
		# If main is 5 ahead of feature, this should be (5, 0)
		self.repo.ahead_behind.return_value = (5, 0)  # main is 5 ahead of feature

		# Act
		# get_branch_relation(branch_to_check_if_ancestor, target_branch_to_check_descendants_of)
		# is_ancestor = self.repo.descendant_of(target_oid, branch_oid)
		# commit_count = commits in target_branch_ref_name that are not in branch_ref_name
		is_ancestor, commit_count = self.pgu.get_branch_relation("feature", "main")

		# Assert
		assert is_ancestor is False
		assert commit_count == 5  # 5 commits in "main" (target) that are not in "feature" (branch)

		self.repo.revparse_single.assert_any_call("feature")
		self.repo.revparse_single.assert_any_call("main")
		self.repo.descendant_of.assert_called_once_with(mock_target_commit_peeled.id, mock_branch_commit_peeled.id)
		self.repo.ahead_behind.assert_called_once_with(mock_target_commit_peeled.id, mock_branch_commit_peeled.id)

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
