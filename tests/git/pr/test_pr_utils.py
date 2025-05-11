"""Tests for PR generator utility functions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
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
		self._patchers = []

		try:
			self.repo = MagicMock(spec=Repository)
			# Patch GitRepoContext.get_repo_root to return a mock path, preventing real repo discovery.
			# Patch codemap.utils.git_utils.Repository so that GitRepoContext's self.repo becomes our mock.
			with (
				patch("codemap.utils.git_utils.GitRepoContext.get_repo_root", return_value=Path("/mock/repo/root")),
				patch("codemap.utils.git_utils.Repository", return_value=self.repo),
			):
				self.pgu_real_instance = PRGitUtils()

			# Ensure the instance's repo attribute is our mock.
			# The patch for codemap.utils.git_utils.Repository should handle this during super().__init__(),
			# but direct assignment is a safeguard.
			self.pgu_real_instance.repo = self.repo
			# Ensure git_root is also set to the mock path, which should happen via get_repo_root patch.
			# ExtendedGitRepoContext sets self.git_root = self.repo_root (from GitRepoContext).
			# The patch to GitRepoContext.get_repo_root should ensure self.repo_root is Path("/mock/repo/root"),
			# and thus self.git_root will be set correctly by ExtendedGitRepoContext's __init__.
			# No explicit assignment to self.pgu_real_instance.git_root should be needed here.

		except Pygit2GitError:  # This block should ideally not be reached with the new patches.
			# Fallback if instantiation is problematic without deeper init patching:
			# Create a dummy PRGitUtils and manually set repo. This is less ideal as it avoids __init__.
			class DummyPRGitUtils(PRGitUtils):
				def __init__(self, mock_repo_to_assign) -> None:
					# Skip ExtendedGitRepoContext's __init__ to avoid repo loading issues
					# by not calling super().__init__(). This is a significant override.
					# pylint: disable=super-init-not-called
					self.repo = mock_repo_to_assign
					self.git_root = Path("/mock/dummy/root")  # Provide a Path object
					self.branch = None
					# Initialize other necessary attributes if PRGitUtils methods depend on them

			self.repo = MagicMock(spec=Repository)
			self.pgu_real_instance = DummyPRGitUtils(self.repo)

		# Patch PRGitUtils.get_instance to return our real instance (which has a mock repo)
		patch_get_instance = patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance")
		self.mock_get_instance_method = patch_get_instance.start()
		self.mock_get_instance_method.return_value = self.pgu_real_instance
		self._patchers.append(patch_get_instance)

		# For convenience in tests, self.pgu refers to the real instance
		self.pgu = self.pgu_real_instance
		# self.repo is already the MagicMock assigned to self.pgu.repo

	def teardown_method(self, _: None) -> None:
		"""Tear down test fixtures."""
		for patcher in self._patchers:
			patcher.stop()
		# Restore original get_instance if it was stored and modified
		# PRGitUtils._pr_git_utils_instance = None # Reset singleton for other tests

	def test_get_current_branch(self) -> None:
		"""Test getting the current branch name using PRGitUtils."""
		# Arrange
		mock_head_attribute = MagicMock()
		mock_head_attribute.shorthand = "feature-branch"
		self.repo.configure_mock(head_is_detached=False, head=mock_head_attribute)
		# No need to mock self.pgu.get_current_branch, we are testing its implementation

		# Act
		result = self.pgu.get_current_branch()  # Call the real method

		# Assert
		assert result == "feature-branch"
		# We can assert that self.repo.head was accessed, if needed for more detailed testing.
		# self.repo.head.assert_called() # or similar, depending on Pygit2's structure

	def test_get_current_branch_detached_head(self) -> None:
		"""Test getting current branch when HEAD is detached."""
		# Arrange
		MagicMock()  # Not used directly for shorthand
		MagicMock()
		mock_peeled_commit = MagicMock(spec=Commit)
		mock_peeled_commit.short_id = "abcdef0"
		# In pygit2, peel() is a method of a Reference or Oid object, not the commit itself for this context.
		# repo.revparse_single("HEAD").peel(Commit)
		mock_head_ref_obj = MagicMock()  # Represents the result of repo.revparse_single("HEAD")
		mock_head_ref_obj.peel.return_value = mock_peeled_commit  # peel(Commit) returns the commit

		self.repo.configure_mock(
			head_is_detached=True,
			# head=mock_head_attribute, # .head is a Reference in non-detached
			revparse_single=MagicMock(return_value=mock_head_ref_obj),  # mock for repo.revparse_single("HEAD")
		)
		# No need to mock self.pgu.get_current_branch itself

		# Act
		result = self.pgu.get_current_branch()  # Call the real method

		# Assert
		assert result == "abcdef0"
		self.repo.revparse_single.assert_called_once_with("HEAD")
		mock_head_ref_obj.peel.assert_called_once_with(Commit)

	def test_get_current_branch_error(self) -> None:
		"""Test error handling when getting the current branch fails via PRGitUtils."""
		# Arrange
		# Configure the mock repo to raise an error when 'head' is accessed or revparse_single is called
		# The actual PRGitUtils.get_current_branch() might use self.repo.head or self.repo.revparse_single
		# Let's assume it uses self.repo.head and that access raises Pygit2GitError
		self.repo.configure_mock(head=MagicMock(side_effect=Pygit2GitError("Pygit2 error on head access")))
		# Or if it uses revparse_single in the error path:
		# self.repo.revparse_single.side_effect = Pygit2GitError("Pygit2 error on revparse")

		# Act and Assert
		with pytest.raises(
			GitError, match="Failed to get current branch: Pygit2 error on head access"
		):  # Match the actual error
			self.pgu.get_current_branch()  # Call the real method

	def test_create_branch(self) -> None:
		"""Test creating a new branch using PRGitUtils."""
		# Arrange
		mock_head_peel_commit = MagicMock(spec=Commit)  # This is the commit object itself
		mock_head_peel_commit.id = "mock_head_commit_id"

		# This mock represents the repo.head reference object
		mock_head_ref = MagicMock()
		mock_head_ref.peel.return_value = mock_head_peel_commit  # repo.head.peel(Commit)

		# self.repo is the MagicMock for the Repository
		self.repo.configure_mock(
			head_is_unborn=False,
			head=mock_head_ref,  # repo.head will return mock_head_ref
			# create_branch, lookup_reference, checkout are already MagicMocks on self.repo
		)

		mock_checkout_ref_obj = MagicMock()
		self.repo.lookup_reference.return_value = mock_checkout_ref_obj

		# Act
		self.pgu.create_branch("feature-branch")  # Call the real method on PRGitUtils instance

		# Assert that the correct methods were called on self.repo (the mock Repository)
		mock_head_ref.peel.assert_called_once_with(Commit)
		self.repo.create_branch.assert_called_once_with("feature-branch", mock_head_peel_commit)
		self.repo.lookup_reference.assert_called_with(
			"refs/heads/feature-branch"
		)  # checkout_branch is called internally
		self.repo.checkout.assert_called_once_with(mock_checkout_ref_obj)

	def test_create_branch_from_reference(self) -> None:
		"""Test creating a new branch from a specific reference."""
		# Arrange
		mock_ref_commit = MagicMock(spec=Commit)  # The commit object from the reference
		mock_ref_commit.id = "ref_commit_id"

		mock_revparse_obj = MagicMock()  # Object returned by revparse_single
		mock_revparse_obj.peel.return_value = mock_ref_commit  # .peel(Commit) gives the commit

		self.repo.revparse_single.return_value = mock_revparse_obj
		# create_branch, lookup_reference, checkout are already MagicMocks on self.repo

		mock_checkout_ref_obj = MagicMock()
		self.repo.lookup_reference.return_value = mock_checkout_ref_obj

		# Act
		self.pgu.create_branch("new-feature", from_reference="main")  # Call real method

		# Assert
		self.repo.revparse_single.assert_called_once_with("main")
		mock_revparse_obj.peel.assert_called_once_with(Commit)
		self.repo.create_branch.assert_called_once_with("new-feature", mock_ref_commit)
		self.repo.lookup_reference.assert_called_with("refs/heads/new-feature")  # from internal checkout_branch
		self.repo.checkout.assert_called_once_with(mock_checkout_ref_obj)

	def test_create_branch_error(self) -> None:
		"""Test error handling when creating a branch fails using PRGitUtils."""
		# Arrange
		# Let's make repo.head.peel() raise the error
		mock_head_ref_for_peel = MagicMock()
		mock_head_ref_for_peel.peel.side_effect = Pygit2GitError("Pygit2 error on peel")
		self.repo.configure_mock(head_is_unborn=False, head=mock_head_ref_for_peel)

		# Act and Assert
		# The error message in PRGitUtils.create_branch is "Failed to create branch '{branch_name}' using pygit2: {e}"
		with pytest.raises(
			GitError, match="Failed to create branch 'feature-branch' using pygit2: Pygit2 error on peel"
		):
			self.pgu.create_branch("feature-branch")  # Call real method

	def test_checkout_branch(self) -> None:
		"""Test checking out an existing branch using PRGitUtils."""
		# Arrange
		mock_branch_ref_obj = MagicMock()  # This is the reference object for the branch
		self.repo.lookup_reference.return_value = mock_branch_ref_obj

		# Mock attributes for updating self.branch after checkout
		mock_current_branch_head_attribute = MagicMock()
		mock_current_branch_head_attribute.shorthand = "feature-branch"
		self.repo.configure_mock(
			head_is_detached=False,  # After checkout, head is not detached
			head=mock_current_branch_head_attribute,  # repo.head after checkout
		)

		# Act
		self.pgu.checkout_branch("feature-branch")  # Call real method

		# Assert
		self.repo.lookup_reference.assert_called_once_with("refs/heads/feature-branch")
		self.repo.checkout.assert_called_once_with(mock_branch_ref_obj)
		# Also, check if self.pgu.branch (from ExtendedGitRepoContext) was updated, if that's a desired side effect to test.
		# For now, focusing on pygit2 calls. Based on PRGitUtils.checkout_branch, it does update self.branch.
		assert self.pgu.branch == "feature-branch"

	def test_checkout_branch_error(self) -> None:
		"""Test error handling when checking out a branch fails using PRGitUtils."""
		# Arrange
		self.repo.lookup_reference.side_effect = Pygit2GitError("Pygit2 error on lookup")

		# Act and Assert
		# The error message in PRGitUtils.checkout_branch is "Failed to checkout branch '{branch_name}' using pygit2: {e}"
		with pytest.raises(
			GitError, match="Failed to checkout branch 'feature-branch' using pygit2: Pygit2 error on lookup"
		):
			self.pgu.checkout_branch("feature-branch")

	def test_push_branch(self) -> None:
		"""Test pushing a branch to remote using PRGitUtils."""
		# Arrange
		mock_remote_obj = MagicMock()  # This is the mock for repo.remotes['origin']
		# mock_remote_obj.push is already a MagicMock

		# Configure self.repo.remotes to be a mock that returns mock_remote_obj on __getitem__
		# self.repo.remotes is already a MagicMock. Its __getitem__ is also a MagicMock.
		self.repo.remotes.__getitem__.return_value = mock_remote_obj

		# Mock the credential callback part to avoid its complex logic for this unit test
		# The push_branch method in PRGitUtils sets up a credential_callback.
		# We need to ensure this doesn't break or try to do real SSH/HTTP calls.
		# We can patch 'pygit2.callbacks.RemoteCallbacks' or the specific credential functions it uses if they are an issue.
		# For now, let's assume the default RemoteCallbacks() works or the error happens before/after callbacks.
		# A minimal Callbacks object might be needed if push requires it.
		with patch("codemap.git.pr_generator.pr_git_utils.RemoteCallbacks") as mock_remote_callbacks_cls:
			mock_callbacks_instance = MagicMock()
			mock_remote_callbacks_cls.return_value = mock_callbacks_instance

			# Act
			self.pgu.push_branch("feature-branch")  # Call real method

			# Assert
			self.repo.remotes.__getitem__.assert_called_once_with("origin")
			mock_remote_obj.push.assert_called_once_with(
				["refs/heads/feature-branch:refs/heads/feature-branch"], callbacks=mock_callbacks_instance
			)

	def test_push_branch_force(self) -> None:
		"""Test force pushing a branch to remote using PRGitUtils."""
		# Arrange
		mock_remote_obj = MagicMock()
		self.repo.remotes.__getitem__.return_value = mock_remote_obj

		with patch("codemap.git.pr_generator.pr_git_utils.RemoteCallbacks") as mock_remote_callbacks_cls:
			mock_callbacks_instance = MagicMock()
			mock_remote_callbacks_cls.return_value = mock_callbacks_instance

			# Act
			self.pgu.push_branch("feature-branch", force=True)  # Call real method

			# Assert
			self.repo.remotes.__getitem__.assert_called_once_with("origin")
			mock_remote_obj.push.assert_called_once_with(
				["+refs/heads/feature-branch:refs/heads/feature-branch"], callbacks=mock_callbacks_instance
			)

	def test_push_branch_error(self) -> None:
		"""Test error handling when pushing a branch fails using PRGitUtils."""
		# Arrange
		# Make the 'push' call on the remote object raise an error
		mock_remote_obj = MagicMock()
		mock_remote_obj.push.side_effect = Pygit2GitError("Pygit2 error on push")
		self.repo.remotes.__getitem__.return_value = mock_remote_obj

		# Act and Assert
		# The error message in PRGitUtils.push_branch is "Failed to push branch '{branch_name}' to remote '{remote_name}' using pygit2: {e}"
		with patch("codemap.git.pr_generator.pr_git_utils.RemoteCallbacks"):  # Mock callbacks to simplify
			with pytest.raises(
				GitError,
				match="Failed to push branch 'feature-branch' to remote 'origin' using pygit2: Pygit2 error on push",
			):
				self.pgu.push_branch("feature-branch")  # Call real method


@pytest.mark.unit
@pytest.mark.git
class TestPRUtilsCommitOperations(GitTestBase):
	"""Tests for PR generator commit-related utility functions."""

	def setup_method(self) -> None:
		"""Set up for tests."""
		self._patchers = []

		try:
			self.repo = MagicMock(spec=Repository)
			with (
				patch("codemap.utils.git_utils.GitRepoContext.get_repo_root", return_value=Path("/mock/repo/root")),
				patch("codemap.utils.git_utils.Repository", return_value=self.repo),
			):
				self.pgu_real_instance = PRGitUtils()
			self.pgu_real_instance.repo = self.repo
			# self.pgu_real_instance.git_root should be set correctly via __init__ chain and patches.
		except Pygit2GitError:  # This block should ideally not be reached.

			class DummyPRGitUtils(PRGitUtils):
				def __init__(self, mock_repo_to_assign) -> None:
					# pylint: disable=super-init-not-called
					self.repo = mock_repo_to_assign
					self.git_root = Path("/mock/dummy/root")  # Provide a Path object
					self.branch = None

			self.repo = MagicMock(spec=Repository)
			self.pgu_real_instance = DummyPRGitUtils(self.repo)

		patch_get_instance = patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance")
		self.mock_get_instance_method = patch_get_instance.start()
		self.mock_get_instance_method.return_value = self.pgu_real_instance
		self._patchers.append(patch_get_instance)

		self.pgu = self.pgu_real_instance

	def teardown_method(self, _: None) -> None:
		"""Tear down test fixtures."""
		for patcher in self._patchers:
			patcher.stop()
		# PRGitUtils._pr_git_utils_instance = None # Reset singleton

	def test_get_commit_messages(self) -> None:
		"""Test getting commit messages between branches using PRGitUtils."""
		# Arrange
		# The real pgu.get_commit_messages will call self.repo.revparse_single and self.repo.walk
		mock_base_commit = MagicMock(spec=Commit, id="base_oid")
		mock_head_commit = MagicMock(spec=Commit, id="head_oid")

		mock_base_rev_obj = MagicMock()
		mock_base_rev_obj.peel.return_value = mock_base_commit
		mock_head_rev_obj = MagicMock()
		mock_head_rev_obj.peel.return_value = mock_head_commit

		# self.repo.revparse_single needs to be a mock that returns different values on subsequent calls
		self.repo.revparse_single.side_effect = [mock_base_rev_obj, mock_head_rev_obj]  # base, then head

		# Mock the walker
		commit1_pygit2 = MagicMock(spec=Commit)
		commit1_pygit2.message = "feat: Add feature"
		commit2_pygit2 = MagicMock(spec=Commit)
		commit2_pygit2.message = "fix: Fix bug"

		mock_walker = MagicMock()
		mock_walker.__iter__.return_value = [commit1_pygit2, commit2_pygit2]  # Commits returned by walk
		self.repo.walk.return_value = mock_walker

		# Act
		result = self.pgu.get_commit_messages("main", "feature")  # Call real method

		# Assert
		assert result == ["feat: Add feature", "fix: Fix bug"]

		# Check calls on self.repo
		assert self.repo.revparse_single.call_count == 2
		self.repo.revparse_single.assert_any_call("main")  # Actually "base_branch" then "head_branch"
		self.repo.revparse_single.assert_any_call("feature")  # This order is from the test, method uses base then head

		self.repo.walk.assert_called_once_with(mock_head_commit.id, SortMode.TOPOLOGICAL)
		mock_walker.hide.assert_called_once_with(mock_base_commit.id)

	def test_get_commit_messages_empty(self) -> None:
		"""Test getting commit messages with empty result using PRGitUtils."""
		# Arrange
		mock_base_commit = MagicMock(spec=Commit, id="base_oid")
		mock_head_commit = MagicMock(spec=Commit, id="head_oid")
		mock_base_rev_obj = MagicMock()
		mock_base_rev_obj.peel.return_value = mock_base_commit
		mock_head_rev_obj = MagicMock()
		mock_head_rev_obj.peel.return_value = mock_head_commit
		self.repo.revparse_single.side_effect = [mock_base_rev_obj, mock_head_rev_obj]

		mock_walk_iter = []  # No commits
		mock_walker = MagicMock()
		mock_walker.__iter__.return_value = mock_walk_iter
		self.repo.walk.return_value = mock_walker

		# Act
		result = self.pgu.get_commit_messages("main", "feature")  # Call real method

		# Assert
		assert result == []
		# self.pgu.get_commit_messages.assert_called_once_with("main", "feature") # No, testing implementation

	def test_get_commit_messages_error(self) -> None:
		"""Test error handling when getting commit messages fails using PRGitUtils."""
		# Arrange
		# Make revparse_single raise an error
		self.repo.revparse_single.side_effect = Pygit2GitError("Resolution failed")

		# Act and Assert
		# The error message in PRGitUtils.get_commit_messages is "Failed to get commit messages between '{base_branch}' and '{head_branch}' using pygit2: {e}"
		with pytest.raises(
			GitError, match="Failed to get commit messages between 'main' and 'feature' using pygit2: Resolution failed"
		):
			self.pgu.get_commit_messages("main", "feature")  # Call real method

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
	def test_create_pull_request(self, mock_subprocess_run) -> None:
		"""Test creating a pull request."""
		# Arrange - Mock subprocess.run to return a successful result with a URL
		# Mock the gh CLI check and PR creation
		mock_subprocess_run.side_effect = [
			MagicMock(returncode=0, stdout="gh version 2.0.0"),  # gh --version check
			MagicMock(returncode=0, stdout="https://github.com/user/repo/pull/1"),  # gh pr create
		]

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
		assert mock_subprocess_run.call_args_list[0][0][0] == ["gh", "--version"]  # First call checks version

		args = mock_subprocess_run.call_args_list[1][0][0]  # Second call creates PR
		assert args[0:3] == ["gh", "pr", "create"]
		assert args[3:5] == ["--base", "main"]
		assert args[5:7] == ["--head", "feature"]
		assert args[7:9] == ["--title", "Add feature"]
		assert args[9:11] == ["--body", "Description"]

	@patch("subprocess.run")
	def test_create_pull_request_error(self, mock_subprocess_run) -> None:
		"""Test error handling when creating a pull request fails."""
		# Arrange
		# Mock the gh --version check to succeed
		# Mock the gh pr create call to fail
		mock_subprocess_run.side_effect = [
			MagicMock(returncode=0, stdout="gh version 2.0.0"),  # Successful gh --version check
			subprocess.CalledProcessError(1, ["gh", "pr", "create"], stderr="Error: Failed to create PR"),
		]

		# Act and Assert
		with pytest.raises(PRCreationError, match="Failed to create PR"):
			create_pull_request("main", "feature", "Add feature", "Description")

	@patch("subprocess.run")
	@patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance")
	def test_update_pull_request(self, mock_get_instance, mock_subprocess_run) -> None:
		"""Test updating a pull request."""
		# Arrange
		mock_pgu = MagicMock()
		mock_pgu.get_current_branch.return_value = "feature"
		mock_get_instance.return_value = mock_pgu

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
	@patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance")
	def test_update_pull_request_error(self, mock_get_instance, mock_subprocess_run) -> None:
		"""Test error handling when updating a pull request fails."""
		# Arrange
		mock_pgu = MagicMock()
		mock_pgu.get_current_branch.return_value = "feature"
		mock_get_instance.return_value = mock_pgu

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

		try:
			self.repo = MagicMock(spec=Repository)
			with (
				patch("codemap.utils.git_utils.GitRepoContext.get_repo_root", return_value=Path("/mock/repo/root")),
				patch("codemap.utils.git_utils.Repository", return_value=self.repo),
			):
				self.pgu_real_instance = PRGitUtils()
			self.pgu_real_instance.repo = self.repo
			# self.pgu_real_instance.git_root should be set correctly via __init__ chain and patches.
		except Pygit2GitError:  # This block should ideally not be reached.

			class DummyPRGitUtils(PRGitUtils):
				def __init__(self, mock_repo_to_assign) -> None:
					# pylint: disable=super-init-not-called
					self.repo = mock_repo_to_assign
					self.git_root = Path("/mock/dummy/root")  # Provide a Path object
					self.branch = None

			self.repo = MagicMock(spec=Repository)
			self.pgu_real_instance = DummyPRGitUtils(self.repo)

		patch_get_instance = patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance")
		self.mock_get_instance_method = patch_get_instance.start()
		self.mock_get_instance_method.return_value = self.pgu_real_instance
		self._patchers.append(patch_get_instance)

		self.pgu = self.pgu_real_instance
		# self.repo is already the MagicMock for Repository

	def teardown_method(self, _: None) -> None:
		"""Tear down test fixtures."""
		for patcher in self._patchers:
			patcher.stop()
		# PRGitUtils._pr_git_utils_instance = None # Reset singleton

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

		# Configure self.repo.revparse_single to return the correct objects for branch and target
		self.repo.revparse_single.side_effect = [mock_branch_rev_obj, mock_target_rev_obj]

		self.repo.descendant_of.return_value = False
		self.repo.ahead_behind.return_value = (5, 0)  # target is 5 ahead of branch

		# Act
		is_ancestor, commit_count = self.pgu.get_branch_relation("feature", "main")  # Call real method

		# Assert
		assert is_ancestor is False
		assert commit_count == 5

		assert self.repo.revparse_single.call_count == 2
		self.repo.revparse_single.assert_any_call("feature")
		self.repo.revparse_single.assert_any_call("main")

		mock_branch_rev_obj.peel.assert_called_once_with(Commit)
		mock_target_rev_obj.peel.assert_called_once_with(Commit)

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
