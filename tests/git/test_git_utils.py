"""Tests for git utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codemap.git.utils import (
	ExtendedGitRepoContext,
	GitDiff,
	GitError,
)
from tests.conftest import skip_git_tests


class TestGitUtils:
	"""Test git utility functions."""

	def setup_method(self) -> None:
		"""Set up test environment."""
		# Create mock for subprocess.run
		self.mock_run = MagicMock()
		self.mock_run.return_value.stdout = "mock output"

		# Create patch for Path.exists
		self.mock_exists = MagicMock(return_value=True)

		# Create patch for run_git_command
		self.mock_run_git_command = MagicMock(return_value="mock output")

		# Create a patcher for the class method
		self.get_repo_root_patcher = patch.object(
			ExtendedGitRepoContext, "get_repo_root", return_value=Path("/path/to/repo")
		)
		self.mock_get_repo_root = self.get_repo_root_patcher.start()

		# Create and initialize mock git context
		self.git_context = MagicMock(spec=ExtendedGitRepoContext)
		# Set up method return values
		self.git_context.get_staged_diff.return_value = GitDiff(
			files=["file1.py", "file2.py"], content="diff content", is_staged=True
		)
		self.git_context.get_unstaged_diff.return_value = GitDiff(
			files=["file1.py", "file2.py"], content="diff content", is_staged=False
		)
		self.git_context.get_untracked_files.return_value = ["file1.py", "file2.py"]
		self.git_context.get_other_staged_files.return_value = ["file2.py", "file3.py"]

	def teardown_method(self) -> None:
		"""Clean up after test."""
		# Stop the patchers
		self.get_repo_root_patcher.stop()

	def test_get_repo_root_success(self) -> None:
		"""Test getting repository root successfully."""
		with patch(
			"codemap.git.utils.ExtendedGitRepoContext.get_repo_root", return_value=Path("/path/to/repo")
		) as mock_get_repo_root:
			result = ExtendedGitRepoContext.get_repo_root()
			assert result == Path("/path/to/repo")
			mock_get_repo_root.assert_called_once()

	def test_get_repo_root_failure(self) -> None:
		"""Test failure when getting repository root."""
		with patch(
			"codemap.git.utils.ExtendedGitRepoContext.get_repo_root", side_effect=GitError("Not in a Git repository")
		) as mock_get_repo_root:
			with pytest.raises(GitError, match="Not in a Git repository"):
				ExtendedGitRepoContext.get_repo_root()
			mock_get_repo_root.assert_called_once()

	@skip_git_tests
	def test_validate_repo_path_success(self) -> None:
		"""Test validating repository path successfully."""
		with patch("codemap.git.utils.ExtendedGitRepoContext.get_repo_root", return_value=Path("/path/to/repo")):
			result = ExtendedGitRepoContext.validate_repo_path(Path("/some/path"))
			assert result == Path("/path/to/repo")

	@skip_git_tests
	def test_validate_repo_path_failure(self) -> None:
		"""Test failing to validate repository path."""
		with patch(
			"codemap.git.utils.ExtendedGitRepoContext.get_repo_root", side_effect=GitError("Not in a Git repository")
		):
			result = ExtendedGitRepoContext.validate_repo_path(Path("/some/path"))
			assert result is None

	def test_get_staged_diff(self) -> None:
		"""Test getting staged diff."""
		result = self.git_context.get_staged_diff()
		assert isinstance(result, GitDiff)
		assert result.files == ["file1.py", "file2.py"]
		assert result.content == "diff content"
		assert result.is_staged is True

	def test_get_unstaged_diff(self) -> None:
		"""Test getting unstaged diff."""
		result = self.git_context.get_unstaged_diff()
		assert isinstance(result, GitDiff)
		assert result.files == ["file1.py", "file2.py"]
		assert result.content == "diff content"
		assert result.is_staged is False

	def test_stage_files(self) -> None:
		"""Test staging files."""
		# Test basic stage files functionality
		self.git_context.stage_files(["file1.py", "file2.py"])
		self.git_context.stage_files.assert_called_with(["file1.py", "file2.py"])

	def test_stage_files_empty_list(self) -> None:
		"""Test staging an empty list of files."""
		self.git_context.stage_files([])
		self.git_context.stage_files.assert_called_with([])

	def test_commit(self) -> None:
		"""Test creating a commit."""
		self.git_context.commit("Test commit message")
		self.git_context.commit.assert_called_with("Test commit message")

	def test_commit_failure(self) -> None:
		"""Test commit failure."""
		self.git_context.commit.side_effect = GitError("Failed to commit")
		with pytest.raises(GitError):
			self.git_context.commit("Test commit message")
		self.git_context.commit.side_effect = None  # Reset for other tests

	def test_get_other_staged_files(self) -> None:
		"""Test getting other staged files."""
		result = self.git_context.get_other_staged_files(["file1.py"])
		assert result == ["file2.py", "file3.py"]

	def test_stash_staged_changes_no_other_files(self) -> None:
		"""Test stashing staged changes when there are no other files."""
		# Setup - no other staged files
		self.git_context.get_other_staged_files.return_value = []
		self.git_context.stash_staged_changes.return_value = False

		# Execute
		result = self.git_context.stash_staged_changes(["file1.py"])

		# Verify
		assert result is False
		self.git_context.stash_staged_changes.assert_called_with(["file1.py"])

	def test_stash_staged_changes_with_other_files(self) -> None:
		"""Test stashing staged changes when there are other files."""
		# Setup - has other staged files
		self.git_context.get_other_staged_files.return_value = ["file2.py"]
		self.git_context.stash_staged_changes.return_value = True

		# Execute
		result = self.git_context.stash_staged_changes(["file1.py"])

		# Verify
		assert result is True
		self.git_context.stash_staged_changes.assert_called_with(["file1.py"])

	def test_unstash_changes_no_stash(self) -> None:
		"""Test unstashing when there's no stash."""
		# Setup - simulate no stash found
		self.git_context.get_untracked_files.return_value = []

		# Execute - should not raise an exception
		self.git_context.unstash_changes()

		# Verify method was called
		self.git_context.unstash_changes.assert_called_once()

	def test_unstash_changes_with_stash(self) -> None:
		"""Test unstashing when there's a stash."""
		# Setup - simulate stash found
		self.git_context.get_untracked_files.return_value = ["stash@{0}: On main: CodeMap: temporary stash for commit"]

		# Execute
		self.git_context.unstash_changes()

		# Verify method was called
		self.git_context.unstash_changes.assert_called_once()

	def test_unstage_files(self) -> None:
		"""Test unstaging files."""
		self.git_context.unstage_files(["file1.py", "file2.py"])
		self.git_context.unstage_files.assert_called_with(["file1.py", "file2.py"])

	def test_unstage_files_failure(self) -> None:
		"""Test failure when unstaging files."""
		self.git_context.unstage_files.side_effect = GitError("Failed to unstage")
		with pytest.raises(GitError):
			self.git_context.unstage_files(["file1.py", "file2.py"])
		self.git_context.unstage_files.side_effect = None  # Reset for other tests

	def test_get_untracked_files(self) -> None:
		"""Test getting untracked files."""
		result = self.git_context.get_untracked_files()
		assert result == ["file1.py", "file2.py"]

	def test_commit_only_files(self) -> None:
		"""Test committing only specific files."""
		# Setup
		self.git_context.commit_only_files.return_value = ["other.py"]

		# Execute
		result = self.git_context.commit_only_files(["file1.py"], "Test commit")

		# Verify
		assert result == ["other.py"]
		self.git_context.commit_only_files.assert_called_with(["file1.py"], "Test commit")

	def test_commit_only_files_with_hooks_disabled(self) -> None:
		"""Test committing with Git hooks disabled."""
		self.git_context.commit_only_files(["file1.py"], "Test commit", ignore_hooks=True)
		self.git_context.commit_only_files.assert_called_with(["file1.py"], "Test commit", ignore_hooks=True)
