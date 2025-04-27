"""Tests for git utility functions."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codemap.git.utils import (
	GitDiff,
	GitError,
	commit,
	commit_only_files,
	get_other_staged_files,
	get_repo_root,
	get_staged_diff,
	get_unstaged_diff,
	get_untracked_files,
	run_git_command,
	stage_files,
	stash_staged_changes,
	unstage_files,
	unstash_changes,
	validate_repo_path,
)


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

	def test_run_git_command_success(self) -> None:
		"""Test successful git command execution."""
		with patch("subprocess.run", self.mock_run):
			result = run_git_command(["git", "status"])
			assert result == "mock output"
			self.mock_run.assert_called_once()

	def test_run_git_command_failure(self) -> None:
		"""Test failed git command execution."""
		self.mock_run.side_effect = subprocess.CalledProcessError(1, ["git", "status"], stderr="error message")
		with patch("subprocess.run", self.mock_run), pytest.raises(GitError):
			run_git_command(["git", "status"])

	def test_get_repo_root_success(self) -> None:
		"""Test getting repository root successfully."""
		with patch("codemap.git.utils.run_git_command", return_value="/path/to/repo\n"):
			result = get_repo_root()
			assert result == Path("/path/to/repo")

	def test_get_repo_root_failure(self) -> None:
		"""Test failure when getting repository root."""
		with (
			patch("codemap.git.utils.run_git_command", side_effect=GitError("Not in a Git repository")),
			pytest.raises(GitError, match="Not in a Git repository"),
		):
			get_repo_root()

	def test_validate_repo_path_success(self) -> None:
		"""Test validating repository path successfully."""
		with patch("codemap.git.utils.get_repo_root", return_value=Path("/path/to/repo")):
			result = validate_repo_path(Path("/some/path"))
			assert result == Path("/path/to/repo")

	def test_validate_repo_path_failure(self) -> None:
		"""Test failing to validate repository path."""
		with patch("codemap.git.utils.get_repo_root", side_effect=GitError("Not in a Git repository")):
			result = validate_repo_path(Path("/some/path"))
			assert result is None

	def test_get_staged_diff(self) -> None:
		"""Test getting staged diff."""
		with patch("codemap.git.utils.run_git_command") as mock_run_cmd:
			mock_run_cmd.side_effect = [
				"file1.py\nfile2.py",  # staged files
				"diff content",  # diff content
			]
			result = get_staged_diff()
			assert isinstance(result, GitDiff)
			assert result.files == ["file1.py", "file2.py"]
			assert result.content == "diff content"
			assert result.is_staged is True

	def test_get_unstaged_diff(self) -> None:
		"""Test getting unstaged diff."""
		with patch("codemap.git.utils.run_git_command") as mock_run_cmd:
			mock_run_cmd.side_effect = [
				"file1.py\nfile2.py",  # unstaged files
				"diff content",  # diff content
			]
			result = get_unstaged_diff()
			assert isinstance(result, GitDiff)
			assert result.files == ["file1.py", "file2.py"]
			assert result.content == "diff content"
			assert result.is_staged is False

	def test_stage_files_empty_list(self) -> None:
		"""Test staging an empty list of files."""
		with patch("codemap.git.utils.logger") as mock_logger:
			stage_files([])
			mock_logger.warning.assert_called_once_with("No files provided to stage_files")

	def test_stage_files_existing_files(self) -> None:
		"""Test staging existing files."""
		with (
			patch("codemap.git.utils.run_git_command") as mock_run_cmd,
			patch("pathlib.Path.exists", return_value=True),
		):
			# Mock git status, tracked files, and index files
			mock_run_cmd.side_effect = [
				" M file1.py\n D file2.py\nD  file3.py",  # git status output
				"file1.py\nfile2.py\nfile3.py",  # tracked files
				"100644 hash file1.py\n100644 hash file2.py\n100644 hash file3.py",  # index files
				"success",  # git add result
			]

			stage_files(["file1.py"])

			# Check that git add was called for the existing file
			mock_run_cmd.assert_any_call(["git", "add", "file1.py"])

	def test_stage_files_deleted_tracked_files(self) -> None:
		"""Test staging deleted files that are tracked by git."""
		with (
			patch("codemap.git.utils.run_git_command") as mock_run_cmd,
			patch("pathlib.Path.exists", return_value=False),
		):
			# Mock git status, tracked files, and index files
			mock_run_cmd.side_effect = [
				" D file2.py\nD  file3.py",  # git status output
				"file1.py\nfile2.py\nfile3.py",  # tracked files
				"100644 hash file1.py\n100644 hash file2.py\n100644 hash file3.py",  # index files
				"success",  # git rm result
			]

			stage_files(["file2.py"])

			# Check that git rm was called for the deleted tracked file
			mock_run_cmd.assert_any_call(["git", "rm", "file2.py"])

	def test_stage_files_deleted_index_files(self) -> None:
		"""Test staging files that are in the index but don't exist."""
		with (
			patch("codemap.git.utils.run_git_command") as mock_run_cmd,
			patch("pathlib.Path.exists", return_value=False),
		):
			# Mock git status, tracked files, and index files
			mock_run_cmd.side_effect = [
				"",  # git status output (empty)
				"",  # tracked files (empty)
				"100644 hash file3.py",  # index files
				"success",  # git rm --cached result
			]

			stage_files(["file3.py"])

			# Check that git rm --cached was called for the file that only exists in the index
			mock_run_cmd.assert_any_call(["git", "rm", "--cached", "file3.py"])

	def test_stage_files_git_rm_fails_fallback_to_cached(self) -> None:
		"""Test fallback to git rm --cached when git rm fails."""
		with (
			patch("codemap.git.utils.run_git_command") as mock_run_cmd,
			patch("pathlib.Path.exists", return_value=False),
		):
			# First attempt to git rm fails, fallback to git rm --cached succeeds
			def side_effect(cmd: list[str], *_, **__) -> str:
				error_msg = (
					"Git command failed: git rm file2.py\nError: fatal: pathspec 'file2.py' did not match any files"
				)

				if cmd == ["git", "status", "--porcelain"]:
					return ""  # No status info
				if cmd == ["git", "ls-files"]:
					return "file2.py"  # file is tracked
				if cmd == ["git", "ls-files", "--stage"]:
					return "100644 hash file2.py"  # file is in index
				if cmd == ["git", "rm", "file2.py"]:
					raise GitError(error_msg)
				if cmd == ["git", "rm", "--cached", "file2.py"]:
					return "success"  # Cached removal succeeds
				return "unexpected command"

			mock_run_cmd.side_effect = side_effect

			# Should not raise an exception
			stage_files(["file2.py"])

			# Verify both commands were attempted
			mock_run_cmd.assert_any_call(["git", "rm", "file2.py"])
			mock_run_cmd.assert_any_call(["git", "rm", "--cached", "file2.py"])

	def test_stage_files_invalid_filenames(self) -> None:
		"""Test that invalid filenames are filtered out."""
		with (
			patch("codemap.git.utils.run_git_command") as mock_run_cmd,
			patch("codemap.git.utils.logger") as mock_logger,
		):
			mock_run_cmd.side_effect = [
				"",  # git status
				"",  # tracked files
				"",  # index files
			]

			stage_files(["file*.py", "valid.py"])

			# Check that warning was logged for invalid filename
			mock_logger.warning.assert_any_call("Skipping invalid filename: %s", "file*.py")

	def test_stage_files_untracked_nonexistent(self) -> None:
		"""Test handling of files that don't exist and aren't tracked."""
		with (
			patch("codemap.git.utils.run_git_command") as mock_run_cmd,
			patch("pathlib.Path.exists", return_value=False),
			patch("codemap.git.utils.logger") as mock_logger,
		):
			mock_run_cmd.side_effect = [
				"",  # git status
				"",  # tracked files (empty)
				"",  # index files (empty)
			]

			stage_files(["nonexistent.py"])

			# Check that warning was logged
			mock_logger.warning.assert_any_call(
				"Skipping file %s: Does not exist and is not tracked by git", "nonexistent.py"
			)

	def test_stage_files_git_status_fails(self) -> None:
		"""Test handling when git status command fails."""
		with (
			patch("codemap.git.utils.run_git_command") as mock_run_cmd,
			patch("pathlib.Path.exists", return_value=True),
			patch("codemap.git.utils.logger") as mock_logger,
		):

			def side_effect(cmd: list[str], *_, **__) -> str:
				status_error_msg = "Failed to get git status"

				if cmd == ["git", "status", "--porcelain"]:
					raise GitError(status_error_msg)
				# These still succeed
				if cmd == ["git", "ls-files"]:
					return "file1.py"
				if cmd == ["git", "ls-files", "--stage"]:
					return "100644 hash file1.py"
				if cmd == ["git", "add", "file1.py"]:
					return "success"
				return ""

			mock_run_cmd.side_effect = side_effect

			# Our implementation raises GitError when git status fails
			with pytest.raises(GitError, match="Failed to get git status information"):
				stage_files(["file1.py"])

			# Check that error was logged
			mock_logger.error.assert_called()

	def test_commit(self) -> None:
		"""Test creating a commit."""
		with patch("subprocess.run") as mock_run:
			commit("Test commit message")
			mock_run.assert_called_once()

	def test_commit_failure(self) -> None:
		"""Test commit failure."""
		with (
			patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error")),
			pytest.raises(GitError),
		):
			commit("Test commit message")

	def test_get_other_staged_files(self) -> None:
		"""Test getting other staged files."""
		with patch("codemap.git.utils.run_git_command", return_value="file1.py\nfile2.py\nfile3.py"):
			result = get_other_staged_files(["file1.py"])
			assert result == ["file2.py", "file3.py"]

	def test_stash_staged_changes_no_other_files(self) -> None:
		"""Test stashing staged changes when there are no other files."""
		with patch("codemap.git.utils.get_other_staged_files", return_value=[]):
			result = stash_staged_changes(["file1.py"])
			assert result is False

	def test_stash_staged_changes_with_other_files(self) -> None:
		"""Test stashing staged changes when there are other files."""
		with (
			patch("codemap.git.utils.get_other_staged_files", return_value=["file2.py"]),
			patch("codemap.git.utils.run_git_command"),
		):
			result = stash_staged_changes(["file1.py"])
			assert result is True

	def test_unstash_changes_no_stash(self) -> None:
		"""Test unstashing when there's no stash."""
		with patch("codemap.git.utils.run_git_command", return_value=""):
			unstash_changes()  # Should not raise or call stash pop

	def test_unstash_changes_with_stash(self) -> None:
		"""Test unstashing when there's a stash."""
		with patch("codemap.git.utils.run_git_command") as mock_run_cmd:
			mock_run_cmd.side_effect = [
				"stash@{0}: On main: CodeMap: temporary stash for commit",  # stash list
				"success",  # stash pop result
			]
			unstash_changes()
			# Verify stash pop was called
			mock_run_cmd.assert_any_call(["git", "stash", "pop"])

	def test_unstage_files(self) -> None:
		"""Test unstaging files."""
		with patch("codemap.git.utils.run_git_command") as mock_run_cmd:
			unstage_files(["file1.py", "file2.py"])
			mock_run_cmd.assert_called_once_with(["git", "restore", "--staged", "file1.py", "file2.py"])

	def test_unstage_files_failure(self) -> None:
		"""Test failure when unstaging files."""
		with (
			patch("codemap.git.utils.run_git_command", side_effect=GitError("Failed to unstage")),
			pytest.raises(GitError, match="Failed to unstage files: file1.py, file2.py"),
		):
			unstage_files(["file1.py", "file2.py"])

	def test_get_untracked_files(self) -> None:
		"""Test getting untracked files."""
		with patch("codemap.git.utils.run_git_command", return_value="file1.py\nfile2.py"):
			result = get_untracked_files()
			assert result == ["file1.py", "file2.py"]

	def test_commit_only_files(self) -> None:
		"""Test committing only specific files."""
		with (
			patch("subprocess.run") as mock_run,
			patch("codemap.git.utils.stage_files") as mock_stage,
			patch("codemap.git.utils.get_other_staged_files", return_value=["other.py"]),
		):
			# Mock subprocess.run for status and commit
			mock_run.side_effect = [
				MagicMock(stdout=" M file1.py\n M file2.py"),  # git status
				MagicMock(),  # git commit
			]

			result = commit_only_files(["file1.py"], "Test commit")

			# Verify stage_files was called
			mock_stage.assert_called_once_with(["file1.py"])

			# Verify result includes other staged files
			assert result == ["other.py"]

	def test_commit_only_files_with_hooks_disabled(self) -> None:
		"""Test committing with Git hooks disabled."""
		with (
			patch("subprocess.run") as mock_run,
			patch("codemap.git.utils.stage_files"),
			patch("codemap.git.utils.get_other_staged_files", return_value=[]),
		):
			# Mock subprocess.run for status and commit
			mock_run.side_effect = [
				MagicMock(stdout=" M file1.py"),  # git status
				MagicMock(),  # git commit
			]

			commit_only_files(["file1.py"], "Test commit", ignore_hooks=True)

			# Check that --no-verify was added to the commit command
			commit_call = next(call for call in mock_run.call_args_list if "commit" in str(call))
			assert "--no-verify" in commit_call[0][0]
