"""Tests for the Git utilities."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from codemap.utils.git_utils import (
	GitError,
	commit_only_files,
	get_other_staged_files,
	get_untracked_files,
	stage_files,
	stash_staged_changes,
	unstash_changes,
)
from tests.base import GitTestBase


@pytest.mark.unit
@pytest.mark.git
class TestGitFileOperations(GitTestBase):
	"""Test cases for Git file operations."""

	def test_get_other_staged_files(self) -> None:
		"""Test getting other staged files."""
		with patch("codemap.utils.git_utils.run_git_command") as mock_run:
			# Mock the git diff command to return a list of staged files
			mock_run.return_value = "file1.txt\nfile2.txt\nfile3.txt"

			# Get files not in the target list
			other_files = get_other_staged_files(["file1.txt"])

			# Assert correct files returned
			assert len(other_files) == 2
			assert "file2.txt" in other_files
			assert "file3.txt" in other_files
			assert "file1.txt" not in other_files

	def test_get_other_staged_files_error(self) -> None:
		"""Test error handling in get_other_staged_files."""
		with patch("codemap.utils.git_utils.run_git_command") as mock_run:
			# Mock a git error
			mock_run.side_effect = GitError("Git command failed")

			# Function should raise GitError
			with pytest.raises(GitError) as excinfo:
				get_other_staged_files(["file1.txt"])

			# Check error message is appropriate
			assert "Failed to check for other staged files" in str(excinfo.value)

	def test_get_untracked_files(self) -> None:
		"""Test getting untracked files."""
		with patch("codemap.utils.git_utils.run_git_command") as mock_run:
			# Mock the git ls-files command to return a list of untracked files
			mock_run.return_value = "new_file1.txt\nnew_file2.txt\nnew_dir/new_file3.txt"

			# Get untracked files
			untracked = get_untracked_files()

			# Assert correct files returned
			assert len(untracked) == 3
			assert "new_file1.txt" in untracked
			assert "new_file2.txt" in untracked
			assert "new_dir/new_file3.txt" in untracked

			# Verify the correct git command was called
			mock_run.assert_called_once()
			args = mock_run.call_args[0][0]
			assert "ls-files" in args
			assert "--others" in args
			assert "--exclude-standard" in args


@pytest.mark.unit
@pytest.mark.git
class TestGitStashOperations(GitTestBase):
	"""Test cases for Git stash operations."""

	def test_stash_staged_changes(self) -> None:
		"""Test stashing staged changes."""
		with (
			patch("codemap.utils.git_utils.get_other_staged_files") as mock_other,
			patch(
				"codemap.utils.git_utils.run_git_command",
			) as mock_run,
		):
			# Test when there are no other files (no stash needed)
			mock_other.return_value = []
			result = stash_staged_changes(["file1.txt"])
			assert not result  # Should return False
			mock_run.assert_not_called()

			# Test when there are other files (stash needed)
			mock_other.return_value = ["file2.txt", "file3.txt"]
			mock_run.reset_mock()
			result = stash_staged_changes(["file1.txt"])
			assert result  # Should return True
			mock_run.assert_called_once()
			assert "stash" in mock_run.call_args[0][0]

	def test_unstash_changes(self) -> None:
		"""Test unstashing changes."""
		with patch("codemap.utils.git_utils.run_git_command") as mock_run:
			# Test when there is a stash
			mock_run.side_effect = [
				"stash@{0}: CodeMap: temporary stash for commit\nstash@{1}: Something else",  # stash list
				"Stash popped",  # stash pop
			]

			unstash_changes()
			assert mock_run.call_count == 2
			assert "stash" in mock_run.call_args_list[0][0][0]
			assert "pop" in mock_run.call_args_list[1][0][0]

			# Test when there's no matching stash
			mock_run.reset_mock()
			mock_run.side_effect = ["stash@{0}: Something else"]

			unstash_changes()
			assert mock_run.call_count == 1  # Only called for stash list


@pytest.mark.unit
@pytest.mark.git
class TestGitCommitOperations(GitTestBase):
	"""Test cases for Git commit operations."""

	def test_commit_only_files_with_hooks(self) -> None:
		"""Test committing only specific files with hook handling."""
		with (
			patch("codemap.utils.git_utils.get_other_staged_files") as mock_other,
			patch(
				"codemap.utils.git_utils.stage_files",
			),
			patch(
				"codemap.utils.git_utils.subprocess.run",
			) as mock_subprocess_run,
			patch(
				"pathlib.Path.exists",
			) as mock_exists,
		):
			# Setup mocks
			mock_other.return_value = []
			mock_exists.return_value = True  # Files exist

			# Test normal commit
			mock_result = Mock()
			mock_result.returncode = 0
			mock_result.stdout = "mock commit output"
			mock_result.stderr = ""
			mock_subprocess_run.return_value = mock_result  # Successful run with stdout

			result = commit_only_files(["file1.txt"], "Test commit")
			assert result == []  # No other staged files

			# Verify subprocess.run was called twice:
			# 1. First call for git status to check for deleted files
			# 2. Second call for the actual commit
			assert mock_subprocess_run.call_count == 2

			# Make sure subprocess.run was called with the expected commands
			status_call = False
			commit_call = False

			for call in mock_subprocess_run.call_args_list:
				args = call[0][0]
				if isinstance(args, list) and "status" in args:
					status_call = True
				elif isinstance(args, list) and "commit" in args:
					commit_call = True

			assert status_call, "Git status command was not called"
			assert commit_call, "Git commit command was not called"


@pytest.mark.unit
@pytest.mark.git
class TestGitStageOperations(GitTestBase):
	"""Test cases for Git staging operations."""

	def test_stage_files_with_deleted_files(self) -> None:
		"""Test staging files that include deleted files."""
		# Create a simpler approach using module-level variables to track calls
		git_command_calls = []

		def mock_run_git_command(args: list[str], **_kwargs) -> str:
			git_command_calls.append(args)
			if args[0:2] == ["git", "ls-files"]:
				return "file1.txt\ndeleted_file.txt\n"
			return ""

		# Apply our patches - more directly patch the internal categorization logic
		with (
			patch("codemap.utils.git_utils.run_git_command", side_effect=mock_run_git_command),
			patch("pathlib.Path.exists", new=lambda path: str(path) == "file1.txt" or str(path).endswith("file1.txt")),
		):
			# Clear call history
			git_command_calls.clear()

			# Test case 1: Existing and tracked deleted files
			stage_files(["file1.txt", "deleted_file.txt"])

			# Expect these calls:
			# 1. git status --porcelain (to check deleted files)
			# 2. git ls-files (to check tracked files)
			# 3. git add file1.txt (for existing file)
			# 4. git rm deleted_file.txt (for deleted file that's tracked)
			assert len(git_command_calls) >= 3
			assert ["git", "status", "--porcelain"] in git_command_calls
			assert ["git", "ls-files"] in git_command_calls

			# Check for appropriate commands based on the file existence
			add_command_found = False
			rm_command_found = False

			for call in git_command_calls:
				if call[0:2] == ["git", "add"] and "file1.txt" in call:
					add_command_found = True
				if call[0:2] == ["git", "rm"] and "deleted_file.txt" in call:
					rm_command_found = True

			assert add_command_found, "Git add command for file1.txt not found"
			assert rm_command_found, "Git rm command for deleted_file.txt not found"

	def test_stage_untracked_deleted_files(self) -> None:
		"""Test staging untracked deleted files."""
		git_command_calls = []

		# Define a function instead of lambda
		def mock_run_git_command_case2(args: list[str], **_kwargs) -> str:
			git_command_calls.append(args)
			if args[0:2] == ["git", "ls-files"]:
				return "file1.txt\n"
			return ""

		# Directly patch the Path.exists method with a simple lambda
		with (
			patch("codemap.utils.git_utils.run_git_command", side_effect=mock_run_git_command_case2),
			patch("pathlib.Path.exists", new=lambda path: str(path) == "file1.txt" or str(path).endswith("file1.txt")),
		):
			# Clear call history
			git_command_calls.clear()

			stage_files(["file1.txt", "untracked.txt"])

			# Should have these calls:
			# 1. git status --porcelain
			# 2. git ls-files (to check tracked files)
			# 3. git add for file1.txt
			assert len(git_command_calls) >= 2
			assert ["git", "status", "--porcelain"] in git_command_calls
			assert ["git", "ls-files"] in git_command_calls

			# Check if file1.txt was added with git add
			add_command_found = False
			for call in git_command_calls:
				if call[0:2] == ["git", "add"] and "file1.txt" in call:
					add_command_found = True
					break

			assert add_command_found, "Git add command for file1.txt not found"

	def test_stage_empty_file_list(self) -> None:
		"""Test staging an empty file list."""
		git_command_calls = []

		def mock_run_git_command(args: list[str], **_kwargs) -> str:
			git_command_calls.append(args)
			return ""

		with patch("codemap.utils.git_utils.run_git_command", side_effect=mock_run_git_command):
			stage_files([])

			# Should have no calls
			assert len(git_command_calls) == 0
