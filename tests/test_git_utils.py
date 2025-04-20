"""Tests for the Git utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


def test_get_other_staged_files() -> None:
    """Test getting other staged files."""
    from codemap.git.utils.git_utils import get_other_staged_files

    with patch("codemap.git.utils.git_utils.run_git_command") as mock_run:
        # Mock the git diff command to return a list of staged files
        mock_run.return_value = "file1.txt\nfile2.txt\nfile3.txt"

        # Get files not in the target list
        other_files = get_other_staged_files(["file1.txt"])

        # Assert correct files returned
        assert len(other_files) == 2
        assert "file2.txt" in other_files
        assert "file3.txt" in other_files
        assert "file1.txt" not in other_files


def test_stash_staged_changes() -> None:
    """Test stashing staged changes."""
    from codemap.git.utils.git_utils import stash_staged_changes

    with patch("codemap.git.utils.git_utils.get_other_staged_files") as mock_other, patch(
        "codemap.git.utils.git_utils.run_git_command",
    ) as mock_run:
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


def test_unstash_changes() -> None:
    """Test unstashing changes."""
    from codemap.git.utils.git_utils import unstash_changes

    with patch("codemap.git.utils.git_utils.run_git_command") as mock_run:
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


def test_commit_only_files_with_hooks() -> None:
    """Test committing only specific files with hook handling."""
    from codemap.git.utils.git_utils import commit_only_files

    with patch("codemap.git.utils.git_utils.get_other_staged_files") as mock_other, patch(
        "codemap.git.utils.git_utils.stage_files",
    ), patch(
        "codemap.git.utils.git_utils.subprocess.run",
    ) as mock_subprocess_run, patch(
        "codemap.git.utils.git_utils.Path.exists",
    ) as mock_exists:
        # Setup mocks
        mock_other.return_value = []
        mock_exists.return_value = True  # Assume file exists to simplify test

        # Test normal commit
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b"mock commit output"
        mock_subprocess_run.return_value = mock_result  # Successful run with stdout

        result = commit_only_files(["file1.txt"], "Test commit")
        assert result == []  # No other staged files

        # Verify subprocess.run was called once for the commit
        assert mock_subprocess_run.call_count == 1

        # The call is for the commit
        commit_call = mock_subprocess_run.call_args_list[0]
        commit_args, commit_kwargs = commit_call
        assert "git commit" in commit_args[0]
        assert "Test commit" in commit_args[0]
        assert commit_kwargs["shell"] is True


def test_get_other_staged_files_error() -> None:
    """Test error handling in get_other_staged_files."""
    from codemap.git.utils.git_utils import GitError, get_other_staged_files

    with patch("codemap.git.utils.git_utils.run_git_command") as mock_run:
        # Mock a git error
        mock_run.side_effect = GitError("Git command failed")

        # Function should raise GitError
        with pytest.raises(GitError) as excinfo:
            get_other_staged_files(["file1.txt"])

        # Check error message is appropriate
        assert "Failed to check for other staged files" in str(excinfo.value)


def test_get_untracked_files() -> None:
    """Test getting untracked files."""
    from codemap.git.utils.git_utils import get_untracked_files

    with patch("codemap.git.utils.git_utils.run_git_command") as mock_run:
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


def test_stage_files_with_deleted_files() -> None:
    """Test staging files that include deleted files."""
    from codemap.git.utils.git_utils import stage_files

    # Create a simpler approach using module-level variables to track calls
    git_command_calls = []

    def mock_run_git_command(args: list[str], **_kwargs) -> str:
        git_command_calls.append(args)
        if args[0:2] == ["git", "ls-files"]:
            return "file1.txt\ndeleted_file.txt\n"
        return ""

    def mock_path_exists(path: str) -> bool:
        # Return True for file1.txt, False for others
        return str(path) == "file1.txt"

    # Apply our patches
    with patch("codemap.git.utils.git_utils.run_git_command", side_effect=mock_run_git_command), patch.object(
        Path, "exists", return_value=False
    ), patch("codemap.git.utils.git_utils.Path") as mock_path:
        # Setup Path mock to return different exists values
        path_instances = {}

        def get_mock_path(file_path: str) -> MagicMock:
            if file_path not in path_instances:
                mock_instance = MagicMock()
                mock_instance.exists.return_value = file_path == "file1.txt"
                path_instances[file_path] = mock_instance
            return path_instances[file_path]

        mock_path.side_effect = get_mock_path

        # Clear call history
        git_command_calls.clear()

        # Test case 1: Existing and tracked deleted files
        stage_files(["file1.txt", "deleted_file.txt"])

        # Expect these calls:
        # 1. git add file1.txt (for existing file)
        # 2. git ls-files (to check tracked files)
        # 3. git rm deleted_file.txt (for deleted file that's tracked)
        assert len(git_command_calls) == 3
        assert git_command_calls[0] == ["git", "add", "file1.txt"]
        assert git_command_calls[1] == ["git", "ls-files"]
        assert git_command_calls[2] == ["git", "rm", "deleted_file.txt"]

        # Test case 2: untracked deleted files
        git_command_calls.clear()

        # Define a function instead of lambda
        def mock_run_git_command_case2(args: list[str], **_kwargs) -> str:
            git_command_calls.append(args)
            if args[0:2] == ["git", "ls-files"]:
                return "file1.txt\n"
            return ""

        with patch("codemap.git.utils.git_utils.run_git_command", side_effect=mock_run_git_command_case2):
            stage_files(["file1.txt", "untracked.txt"])

            # Should have 2 calls: git add for file1.txt, git ls-files
            # (no git rm for untracked)
            assert len(git_command_calls) == 2
            assert git_command_calls[0] == ["git", "add", "file1.txt"]
            assert git_command_calls[1] == ["git", "ls-files"]

        # Test case 3: empty file list
        git_command_calls.clear()
        stage_files([])

        # Should have no calls
        assert len(git_command_calls) == 0
