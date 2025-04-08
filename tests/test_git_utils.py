"""Tests for the Git utilities."""

from unittest.mock import patch

import pytest


def test_get_other_staged_files() -> None:
    """Test getting other staged files."""
    from codemap.utils.git_utils import get_other_staged_files

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


def test_stash_staged_changes() -> None:
    """Test stashing staged changes."""
    from codemap.utils.git_utils import stash_staged_changes

    with patch("codemap.utils.git_utils.get_other_staged_files") as mock_other, patch(
        "codemap.utils.git_utils.run_git_command",
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
    from codemap.utils.git_utils import unstash_changes

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


def test_commit_only_files_with_hooks() -> None:
    """Test committing only specific files with hook handling."""
    from codemap.utils.git_utils import GitError, commit_only_files

    with patch("codemap.utils.git_utils.get_other_staged_files") as mock_other, patch(
        "codemap.utils.git_utils.run_git_command",
    ) as mock_run:
        # Setup mocks
        mock_other.return_value = []

        # Test normal commit
        commit_only_files(["file1.txt"], "Test commit")
        assert mock_run.call_count == 2  # add + commit

        # Test hook failure and bypass
        mock_run.reset_mock()
        mock_run.side_effect = [
            None,  # git add succeeds
            GitError("Failed: hook script returned error"),  # First commit fails
            None,  # Second commit with --no-verify succeeds
        ]

        # Should fail first time
        with pytest.raises(GitError):
            commit_only_files(["file1.txt"], "Test commit")

        # Reset for next test
        mock_run.reset_mock()
        mock_run.side_effect = [
            None,  # git add succeeds
            GitError("Failed: hook script returned error"),  # First commit fails
            None,  # Second commit with --no-verify succeeds
        ]

        # Should succeed with ignore_hooks=True
        commit_only_files(["file1.txt"], "Test commit", ignore_hooks=True)
        assert mock_run.call_count == 3  # add + failed commit + successful commit with --no-verify
        assert "--no-verify" in mock_run.call_args[0][0]


def test_get_other_staged_files_error() -> None:
    """Test error handling in get_other_staged_files."""
    from codemap.utils.git_utils import GitError, get_other_staged_files

    with patch("codemap.utils.git_utils.run_git_command") as mock_run:
        # Mock a git error
        mock_run.side_effect = GitError("Git command failed")

        # Function should raise GitError
        with pytest.raises(GitError) as excinfo:
            get_other_staged_files(["file1.txt"])

        # Check error message is appropriate
        assert "Failed to check for other staged files" in str(excinfo.value)


def test_get_untracked_files() -> None:
    """Test getting untracked files."""
    from codemap.utils.git_utils import get_untracked_files

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
