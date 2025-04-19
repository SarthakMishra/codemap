"""Git utilities for CodeMap."""

from .git_utils import (
    GitDiff,
    GitError,
    commit,
    get_repo_root,
    get_staged_diff,
    get_unstaged_diff,
    get_untracked_files,
    stage_files,
    unstage_files,
)

__all__ = [
    "GitDiff",
    "GitError",
    "commit",
    "get_repo_root",
    "get_staged_diff",
    "get_unstaged_diff",
    "get_untracked_files",
    "stage_files",
    "unstage_files",
]
