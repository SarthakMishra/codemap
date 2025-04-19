"""Git functionality for CodeMap."""

from .commit import (
    ChunkAction,
    ChunkResult,
    CommitCommand,
    CommitUI,
    DiffChunk,
    DiffSplitter,
    SplitStrategy,
)
from .git import GitWrapper
from .utils import (
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
    "ChunkAction",
    "ChunkResult",
    "CommitCommand",
    "CommitUI",
    "DiffChunk",
    "DiffSplitter",
    "GitDiff",
    "GitError",
    "GitWrapper",
    "SplitStrategy",
    "commit",
    "get_repo_root",
    "get_staged_diff",
    "get_unstaged_diff",
    "get_untracked_files",
    "stage_files",
    "unstage_files",
]
