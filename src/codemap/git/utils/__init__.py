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

try:
    from .pr_utils import (
        PullRequest,
        branch_exists,
        checkout_branch,
        create_branch,
        create_pull_request,
        generate_pr_description_from_commits,
        generate_pr_title_from_commits,
        get_commit_messages,
        get_current_branch,
        get_default_branch,
        get_existing_pr,
        push_branch,
        suggest_branch_name,
        update_pull_request,
    )
except ImportError:
    # PR utils might not be available in all environments
    pass

__all__ = [
    "GitDiff",
    "GitError",
    "PullRequest",
    "branch_exists",
    "checkout_branch",
    "commit",
    "create_branch",
    "create_pull_request",
    "generate_pr_description_from_commits",
    "generate_pr_title_from_commits",
    "get_commit_messages",
    "get_current_branch",
    "get_default_branch",
    "get_existing_pr",
    "get_repo_root",
    "get_staged_diff",
    "get_unstaged_diff",
    "get_untracked_files",
    "push_branch",
    "stage_files",
    "suggest_branch_name",
    "unstage_files",
    "update_pull_request",
]
