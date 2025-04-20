"""Pull request utilities for CodeMap."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from codemap.git.utils.git_utils import GitError, run_git_command

logger = logging.getLogger(__name__)


@dataclass
class PullRequest:
    """Represents a GitHub Pull Request."""

    branch: str
    title: str
    description: str
    url: Optional[str] = None
    number: Optional[int] = None


def get_current_branch() -> str:
    """Get the name of the current branch.

    Returns:
        Name of the current branch

    Raises:
        GitError: If git command fails
    """
    try:
        return run_git_command(["git", "branch", "--show-current"]).strip()
    except GitError as e:
        msg = "Failed to get current branch"
        raise GitError(msg) from e


def get_default_branch() -> str:
    """Get the default branch of the repository.

    Returns:
        Name of the default branch (usually main or master)

    Raises:
        GitError: If git command fails
    """
    try:
        # Try to get the default branch from the remote
        remote_info = run_git_command(["git", "remote", "show", "origin"])
        match = re.search(r"HEAD branch: (\S+)", remote_info)
        if match:
            return match.group(1)

        # Fallback to checking if main or master exists
        branches = run_git_command(["git", "branch", "-r"]).splitlines()
        if any("origin/main" in branch for branch in branches):
            return "main"
        if any("origin/master" in branch for branch in branches):
            return "master"

        # Last resort, use the current branch
        return get_current_branch()
    except GitError as e:
        msg = "Failed to determine default branch, using 'main'"
        logger.warning(msg)
        return "main"


def create_branch(branch_name: str) -> None:
    """Create a new branch and switch to it.

    Args:
        branch_name: Name of the branch to create

    Raises:
        GitError: If git command fails
    """
    try:
        run_git_command(["git", "checkout", "-b", branch_name])
    except GitError as e:
        msg = f"Failed to create branch: {branch_name}"
        raise GitError(msg) from e


def checkout_branch(branch_name: str) -> None:
    """Checkout an existing branch.

    Args:
        branch_name: Name of the branch to checkout

    Raises:
        GitError: If git command fails
    """
    try:
        run_git_command(["git", "checkout", branch_name])
    except GitError as e:
        msg = f"Failed to checkout branch: {branch_name}"
        raise GitError(msg) from e


def branch_exists(branch_name: str, include_remote: bool = True) -> bool:
    """Check if a branch exists.

    Args:
        branch_name: Name of the branch to check
        include_remote: Whether to check remote branches as well

    Returns:
        True if the branch exists, False otherwise
    """
    try:
        branches = run_git_command(["git", "branch", "--list", branch_name]).strip()
        if branches:
            return True

        if include_remote:
            remote_branches = run_git_command(["git", "branch", "-r", "--list", f"origin/{branch_name}"]).strip()
            return bool(remote_branches)

        return False
    except GitError:
        return False


def push_branch(branch_name: str, force: bool = False) -> None:
    """Push a branch to the remote.

    Args:
        branch_name: Name of the branch to push
        force: Whether to force push

    Raises:
        GitError: If git command fails
    """
    try:
        cmd = ["git", "push", "-u", "origin", branch_name]
        if force:
            cmd.insert(2, "--force")
        run_git_command(cmd)
    except GitError as e:
        msg = f"Failed to push branch: {branch_name}"
        raise GitError(msg) from e


def get_commit_messages(base_branch: str, head_branch: str) -> List[str]:
    """Get commit messages between two branches.

    Args:
        base_branch: Base branch (e.g., main)
        head_branch: Head branch (e.g., feature-branch)

    Returns:
        List of commit messages

    Raises:
        GitError: If git command fails
    """
    try:
        # Get commit messages between base and head
        log_output = run_git_command(["git", "log", f"{base_branch}..{head_branch}", "--pretty=format:%s"])
        return log_output.splitlines() if log_output.strip() else []
    except GitError as e:
        msg = f"Failed to get commit messages between {base_branch} and {head_branch}"
        raise GitError(msg) from e


def generate_pr_title_from_commits(commits: List[str]) -> str:
    """Generate a PR title from commit messages.

    Args:
        commits: List of commit messages

    Returns:
        Generated PR title
    """
    if not commits:
        return "Update branch"

    # Use the first commit message as the PR title
    title = commits[0]

    # Remove any conventional commit prefixes (e.g., "feat: ", "fix: ")
    title = re.sub(r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([^)]+\))?:\s*", "", title)

    # Capitalize the first letter
    if title:
        title = title[0].upper() + title[1:]

    return title


def generate_pr_description_from_commits(commits: List[str]) -> str:
    """Generate a PR description from commit messages.

    Args:
        commits: List of commit messages

    Returns:
        Generated PR description
    """
    if not commits:
        return "No changes"

    # Group commits by type
    features = []
    fixes = []
    docs = []
    other = []

    for commit in commits:
        if commit.startswith("feat"):
            features.append(commit)
        elif commit.startswith("fix"):
            fixes.append(commit)
        elif commit.startswith("docs"):
            docs.append(commit)
        else:
            other.append(commit)

    # Build description
    description = "## Changes\n\n"

    if features:
        description += "### Features\n\n"
        for feat in features:
            # Remove the prefix and format as a list item
            clean_msg = re.sub(r"^feat(\([^)]+\))?:\s*", "", feat)
            description += f"- {clean_msg}\n"
        description += "\n"

    if fixes:
        description += "### Fixes\n\n"
        for fix in fixes:
            clean_msg = re.sub(r"^fix(\([^)]+\))?:\s*", "", fix)
            description += f"- {clean_msg}\n"
        description += "\n"

    if docs:
        description += "### Documentation\n\n"
        for doc in docs:
            clean_msg = re.sub(r"^docs(\([^)]+\))?:\s*", "", doc)
            description += f"- {clean_msg}\n"
        description += "\n"

    if other:
        description += "### Other\n\n"
        for msg in other:
            # Try to clean up conventional commit prefixes
            clean_msg = re.sub(r"^(refactor|style|perf|test|build|ci|chore|revert)(\([^)]+\))?:\s*", "", msg)
            description += f"- {clean_msg}\n"
        description += "\n"

    return description


def create_pull_request(base_branch: str, head_branch: str, title: str, description: str) -> PullRequest:
    """Create a pull request on GitHub.

    Args:
        base_branch: Base branch (e.g., main)
        head_branch: Head branch (e.g., feature-branch)
        title: PR title
        description: PR description

    Returns:
        PullRequest object with PR details

    Raises:
        GitError: If PR creation fails
    """
    try:
        # Check if gh CLI is installed
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True, text=True)  # noqa: S603, S607
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msg = "GitHub CLI (gh) is not installed or not in PATH. Please install it to create PRs."
            raise GitError(msg) from e

        # Create PR using GitHub CLI
        cmd = [
            "gh",
            "pr",
            "create",
            "--base",
            base_branch,
            "--head",
            head_branch,
            "--title",
            title,
            "--body",
            description,
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603, S607
        output = result.stdout.strip()

        # Extract PR URL and number
        pr_url = output.strip()
        pr_number = None

        # Try to extract PR number from URL
        match = re.search(r"/pull/(\d+)$", pr_url)
        if match:
            pr_number = int(match.group(1))

        return PullRequest(
            branch=head_branch,
            title=title,
            description=description,
            url=pr_url,
            number=pr_number,
        )
    except subprocess.CalledProcessError as e:
        msg = f"Failed to create PR: {e.stderr}"
        raise GitError(msg) from e


def update_pull_request(pr_number: int, title: str, description: str) -> PullRequest:
    """Update an existing pull request.

    Args:
        pr_number: PR number
        title: New PR title
        description: New PR description

    Returns:
        Updated PullRequest object

    Raises:
        GitError: If PR update fails
    """
    try:
        # Check if gh CLI is installed
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True, text=True)  # noqa: S603, S607
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msg = "GitHub CLI (gh) is not installed or not in PATH. Please install it to update PRs."
            raise GitError(msg) from e

        # Get current branch
        branch = get_current_branch()

        # Update PR using GitHub CLI
        cmd = [
            "gh",
            "pr",
            "edit",
            str(pr_number),
            "--title",
            title,
            "--body",
            description,
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603, S607

        # Get PR URL
        url_cmd = ["gh", "pr", "view", str(pr_number), "--json", "url", "--jq", ".url"]
        result = subprocess.run(url_cmd, check=True, capture_output=True, text=True)  # noqa: S603, S607
        pr_url = result.stdout.strip()

        return PullRequest(
            branch=branch,
            title=title,
            description=description,
            url=pr_url,
            number=pr_number,
        )
    except subprocess.CalledProcessError as e:
        msg = f"Failed to update PR: {e.stderr}"
        raise GitError(msg) from e


def get_existing_pr(branch_name: str) -> Optional[PullRequest]:
    """Get an existing PR for a branch.

    Args:
        branch_name: Branch name

    Returns:
        PullRequest object if found, None otherwise
    """
    try:
        # Check if gh CLI is installed
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True, text=True)  # noqa: S603, S607
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        # List PRs for the branch
        cmd = [
            "gh",
            "pr",
            "list",
            "--head",
            branch_name,
            "--json",
            "number,title,body,url",
            "--jq",
            ".[0]",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603, S607
        if result.returncode != 0 or not result.stdout.strip():
            return None

        # Parse JSON output
        import json

        pr_data = json.loads(result.stdout)
        if not pr_data:
            return None

        return PullRequest(
            branch=branch_name,
            title=pr_data.get("title", ""),
            description=pr_data.get("body", ""),
            url=pr_data.get("url", ""),
            number=pr_data.get("number"),
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def suggest_branch_name(commits: List[str]) -> str:
    """Suggest a branch name based on commit messages.

    Args:
        commits: List of commit messages

    Returns:
        Suggested branch name
    """
    if not commits:
        return f"update-{get_timestamp()}"

    # Use the first commit message to generate a branch name
    first_commit = commits[0]

    # Extract the type and scope if it's a conventional commit
    match = re.match(r"^(feat|fix|docs|refactor|style|perf|test|build|ci|chore|revert)(?:\(([^)]+)\))?:\s*(.+)$", first_commit)
    
    if match:
        commit_type, scope, subject = match.groups()
        if scope:
            branch_prefix = f"{commit_type}-{scope}"
        else:
            branch_prefix = commit_type
    else:
        # Not a conventional commit, use a generic prefix
        branch_prefix = "update"

    # Extract a few words from the subject for the branch name
    if match:
        subject = match.group(3)
    else:
        subject = first_commit

    # Clean up the subject and take first few words
    words = re.sub(r"[^\w\s-]", "", subject.lower()).split()[:3]
    branch_suffix = "-".join(words)

    # Combine prefix and suffix
    branch_name = f"{branch_prefix}-{branch_suffix}"

    # Ensure the branch name is valid
    branch_name = re.sub(r"[^\w-]", "-", branch_name)
    branch_name = re.sub(r"-+", "-", branch_name)  # Replace multiple hyphens with a single one
    branch_name = branch_name.strip("-")

    return branch_name


def get_timestamp() -> str:
    """Get a timestamp string for branch names.

    Returns:
        Timestamp string in format YYYYMMDD-HHMMSS
    """
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")