"""Pull request utilities for CodeMap."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from codemap.utils.git_utils import GitError, run_git_command

logger = logging.getLogger(__name__)


@dataclass
class PullRequest:
    """Represents a GitHub Pull Request."""

    branch: str
    title: str
    description: str
    url: str | None = None
    number: int | None = None


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
    except GitError:
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


def get_commit_messages(base_branch: str, head_branch: str) -> list[str]:
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


def generate_pr_title_from_commits(commits: list[str]) -> str:
    """Generate a PR title from commit messages.

    Args:
        commits: List of commit messages

    Returns:
        Generated PR title
    """
    if not commits:
        return "Update branch"

    # Define mapping from commit prefixes to PR title prefixes
    prefix_mapping = {"feat": "Feature:", "fix": "Fix:", "docs": "Docs:", "refactor": "Refactor:", "perf": "Optimize:"}

    for prefix, title_prefix in prefix_mapping.items():
        for commit in commits:
            if commit.startswith(prefix):
                # Strip the prefix and use as title
                title = re.sub(r"^[a-z]+(\([^)]+\))?:\s*", "", commit)
                # Capitalize first letter and add PR type prefix
                return f"{title_prefix} {title[0].upper() + title[1:]}"

    # Fallback to first commit
    title = re.sub(r"^[a-z]+(\([^)]+\))?:\s*", "", commits[0])
    return title[0].upper() + title[1:]


def generate_pr_title_with_llm(
    commits: list[str],
    model: str | None = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Generate a PR title using an LLM.

    Args:
        commits: List of commit messages
        model: LLM model to use
        api_key: API key for LLM provider
        api_base: Custom API base URL

    Returns:
        Generated PR title
    """
    import logging

    from codemap.utils.llm_utils import generate_text_with_llm

    logger = logging.getLogger(__name__)

    # Ensure model is never None
    actual_model = model or "gpt-4o-mini"

    if not commits:
        return "Update branch"

    try:
        # Format commit messages
        commit_list = "\n".join([f"- {commit}" for commit in commits])

        # Prepare prompt
        prompt = """Based on the following commits, generate a clear, concise PR title that captures the
essence of the changes.
        Follow these guidelines:
        - Focus on the most important change
        - If there are multiple related changes, summarize them
        - Keep it under 80 characters
        - Start with a capital letter
        - Don't use a period at the end
        - Use present tense (e.g., "Add feature" not "Added feature")
        - Be descriptive and specific (e.g., "Fix memory leak in data processing" not just "Fix bug")
        - Include the type of change if clear (Feature, Fix, Refactor, etc.)

        Commits:
        """

        prompt += commit_list + "\n\n        PR Title:"

        # Call LLM with repo_path used for context
        title = generate_text_with_llm(prompt, actual_model, api_key, api_base)

        # Clean up the title
        title = title.strip()
        if title.endswith("."):
            title = title[:-1]

        return title
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.warning("Failed to generate PR title with LLM: %s", str(e))
        # Fallback to rule-based approach
        return generate_pr_title_from_commits(commits)


def generate_pr_description_from_commits(commits: list[str]) -> str:
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
    refactors = []
    optimizations = []
    other = []

    for commit in commits:
        if commit.startswith("feat"):
            features.append(commit)
        elif commit.startswith("fix"):
            fixes.append(commit)
        elif commit.startswith("docs"):
            docs.append(commit)
        elif commit.startswith("refactor"):
            refactors.append(commit)
        elif commit.startswith("perf"):
            optimizations.append(commit)
        else:
            other.append(commit)

    # Determine PR type checkboxes
    has_refactor = bool(refactors)
    has_feature = bool(features)
    has_bug_fix = bool(fixes)
    has_optimization = bool(optimizations)
    has_docs_update = bool(docs)

    # Build description
    description = "## What type of PR is this? (check all applicable)\n\n"
    description += f"- [{' ' if not has_refactor else 'x'}] Refactor\n"
    description += f"- [{' ' if not has_feature else 'x'}] Feature\n"
    description += f"- [{' ' if not has_bug_fix else 'x'}] Bug Fix\n"
    description += f"- [{' ' if not has_optimization else 'x'}] Optimization\n"
    description += f"- [{' ' if not has_docs_update else 'x'}] Documentation Update\n\n"

    description += "## Description\n\n"

    # Add categorized changes to description
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

    if refactors:
        description += "### Refactors\n\n"
        for refactor in refactors:
            clean_msg = re.sub(r"^refactor(\([^)]+\))?:\s*", "", refactor)
            description += f"- {clean_msg}\n"
        description += "\n"

    if optimizations:
        description += "### Optimizations\n\n"
        for perf in optimizations:
            clean_msg = re.sub(r"^perf(\([^)]+\))?:\s*", "", perf)
            description += f"- {clean_msg}\n"
        description += "\n"

    if other:
        description += "### Other\n\n"
        for msg in other:
            # Try to clean up conventional commit prefixes
            clean_msg = re.sub(r"^(style|test|build|ci|chore|revert)(\([^)]+\))?:\s*", "", msg)
            description += f"- {clean_msg}\n"
        description += "\n"

    description += "## Related Tickets & Documents\n\n"
    description += "- Related Issue #\n"
    description += "- Closes #\n\n"

    description += "## Added/updated tests?\n\n"
    description += "- [ ] Yes\n"
    description += (
        "- [ ] No, and this is why: _please replace this line with details on why tests have not been included_\n"
    )
    description += "- [ ] I need help with writing tests\n"

    return description


def generate_pr_description_with_llm(
    commits: list[str],
    model: str | None = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Generate a PR description using an LLM.

    Args:
        commits: List of commit messages
        model: LLM model to use
        api_key: API key for LLM provider
        api_base: Custom API base URL

    Returns:
        Generated PR description
    """
    import logging

    from codemap.utils.llm_utils import generate_text_with_llm

    logger = logging.getLogger(__name__)

    # Ensure model is never None
    actual_model = model or "gpt-4o-mini"

    if not commits:
        return "No changes"

    try:
        # Format commit messages
        commit_list = "\n".join([f"- {commit}" for commit in commits])

        # Prepare prompt
        prompt = f"""Based on the following commits, generate a comprehensive PR description following this template:

        ## What type of PR is this? (check all applicable)

        - [ ] Refactor
        - [ ] Feature
        - [ ] Bug Fix
        - [ ] Optimization
        - [ ] Documentation Update

        ## Description
        [Fill this section with a detailed description of the changes]

        ## Related Tickets & Documents
        - Related Issue #
        - Closes #

        ## Added/updated tests?
        - [ ] Yes
        - [ ] No, and this is why: [explanation]
        - [ ] I need help with writing tests

        Consider the following guidelines:
        - Check the appropriate PR type boxes based on the commit messages
        - Provide a clear, detailed description of the changes
        - Include any relevant issue numbers that this PR relates to or closes
        - Indicate if tests were added, and if not, explain why
        - Use bullet points for clarity

        Commits:
        {commit_list}

        PR Description:"""

        # Call LLM with repo_path used for context
        return generate_text_with_llm(prompt, actual_model, api_key, api_base)
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.warning("Failed to generate PR description with LLM: %s", str(e))
        # Fallback to rule-based approach
        return generate_pr_description_from_commits(commits)


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

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
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


def update_pull_request(pr_number: int | None, title: str, description: str) -> PullRequest:
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
    if pr_number is None:
        msg = "PR number cannot be None"
        raise GitError(msg)

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

        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603

        # Get PR URL
        url_cmd = ["gh", "pr", "view", str(pr_number), "--json", "url", "--jq", ".url"]
        result = subprocess.run(url_cmd, check=True, capture_output=True, text=True)  # noqa: S603
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


def get_existing_pr(branch_name: str) -> PullRequest | None:
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

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode != 0 or not result.stdout.strip():
            return None

        # Parse JSON output
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


def suggest_branch_name(commits: list[str]) -> str:
    """Suggest a branch name based on commit messages.

    Args:
        commits: List of commit messages

    Returns:
        Suggested branch name
    """
    if not commits:
        # If no commits, use a timestamp
        return f"update-{get_timestamp()}"

    # Use the first commit as the basis for the branch name
    first_commit = commits[0]

    # Extract the type and scope if it's a conventional commit
    pattern = (
        r"^(feat|fix|docs|refactor|style|perf|test|build|ci|chore|revert)"
        r"(?:\(([^)]+)\))?:\s*(.+)$"
    )
    match = re.match(pattern, first_commit)

    if match:
        commit_type, scope, subject = match.groups()
        branch_prefix = f"{commit_type}-{scope}" if scope else commit_type
    else:
        # Not a conventional commit, use a generic prefix
        branch_prefix = "update"

    # Extract a few words from the subject for the branch name
    subject = match.group(3) if match else first_commit

    # Clean up the subject and take first few words
    subject = re.sub(r"[^\w\s-]", "", subject).lower()
    subject = re.sub(r"\s+", "-", subject)
    words = subject.split("-")
    short_subject = "-".join(words[:3])  # Take up to 3 words

    # Build the branch name
    branch_name = f"{branch_prefix}-{short_subject}"

    # Ensure the branch name meets git's requirements
    branch_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", branch_name)
    branch_name = re.sub(r"-+", "-", branch_name)  # Replace multiple dashes with a single one

    return branch_name.strip("-")


def get_timestamp() -> str:
    """Get a timestamp for branch naming.

    Returns:
        Timestamp string in format YYYYmmdd-HHMMSS
    """
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
