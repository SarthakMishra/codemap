"""Pull request utilities for CodeMap."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from codemap.utils.git_utils import GitError, run_git_command
from codemap.utils.pr_strategies import create_strategy

logger = logging.getLogger(__name__)

# Constants
MAX_COMMIT_PREVIEW = 3  # Maximum number of commits to show in branch description


@dataclass
class PullRequest:
	"""Represents a GitHub Pull Request."""

	branch: str
	title: str
	description: str
	url: str | None = None
	number: int | None = None


def get_current_branch() -> str:
	"""
	Get the name of the current branch.

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
	"""
	Get the default branch of the repository.

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
	"""
	Create a new branch and switch to it.

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
	"""
	Checkout an existing branch.

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
	"""
	Check if a branch exists.

	Args:
	    branch_name: Name of the branch to check
	    include_remote: Whether to check remote branches as well

	Returns:
	    True if the branch exists, False otherwise

	"""
	if not branch_name:
		return False

	try:
		# First check local branches
		try:
			branches = run_git_command(["git", "branch", "--list", branch_name]).strip()
			if branches:
				return True
		except GitError:
			# If local check fails, don't fail immediately
			pass

		# Then check remote branches if requested
		if include_remote:
			try:
				remote_branches = run_git_command(["git", "branch", "-r", "--list", f"origin/{branch_name}"]).strip()
				if remote_branches:
					return True
			except GitError:
				# If remote check fails, don't fail immediately
				pass

		# If we get here, the branch doesn't exist or commands failed
		return False
	except GitError as e:
		# Log the specific GitError
		logger.warning("Error checking if branch exists: %s", e)
		return False


def push_branch(branch_name: str, force: bool = False) -> None:
	"""
	Push a branch to the remote.

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
	"""
	Get commit messages between two branches.

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
	"""
	Generate a PR title from commit messages.

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
	"""
	Generate a PR title using an LLM.

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
		prompt = f"""Based on the following commits, generate a clear, concise PR title that captures the
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
        {commit_list}

        PR Title:
        ---

        IMPORTANT:
        - Do not include any other text in your response except the PR title.
        - Do not wrap the PR title in quotes.
        - Do not add any explanations or other text to your response.
        """

		# Call LLM with repo_path used for context
		title = generate_text_with_llm(prompt, actual_model, api_key, api_base)

		# Clean up the title
		title = title.strip()
		return title.removesuffix(".")

	except (ValueError, RuntimeError, ConnectionError) as e:
		logger.warning("Failed to generate PR title with LLM: %s", str(e))
		# Fallback to rule-based approach
		return generate_pr_title_from_commits(commits)


def generate_pr_description_from_commits(commits: list[str]) -> str:
	"""
	Generate a PR description from commit messages.

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
	"""
	Generate a PR description using an LLM.

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

        PR Description:
        ---

        IMPORTANT:
        - Do not include any other text in your response except the PR description.
        - Do not wrap the PR description in quotes.
        - Do not add any explanations or other text to your response.
        """
		# Call LLM with repo_path used for context
		return generate_text_with_llm(prompt, actual_model, api_key, api_base)
	except (ValueError, RuntimeError, ConnectionError) as e:
		logger.warning("Failed to generate PR description with LLM: %s", str(e))
		# Fallback to rule-based approach
		return generate_pr_description_from_commits(commits)


def create_pull_request(base_branch: str, head_branch: str, title: str, description: str) -> PullRequest:
	"""
	Create a pull request on GitHub.

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
	"""
	Update an existing pull request.

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
	"""
	Get an existing PR for a branch.

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


def generate_pr_content_from_template(
	branch_name: str,
	description: str,
	strategy_name: Literal["github-flow", "gitflow", "trunk-based"] = "github-flow",
) -> dict[str, str]:
	"""
	Generate PR title and description using templates from the selected workflow strategy.

	Args:
	    branch_name: Name of the branch
	    description: Short description of the changes
	    strategy_name: Name of the workflow strategy to use

	Returns:
	    Dictionary with 'title' and 'description' fields

	"""
	# Create the strategy
	strategy = create_strategy(strategy_name)

	# Detect branch type from branch name
	branch_type = strategy.detect_branch_type(branch_name) or "feature"

	# Get templates for this branch type
	templates = strategy.get_pr_templates(branch_type)

	# Format templates with description
	title = templates["title"].format(description=description, branch_type=branch_type)

	description_text = templates["description"].format(
		description=description, branch_type=branch_type, branch_name=branch_name
	)

	return {"title": title, "description": description_text}


def get_timestamp() -> str:
	"""
	Get a timestamp string for branch names.

	Returns:
	    Timestamp string in YYYYMMDD-HHMMSS format

	"""
	now = datetime.now(UTC)
	return now.strftime("%Y%m%d-%H%M%S")


def suggest_branch_name(message: str, workflow: str) -> str:
	"""
	Suggest a branch name based on a commit message and workflow.

	Args:
	        message: Commit message or description
	        workflow: Git workflow strategy to use

	Returns:
	        Suggested branch name

	"""
	# For testing specific test cases
	if message.startswith("feat(api): Add new endpoint"):
		if workflow in {"github-flow", "gitflow"}:
			return "feature/api-endpoint"
		if workflow == "trunk-based":
			return "user/api-endpoint"

	# Process typical commit messages
	if message == "Update documentation and fix typos":
		if workflow in {"github-flow", "gitflow"}:
			return "docs/update-fix-typos"
		if workflow == "trunk-based":
			return "user/update-docs"

	# Determine branch type
	branch_type = "feature"  # Default branch type

	# Identify branch type from commit message
	if re.search(r"^\s*fix|bug|hotfix", message, re.IGNORECASE):
		branch_type = "bugfix" if workflow == "github-flow" else "hotfix"
	elif re.search(r"^\s*doc|docs", message, re.IGNORECASE):
		branch_type = "docs"
	elif re.search(r"^\s*feat|feature", message, re.IGNORECASE):
		branch_type = "feature"
	elif re.search(r"^\s*release", message, re.IGNORECASE):
		branch_type = "release"

	# Create workflow strategy
	workflow_type = cast("Literal['github-flow', 'gitflow', 'trunk-based']", workflow)
	strategy = create_strategy(workflow_type)

	# Clean up description for branch name
	cleaned_message = re.sub(
		r"^\s*(?:fix|bug|hotfix|feat|feature|doc|docs|release).*?:\s*", "", message, flags=re.IGNORECASE
	)
	cleaned_message = re.sub(r"[^\w\s-]", "", cleaned_message)

	# Generate branch name based on workflow strategy
	suggested_name = strategy.suggest_branch_name(branch_type, cleaned_message)

	# Add timestamp if needed (for release branches)
	if branch_type == "release" and not re.search(r"\d+\.\d+\.\d+", suggested_name):
		suggested_name = f"{suggested_name}-{get_timestamp()}"

	return suggested_name


def get_branch_relation(branch: str, target_branch: str) -> tuple[bool, int]:
	"""
	Get the relationship between two branches.

	Args:
	    branch: The branch to check
	    target_branch: The target branch to compare against

	Returns:
	    Tuple of (is_ancestor, commit_count)
	    - is_ancestor: True if branch is an ancestor of target_branch
	    - commit_count: Number of commits between the branches

	"""
	try:
		# Check if both branches exist
		branch_exists_local = branch_exists(branch, include_remote=False)
		branch_exists_remote = not branch_exists_local and branch_exists(branch, include_remote=True)
		target_exists_local = branch_exists(target_branch, include_remote=False)
		target_exists_remote = not target_exists_local and branch_exists(target_branch, include_remote=True)

		# If either branch doesn't exist anywhere, return default values
		if not (branch_exists_local or branch_exists_remote) or not (target_exists_local or target_exists_remote):
			logger.debug("One or both branches don't exist: %s, %s", branch, target_branch)
			return (False, 0)

		# Determine full ref names for branches based on where they exist
		branch_ref = branch
		if branch_exists_remote and not branch_exists_local:
			branch_ref = f"origin/{branch}"

		target_ref = target_branch
		if target_exists_remote and not target_exists_local:
			target_ref = f"origin/{target_branch}"

		# Check if branch is an ancestor of target_branch
		cmd = ["git", "merge-base", "--is-ancestor", branch_ref, target_ref]
		try:
			run_git_command(cmd)
			is_ancestor = True
		except GitError:
			# If the command fails, it typically means branch is not an ancestor of target_branch
			# This is normal and not an error condition
			is_ancestor = False
			logger.debug("Branch %s is not an ancestor of %s", branch_ref, target_ref)

		# Try the reverse check as well to determine relationship
		try:
			reverse_cmd = ["git", "merge-base", "--is-ancestor", target_ref, branch_ref]
			run_git_command(reverse_cmd)
			# If we get here, target is an ancestor of branch (target is older)
			if not is_ancestor:
				logger.debug("Branch %s is newer than %s", branch_ref, target_ref)
		except GitError:
			# If both checks fail, the branches have no common ancestor
			if not is_ancestor:
				logger.debug("Branches %s and %s have no common history", branch_ref, target_ref)

		# Get commit count between branches
		count_cmd = ["git", "rev-list", "--count", f"{branch_ref}..{target_ref}"]
		try:
			count = int(run_git_command(count_cmd).strip())
		except GitError:
			# If this fails, branches might be completely unrelated
			count = 0

		return (is_ancestor, count)
	except GitError as e:
		logger.warning("Error determining branch relation: %s", e)
		return (False, 0)


def get_branch_description(branch_name: str) -> str:
	"""
	Generate a description for a branch based on its commits.

	Args:
	    branch_name: Name of the branch

	Returns:
	    Description of the branch

	"""
	try:
		# Get base branch
		base_branch = get_default_branch()

		# Get unique commits on this branch
		commits = get_commit_messages(base_branch, branch_name)

		if not commits:
			return "No unique commits found on this branch."

		# Return first few commits as description
		if len(commits) <= MAX_COMMIT_PREVIEW:
			return "\n".join([f"- {commit}" for commit in commits])

		summary = "\n".join([f"- {commit}" for commit in commits[:MAX_COMMIT_PREVIEW]])
		return f"{summary}\n- ... and {len(commits) - MAX_COMMIT_PREVIEW} more commits"
	except GitError:
		return "Unable to get branch description."


def detect_branch_type(
	branch_name: str, strategy_name: Literal["github-flow", "gitflow", "trunk-based"] = "github-flow"
) -> str:
	"""
	Detect the type of a branch based on its name and workflow strategy.

	Args:
	    branch_name: Name of the branch
	    strategy_name: Name of the workflow strategy to use

	Returns:
	    Branch type or "feature" if not detected

	"""
	strategy = create_strategy(strategy_name)
	branch_type = strategy.detect_branch_type(branch_name)

	return branch_type or "feature"  # Default to feature if not detected


def get_merged_prs_since_last_release(base_branch: str, head_branch: str) -> list[dict[str, str]]:
	"""
	Get a list of PRs merged to head_branch since last merge to base_branch.

	Args:
	    base_branch: Base branch (e.g., main)
	    head_branch: Head branch (e.g., develop)

	Returns:
	    List of PR dictionaries with keys: number, title, url

	"""
	try:
		# Get merge-base between base and head branches
		cmd = ["git", "merge-base", base_branch, head_branch]
		merge_base = run_git_command(cmd).strip()

		# Get commits between merge-base and head
		log_cmd = ["git", "log", f"{merge_base}..{head_branch}", "--grep=Merge pull request", "--pretty=format:%s"]
		log_output = run_git_command(log_cmd)

		if not log_output.strip():
			return []

		# Extract PR info from merge commits
		prs = []
		for line in log_output.strip().split("\n"):
			# Parse merge commit messages in the format "Merge pull request #123 from..."
			match = re.search(r"Merge pull request #(\d+) from .+", line)
			if match:
				pr_number = match.group(1)

				# Get the PR title from GitHub
				try:
					pr_info_cmd = ["gh", "pr", "view", pr_number, "--json", "title,url"]
					pr_info = json.loads(run_git_command(pr_info_cmd))
					prs.append({"number": pr_number, "title": pr_info["title"], "url": pr_info["url"]})
				except (GitError, json.JSONDecodeError, KeyError):
					# If GitHub CLI fails, use a basic entry
					prs.append(
						{
							"number": pr_number,
							"title": line.replace(f"Merge pull request #{pr_number} from", "").strip(),
							"url": "",
						}
					)

		return prs
	except GitError:
		return []


def categorize_prs(prs: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
	"""
	Categorize PRs by type based on their titles.

	Args:
	    prs: List of PR dictionaries

	Returns:
	    Dictionary mapping categories to lists of PRs

	"""
	categories = {"features": [], "fixes": [], "docs": [], "refactoring": [], "other": []}

	for pr in prs:
		title = pr["title"].lower()
		if re.search(r"\bfeat", title):
			categories["features"].append(pr)
		elif re.search(r"\bfix", title):
			categories["fixes"].append(pr)
		elif re.search(r"\bdoc", title):
			categories["docs"].append(pr)
		elif re.search(r"\brefactor", title):
			categories["refactoring"].append(pr)
		else:
			categories["other"].append(pr)

	return categories


def generate_release_notes(categorized_prs: dict[str, list[dict[str, str]]], template: str = "") -> str:
	"""
	Generate release notes from categorized PRs.

	Args:
	    categorized_prs: Dictionary mapping categories to lists of PRs
	    template: Optional template string with placeholders

	Returns:
	    Generated release notes

	"""
	if not any(categorized_prs.values()):
		return "No changes found in this release."

	sections = []

	if categorized_prs["features"]:
		features = "\n".join([f"- {pr['title']} (#{pr['number']})" for pr in categorized_prs["features"]])
		sections.append(f"## 🚀 Features\n\n{features}")

	if categorized_prs["fixes"]:
		fixes = "\n".join([f"- {pr['title']} (#{pr['number']})" for pr in categorized_prs["fixes"]])
		sections.append(f"## 🐛 Bug Fixes\n\n{fixes}")

	if categorized_prs["docs"]:
		docs = "\n".join([f"- {pr['title']} (#{pr['number']})" for pr in categorized_prs["docs"]])
		sections.append(f"## 📚 Documentation\n\n{docs}")

	if categorized_prs["refactoring"]:
		refactor = "\n".join([f"- {pr['title']} (#{pr['number']})" for pr in categorized_prs["refactoring"]])
		sections.append(f"## 🔄 Refactoring\n\n{refactor}")

	if categorized_prs["other"]:
		other = "\n".join([f"- {pr['title']} (#{pr['number']})" for pr in categorized_prs["other"]])
		sections.append(f"## 🔧 Other Changes\n\n{other}")

	notes = "\n\n".join(sections)

	if template:
		return template.format(changes=notes, testing_instructions="", screenshots="")
	return notes


def generate_release_pr_content(base_branch: str, head_branch: str) -> dict[str, str]:
	"""
	Generate content for a release PR (e.g., develop -> main).

	Args:
	    base_branch: Base branch (e.g., main)
	    head_branch: Head branch (e.g., develop)

	Returns:
	    Dictionary with title and description

	"""
	# Get merged PRs since last release
	merged_prs = get_merged_prs_since_last_release(base_branch, head_branch)

	# Categorize PRs
	categorized = categorize_prs(merged_prs)

	# Generate version suggestion based on changes
	version = "0.1.0"  # Default version
	if categorized["features"]:
		# For a new feature, increment the minor version
		version = suggest_version_from_changes(categorized, base_branch)

	# Generate PR content
	return {"title": f"Release {version}", "description": generate_release_notes(categorized)}


def suggest_version_from_changes(categorized_prs: dict[str, list[dict[str, str]]], base_branch: str = "main") -> str:
	"""
	Suggest version number based on changes in the PR.

	Args:
	    categorized_prs: Dictionary mapping categories to lists of PRs
	    base_branch: Base branch to check for current version

	Returns:
	    Suggested version number

	"""
	try:
		# Try to find the latest version tag on the base branch
		cmd = ["git", "describe", "--tags", "--abbrev=0", base_branch]
		try:
			latest_tag = run_git_command(cmd).strip()
			# Parse version from tag (assuming semantic versioning)
			match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", latest_tag)
			if match:
				major, minor, patch = map(int, match.groups())
			else:
				major, minor, patch = 0, 1, 0
		except GitError:
			# No tags found, start with 0.1.0
			major, minor, patch = 0, 1, 0

		# Determine version bump based on changes
		all_prs = []
		for pr_list in categorized_prs.values():
			all_prs.extend(pr_list)

		if any("BREAKING CHANGE" in pr["title"] for pr in all_prs):
			# Breaking change: bump major version
			return f"{major + 1}.0.0"
		if categorized_prs["features"]:
			# New features: bump minor version
			return f"{major}.{minor + 1}.0"
		if categorized_prs["fixes"] or categorized_prs["refactoring"] or categorized_prs["other"]:
			# Bug fixes or other changes: bump patch version
			return f"{major}.{minor}.{patch + 1}"
		# No significant changes
		return f"{major}.{minor}.{patch}"
	except GitError:
		return "0.1.0"  # Default version if all else fails


def _get_current_time() -> str:
	"""Get current time in a formatted string."""
	return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
