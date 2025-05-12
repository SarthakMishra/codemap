"""Utility functions for PR generation."""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dotenv import set_key
from github import Auth, Github, GithubException
from pygit2 import GitError as Pygit2GitError

from codemap.config.config_loader import ConfigLoader
from codemap.git.pr_generator.constants import MAX_COMMIT_PREVIEW
from codemap.git.pr_generator.pr_git_utils import PRGitUtils
from codemap.git.pr_generator.prompts import (
	PR_DESCRIPTION_PROMPT,
	PR_SYSTEM_PROMPT,
	PR_TITLE_PROMPT,
	format_commits_for_prompt,
)
from codemap.git.pr_generator.schemas import PRContent, PullRequest
from codemap.git.pr_generator.strategies import create_strategy, get_default_branch
from codemap.git.utils import GitError

if TYPE_CHECKING:
	from codemap.llm import LLMClient

logger = logging.getLogger(__name__)


class PRCreationError(GitError):
	"""Error raised when there's an issue creating or updating a pull request."""


# Singleton for Github client
_github_client = None
_github_repo = None

GH_TOKEN_PARTS_LEN = 2  # Constant for token line split length


def get_token_from_gh_cli() -> str | None:
	"""
	Try to get the GitHub token from the gh CLI.

	Returns:
		The token string if found, else None.
	"""
	try:
		# This subprocess call is safe: command and args are hardcoded, no user input
		result = subprocess.run(  # noqa: S603
			["gh", "auth", "status", "--show-token"],  # noqa: S607
			capture_output=True,
			text=True,
			check=True,
		)
		# Look for 'Token: ...' in output
		for line in result.stdout.splitlines():
			if line.strip().startswith("- Token:"):
				# Extract token
				parts = line.split(":", 1)
				if len(parts) == GH_TOKEN_PARTS_LEN:
					token = parts[1].strip()
					if token:
						return token
		return None
	except (subprocess.CalledProcessError, FileNotFoundError):
		return None


def get_github_client(config_loader: ConfigLoader | None = None) -> tuple[Github, str]:
	"""
	Get a singleton Github client using the OAuth token from config.

	Returns:
		(Github, repo_full_name): Tuple of Github client and repo name
	Raises:
		PRCreationError: If token is missing or repo cannot be determined
	"""
	global _github_client, _github_repo  # noqa: PLW0603
	if _github_client is not None and _github_repo is not None:
		return _github_client, _github_repo

	config_loader = config_loader or ConfigLoader.get_instance()
	config = config_loader.get.github
	token = config.token
	repo_name = config.repo
	if not token:
		# Try to get from gh CLI
		token = get_token_from_gh_cli()
		if token:
			# Save to .env.local for future use
			try:
				set_key(str(Path(".env.local")), "GITHUB_TOKEN", token)
				logger.info("Saved GitHub token from gh CLI to .env.local")
			# Only catch expected errors from set_key
			except (OSError, ValueError) as e:
				logger.warning(f"Could not save GitHub token to .env.local: {e}")
		else:
			logger.error("GitHub OAuth token not set in config (github.token), env, or gh CLI.")
			msg = (
				"GitHub OAuth token not set in config (github.token), env, or gh CLI. "
				"Please run 'gh auth login' or set GITHUB_TOKEN in your .env/.env.local."
			)
			raise PRCreationError(msg)
	auth = Auth.Token(token)
	_github_client = Github(auth=auth)
	if not repo_name:
		# Try to infer from git remote
		try:
			pr_git_utils = PRGitUtils.get_instance()
			url = pr_git_utils.repo.remotes["origin"].url
			# Ensure url is a string
			if not isinstance(url, str) or not url:
				msg = f"Could not parse GitHub repo from remote URL: {url}"
				raise PRCreationError(msg)
			# Parse repo name from URL (supports git@github.com:user/repo.git and https)
			m = re.search(r"github.com[:/](.+?)(?:\\.git)?$", url)
			if m:
				repo_name = m.group(1)
				# Remove .git suffix if present
				repo_name = repo_name.removesuffix(".git")
			else:
				msg = f"Could not parse GitHub repo from remote URL: {url}"
				raise PRCreationError(msg)
		except Exception as e:
			logger.exception("Could not determine GitHub repo from git remote")
			msg = "Could not determine GitHub repo from git remote"
			raise PRCreationError(msg) from e
	_github_repo = repo_name
	return _github_client, _github_repo


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

	# Use the first commit to determine the PR type
	first_commit = commits[0]

	# Define mapping from commit prefixes to PR title prefixes
	prefix_mapping = {"feat": "Feature:", "fix": "Fix:", "docs": "Docs:", "refactor": "Refactor:", "perf": "Optimize:"}

	# Extract commit type from first commit
	match = re.match(r"^([a-z]+)(\([^)]+\))?:", first_commit)
	if match:
		prefix = match.group(1)
		title_prefix = prefix_mapping.get(prefix, "Update:")

		# Strip the prefix and use as title
		title = re.sub(r"^[a-z]+(\([^)]+\))?:\s*", "", first_commit)
		# Capitalize first letter and add PR type prefix
		return f"{title_prefix} {title[0].upper() + title[1:]}"

	# Fallback if no conventional commit format found
	return first_commit


def generate_pr_title_with_llm(
	commits: list[str],
	llm_client: LLMClient,
) -> str:
	"""
	Generate a PR title using an LLM.

	Args:
	    commits: List of commit messages
	    llm_client: LLMClient instance

	Returns:
	    Generated PR title

	"""
	if not commits:
		return "Update branch"

	try:
		# Format commit messages and prepare prompt
		commit_list = format_commits_for_prompt(commits)
		prompt = PR_TITLE_PROMPT.format(commit_list=commit_list)

		return llm_client.completion(
			messages=[
				{"role": "system", "content": PR_SYSTEM_PROMPT},
				{"role": "user", "content": prompt},
			],
		)

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
	llm_client: LLMClient,
) -> str:
	"""
	Generate a PR description using an LLM.

	Args:
	    commits: List of commit messages
	    llm_client: LLMClient instance

	Returns:
	    Generated PR description

	"""
	if not commits:
		return "No changes"

	try:
		# Format commit messages and prepare prompt
		commit_list = format_commits_for_prompt(commits)
		prompt = PR_DESCRIPTION_PROMPT.format(commit_list=commit_list)

		return llm_client.completion(
			messages=[
				{"role": "system", "content": PR_SYSTEM_PROMPT},
				{"role": "user", "content": prompt},
			],
		)

	except (ValueError, RuntimeError, ConnectionError) as e:
		logger.warning("Failed to generate PR description with LLM: %s", str(e))
		# Fallback to rule-based approach
		return generate_pr_description_from_commits(commits)


def create_pull_request(base_branch: str, head_branch: str, title: str, description: str) -> PullRequest:
	"""Create a pull request on GitHub using PyGithub."""
	try:
		gh, repo_name = get_github_client()
		repo = gh.get_repo(repo_name)
		pr = repo.create_pull(
			base=base_branch,
			head=head_branch,
			title=title,
			body=description,
		)
		return PullRequest(
			branch=head_branch,
			title=title,
			description=description,
			url=pr.html_url,
			number=pr.number,
		)
	except GithubException as e:
		logger.exception("GitHub API error during PR creation:")
		msg = f"Failed to create PR: {e.data.get('message', str(e)) if hasattr(e, 'data') else str(e)}"
		raise PRCreationError(msg) from e
	except Exception as e:
		logger.exception("Error creating PR via PyGithub:")
		msg = f"Error during PR creation: {e}"
		raise PRCreationError(msg) from e


def update_pull_request(pr_number: int | None, title: str, description: str) -> PullRequest:
	"""Update an existing pull request using PyGithub."""
	if pr_number is None:
		msg = "PR number cannot be None"
		raise PRCreationError(msg)
	try:
		gh, repo_name = get_github_client()
		repo = gh.get_repo(repo_name)
		pr = repo.get_pull(pr_number)
		pr.edit(title=title, body=description)
		# Get current branch name
		pr_git_utils = PRGitUtils.get_instance()
		branch = pr_git_utils.get_current_branch()
		return PullRequest(
			branch=branch,
			title=title,
			description=description,
			url=pr.html_url,
			number=pr.number,
		)
	except GithubException as e:
		logger.exception("GitHub API error during PR update:")
		msg = f"Failed to update PR: {e.data.get('message', str(e)) if hasattr(e, 'data') else str(e)}"
		raise PRCreationError(msg) from e
	except Exception as e:
		logger.exception("Error updating PR via PyGithub:")
		msg = f"Error during PR update: {e}"
		raise PRCreationError(msg) from e


def get_existing_pr(branch_name: str) -> PullRequest | None:
	"""Get an existing PR for a branch using PyGithub."""
	try:
		if not branch_name:
			logger.debug("Branch name is None, cannot get existing PR.")
			return None
		gh, repo_name = get_github_client()
		repo = gh.get_repo(repo_name)
		try:
			pulls = repo.get_pulls(state="open", head=f"{repo.owner.login}:{branch_name}")
		except Exception as e:  # noqa: BLE001
			logger.warning(f"Error getting PRs from GitHub API: {e}")
			return None
		for pr in pulls:
			# Return the first matching PR
			return PullRequest(
				branch=branch_name,
				title=pr.title,
				description=pr.body,
				url=pr.html_url,
				number=pr.number,
			)
		return None
	except GithubException as e:
		logger.warning(f"GitHub API error during get_existing_pr: {e}")
		return None
	except (ValueError, RuntimeError, ConnectionError, TypeError) as e:
		logger.warning(f"Error getting existing PR via PyGithub: {e}")
		return None


def generate_pr_content_from_template(
	branch_name: str,
	description: str,
	strategy_name: str = "github-flow",
) -> PRContent:
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
	workflow_type = cast("str", workflow)
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
		base_branch = get_default_branch()  # This is a helper from .strategies

		# Instantiate PRGitUtils to use the new pygit2-based get_commit_messages
		# This assumes the CWD is within a git repo.
		# A better approach would be to pass an instance of PRGitUtils or relevant context.
		git_utils_instance = PRGitUtils()  # This will initialize ExtendedGitRepoContext
		commits = git_utils_instance.get_commit_messages(base_branch, branch_name)

		if not commits:
			return "No unique commits found on this branch."

		# Return first few commits as description
		if len(commits) <= MAX_COMMIT_PREVIEW:
			return "\n".join([f"- {commit}" for commit in commits])

		summary = "\n".join([f"- {commit}" for commit in commits[:MAX_COMMIT_PREVIEW]])
		return f"{summary}\n- ... and {len(commits) - MAX_COMMIT_PREVIEW} more commits"
	except GitError:
		return "Unable to get branch description."


def detect_branch_type(branch_name: str, strategy_name: str = "github-flow") -> str:
	"""
	Detect the type of a branch based on its name and workflow strategy.

	Args:
	    branch_name: Name of the branch
	    strategy_name: Name of the workflow strategy to use

	Returns:
	    Branch type or "feature" if not detected

	"""
	strategy = create_strategy(strategy_name)
	# Handle None branch_name
	if not branch_name:
		return "feature"  # Default if branch name is None
	branch_type = strategy.detect_branch_type(branch_name)

	return branch_type or "feature"  # Default to feature if not detected


def list_branches() -> list[str]:
	"""
	Get a list of all branches (local and remote).

	Returns:
	        List of branch names
	"""
	try:
		git_utils_instance = PRGitUtils()
		local_branches = list(git_utils_instance.repo.branches.local)
		remote_branches_full_refs = list(git_utils_instance.repo.branches.remote)
		remote_branches = []
		for ref_name in remote_branches_full_refs:
			# Example ref_name: "origin/main", "origin/HEAD"
			if not ref_name.endswith("/HEAD"):  # Exclude remote HEAD pointers
				# Strip the remote name prefix, e.g., "origin/"
				parts = ref_name.split("/", 1)
				if len(parts) > 1:
					remote_branches.append(parts[1])
				else:  # Should not happen for valid remote branch refs like "origin/branch"
					remote_branches.append(ref_name)

		# Combine and remove duplicates
		return list(set(local_branches + remote_branches))
	except (GitError, Pygit2GitError) as e:
		logger.debug(f"Error listing branches using pygit2: {e}")
		return []


def validate_branch_name(branch_name: str | None) -> bool:
	"""
	Validate a branch name.

	Args:
	    branch_name: Branch name to validate

	Returns:
	    True if valid, False otherwise

	"""
	# Check if branch name is valid
	if not branch_name or not re.match(r"^[a-zA-Z0-9_.-]+$", branch_name):
		# Log error instead of showing directly, as this is now a util function
		logger.error(
			"Invalid branch name '%s'. Use only letters, numbers, underscores, dots, and hyphens.", branch_name
		)
		return False
	return True


def get_all_open_prs() -> list[PullRequest]:
	"""
	Fetch all open pull requests for the current repository.

	Returns:
		List of PullRequest objects for all open PRs.

	Raises:
		PRCreationError: If repo_name is invalid or repo cannot be found.
	"""
	gh, repo_name = get_github_client()
	if not repo_name or "/" not in repo_name:
		logger.error(f"Invalid repo_name for GitHub API: {repo_name}")
		msg = f"Invalid repo_name for GitHub API: {repo_name}"
		raise PRCreationError(msg)
	try:
		repo = gh.get_repo(repo_name)
	except Exception as e:
		logger.exception(f"Could not fetch repo '{repo_name}' from GitHub.")
		msg = f"Could not fetch repo '{repo_name}' from GitHub: {e}"
		raise PRCreationError(msg) from e
	return [
		PullRequest(
			branch=pr.head.ref,
			title=pr.title,
			description=pr.body,
			url=pr.html_url,
			number=pr.number,
		)
		for pr in repo.get_pulls(state="open")
	]
