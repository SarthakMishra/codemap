"""
PR generator for the CodeMap Git module.

This class generates pull requests for git repositories.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codemap.git.pr_generator.pr_git_utils import PRGitUtils
from codemap.git.pr_generator.schemas import PRContent, PullRequest
from codemap.git.pr_generator.utils import (
	create_pull_request,
	generate_pr_content_from_template,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	get_default_branch,
	get_existing_pr,
	suggest_branch_name,
	update_pull_request,
)
from codemap.git.utils import GitError
from codemap.llm import LLMClient

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class PRGenerator:
	"""
	Generator for Pull Requests.

	This class handles generating pull request content (title and
	description) and creating/updating PRs on GitHub.

	"""

	def __init__(
		self,
		repo_path: Path,
		llm_client: LLMClient,
	) -> None:
		"""
		Initialize the PR generator.

		Args:
		    repo_path: Path to the git repository
		    llm_client: LLMClient instance to use for content generation

		"""
		self.repo_path = repo_path
		self.client = llm_client

	def generate_content_from_commits(self, base_branch: str, head_branch: str, use_llm: bool = True) -> PRContent:
		"""
		Generate PR content (title and description) from commits.

		Args:
		    base_branch: Base branch (e.g., main)
		    head_branch: Head branch (e.g., feature-branch)
		    use_llm: Whether to use LLM for generation

		Returns:
		    Dictionary with 'title' and 'description' keys

		"""
		# Get commit messages between branches
		pgu = PRGitUtils.get_instance()
		commits = pgu.get_commit_messages(base_branch, head_branch)

		if not commits:
			return {"title": "Update branch", "description": "No changes in this PR."}

		if use_llm:
			# Generate title and description using LLM
			title = generate_pr_title_with_llm(commits, self.client)
			description = generate_pr_description_with_llm(commits, self.client)
		else:
			# Generate title and description using rule-based approach
			title = generate_pr_title_from_commits(commits)
			description = generate_pr_description_from_commits(commits)

		return {"title": title, "description": description}

	def generate_content_from_template(
		self, branch_name: str, description: str, workflow_strategy: str = "github-flow"
	) -> PRContent:
		"""
		Generate PR content (title and description) from a template.

		Args:
		    branch_name: Name of the branch
		    description: Short description of the changes
		    workflow_strategy: Git workflow strategy to use

		Returns:
		    Dictionary with 'title' and 'description' keys

		"""
		return generate_pr_content_from_template(branch_name, description, workflow_strategy)

	def suggest_branch_name(self, description: str, workflow_strategy: str = "github-flow") -> str:
		"""
		Suggest a branch name based on a description.

		Args:
		    description: Description of the branch
		    workflow_strategy: Git workflow strategy to use

		Returns:
		    Suggested branch name

		"""
		return suggest_branch_name(description, workflow_strategy)

	def create_pr(self, base_branch: str, head_branch: str, title: str, description: str) -> PullRequest:
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
		return create_pull_request(base_branch, head_branch, title, description)

	def update_pr(self, pr_number: int, title: str, description: str) -> PullRequest:
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
		return update_pull_request(pr_number, title, description)

	def get_existing_pr(self, branch_name: str) -> PullRequest | None:
		"""
		Get an existing PR for a branch.

		Args:
		    branch_name: Branch name

		Returns:
		    PullRequest object if found, None otherwise

		"""
		return get_existing_pr(branch_name)

	def create_or_update_pr(
		self,
		base_branch: str | None = None,
		head_branch: str | None = None,
		title: str | None = None,
		description: str | None = None,
		use_llm: bool = True,
		pr_number: int | None = None,
	) -> PullRequest:
		"""
		Create a new PR or update an existing one.

		Args:
		    base_branch: Base branch (defaults to default branch)
		    head_branch: Head branch
		    title: PR title (if None, will be generated)
		    description: PR description (if None, will be generated)
		    use_llm: Whether to use LLM for content generation
		    pr_number: PR number for update (if None, will create new PR)

		Returns:
		    PullRequest object

		Raises:
		    GitError: If PR creation/update fails

		"""
		# Get default branch if base_branch is not specified
		if base_branch is None:
			base_branch = get_default_branch()

		# Set default head_branch to current branch if not specified
		pgu = PRGitUtils.get_instance()
		if head_branch is None:
			try:
				head_branch = pgu.get_current_branch()
			except GitError as err:
				msg = "Failed to determine current branch"
				raise GitError(msg) from err

		# Check if PR exists
		existing_pr = None
		if pr_number is not None:
			# Updating an existing PR by number
			if title is None or description is None:
				# Need to fetch the PR to get current title/description
				existing_pr = self.get_existing_pr(head_branch)
				if existing_pr is None:
					msg = f"No PR found for branch {head_branch} with number {pr_number}"
					raise GitError(msg)

		else:
			# Look for existing PR for this branch
			existing_pr = self.get_existing_pr(head_branch)
			if existing_pr is not None:
				pr_number = existing_pr.number

		# Generate content if not provided
		if title is None or description is None:
			content = self.generate_content_from_commits(base_branch, head_branch, use_llm)
			if title is None:
				title = content["title"]
			if description is None:
				description = content["description"]

		# Create or update PR
		if pr_number is not None:
			# Update existing PR
			return self.update_pr(pr_number, title, description)
		# Create new PR
		return self.create_pr(base_branch, head_branch, title, description)
