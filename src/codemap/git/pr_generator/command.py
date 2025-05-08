"""Main PR generation command implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codemap.config import ConfigLoader
from codemap.git.pr_generator.schemas import PullRequest
from codemap.git.pr_generator.strategies import create_strategy
from codemap.git.pr_generator.utils import (
	PRCreationError,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	get_commit_messages,
	get_existing_pr,
)
from codemap.git.utils import GitError, get_repo_root, run_git_command
from codemap.llm import LLMClient, LLMError
from codemap.utils.cli_utils import progress_indicator

from . import PRGenerator
from .constants import MIN_COMMIT_PARTS

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class PRCommand:
	"""Handles the PR generation command workflow."""

	def __init__(self, config_loader: ConfigLoader, path: Path | None = None) -> None:
		"""
		Initialize the PR command.

		Args:
		    config_loader: ConfigLoader instance
		    path: Optional path to start from

		"""
		try:
			self.repo_root = get_repo_root(path)

			# Create LLM client and configs
			from codemap.llm import LLMClient

			llm_client = LLMClient(config_loader=config_loader, repo_path=self.repo_root)

			# Create the PR generator with required parameters
			self.pr_generator = PRGenerator(
				repo_path=self.repo_root,
				llm_client=llm_client,
			)

			self.error_state = None  # Tracks reason for failure: "failed", "aborted", etc.
		except GitError as e:
			raise RuntimeError(str(e)) from e

	def _get_branch_info(self) -> dict[str, str]:
		"""
		Get information about the current branch and its target.

		Returns:
		    Dictionary with branch information

		Raises:
		    RuntimeError: If Git operations fail

		"""
		try:
			# Get current branch
			current_branch = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()

			# Get default branch (usually main or master)
			default_branch = run_git_command(["git", "remote", "show", "origin"]).strip()
			# Parse the default branch from the output
			for line in default_branch.splitlines():
				if "HEAD branch" in line:
					default_branch = line.split(":")[-1].strip()
					break

			return {"current_branch": current_branch, "target_branch": default_branch}
		except GitError as e:
			msg = f"Failed to get branch information: {e}"
			raise RuntimeError(msg) from e

	def _get_commit_history(self, base_branch: str) -> list[dict[str, str]]:
		"""
		Get commit history between the current branch and the base branch.

		Args:
		    base_branch: The base branch to compare against

		Returns:
		    List of commits with their details

		Raises:
		    RuntimeError: If Git operations fail

		"""
		try:
			# Get list of commits that are in the current branch but not in the base branch
			commits_output = run_git_command(["git", "log", f"{base_branch}..HEAD", "--pretty=format:%H||%an||%s"])

			commits = []
			if commits_output.strip():
				for commit_line in commits_output.strip().split("\n"):
					if not commit_line.strip():
						continue

					parts = commit_line.split("||")
					if len(parts) >= MIN_COMMIT_PARTS:
						commit_hash, author, subject = parts[0], parts[1], parts[2]
						commits.append({"hash": commit_hash, "author": author, "subject": subject})

			return commits
		except GitError as e:
			msg = f"Failed to get commit history: {e}"
			raise RuntimeError(msg) from e

	def _generate_pr_description(self, branch_info: dict[str, str], _commits: list[dict[str, str]]) -> str:
		"""
		Generate PR description based on branch info and commit history.

		Args:
		    branch_info: Information about the branches
		    _commits: List of commits to include in the description (fetched internally by PRGenerator)

		Returns:
		    Generated PR description

		Raises:
		    RuntimeError: If description generation fails

		"""
		try:
			with progress_indicator("Generating PR description using LLM..."):
				# Use the PR generator to create content
				content = self.pr_generator.generate_content_from_commits(
					base_branch=branch_info["target_branch"], head_branch=branch_info["current_branch"], use_llm=True
				)
				return content["description"]
		except LLMError as e:
			logger.exception("LLM description generation failed")
			logger.warning("LLM error: %s", str(e))

			# Generate a simple fallback description without LLM
			with progress_indicator("Falling back to simple PR description generation..."):
				content = self.pr_generator.generate_content_from_commits(
					base_branch=branch_info["target_branch"], head_branch=branch_info["current_branch"], use_llm=False
				)
				return content["description"]
		except (ValueError, RuntimeError) as e:
			logger.warning("Error generating PR description: %s", str(e))
			msg = f"Failed to generate PR description: {e}"
			raise RuntimeError(msg) from e

	def _raise_no_commits_error(self, branch_info: dict[str, str]) -> None:
		"""
		Raise an error when no commits are found between branches.

		Args:
		    branch_info: Information about the branches

		Raises:
		    RuntimeError: Always raises this error with appropriate message

		"""
		msg = f"No commits found between {branch_info['current_branch']} and {branch_info['target_branch']}"
		logger.warning(msg)
		raise RuntimeError(msg)

	def run(self) -> dict[str, Any]:
		"""
		Run the PR generation command.

		Returns:
		    Dictionary with PR information and generated description

		Raises:
		    RuntimeError: If the command fails

		"""
		try:
			# Get branch information
			with progress_indicator("Getting branch information..."):
				branch_info = self._get_branch_info()

			# Get commit history
			with progress_indicator("Retrieving commit history..."):
				commits = self._get_commit_history(branch_info["target_branch"])

			if not commits:
				self._raise_no_commits_error(branch_info)

			# Generate PR description
			description = self._generate_pr_description(branch_info, commits)

			return {"branch_info": branch_info, "commits": commits, "description": description}
		except (RuntimeError, ValueError) as e:
			self.error_state = "failed"
			raise RuntimeError(str(e)) from e


class PRWorkflowCommand:
	"""Handles the core PR creation and update workflow logic."""

	def __init__(
		self,
		config_loader: ConfigLoader,
		llm_client: LLMClient | None = None,
	) -> None:
		"""
		Initialize the PR workflow command helper.

		Args:
		        config_loader: ConfigLoader instance.
		        llm_client: Optional pre-configured LLMClient.

		"""
		self.config_loader = config_loader

		if self.config_loader.get.repo_root is None:
			self.repo_root = get_repo_root()
		else:
			self.repo_root = self.config_loader.get.repo_root

		self.pr_config = self.config_loader.get.pr
		self.content_config = self.pr_config.generate
		self.workflow_strategy_name = self.config_loader.get.pr.strategy
		self.workflow = create_strategy(self.workflow_strategy_name)

		# Initialize LLM client if needed
		if llm_client:
			self.llm_client = llm_client
		else:
			from codemap.llm import LLMClient

			self.llm_client = LLMClient(
				config_loader=self.config_loader,
				repo_path=self.repo_root,
			)

		self.pr_generator = PRGenerator(repo_path=self.repo_root, llm_client=self.llm_client)

	def _generate_release_pr_content(self, base_branch: str, branch_name: str) -> dict[str, str]:
		"""
		Generate PR content for a release.

		Args:
		        base_branch: The branch to merge into (e.g. main)
		        branch_name: The release branch name (e.g. release/1.0.0)

		Returns:
		        Dictionary with title and description

		"""
		# Extract version from branch name
		version = branch_name.replace("release/", "")
		title = f"Release {version}"
		# Include base branch information in the description
		description = f"# Release {version}\n\nThis pull request merges release {version} into {base_branch}."
		return {"title": title, "description": description}

	def _generate_title(self, commits: list[str], branch_name: str, branch_type: str) -> str:
		"""Core logic for generating PR title."""
		title_strategy = self.content_config.title_strategy

		if not commits:
			if branch_type == "release":
				return f"Release {branch_name.replace('release/', '')}"
			clean_name = branch_name.replace(f"{branch_type}/", "").replace("-", " ").replace("_", " ")
			return f"{branch_type.capitalize()}: {clean_name.capitalize()}"

		if title_strategy == "llm":
			return generate_pr_title_with_llm(commits, llm_client=self.llm_client)

		return generate_pr_title_from_commits(commits)

	def _generate_description(self, commits: list[str], branch_name: str, branch_type: str, base_branch: str) -> str:
		"""Core logic for generating PR description."""
		description_strategy = self.content_config.description_strategy

		if not commits:
			if branch_type == "release" and self.workflow_strategy_name == "gitflow":
				# Call the internal helper method
				content = self._generate_release_pr_content(base_branch, branch_name)
				return content["description"]
			return f"Changes in {branch_name}"

		if description_strategy == "llm":
			return generate_pr_description_with_llm(commits, llm_client=self.llm_client)

		if description_strategy == "template" and self.content_config.use_workflow_templates:
			template = self.content_config.description_template
			if template:
				commit_description = "\n".join([f"- {commit}" for commit in commits])
				# Note: Other template variables like testing_instructions might need context
				return template.format(
					changes=commit_description,
					testing_instructions="[Testing instructions]",
					screenshots="[Screenshots]",
				)

		return generate_pr_description_from_commits(commits)

	def create_pr_workflow(
		self, base_branch: str, head_branch: str, title: str | None = None, description: str | None = None
	) -> PullRequest:
		"""Orchestrates the PR creation process (non-interactive part)."""
		try:
			# Check for existing PR first
			existing_pr = get_existing_pr(head_branch)
			if existing_pr:
				logger.warning(
					f"PR #{existing_pr.number} already exists for branch '{head_branch}'. Returning existing PR."
				)
				return existing_pr

			# Get commits
			commits = get_commit_messages(base_branch, head_branch)

			# Determine branch type
			branch_type = self.workflow.detect_branch_type(head_branch) or "feature"

			# Generate title and description if not provided
			final_title = title or self._generate_title(commits, head_branch, branch_type)
			final_description = description or self._generate_description(
				commits, head_branch, branch_type, base_branch
			)

			# Create PR using PRGenerator
			pr = self.pr_generator.create_pr(base_branch, head_branch, final_title, final_description)
			logger.info(f"Successfully created PR #{pr.number}: {pr.url}")
			return pr
		except GitError:
			# Specific handling for unrelated histories might go here or be handled in CLI
			logger.exception("GitError during PR creation workflow")
			raise
		except Exception as e:
			logger.exception("Unexpected error during PR creation workflow")
			msg = f"Unexpected error creating PR: {e}"
			raise PRCreationError(msg) from e

	def update_pr_workflow(
		self,
		pr_number: int,
		title: str | None = None,
		description: str | None = None,
		base_branch: str | None = None,
		head_branch: str | None = None,
	) -> PullRequest:
		"""Orchestrates the PR update process (non-interactive part)."""
		try:
			# Fetch existing PR info if needed to regenerate title/description
			# This might require gh cli or GitHub API interaction if pr_generator doesn't fetch
			# For now, assume base/head are provided if regeneration is needed

			final_title = title
			final_description = description

			# Regenerate if title/description are None
			if title is None or description is None:
				if not base_branch or not head_branch:
					msg = "Cannot regenerate content for update without base and head branches."
					raise PRCreationError(msg)

				commits = get_commit_messages(base_branch, head_branch)
				branch_type = self.workflow.detect_branch_type(head_branch) or "feature"

				if title is None:
					final_title = self._generate_title(commits, head_branch, branch_type)
				if description is None:
					final_description = self._generate_description(commits, head_branch, branch_type, base_branch)

			if final_title is None or final_description is None:
				msg = "Could not determine final title or description for PR update."
				raise PRCreationError(msg)

			# Update PR using PRGenerator
			updated_pr = self.pr_generator.update_pr(pr_number, final_title, final_description)
			logger.info(f"Successfully updated PR #{updated_pr.number}: {updated_pr.url}")
			return updated_pr
		except GitError:
			logger.exception("GitError during PR update workflow")
			raise
		except Exception as e:
			logger.exception("Unexpected error during PR update workflow")
			msg = f"Unexpected error updating PR: {e}"
			raise PRCreationError(msg) from e
