"""Main PR generation command implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pygit2 import Commit
from pygit2 import GitError as Pygit2GitError
from pygit2.enums import SortMode

from codemap.config import ConfigLoader
from codemap.git.pr_generator.pr_git_utils import PRGitUtils
from codemap.git.pr_generator.schemas import PullRequest
from codemap.git.pr_generator.strategies import create_strategy, get_default_branch
from codemap.git.pr_generator.utils import (
	PRCreationError,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	get_existing_pr,
)
from codemap.git.utils import ExtendedGitRepoContext, GitError
from codemap.llm import LLMClient, LLMError
from codemap.utils.cli_utils import progress_indicator

from . import PRGenerator

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
			self.repo_root = ExtendedGitRepoContext.get_repo_root(path)

			# Create LLM client and configs
			from codemap.llm import LLMClient

			llm_client = LLMClient(config_loader=config_loader, repo_path=self.repo_root)

			# Create the PR generator with required parameters
			self.pr_generator = PRGenerator(
				repo_path=self.repo_root,
				llm_client=llm_client,
			)

			self.error_state: str | None = None  # Tracks reason for failure: "failed", "aborted", etc.
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
			pgu = PRGitUtils.get_instance()
			repo = pgu.repo

			# Get current branch
			current_branch = pgu.get_current_branch()
			if not current_branch:
				msg = "Failed to determine current branch using PRGitUtils."
				raise GitError(msg)

			# Get default branch (usually main or master)
			# get_default_branch from strategies.py uses PRGitUtils instance
			default_branch_name = get_default_branch(pgu_instance=pgu)
			if not default_branch_name:
				msg = "Failed to determine default branch using PRGitUtils and strategies."
				# Attempt to find common names if strategy failed, as a fallback.
				common_defaults = ["main", "master"]
				for common_default in common_defaults:
					if f"origin/{common_default}" in repo.branches.remote or common_default in repo.branches.local:
						default_branch_name = common_default
						break
				if not default_branch_name:  # Still not found
					msg = "Could not determine default/target branch."
					raise GitError(msg)

			return {"current_branch": current_branch, "target_branch": default_branch_name}
		except (GitError, Pygit2GitError) as e:
			msg = f"Failed to get branch information: {e}"
			raise RuntimeError(msg) from e
		except Exception as e:
			msg = f"Unexpected error getting branch information: {e}"
			logger.exception("Unexpected error in _get_branch_info")
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
		pgu = PRGitUtils.get_instance()
		repo = pgu.repo
		commits_data = []
		try:
			head_commit_obj = repo.revparse_single("HEAD").peel(Commit)
			base_commit_obj = repo.revparse_single(base_branch).peel(Commit)

			# Find the merge base between head and base
			merge_base_oid = repo.merge_base(base_commit_obj.id, head_commit_obj.id)

			# Walk from HEAD, newest first
			for commit in repo.walk(head_commit_obj.id, SortMode.TOPOLOGICAL):
				if merge_base_oid and commit.id == merge_base_oid:
					break  # Stop if we reached the merge base

				# Additional check: if the commit is an ancestor of the base_commit_obj
				# and it's not the merge_base itself, it means we're on the base branch's
				# history before the divergence. This can happen in complex histories
				# if merge_base_oid is None or the walk somehow includes them.
				# The primary stop condition is hitting the merge_base_oid.
				if (
					repo.descendant_of(commit.id, base_commit_obj.id)
					and (not merge_base_oid or commit.id != merge_base_oid)
					and commit.id != head_commit_obj.id
				):  # Don't stop if base is HEAD or commit is HEAD
					# This commit is reachable from base_commit_obj, so it's part of base's history
					# This logic ensures we only take commits unique to the current branch after divergence
					# For `base..HEAD` this means commit is reachable from HEAD but not base.
					# The simple walk from head_commit_obj stopping at merge_base_oid should correctly
					# implement "commits on head since it diverged from base".
					pass  # The break at merge_base_oid is the key.

				commit_subject = commit.message.splitlines()[0].strip() if commit.message else ""
				commits_data.append(
					{
						"hash": str(commit.short_id),
						"author": commit.author.name if commit.author else "Unknown",
						"subject": commit_subject,
					}
				)
			return commits_data
		except Pygit2GitError as e:
			msg = f"Failed to get commit history using pygit2: {e}"
			logger.exception("pygit2 error in _get_commit_history")
			raise RuntimeError(msg) from e
		except Exception as e:
			# Catch other potential errors like branch not found from revparse_single
			msg = f"Unexpected error getting commit history: {e}"
			logger.exception("Unexpected error in _get_commit_history")
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
			self.repo_root = ExtendedGitRepoContext.get_repo_root()
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

			pgu = PRGitUtils.get_instance()
			# Get commits
			commits = pgu.get_commit_messages(base_branch, head_branch)

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

				pgu = PRGitUtils.get_instance()
				commits = pgu.get_commit_messages(base_branch, head_branch)
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
