"""Main PR generation command implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codemap.git.utils import GitError, get_repo_root, run_git_command
from codemap.llm import LLMError
from codemap.utils.cli_utils import loading_spinner

from . import PRGenerator
from .constants import MIN_COMMIT_PARTS

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class PRCommand:
	"""Handles the PR generation command workflow."""

	def __init__(self, path: Path | None = None, model: str = "gpt-4o-mini") -> None:
		"""
		Initialize the PR command.

		Args:
		    path: Optional path to start from
		    model: LLM model to use for PR description generation

		"""
		try:
			self.repo_root = get_repo_root(path)

			# Create LLM client and configs
			from codemap.llm import create_client

			llm_client = create_client(repo_path=self.repo_root, model=model)

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
			with loading_spinner("Generating PR description using LLM..."):
				# Use the PR generator to create content
				content = self.pr_generator.generate_content_from_commits(
					base_branch=branch_info["target_branch"], head_branch=branch_info["current_branch"], use_llm=True
				)
				return content["description"]
		except LLMError as e:
			logger.exception("LLM description generation failed")
			logger.warning("LLM error: %s", str(e))

			# Generate a simple fallback description without LLM
			with loading_spinner("Falling back to simple PR description generation..."):
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
			with loading_spinner("Getting branch information..."):
				branch_info = self._get_branch_info()

			# Get commit history
			with loading_spinner("Retrieving commit history..."):
				commits = self._get_commit_history(branch_info["target_branch"])

			if not commits:
				self._raise_no_commits_error(branch_info)

			# Generate PR description
			description = self._generate_pr_description(branch_info, commits)

			return {"branch_info": branch_info, "commits": commits, "description": description}
		except (RuntimeError, ValueError) as e:
			self.error_state = "failed"
			raise RuntimeError(str(e)) from e
