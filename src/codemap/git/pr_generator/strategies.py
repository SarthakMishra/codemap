"""Git workflow strategy implementations for PR management."""

from __future__ import annotations

import re
from typing import Any

from codemap.git.pr_generator.constants import MIN_SIGNIFICANT_WORD_LENGTH
from codemap.git.pr_generator.templates import (
	DEFAULT_PR_TEMPLATE,
	GITFLOW_PR_TEMPLATES,
	GITHUB_FLOW_PR_TEMPLATE,
	TRUNK_BASED_PR_TEMPLATE,
)
from codemap.git.utils import GitError, run_git_command


class WorkflowStrategy:
	"""Base class for git workflow strategies."""

	def get_default_base(self, branch_type: str) -> str:
		"""
		Get the default base branch for a given branch type.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)

		Returns:
		    Name of the default base branch

		"""
		raise NotImplementedError

	def suggest_branch_name(self, branch_type: str, description: str) -> str:
		"""
		Suggest a branch name based on the workflow.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)
		    description: Description of the branch

		Returns:
		    Suggested branch name

		"""
		# Default implementation
		clean_description = re.sub(r"[^a-zA-Z0-9]+", "-", description.lower())
		clean_description = clean_description.strip("-")
		prefix = self.get_branch_prefix(branch_type)
		return f"{prefix}{clean_description}"

	def get_branch_prefix(self, branch_type: str) -> str:
		"""
		Get the branch name prefix for a given branch type.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)

		Returns:
		    Branch name prefix

		"""
		raise NotImplementedError

	def get_branch_types(self) -> list[str]:
		"""
		Get valid branch types for this workflow.

		Returns:
		    List of valid branch types

		"""
		raise NotImplementedError

	def detect_branch_type(self, branch_name: str) -> str | None:
		"""
		Detect the type of a branch from its name.

		Args:
		    branch_name: Name of the branch

		Returns:
		    Branch type or None if not detected

		"""
		for branch_type in self.get_branch_types():
			prefix = self.get_branch_prefix(branch_type)
			if branch_name.startswith(prefix):
				return branch_type
		return None

	def get_pr_templates(self, branch_type: str) -> dict[str, str]:  # noqa: ARG002
		"""
		Get PR title and description templates for a given branch type.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)

		Returns:
		    Dictionary with 'title' and 'description' templates

		"""
		# Return the default templates
		return DEFAULT_PR_TEMPLATE

	def get_remote_branches(self) -> list[str]:
		"""
		Get list of remote branches.

		Returns:
		    List of remote branch names (without 'origin/' prefix)

		"""
		try:
			branches = run_git_command(["git", "branch", "-r"]).strip().split("\n")
			# Clean up branch names and remove 'origin/' prefix
			remote_branches = []
			for branch_name in branches:
				branch_clean = branch_name.strip()
				if branch_clean.startswith("origin/"):
					branch_name_without_prefix = branch_clean[7:]  # Remove 'origin/' prefix
					# Exclude HEAD branches
					if not branch_name_without_prefix.startswith("HEAD"):
						remote_branches.append(branch_name_without_prefix)
			return remote_branches
		except GitError:
			return []

	def get_local_branches(self) -> list[str]:
		"""
		Get list of local branches.

		Returns:
		    List of local branch names

		"""
		try:
			branches = run_git_command(["git", "branch"]).strip().split("\n")
			# Clean up branch names and remove the '*' from current branch
			local_branches = []
			for branch_name in branches:
				branch_clean = branch_name.strip().removeprefix("* ")  # Remove '* ' prefix
				local_branches.append(branch_clean)
			return local_branches
		except GitError:
			return []

	def get_branches_by_type(self) -> dict[str, list[str]]:
		"""
		Group branches by their type.

		Returns:
		    Dictionary mapping branch types to lists of branch names

		"""
		result = {branch_type: [] for branch_type in self.get_branch_types()}
		result["other"] = []  # For branches that don't match any type

		# Get all branches (local and remote)
		all_branches = set(self.get_local_branches() + self.get_remote_branches())

		for branch in all_branches:
			branch_type = self.detect_branch_type(branch)
			if branch_type:
				result[branch_type].append(branch)
			else:
				result["other"].append(branch)

		return result

	def get_branch_metadata(self, branch_name: str) -> dict[str, Any]:
		"""
		Get metadata for a specific branch.

		Args:
		    branch_name: Name of the branch

		Returns:
		    Dictionary with branch metadata

		"""
		try:
			# Get last commit date
			date_cmd = [
				"git",
				"log",
				"-1",
				"--format=%ad",
				"--date=relative",
				branch_name if branch_exists(branch_name) else f"origin/{branch_name}",
			]
			date = run_git_command(date_cmd).strip()

			# Get commit count (compared to default branch)
			default = get_default_branch()
			count_cmd = ["git", "rev-list", "--count", f"{default}..{branch_name}"]
			try:
				count = run_git_command(count_cmd).strip()
			except GitError:
				count = "0"

			# Detect branch type
			branch_type = self.detect_branch_type(branch_name)

			return {
				"last_commit_date": date,
				"commit_count": count,
				"branch_type": branch_type,
				"is_local": branch_name in self.get_local_branches(),
				"is_remote": branch_name in self.get_remote_branches(),
			}
		except GitError:
			# Return default metadata if there's an error
			return {
				"last_commit_date": "unknown",
				"commit_count": "0",
				"branch_type": self.detect_branch_type(branch_name),
				"is_local": False,
				"is_remote": False,
			}

	def get_all_branches_with_metadata(self) -> dict[str, dict[str, Any]]:
		"""
		Get all branches with metadata.

		Returns:
		    Dictionary mapping branch names to metadata dictionaries

		"""
		result = {}
		all_branches = set(self.get_local_branches() + self.get_remote_branches())

		for branch in all_branches:
			result[branch] = self.get_branch_metadata(branch)

		return result


class GitHubFlowStrategy(WorkflowStrategy):
	"""Implementation of GitHub Flow workflow strategy."""

	def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
		"""
		Get the default base branch for GitHub Flow.

		Args:
		    branch_type: Type of branch (always 'feature' in GitHub Flow)

		Returns:
		    Name of the default base branch (usually 'main')

		"""
		# Ignoring branch_type as GitHub Flow always uses the default branch
		return get_default_branch()

	def get_branch_prefix(self, branch_type: str) -> str:  # noqa: ARG002
		"""
		Get the branch name prefix for GitHub Flow.

		Args:
		    branch_type: Type of branch (always 'feature' in GitHub Flow)

		Returns:
		    Branch name prefix (empty string for GitHub Flow)

		"""
		# Ignoring branch_type as GitHub Flow doesn't use prefixes
		return ""

	def get_branch_types(self) -> list[str]:
		"""
		Get valid branch types for GitHub Flow.

		Returns:
		    List containing only 'feature'

		"""
		return ["feature"]

	def get_pr_templates(self, branch_type: str) -> dict[str, str]:  # noqa: ARG002
		"""
		Get PR title and description templates for GitHub Flow.

		Args:
		    branch_type: Type of branch (always 'feature' in GitHub Flow)

		Returns:
		    Dictionary with 'title' and 'description' templates

		"""
		return GITHUB_FLOW_PR_TEMPLATE


class GitFlowStrategy(WorkflowStrategy):
	"""Implementation of GitFlow workflow strategy."""

	def get_default_base(self, branch_type: str) -> str:
		"""
		Get the default base branch for GitFlow.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, bugfix)

		Returns:
		    Name of the default base branch

		"""
		mapping = {
			"feature": "develop",
			"release": "main",
			"hotfix": "main",
			"bugfix": "develop",
		}
		default = get_default_branch()
		return mapping.get(branch_type, default)

	def get_branch_prefix(self, branch_type: str) -> str:
		"""
		Get the branch name prefix for GitFlow.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)

		Returns:
		    Branch name prefix

		"""
		mapping = {
			"feature": "feature/",
			"release": "release/",
			"hotfix": "hotfix/",
			"bugfix": "bugfix/",
		}
		return mapping.get(branch_type, "")

	def get_branch_types(self) -> list[str]:
		"""
		Get valid branch types for GitFlow.

		Returns:
		    List of valid branch types for GitFlow

		"""
		return ["feature", "release", "hotfix", "bugfix"]

	def suggest_branch_name(self, branch_type: str, description: str) -> str:
		"""
		Suggest a branch name based on GitFlow conventions.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)
		    description: Description of the branch

		Returns:
		    Suggested branch name

		"""
		prefix = self.get_branch_prefix(branch_type)

		if branch_type == "release":
			# Extract version number from description if it looks like a version
			version_match = re.search(r"(\d+\.\d+\.\d+)", description)
			if version_match:
				return f"{prefix}{version_match.group(1)}"

		# For other branch types, use the default implementation
		return super().suggest_branch_name(branch_type, description)

	def get_pr_templates(self, branch_type: str) -> dict[str, str]:
		"""
		Get PR title and description templates for GitFlow.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, bugfix)

		Returns:
		    Dictionary with 'title' and 'description' templates

		"""
		return GITFLOW_PR_TEMPLATES.get(branch_type, DEFAULT_PR_TEMPLATE)


class TrunkBasedStrategy(WorkflowStrategy):
	"""Implementation of Trunk-Based Development workflow strategy."""

	def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
		"""
		Get the default base branch for Trunk-Based Development.

		Args:
		    branch_type: Type of branch

		Returns:
		    Name of the default base branch (trunk, which is usually 'main')

		"""
		# Ignoring branch_type as Trunk-Based Development always uses the main branch
		return get_default_branch()

	def get_branch_prefix(self, branch_type: str) -> str:
		"""
		Get the branch name prefix for Trunk-Based Development.

		Args:
		    branch_type: Type of branch

		Returns:
		    Branch name prefix

		"""
		return "fb/" if branch_type == "feature" else ""

	def get_branch_types(self) -> list[str]:
		"""
		Get valid branch types for Trunk-Based Development.

		Returns:
		    List containing only 'feature'

		"""
		return ["feature"]

	def suggest_branch_name(self, branch_type: str, description: str) -> str:
		"""
		Suggest a branch name based on Trunk-Based Development conventions.

		Emphasizes short-lived, descriptive branches.

		Args:
		    branch_type: Type of branch
		    description: Description of the branch

		Returns:
		    Suggested branch name

		"""
		# For trunk-based development, try to generate very short names
		words = description.split()
		# Filter out common words like "implement", "the", "and", etc.
		common_words = ["the", "and", "for", "with", "implement", "implementing", "implementation"]
		words = [w for w in words if len(w) > MIN_SIGNIFICANT_WORD_LENGTH and w.lower() not in common_words]

		# Take up to 3 significant words
		short_desc = "-".join(words[:3]).lower()
		short_desc = re.sub(r"[^a-zA-Z0-9-]", "-", short_desc)
		short_desc = re.sub(r"-+", "-", short_desc)
		short_desc = short_desc.strip("-")

		# Add username prefix for trunk-based (optional)
		try:
			username = run_git_command(["git", "config", "user.name"]).strip().split()[0].lower()
			username = re.sub(r"[^a-zA-Z0-9]", "", username)
			return f"{username}/{short_desc}"
		except (GitError, IndexError):
			# Fall back to standard prefix if username not available
			prefix = self.get_branch_prefix(branch_type)
			return f"{prefix}{short_desc}"

	def get_pr_templates(self, branch_type: str) -> dict[str, str]:  # noqa: ARG002
		"""
		Get PR title and description templates for Trunk-Based Development.

		Args:
		    branch_type: Type of branch

		Returns:
		    Dictionary with 'title' and 'description' templates

		"""
		return TRUNK_BASED_PR_TEMPLATE


def get_strategy_class(strategy_name: str) -> type[WorkflowStrategy] | None:
	"""
	Get the workflow strategy class corresponding to the strategy name.

	Args:
	    strategy_name: Name of the workflow strategy

	Returns:
	    Workflow strategy class or None if not found

	"""
	strategy_map = {
		"github-flow": GitHubFlowStrategy,
		"gitflow": GitFlowStrategy,
		"trunk-based": TrunkBasedStrategy,
	}
	return strategy_map.get(strategy_name)


def create_strategy(strategy_name: str) -> WorkflowStrategy:
	"""
	Create a workflow strategy instance based on the strategy name.

	Args:
	    strategy_name: The name of the workflow strategy to create.

	Returns:
	    An instance of the requested workflow strategy.

	Raises:
	    ValueError: If the strategy name is unknown.

	"""
	strategy_class = get_strategy_class(strategy_name)
	if not strategy_class:
		error_msg = f"Unknown workflow strategy: {strategy_name}"
		raise ValueError(error_msg)

	return strategy_class()


# Utility functions to avoid circular imports
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
	except GitError:
		return False


def get_default_branch() -> str:
	"""
	Get the default branch of the repository.

	Returns:
	    Name of the default branch (usually main or master)

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

		# Last resort, use current branch
		return run_git_command(["git", "branch", "--show-current"]).strip()
	except GitError:
		return "main"
