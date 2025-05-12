"""Git workflow strategy implementations for PR management."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from pygit2 import Commit
from pygit2 import GitError as Pygit2GitError
from pygit2.enums import BranchType, ReferenceType

from codemap.git.pr_generator.constants import MIN_SIGNIFICANT_WORD_LENGTH
from codemap.git.pr_generator.pr_git_utils import PRGitUtils
from codemap.git.pr_generator.templates import (
	DEFAULT_PR_TEMPLATE,
	GITFLOW_PR_TEMPLATES,
	GITHUB_FLOW_PR_TEMPLATE,
	TRUNK_BASED_PR_TEMPLATE,
)
from codemap.git.utils import GitError


class WorkflowStrategy(ABC):
	"""Base class for git workflow strategies."""

	@abstractmethod
	def get_default_base(self, branch_type: str) -> str | None:
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
		clean_description = re.sub(r"[^a-zA-Z0-9]+", "-", description.lower()).strip("-")
		prefix = self.get_branch_prefix(branch_type)
		return f"{prefix}{clean_description}"

	@abstractmethod
	def get_branch_prefix(self, branch_type: str) -> str:
		"""
		Get the branch name prefix for a given branch type.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)

		Returns:
		    Branch name prefix

		"""
		raise NotImplementedError

	@abstractmethod
	def get_branch_types(self) -> list[str]:
		"""
		Get valid branch types for this workflow.

		Returns:
		    List of valid branch types

		"""
		raise NotImplementedError

	def detect_branch_type(self, branch_name: str | None) -> str | None:
		"""
		Detect the type of a branch from its name.

		Args:
		    branch_name: Name of the branch

		Returns:
		    Branch type or None if not detected

		"""
		for branch_type in self.get_branch_types():
			prefix = self.get_branch_prefix(branch_type)
			if branch_name and branch_name.startswith(prefix):
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
		return DEFAULT_PR_TEMPLATE

	def get_remote_branches(self) -> list[str]:
		"""
		Get list of remote branches.

		Returns:
		    List of remote branch names (without 'origin/' prefix typically)

		"""
		try:
			pgu = PRGitUtils.get_instance()
			remote_branches = []
			for b_name in pgu.repo.branches.remote:
				if b_name.startswith("origin/"):
					branch_name_without_prefix = b_name[len("origin/") :]
					if not branch_name_without_prefix.startswith("HEAD"):
						remote_branches.append(branch_name_without_prefix)
				elif "/" in b_name and not b_name.endswith("/HEAD"):
					parts = b_name.split("/", 1)
					if len(parts) > 1:
						remote_branches.append(parts[1])
			return list(set(remote_branches))
		except (GitError, Pygit2GitError) as e:
			PRGitUtils.logger.debug(f"Error getting remote branches: {e}")
			return []
		except Exception as e:  # noqa: BLE001
			PRGitUtils.logger.debug(f"Unexpected error getting remote branches: {e}")
			return []

	def get_local_branches(self) -> list[str]:
		"""
		Get list of local branches.

		Returns:
		    List of local branch names

		"""
		try:
			pgu = PRGitUtils.get_instance()
			return list(pgu.repo.branches.local)
		except (GitError, Pygit2GitError) as e:
			PRGitUtils.logger.debug(f"Error getting local branches: {e}")
			return []
		except Exception as e:  # noqa: BLE001
			PRGitUtils.logger.debug(f"Unexpected error getting local branches: {e}")
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
			pgu = PRGitUtils.get_instance()
			repo = pgu.repo

			# Determine full ref for revparse_single (try local, then remote)
			branch_ref_to_parse = branch_name
			if not branch_exists(branch_name, pgu_instance=pgu, include_remote=False) and branch_exists(
				branch_name, pgu_instance=pgu, remote_name="origin", include_local=False
			):
				branch_ref_to_parse = f"origin/{branch_name}"
			# If still not found by branch_exists, revparse_single might fail, which is caught below.

			last_commit_iso_date = "unknown"
			try:
				commit_obj = repo.revparse_single(branch_ref_to_parse).peel(Commit)
				commit_time = datetime.fromtimestamp(commit_obj.commit_time, tz=UTC)
				last_commit_iso_date = commit_time.isoformat()
			except (Pygit2GitError, GitError) as e:  # Catch errors resolving commit
				PRGitUtils.logger.debug(f"Could not get last commit date for {branch_ref_to_parse}: {e}")

			commit_count_str = "0"
			try:
				default_b = get_default_branch(pgu_instance=pgu)
				if default_b:
					# get_branch_relation expects full ref names or resolvable names
					_, count = pgu.get_branch_relation(default_b, branch_ref_to_parse)
					commit_count_str = str(count)
			except (GitError, Pygit2GitError) as e:
				PRGitUtils.logger.debug(f"Could not get commit count for {branch_ref_to_parse} vs default: {e}")

			branch_type_detected = self.detect_branch_type(branch_name)

			return {
				"last_commit_date": last_commit_iso_date,
				"commit_count": commit_count_str,
				"branch_type": branch_type_detected,
				"is_local": branch_exists(branch_name, pgu_instance=pgu, include_remote=False, include_local=True),
				"is_remote": branch_exists(branch_name, pgu_instance=pgu, remote_name="origin", include_local=False),
			}
		except (GitError, Pygit2GitError) as e:
			PRGitUtils.logger.warning(f"Error getting branch metadata for {branch_name}: {e}")
			return {  # Fallback for broader errors during pgu instantiation or initial checks
				"last_commit_date": "unknown",
				"commit_count": "0",
				"branch_type": self.detect_branch_type(branch_name),
				"is_local": False,
				"is_remote": False,
			}
		except Exception as e:  # noqa: BLE001
			PRGitUtils.logger.warning(f"Unexpected error getting branch metadata for {branch_name}: {e}")
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
		# Using PRGitUtils for a consistent list of branches
		pgu = PRGitUtils.get_instance()
		local_b = list(pgu.repo.branches.local)
		remote_b_parsed = []
		for rb_name in pgu.repo.branches.remote:
			if rb_name.startswith("origin/") and not rb_name.endswith("/HEAD"):
				remote_b_parsed.append(rb_name[len("origin/") :])
			elif "/" in rb_name and not rb_name.endswith("/HEAD"):
				parts = rb_name.split("/", 1)
				if len(parts) > 1:
					remote_b_parsed.append(parts[1])

		all_branches = set(local_b + remote_b_parsed)

		for branch in all_branches:
			result[branch] = self.get_branch_metadata(branch)

		return result


class GitHubFlowStrategy(WorkflowStrategy):
	"""Implementation of GitHub Flow workflow strategy."""

	def get_default_base(self, branch_type: str) -> str | None:  # noqa: ARG002
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

	def get_default_base(self, branch_type: str) -> str | None:
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

	def get_default_base(self, branch_type: str) -> str | None:  # noqa: ARG002
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
			pgu = PRGitUtils.get_instance()
			# Ensure config is available and handle potential errors
			user_name_config = pgu.repo.config["user.name"]
			if user_name_config:
				# Ensure user_name_config is treated as a string before strip/split
				username = str(user_name_config).strip().split()[0].lower()
				username = re.sub(r"[^a-zA-Z0-9]", "", username)
				return f"{username}/{short_desc if short_desc else 'update'}"  # ensure short_desc is not empty
			# Fallback if username is not configured
			prefix = self.get_branch_prefix(branch_type)
			return f"{prefix}{short_desc if short_desc else 'update'}"
		except (GitError, Pygit2GitError, KeyError, IndexError, AttributeError) as e:  # Catch more specific errors
			PRGitUtils.logger.debug(f"Could not get username for branch prefix: {e}")
			prefix = self.get_branch_prefix(branch_type)
			return f"{prefix}{short_desc if short_desc else 'update'}"
		except Exception as e:  # noqa: BLE001
			PRGitUtils.logger.debug(f"Unexpected error getting username for branch prefix: {e}")
			prefix = self.get_branch_prefix(branch_type)
			return f"{prefix}{short_desc if short_desc else 'update'}"

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


# Utility functions, now using PRGitUtils
def branch_exists(
	branch_name: str,
	pgu_instance: PRGitUtils | None = None,
	remote_name: str = "origin",
	include_remote: bool = True,
	include_local: bool = True,
) -> bool:
	"""
	Check if a branch exists using pygit2.

	Args:
	    branch_name: Name of the branch to check (e.g., "main", "feature/foo").
	    pgu_instance: Optional instance of PRGitUtils. If None, one will be created.
	    remote_name: The name of the remote to check (default: "origin").
	    include_remote: Whether to check remote branches.
	    include_local: Whether to check local branches.

	Returns:
	    True if the branch exists in the specified locations, False otherwise.
	"""
	if not branch_name:
		return False

	pgu = pgu_instance or PRGitUtils.get_instance()
	repo = pgu.repo

	if include_local:
		try:
			# lookup_branch checks local branches by default if branch_type is not specified
			# or if BranchType.LOCAL is used.
			if repo.lookup_branch(branch_name, BranchType.LOCAL):  # Explicitly check local
				return True
		except (KeyError, Pygit2GitError):  # lookup_branch raises KeyError if not found
			pass  # Not found locally

	if include_remote:
		remote_branch_ref = f"{remote_name}/{branch_name}"
		try:
			# To check a remote branch, we look it up by its full remote-prefixed name
			# in the list of remote branches pygit2 knows.
			# An alternative is repo.lookup_reference(f"refs/remotes/{remote_name}/{branch_name}")
			if remote_branch_ref in repo.branches.remote:
				return True
		except (KeyError, Pygit2GitError):  # Should not happen with `in` check
			pass  # Not found remotely

	return False


def get_default_branch(pgu_instance: PRGitUtils | None = None) -> str:
	"""
	Get the default branch of the repository using pygit2.

	Args:
	    pgu_instance: Optional instance of PRGitUtils.

	Returns:
	    Name of the default branch (e.g., "main", "master").
	"""
	pgu = pgu_instance or PRGitUtils.get_instance()
	repo = pgu.repo
	try:
		if not repo.head_is_detached:
			# Current HEAD is a symbolic ref to a branch, this is often the default
			# if on the default branch.
			# However, this returns the current branch, not necessarily default.
			# current_branch = repo.head.shorthand
			# if current_branch in ["main", "master"]: return current_branch
			pass  # Fall through to more robust checks for default

		# Try to get the symbolic-ref of refs/remotes/origin/HEAD
		try:
			origin_head_ref = repo.lookup_reference("refs/remotes/origin/HEAD")
			if origin_head_ref and origin_head_ref.type == ReferenceType.SYMBOLIC:
				target_as_val = origin_head_ref.target
				# Target is like 'refs/remotes/origin/main'
				# If type is SYMBOLIC, target should be str. Add isinstance to help linter.
				if isinstance(target_as_val, str) and target_as_val.startswith("refs/remotes/origin/"):
					return target_as_val[len("refs/remotes/origin/") :]
		except (KeyError, Pygit2GitError):
			pass  # origin/HEAD might not exist or not be symbolic

		# Fallback: check for common default branch names ('main', then 'master')
		# Check remote branches first as they are more indicative of shared default
		if "origin/main" in repo.branches.remote:
			return "main"
		if "origin/master" in repo.branches.remote:
			return "master"
		# Then check local branches
		if "main" in repo.branches.local:
			return "main"
		if "master" in repo.branches.local:
			return "master"

		# If still not found, and HEAD is not detached, use current branch as last resort.
		if not repo.head_is_detached:
			return repo.head.shorthand

	except (GitError, Pygit2GitError) as e:
		PRGitUtils.logger.warning(f"Could not determine default branch via pygit2: {e}. Falling back to 'main'.")
	except Exception as e:  # noqa: BLE001
		PRGitUtils.logger.warning(f"Unexpected error determining default branch: {e}. Falling back to 'main'.")

	return "main"  # Ultimate fallback
