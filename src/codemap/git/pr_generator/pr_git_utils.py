"""Utility functions for PR generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pygit2 import Commit
from pygit2 import GitError as Pygit2GitError
from pygit2.enums import SortMode

from codemap.git.utils import ExtendedGitRepoContext, GitError

if TYPE_CHECKING:
	from pygit2 import Oid


logger = logging.getLogger(__name__)


class PRGitUtils(ExtendedGitRepoContext):
	"""Provides Git operations for PR generation using pygit2."""

	_pr_git_utils_instance: PRGitUtils | None = None

	@classmethod
	def get_instance(cls) -> PRGitUtils:
		"""Get an instance of the PRGitUtils class."""
		if cls._pr_git_utils_instance is None:
			cls._pr_git_utils_instance = cls()
		return cls._pr_git_utils_instance

	def __init__(self) -> None:
		"""Initialize the PRGitUtils with the given repository path."""
		super().__init__()

	def create_branch(self, branch_name: str, from_reference: str | None = None) -> None:
		"""
		Create a new branch and switch to it using pygit2.

		Args:
		    branch_name: Name of the branch to create.
		    from_reference: Optional reference (branch name, commit SHA) to create the branch from.
		                    Defaults to current HEAD.

		Raises:
		    GitError: If branch creation or checkout fails.
		"""
		try:
			if from_reference:
				commit_obj = self.repo.revparse_single(from_reference)
				if not commit_obj:
					msg = f"Could not resolve 'from_reference': {from_reference}"
					logger.error(msg)
					raise GitError(msg)
				source_commit = commit_obj.peel(Commit)
			else:
				if self.repo.head_is_unborn:
					msg = "Cannot create branch from unborn HEAD. Please make an initial commit."
					logger.error(msg)
					raise GitError(msg)
				source_commit = self.repo.head.peel(Commit)

			self.repo.create_branch(branch_name, source_commit)
			logger.info(f"Branch '{branch_name}' created from '{source_commit.id}'.")
			self.checkout_branch(branch_name)  # Checkout after creation
		except GitError as e:
			msg = f"Failed to create branch '{branch_name}' using pygit2: {e}"
			logger.exception(msg)
			raise GitError(msg) from e
		except Exception as e:
			msg = f"An unexpected error occurred while creating branch '{branch_name}': {e}"
			logger.exception(msg)
			raise GitError(msg) from e

	def checkout_branch(self, branch_name: str) -> None:
		"""
		Checkout an existing branch using pygit2.

		Args:
		    branch_name: Name of the branch to checkout.

		Raises:
		    GitError: If checkout fails.
		"""
		try:
			# Construct the full ref name
			ref_name = f"refs/heads/{branch_name}"
			branch_obj = self.repo.lookup_reference(ref_name)
			self.repo.checkout(branch_obj)
			# Update self.branch after checkout, consistent with GitRepoContext constructor
			current_branch_obj = self.repo
			if not current_branch_obj.head_is_detached:
				self.branch = current_branch_obj.head.shorthand
			else:
				self.branch = ""  # Or perhaps the SHA for detached head
			logger.info(f"Checked out branch '{branch_name}' using pygit2.")
		except GitError as e:
			msg = f"Failed to checkout branch '{branch_name}' using pygit2: {e}"
			logger.exception(msg)
			raise GitError(msg) from e
		except Exception as e:
			msg = f"An unexpected error occurred while checking out branch '{branch_name}': {e}"
			logger.exception(msg)
			raise GitError(msg) from e

	def push_branch(self, branch_name: str, force: bool = False, remote_name: str = "origin") -> None:
		"""
		Push a branch to the remote using pygit2.

		Args:
		    branch_name: Name of the branch to push.
		    force: Whether to force push.
		    remote_name: Name of the remote (e.g., "origin").

		Raises:
		    GitError: If push fails.
		"""
		try:
			remote = self.repo.remotes[remote_name]
			local_ref = f"refs/heads/{branch_name}"
			remote_ref = f"refs/heads/{branch_name}"

			refspec = f"{'+' if force else ''}{local_ref}:{remote_ref}"

			logger.info(
				f"Attempting to push branch '{branch_name}' to remote "
				f"'{remote_name}' with refspec '{refspec}' using pygit2."
			)
			# Note: pygit2 push may require credential callbacks for private repos
			# or specific auth methods. This implementation assumes credentials
			# are globally configured for Git (e.g., via credential helper, SSH agent).
			remote.push([refspec])
			logger.info(f"Branch '{branch_name}' pushed to remote '{remote_name}' using pygit2.")
		except (Pygit2GitError, KeyError) as e:  # KeyError for remote_name not found
			msg = f"Failed to push branch '{branch_name}' to remote '{remote_name}' using pygit2: {e}"
			logger.exception(msg)
			raise GitError(msg) from e
		except GitError as e:  # Catch codemap's GitError if it somehow occurred before
			msg = f"Git operation error while pushing branch '{branch_name}': {e}"
			logger.exception(msg)
			raise GitError(msg) from e
		except Exception as e:
			msg = f"An unexpected error occurred while pushing branch '{branch_name}': {e}"
			logger.exception(msg)
			raise GitError(msg) from e

	def get_commit_messages(self, base_branch: str, head_branch: str) -> list[str]:
		"""
		Get commit messages (summaries) between two branches using pygit2.

		This lists commits that are in head_branch but not in base_branch.

		Args:
		    base_branch: Base branch name/ref (e.g., "main").
		    head_branch: Head branch name/ref (e.g., "feature-branch").

		Returns:
		    List of commit message summaries.

		Raises:
		    GitError: If retrieving commits fails.
		"""
		try:
			if not base_branch or not head_branch:
				logger.warning("Base or head branch is None/empty, cannot get commit messages.")
				return []

			def _resolve_to_commit_oid(branch_spec: str) -> Oid:
				obj = self.repo.revparse_single(branch_spec)
				if not obj:
					msg = f"Could not resolve '{branch_spec}'"
					logger.error(msg)
					raise GitError(msg)
				# Ensure it's a commit (could be a tag pointing to another tag, etc.)
				commit_obj = obj.peel(Commit)
				return commit_obj.id

			base_oid = _resolve_to_commit_oid(base_branch)
			head_oid = _resolve_to_commit_oid(head_branch)

			walker = self.repo.walk(head_oid, SortMode.TOPOLOGICAL)
			walker.hide(base_oid)

			commit_messages = []
			for commit_pygit2 in walker:
				# commit_pygit2.message is the full message. Get summary (first line).
				message_summary = commit_pygit2.message.splitlines()[0].strip() if commit_pygit2.message else ""
				commit_messages.append(message_summary)

			logger.info(f"Found {len(commit_messages)} commit messages between '{base_branch}' and '{head_branch}'.")
			return commit_messages

		except (Pygit2GitError, GitError) as e:
			msg = f"Failed to get commit messages between '{base_branch}' and '{head_branch}' using pygit2: {e}"
			logger.exception(msg)
			raise GitError(msg) from e
		except Exception as e:
			msg = (
				f"An unexpected error occurred while getting commit messages "
				f"between '{base_branch}' and '{head_branch}': {e}"
			)
			logger.exception(msg)
			raise GitError(msg) from e

	def get_branch_relation(self, branch_ref_name: str, target_branch_ref_name: str) -> tuple[bool, int]:
		"""
		Get the relationship between two branches using pygit2.

		Args:
			branch_ref_name: The branch to check (e.g., "main", "origin/main").
			target_branch_ref_name: The target branch to compare against (e.g., "feature/foo").

		Returns:
			Tuple of (is_ancestor, commit_count)
			- is_ancestor: True if branch_ref_name is an ancestor of target_branch_ref_name.
			- commit_count: Number of commits in target_branch_ref_name that are not in branch_ref_name.
						(i.e., how many commits target is "ahead" of branch).

		Raises:
			GitError: If branches cannot be resolved or other git issues occur.
		"""
		try:
			if not branch_ref_name or not target_branch_ref_name:
				logger.warning("Branch or target branch name is None/empty for relation check.")
				return False, 0

			# Resolve branch names to Oids. revparse_single can handle local and remote-like refs.
			branch_commit_obj = self.repo.revparse_single(branch_ref_name)
			if not branch_commit_obj:
				msg = f"Could not resolve branch: {branch_ref_name}"
				logger.error(msg)
				raise GitError(msg)
			branch_oid = branch_commit_obj.peel(Commit).id

			target_commit_obj = self.repo.revparse_single(target_branch_ref_name)
			if not target_commit_obj:
				msg = f"Could not resolve target branch: {target_branch_ref_name}"
				logger.error(msg)
				raise GitError(msg)
			target_oid = target_commit_obj.peel(Commit).id

			# Check if branch_oid is an ancestor of target_oid
			# pygit2's descendant_of(A, B) means "is A a descendant of B?"
			# So, is_ancestor (branch is ancestor of target) means target is descendant of branch.
			is_ancestor = self.repo.descendant_of(target_oid, branch_oid)

			# Get commit count: commits in target_oid that are not in branch_oid.
			# ahead_behind(A, B) returns (commits in A not in B, commits in B not in A)
			# We want commits in target_oid not in branch_oid.
			# So, if A=target_oid, B=branch_oid, we want the first value (ahead).
			ahead, _ = self.repo.ahead_behind(target_oid, branch_oid)
			commit_count_target_ahead = ahead  # Renaming for clarity

			logger.debug(
				f"Branch relation: {branch_ref_name} vs {target_branch_ref_name}. "
				f"Is ancestor: {is_ancestor}, Target ahead by: {commit_count_target_ahead}"
			)
			return is_ancestor, commit_count_target_ahead

		except Pygit2GitError as e:
			msg = (
				f"Pygit2 error determining branch relation between "
				f"'{branch_ref_name}' and '{target_branch_ref_name}': {e}"
			)
			logger.warning(msg)
			raise GitError(msg) from e  # Wrap in codemap's GitError
		except GitError as e:  # Catch codemap's GitError if raised by _resolve_to_commit_oid or similar
			msg = (
				f"Codemap GitError determining branch relation between '{branch_ref_name}' and "
				f"'{target_branch_ref_name}': {e}"
			)
			logger.warning(msg)
			raise  # Re-raise as it's already the correct type
		except Exception as e:  # Catch any other unexpected non-Git errors
			msg = (
				f"Unexpected error determining branch relation between '{branch_ref_name}' and "
				f"'{target_branch_ref_name}': {e}"
			)
			logger.warning(msg)
			raise GitError(msg) from e  # Wrap in codemap's GitError
