"""Utility functions for PR generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pygit2 import Commit
from pygit2 import GitError as Pygit2GitError
from pygit2.enums import SortMode

from codemap.git.utils import ExtendedGitRepoContext, GitError
from codemap.utils.git_hooks import hook_exists, run_hook

if TYPE_CHECKING:
	from pygit2 import Oid


logger = logging.getLogger(__name__)


class PRGitUtils(ExtendedGitRepoContext):
	"""Provides Git operations for PR generation using pygit2."""

	_pr_git_utils_instance: PRGitUtils | None = None
	branch: str  # Explicitly declare the inherited attribute

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
			# Run post-checkout hook if present
			if hook_exists("post-checkout"):
				exit_code = run_hook("post-checkout")
				if exit_code != 0:
					logger.warning("post-checkout hook failed (branch already checked out)")
		except GitError as e:
			msg = f"Failed to checkout branch '{branch_name}' using pygit2: {e}"
			logger.exception(msg)
			raise GitError(msg) from e
		except Exception as e:
			msg = f"An unexpected error occurred while checking out branch '{branch_name}': {e}"
			logger.exception(msg)
			raise GitError(msg) from e

	def push_branch(
		self, branch_name: str, force: bool = False, remote_name: str = "origin", ignore_hooks: bool = False
	) -> None:
		"""
		Push a branch to the remote using pygit2.

		Args:
		    branch_name: Name of the branch to push.
		    force: Whether to force push.
		    remote_name: Name of the remote (e.g., "origin").
		    ignore_hooks: If True, skip running the pre-push hook.

		Raises:
		    GitError: If push fails or pre-push hook fails.
		"""
		# Run pre-push hook if not ignored
		if not ignore_hooks:
			exit_code = run_hook("pre-push")
			if exit_code != 0:
				msg = "pre-push hook failed, aborting push."
				logger.error(msg)
				raise GitError(msg)
		try:
			remote = self.repo.remotes[remote_name]
			local_ref = f"refs/heads/{branch_name}"
			remote_ref = f"refs/heads/{branch_name}"

			refspec = f"{'+' if force else ''}{local_ref}:{remote_ref}"

			logger.info(
				f"Attempting to push branch '{branch_name}' to remote "
				f"'{remote_name}' with refspec '{refspec}' using pygit2."
			)

			# Import the proper pygit2 credential classes
			import shlex
			import subprocess
			from pathlib import Path
			from urllib.parse import urlparse

			from pygit2.callbacks import RemoteCallbacks
			from pygit2.enums import CredentialType

			# Create a credential class to handle SSH and username/password authentication
			class GitCredential:
				def __init__(self, cred_type: CredentialType, *args: str | None) -> None:
					self.credential_type = cred_type
					self.credential_tuple = args

			def credential_callback(
				url: str, username_from_url: str | None, allowed_types: CredentialType
			) -> GitCredential:
				"""
				Callback to handle credential requests from pygit2.

				Args:
					url: The URL being authenticated against
					username_from_url: Username extracted from the URL if present
					allowed_types: Bitmask of allowed credential types

				Returns:
					A credential object for authentication
				"""
				logger.debug(f"Authentication required for {url} (allowed types: {allowed_types})")

				# Get username from URL or use default from git config
				username = username_from_url
				if not username:
					try:
						config = self.repo.config
						username = config["user.name"]
					except (KeyError, AttributeError) as e:
						# Default if we can't get from config
						logger.debug(f"Could not get username from git config: {e}")
						username = "git"

				# Try SSH agent authentication first (if available)
				if CredentialType.SSH_KEY in allowed_types:
					logger.debug(f"Attempting SSH agent authentication for {username}")
					return GitCredential(CredentialType.SSH_KEY, username, None, None, "")

				# Try SSH key authentication if agent is not available
				if CredentialType.SSH_KEY in allowed_types:
					try:
						# Common SSH key paths
						ssh_dir = Path.home() / ".ssh"
						key_paths = [
							ssh_dir / "id_rsa",
							ssh_dir / "id_ed25519",
							ssh_dir / "id_ecdsa",
							ssh_dir / "id_dsa",
						]

						for private_key_path in key_paths:
							public_key_path = Path(f"{private_key_path}.pub")
							if private_key_path.exists() and public_key_path.exists():
								logger.debug(f"Attempting SSH key authentication with {private_key_path}")
								return GitCredential(
									CredentialType.SSH_KEY, username, str(public_key_path), str(private_key_path), ""
								)
					except OSError as e:
						logger.debug(f"SSH key authentication failed: {e}")

				# Try username/password if SSH is not available or didn't work
				if CredentialType.USERPASS_PLAINTEXT in allowed_types:
					try:
						# Extract hostname from URL
						parsed_url = urlparse(url)
						hostname = parsed_url.netloc

						# Use git credential fill to get credentials - this command is safe as it's hardcoded
						cmd = "git credential fill"
						# Use shlex.split for secure command execution
						process = subprocess.Popen(  # noqa: S603
							shlex.split(cmd),
							stdin=subprocess.PIPE,
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE,
							text=True,
						)

						# Provide input for git credential fill
						input_data = f"protocol={parsed_url.scheme}\nhost={hostname}\n\n"
						stdout, _ = process.communicate(input=input_data)

						if process.returncode == 0 and stdout:
							# Parse the output
							credentials = {}
							for line in stdout.splitlines():
								if "=" in line:
									key, value = line.split("=", 1)
									credentials[key] = value

							if "username" in credentials and "password" in credentials:
								logger.debug(f"Using username/password authentication for {credentials['username']}")
								return GitCredential(
									CredentialType.USERPASS_PLAINTEXT, credentials["username"], credentials["password"]
								)
					except (subprocess.SubprocessError, OSError) as e:
						logger.debug(f"Username/password authentication failed: {e}")

				# If nothing else works, try username-only authentication
				if CredentialType.USERNAME in allowed_types:
					logger.debug(f"Falling back to username-only authentication for {username}")
					return GitCredential(CredentialType.USERNAME, username)

				# If we get here, we couldn't find suitable credentials
				logger.warning(f"No suitable authentication method found for {url}")
				msg = "No suitable authentication method available"
				raise Pygit2GitError(msg)

			# Create callback object with our credential callback
			callbacks = RemoteCallbacks(credentials=credential_callback)

			# Pass callbacks to the push method
			remote.push([refspec], callbacks=callbacks)
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
