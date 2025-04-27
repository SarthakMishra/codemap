"""Tests for the PRCommand class in git/pr_generator/command.py."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from codemap.git.pr_generator.command import PRCommand
from codemap.git.utils import GitError
from codemap.llm import LLMError
from tests.base import GitTestBase


@pytest.mark.unit
@pytest.mark.git
class TestPRCommandInitialization(GitTestBase):
	"""Test cases for the initialization of the PRCommand class."""

	def setup_method(self) -> None:
		"""Set up test environment with mocks."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Patch get_repo_root to avoid actual Git operations
		self.mock_repo_path("/mock/repo/path")
		self.repo_path = "/mock/repo/path"  # Store the path value for later assertions

	def test_init_success(self) -> None:
		"""Test successful initialization of PRCommand."""
		# Arrange: Set up mocks
		with (
			patch("codemap.llm.create_client") as mock_create_client,
			patch("codemap.git.pr_generator.command.get_repo_root") as mock_get_repo_root,
		):
			mock_llm_client = Mock()
			mock_create_client.return_value = mock_llm_client
			mock_get_repo_root.return_value = self.repo_path  # Ensure consistent repo path

			# Act: Initialize PRCommand
			pr_command = PRCommand(path=None, model="gpt-4o-mini")

			# Assert: Verify repository root and error state
			assert pr_command.repo_root == self.repo_path
			assert pr_command.error_state is None
			assert pr_command.pr_generator is not None

			# Verify create_client was called with correct parameters
			mock_create_client.assert_called_once_with(repo_path=self.repo_path, model="gpt-4o-mini")

	def test_init_git_error(self) -> None:
		"""Test error handling when Git operations fail during initialization."""
		# Arrange: Set up mocks
		with patch("codemap.git.pr_generator.command.get_repo_root") as mock_get_repo_root:
			# Configure mocks to raise GitError
			mock_get_repo_root.side_effect = GitError("Not a git repository")

			# Act and Assert: Should raise RuntimeError
			with pytest.raises(RuntimeError) as excinfo:
				PRCommand(path=None)

			# Verify error message
			assert "Not a git repository" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.git
class TestPRCommandBranchInfo(GitTestBase):
	"""Test cases for the _get_branch_info method in PRCommand."""

	def setup_method(self) -> None:
		"""Set up test environment with mocks."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Patch get_repo_root to avoid actual Git operations
		self.mock_repo_path("/mock/repo/path")
		self.repo_path = "/mock/repo/path"

		# Create mock LLM client
		self.mock_llm_client = Mock()

		# Create the PRCommand with patched dependencies
		with (
			patch("codemap.llm.create_client", return_value=self.mock_llm_client),
			patch("codemap.git.pr_generator.command.get_repo_root", return_value=self.repo_path),
		):
			self.pr_command = PRCommand()

	def test_get_branch_info_success(self) -> None:
		"""Test successful retrieval of branch information."""
		# Arrange: Set up mocks for git operations
		with patch("codemap.git.pr_generator.command.run_git_command") as mock_run_git:
			# Configure mocks
			mock_run_git.side_effect = [
				"feature-branch",  # Current branch
				"  HEAD branch: main\n  Remote branches:\n    main\n    develop",  # Default branch
			]

			# Act: Call the method
			branch_info = self.pr_command._get_branch_info()

			# Assert: Verify results
			assert branch_info["current_branch"] == "feature-branch"
			assert branch_info["target_branch"] == "main"

			# Verify git commands were called
			assert mock_run_git.call_count == 2
			assert mock_run_git.call_args_list[0][0][0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
			assert mock_run_git.call_args_list[1][0][0] == ["git", "remote", "show", "origin"]

	def test_get_branch_info_git_error(self) -> None:
		"""Test error handling when Git operations fail."""
		# Arrange: Set up mocks for git operations
		with patch("codemap.git.pr_generator.command.run_git_command") as mock_run_git:
			# Configure mocks to raise GitError
			mock_run_git.side_effect = GitError("Git operation failed")

			# Act and Assert: Should raise RuntimeError
			with pytest.raises(RuntimeError) as excinfo:
				self.pr_command._get_branch_info()

			# Verify error message
			assert "Failed to get branch information: Git operation failed" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.git
class TestPRCommandCommitHistory(GitTestBase):
	"""Test cases for the _get_commit_history method in PRCommand."""

	def setup_method(self) -> None:
		"""Set up test environment with mocks."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Patch get_repo_root to avoid actual Git operations
		self.mock_repo_path("/mock/repo/path")
		self.repo_path = "/mock/repo/path"

		# Create mock LLM client
		self.mock_llm_client = Mock()

		# Create the PRCommand with patched dependencies
		with (
			patch("codemap.llm.create_client", return_value=self.mock_llm_client),
			patch("codemap.git.pr_generator.command.get_repo_root", return_value=self.repo_path),
		):
			self.pr_command = PRCommand()

	def test_get_commit_history_success(self) -> None:
		"""Test successful retrieval of commit history."""
		# Arrange: Set up mocks for git operations
		with patch("codemap.git.pr_generator.command.run_git_command") as mock_run_git:
			# Configure mocks
			mock_run_git.return_value = "abc123||John Doe||feat: Add new feature\ndef456||Jane Smith||fix: Fix a bug"

			# Act: Call the method
			commits = self.pr_command._get_commit_history("main")

			# Assert: Verify results
			assert len(commits) == 2
			assert commits[0]["hash"] == "abc123"
			assert commits[0]["author"] == "John Doe"
			assert commits[0]["subject"] == "feat: Add new feature"
			assert commits[1]["hash"] == "def456"
			assert commits[1]["author"] == "Jane Smith"
			assert commits[1]["subject"] == "fix: Fix a bug"

			# Verify git command was called
			mock_run_git.assert_called_once_with(["git", "log", "main..HEAD", "--pretty=format:%H||%an||%s"])

	def test_get_commit_history_empty(self) -> None:
		"""Test retrieving an empty commit history."""
		# Arrange: Set up mocks for git operations
		with patch("codemap.git.pr_generator.command.run_git_command") as mock_run_git:
			# Configure mocks
			mock_run_git.return_value = ""

			# Act: Call the method
			commits = self.pr_command._get_commit_history("main")

			# Assert: Verify results
			assert commits == []

	def test_get_commit_history_with_invalid_format(self) -> None:
		"""Test handling of commits with invalid format."""
		# Arrange: Set up mocks for git operations
		with patch("codemap.git.pr_generator.command.run_git_command") as mock_run_git:
			# Configure mocks with an invalid format commit
			mock_run_git.return_value = (
				"abc123||John Doe||feat: Add new feature\n"
				"invalid_format_commit\n"  # Not enough parts
				"def456||Jane Smith||fix: Fix a bug"
			)

			# Act: Call the method
			commits = self.pr_command._get_commit_history("main")

			# Assert: Verify results
			assert len(commits) == 2  # Invalid commit should be skipped
			assert commits[0]["hash"] == "abc123"
			assert commits[1]["hash"] == "def456"

	def test_get_commit_history_git_error(self) -> None:
		"""Test error handling when Git operations fail."""
		# Arrange: Set up mocks for git operations
		with patch("codemap.git.pr_generator.command.run_git_command") as mock_run_git:
			# Configure mocks to raise GitError
			mock_run_git.side_effect = GitError("Git operation failed")

			# Act and Assert: Should raise RuntimeError
			with pytest.raises(RuntimeError) as excinfo:
				self.pr_command._get_commit_history("main")

			# Verify error message
			assert "Failed to get commit history: Git operation failed" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.git
class TestPRCommandDescriptionGeneration(GitTestBase):
	"""Test cases for PR description generation in PRCommand."""

	def setup_method(self) -> None:
		"""Set up test environment with mocks."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Patch get_repo_root to avoid actual Git operations
		self.mock_repo_path("/mock/repo/path")
		self.repo_path = "/mock/repo/path"

		# Create mock objects
		self.mock_llm_client = Mock()
		self.mock_pr_generator = Mock()

		# Create the PRCommand with patched dependencies
		# First patch the create_client in llm module
		patcher1 = patch("codemap.llm.create_client", return_value=self.mock_llm_client)
		self.mock_create_client = patcher1.start()
		self._patchers.append(patcher1)

		# Patch get_repo_root
		patcher2 = patch("codemap.git.pr_generator.command.get_repo_root", return_value=self.repo_path)
		self.mock_get_repo_root = patcher2.start()
		self._patchers.append(patcher2)

		# Then patch the PRGenerator class
		patcher3 = patch("codemap.git.pr_generator.command.PRGenerator", return_value=self.mock_pr_generator)
		self.mock_pr_generator_class = patcher3.start()
		self._patchers.append(patcher3)

		# Create the command
		self.pr_command = PRCommand()

		# Set up branch_info and commits for testing
		self.branch_info = {
			"current_branch": "feature-branch",
			"target_branch": "main",
		}
		self.commits = [
			{"hash": "abc123", "author": "John Doe", "subject": "feat: Add new feature"},
			{"hash": "def456", "author": "Jane Smith", "subject": "fix: Fix a bug"},
		]

	def test_generate_pr_description_success(self) -> None:
		"""Test successful PR description generation."""
		# Arrange: Set up mocks
		with patch("codemap.git.pr_generator.command.loading_spinner"):
			# Configure mock to return a PR content dictionary
			self.mock_pr_generator.generate_content_from_commits.return_value = {
				"title": "Feature: Add new functionality",
				"description": "This PR adds a new feature and fixes a bug.",
			}

			# Act: Call the method
			description = self.pr_command._generate_pr_description(self.branch_info, self.commits)

			# Assert: Verify results
			assert description == "This PR adds a new feature and fixes a bug."

			# Verify generate_content_from_commits was called with correct parameters
			self.mock_pr_generator.generate_content_from_commits.assert_called_once_with(
				base_branch="main", head_branch="feature-branch", use_llm=True
			)

	def test_generate_pr_description_llm_error_fallback(self) -> None:
		"""Test fallback when LLM PR description generation fails."""
		# Arrange: Set up mocks
		with patch("codemap.git.pr_generator.command.loading_spinner"):
			# Configure first call to raise LLMError, second call to succeed
			self.mock_pr_generator.generate_content_from_commits.side_effect = [
				LLMError("LLM service unavailable"),
				{
					"title": "Update feature-branch",
					"description": "Fallback description without LLM.",
				},
			]

			# Act: Call the method
			description = self.pr_command._generate_pr_description(self.branch_info, self.commits)

			# Assert: Verify results
			assert description == "Fallback description without LLM."

			# Verify generate_content_from_commits was called with correct parameters
			assert self.mock_pr_generator.generate_content_from_commits.call_count == 2
			# First call with use_llm=True
			assert self.mock_pr_generator.generate_content_from_commits.call_args_list[0][1]["use_llm"] is True
			# Second call with use_llm=False
			assert self.mock_pr_generator.generate_content_from_commits.call_args_list[1][1]["use_llm"] is False

	def test_generate_pr_description_other_error(self) -> None:
		"""Test error handling for other types of errors during PR description generation."""
		# Arrange: Set up mocks
		with patch("codemap.git.pr_generator.command.loading_spinner"):
			# Configure mock to raise ValueError
			self.mock_pr_generator.generate_content_from_commits.side_effect = ValueError("Invalid data")

			# Act and Assert: Should raise RuntimeError
			with pytest.raises(RuntimeError) as excinfo:
				self.pr_command._generate_pr_description(self.branch_info, self.commits)

			# Verify error message
			assert "Failed to generate PR description: Invalid data" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.git
class TestPRCommandRun(GitTestBase):
	"""Test cases for the run method in PRCommand."""

	def setup_method(self) -> None:
		"""Set up test environment with mocks."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Patch get_repo_root to avoid actual Git operations
		self.mock_repo_path("/mock/repo/path")
		self.repo_path = "/mock/repo/path"

		# Create mock objects
		self.mock_llm_client = Mock()
		self.mock_pr_generator = Mock()

		# Create the PRCommand with patched dependencies
		# First patch the create_client in llm module
		patcher1 = patch("codemap.llm.create_client", return_value=self.mock_llm_client)
		self.mock_create_client = patcher1.start()
		self._patchers.append(patcher1)

		# Patch get_repo_root
		patcher2 = patch("codemap.git.pr_generator.command.get_repo_root", return_value=self.repo_path)
		self.mock_get_repo_root = patcher2.start()
		self._patchers.append(patcher2)

		# Then patch the PRGenerator class
		patcher3 = patch("codemap.git.pr_generator.command.PRGenerator", return_value=self.mock_pr_generator)
		self.mock_pr_generator_class = patcher3.start()
		self._patchers.append(patcher3)

		# Create the command
		self.pr_command = PRCommand()

	@patch.object(PRCommand, "_get_branch_info")
	@patch.object(PRCommand, "_get_commit_history")
	@patch.object(PRCommand, "_generate_pr_description")
	def test_run_success(
		self, mock_generate_description: Mock, mock_get_commits: Mock, mock_get_branch_info: Mock
	) -> None:
		"""Test successful execution of the run method."""
		# Arrange: Set up mocks
		branch_info = {
			"current_branch": "feature-branch",
			"target_branch": "main",
		}
		commits = [
			{"hash": "abc123", "author": "John Doe", "subject": "feat: Add new feature"},
			{"hash": "def456", "author": "Jane Smith", "subject": "fix: Fix a bug"},
		]
		description = "This PR adds a new feature and fixes a bug."

		mock_get_branch_info.return_value = branch_info
		mock_get_commits.return_value = commits
		mock_generate_description.return_value = description

		# Act: Call the run method
		result = self.pr_command.run()

		# Assert: Verify results
		assert result["branch_info"] == branch_info
		assert result["commits"] == commits
		assert result["description"] == description

		# Verify methods were called
		mock_get_branch_info.assert_called_once()
		mock_get_commits.assert_called_once_with(branch_info["target_branch"])
		mock_generate_description.assert_called_once_with(branch_info, commits)

	@patch.object(PRCommand, "_get_branch_info")
	@patch.object(PRCommand, "_get_commit_history")
	@patch.object(PRCommand, "_raise_no_commits_error")
	def test_run_no_commits(self, mock_raise_error: Mock, mock_get_commits: Mock, mock_get_branch_info: Mock) -> None:
		"""Test run method when there are no commits found."""
		# Arrange: Set up mocks
		branch_info = {
			"current_branch": "feature-branch",
			"target_branch": "main",
		}
		mock_get_branch_info.return_value = branch_info
		mock_get_commits.return_value = []  # No commits found
		mock_raise_error.side_effect = RuntimeError("No commits found between feature-branch and main")

		# Act and Assert: Should raise RuntimeError
		with pytest.raises(RuntimeError) as excinfo:
			self.pr_command.run()

		# Verify error message
		assert "No commits found between feature-branch and main" in str(excinfo.value)
		assert self.pr_command.error_state == "failed"

		# Verify methods were called
		mock_get_branch_info.assert_called_once()
		mock_get_commits.assert_called_once_with(branch_info["target_branch"])
		mock_raise_error.assert_called_once_with(branch_info)

	@patch.object(PRCommand, "_get_branch_info")
	def test_run_with_branch_info_error(self, mock_get_branch_info: Mock) -> None:
		"""Test error handling when getting branch information fails."""
		# Arrange: Set up mocks
		mock_get_branch_info.side_effect = RuntimeError("Failed to get branch information")

		# Act and Assert: Should raise RuntimeError
		with pytest.raises(RuntimeError) as excinfo:
			self.pr_command.run()

		# Verify error message
		assert "Failed to get branch information" in str(excinfo.value)
		assert self.pr_command.error_state == "failed"

	def test_raise_no_commits_error(self) -> None:
		"""Test the _raise_no_commits_error method."""
		# Arrange: Set up branch_info
		branch_info = {
			"current_branch": "feature-branch",
			"target_branch": "main",
		}

		# Set up logger mock for this test specifically
		with (
			pytest.raises(RuntimeError) as excinfo,
			patch("codemap.git.pr_generator.command.logger") as mock_logger,
		):
			# Act: Call the method directly
			self.pr_command._raise_no_commits_error(branch_info)

		# Assert: Verify error message and logger call
		assert "No commits found between feature-branch and main" in str(excinfo.value)

		# The logger is called with the formatted string directly, not with format specifiers and args
		expected_message = (
			f"No commits found between {branch_info['current_branch']} and {branch_info['target_branch']}"
		)
		mock_logger.warning.assert_called_with(expected_message)
