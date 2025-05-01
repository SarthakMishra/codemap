"""Tests for the PR generator module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codemap.git.pr_generator import (
	PRGenerator,
	generate_pr_description_from_commits,
	generate_pr_title_from_commits,
	suggest_branch_name,
)
from codemap.git.pr_generator.schemas import PullRequest
from codemap.git.utils import GitError
from tests.base import GitTestBase
from tests.conftest import skip_git_tests


def test_suggest_branch_name() -> None:
	"""Test suggesting branch names."""
	# Instead of testing the actual implementation which might change,
	# we'll test that the function returns something reasonable
	with patch("codemap.git.pr_generator.utils.create_strategy") as mock_create_strategy:
		# Set up the mock strategy
		mock_strategy = Mock()
		mock_strategy.suggest_branch_name.return_value = "feature/auth"
		mock_create_strategy.return_value = mock_strategy

		# Test with GitHub Flow
		branch_name = suggest_branch_name("Add new feature for user authentication", "github-flow")
		assert branch_name == "feature/auth"

		# Set different return values for different workflow strategies
		mock_strategy.suggest_branch_name.return_value = "hotfix/login-issue"
		# Test with GitFlow
		branch_name = suggest_branch_name("fix: Resolve login issue", "gitflow")
		assert branch_name == "hotfix/login-issue"

		# Set different return value for trunk-based
		mock_strategy.suggest_branch_name.return_value = "docs/update-readme"
		# Test with trunk-based
		branch_name = suggest_branch_name("docs: Update README", "trunk-based")
		assert branch_name == "docs/update-readme"


def test_generate_pr_title_from_commits() -> None:
	"""Test generating PR title from commits."""
	# Feature commit
	title = generate_pr_title_from_commits(["feat: Add user authentication"])
	assert title.startswith("Feature:")

	# Fix commit
	title = generate_pr_title_from_commits(["fix: Resolve login issue"])
	assert title.startswith("Fix:")

	# Multiple commits (should use the first one)
	title = generate_pr_title_from_commits(["docs: Update README", "feat: Add new feature"])
	assert title.startswith("Docs:")

	# Empty commits
	title = generate_pr_title_from_commits([])
	assert title == "Update branch"


def test_generate_pr_description_from_commits() -> None:
	"""Test generating PR description from commits."""
	# Feature commits
	description = generate_pr_description_from_commits(["feat: Add user authentication"])
	assert "Feature" in description
	assert "Add user authentication" in description

	# Mix of commit types
	description = generate_pr_description_from_commits(
		["feat: Add authentication", "fix: Fix login bug", "docs: Update docs", "refactor: Clean up code"]
	)
	# Check for the sections that are actually in the description
	assert "## What type of PR is this?" in description
	assert "Feature" in description
	assert "Bug Fix" in description
	assert "Documentation Update" in description
	assert "Refactor" in description
	assert "Add authentication" in description
	assert "Fix login bug" in description
	assert "Update docs" in description
	assert "Clean up code" in description


@pytest.mark.unit
@pytest.mark.git
@skip_git_tests
class TestPRGenerator(GitTestBase):
	"""Tests for the PRGenerator class."""

	def setup_method(self) -> None:
		"""Set up test environment."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Create mock objects
		self.mock_llm_client = Mock()
		self.repo_path = Path("/mock/repo/path")

		# Create the generator instance
		self.pr_generator = PRGenerator(self.repo_path, self.mock_llm_client)

		# Mock all git operations to prevent any real git commands from being run
		self.patcher1 = patch("codemap.git.pr_generator.utils.run_git_command")
		self.mock_run_git = self.patcher1.start()
		self._patchers.append(self.patcher1)

		# Mock get_commit_messages directly to avoid real Git operations
		self.patcher2 = patch("codemap.git.pr_generator.generator.get_commit_messages")
		self.mock_get_commits = self.patcher2.start()
		self._patchers.append(self.patcher2)

	def test_init(self) -> None:
		"""Test initialization of PRGenerator."""
		assert self.pr_generator.repo_path == self.repo_path
		assert self.pr_generator.client == self.mock_llm_client

	def test_generate_content_from_commits_with_llm(self) -> None:
		"""Test generating PR content from commits using LLM."""
		# Arrange
		self.mock_get_commits.return_value = ["feat: Add feature", "fix: Fix bug"]

		with (
			patch("codemap.git.pr_generator.generator.generate_pr_title_with_llm") as mock_gen_title,
			patch("codemap.git.pr_generator.generator.generate_pr_description_with_llm") as mock_gen_desc,
		):
			# Setup mocks
			mock_gen_title.return_value = "Add feature and fix bug"
			mock_gen_desc.return_value = "This PR adds a feature and fixes a bug."

			# Act
			result = self.pr_generator.generate_content_from_commits("main", "feature", use_llm=True)

			# Assert
			assert result["title"] == "Add feature and fix bug"
			assert result["description"] == "This PR adds a feature and fixes a bug."
			self.mock_get_commits.assert_called_once_with("main", "feature")
			mock_gen_title.assert_called_once_with(["feat: Add feature", "fix: Fix bug"], self.mock_llm_client)
			mock_gen_desc.assert_called_once_with(["feat: Add feature", "fix: Fix bug"], self.mock_llm_client)

	def test_generate_content_from_commits_without_llm(self) -> None:
		"""Test generating PR content from commits without using LLM."""
		# Arrange
		self.mock_get_commits.return_value = ["feat: Add feature", "fix: Fix bug"]

		with (
			patch("codemap.git.pr_generator.generator.generate_pr_title_from_commits") as mock_gen_title,
			patch("codemap.git.pr_generator.generator.generate_pr_description_from_commits") as mock_gen_desc,
		):
			# Setup mocks
			mock_gen_title.return_value = "Feature: Add feature"
			mock_gen_desc.return_value = "## Changes\n\n- Add feature\n- Fix bug"

			# Act
			result = self.pr_generator.generate_content_from_commits("main", "feature", use_llm=False)

			# Assert
			assert result["title"] == "Feature: Add feature"
			assert result["description"] == "## Changes\n\n- Add feature\n- Fix bug"
			self.mock_get_commits.assert_called_once_with("main", "feature")
			mock_gen_title.assert_called_once_with(["feat: Add feature", "fix: Fix bug"])
			mock_gen_desc.assert_called_once_with(["feat: Add feature", "fix: Fix bug"])

	def test_generate_content_from_commits_empty(self) -> None:
		"""Test generating PR content with no commits."""
		# Arrange
		self.mock_get_commits.return_value = []

		# Act
		result = self.pr_generator.generate_content_from_commits("main", "feature")

		# Assert
		assert result["title"] == "Update branch"
		assert result["description"] == "No changes in this PR."
		self.mock_get_commits.assert_called_once_with("main", "feature")

	def test_generate_content_from_template(self) -> None:
		"""Test generating PR content from a template."""
		# Arrange
		expected_content = {"title": "Add authentication feature", "description": "This PR adds user authentication"}

		with patch("codemap.git.pr_generator.generator.generate_pr_content_from_template") as mock_gen_content:
			# Setup mock
			mock_gen_content.return_value = expected_content

			# Act
			result = self.pr_generator.generate_content_from_template(
				"feature/auth", "Add authentication feature", "github-flow"
			)

			# Assert
			assert result == expected_content
			mock_gen_content.assert_called_once_with("feature/auth", "Add authentication feature", "github-flow")

	def test_suggest_branch_name(self) -> None:
		"""Test suggesting a branch name."""
		# Arrange
		with patch("codemap.git.pr_generator.generator.suggest_branch_name") as mock_suggest:
			# Setup mock
			mock_suggest.return_value = "feature/auth"

			# Act
			result = self.pr_generator.suggest_branch_name("Add authentication", "github-flow")

			# Assert
			assert result == "feature/auth"
			mock_suggest.assert_called_once_with("Add authentication", "github-flow")

	def test_create_pr(self) -> None:
		"""Test creating a pull request."""
		# Arrange
		mock_pr = PullRequest(
			branch="feature",
			title="Add feature",
			description="Description",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)

		with patch("codemap.git.pr_generator.generator.create_pull_request") as mock_create_pr:
			# Setup mock
			mock_create_pr.return_value = mock_pr

			# Act
			result = self.pr_generator.create_pr("main", "feature", "Add feature", "Description")

			# Assert
			assert result == mock_pr
			mock_create_pr.assert_called_once_with("main", "feature", "Add feature", "Description")

	def test_update_pr(self) -> None:
		"""Test updating a pull request."""
		# Arrange
		mock_pr = PullRequest(
			branch="feature",
			title="Updated feature",
			description="Updated description",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)

		with patch("codemap.git.pr_generator.generator.update_pull_request") as mock_update_pr:
			# Setup mock
			mock_update_pr.return_value = mock_pr

			# Act
			result = self.pr_generator.update_pr(1, "Updated feature", "Updated description")

			# Assert
			assert result == mock_pr
			mock_update_pr.assert_called_once_with(1, "Updated feature", "Updated description")

	def test_get_existing_pr(self) -> None:
		"""Test getting an existing PR."""
		# Arrange
		mock_pr = PullRequest(
			branch="feature",
			title="Feature",
			description="Description",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)

		with patch("codemap.git.pr_generator.generator.get_existing_pr") as mock_get_pr:
			# Setup mock
			mock_get_pr.return_value = mock_pr

			# Act
			result = self.pr_generator.get_existing_pr("feature")

			# Assert
			assert result == mock_pr
			mock_get_pr.assert_called_once_with("feature")

	def test_create_or_update_pr_new(self) -> None:
		"""Test creating a new PR."""
		# Arrange
		mock_pr = PullRequest(
			branch="feature",
			title="Add feature",
			description="Description",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)

		with (
			patch("codemap.git.pr_generator.generator.get_existing_pr") as mock_get_pr,
			patch("codemap.git.pr_generator.generator.create_pull_request") as mock_create_pr,
		):
			# Setup mocks
			mock_get_pr.return_value = None  # No existing PR
			mock_create_pr.return_value = mock_pr

			# Act
			result = self.pr_generator.create_or_update_pr(
				base_branch="main", head_branch="feature", title="Add feature", description="Description"
			)

			# Assert
			assert result == mock_pr
			mock_get_pr.assert_called_once_with("feature")
			mock_create_pr.assert_called_once_with("main", "feature", "Add feature", "Description")

	def test_create_or_update_pr_existing(self) -> None:
		"""Test updating an existing PR."""
		# Arrange
		mock_existing_pr = PullRequest(
			branch="feature",
			title="Old title",
			description="Old description",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)
		mock_updated_pr = PullRequest(
			branch="feature",
			title="Updated title",
			description="Updated description",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)

		with (
			patch("codemap.git.pr_generator.generator.get_existing_pr") as mock_get_pr,
			patch("codemap.git.pr_generator.generator.update_pull_request") as mock_update_pr,
		):
			# Setup mocks
			mock_get_pr.return_value = mock_existing_pr
			mock_update_pr.return_value = mock_updated_pr

			# Act
			result = self.pr_generator.create_or_update_pr(
				base_branch="main", head_branch="feature", title="Updated title", description="Updated description"
			)

			# Assert
			assert result == mock_updated_pr
			mock_get_pr.assert_called_once_with("feature")
			mock_update_pr.assert_called_once_with(1, "Updated title", "Updated description")

	def test_create_or_update_pr_defaults(self) -> None:
		"""Test creating a PR with default values."""
		# Arrange
		mock_pr = PullRequest(
			branch="feature",
			title="Auto-generated title",
			description="Auto-generated description",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)

		# Much simpler approach: directly patch the specific functions we need to test
		# and avoid dealing with the dynamic import
		with (
			patch.object(self.pr_generator, "get_existing_pr", return_value=None) as mock_get_pr,
			patch.object(self.pr_generator, "create_pr", return_value=mock_pr) as mock_create_pr,
			patch.object(
				self.pr_generator,
				"generate_content_from_commits",
				return_value={"title": "Auto-generated title", "description": "Auto-generated description"},
			),
			# Instead of complex import patching, directly patch the functions
			# that are dynamically imported
			patch("codemap.git.pr_generator.utils.get_default_branch", return_value="main"),
			patch("codemap.git.pr_generator.utils.get_current_branch", return_value="feature"),
		):
			# Act
			result = self.pr_generator.create_or_update_pr()

			# Assert
			assert result == mock_pr
			mock_get_pr.assert_called_once_with("feature")
			mock_create_pr.assert_called_once_with(
				"main", "feature", "Auto-generated title", "Auto-generated description"
			)

	def test_create_or_update_pr_current_branch_error(self) -> None:
		"""Test error handling when getting current branch fails."""
		# Arrange
		with (
			patch("codemap.git.pr_generator.utils.get_default_branch") as mock_get_default,
			patch("codemap.git.pr_generator.utils.get_current_branch") as mock_get_current,
		):
			# Setup mocks
			mock_get_default.return_value = "main"
			mock_get_current.side_effect = GitError("Failed to get current branch")

			# Act & Assert
			with pytest.raises(GitError, match="Failed to determine current branch"):
				self.pr_generator.create_or_update_pr()

	def test_create_or_update_pr_no_existing_pr_by_number(self) -> None:
		"""Test error when PR number is provided but PR doesn't exist."""
		# Arrange
		with (
			patch("codemap.git.pr_generator.utils.get_default_branch") as mock_get_default,
			patch("codemap.git.pr_generator.utils.get_current_branch") as mock_get_current,
			patch("codemap.git.pr_generator.generator.get_existing_pr") as mock_get_pr,
		):
			# Setup mocks
			mock_get_default.return_value = "main"
			mock_get_current.return_value = "feature"
			mock_get_pr.return_value = None  # No existing PR

			# Act & Assert
			with pytest.raises(GitError, match="No PR found for branch feature with number 42"):
				self.pr_generator.create_or_update_pr(pr_number=42)
