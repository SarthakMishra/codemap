"""Tests for PR workflow strategies."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codemap.utils.pr_strategies import (
	GitFlowStrategy,
	GitHubFlowStrategy,
	TrunkBasedStrategy,
	WorkflowStrategy,
	create_strategy,
	get_strategy_class,
)


class TestWorkflowStrategyBase:
	"""Tests for the WorkflowStrategy base class."""

	def test_suggest_branch_name_base(self) -> None:
		"""Test the base implementation of suggest_branch_name."""

		class TestStrategy(WorkflowStrategy):
			"""Test implementation of WorkflowStrategy."""

			def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
				return "main"

			def get_branch_prefix(self, branch_type: str) -> str:  # noqa: ARG002
				return "test-"

			def get_branch_types(self) -> list[str]:
				return ["test"]

		strategy = TestStrategy()
		result = strategy.suggest_branch_name("test", "This is a test description")

		assert result == "test-this-is-a-test-description"

	def test_detect_branch_type(self) -> None:
		"""Test detect_branch_type method."""

		class TestStrategy(WorkflowStrategy):
			"""Test implementation of WorkflowStrategy."""

			def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
				return "main"

			def get_branch_prefix(self, branch_type: str) -> str:
				return "test-" if branch_type == "test" else "other-"

			def get_branch_types(self) -> list[str]:
				return ["test", "other"]

		strategy = TestStrategy()

		# Test with matching prefix
		assert strategy.detect_branch_type("test-branch") == "test"
		assert strategy.detect_branch_type("other-branch") == "other"

		# Test with non-matching prefix
		assert strategy.detect_branch_type("feature/branch") is None

	def test_get_pr_templates_base(self) -> None:
		"""Test the base implementation of get_pr_templates."""

		class TestStrategy(WorkflowStrategy):
			"""Test implementation of WorkflowStrategy."""

			def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
				return "main"

			def get_branch_prefix(self, branch_type: str) -> str:  # noqa: ARG002
				return "test-"

			def get_branch_types(self) -> list[str]:
				return ["test"]

		strategy = TestStrategy()
		templates = strategy.get_pr_templates("test")

		assert "title" in templates
		assert "description" in templates
		assert "{branch_type}" in templates["title"]
		assert "{description}" in templates["title"]
		assert "{description}" in templates["description"]

	def test_get_remote_branches(self) -> None:
		"""Test get_remote_branches method."""
		with patch("codemap.utils.pr_strategies.run_git_command") as mock_run_git_command:
			# Set up the return value that matches the expected format
			mock_run_git_command.return_value = "  origin/main\n  origin/feature/branch\n  origin/HEAD"

			class TestStrategy(WorkflowStrategy):
				"""Test implementation of WorkflowStrategy."""

				def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
					return "main"

				def get_branch_prefix(self, branch_type: str) -> str:  # noqa: ARG002
					return "test-"

				def get_branch_types(self) -> list[str]:
					return ["test"]

			strategy = TestStrategy()
			branches = strategy.get_remote_branches()

			assert "main" in branches
			assert "feature/branch" in branches
			assert "HEAD" not in branches  # Should be cleaned up

	def test_get_local_branches(self) -> None:
		"""Test get_local_branches method."""
		with patch("codemap.utils.pr_strategies.run_git_command") as mock_run_git_command:
			mock_run_git_command.return_value = "  main\n* feature/branch\n  release/1.0.0"

			class TestStrategy(WorkflowStrategy):
				"""Test implementation of WorkflowStrategy."""

				def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
					return "main"

				def get_branch_prefix(self, branch_type: str) -> str:  # noqa: ARG002
					return "test-"

				def get_branch_types(self) -> list[str]:
					return ["test"]

			strategy = TestStrategy()
			branches = strategy.get_local_branches()

			assert "main" in branches
			assert "feature/branch" in branches
			assert "release/1.0.0" in branches
			# Current branch should not have * prefix
			assert all("*" not in branch for branch in branches)

	def test_get_branches_by_type(self) -> None:
		"""Test get_branches_by_type method."""

		class TestStrategy(WorkflowStrategy):
			"""Test implementation of WorkflowStrategy."""

			def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
				return "main"

			def get_branch_prefix(self, branch_type: str) -> str:
				return f"{branch_type}/"

			def get_branch_types(self) -> list[str]:
				return ["feature", "release"]

			def get_local_branches(self) -> list[str]:
				return ["main", "feature/one", "release/1.0.0", "unknown"]

			def get_remote_branches(self) -> list[str]:
				return ["main", "feature/two", "hotfix/bug"]

		strategy = TestStrategy()
		branches_by_type = strategy.get_branches_by_type()

		assert set(branches_by_type.keys()) == {"feature", "release", "other"}
		assert set(branches_by_type["feature"]) == {"feature/one", "feature/two"}
		assert set(branches_by_type["release"]) == {"release/1.0.0"}
		assert set(branches_by_type["other"]) == {"main", "unknown", "hotfix/bug"}

	def test_get_branch_metadata(self) -> None:
		"""Test get_branch_metadata method."""
		with (
			patch("codemap.utils.pr_strategies.run_git_command") as mock_run_git_command,
			patch("codemap.utils.pr_strategies._branch_exists") as mock_branch_exists,
			patch("codemap.utils.pr_strategies._get_default_branch") as mock_get_default_branch,
		):
			mock_branch_exists.return_value = True
			mock_get_default_branch.return_value = "main"
			mock_run_git_command.side_effect = [
				"2 days ago",  # Last commit date
				"5",  # Commit count
			]

			class TestStrategy(WorkflowStrategy):
				"""Test implementation of WorkflowStrategy."""

				def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
					return "main"

				def get_branch_prefix(self, branch_type: str) -> str:  # noqa: ARG002
					return "feature/"

				def get_branch_types(self) -> list[str]:
					return ["feature"]

				def get_local_branches(self) -> list[str]:
					return ["main", "feature/test"]

				def get_remote_branches(self) -> list[str]:
					return ["main", "feature/test"]

				def detect_branch_type(self, branch_name: str) -> str | None:
					return "feature" if branch_name.startswith("feature/") else None

			strategy = TestStrategy()
			metadata = strategy.get_branch_metadata("feature/test")

			assert metadata["last_commit_date"] == "2 days ago"
			assert metadata["commit_count"] == "5"
			assert metadata["branch_type"] == "feature"
			assert metadata["is_local"] is True
			assert metadata["is_remote"] is True

	def test_get_all_branches_with_metadata(self) -> None:
		"""Test get_all_branches_with_metadata method."""

		class TestStrategy(WorkflowStrategy):
			"""Test implementation of WorkflowStrategy."""

			def get_default_base(self, branch_type: str) -> str:  # noqa: ARG002
				return "main"

			def get_branch_prefix(self, branch_type: str) -> str:  # noqa: ARG002
				return "feature/"

			def get_branch_types(self) -> list[str]:
				return ["feature"]

			def get_local_branches(self) -> list[str]:
				return ["main", "feature/test"]

			def get_remote_branches(self) -> list[str]:
				return ["main", "feature/other"]

			def get_branch_metadata(self, branch_name: str) -> dict:
				return {
					"last_commit_date": "2 days ago",
					"commit_count": "5",
					"branch_type": "feature" if branch_name.startswith("feature/") else None,
					"is_local": branch_name in self.get_local_branches(),
					"is_remote": branch_name in self.get_remote_branches(),
				}

		strategy = TestStrategy()
		all_branches_with_metadata = strategy.get_all_branches_with_metadata()

		assert set(all_branches_with_metadata.keys()) == {"main", "feature/test", "feature/other"}
		assert all_branches_with_metadata["main"]["branch_type"] is None
		assert all_branches_with_metadata["feature/test"]["branch_type"] == "feature"
		assert all_branches_with_metadata["feature/test"]["is_local"] is True
		assert all_branches_with_metadata["feature/other"]["is_remote"] is True


class TestGitHubFlowStrategy:
	"""Tests for the GitHubFlowStrategy class."""

	def test_get_default_base(self) -> None:
		"""Test get_default_base method."""
		with patch("codemap.utils.pr_strategies._get_default_branch") as mock_get_default_branch:
			mock_get_default_branch.return_value = "main"

			strategy = GitHubFlowStrategy()
			result = strategy.get_default_base("feature")

			assert result == "main"
			mock_get_default_branch.assert_called_once()

	def test_get_branch_prefix(self) -> None:
		"""Test get_branch_prefix method."""
		strategy = GitHubFlowStrategy()
		result = strategy.get_branch_prefix("feature")

		assert result == ""  # GitHub Flow doesn't use prefixes

	def test_get_branch_types(self) -> None:
		"""Test get_branch_types method."""
		strategy = GitHubFlowStrategy()
		result = strategy.get_branch_types()

		assert result == ["feature"]

	def test_get_pr_templates(self) -> None:
		"""Test get_pr_templates method."""
		strategy = GitHubFlowStrategy()
		templates = strategy.get_pr_templates("feature")

		assert "title" in templates
		assert "description" in templates
		assert "{description}" in templates["title"]
		assert "## Description" in templates["description"]
		assert "## Changes" in templates["description"]


class TestGitFlowStrategy:
	"""Tests for the GitFlowStrategy class."""

	def test_get_default_base(self) -> None:
		"""Test get_default_base method for different branch types."""
		strategy = GitFlowStrategy()

		assert strategy.get_default_base("feature") == "develop"
		assert strategy.get_default_base("release") == "main"
		assert strategy.get_default_base("hotfix") == "main"
		assert strategy.get_default_base("bugfix") == "develop"

	def test_get_branch_prefix(self) -> None:
		"""Test get_branch_prefix method for different branch types."""
		strategy = GitFlowStrategy()

		assert strategy.get_branch_prefix("feature") == "feature/"
		assert strategy.get_branch_prefix("release") == "release/"
		assert strategy.get_branch_prefix("hotfix") == "hotfix/"
		assert strategy.get_branch_prefix("bugfix") == "bugfix/"
		assert strategy.get_branch_prefix("unknown") == ""  # Default

	def test_get_branch_types(self) -> None:
		"""Test get_branch_types method."""
		strategy = GitFlowStrategy()
		result = strategy.get_branch_types()

		assert set(result) == {"feature", "release", "hotfix", "bugfix"}

	def test_suggest_branch_name_release(self) -> None:
		"""Test suggest_branch_name method for release branches."""
		strategy = GitFlowStrategy()

		# With version number
		result1 = strategy.suggest_branch_name("release", "Release version 1.2.3 with new features")
		assert result1 == "release/1.2.3"

		# Without version number
		result2 = strategy.suggest_branch_name("release", "January release")
		assert result2 == "release/january-release"

	def test_suggest_branch_name_other(self) -> None:
		"""Test suggest_branch_name method for non-release branches."""
		strategy = GitFlowStrategy()

		result = strategy.suggest_branch_name("feature", "Add new search functionality")
		assert result == "feature/add-new-search-functionality"

	def test_get_pr_templates_different_types(self) -> None:
		"""Test get_pr_templates method for different branch types."""
		strategy = GitFlowStrategy()

		feature_templates = strategy.get_pr_templates("feature")
		assert "Feature:" in feature_templates["title"]
		assert "## Feature Description" in feature_templates["description"]

		release_templates = strategy.get_pr_templates("release")
		assert "Release" in release_templates["title"]
		assert "## Release" in release_templates["description"]
		assert "### Features" in release_templates["description"]

		hotfix_templates = strategy.get_pr_templates("hotfix")
		assert "Hotfix:" in hotfix_templates["title"]
		assert "## Hotfix:" in hotfix_templates["description"]

		bugfix_templates = strategy.get_pr_templates("bugfix")
		assert "Fix:" in bugfix_templates["title"]
		assert "## Bug Fix" in bugfix_templates["description"]


class TestTrunkBasedStrategy:
	"""Tests for the TrunkBasedStrategy class."""

	def test_get_default_base(self) -> None:
		"""Test get_default_base method."""
		with patch("codemap.utils.pr_strategies._get_default_branch") as mock_get_default_branch:
			mock_get_default_branch.return_value = "main"

			strategy = TrunkBasedStrategy()
			result = strategy.get_default_base("feature")

			assert result == "main"
			mock_get_default_branch.assert_called_once()

	def test_get_branch_prefix(self) -> None:
		"""Test get_branch_prefix method."""
		strategy = TrunkBasedStrategy()

		assert strategy.get_branch_prefix("feature") == "fb/"
		assert strategy.get_branch_prefix("other") == ""

	def test_get_branch_types(self) -> None:
		"""Test get_branch_types method."""
		strategy = TrunkBasedStrategy()
		result = strategy.get_branch_types()

		assert result == ["feature"]

	def test_suggest_branch_name_with_username(self) -> None:
		"""Test suggest_branch_name method with username."""
		with patch("codemap.utils.pr_strategies.run_git_command") as mock_run_git_command:
			mock_run_git_command.return_value = "John Doe"

			strategy = TrunkBasedStrategy()
			result = strategy.suggest_branch_name("feature", "Implement search functionality")

			assert result.startswith("john/")
			assert "search" in result
			assert "functionality" in result
			# Should exclude short words and common words like "implement"
			assert "implement" not in result

	def test_suggest_branch_name_without_username(self) -> None:
		"""Test suggest_branch_name method when username is not available."""
		with patch("codemap.utils.pr_strategies.run_git_command") as mock_run_git_command:
			from codemap.utils.git_utils import GitError

			mock_run_git_command.side_effect = GitError("Command failed")

			strategy = TrunkBasedStrategy()
			result = strategy.suggest_branch_name("feature", "Implement search functionality")

			assert result.startswith("fb/")
			assert "search" in result
			assert "functionality" in result

	def test_get_pr_templates(self) -> None:
		"""Test get_pr_templates method."""
		strategy = TrunkBasedStrategy()
		templates = strategy.get_pr_templates("feature")

		assert "title" in templates
		assert "description" in templates
		assert "{description}" in templates["title"]
		assert "## Change Description" in templates["description"]
		assert "## Implementation" in templates["description"]
		assert "## Test Plan" in templates["description"]
		assert "## Rollout Plan" in templates["description"]


class TestUtilityFunctions:
	"""Tests for utility functions in the PR strategies module."""

	def test_get_strategy_class(self) -> None:
		"""Test get_strategy_class function."""
		assert get_strategy_class("github-flow") == GitHubFlowStrategy
		assert get_strategy_class("gitflow") == GitFlowStrategy
		assert get_strategy_class("trunk-based") == TrunkBasedStrategy
		assert get_strategy_class("unknown-strategy") is None

	def test_create_strategy(self) -> None:
		"""Test create_strategy function."""
		github_flow = create_strategy("github-flow")
		assert isinstance(github_flow, GitHubFlowStrategy)

		gitflow = create_strategy("gitflow")
		assert isinstance(gitflow, GitFlowStrategy)

		trunk_based = create_strategy("trunk-based")
		assert isinstance(trunk_based, TrunkBasedStrategy)

		with pytest.raises(ValueError, match="Unknown workflow strategy: unknown-strategy"):
			create_strategy("unknown-strategy")  # type: ignore[arg-type]
