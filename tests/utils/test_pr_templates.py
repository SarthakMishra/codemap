"""Tests for PR template functionality."""

from __future__ import annotations

from codemap.utils.pr_strategies import (
	GitFlowStrategy,
	GitHubFlowStrategy,
	TrunkBasedStrategy,
)
from codemap.utils.pr_utils import generate_pr_content_from_template


class TestPRTemplates:
	"""Tests for PR template functionality."""

	def test_base_strategy_templates(self) -> None:
		"""Test that all strategy implementations provide templates."""
		github_flow = GitHubFlowStrategy()
		gitflow = GitFlowStrategy()
		trunk_based = TrunkBasedStrategy()

		# Test that all strategies have implemented the get_pr_templates method
		for strategy in [github_flow, gitflow, trunk_based]:
			for branch_type in strategy.get_branch_types():
				templates = strategy.get_pr_templates(branch_type)
				assert "title" in templates
				assert "description" in templates
				assert isinstance(templates["title"], str)
				assert isinstance(templates["description"], str)

	def test_github_flow_templates(self) -> None:
		"""Test that GitHub Flow strategy returns appropriate templates."""
		strategy = GitHubFlowStrategy()
		templates = strategy.get_pr_templates("feature")

		assert "{description}" in templates["title"]
		assert "## Description" in templates["description"]
		assert "## What does this PR do?" in templates["description"]
		assert "## Testing completed" in templates["description"]

	def test_gitflow_strategy_templates(self) -> None:
		"""Test that GitFlow strategy returns appropriate templates for different branch types."""
		strategy = GitFlowStrategy()

		# Test feature branch templates
		feature_templates = strategy.get_pr_templates("feature")
		assert "Feature: {description}" in feature_templates["title"]
		assert "## Feature Description" in feature_templates["description"]

		# Test release branch templates
		release_templates = strategy.get_pr_templates("release")
		assert "Release {description}" in release_templates["title"]
		assert "### Features" in release_templates["description"]
		assert "### Bug Fixes" in release_templates["description"]

		# Test hotfix branch templates
		hotfix_templates = strategy.get_pr_templates("hotfix")
		assert "Hotfix: {description}" in hotfix_templates["title"]
		assert "### Issue Description" in hotfix_templates["description"]

		# Test bugfix branch templates
		bugfix_templates = strategy.get_pr_templates("bugfix")
		assert "Fix: {description}" in bugfix_templates["title"]
		assert "## Bug Fix" in bugfix_templates["description"]

	def test_trunk_based_strategy_templates(self) -> None:
		"""Test that Trunk-Based strategy returns appropriate templates."""
		strategy = TrunkBasedStrategy()
		templates = strategy.get_pr_templates("feature")

		assert "{description}" in templates["title"]
		assert "## Change Description" in templates["description"]
		assert "## Implementation" in templates["description"]
		assert "## Test Plan" in templates["description"]
		assert "## Rollout Plan" in templates["description"]

	def test_generate_pr_content_from_template(self) -> None:
		"""Test generating PR content from templates."""
		# Test with GitHub Flow
		github_flow_content = generate_pr_content_from_template(
			branch_name="new-feature", description="Add user authentication", strategy_name="github-flow"
		)

		assert github_flow_content["title"] == "Add user authentication"
		assert "Add user authentication" in github_flow_content["description"]

		# Test with GitFlow for a feature branch
		gitflow_content = generate_pr_content_from_template(
			branch_name="feature/user-auth", description="Add user authentication", strategy_name="gitflow"
		)

		assert gitflow_content["title"] == "Feature: Add user authentication"
		assert "## Feature Description" in gitflow_content["description"]
		assert "Add user authentication" in gitflow_content["description"]

		# Test with GitFlow for a hotfix branch
		gitflow_hotfix_content = generate_pr_content_from_template(
			branch_name="hotfix/login-issue", description="Fix login regression", strategy_name="gitflow"
		)

		assert gitflow_hotfix_content["title"] == "Hotfix: Fix login regression"
		assert "### Issue Description" in gitflow_hotfix_content["description"]
		assert "Fix login regression" in gitflow_hotfix_content["description"]

		# Test with Trunk-Based
		trunk_based_content = generate_pr_content_from_template(
			branch_name="username/auth-fix", description="Fix authentication issues", strategy_name="trunk-based"
		)

		assert trunk_based_content["title"] == "Fix authentication issues"
		assert "## Change Description" in trunk_based_content["description"]
		assert "Fix authentication issues" in trunk_based_content["description"]
