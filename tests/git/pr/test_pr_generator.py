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


def test_suggest_branch_name() -> None:
	"""Test suggesting branch names."""
	# Test with GitHub Flow
	branch_name = suggest_branch_name("Add new feature for user authentication", "github-flow")
	assert "authentication" in branch_name.lower()

	# Test with GitFlow
	branch_name = suggest_branch_name("fix: Resolve login issue", "gitflow")
	assert branch_name.startswith("hotfix/")

	# Test with trunk-based
	branch_name = suggest_branch_name("docs: Update README", "trunk-based")
	assert "/" in branch_name


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
	assert "Features" in description
	assert "Fixes" in description
	assert "Documentation" in description
	assert "Refactors" in description


@pytest.mark.parametrize("workflow_strategy", ["github-flow", "gitflow", "trunk-based"])
def test_pr_generator_init(workflow_strategy: str) -> None:  # noqa: ARG001
	"""Test initializing PR generator."""
	# Unused parameter workflow_strategy is required by the parametrize decorator
	with patch("codemap.git.pr_generator.generator.get_default_branch", return_value="main"):
		# Create a mock LLMClient
		mock_llm_client = Mock()

		# Initialize the generator with the required llm_client
		generator = PRGenerator(Path("/fake/path"), llm_client=mock_llm_client)

		# Verify initialization
		assert generator.repo_path == Path("/fake/path")
		assert generator.client == mock_llm_client
