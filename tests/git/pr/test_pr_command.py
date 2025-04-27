"""Tests for the PR command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
import typer

from codemap.git.pr_generator.schemas import PullRequest
from codemap.git.utils import GitDiff

if TYPE_CHECKING:
	from collections.abc import Generator

# Get app from the CLI module
from codemap import cli_app

app = cli_app.app


@pytest.fixture
def mock_branch_operations() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock git branch operations."""
	with (
		patch("codemap.git.pr_generator.utils.get_current_branch") as mock_get_current_branch,
		patch("codemap.git.pr_generator.utils.get_default_branch") as mock_get_default_branch,
		patch("codemap.git.pr_generator.utils.branch_exists") as mock_branch_exists,
		patch("codemap.git.pr_generator.utils.create_branch") as mock_create_branch,
		patch("codemap.git.pr_generator.utils.checkout_branch") as mock_checkout_branch,
		patch("codemap.git.pr_generator.utils.push_branch") as mock_push_branch,
		patch("codemap.git.pr_generator.utils.list_branches") as mock_list_branches,
	):
		mock_get_current_branch.return_value = "feature-branch"
		mock_get_default_branch.return_value = "main"
		mock_branch_exists.return_value = False
		mock_list_branches.return_value = ["main", "dev", "feature-branch"]

		yield {
			"get_current_branch": mock_get_current_branch,
			"get_default_branch": mock_get_default_branch,
			"branch_exists": mock_branch_exists,
			"create_branch": mock_create_branch,
			"checkout_branch": mock_checkout_branch,
			"push_branch": mock_push_branch,
			"list_branches": mock_list_branches,
		}


@pytest.fixture
def mock_pr_operations() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock PR operations."""
	with (
		patch("codemap.git.pr_generator.utils.get_commit_messages") as mock_get_commit_messages,
		patch("codemap.git.pr_generator.utils.create_pull_request") as mock_create_pull_request,
		patch("codemap.git.pr_generator.utils.get_existing_pr") as mock_get_existing_pr,
		patch("codemap.git.pr_generator.utils.update_pull_request") as mock_update_pull_request,
	):
		# Mock commit messages
		mock_get_commit_messages.return_value = ["feat: Add new feature", "fix: Fix bug"]

		# Mock PR creation
		mock_pr = PullRequest(
			branch="feature-branch",
			title="Add new feature",
			description="## Changes\n\n### Features\n\n- Add new feature\n\n### Fixes\n\n- Fix bug\n\n",
			url="https://github.com/user/repo/pull/1",
			number=1,
		)
		mock_create_pull_request.return_value = mock_pr
		mock_update_pull_request.return_value = mock_pr
		mock_get_existing_pr.return_value = None

		yield {
			"get_commit_messages": mock_get_commit_messages,
			"create_pull_request": mock_create_pull_request,
			"get_existing_pr": mock_get_existing_pr,
			"update_pull_request": mock_update_pull_request,
		}


@pytest.fixture
def mock_git_diff_operations() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock git diff operations."""
	with (
		patch("codemap.git.utils.validate_repo_path") as mock_validate_repo_path,
		patch("codemap.git.utils.get_staged_diff") as mock_get_staged_diff,
		patch("codemap.git.utils.get_unstaged_diff") as mock_get_unstaged_diff,
		patch("codemap.git.utils.get_untracked_files") as mock_get_untracked_files,
	):
		mock_validate_repo_path.return_value = Path("/fake/repo")

		# Mock git utilities
		mock_staged_diff = GitDiff(
			files=["file1.py"],
			content="diff content for file1.py",
			is_staged=True,
		)
		mock_get_staged_diff.return_value = mock_staged_diff

		mock_unstaged_diff = GitDiff(
			files=["file2.py"],
			content="diff content for file2.py",
			is_staged=False,
		)
		mock_get_unstaged_diff.return_value = mock_unstaged_diff

		mock_get_untracked_files.return_value = ["file3.py"]

		yield {
			"validate_repo_path": mock_validate_repo_path,
			"get_staged_diff": mock_get_staged_diff,
			"get_unstaged_diff": mock_get_unstaged_diff,
			"get_untracked_files": mock_get_untracked_files,
		}


@pytest.fixture
def mock_diff_processing() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock diff processing operations."""
	with (
		patch("codemap.git.diff_splitter.DiffSplitter") as mock_diff_splitter,
		patch("codemap.cli.pr_cmd.create_universal_generator") as mock_create_universal_generator,
		patch("codemap.git.commit_generator.command.CommitCommand.process_all_chunks") as mock_process_all_chunks,
	):
		# Mock DiffSplitter
		mock_splitter = MagicMock()
		mock_chunk = MagicMock()
		mock_chunk.files = ["file1.py"]
		mock_splitter.split_diff.return_value = [mock_chunk]
		mock_diff_splitter.return_value = mock_splitter

		# Mock message generator
		mock_generator = MagicMock()
		mock_generator.generate_message.return_value = ("feat: Add new feature", True)
		mock_create_universal_generator.return_value = mock_generator

		# Mock process_all_chunks
		mock_process_all_chunks.return_value = 0

		yield {
			"diff_splitter": mock_diff_splitter,
			"create_universal_generator": mock_create_universal_generator,
			"process_all_chunks": mock_process_all_chunks,
		}


@pytest.fixture
def mock_user_input() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock user input operations."""
	with (
		patch("questionary.confirm") as mock_confirm,
		patch("questionary.text") as mock_text,
		patch("questionary.select") as mock_select,
	):
		# Mock questionary
		mock_confirm.return_value.ask.return_value = True
		mock_text.return_value.ask.return_value = "feature-branch"
		mock_select.return_value.ask.return_value = "commit"

		yield {
			"confirm": mock_confirm,
			"text": mock_text,
			"select": mock_select,
		}


@pytest.fixture
def mock_llm_config() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock LLM configuration loading."""
	with patch("codemap.cli.pr_cmd._load_llm_config") as mock_load_config:
		# Return a default configuration for testing
		mock_load_config.return_value = {
			"model": "gpt-3.5-turbo",
			"api_key": "test-api-key",
			"api_base": "https://api.example.com",
		}

		yield {
			"load_llm_config": mock_load_config,
		}


@pytest.fixture
def mock_subprocess() -> Generator[dict[str, MagicMock], None, None]:
	"""Mock subprocess run calls."""
	with patch("subprocess.run") as mock_run:
		# Create a mock result for GitHub CLI commands
		mock_result = MagicMock()
		mock_result.returncode = 0
		mock_result.stdout = json.dumps(
			{
				"number": 42,
				"title": "Test PR",
				"body": "Test description",
				"headRefName": "feature-branch",
				"url": "https://github.com/user/repo/pull/42",
			}
		)

		# Set the return value of the mock
		mock_run.return_value = mock_result

		yield {"subprocess_run": mock_run}


@pytest.fixture
def mock_git_utils(
	mock_branch_operations: dict[str, MagicMock],
	mock_pr_operations: dict[str, MagicMock],
	mock_git_diff_operations: dict[str, MagicMock],
	mock_diff_processing: dict[str, MagicMock],
	mock_user_input: dict[str, MagicMock],
	mock_llm_config: dict[str, MagicMock],
	mock_subprocess: dict[str, MagicMock],
) -> dict[str, Any]:
	"""Combine all mock fixtures into one dictionary for convenience."""
	return {
		**mock_branch_operations,
		**mock_pr_operations,
		**mock_git_diff_operations,
		**mock_diff_processing,
		**mock_user_input,
		**mock_llm_config,
		**mock_subprocess,
	}


@pytest.fixture
def mock_exit_with_error() -> Generator[MagicMock, None, None]:
	"""Mock the _exit_with_error function."""
	with patch("codemap.cli.pr_cmd._exit_with_error") as mock_exit:
		# Instead of exiting, just capture the error message for debugging
		mock_exit.side_effect = lambda _msg, *_args, **_kwargs: None
		yield mock_exit


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.cli
@pytest.mark.skip(reason="PR command integration test needs deeper mocking strategy")
def test_pr_create_command(mock_git_utils: dict[str, Any], mock_exit_with_error: MagicMock) -> None:
	"""Test the PR create command."""
	# Test skipped - requires deeper mocking strategy


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.cli
@pytest.mark.skip(reason="PR command integration test needs deeper mocking strategy")
def test_pr_update_command(mock_git_utils: dict[str, Any], mock_exit_with_error: MagicMock) -> None:
	"""Test the PR update command."""
	# Test skipped - requires deeper mocking strategy


@pytest.mark.unit
@pytest.mark.git
class TestPRCommandHelpers:
	"""Test helper functions in PR command module."""

	def test_validate_branch_name_valid(self) -> None:
		"""Test branch name validation with valid inputs."""
		from codemap.cli.pr_cmd import _validate_branch_name

		valid_names = ["feature-branch", "feature_branch", "hotfix123", "bugfix.security"]
		for name in valid_names:
			assert _validate_branch_name(name) is True

	def test_validate_branch_name_invalid(self) -> None:
		"""Test branch name validation with invalid inputs."""
		from codemap.cli.pr_cmd import _validate_branch_name

		with patch("codemap.cli.pr_cmd.console.print") as mock_print:
			invalid_names = ["", "feature/branch", "release:1.0.0", "hot&fix"]
			for name in invalid_names:
				assert _validate_branch_name(name) is False
				mock_print.assert_called_with(
					"[red]Invalid branch name. Use only letters, numbers, underscores, dots, and hyphens.[/red]"
				)

	def test_exit_with_error(self) -> None:
		"""Test exit with error function."""
		from codemap.cli.pr_cmd import _exit_with_error

		with patch("codemap.cli.pr_cmd.console.print") as mock_print:
			with pytest.raises(typer.Exit) as excinfo:
				_exit_with_error("Test error message", 123)

			mock_print.assert_called_once_with("Test error message")
			assert excinfo.value.exit_code == 123

	def test_exit_with_error_exception(self) -> None:
		"""Test exit with error function with exception."""
		from codemap.cli.pr_cmd import _exit_with_error

		test_exception = ValueError("Test error")
		with patch("codemap.cli.pr_cmd.console.print") as mock_print:
			with pytest.raises(typer.Exit) as excinfo:
				_exit_with_error("Error message", 2, test_exception)

			mock_print.assert_called_once_with("Error message")
			assert excinfo.value.exit_code == 2
			assert excinfo.value.__cause__ == test_exception

	def test_generate_release_pr_content(self) -> None:
		"""Test generation of release PR content."""
		from codemap.cli.pr_cmd import generate_release_pr_content

		result = generate_release_pr_content("main", "release/1.0.0")
		assert result["title"] == "Release 1.0.0"
		assert "Release 1.0.0" in result["description"]
		assert "main" in result["description"]


@pytest.mark.unit
@pytest.mark.git
class TestHandleBranchCreation:
	"""Test branch creation and selection handling."""

	def test_handle_branch_creation_existing_branch(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test handling branch creation when user wants to use current branch."""
		from codemap.cli.pr_cmd import PROptions, _handle_branch_creation

		# Configure mocks
		mock_git_utils["get_current_branch"].return_value = "feature-branch"
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["confirm"].return_value.ask.return_value = True

		# Create options with interactive mode
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Call the function
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.get_current_branch", return_value="feature-branch"),
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"
			mock_confirm.return_value.ask.return_value = True

			# Call the function under test
			result = _handle_branch_creation(options)

		# Verify result
		assert result == "feature-branch"

	def test_handle_branch_creation_specified_branch(self) -> None:
		"""Test handling branch creation with specified branch name."""
		from codemap.cli.pr_cmd import PROptions, _handle_branch_creation

		# Create options with branch name
		options = PROptions(
			repo_path=Path("/fake/repo"),
			branch_name="specified-branch",
			interactive=False,
		)

		# Call the function
		with patch("codemap.cli.pr_cmd._validate_branch_name", return_value=True):
			result = _handle_branch_creation(options)

		# Verify result
		assert result == "specified-branch"

	def test_handle_branch_creation_invalid_branch(self) -> None:
		"""Test handling branch creation with invalid branch name."""
		from codemap.cli.pr_cmd import PROptions, _handle_branch_creation

		# Create options with invalid branch name
		options = PROptions(
			repo_path=Path("/fake/repo"),
			branch_name="invalid$branch",
			interactive=False,
		)

		# Call the function
		with patch("codemap.cli.pr_cmd._validate_branch_name", return_value=False):
			result = _handle_branch_creation(options)

		# Verify result
		assert result is None


@pytest.mark.unit
@pytest.mark.git
class TestGenerateTitleAndDescription:
	"""Test generation of PR title and description."""

	def test_generate_title_from_commits(self) -> None:
		"""Test generating PR title from commits."""
		from codemap.cli.pr_cmd import PROptions, _generate_title

		# Setup
		options = PROptions(
			repo_path=Path("/fake/repo"),
			model="test-model",
		)
		commits = ["feat: Add feature A", "feat: Add feature B"]

		# Test with commits strategy
		with patch(
			"codemap.cli.pr_cmd.generate_pr_title_from_commits", return_value="Features: Add A and B"
		) as mock_title:
			result = _generate_title(options, "commits", commits, "feature-branch", "feature")
			mock_title.assert_called_once_with(commits)
			assert result == "Features: Add A and B"

	def test_generate_title_with_llm(self) -> None:
		"""Test generating PR title with LLM."""
		from codemap.cli.pr_cmd import PROptions, _generate_title

		# Setup
		options = PROptions(
			repo_path=Path("/fake/repo"),
			model="test-model",
		)
		commits = ["feat: Add feature A", "feat: Add feature B"]

		# Test with LLM strategy
		with (
			patch("codemap.cli.pr_cmd.generate_pr_title_with_llm", return_value="AI: Smart PR Title") as mock_title,
			patch("codemap.cli.pr_cmd.create_client", return_value=MagicMock()),
		):
			result = _generate_title(options, "llm", commits, "feature-branch", "feature")
			mock_title.assert_called_once()
			assert result == "AI: Smart PR Title"

	def test_generate_description_from_commits(self) -> None:
		"""Test generating PR description from commits."""
		from codemap.cli.pr_cmd import PROptions, _generate_description

		# Setup
		options = PROptions(
			repo_path=Path("/fake/repo"),
			model="test-model",
		)
		commits = ["feat: Add feature A", "feat: Add feature B"]
		content_config = {"template": "default"}

		# Test with commits strategy
		with patch(
			"codemap.cli.pr_cmd.generate_pr_description_from_commits",
			return_value="# Changes\n\n- Add feature A\n- Add feature B",
		) as mock_desc:
			result = _generate_description(
				options, "commits", commits, "feature-branch", "feature", "github-flow", "main", content_config
			)
			mock_desc.assert_called_once()
			assert "# Changes" in result
			assert "Add feature A" in result

	def test_generate_description_with_llm(self) -> None:
		"""Test generating PR description with LLM."""
		from codemap.cli.pr_cmd import PROptions, _generate_description

		# Setup
		options = PROptions(
			repo_path=Path("/fake/repo"),
			model="test-model",
		)
		commits = ["feat: Add feature A", "feat: Add feature B"]
		content_config = {"template": "default"}

		# Test with LLM strategy
		with (
			patch(
				"codemap.cli.pr_cmd.generate_pr_description_with_llm",
				return_value="# AI Generated Description\n\nThis is a smart description.",
			) as mock_desc,
			patch("codemap.cli.pr_cmd.create_client", return_value=MagicMock()),
		):
			result = _generate_description(
				options, "llm", commits, "feature-branch", "feature", "github-flow", "main", content_config
			)
			mock_desc.assert_called_once()
			assert "# AI Generated Description" in result


@pytest.mark.unit
@pytest.mark.git
class TestHandleCommits:
	"""Test handling commits for PR creation."""

	def test_handle_commits_skip(self) -> None:
		"""Test handling commits when skipping commit process."""
		from codemap.cli.pr_cmd import PROptions, _handle_commits

		# Setup options to skip commit
		options = PROptions(
			repo_path=Path("/fake/repo"),
			commit_first=False,
		)

		# Call the function
		with patch("codemap.git.commit_generator.command.CommitCommand.process_all_chunks") as mock_process:
			result = _handle_commits(options)

		# Verify skip behavior
		assert result is True
		mock_process.assert_not_called()

	def test_handle_commits_no_changes(self) -> None:
		"""Test handling commits when there are no changes."""
		# Skip this test for now as it requires too much mocking
		import pytest

		from codemap.cli.pr_cmd import PROptions, _handle_commits
		from codemap.git.utils import GitDiff

		pytest.skip("This test needs to be refactored to handle file system operations")

		# Setup options and empty diffs
		options = PROptions(
			repo_path=Path("/fake/repo"),
			commit_first=True,
			interactive=False,  # Skip the interactive prompt
		)

		# Mock empty diffs and questionary
		with (
			patch("codemap.git.utils.get_staged_diff") as mock_get_staged_diff,
			patch("codemap.git.utils.get_unstaged_diff") as mock_get_unstaged_diff,
			patch("codemap.git.utils.get_untracked_files") as mock_get_untracked_files,
			patch("codemap.cli.pr_cmd.console.print") as mock_print,
		):
			# Set up return values for the mocks
			mock_get_staged_diff.return_value = GitDiff(files=[], content="", is_staged=True)
			mock_get_unstaged_diff.return_value = GitDiff(files=[], content="", is_staged=False)
			mock_get_untracked_files.return_value = []

			# Call the function
			result = _handle_commits(options)

		# Verify behavior with no changes
		mock_print.assert_called_with("[yellow]No uncommitted changes to commit.[/yellow]")
		assert result is True


@pytest.mark.unit
@pytest.mark.git
class TestLoadLLMConfig:
	"""Test loading LLM configuration."""

	def test_load_llm_config(self) -> None:
		"""Test loading LLM configuration."""
		from codemap.cli.pr_cmd import _load_llm_config

		# Mock ConfigLoader
		with patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config_class:
			# Setup mock
			mock_config = MagicMock()
			mock_config.get_llm_config.return_value = {
				"model": "test-model",
				"api_key": "test-key",
				"api_base": "https://test-api.com",
			}
			mock_config_class.return_value = mock_config

			# Call the function
			result = _load_llm_config(Path("/fake/repo"))

			# Verify result
			assert result["model"] == "test-model"
			assert result["api_key"] == "test-key"
			assert result["api_base"] == "https://test-api.com"


@pytest.mark.unit
@pytest.mark.git
class TestWorkflowStrategy:
	"""Test workflow strategy validation."""

	def test_validate_workflow_strategy_valid(self) -> None:
		"""Test validation of valid workflow strategies."""
		from codemap.cli.pr_cmd import validate_workflow_strategy

		valid_strategies = ["github-flow", "gitflow", "trunk-based"]
		for strategy in valid_strategies:
			assert validate_workflow_strategy(strategy) == strategy

	def test_validate_workflow_strategy_invalid(self) -> None:
		"""Test validation of invalid workflow strategies."""
		from codemap.cli.pr_cmd import validate_workflow_strategy

		with patch("codemap.cli.pr_cmd.console.print") as mock_print, pytest.raises(typer.BadParameter):
			validate_workflow_strategy("invalid-strategy")

		mock_print.assert_called()


@pytest.mark.unit
@pytest.mark.git
class TestHandlePRCreation:
	"""Test PR creation handling."""

	def test_pr_creation_with_branch_selection(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test PR creation with branch selection when no branch mapping exists."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_creation

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "chore/refactor"
		mock_git_utils["branch_exists"].return_value = True

		# Create options with interactive mode
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.questionary.select") as mock_select,
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_title") as mock_generate_title,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			patch("codemap.git.utils.run_git_command") as mock_run_git,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up PR config without branch mapping for the branch type
			pr_config = {
				"defaults": {"base_branch": "main", "feature_prefix": "feat/"},
				"strategy": "github-flow",
				"branch_mapping": {},  # Empty branch mapping
				"generate": {"title_strategy": "commits", "description_strategy": "llm"},
			}
			mock_config.return_value.get_pr_config.return_value = pr_config
			mock_config.return_value.get_content_generation_config.return_value = pr_config["generate"]

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			strategy.get_default_base.return_value = "main"
			strategy.get_remote_branches.return_value = ["main", "develop", "staging"]
			strategy.get_local_branches.return_value = ["chore/refactor", "main", "feature/test"]
			mock_strategy.return_value = strategy

			# Set up PR generator
			pr_gen = MagicMock()
			pr_gen.get_existing_pr.return_value = None
			mock_pull_request = MagicMock()
			mock_pull_request.number = 123
			mock_pull_request.url = "https://github.com/user/repo/pull/123"
			pr_gen.create_pr.return_value = mock_pull_request
			mock_pr_generator.return_value = pr_gen

			# Set up branch selection
			mock_select.return_value.ask.side_effect = [
				"develop",
				"edit",
			]  # First for branch selection, second for edit method
			# Set up confirm to edit the title/description
			mock_confirm.return_value.ask.side_effect = [False, True]  # No for title edit, Yes for description edit

			# Other mocks
			mock_generate_title.return_value = "Test PR title"
			mock_generate_desc.return_value = "Test PR description"
			mock_get_commits.return_value = ["commit1", "commit2"]
			mock_run_git.return_value = ""

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test
			result = _handle_pr_creation(options, "chore/refactor")

		# Verify select was called (now we expect 2 calls - one for branch selection, one for edit method)
		assert mock_select.call_count == 2
		# First call should be for branch selection
		assert mock_select.call_args_list[0][0][0] == "Select target branch for PR:"
		# Second call should be for edit method
		assert mock_select.call_args_list[1][0][0] == "Edit method:"

		# Verify PR was created with the correct base branch
		pr_gen.create_pr.assert_called_once()
		call_args = pr_gen.create_pr.call_args[0]
		assert call_args[0] == "develop"  # Base branch
		assert call_args[1] == "chore/refactor"  # Head branch

		# Make sure the result is correct
		assert result is not None
		assert result.number == 123
		assert result.url == "https://github.com/user/repo/pull/123"

	def test_pr_display_panels(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test that PR title and description are displayed in panels."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_creation

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "feature/test"
		mock_git_utils["branch_exists"].return_value = True

		# Create options with interactive mode
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_title") as mock_generate_title,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			patch("codemap.git.utils.run_git_command") as mock_run_git,
			patch("codemap.cli.pr_cmd.console.print") as mock_console_print,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up PR config
			pr_config = {
				"defaults": {"base_branch": "main", "feature_prefix": "feat/"},
				"strategy": "github-flow",
				"branch_mapping": {"feature": {"base": "main", "prefix": "feature/"}},
				"generate": {"title_strategy": "commits", "description_strategy": "llm"},
			}
			mock_config.return_value.get_pr_config.return_value = pr_config
			mock_config.return_value.get_content_generation_config.return_value = pr_config["generate"]

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			strategy.get_default_base.return_value = "main"
			mock_strategy.return_value = strategy

			# Set up PR generator
			pr_gen = MagicMock()
			pr_gen.get_existing_pr.return_value = None
			mock_pull_request = MagicMock()
			mock_pull_request.number = 123
			mock_pull_request.url = "https://github.com/user/repo/pull/123"
			pr_gen.create_pr.return_value = mock_pull_request
			mock_pr_generator.return_value = pr_gen

			# Set up confirm to not edit title/description
			mock_confirm.return_value.ask.return_value = False

			# Other mocks
			mock_generate_title.return_value = "My Test PR Title"
			mock_generate_desc.return_value = (
				"This is a test PR description\n\n- With multiple lines\n- And bullet points"
			)
			mock_get_commits.return_value = ["commit1", "commit2"]
			mock_run_git.return_value = ""

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test
			result = _handle_pr_creation(options, "feature/test")

		# Verify Panel output was called
		mock_console_print.assert_called()

		# Verify PR creation happened with correct data
		pr_gen.create_pr.assert_called_once_with(
			"main",
			"feature/test",
			"My Test PR Title",
			"This is a test PR description\n\n- With multiple lines\n- And bullet points",
		)

		# Make sure the result is correct
		assert result is not None
		assert result.number == 123
		assert result.url == "https://github.com/user/repo/pull/123"

	def test_pr_edit_title_and_description(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test editing PR title and description in interactive mode."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_creation

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "feature/test"
		mock_git_utils["branch_exists"].return_value = True

		# Create options with interactive mode
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.questionary.select") as mock_select,
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
			patch("codemap.cli.pr_cmd.questionary.text") as mock_text,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_title") as mock_generate_title,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			patch("codemap.git.utils.run_git_command") as mock_run_git,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up PR config
			pr_config = {
				"defaults": {"base_branch": "main", "feature_prefix": "feat/"},
				"strategy": "github-flow",
				"branch_mapping": {"feature": {"base": "main", "prefix": "feature/"}},
				"generate": {"title_strategy": "commits", "description_strategy": "llm"},
			}
			mock_config.return_value.get_pr_config.return_value = pr_config
			mock_config.return_value.get_content_generation_config.return_value = pr_config["generate"]

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			strategy.get_default_base.return_value = "main"
			mock_strategy.return_value = strategy

			# Set up PR generator
			pr_gen = MagicMock()
			pr_gen.get_existing_pr.return_value = None
			mock_pull_request = MagicMock()
			mock_pull_request.number = 123
			mock_pull_request.url = "https://github.com/user/repo/pull/123"
			pr_gen.create_pr.return_value = mock_pull_request
			mock_pr_generator.return_value = pr_gen

			# Set up confirm responses to edit both title and description
			mock_confirm.return_value.ask.side_effect = [True, True]  # Edit title, edit description

			# Set up text input for new title
			mock_text.return_value.ask.return_value = "Edited PR Title"

			# Set up select for edit method (edit in text input)
			mock_select.return_value.ask.return_value = "edit"

			# Other mocks
			mock_generate_title.return_value = "Original PR Title"
			mock_generate_desc.return_value = "Original PR description"
			mock_get_commits.return_value = ["commit1", "commit2"]
			mock_run_git.return_value = ""

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test
			result = _handle_pr_creation(options, "feature/test")

		# Verify user was asked to edit title and description
		assert mock_confirm.call_count == 2

		# Verify PR creation happened with edited title
		pr_gen.create_pr.assert_called_once_with("main", "feature/test", "Edited PR Title", "Edited PR Title")

		# Verify result
		assert result is not None
		assert result.number == 123
		assert result.url == "https://github.com/user/repo/pull/123"

	def test_pr_regenerate_description(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test regenerating PR description with LLM."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_creation

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "feature/test"
		mock_git_utils["branch_exists"].return_value = True

		# Create options with interactive mode
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.questionary.select") as mock_select,
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_title") as mock_generate_title,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			patch("codemap.git.utils.run_git_command") as mock_run_git,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up PR config
			pr_config = {
				"defaults": {"base_branch": "main", "feature_prefix": "feat/"},
				"strategy": "github-flow",
				"branch_mapping": {"feature": {"base": "main", "prefix": "feature/"}},
				"generate": {"title_strategy": "commits", "description_strategy": "llm"},
			}
			mock_config.return_value.get_pr_config.return_value = pr_config
			mock_config.return_value.get_content_generation_config.return_value = pr_config["generate"]

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			strategy.get_default_base.return_value = "main"
			mock_strategy.return_value = strategy

			# Set up PR generator
			pr_gen = MagicMock()
			pr_gen.get_existing_pr.return_value = None
			mock_pull_request = MagicMock()
			mock_pull_request.number = 123
			mock_pull_request.url = "https://github.com/user/repo/pull/123"
			pr_gen.create_pr.return_value = mock_pull_request
			mock_pr_generator.return_value = pr_gen

			# Set up confirm responses to not edit title but edit description
			mock_confirm.return_value.ask.side_effect = [False, True]  # Don't edit title, edit description

			# Set up select for regeneration option
			mock_select.return_value.ask.return_value = "regenerate"

			# Other mocks
			mock_generate_title.return_value = "Original PR Title"
			# First call is for initial description, second for regenerated description
			mock_generate_desc.side_effect = ["Original PR description", "Regenerated PR description with LLM"]
			mock_get_commits.return_value = ["commit1", "commit2"]
			mock_run_git.return_value = ""

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test
			result = _handle_pr_creation(options, "feature/test")

		# Verify _generate_description was called twice (initial + regeneration)
		assert mock_generate_desc.call_count == 2

		# Verify PR was created with regenerated description
		pr_gen.create_pr.assert_called_once_with(
			"main", "feature/test", "Original PR Title", "Regenerated PR description with LLM"
		)

		# Verify result
		assert result is not None
		assert result.number == 123
		assert result.url == "https://github.com/user/repo/pull/123"

	def test_pr_update_with_panels(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test PR update with panel display."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_update
		from codemap.git.pr_generator.schemas import PullRequest

		# Create an existing PR
		existing_pr = PullRequest(
			number=42,
			url="https://github.com/user/repo/pull/42",
			title="Original PR Title",
			description="Original description",
			branch="feature/test",
		)

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "feature/test"

		# Create options with interactive mode
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			patch("codemap.cli.pr_cmd.console.print") as mock_console_print,
			patch("codemap.cli.pr_cmd.Markdown") as mock_markdown,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up content generation config
			content_config = {
				"title_strategy": "commits",
				"description_strategy": "llm",
			}
			mock_config.return_value.get_content_generation_config.return_value = content_config

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			mock_strategy.return_value = strategy

			# Set up PR generator
			pr_gen = MagicMock()
			updated_pr = PullRequest(
				number=42,
				url="https://github.com/user/repo/pull/42",
				title="Original PR Title",
				description="Updated description with Markdown formatting",
				branch="feature/test",
			)
			pr_gen.update_pr.return_value = updated_pr
			mock_pr_generator.return_value = pr_gen

			# Set up confirm to not edit title but edit description
			mock_confirm.return_value.ask.side_effect = [False, True]  # Don't edit title, edit description

			# Other mocks
			mock_generate_desc.return_value = "Updated description with Markdown formatting"
			mock_get_commits.return_value = ["commit1", "commit2"]

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test
			result = _handle_pr_update(options, existing_pr)

		# Verify Panel was created with Markdown for description
		mock_markdown.assert_called_with("Updated description with Markdown formatting")

		# Verify PR update happened with original title and updated description
		pr_gen.update_pr.assert_called_once_with(
			42, "Original PR Title", "Updated description with Markdown formatting"
		)

		# Verify console.print was called to display panels
		mock_console_print.assert_called()

		# Make sure the result is correct
		assert result is not None
		assert result.number == 42
		assert result.url == "https://github.com/user/repo/pull/42"
		assert result.description == "Updated description with Markdown formatting"

	def test_pr_update_by_number(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test PR update by number."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_update

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "feature/test"

		# Create options with PR number
		options = PROptions(
			repo_path=Path("/fake/repo"),
			pr_number=42,
			interactive=False,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			patch("codemap.cli.pr_cmd.console.print") as mock_console_print,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up content generation config
			content_config = {
				"title_strategy": "commits",
				"description_strategy": "llm",
			}
			mock_config.return_value.get_content_generation_config.return_value = content_config

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			mock_strategy.return_value = strategy

			# Set up PR generator
			pr_gen = MagicMock()
			updated_pr = MagicMock()
			updated_pr.number = 42
			updated_pr.url = "https://github.com/user/repo/pull/42"
			updated_pr.title = "Original PR Title"
			updated_pr.description = "Original description"
			pr_gen.update_pr.return_value = updated_pr
			mock_pr_generator.return_value = pr_gen

			# Other mocks
			mock_get_commits.return_value = ["commit1", "commit2"]

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test
			result = _handle_pr_update(options, None)

		# Verify PR was updated by number
		pr_gen.update_pr.assert_called_once()

		# Verify console.print was called to display success message
		mock_console_print.assert_called()

		# Make sure the result is correct
		assert result is not None
		assert result.number == 42
		assert result.url == "https://github.com/user/repo/pull/42"

	def test_markdown_formatting_in_panels(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test Markdown formatting in description panels."""
		from rich.markdown import Markdown
		from rich.text import Text

		from codemap.cli.pr_cmd import PROptions, _handle_pr_creation

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "feature/test"
		mock_git_utils["branch_exists"].return_value = True

		# Create options with interactive mode
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_title") as mock_generate_title,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			# Create real Text and Markdown objects instead of mocks
			patch("codemap.cli.pr_cmd.Text", return_value=Text("Mocked Title")),
			patch("codemap.cli.pr_cmd.Markdown", return_value=Markdown("Mocked Description")),
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up PR config
			pr_config = {
				"defaults": {"base_branch": "main", "feature_prefix": "feat/"},
				"strategy": "github-flow",
				"branch_mapping": {"feature": {"base": "main", "prefix": "feature/"}},
				"generate": {"title_strategy": "commits", "description_strategy": "llm"},
			}
			mock_config.return_value.get_pr_config.return_value = pr_config
			mock_config.return_value.get_content_generation_config.return_value = pr_config["generate"]

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			strategy.get_default_base.return_value = "main"
			mock_strategy.return_value = strategy

			# Set up PR generator
			pr_gen = MagicMock()
			pr_gen.get_existing_pr.return_value = None
			mock_pull_request = MagicMock()
			mock_pull_request.number = 123
			mock_pull_request.url = "https://github.com/user/repo/pull/123"
			pr_gen.create_pr.return_value = mock_pull_request
			mock_pr_generator.return_value = pr_gen

			# Set up confirm to not edit title/description
			mock_confirm.return_value.ask.return_value = False

			# Set up complex markdown description
			markdown_description = """# Feature Description

## Overview
This PR adds a new feature that:
- Improves user experience
- Fixes several bugs
- Adds new functionality

## Code Changes
```python
def new_function():
    return "Hello World"
```

## Testing
1. Test case 1
2. Test case 2
"""

			# Other mocks
			mock_generate_title.return_value = "New Feature PR Title"
			mock_generate_desc.return_value = markdown_description
			mock_get_commits.return_value = ["commit1", "commit2"]

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test - this now should not error
			result = _handle_pr_creation(options, "feature/test")

			# Verify PR was created successfully
			assert result is not None
			assert result.number == 123
			assert result.url == "https://github.com/user/repo/pull/123"

	def test_branch_selection_questionnaire(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test branch selection through questionnaire."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_creation

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "dev"
		mock_git_utils["branch_exists"].return_value = True
		mock_git_utils["list_branches"].return_value = ["main", "dev", "feature/one", "feature/two"]

		# Create options with interactive mode and no branch specified
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=True,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.questionary.select") as mock_select,
			patch("codemap.cli.pr_cmd.questionary.confirm") as mock_confirm,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_title") as mock_generate_title,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"

			# Set up PR config
			pr_config = {
				"defaults": {"base_branch": "main", "feature_prefix": "feature/"},
				"strategy": "github-flow",
				"branch_mapping": {"feature": {"base": "main", "prefix": "feature/"}},
				"generate": {"title_strategy": "commits", "description_strategy": "llm"},
			}
			mock_config.return_value.get_pr_config.return_value = pr_config
			mock_config.return_value.get_content_generation_config.return_value = pr_config["generate"]

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			strategy.get_default_base.return_value = "main"
			mock_strategy.return_value = strategy

			# Set up branch selection mock to choose feature/one
			mock_select.return_value.ask.return_value = "feature/one"

			# Set up PR generator
			pr_gen = MagicMock()
			pr_gen.get_existing_pr.return_value = None
			mock_pull_request = MagicMock()
			mock_pull_request.number = 123
			mock_pull_request.url = "https://github.com/user/repo/pull/123"
			pr_gen.create_pr.return_value = mock_pull_request
			mock_pr_generator.return_value = pr_gen

			# Set up confirm to not edit title/description
			mock_confirm.return_value.ask.return_value = False

			# Other mocks
			mock_generate_title.return_value = "Feature One PR Title"
			mock_generate_desc.return_value = "Feature one description"
			mock_get_commits.return_value = ["commit1", "commit2"]

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test with 'feature/one' instead of None
			result = _handle_pr_creation(options, "feature/one")

		# Verify PR was created with selected branch
		assert result is not None
		assert result.number == 123
		assert result.url == "https://github.com/user/repo/pull/123"

	def test_pr_creation_error_handling(self, mock_git_utils: dict[str, Any]) -> None:
		"""Test error handling during PR creation."""
		from codemap.cli.pr_cmd import PROptions, _handle_pr_creation
		from codemap.git.pr_generator import PRCreationError

		# Configure mocks
		mock_git_utils["get_default_branch"].return_value = "main"
		mock_git_utils["get_current_branch"].return_value = "feature/test"
		mock_git_utils["branch_exists"].return_value = True

		# Create options
		options = PROptions(
			repo_path=Path("/fake/repo"),
			interactive=False,
		)

		# Mock PR generator and dependencies
		with (
			patch("codemap.cli.pr_cmd.ConfigLoader") as mock_config,
			patch("codemap.cli.pr_cmd.PRGenerator") as mock_pr_generator,
			patch("codemap.cli.pr_cmd.create_strategy") as mock_strategy,
			patch("codemap.cli.pr_cmd.get_commit_messages") as mock_get_commits,
			patch("codemap.cli.pr_cmd._generate_title") as mock_generate_title,
			patch("codemap.cli.pr_cmd._generate_description") as mock_generate_desc,
			patch("codemap.cli.pr_cmd.loading_spinner") as mock_spinner,
			patch("codemap.cli.pr_cmd.console.print") as mock_console_print,
		):
			# Configure mocks
			mock_config.return_value.get_workflow_strategy.return_value = "github-flow"
			mock_config.return_value.get_content_generation_config.return_value = {
				"title_strategy": "commits",
				"description_strategy": "commits",
			}

			# Set up strategy
			strategy = MagicMock()
			strategy.detect_branch_type.return_value = "feature"
			strategy.get_default_base.return_value = "main"
			mock_strategy.return_value = strategy

			# Set up PR generator to raise error
			pr_gen = MagicMock()
			pr_gen.get_existing_pr.return_value = None
			pr_gen.create_pr.side_effect = PRCreationError("Failed to create PR: GitHub CLI error")
			mock_pr_generator.return_value = pr_gen

			# Other mocks
			mock_generate_title.return_value = "Test PR Title"
			mock_generate_desc.return_value = "Test PR description"
			mock_get_commits.return_value = ["commit1", "commit2"]

			# Context manager for spinner
			mock_spinner.return_value.__enter__.return_value = None

			# Call the function under test
			result = _handle_pr_creation(options, "feature/test")

			# Verify error was printed
			mock_console_print.assert_any_call("[red]Error creating PR: Failed to create PR: GitHub CLI error[/red]")
			assert result is None
