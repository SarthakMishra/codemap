"""Tests for the PR command."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

if TYPE_CHECKING:
	from collections.abc import Iterator

	from codemap.git.pr_generator.command import PRWorkflowCommand


@pytest.fixture
def mock_pr_command_impl() -> Iterator[AsyncMock]:
	"""Fixture to mock the _pr_command_impl function."""
	with patch("codemap.cli.pr_cmd._pr_command_impl") as mock_impl:
		# Use AsyncMock since it's an async function
		mock_impl.return_value = None
		yield mock_impl


@pytest.mark.unit
@pytest.mark.cli
@pytest.mark.git
class TestPRCommandModule:
	"""Test the PR command module structure."""

	def test_pr_command_structure(self) -> None:
		"""Test that the PR command module has the expected structure."""
		# Import the module
		pr_cmd = importlib.import_module("codemap.cli.pr_cmd")

		# Check that the key functions exist
		assert hasattr(pr_cmd, "register_command"), "register_command function is missing"
		assert hasattr(pr_cmd, "_pr_command_impl"), "Implementation function is missing"
		assert hasattr(pr_cmd, "PRAction"), "PRAction enum is missing"
		assert hasattr(pr_cmd, "validate_workflow_strategy"), "Validation function is missing"

		# Check that PRAction has the expected values
		assert pr_cmd.PRAction.CREATE == "create", "PRAction.CREATE should be 'create'"
		assert pr_cmd.PRAction.UPDATE == "update", "PRAction.UPDATE should be 'update'"

		# Check that the command annotations are defined
		assert hasattr(pr_cmd, "ActionArg"), "ActionArg annotation is missing"
		assert hasattr(pr_cmd, "BranchNameOpt"), "BranchNameOpt annotation is missing"
		assert hasattr(pr_cmd, "WorkflowOpt"), "WorkflowOpt annotation is missing"

	def test_pr_command_impl_signature(self) -> None:
		"""Test the signature of the _pr_command_impl function."""
		# Ensure the function has the correct signature
		from inspect import iscoroutinefunction, signature

		from codemap.cli.pr_cmd import _pr_command_impl

		# Check if it's an async function directly
		assert iscoroutinefunction(_pr_command_impl), "Should be an async function"

		sig = signature(_pr_command_impl)

		# Check required parameters
		parameters = sig.parameters
		assert "action" in parameters, "action parameter is missing"
		assert "branch_name" in parameters, "branch_name parameter is missing"
		assert "workflow" in parameters, "workflow parameter is missing"
		assert "non_interactive" in parameters, "non_interactive parameter is missing"

	def test_validate_workflow_strategy(self) -> None:
		"""Test the validate_workflow_strategy function."""
		from codemap.cli.pr_cmd import validate_workflow_strategy

		# Test valid strategies
		valid_strategies = ["github-flow", "gitflow", "trunk-based"]
		for strategy in valid_strategies:
			assert validate_workflow_strategy(strategy) == strategy

		# Test None is allowed
		assert validate_workflow_strategy(None) is None

		# Test invalid strategy raises error
		with pytest.raises(typer.BadParameter):
			validate_workflow_strategy("invalid-strategy")


@pytest.mark.unit
@pytest.mark.git
class TestPRWorkflowCommand:
	"""Test the PRWorkflowCommand class."""

	@pytest.fixture
	def mock_llm_client(self) -> MagicMock:
		"""Create a mock LLM client for testing."""
		client = MagicMock()

		# Support different completion methods based on API changes
		client.chat_completion.return_value = '{"title": "Test PR", "description": "Test description"}'
		client.get_completion.return_value = '{"title": "Test PR", "description": "Test description"}'
		client.completion.return_value = '{"title": "Test PR", "description": "Test description"}'

		return client

	@pytest.fixture
	def mock_config_loader(self) -> MagicMock:
		"""Create a mock config loader."""
		config = MagicMock()

		# Set up the config to return values for PR generation
		config.get.return_value = {
			"pr": {"generate": {"title_strategy": "llm", "description_strategy": "llm"}, "strategy": "github-flow"}
		}
		return config

	@pytest.fixture
	def mock_pr_generator(self) -> Iterator[MagicMock]:
		"""Create a mock PR generator."""
		generator = MagicMock()
		generator.create_pr.return_value = MagicMock(
			branch="feature/test",
			title="Test PR",
			description="Test description",
			url="https://github.com/org/repo/pull/1",
			number=1,
		)
		generator.update_pr.return_value = MagicMock(
			branch="feature/test",
			title="Updated PR",
			description="Updated description",
			url="https://github.com/org/repo/pull/1",
			number=1,
		)
		return generator

	@pytest.fixture
	def mock_pgu(self) -> MagicMock:
		"""Create a mock PRGitUtils instance."""
		mock = MagicMock()
		# Set up necessary mock methods
		mock.get_commit_messages.return_value = ["fix: bug fix", "feat: new feature"]
		mock.branch = "feature/test-branch"  # Default branch for tests
		return mock

	@pytest.fixture
	def workflow_command(
		self,
		mock_llm_client: MagicMock,
		mock_config_loader: MagicMock,
		mock_pr_generator: MagicMock,
		mock_pgu: MagicMock,
	) -> Iterator[PRWorkflowCommand]:
		"""Create a mock PRWorkflowCommand with necessary patches."""
		with (
			patch("codemap.git.pr_generator.command.PRGenerator", return_value=mock_pr_generator),
			patch("codemap.git.utils.ExtendedGitRepoContext.get_repo_root", return_value=Path("/path/to/repo")),
			patch("codemap.git.pr_generator.command.create_strategy") as mock_create_strategy,
			patch(
				"codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_commit_messages",
				return_value=["fix: bug fix", "feat: new feature"],
			),
			patch("codemap.git.pr_generator.command.get_existing_pr", return_value=None),
			patch("codemap.git.pr_generator.command.generate_pr_title_with_llm", return_value="Test PR Title"),
			patch(
				"codemap.git.pr_generator.command.generate_pr_description_with_llm", return_value="Test PR Description"
			),
			patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance", return_value=mock_pgu),
		):
			# Configure strategy mock
			strategy_mock = MagicMock()
			strategy_mock.detect_branch_type.return_value = "feature"
			mock_create_strategy.return_value = strategy_mock

			# Import here to avoid circular imports
			from codemap.git.pr_generator.command import PRWorkflowCommand

			command = PRWorkflowCommand(
				config_loader=mock_config_loader,
				llm_client=mock_llm_client,
			)

			# Manually set some attributes that might not be initialized correctly in tests
			command.content_config = MagicMock()
			command.content_config.title_strategy = "llm"
			command.content_config.description_strategy = "llm"
			command.pr_generator = mock_pr_generator

			yield command

	@pytest.mark.asyncio
	async def test_generate_title(self, workflow_command, mock_llm_client: MagicMock) -> None:
		"""Test title generation from commits."""
		with patch("codemap.git.pr_generator.command.generate_pr_title_with_llm") as mock_title_gen:
			mock_title_gen.return_value = "Test PR Title"

			# Test LLM-based title generation
			commits = ["fix: bug fix", "feat: new feature"]
			title = workflow_command._generate_title(commits, "feature/test-branch", "feature")

			# Verify LLM was called
			mock_title_gen.assert_called_once_with(commits, llm_client=mock_llm_client)
			assert title == "Test PR Title"

			# Test fallback for empty commits
			with patch("codemap.git.pr_generator.command.generate_pr_title_with_llm") as mock_empty_title:
				title = workflow_command._generate_title([], "feature/test-branch", "feature")
				assert "Feature: Test branch" in title
				assert not mock_empty_title.called

	@pytest.mark.asyncio
	async def test_generate_description(self, workflow_command, mock_llm_client: MagicMock) -> None:
		"""Test description generation from commits."""
		with patch("codemap.git.pr_generator.command.generate_pr_description_with_llm") as mock_desc_gen:
			mock_desc_gen.return_value = "Test PR Description"

			# Test LLM-based description generation
			commits = ["fix: bug fix", "feat: new feature"]
			description = workflow_command._generate_description(commits, "feature/test-branch", "feature", "main")

			# Verify LLM was called
			mock_desc_gen.assert_called_once_with(commits, llm_client=mock_llm_client)
			assert description == "Test PR Description"

			# Test fallback for empty commits
			with patch("codemap.git.pr_generator.command.generate_pr_description_with_llm") as mock_empty_desc:
				description = workflow_command._generate_description([], "feature/test-branch", "feature", "main")
				assert "Changes in feature/test-branch" in description
				assert not mock_empty_desc.called

	@pytest.mark.asyncio
	async def test_create_pr_workflow(
		self, workflow_command, mock_pr_generator: MagicMock, mock_pgu: MagicMock
	) -> None:
		"""Test the PR creation workflow."""
		with (
			patch("codemap.git.pr_generator.command.generate_pr_title_with_llm") as mock_title_gen,
			patch("codemap.git.pr_generator.command.generate_pr_description_with_llm") as mock_desc_gen,
		):
			# Set up mocks
			mock_pgu.get_commit_messages.return_value = ["fix: bug fix", "feat: new feature"]
			mock_title_gen.return_value = "Test PR Title"
			mock_desc_gen.return_value = "Test PR Description"

			# Test PR creation
			pr = workflow_command.create_pr_workflow("main", "feature/test-branch")

			# Verify the correct methods were called
			mock_pgu.get_commit_messages.assert_called_once_with("main", "feature/test-branch")
			mock_title_gen.assert_called_once()
			mock_desc_gen.assert_called_once()

			# Verify PR generator was called correctly
			mock_pr_generator.create_pr.assert_called_once_with(
				"main", "feature/test-branch", "Test PR Title", "Test PR Description"
			)

			# Verify the returned PR object
			assert pr.number == 1
			assert pr.url is not None

	@pytest.mark.asyncio
	async def test_update_pr_workflow(
		self, workflow_command, mock_pr_generator: MagicMock, mock_pgu: MagicMock
	) -> None:
		"""Test the PR update workflow."""
		with (
			patch("codemap.git.pr_generator.command.generate_pr_title_with_llm") as mock_title_gen,
			patch("codemap.git.pr_generator.command.generate_pr_description_with_llm") as mock_desc_gen,
		):
			# Set up mocks
			mock_pgu.get_commit_messages.return_value = ["fix: bug fix", "feat: new feature"]
			mock_title_gen.return_value = "Updated PR Title"
			mock_desc_gen.return_value = "Updated PR Description"

			# Test PR update with auto-generated title and description
			pr = workflow_command.update_pr_workflow(pr_number=1, base_branch="main", head_branch="feature/test-branch")

			# Verify the correct methods were called
			mock_pgu.get_commit_messages.assert_called_once_with("main", "feature/test-branch")
			mock_title_gen.assert_called_once()
			mock_desc_gen.assert_called_once()

			# Verify PR generator was called correctly
			mock_pr_generator.update_pr.assert_called_once_with(1, "Updated PR Title", "Updated PR Description")

			# Verify the returned PR object
			assert pr.number == 1
			assert pr.url is not None

			# Reset mocks
			mock_pgu.get_commit_messages.reset_mock()
			mock_title_gen.reset_mock()
			mock_desc_gen.reset_mock()
			mock_pr_generator.update_pr.reset_mock()

			# Test PR update with provided title and description
			pr = workflow_command.update_pr_workflow(
				pr_number=1, title="Custom Title", description="Custom Description"
			)

			# Verify no generation was needed
			assert not mock_pgu.get_commit_messages.called
			assert not mock_title_gen.called
			assert not mock_desc_gen.called

			# Verify PR generator was called correctly with provided values
			mock_pr_generator.update_pr.assert_called_once_with(1, "Custom Title", "Custom Description")
