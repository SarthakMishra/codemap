"""Base test classes for different types of tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

if TYPE_CHECKING:
	from pathlib import Path

	from click.testing import Result as CliResult


# Define type for message generator
class MessageGeneratorLike(Protocol):
	"""Protocol for message generator."""

	generate_message: Any
	fallback_generation: Any


# For type checking
T = TypeVar("T")


class GitTestBase:
	"""Base class for Git-related tests."""

	@pytest.fixture(autouse=True)
	def setup_git_mocks(self, mock_git_utils: dict[str, Mock]) -> None:
		"""Set up Git mocks."""
		self.git_utils = mock_git_utils
		self._patchers = []

	def mock_repo_path(self, path: str = "/mock/repo") -> None:
		"""
		Mock the repository path.

		Args:
		    path: Mock path to use

		"""
		patcher = patch("codemap.git.utils.ExtendedGitRepoContext.get_repo_root")
		self.mock_get_repo_root = patcher.start()
		self.mock_get_repo_root.return_value = path
		self._patchers.append(patcher)

	def teardown_method(self, _: None) -> None:
		"""Clean up patchers after test method execution."""
		for patcher in self._patchers:
			patcher.stop()
		self._patchers = []


class LLMTestBase:
	"""Base class for LLM-related tests."""

	message_generator: MessageGeneratorLike

	@pytest.fixture(autouse=True)
	def setup_llm_mocks(self, mock_message_generator: MessageGeneratorLike) -> None:
		"""Set up LLM mocks."""
		self.message_generator = mock_message_generator

	def mock_llm_response(self, response: str, success: bool = True) -> None:
		"""
		Mock the LLM response.

		Args:
		    response: Response to return
		    success: Whether the generation was successful

		"""
		self.message_generator.generate_message.return_value = (response, success)


class CLITestBase:
	"""Base class for CLI tests."""

	@pytest.fixture(autouse=True)
	def setup_cli(self) -> None:
		"""Set up CLI test environment."""
		self.runner = CliRunner()

	def invoke_command(self, command: list[str], input_text: str | None = None) -> CliResult:
		"""
		Invoke a CLI command.

		Args:
		    command: Command to invoke
		    input_text: Optional input to provide

		Returns:
		    Command result

		"""
		import codemap.cli

		return self.runner.invoke(codemap.cli.app, command, input=input_text)


class FileSystemTestBase:
	"""Base class for filesystem-related tests."""

	@pytest.fixture(autouse=True)
	def setup_fs(self, temp_dir: Path) -> None:
		"""Set up filesystem test environment."""
		self.temp_dir = temp_dir

	def create_test_file(self, rel_path: str, content: str) -> Path:
		"""
		Create a test file with the specified content.

		Args:
		    rel_path: Relative path to create
		    content: File content

		Returns:
		    Path to the created file

		"""
		path = self.temp_dir / rel_path
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(content)
		return path
