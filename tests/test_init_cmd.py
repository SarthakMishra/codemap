"""Tests for the init command."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from codemap.cli.init_cmd import init_command
from codemap.config import DEFAULT_CONFIG

if TYPE_CHECKING:
	from collections.abc import Generator


# Define missing fixtures
@pytest.fixture
def mock_parser_cls() -> Generator[Mock, None, None]:
	"""Fixture to mock the CodeParser class."""
	with patch("codemap.cli.init_cmd.CodeParser") as mock:
		mock_instance = Mock()
		mock.return_value = mock_instance
		yield mock


@pytest.fixture
def mock_console() -> Generator[Mock, None, None]:
	"""Fixture to mock the console."""
	with patch("codemap.cli.init_cmd.console") as mock:
		yield mock


@pytest.fixture
def mock_questionary() -> Generator[dict[str, Mock], None, None]:
	"""Fixture to mock questionary interactive prompts."""
	mocks = {}

	with patch("codemap.cli.init_cmd.questionary.text") as mock_text:
		mock_text_instance = Mock()
		mock_text_instance.ask.return_value = "test_repo"
		mock_text.return_value = mock_text_instance
		mocks["text"] = mock_text

		with patch("codemap.cli.init_cmd.questionary.confirm") as mock_confirm:
			mock_confirm_instance = Mock()
			mock_confirm_instance.ask.return_value = True
			mock_confirm.return_value = mock_confirm_instance
			mocks["confirm"] = mock_confirm

			with patch("codemap.cli.init_cmd.questionary.select") as mock_select:
				mock_select_instance = Mock()
				mock_select_instance.ask.return_value = "OpenAI"
				mock_select.return_value = mock_select_instance
				mocks["select"] = mock_select

				# Configure specific responses for known prompts
				def text_side_effect(prompt: str, **kwargs) -> Mock:
					mock = Mock()
					if "repository" in prompt:
						mock.ask.return_value = "test_repo"
					elif "extensions" in prompt:
						mock.ask.return_value = "py,js,ts"
					elif "token" in prompt:
						mock.ask.return_value = "10000"
					elif "API key" in prompt:
						mock.ask.return_value = "sk-test123456"
					else:
						mock.ask.return_value = kwargs.get("default", "")
					return mock

				mock_text.side_effect = text_side_effect

				def select_side_effect(prompt: str, **kwargs) -> Mock:
					mock = Mock()
					if "LLM provider" in prompt:
						mock.ask.return_value = "OpenAI"
					elif "OpenAI model" in prompt:
						mock.ask.return_value = "gpt-4o"
					elif "API key" in prompt:
						mock.ask.return_value = "Use environment variable (recommended)"
					elif "worker" in prompt:
						mock.ask.return_value = "4"
					else:
						mock.ask.return_value = kwargs.get("default", "")
					return mock

				mock_select.side_effect = select_side_effect

				yield mocks


@pytest.fixture
def mock_show_warning() -> Generator[Mock, None, None]:
	"""Fixture to mock the show_warning function."""
	with patch("codemap.cli.init_cmd.show_warning") as mock:
		yield mock


@pytest.mark.cli
@pytest.mark.unit
class TestInitCommand:
	"""Test cases for the init command."""

	def setup_method(self) -> None:
		"""Set up test environment."""
		# Create a temporary directory for testing
		self.temp_dir = Path(tempfile.mkdtemp())

		# Create test directory structures
		self.repo_root = self.temp_dir / "test_repo"
		self.repo_root.mkdir(exist_ok=True)
		self.config_file = self.repo_root / ".codemap.yml"
		self.docs_dir = self.repo_root / str(DEFAULT_CONFIG["output_dir"])
		self.cache_dir = self.repo_root / ".codemap_cache"

		# Set up a CLI runner for testing commands
		self.runner = CliRunner()

	def teardown_method(self) -> None:
		"""Clean up after tests."""
		# Make sure to clean up any files created during testing
		if self.temp_dir.exists():
			shutil.rmtree(self.temp_dir)

	@pytest.mark.path_sensitive
	@patch("codemap.cli.init_cmd.console")
	@patch("codemap.cli.init_cmd.CodeParser")
	def test_init_new_project(self, mock_parser_cls: Mock, mock_console: Mock) -> None:
		"""Test initialization of a completely new project."""
		# Setup
		mock_parser_instance = Mock()
		mock_parser_cls.return_value = mock_parser_instance

		# Make sure cache directory doesn't exist for this test
		if self.cache_dir.exists():
			shutil.rmtree(self.cache_dir)

		# Need to patch multiple things to make this test work
		with (
			patch("pathlib.Path.exists", return_value=False) as mock_exists,
			patch("codemap.cli.init_cmd.get_directory_manager") as mock_dir_manager,
			patch("codemap.cli.init_cmd.get_config_manager") as mock_config_manager,
			patch("codemap.cli.init_cmd.run_global_config_wizard", return_value=True),
			patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard,
			patch("codemap.cli.init_cmd.update_gitignore"),
			patch("pathlib.Path.mkdir"),
		):
			# Special handling for repo_root path check
			mock_exists.side_effect = lambda path=None: str(path) == str(self.repo_root)

			# Set up mocks
			mock_dir_instance = Mock()
			mock_dir_instance.config_dir = Path(tempfile.mkdtemp())
			mock_dir_instance.ensure_directories.return_value = None
			mock_dir_instance.get_project_cache_dir.return_value = self.cache_dir
			mock_dir_manager.return_value = mock_dir_instance

			mock_config_instance = Mock()
			mock_config_instance.get_config.return_value = {}
			mock_config_instance.initialize_project_config.return_value = None
			mock_config_manager.return_value = mock_config_instance

			# Set up repository config
			mock_repo_wizard.return_value = {
				"name": "test_repo",
				"token_limit": 10000,
				"use_gitignore": True,
				"extensions": ["py", "js", "ts"],
			}

			# Execute
			init_command(path=self.repo_root, force_flag=False, is_verbose=False)

		# Verify mock calls
		mock_parser_cls.assert_called_once()

		# Check success messages in a flexible way to handle path differences in CI
		mock_console.print.assert_any_call("\n✨ CodeMap initialized successfully!")

		# Verify config file and docs directory creation
		config_file_message_found = False
		docs_dir_message_found = False

		for call in mock_console.print.call_args_list:
			if str(call).find("[green]Created config file:") >= 0:
				config_file_message_found = True
			if str(call).find("[green]Created documentation directory:") >= 0:
				docs_dir_message_found = True

		assert config_file_message_found, "No message for config file creation found"
		assert docs_dir_message_found, "No message for docs directory creation found"

	@patch("codemap.cli.init_cmd.show_warning")
	@patch("codemap.cli.init_cmd.CodeParser")
	@pytest.mark.usefixtures("mock_console")
	def test_init_with_existing_files_no_force(self, mock_parser_cls: Mock, mock_show_warning: Mock) -> None:
		"""Test initialization with existing files without force flag."""
		# Setup - create existing files
		self.config_file.write_text("existing config")
		self.docs_dir.mkdir(exist_ok=True)
		self.cache_dir.mkdir(exist_ok=True)

		# Execute and verify
		with pytest.raises(typer.Exit):
			init_command(path=self.repo_root, force_flag=False, is_verbose=False)

		# Verify warning message was shown via show_warning
		assert mock_show_warning.called

		# Verify files were not overwritten
		assert self.config_file.read_text() == "existing config"

		# Verify CodeParser was not initialized
		mock_parser_cls.assert_not_called()

	@patch("codemap.cli.init_cmd.CodeParser")
	@pytest.mark.usefixtures("mock_console")
	def test_init_with_existing_files_with_force(self, mock_parser_cls: Mock) -> None:
		"""Test initialization with existing files with force flag."""
		# Setup - create existing files
		self.config_file.write_text("existing config")
		# Create a file inside the docs dir to verify it's removed when using force flag
		self.docs_dir.mkdir(exist_ok=True, parents=True)
		test_file = self.docs_dir / "test_file.txt"
		test_file.write_text("test content")

		# Create cache directory with a test file
		self.cache_dir.mkdir(exist_ok=True, parents=True)
		cache_test_file = self.cache_dir / "test_cache_file.txt"
		cache_test_file.write_text("cache test content")

		# Force remove cache directory for test consistency
		with (
			patch("codemap.cli.init_cmd.run_global_config_wizard", return_value=True),
			patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard,
			patch("codemap.cli.init_cmd.get_directory_manager") as mock_dir_manager,
			patch("codemap.cli.init_cmd.get_config_manager") as mock_config_manager,
			patch("codemap.cli.init_cmd.initialize_processor") as mock_processor,
		):
			# Set up repository config
			mock_repo_wizard.return_value = {
				"name": "test_repo",
				"token_limit": 10000,
				"use_gitignore": True,
				"extensions": ["py", "js", "ts"],
			}

			# Mock dir_manager to prevent TypeError with the config_dir / "settings.yml"
			mock_dir_instance = Mock()
			mock_dir_instance.get_project_cache_dir.return_value = self.cache_dir

			# Fix the TypeError with config_dir
			mock_dir_instance.config_dir = Path(tempfile.mkdtemp())
			mock_dir_instance.user_config_dir = Path(tempfile.mkdtemp())
			mock_dir_instance.user_data_dir = Path(tempfile.mkdtemp())
			mock_dir_instance.user_cache_dir = Path(tempfile.mkdtemp())
			mock_dir_instance.user_log_dir = Path(tempfile.mkdtemp())

			mock_dir_manager.return_value = mock_dir_instance

			# Patch config_manager
			mock_config_instance = Mock()
			mock_config_instance.get_config.return_value = {}
			mock_config_instance.initialize_project_config.return_value = None
			mock_config_manager.return_value = mock_config_instance

			# Prevent the processor initialization
			mock_processor.return_value = Mock()

			# Execute
			init_command(path=self.repo_root, force_flag=True, is_verbose=False)

		# Verify CodeParser was initialized
		mock_parser_cls.assert_called_once()

	@patch("codemap.cli.init_cmd.exit_with_error")
	@pytest.mark.usefixtures("mock_console")
	def test_init_filesystem_error(self, mock_exit_error: Mock) -> None:
		"""Test handling of filesystem errors during initialization."""
		# Create a simpler, more direct test that just verifies the error handler

		# Define a function that will be used to simulate the initialization
		# and raise the filesystem error we want to test
		def initialize_with_error() -> None:
			msg = "Permission denied"
			raise PermissionError(msg)

		# Patch config_manager.initialize_project_config to raise our error
		with (
			patch(
				"codemap.utils.config_manager.ConfigManager.initialize_project_config",
				side_effect=initialize_with_error,
			),
			# Patch exists to avoid early exit due to existing files
			patch.object(Path, "exists", return_value=False),
			# Also patch global and repo wizards to avoid interactive prompts
			patch("codemap.cli.init_cmd.run_global_config_wizard", return_value=True),
			patch("codemap.cli.init_cmd.run_repo_config_wizard", return_value={}),
			# Patch progress_indicator to avoid context manager issues
			patch("codemap.cli.init_cmd.progress_indicator") as mock_progress,
		):
			mock_progress_ctx = Mock()
			mock_progress_ctx.__enter__ = Mock(return_value=lambda: None)
			mock_progress_ctx.__exit__ = Mock(return_value=None)
			mock_progress.return_value = mock_progress_ctx

			# Run the command - should trigger the error handler
			init_command(path=self.repo_root, force_flag=True, is_verbose=False)

		# Verify exit_with_error was called with the correct error message
		mock_exit_error.assert_called_once()
		error_msg = mock_exit_error.call_args[0][0]
		assert "File system error: Permission denied" in error_msg

	@patch("codemap.cli.init_cmd.CodeParser")
	@patch("codemap.cli.init_cmd.exit_with_error")
	@pytest.mark.usefixtures("mock_console")
	def test_init_code_parser_error(self, mock_exit_error: Mock, mock_parser_cls: Mock) -> None:
		"""Test handling of CodeParser initialization errors."""
		# Setup - simulate error when initializing CodeParser
		mock_parser_cls.side_effect = ValueError("Parser error")

		# Patch questionary to avoid interactive prompts
		with (
			patch("codemap.cli.init_cmd.questionary") as mock_questionary,
			# Patch all file operations to bypass existing files check
			patch("pathlib.Path.exists", return_value=False),
			# Patch directory manager and config manager
			patch("codemap.cli.init_cmd.get_directory_manager") as mock_dir_manager,
			patch("codemap.cli.init_cmd.get_config_manager") as mock_config_manager,
			# Patch to skip file operations
			patch("pathlib.Path.mkdir"),
			patch("codemap.cli.init_cmd.update_gitignore"),
			patch("codemap.cli.init_cmd.run_global_config_wizard", return_value=True),
			patch("codemap.cli.init_cmd.run_repo_config_wizard", return_value={}),
			patch("codemap.cli.init_cmd.setup_api_keys"),
			patch("codemap.cli.init_cmd.progress_indicator") as mock_progress,
		):
			# Set up questionary mocks
			mock_text = Mock()
			mock_text.ask.return_value = "test_value"
			mock_questionary.text.return_value = mock_text

			mock_select = Mock()
			mock_select.ask.return_value = "OpenAI"
			mock_questionary.select.return_value = mock_select

			mock_confirm = Mock()
			mock_confirm.ask.return_value = True
			mock_questionary.confirm.return_value = mock_confirm

			# Set up directory manager mock
			mock_dir_instance = Mock()
			mock_dir_instance.config_dir = Path(tempfile.mkdtemp())
			mock_dir_instance.user_config_dir = Path(tempfile.mkdtemp())
			mock_dir_instance.ensure_directories.return_value = None
			mock_dir_instance.get_project_cache_dir.return_value = self.cache_dir
			mock_dir_manager.return_value = mock_dir_instance

			# Set up config manager mock
			mock_config_instance = Mock()
			mock_config_instance.get_config.return_value = {}
			mock_config_instance.initialize_project_config.return_value = None
			mock_config_manager.return_value = mock_config_instance

			# Mock progress context manager
			mock_progress_ctx = Mock()
			mock_progress_ctx.__enter__ = Mock(return_value=lambda: None)
			mock_progress_ctx.__exit__ = Mock(return_value=None)
			mock_progress.return_value = mock_progress_ctx

			# Execute and verify
			init_command(path=self.repo_root, force_flag=False, is_verbose=False)

		# Verify exit_with_error was called with the correct error message
		mock_exit_error.assert_called_once()
		error_msg = mock_exit_error.call_args[0][0]
		assert "Configuration error: Parser error" in error_msg

	def test_init_command_integration_with_cli(self) -> None:
		"""Test init command through the CLI interface."""
		# Setup
		parser_patcher = patch("codemap.cli.init_cmd.CodeParser")
		parser_patcher.start()

		# Patch run_global_config_wizard to return immediately
		with patch("codemap.cli.init_cmd.run_global_config_wizard") as mock_global_wizard:
			mock_global_wizard.return_value = True

			# Patch run_repo_config_wizard to return a simple config
			with patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard:
				mock_repo_wizard.return_value = {
					"name": "test_repo",
					"token_limit": 10000,
				}

				try:
					import codemap.cli

					# Execute
					result = self.runner.invoke(codemap.cli.app, ["init", str(self.repo_root), "--force"])

					# Verify
					assert result.exit_code == 0, f"Command failed with output: {result.output}"
				finally:
					# Clean up
					parser_patcher.stop()
