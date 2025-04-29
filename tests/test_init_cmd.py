"""Tests for the init command."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
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

		# Patch run_global_config_wizard to return immediately
		with patch("codemap.cli.init_cmd.run_global_config_wizard") as mock_global_wizard:
			mock_global_wizard.return_value = True

			# Patch run_repo_config_wizard to return a simple config
			with patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard:
				mock_repo_wizard.return_value = {
					"name": "test_repo",
					"token_limit": 10000,
					"use_gitignore": True,
					"extensions": ["py", "js", "ts"],
				}

				# Execute
				init_command(path=self.repo_root, force_flag=False, is_verbose=False)

		# Verify
		assert self.config_file.exists()
		assert self.docs_dir.exists()

		# Verify CodeParser was initialized
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

	@patch("codemap.cli.init_cmd.console")
	@patch("codemap.cli.init_cmd.CodeParser")
	@patch("codemap.cli.init_cmd.show_warning")
	@pytest.mark.usefixtures("mock_console")
	def test_init_with_existing_files_no_force(self, mock_show_warning: Mock, mock_parser_cls: Mock) -> None:
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

		# Check the warning message
		warning_message = mock_show_warning.call_args[0][0]
		assert "CodeMap files already exist:" in warning_message
		assert str(self.config_file) in warning_message
		assert str(self.docs_dir) in warning_message
		assert str(self.cache_dir) in warning_message
		assert "Use --force to overwrite." in warning_message

		# Verify files were not overwritten
		assert self.config_file.read_text() == "existing config"

		# Verify CodeParser was not initialized
		mock_parser_cls.assert_not_called()

	@patch("codemap.cli.init_cmd.console")
	@patch("codemap.cli.init_cmd.CodeParser")
	@pytest.mark.usefixtures("mock_show_warning", "mock_console")
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

		# Patch run_global_config_wizard to return immediately
		with patch("codemap.cli.init_cmd.run_global_config_wizard") as mock_global_wizard:
			mock_global_wizard.return_value = True

			# Patch run_repo_config_wizard to return a simple config
			with patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard:
				mock_repo_wizard.return_value = {
					"name": "test_repo",
					"token_limit": 10000,
					"use_gitignore": True,
					"extensions": ["py", "js", "ts"],
				}

				# Execute
				init_command(path=self.repo_root, force_flag=True, is_verbose=False)

		# Verify
		assert self.config_file.exists()
		assert self.docs_dir.exists()
		# The file should no longer exist as the directory is completely removed and recreated
		assert not test_file.exists()
		# The cache file should no longer exist as the directory is completely removed and recreated
		assert not cache_test_file.exists()

		# Verify CodeParser was initialized
		mock_parser_cls.assert_called_once()

	@pytest.mark.path_sensitive
	@patch("codemap.cli.init_cmd.console")
	@patch("codemap.cli.init_cmd.CodeParser")
	def test_init_with_verbose_flag(self, mock_parser_cls: Mock, mock_console: Mock) -> None:
		"""Test initialization with verbose flag."""
		# Setup
		mock_parser_instance = Mock()
		mock_parser_cls.return_value = mock_parser_instance

		# Make sure cache directory doesn't exist for this test
		if self.cache_dir.exists():
			shutil.rmtree(self.cache_dir)

		# Patch run_global_config_wizard to return immediately
		with patch("codemap.cli.init_cmd.run_global_config_wizard") as mock_global_wizard:
			mock_global_wizard.return_value = True

			# Patch run_repo_config_wizard to return a simple config
			with patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard:
				mock_repo_wizard.return_value = {
					"name": "test_repo",
					"token_limit": 10000,
					"use_gitignore": True,
					"extensions": ["py", "js", "ts"],
				}

				# Execute
				init_command(path=self.repo_root, force_flag=False, is_verbose=True)

		# Verify
		assert self.config_file.exists()
		assert self.docs_dir.exists()
		assert mock_console.print.called

		# Verify verbose messaging
		verbose_calls = [call_args[0][0] for call_args in mock_console.print.call_args_list]
		assert any("Creating configuration file" in message for message in verbose_calls)
		assert any("Creating documentation directory" in message for message in verbose_calls)
		assert any("Initialization complete" in message for message in verbose_calls)

	@patch("codemap.cli.init_cmd.console")
	@patch("codemap.cli.init_cmd.show_warning")
	@pytest.mark.usefixtures("mock_show_warning", "mock_parser_cls")
	def test_init_filesystem_error(self, mock_console: Mock) -> None:
		"""Test handling of filesystem errors during initialization."""

		# Setup - simulate error when writing the config file
		def side_effect(*_args, **_kwargs) -> NoReturn:
			msg = "Permission denied"
			raise PermissionError(msg)

		# Patch run_global_config_wizard to return immediately
		with patch("codemap.cli.init_cmd.run_global_config_wizard") as mock_global_wizard:
			mock_global_wizard.return_value = True

			# Patch run_repo_config_wizard to return a simple config
			with patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard:
				mock_repo_wizard.return_value = {
					"name": "test_repo",
					"token_limit": 10000,
				}

				with patch.object(Path, "write_text", side_effect=side_effect):
					# Execute and verify
					with pytest.raises(typer.Exit):
						init_command(path=self.repo_root, force_flag=False, is_verbose=False)

					# Verify error message
					mock_console.print.assert_any_call("[red]File system error: Permission denied")

	@patch("codemap.cli.init_cmd.console")
	def test_init_code_parser_error(self, mock_console: Mock, mock_parser_cls: Mock) -> None:
		"""Test handling of CodeParser initialization errors."""
		# Setup - simulate error when initializing CodeParser
		mock_parser_cls.side_effect = ValueError("Parser error")

		# Patch run_global_config_wizard to return immediately
		with patch("codemap.cli.init_cmd.run_global_config_wizard") as mock_global_wizard:
			mock_global_wizard.return_value = True

			# Patch run_repo_config_wizard to return a simple config
			with patch("codemap.cli.init_cmd.run_repo_config_wizard") as mock_repo_wizard:
				mock_repo_wizard.return_value = {
					"name": "test_repo",
					"token_limit": 10000,
				}

				# Execute and verify
				with pytest.raises(typer.Exit):
					init_command(path=self.repo_root, force_flag=False, is_verbose=False)

				# Verify error message
				mock_console.print.assert_any_call("[red]Configuration error: Parser error")

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
