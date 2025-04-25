"""Tests for the init command."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from unittest.mock import Mock, patch

import pytest
import typer
import yaml
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

		# Execute
		init_command(path=self.repo_root, force_flag=False, is_verbose=False)

		# Verify
		assert self.config_file.exists()
		assert self.docs_dir.exists()

		# Check config file contents
		config_data = yaml.safe_load(self.config_file.read_text())
		assert config_data == DEFAULT_CONFIG

		# Verify CodeParser was initialized
		mock_parser_cls.assert_called_once()

		# Check success messages in a flexible way to handle path differences in CI
		mock_console.print.assert_any_call("\n✨ CodeMap initialized successfully!")

		# Check if the file path appears in any call - don't rely on exact string
		config_file_calls = [
			call
			for call in mock_console.print.call_args_list
			if "[green]Created config file:" in str(call) and ".codemap.yml" in str(call)
		]
		assert config_file_calls, "No call found for creating config file"

		# Check for docs directory creation - more flexible matching
		docs_dir_calls = [
			call
			for call in mock_console.print.call_args_list
			if "[green]Created documentation directory:" in str(call) and "docs" in str(call)
		]
		if not docs_dir_calls:
			# Alternative check - sometimes the path format varies
			docs_dir_calls = [
				call for call in mock_console.print.call_args_list if "documentation directory" in str(call).lower()
			]
		assert docs_dir_calls, "No call found for creating docs directory"

		# Check next steps are shown
		mock_console.print.assert_any_call("\nNext steps:")
		mock_console.print.assert_any_call("1. Review and customize .codemap.yml")
		mock_console.print.assert_any_call("2. Run 'codemap generate' to create documentation")

	@pytest.mark.path_sensitive
	@patch("codemap.cli.init_cmd.console")
	@patch("codemap.cli.init_cmd.CodeParser")
	def test_init_with_existing_files_no_force(self, mock_parser_cls: Mock, mock_console: Mock) -> None:
		"""Test initialization with existing files without force flag."""
		# Setup - create existing files
		self.config_file.write_text("existing config")
		self.docs_dir.mkdir(exist_ok=True)

		# Execute and verify
		with pytest.raises(typer.Exit):
			init_command(path=self.repo_root, force_flag=False, is_verbose=False)

		# Verify warning message about existing files
		mock_console.print.assert_any_call("[yellow]CodeMap files already exist:")

		# Check if warning about existing files is shown
		all_calls = [str(call) for call in mock_console.print.call_args_list]
		config_file_mentioned = any(".codemap.yml" in call and "[yellow]" in call for call in all_calls)
		assert config_file_mentioned, "No warning shown for existing config file"

		# More flexible docs directory check - sometimes paths are formatted differently
		docs_dir_mentioned = any(
			("docs" in call or "documentation" in call.lower()) and "[yellow]" in call for call in all_calls
		)
		assert docs_dir_mentioned, "No warning shown for existing docs directory"

		mock_console.print.assert_any_call("[yellow]Use --force to overwrite.")

		# Verify files were not overwritten
		assert self.config_file.read_text() == "existing config"

		# Verify CodeParser was not initialized
		mock_parser_cls.assert_not_called()

	@pytest.mark.usefixtures("mock_console")
	def test_init_with_existing_files_with_force(self, mock_parser_cls: Mock) -> None:
		"""Test initialization with existing files with force flag."""
		# Setup - create existing files
		self.config_file.write_text("existing config")
		# Create a file inside the docs dir to verify it's removed when using force flag
		self.docs_dir.mkdir(exist_ok=True, parents=True)
		test_file = self.docs_dir / "test_file.txt"
		test_file.write_text("test content")

		# Execute
		init_command(path=self.repo_root, force_flag=True, is_verbose=False)

		# Verify
		assert self.config_file.exists()
		assert self.docs_dir.exists()
		# The file should no longer exist as the directory is completely removed and recreated
		assert not test_file.exists()

		# Check config file was overwritten
		config_data = yaml.safe_load(self.config_file.read_text())
		assert config_data == DEFAULT_CONFIG

		# Verify CodeParser was initialized
		mock_parser_cls.assert_called_once()

	@pytest.mark.usefixtures("mock_parser_cls", "mock_console")
	def test_init_with_verbose_flag(self) -> None:
		"""Test initialization with verbose flag."""
		# Setup
		setup_logging_patcher = patch("codemap.cli.init_cmd.setup_logging")
		mock_setup_logging = setup_logging_patcher.start()

		try:
			# Execute
			init_command(path=self.repo_root, force_flag=False, is_verbose=True)

			# Verify
			mock_setup_logging.assert_called_once_with(is_verbose=True)
		finally:
			# Clean up
			setup_logging_patcher.stop()

	@pytest.mark.usefixtures("mock_parser_cls")
	def test_init_filesystem_error(self, mock_console: Mock) -> None:
		"""Test handling of filesystem errors during initialization."""

		# Setup - simulate error when writing the config file
		def side_effect(*_args, **_kwargs) -> NoReturn:
			msg = "Permission denied"
			raise PermissionError(msg)

		with patch.object(Path, "write_text", side_effect=side_effect):
			# Execute and verify
			with pytest.raises(typer.Exit):
				init_command(path=self.repo_root, force_flag=False, is_verbose=False)

			# Verify error message
			mock_console.print.assert_called_with("[red]File system error: Permission denied")

	@patch("codemap.cli.init_cmd.console")
	@patch("codemap.cli.init_cmd.CodeParser")
	def test_init_code_parser_error(self, mock_parser_cls: Mock, mock_console: Mock) -> None:
		"""Test handling of CodeParser initialization errors."""
		# Setup - simulate error when initializing CodeParser
		mock_parser_cls.side_effect = ValueError("Parser error")

		# Execute and verify
		with pytest.raises(typer.Exit):
			init_command(path=self.repo_root, force_flag=False, is_verbose=False)

		# Verify error message
		mock_console.print.assert_called_with("[red]Configuration error: Parser error")

	@pytest.mark.usefixtures("mock_console")
	def test_init_command_integration_with_cli_app(self) -> None:
		"""Test init command through the CLI interface."""
		# Setup
		parser_patcher = patch("codemap.cli.init_cmd.CodeParser")
		parser_patcher.start()

		try:
			import codemap.cli_app

			# Execute
			result = self.runner.invoke(codemap.cli_app.app, ["init", str(self.repo_root), "--force"])

			# Verify
			assert result.exit_code == 0
			assert self.config_file.exists()
			assert self.docs_dir.exists()

			# Check config file contents
			config_data = yaml.safe_load(self.config_file.read_text())
			assert config_data == DEFAULT_CONFIG
		finally:
			parser_patcher.stop()
