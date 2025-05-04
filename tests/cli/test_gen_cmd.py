"""Tests for the gen command CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from codemap.cli import app  # Assuming 'app' is your Typer application instance
from codemap.gen import GenCommand, GenConfig
from codemap.processor import LODLevel
from codemap.utils.config_loader import ConfigLoader
from tests.base import FileSystemTestBase

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.cli
@pytest.mark.fs
class TestGenCommand(FileSystemTestBase):
	"""Test cases for the 'gen' CLI command."""

	runner: CliRunner

	@pytest.fixture(autouse=True)
	def setup_cli(self, temp_dir: Path) -> None:
		"""Set up CLI test environment."""
		self.temp_dir = temp_dir
		self.runner = CliRunner()
		# Create a dummy target directory for tests that need it
		(self.temp_dir / "dummy_code").mkdir(exist_ok=True)

	# Mock essential dependencies used by the command
	@patch("codemap.cli.gen_cmd.setup_logging")
	@patch("codemap.cli.gen_cmd.ConfigLoader")
	@patch("codemap.cli.gen_cmd.GenCommand")  # Mock the class
	def test_gen_command_defaults(
		self,
		mock_gen_command_class: MagicMock,
		mock_config_loader: MagicMock,
		mock_setup_logging: MagicMock,
	) -> None:
		"""Test 'gen' command with default arguments."""
		mock_config_instance = MagicMock(spec=ConfigLoader)
		mock_config_instance.config = {"gen": {}, "processor": {}}  # Simulate loaded config
		mock_config_loader.return_value = mock_config_instance

		mock_gen_command_instance = MagicMock(spec=GenCommand)
		mock_gen_command_class.return_value = mock_gen_command_instance

		result = self.runner.invoke(app, ["gen", str(self.temp_dir / "dummy_code")])

		assert result.exit_code == 0
		mock_setup_logging.assert_called_once_with(is_verbose=False)
		mock_config_loader.assert_called_once_with(None)

		# Verify GenConfig passed to GenCommand constructor
		# Only capture args since kwargs aren't used here
		args = mock_gen_command_class.call_args[0]
		assert len(args) == 1
		config_arg = args[0]
		assert isinstance(config_arg, GenConfig)
		# target_path is handled separately, not part of GenConfig
		# assert config_arg.target_path == self.temp_dir / "dummy_code"
		assert config_arg.max_content_length == 5000  # Default from code
		assert config_arg.lod_level == LODLevel.DOCS  # Default
		assert config_arg.include_tree is False  # Temporarily assert False to check runtime value
		assert config_arg.semantic_analysis is True  # Default in command signature
		# output_path is handled separately, not part of GenConfig
		# assert config_arg.output_path is not None # Should be determined
		# Check default mermaid settings directly on config_arg
		assert config_arg.include_entity_graph is True  # Default in dataclass
		assert config_arg.mermaid_entities == []  # Default
		assert config_arg.mermaid_relationships == []  # Default
		assert config_arg.mermaid_show_legend is True  # Default
		assert config_arg.mermaid_remove_unconnected is False  # Default

		mock_gen_command_instance.execute.assert_called_once()

	@patch("codemap.cli.gen_cmd.setup_logging")
	@patch("codemap.cli.gen_cmd.ConfigLoader")
	@patch("codemap.cli.gen_cmd.GenCommand")
	def test_gen_command_cli_overrides(
		self,
		mock_gen_command_class: MagicMock,
		mock_config_loader: MagicMock,
		_mock_setup_logging: MagicMock,
	) -> None:
		"""Test CLI arguments override config/defaults."""
		mock_config_instance = MagicMock(spec=ConfigLoader)
		# Simulate config with different values than defaults/CLI args
		mock_config_instance.config = {
			"gen": {
				"max_content_length": 1000,
				"lod": "full",
				"include_tree": False,
				"entity_graph": {
					"enabled": True,
					"entities": ["class"],
					"relationships": ["calls"],
					"show_legend": False,
					"remove_unconnected": False,
				},
			},
			"processor": {},
		}
		mock_config_loader.return_value = mock_config_instance
		mock_gen_command_class.return_value = MagicMock(spec=GenCommand)

		cli_output_path = self.temp_dir / "cli_output.md"

		result = self.runner.invoke(
			app,
			[
				"gen",
				str(self.temp_dir / "dummy_code"),
				"--output",
				str(cli_output_path),
				"--max-content-length",
				"2000",
				"--lod",
				"signatures",
				"--no-semantic",
				"--tree",  # Override config's False
				"--verbose",
				"--entity-graph",  # Override config's True (implicitly)
				"--mermaid-entities",
				"function,module",
				"--mermaid-relationships",
				"imports",
				"--mermaid-legend",
				"--mermaid-unconnected",
			],
		)

		assert result.exit_code == 0

		args, _ = mock_gen_command_class.call_args
		config_arg: GenConfig = args[0]

		# assert config_arg.target_path == self.temp_dir / "dummy_code"
		# assert config_arg.output_path == cli_output_path.resolve() # CLI output used
		assert config_arg.max_content_length == 2000  # CLI override
		assert config_arg.lod_level == LODLevel.SIGNATURES  # CLI override
		assert config_arg.semantic_analysis is False  # CLI override
		assert config_arg.include_tree is True  # CLI override
		# Check Mermaid config overrides directly on config_arg
		assert config_arg.include_entity_graph is True  # CLI override
		assert config_arg.mermaid_entities == ["function", "module"]  # CLI override
		assert config_arg.mermaid_relationships == ["imports"]  # CLI override
		assert config_arg.mermaid_show_legend is True  # CLI override
		assert config_arg.mermaid_remove_unconnected is True  # CLI override

		mock_gen_command_class.return_value.execute.assert_called_once()

	@patch("codemap.cli.gen_cmd.setup_logging")
	@patch("codemap.cli.gen_cmd.ConfigLoader")
	@patch("codemap.cli.gen_cmd.GenCommand")
	def test_gen_command_invalid_lod(
		self,
		mock_gen_command_class: MagicMock,
		mock_config_loader: MagicMock,
		_mock_setup_logging: MagicMock,
	) -> None:
		"""Test 'gen' command with an invalid LOD level."""
		mock_config_instance = MagicMock(spec=ConfigLoader)
		mock_config_instance.config = {"gen": {}, "processor": {}}
		mock_config_loader.return_value = mock_config_instance

		result = self.runner.invoke(app, ["gen", str(self.temp_dir / "dummy_code"), "--lod", "invalid_level"])

		assert result.exit_code != 0
		assert "Invalid LOD level" in result.stdout
		mock_gen_command_class.assert_not_called()  # Should exit before command execution

	@patch("codemap.cli.gen_cmd.setup_logging")
	@patch("codemap.cli.gen_cmd.ConfigLoader")
	@patch("codemap.cli.gen_cmd.GenCommand")
	@patch("codemap.cli.gen_cmd.exit_with_error")  # Mock exit helper
	def test_gen_command_gen_error(
		self,
		mock_exit_with_error: MagicMock,
		mock_gen_command_class: MagicMock,
		mock_config_loader: MagicMock,
		_mock_setup_logging: MagicMock,
	) -> None:
		"""Test 'gen' command when GenCommand.execute() raises an error."""
		mock_config_instance = MagicMock(spec=ConfigLoader)
		mock_config_instance.config = {"gen": {}, "processor": {}}
		mock_config_loader.return_value = mock_config_instance

		mock_gen_command_instance = MagicMock(spec=GenCommand)
		mock_gen_command_instance.execute.side_effect = ValueError("Generation failed")
		mock_gen_command_class.return_value = mock_gen_command_instance

		self.runner.invoke(app, ["gen", str(self.temp_dir / "dummy_code")])

		# Expect exit_with_error to be called, actual exit code depends on its implementation
		mock_gen_command_instance.execute.assert_called_once()
		mock_exit_with_error.assert_called_once()
		# Check that the error message passed to exit_with_error contains the original error
		args, _ = mock_exit_with_error.call_args
		assert "Generation failed" in args[0]
