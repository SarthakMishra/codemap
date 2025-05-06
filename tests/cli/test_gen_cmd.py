"""Tests for the gen command CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from codemap.cli import app  # Assuming 'app' is your Typer application instance
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
	@patch("codemap.cli.gen_cmd._gen_command_impl")  # Mock the implementation function
	def test_gen_command_defaults(
		self,
		mock_gen_command_impl: MagicMock,
		_mock_config_loader: MagicMock,
		mock_setup_logging: MagicMock,
	) -> None:
		"""Test 'gen' command with default arguments."""
		result = self.runner.invoke(app, ["gen", str(self.temp_dir / "dummy_code")])

		assert result.exit_code == 0
		mock_setup_logging.assert_not_called()  # Not called in register_command

		# Verify _gen_command_impl was called with correct args
		mock_gen_command_impl.assert_called_once()
		_, kwargs = mock_gen_command_impl.call_args
		assert kwargs["path"] == self.temp_dir / "dummy_code"
		assert kwargs["lod_level_str"] == "docs"  # Default
		assert kwargs["semantic_analysis"] is True  # Default
		assert kwargs["is_verbose"] is False  # Default

	@patch("codemap.cli.gen_cmd.setup_logging")
	@patch("codemap.cli.gen_cmd.ConfigLoader")
	@patch("codemap.cli.gen_cmd._gen_command_impl")
	def test_gen_command_cli_overrides(
		self,
		mock_gen_command_impl: MagicMock,
		_mock_config_loader: MagicMock,
		_mock_setup_logging: MagicMock,
	) -> None:
		"""Test CLI arguments override config/defaults."""
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

		mock_gen_command_impl.assert_called_once()
		_, kwargs = mock_gen_command_impl.call_args

		assert kwargs["path"] == self.temp_dir / "dummy_code"
		assert kwargs["output"] == cli_output_path
		assert kwargs["max_content_length"] == 2000
		assert kwargs["lod_level_str"] == "signatures"
		assert kwargs["semantic_analysis"] is False
		assert kwargs["tree"] is True
		assert kwargs["is_verbose"] is True
		assert kwargs["entity_graph"] is True
		assert kwargs["mermaid_entities_str"] == "function,module"
		assert kwargs["mermaid_relationships_str"] == "imports"
		assert kwargs["mermaid_show_legend_flag"] is True
		assert kwargs["mermaid_remove_unconnected_flag"] is True

	@patch("codemap.cli.gen_cmd._gen_command_impl")
	def test_gen_command_invalid_lod(
		self,
		mock_gen_command_impl: MagicMock,
	) -> None:
		"""Test 'gen' command with an invalid LOD level."""

		# Implement a side effect that raises ValueError when given invalid lod_level_str
		def impl_side_effect(**kwargs):
			if kwargs.get("lod_level_str") == "invalid_level":
				msg = "Invalid LOD level"
				raise ValueError(msg)

		mock_gen_command_impl.side_effect = impl_side_effect

		result = self.runner.invoke(app, ["gen", str(self.temp_dir / "dummy_code"), "--lod", "invalid_level"])

		assert "Invalid LOD level" in result.stdout
		mock_gen_command_impl.assert_called_once()

	@patch("codemap.cli.gen_cmd._gen_command_impl")
	def test_gen_command_gen_error(
		self,
		mock_gen_command_impl: MagicMock,
	) -> None:
		"""Test 'gen' command when implementation raises an error."""
		mock_gen_command_impl.side_effect = ValueError("Generation failed")

		result = self.runner.invoke(app, ["gen", str(self.temp_dir / "dummy_code")])

		assert result.exit_code != 0
		assert "Generation failed" in result.stdout
		mock_gen_command_impl.assert_called_once()
