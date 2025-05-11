"""Tests for configuration loading and validation."""

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from codemap.config import AppConfigSchema, ConfigError, ConfigLoader
from codemap.config.config_schema import CommitSchema
from tests.base import FileSystemTestBase
from tests.helpers import create_file_content


@pytest.fixture
def mock_yaml_loader() -> Generator[Mock, None, None]:
	"""Return a mock for the YAML safe_load function to use in config loader tests."""
	with patch("codemap.config.config_loader.yaml.safe_load") as mock_loader:
		yield mock_loader


@pytest.mark.unit
class TestConfigLoader(FileSystemTestBase):
	"""Test cases for configuration loading and validation."""

	def setup_method(self) -> None:
		"""Set up test environment."""
		# Ensure temp_dir exists before using it
		if not hasattr(self, "temp_dir"):
			# Create a temporary directory if it doesn't exist
			import tempfile

			self.temp_dir = Path(tempfile.mkdtemp())

		self.config_file = self.temp_dir / ".codemap.yml"
		self.old_cwd = Path.cwd()

	def teardown_method(self) -> None:
		"""Clean up test environment."""
		# Ensure we return to the original directory
		if hasattr(self, "old_cwd"):
			os.chdir(self.old_cwd)

		# Clean up temp directory if we created it ourselves
		if hasattr(self, "temp_dir") and not hasattr(self, "_temp_dir_from_fixture"):
			import shutil

			shutil.rmtree(self.temp_dir)

	# Add a fixture hook to detect when temp_dir is set by the fixture
	@pytest.fixture(autouse=True)
	def _setup_temp_dir_marker(self, temp_dir: Path) -> None:
		self.temp_dir = temp_dir
		self._temp_dir_from_fixture = True

	def test_default_config_loading(self) -> None:
		"""Test loading default configuration when no config file is provided."""
		# Change to a temporary directory to ensure we don't pick up any .codemap.yml
		os.chdir(str(self.temp_dir))

		with (
			patch("codemap.git.utils.GitRepoContext.get_repo_root", return_value=self.temp_dir),
			patch.object(ConfigLoader, "_get_config_file", return_value=None),
			patch.object(ConfigLoader, "repo_root", self.temp_dir, create=True),
		):
			config_loader = ConfigLoader()
			# Verify the config is an instance of AppConfigSchema
			assert isinstance(config_loader._app_config, AppConfigSchema)
			# Verify default values from schema are present
			assert config_loader._app_config.llm.model == "openai:gpt-4o-mini"
			assert config_loader._app_config.embedding.model_name == "minishlab/potion-base-8M"

	def test_custom_config_loading(self) -> None:
		"""Test loading custom configuration from file."""
		custom_config = {"llm": {"model": "openai:gpt-4", "temperature": 0.7}, "gen": {"output_dir": "custom_docs"}}

		create_file_content(self.config_file, yaml.dump(custom_config))

		with (
			patch("codemap.git.utils.GitRepoContext.get_repo_root", return_value=self.temp_dir),
			patch.object(ConfigLoader, "_parse_yaml_file", return_value=custom_config),
			patch.object(ConfigLoader, "_get_config_file", return_value=self.config_file),
			patch.object(ConfigLoader, "repo_root", self.temp_dir, create=True),
		):
			config_loader = ConfigLoader()
			# Verify custom values are loaded
			assert config_loader._app_config.llm.model == "openai:gpt-4"
			assert config_loader._app_config.llm.temperature == 0.7
			assert config_loader._app_config.gen.output_dir == "custom_docs"
			# Verify default values are still present for unspecified fields
			assert config_loader._app_config.gen.use_gitignore is True

	def test_invalid_yaml_config(self) -> None:
		"""Test handling of invalid YAML in config file."""
		create_file_content(self.config_file, "invalid: yaml: content: :")

		with (
			pytest.raises(ConfigError),
			patch("codemap.git.utils.GitRepoContext.get_repo_root", return_value=self.temp_dir),
			patch.object(ConfigLoader, "_get_config_file", return_value=self.config_file),
			patch.object(ConfigLoader, "repo_root", self.temp_dir, create=True),
			patch.object(ConfigLoader, "_parse_yaml_file", side_effect=yaml.YAMLError("Invalid YAML")),
		):
			ConfigLoader()


@pytest.mark.unit
class TestAppConfigSchema:
	"""Test cases for AppConfigSchema functionality."""

	def test_commit_config_defaults(self) -> None:
		"""Test default values for commit configuration."""
		# Create config with defaults
		config = AppConfigSchema()

		# Check default values
		assert config.commit.bypass_hooks is False
		assert config.commit.strategy == "file"
		assert config.commit.use_lod_context is True

	def test_commit_config_override(self) -> None:
		"""Test overriding commit configuration values."""
		# Create config with custom commit settings
		custom_commit = CommitSchema(bypass_hooks=True, strategy="semantic")
		config = AppConfigSchema(commit=custom_commit)

		# Check custom values
		assert config.commit.bypass_hooks is True
		assert config.commit.strategy == "semantic"
		# Check defaults for values not overridden
		assert config.commit.use_lod_context is True
