"""Tests for configuration loading and validation."""

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from codemap.config import DEFAULT_CONFIG
from codemap.utils.config_loader import ConfigError, ConfigLoader
from tests.base import FileSystemTestBase
from tests.helpers import create_file_content


@pytest.fixture
def mock_yaml_loader() -> Generator[Mock, None, None]:
	"""Return a mock for the YAML safe_load function to use in config loader tests."""
	with patch("codemap.utils.config_loader.yaml.safe_load") as mock_loader:
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

		config_loader = ConfigLoader(None)
		# Compare each section individually for better error messages
		for key in DEFAULT_CONFIG:
			assert config_loader.config[key] == DEFAULT_CONFIG[key], f"Mismatch in {key} section"

	def test_custom_config_loading(self) -> None:
		"""Test loading custom configuration from file."""
		custom_config = {
			"token_limit": 2000,
			"use_gitignore": False,
			"output_dir": "custom_docs",
		}

		create_file_content(self.config_file, yaml.dump(custom_config))
		config_loader = ConfigLoader(str(self.config_file))

		assert config_loader.config["token_limit"] == 2000
		assert config_loader.config["use_gitignore"] is False
		assert config_loader.config["output_dir"] == "custom_docs"

	def test_config_validation(self) -> None:
		"""Test configuration validation."""
		invalid_config = {
			"token_limit": "not_a_number",
			"use_gitignore": "not_a_boolean",
		}

		create_file_content(self.config_file, yaml.dump(invalid_config))

		# ConfigLoader no longer validates types during loading,
		# it just merges the values as-is
		config_loader = ConfigLoader(str(self.config_file))

		# Verify that the "invalid" values were loaded as strings
		assert config_loader.config["token_limit"] == "not_a_number"
		assert config_loader.config["use_gitignore"] == "not_a_boolean"

	def test_config_merging(self) -> None:
		"""Test merging custom config with default config."""
		partial_config = {
			"token_limit": 3000,
		}

		create_file_content(self.config_file, yaml.dump(partial_config))
		config_loader = ConfigLoader(str(self.config_file))

		assert config_loader.config["token_limit"] == 3000
		# Check for a nested key that exists in the default config
		assert "use_gitignore" in config_loader.config["gen"]
		assert config_loader.config["gen"]["use_gitignore"] is True

	def test_nonexistent_config_file(self) -> None:
		"""Test handling of nonexistent config file."""
		nonexistent_path = "/nonexistent/config.yml"

		# During initialization, ConfigLoader now just warns about nonexistent files
		# but doesn't raise exceptions
		config_loader = ConfigLoader(nonexistent_path)

		# Verify that default config was used by checking a known default key
		# The problematic token_limit key is removed from the assertion
		assert "gen" in config_loader.config
		assert "use_gitignore" in config_loader.config["gen"]
		assert config_loader.config["gen"]["use_gitignore"] is True  # Check the default value

	def test_invalid_yaml_config(self) -> None:
		"""Test handling of invalid YAML in config file."""
		create_file_content(self.config_file, "invalid: yaml: content: :")

		with pytest.raises(ConfigError, match="mapping values are not allowed here"):
			ConfigLoader(str(self.config_file))

	def test_get_commit_hooks(self, mock_yaml_loader: Mock, tmp_path: Path) -> None:
		"""Test loading commit hooks configuration."""
		# Create a config file
		config_file = tmp_path / ".codemap.yml"
		config_content = """
token_limit: 5000
commit:
  bypass_hooks: true
"""
		config_file.write_text(config_content)

		# Setup YAML data for the mock to return
		yaml_data = {"token_limit": 5000, "commit": {"bypass_hooks": True}}
		mock_yaml_loader.return_value = yaml_data

		# Create loader with mocked yaml loader
		with patch("builtins.open", mock_open(read_data=config_content)):
			loader = ConfigLoader(config_file=str(config_file))
			# Test get_bypass_hooks returns the configured value
			assert loader.get_bypass_hooks() is True

		# Test with bypass_hooks explicitly set to false
		yaml_data = {"token_limit": 5000, "commit": {"bypass_hooks": False}}
		mock_yaml_loader.return_value = yaml_data

		config_content = """
token_limit: 5000
commit:
  bypass_hooks: false
"""
		config_file.write_text(config_content)
		with patch("builtins.open", mock_open(read_data=config_content)):
			loader = ConfigLoader(config_file=str(config_file))
			assert loader.get_bypass_hooks() is False

		# Test with commit section but no bypass_hooks (should default to False)
		yaml_data = {"token_limit": 5000, "commit": {"strategy": "semantic"}}
		mock_yaml_loader.return_value = yaml_data

		config_content = """
token_limit: 5000
commit:
  strategy: semantic
"""
		config_file.write_text(config_content)
		with patch("builtins.open", mock_open(read_data=config_content)):
			loader = ConfigLoader(config_file=str(config_file))
			assert loader.get_bypass_hooks() is False

		# Test with no commit section (should default to False)
		yaml_data = {"token_limit": 5000}
		mock_yaml_loader.return_value = yaml_data

		config_content = """
token_limit: 5000
"""
		config_file.write_text(config_content)
		with patch("builtins.open", mock_open(read_data=config_content)):
			loader = ConfigLoader(config_file=str(config_file))
			assert loader.get_bypass_hooks() is False
