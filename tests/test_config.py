"""Tests for configuration loading and validation."""

import os
from pathlib import Path

import pytest
import yaml

from codemap.config import DEFAULT_CONFIG
from codemap.utils.config_loader import ConfigError, ConfigLoader


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    return tmp_path / ".codemap.yml"


def test_default_config_loading(tmp_path: Path) -> None:
    """Test loading default configuration when no config file is provided."""
    # Change to a temporary directory to ensure we don't pick up any .codemap.yml
    old_cwd = Path.cwd()
    os.chdir(str(tmp_path))
    try:
        config_loader = ConfigLoader(None)
        # Compare each section individually for better error messages
        for key in DEFAULT_CONFIG:
            assert config_loader.config[key] == DEFAULT_CONFIG[key], f"Mismatch in {key} section"
    finally:
        os.chdir(old_cwd)


def test_custom_config_loading(temp_config_file: Path) -> None:
    """Test loading custom configuration from file."""
    custom_config = {
        "token_limit": 2000,
        "use_gitignore": False,
        "output_dir": "custom_docs",
    }

    temp_config_file.write_text(yaml.dump(custom_config))
    config_loader = ConfigLoader(str(temp_config_file))

    assert config_loader.config["token_limit"] == 2000
    assert config_loader.config["use_gitignore"] is False
    assert config_loader.config["output_dir"] == "custom_docs"


def test_config_validation(temp_config_file: Path) -> None:
    """Test configuration validation."""
    invalid_config = {
        "token_limit": "not_a_number",
        "use_gitignore": "not_a_boolean",
    }

    temp_config_file.write_text(yaml.dump(invalid_config))

    with pytest.raises(ConfigError, match="token_limit must be an integer"):
        ConfigLoader(str(temp_config_file))


def test_config_merging(temp_config_file: Path) -> None:
    """Test merging custom config with default config."""
    partial_config = {
        "token_limit": 3000,
    }

    temp_config_file.write_text(yaml.dump(partial_config))
    config_loader = ConfigLoader(str(temp_config_file))

    assert config_loader.config["token_limit"] == 3000
    assert "use_gitignore" in config_loader.config
    assert config_loader.config["output_dir"] == "documentation"


def test_nonexistent_config_file() -> None:
    """Test handling of nonexistent config file."""
    with pytest.raises(FileNotFoundError, match="Config file not found:"):
        ConfigLoader("/nonexistent/config.yml")


def test_invalid_yaml_config(temp_config_file: Path) -> None:
    """Test handling of invalid YAML in config file."""
    temp_config_file.write_text("invalid: yaml: content: :")

    with pytest.raises(yaml.YAMLError, match="mapping values are not allowed here"):
        ConfigLoader(str(temp_config_file))
