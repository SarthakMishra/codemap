"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from codemap.config import DEFAULT_CONFIG
from codemap.utils.config_loader import ConfigError, ConfigLoader


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    return tmp_path / ".codemap.yml"


def test_default_config_loading() -> None:
    """Test loading default configuration when no config file is provided."""
    config_loader = ConfigLoader(None)
    assert config_loader.config == DEFAULT_CONFIG


def test_custom_config_loading(temp_config_file: Path) -> None:
    """Test loading custom configuration from file."""
    custom_config = {
        "token_limit": 2000,
        "exclude_patterns": ["*.test.js", "*.spec.py"],
        "include_patterns": ["*.py", "*.js", "*.ts"],
    }

    temp_config_file.write_text(yaml.dump(custom_config))
    config_loader = ConfigLoader(str(temp_config_file))

    assert config_loader.config["token_limit"] == 2000
    assert "*.test.js" in config_loader.config["exclude_patterns"]


def test_config_validation(temp_config_file: Path) -> None:
    """Test configuration validation."""
    invalid_config = {
        "token_limit": "not_a_number",
        "exclude_patterns": "not_a_list",
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
    assert "include_patterns" in config_loader.config
    assert isinstance(config_loader.config["include_patterns"], list)


def test_nonexistent_config_file() -> None:
    """Test handling of nonexistent config file."""
    with pytest.raises(FileNotFoundError, match="Config file not found:"):
        ConfigLoader("/nonexistent/config.yml")


def test_invalid_yaml_config(temp_config_file: Path) -> None:
    """Test handling of invalid YAML in config file."""
    temp_config_file.write_text("invalid: yaml: content: :")

    with pytest.raises(yaml.YAMLError, match="mapping values are not allowed here"):
        ConfigLoader(str(temp_config_file))
