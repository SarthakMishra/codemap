"""Configuration loading and management for the CodeMap tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from codemap.config import DEFAULT_CONFIG


class ConfigError(TypeError):
    """Custom error for configuration validation."""

    INVALID_TOKEN_LIMIT = "token_limit must be an integer"  # noqa: S105
    EXCLUDE_PATTERNS_ERROR = "exclude_patterns must be a list"
    INCLUDE_PATTERNS_ERROR = "include_patterns must be a list"
    OUTPUT_CONFIG_ERROR = "output configuration must be a dictionary"
    OUTPUT_DIRECTORY_ERROR = "output.directory must be a string"
    OUTPUT_FORMAT_ERROR = "output.filename_format must be a string"
    OUTPUT_TIMESTAMP_ERROR = "output.timestamp_format must be a string"


class ConfigLoader:
    """Handles loading and merging of default and user-provided configurations."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the config loader.

        Args:
            config_path: Path to a custom config file. Uses .codemap.yml if not provided.
        """
        self.config_path = config_path or ".codemap.yml"
        self.config_file = Path(self.config_path)
        self.project_root = self.config_file.parent if self.config_path != ".codemap.yml" else Path.cwd()
        self.config = self._load_config()

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration values.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ConfigError: If any configuration values are invalid.
        """
        if not isinstance(config.get("token_limit"), int):
            raise ConfigError(ConfigError.INVALID_TOKEN_LIMIT)

        if "exclude_patterns" in config and not isinstance(config["exclude_patterns"], list):
            raise ConfigError(ConfigError.EXCLUDE_PATTERNS_ERROR)

        if "include_patterns" in config and not isinstance(config["include_patterns"], list):
            raise ConfigError(ConfigError.INCLUDE_PATTERNS_ERROR)

        # Validate output configuration
        if "output" in config:
            if not isinstance(config["output"], dict):
                raise ConfigError(ConfigError.OUTPUT_CONFIG_ERROR)

            output_config = config["output"]
            if "directory" in output_config and not isinstance(output_config["directory"], str):
                raise ConfigError(ConfigError.OUTPUT_DIRECTORY_ERROR)

            if "filename_format" in output_config and not isinstance(output_config["filename_format"], str):
                raise ConfigError(ConfigError.OUTPUT_FORMAT_ERROR)

            if "timestamp_format" in output_config and not isinstance(output_config["timestamp_format"], str):
                raise ConfigError(ConfigError.OUTPUT_TIMESTAMP_ERROR)

    def _load_config(self) -> dict[str, Any]:
        """Load and merge configuration.

        Returns:
            Merged configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ConfigError: If config values are invalid.
        """
        # If we're looking for a default .codemap.yml in the current directory, try to find it in parent directories
        if self.config_path == ".codemap.yml":
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                config_file_path = current_dir / ".codemap.yml"
                if config_file_path.exists():
                    self.config_file = config_file_path
                    self.project_root = current_dir
                    break
                current_dir = current_dir.parent

        # If no config path was specified and not found in parent directories, use default config
        if self.config_path == ".codemap.yml" and not self.config_file.exists():
            return DEFAULT_CONFIG.copy()

        # If specific config path was provided but doesn't exist, raise error
        if not self.config_file.exists():
            msg = f"Config file not found: {self.config_path}"
            raise FileNotFoundError(msg)

        with self.config_file.open() as f:
            user_config = yaml.safe_load(f) or {}

        self._validate_config(user_config)
        return {**DEFAULT_CONFIG, **user_config}
