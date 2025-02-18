"""Configuration loading and management for the CodeMap tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from codemap.config import DEFAULT_CONFIG


class ConfigError(TypeError):
    """Custom error for configuration validation."""

    TOKEN_LIMIT_ERROR = "token_limit must be an integer"  # noqa: S105
    EXCLUDE_PATTERNS_ERROR = "exclude_patterns must be a list"
    INCLUDE_PATTERNS_ERROR = "include_patterns must be a list"


class ConfigLoader:
    """Handles loading and merging of default and user-provided configurations."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the config loader.

        Args:
            config_path: Path to a custom config file. Uses .codemap.yml if not provided.
        """
        self.config_path = config_path or ".codemap.yml"
        self.config = self._load_config()

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration values.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ConfigError: If any configuration values are invalid.
        """
        if not isinstance(config.get("token_limit"), int):
            raise ConfigError(ConfigError.TOKEN_LIMIT_ERROR)

        if "exclude_patterns" in config and not isinstance(config["exclude_patterns"], list):
            raise ConfigError(ConfigError.EXCLUDE_PATTERNS_ERROR)

        if "include_patterns" in config and not isinstance(config["include_patterns"], list):
            raise ConfigError(ConfigError.INCLUDE_PATTERNS_ERROR)

    def _load_config(self) -> dict[str, Any]:
        """Load and merge configuration.

        Returns:
            Merged configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ConfigError: If config values are invalid.
        """
        config_file = Path(self.config_path)

        # If no config path was specified, use default config
        if self.config_path == ".codemap.yml" and not config_file.exists():
            return DEFAULT_CONFIG.copy()

        # If specific config path was provided but doesn't exist, raise error
        if not config_file.exists():
            msg = f"Config file not found: {self.config_path}"
            raise FileNotFoundError(msg)

        with config_file.open() as f:
            user_config = yaml.safe_load(f) or {}

        self._validate_config(user_config)
        return {**DEFAULT_CONFIG, **user_config}
