"""Configuration loading and management for the CodeMap tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from codemap.config import DEFAULT_CONFIG


class ConfigError(TypeError):
    """Custom error for configuration validation."""

    INVALID_TOKEN_LIMIT = "token_limit must be an integer"  # noqa: S105
    INVALID_USE_GITIGNORE = "use_gitignore must be a boolean"
    INVALID_OUTPUT_DIR = "output_dir must be a string"


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
        if "token_limit" in config and not isinstance(config["token_limit"], int):
            raise ConfigError(ConfigError.INVALID_TOKEN_LIMIT)

        if "use_gitignore" in config and not isinstance(config["use_gitignore"], bool):
            raise ConfigError(ConfigError.INVALID_USE_GITIGNORE)

        if "output_dir" in config and not isinstance(config["output_dir"], str):
            raise ConfigError(ConfigError.INVALID_OUTPUT_DIR)

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
