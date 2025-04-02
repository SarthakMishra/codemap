"""Configuration loading utilities for the CodeMap tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from codemap.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ConfigError(TypeError):
    """Custom error for configuration validation."""

    INVALID_TOKEN_LIMIT = "token_limit must be an integer"  # noqa: S105
    INVALID_USE_GITIGNORE = "use_gitignore must be a boolean"
    INVALID_OUTPUT_DIR = "output_dir must be a string"


class ConfigLoader:
    """Load and validate configuration for the CodeMap tool."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the configuration loader.

        Args:
            config_path: Optional path to the configuration file.
                         If None, will search for .codemap.yml in the current
                         directory and parent directories.
        """
        self._config: dict[str, Any] = {}
        self.config_file_dir: Path | None = None
        self._load_config(config_path)

    def _find_config_file(self) -> str | None:
        """Find a configuration file by searching current and parent directories.

        Returns:
            Path to the configuration file, or None if not found.
        """
        # Start in current directory
        current_dir = Path.cwd()
        max_depth = 10  # Allow more levels to find the repo root

        # Try current directory and parents
        for _ in range(max_depth):
            config_path = current_dir / ".codemap.yml"
            if config_path.exists():
                logger.debug("Found config file at: %s", config_path)
                return str(config_path.resolve())

            # Go up one level
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # We've reached the root
                break
            current_dir = parent_dir

        logger.debug("No config file found in directory hierarchy")
        return None

    def _load_config(self, config_path: str | None) -> None:
        """Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration file.
        """
        # Start with default config
        self._config = DEFAULT_CONFIG.copy()

        # Try to find a config file if not specified
        config_file = None
        if config_path:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning("Specified config file not found: %s", config_path)
                config_file = None
        else:
            found_path = self._find_config_file()
            if found_path:
                config_file = Path(found_path)
                logger.debug("Using config file: %s", found_path)

        # Load from file if available
        if config_file and config_file.exists():
            try:
                with config_file.open(encoding="utf-8") as f:
                    file_config = yaml.safe_load(f)

                # Store the directory containing the config file
                self.config_file_dir = config_file.parent.resolve()
                logger.debug("Config file directory: %s", self.config_file_dir)

                if file_config:
                    # Update default config with file config
                    self._config.update(file_config)
                    logger.debug("Loaded config from %s", config_file)
            except (yaml.YAMLError, OSError) as e:
                logger.warning("Error loading config file: %s", e)
        else:
            logger.debug("Using default configuration")

    @property
    def config(self) -> dict[str, Any]:
        """Get the configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config

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
