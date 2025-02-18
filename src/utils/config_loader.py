"""Configuration loading and management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from codemap.config import DEFAULT_CONFIG


class ConfigLoader:
    """Handles loading and merging of default and user-provided configurations."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the config loader.

        Args:
            config_path: Optional path to a custom config file. Uses .codemap.yml if not provided.
        """
        self.config_path = config_path or ".codemap.yml"
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load and merge configuration from file.

        Returns:
            Merged configuration dictionary with default values.
        """
        config_file = Path(self.config_path)
        if config_file.exists():
            with config_file.open() as f:
                user_config = yaml.safe_load(f)
            return {**DEFAULT_CONFIG, **(user_config or {})}
        return DEFAULT_CONFIG
