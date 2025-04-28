"""
Configuration management utilities for CodeMap.

This module provides functions and classes to manage CodeMap
configuration across different scopes (global, project, etc.).

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from codemap.utils.directory_manager import get_directory_manager

logger = logging.getLogger(__name__)
console = Console()


class ConfigManager:
	"""Manages CodeMap configuration at different scopes."""

	def __init__(self) -> None:
		"""Initialize the configuration manager."""
		self.dir_manager = get_directory_manager()
		self.dir_manager.ensure_directories()

		# Default configuration
		self.default_config = {
			# Basic settings
			"token_limit": 10000,
			"use_gitignore": True,
			"output_dir": "documentation",
			"max_content_length": 5000,
			# LLM configuration
			"llm": {
				"provider": "openai",
				"model": "openai/gpt-4o-mini",
			},
			# Processor configuration
			"processor": {
				"enabled": True,
				"enable_lsp": True,
				"max_workers": 4,
				"cache_dir": ".codemap_cache",
				"embedding_model": "Qodo/Qodo-Embed-1-1.5B",
				"batch_size": 32,
				"ignored_patterns": [
					"**/.git/**",
					"**/__pycache__/**",
					"**/.venv/**",
					"**/node_modules/**",
					"**/.codemap_cache/**",
					"**/*.pyc",
					"**/dist/**",
					"**/build/**",
				],
			},
			# Commit feature configuration
			"commit": {
				"strategy": "semantic",
				"llm": {
					"model": "openai/gpt-4o-mini",
				},
				"convention": {
					"types": [
						"feat",
						"fix",
						"docs",
						"style",
						"refactor",
						"perf",
						"test",
						"build",
						"ci",
						"chore",
					],
					"scopes": [],
					"max_length": 72,
				},
			},
		}

		# Load configurations
		self._global_config = self._load_global_config()
		self._project_config = None
		self._project_path = None

	def set_project(self, project_path: str | Path) -> None:
		"""
		Set the current project and load its configuration.

		Args:
		    project_path: Path to the project

		"""
		self._project_path = Path(project_path).resolve()
		self.dir_manager.set_project_dir(self._project_path)
		self._project_config = self._load_project_config()

	def get_config(self, scope: str = "global") -> dict[str, Any]:
		"""
		Get the configuration for a specific scope.

		Args:
		        scope: Configuration scope (global, project, or merged)

		Returns:
		        Dict[str, Any]: Configuration data

		"""
		if scope == "global":
			return self._global_config.copy()
		if scope == "project":
			if self._project_config is None:
				logger.warning("No project configuration loaded")
				return {}
			return self._project_config.copy()
		if scope == "merged":
			config = self._global_config.copy()
			if self._project_config:
				config.update(self._project_config)
			return config
		logger.warning("Unknown configuration scope: %s", scope)
		return {}

	def update_config(self, scope: str, config_updates: dict[str, Any]) -> bool:
		"""
		Update configuration for the specified scope.

		Args:
		        scope: Scope of configuration to update ('global' or 'project')
		        config_updates: Dictionary of configuration updates

		Returns:
		        bool: True if successful, False otherwise

		"""
		if scope == "global":
			self._global_config.update(config_updates)
			return self._save_global_config()
		if scope == "project":
			if self._project_config is None:
				logger.warning("No project configuration loaded")
				return False
			self._project_config.update(config_updates)
			return self._save_project_config()
		logger.warning("Cannot update config for scope: %s", scope)
		return False

	def initialize_project_config(self, config: dict[str, Any] | None = None) -> bool:
		"""
		Initialize project configuration.

		Args:
		    config: Initial configuration (uses default if None)

		Returns:
		    True if initialization was successful, False otherwise

		"""
		if self._project_path is None:
			logger.error("Cannot initialize project config: No project set")
			return False

		# Start with defaults, then override with provided config
		project_config = self.default_config.copy()
		if config:
			project_config.update(config)

		self._project_config = project_config
		return self._save_project_config()

	def _load_global_config(self) -> dict[str, Any]:
		"""
		Load global configuration.

		Returns:
		    Global configuration dictionary

		"""
		config_path = self.dir_manager.config_dir / "settings.yml"
		if not config_path.exists():
			# Create default global config if it doesn't exist
			config = {}
			try:
				config_path.parent.mkdir(parents=True, exist_ok=True)
				with config_path.open("w") as f:
					yaml.dump(config, f, default_flow_style=False)
			except Exception:
				logger.exception("Failed to create global config")
			return config

		try:
			with config_path.open() as f:
				return yaml.safe_load(f) or {}
		except Exception:
			logger.exception("Failed to load global config")
			return {}

	def _save_global_config(self) -> bool:
		"""
		Save global configuration.

		Returns:
		    True if save was successful, False otherwise

		"""
		config_path = self.dir_manager.config_dir / "settings.yml"
		try:
			config_path.parent.mkdir(parents=True, exist_ok=True)
			with config_path.open("w") as f:
				yaml.dump(self._global_config, f, default_flow_style=False)
			return True
		except Exception:
			logger.exception("Failed to save global config")
			return False

	def _load_project_config(self) -> dict[str, Any]:
		"""
		Load project configuration.

		Returns:
		    Project configuration dictionary

		"""
		if self._project_path is None:
			return {}

		config_path = self._project_path / ".codemap.yml"
		if not config_path.exists():
			return {}

		try:
			with config_path.open() as f:
				return yaml.safe_load(f) or {}
		except Exception:
			logger.exception("Failed to load project config")
			return {}

	def _save_project_config(self) -> bool:
		"""
		Save project configuration.

		Returns:
		    True if save was successful, False otherwise

		"""
		if self._project_path is None or self._project_config is None:
			return False

		config_path = self._project_path / ".codemap.yml"
		try:
			with config_path.open("w") as f:
				yaml.dump(self._project_config, f, default_flow_style=False)
			return True
		except Exception:
			logger.exception("Failed to save project config")
			return False

	def _create_default_global_config(self) -> dict[str, Any]:
		"""
		Create a default global configuration.

		Returns:
		        Dict[str, Any]: Default global configuration

		"""
		global_dir = self.dir_manager.config_dir / "codemap"
		config_path = global_dir / "config.yml"

		# Create a default configuration
		config = {
			"user": {
				"name": "",  # Default empty, will be filled by user
				"email": "",  # Default empty, will be filled by user
			},
			"editor": {
				"path": "",  # Default empty, will be detected at runtime
			},
			"system": {
				"cache_dir": str(self.dir_manager.cache_dir),
				"data_dir": str(self.dir_manager.user_data_dir),
			},
		}

		# Save the default configuration
		if not config_path.exists():
			try:
				config_path.parent.mkdir(parents=True, exist_ok=True)
				with config_path.open("w") as f:
					yaml.dump(config, f, default_flow_style=False)
			except Exception:
				logger.exception("Failed to create global config")
			return config

		try:
			with config_path.open() as f:
				return yaml.safe_load(f) or {}
		except Exception:
			logger.exception("Failed to load global config")
			return {}


def get_config_manager() -> ConfigManager:
	"""
	Get an instance of the ConfigManager.

	Returns:
	    ConfigManager instance

	"""
	return ConfigManager()
