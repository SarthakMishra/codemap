"""
Configuration loader for CodeMap.

This module provides functionality for loading and managing
configuration settings.

"""

import logging
import os
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from xdg.BaseDirectory import xdg_config_home

from codemap.config import DEFAULT_CONFIG

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for config values with better type safety
T = TypeVar("T")

# Constant for minimum number of parts in environment variable
MIN_ENV_VAR_PARTS = 2

# Type for configuration values
ConfigValue = str | int | float | bool | dict[str, Any] | list[Any] | None


class ConfigError(Exception):
	"""Exception raised for configuration errors."""


class ConfigLoader:
	"""
	Loads and manages configuration for CodeMap.

	This class handles loading configuration from files, environment
	variables, and default values, with proper error handling and path
	resolution.

	"""

	_instance = None  # For singleton pattern

	@classmethod
	def get_instance(
		cls, config_file: str | None = None, reload: bool = False, repo_root: Path | None = None
	) -> "ConfigLoader":
		"""
		Get the singleton instance of ConfigLoader.

		Args:
		        config_file: Path to configuration file (optional)
		        reload: Whether to reload config even if already loaded
		        repo_root: Repository root path (optional)

		Returns:
		        ConfigLoader: Singleton instance

		"""
		if cls._instance is None or reload:
			cls._instance = cls(config_file, repo_root=repo_root)
		return cls._instance

	def __init__(self, config_file: str | None = None, repo_root: Path | None = None) -> None:
		"""
		Initialize the configuration loader.

		Args:
		        config_file: Path to configuration file (optional)
		        repo_root: Repository root path (optional)

		"""
		self.config: dict[str, Any] = {}
		self.repo_root = repo_root
		self.config_file = self._resolve_config_file(config_file)
		self.load_config()

	def _resolve_config_file(self, config_file: str | None = None) -> Path | None:
		"""
		Resolve the configuration file path.

		If a config file is specified, use that. Otherwise, look in standard locations:
		1. ./.codemap.yml in the current directory
		2. $XDG_CONFIG_HOME/codemap/config.yml
		3. ~/.config/codemap/config.yml (fallback if XDG_CONFIG_HOME not set)

		Args:
		        config_file: Explicitly provided config file path (optional)

		Returns:
		        Optional[Path]: Resolved config file path or None if no suitable file found

		"""
		if config_file:
			path = Path(config_file).expanduser().resolve()
			if path.exists():
				return path
			logger.warning("Specified config file not found: %s", path)
			return path  # Return it anyway, we'll handle the missing file in load_config

		# Try current directory
		local_config = Path(".codemap.yml")
		if local_config.exists():
			return local_config

		# Try XDG config path
		xdg_config_dir = Path(xdg_config_home) / "codemap"
		xdg_config_file = xdg_config_dir / "config.yml"
		if xdg_config_file.exists():
			return xdg_config_file

		# As a last resort, try the legacy ~/.codemap location
		legacy_config = Path.home() / ".codemap" / "config.yml"
		if legacy_config.exists():
			return legacy_config

		# If we get here, no config file was found
		return None

	def load_config(self) -> dict[str, Any]:
		"""
		Load configuration from file and apply environment variable overrides.

		Returns:
		        Dict[str, Any]: Loaded configuration

		Raises:
		        ConfigError: If configuration file exists but cannot be loaded

		"""
		# Start with default configuration
		self.config = DEFAULT_CONFIG.copy()

		# Try to load from file if available
		if self.config_file:
			try:
				if self.config_file.exists():
					with self.config_file.open(encoding="utf-8") as f:
						file_config = yaml.safe_load(f)
						if file_config:
							self._merge_configs(self.config, file_config)
					logger.info("Loaded configuration from %s", self.config_file)
				else:
					logger.warning("Configuration file not found: %s", self.config_file)
			except (OSError, yaml.YAMLError) as e:
				error_msg = f"Error loading configuration from {self.config_file}: {e}"
				logger.exception(error_msg)
				raise ConfigError(error_msg) from e

		# Apply environment variable overrides
		self._apply_env_overrides()

		# Resolve any paths in the config
		self._resolve_paths()

		return self.config

	def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> None:
		"""
		Recursively merge two configuration dictionaries.

		Args:
		        base: Base configuration dictionary to merge into
		        override: Override configuration to apply

		"""
		for key, value in override.items():
			if isinstance(value, dict) and key in base and isinstance(base[key], dict):
				self._merge_configs(base[key], value)
			else:
				base[key] = value

	def _apply_env_overrides(self) -> None:
		"""Apply environment variable overrides to configuration."""
		# Look for environment variables in the form CODEMAP_SECTION_KEY
		for env_var, value in os.environ.items():
			if env_var.startswith("CODEMAP_"):
				parts = env_var.lower().split("_")[1:]
				if len(parts) >= MIN_ENV_VAR_PARTS:
					section, key = parts[0], "_".join(parts[1:])

					# Try to convert value to appropriate type
					if value.lower() in ("true", "yes", "1"):
						typed_value = True
					elif value.lower() in ("false", "no", "0"):
						typed_value = False
					else:
						try:
							typed_value = int(value)
						except ValueError:
							try:
								typed_value = float(value)
							except ValueError:
								typed_value = value

					# Ensure section exists
					if section not in self.config:
						self.config[section] = {}

					# Apply override
					self.config[section][key] = typed_value
					logger.debug("Applied environment override %s: %s", env_var, typed_value)

	def _resolve_paths(self) -> None:
		"""Resolve and expand any paths in the configuration."""
		# Define paths that need resolution
		path_keys = [
			("daemon", "pid_file"),
			("daemon", "log_file"),
			("storage", "data_dir"),
		]

		for section, key in path_keys:
			if section in self.config and key in self.config[section]:
				path_str = self.config[section][key]
				if isinstance(path_str, str):
					resolved_path = Path(path_str).expanduser().resolve()
					self.config[section][key] = str(resolved_path)

	def get(self, key: str, default: T = None) -> T:
		"""
		Get a configuration value, optionally with a section.

		Examples:
		        # Get a top-level key
		        config.get("daemon")

		        # Get a nested key with dot notation
		        config.get("daemon.host")

		Args:
		        key: Configuration key, can include dots for nested access
		        default: Default value if key not found

		Returns:
		        T: Configuration value or default

		"""
		parts = key.split(".")

		# Start with the whole config
		current = self.config

		# Traverse the parts
		for part in parts:
			if isinstance(current, dict) and part in current:
				current = current[part]
			else:
				return default

		return cast("T", current)

	def set(self, key: str, value: ConfigValue) -> None:
		"""
		Set a configuration value.

		Args:
		        key: Configuration key, can include dots for nested access
		        value: Value to set

		"""
		parts = key.split(".")

		# Start with the whole config
		current = self.config

		# Traverse to the parent of the leaf
		for _i, part in enumerate(parts[:-1]):
			if part not in current:
				current[part] = {}
			elif not isinstance(current[part], dict):
				# Convert to dict if it wasn't already
				current[part] = {}
			current = current[part]

		# Set the leaf value
		current[parts[-1]] = value

	def save(self, config_file: str | None = None) -> None:
		"""
		Save the current configuration to a file.

		Args:
		        config_file: Path to save configuration to (optional, defaults to current config_file)

		Raises:
		        ConfigError: If configuration cannot be saved

		"""
		save_path = Path(config_file) if config_file else self.config_file

		if not save_path:
			error_msg = "No configuration file specified for saving"
			logger.error(error_msg)
			raise ConfigError(error_msg)

		# Ensure parent directory exists
		save_path.parent.mkdir(parents=True, exist_ok=True)

		try:
			with save_path.open("w", encoding="utf-8") as f:
				yaml.dump(self.config, f, default_flow_style=False)
			logger.info("Configuration saved to %s", save_path)
		except OSError as e:
			error_msg = f"Error saving configuration to {save_path}: {e}"
			logger.exception(error_msg)
			raise ConfigError(error_msg) from e

	# Helper methods for specific configuration sections
	def get_commit_hooks(self) -> list[dict[str, Any]]:
		"""
		Get commit hooks configuration.

		Returns:
		        List[Dict[str, Any]]: List of commit hooks

		"""
		return self.get("git.commit_hooks", self.get("commit.hooks", []))

	def get_bypass_hooks(self) -> bool:
		"""
		Get whether to bypass git hooks.

		Returns:
		        bool: True if git hooks should be bypassed, False otherwise

		"""
		return self.get("git.bypass_hooks", self.get("commit.bypass_hooks", False))

	def get_commit_convention(self) -> dict[str, Any]:
		"""
		Get commit convention configuration.

		Returns:
		        Dict[str, Any]: Commit convention configuration

		"""
		convention = self.get("commit.convention", {})

		# Ensure 'types' is always present with a default value if missing
		if "types" not in convention:
			convention["types"] = DEFAULT_CONFIG["commit"]["convention"]["types"]

		return convention

	def get_workflow_strategy(self) -> str:
		"""
		Get the workflow strategy configuration.

		Returns:
		        str: Workflow strategy name

		"""
		return self.get("git.workflow_strategy", "default")

	def get_pr_config(self) -> dict[str, Any]:
		"""
		Get PR configuration.

		Returns:
		        Dict[str, Any]: PR configuration

		"""
		return self.get("git.pr", {})

	def get_content_generation_config(self) -> dict[str, Any]:
		"""
		Get content generation configuration.

		Returns:
		        Dict[str, Any]: Content generation configuration

		"""
		return self.get("generation.content", {})

	def get_llm_config(self) -> dict[str, Any]:
		"""
		Get LLM configuration.

		Returns:
		        Dict[str, Any]: LLM configuration

		"""
		# Check the standalone llm config first
		llm_config = self.get("llm", {})

		# If no standalone config, check the nested commit.llm config
		if not llm_config:
			llm_config = self.get("commit.llm", {})

		# If still empty, use the default config
		if not llm_config:
			llm_config = DEFAULT_CONFIG["commit"]["llm"]

		# Ensure we have the proper model format with a provider
		model = llm_config.get("model", DEFAULT_CONFIG["commit"]["llm"]["model"])
		if model and "/" not in model:
			# Add openai/ prefix if provider is missing
			llm_config["model"] = f"openai/{model}"

		return llm_config

	def get_server_config(self) -> dict[str, Any]:
		"""
		Get consolidated server configuration.

		Returns:
		        dict[str, Any]: Server configuration

		"""
		return self.get("server", {})

	def get_daemon_config(self) -> dict[str, Any]:
		"""
		Get daemon configuration from the server section.

		Returns:
		        dict[str, Any]: Daemon configuration

		"""
		server_config = self.get_server_config()

		# Extract daemon-specific configuration from server
		return {
			"host": server_config.get("host", "127.0.0.1"),
			"port": server_config.get("port", 8765),
			"log_level": server_config.get("log_level", "info"),
			"pid_file": server_config.get("pid_file", "~/.codemap/daemon.pid"),
			"log_file": server_config.get("log_file", "~/.codemap/daemon.log"),
			"auto_start": server_config.get("auto_start", False),
		}

	def get_api_config(self) -> dict[str, Any]:
		"""
		Get API configuration from the server section.

		Returns:
		        dict[str, Any]: API configuration

		"""
		server_config = self.get_server_config()

		# Extract API-specific configuration from server
		return {
			"host": server_config.get("host", "127.0.0.1"),
			"port": server_config.get("port", 8765),
			"allow_cors": server_config.get("allow_cors", True),
			"cors_origins": server_config.get("cors_origins", ["*"]),
			"max_connections": server_config.get("max_connections", 10),
		}
