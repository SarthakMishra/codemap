"""
Configuration loader for CodeMap.

This module provides functionality for loading and managing
configuration settings.

"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from xdg.BaseDirectory import xdg_config_home

from codemap.config.config_schema import AppConfigSchema

# For type checking only
if TYPE_CHECKING:
	from pydantic import BaseModel

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Type for configuration values
if TYPE_CHECKING:
	from pydantic import BaseModel

	ConfigValue = BaseModel | dict[str, Any] | list[Any] | str | int | float | bool | None
else:
	ConfigValue = Any


class ConfigError(Exception):
	"""Exception raised for configuration errors."""


class ConfigFileNotFoundError(ConfigError):
	"""Exception raised when configuration file is not found."""


class ConfigParsingError(ConfigError):
	"""Exception raised when configuration file cannot be parsed."""


class ConfigLoader:
	"""
	Loads and manages configuration for CodeMap using Pydantic schemas.

	This class handles loading configuration from files, applying defaults
	from Pydantic models, with proper error handling and path
	resolution.

	"""

	_instance = None  # For singleton pattern

	@classmethod
	def get_instance(
		cls, config_file: Path | None = None, reload: bool = False, repo_root: Path | None = None
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
		if cls._instance is None:
			cls._instance = cls(config_file, repo_root=repo_root)
		elif reload:
			# Update existing instance if reload is requested
			cls._instance.reload_config(config_file, repo_root)
		return cls._instance

	def __init__(self, config_file: Path | None = None, repo_root: Path | None = None) -> None:
		"""
		Initialize the configuration loader.

		Args:
			config_file: Path to configuration file (optional)
			repo_root: Repository root path (optional)

		"""
		self.repo_root = repo_root
		self._config_file = config_file
		self._resolved_config_file = self._resolve_config_file(config_file)
		# Load configuration eagerly during initialization instead of lazy loading
		self._app_config = self._load_config()
		logger.debug("ConfigLoader initialized with eager configuration loading")

	def reload_config(self, config_file: Path | None = None, repo_root: Path | None = None) -> None:
		"""
		Reload configuration with new settings.

		Args:
			config_file: New configuration file path
			repo_root: New repository root path
		"""
		if config_file is not None:
			self._config_file = config_file
		if repo_root is not None:
			self.repo_root = repo_root
		self._resolved_config_file = self._resolve_config_file(self._config_file)
		# Reload the configuration immediately
		self._app_config = self._load_config()
		logger.debug("Configuration reloaded")

	def _resolve_config_file(self, config_file: Path | None = None) -> Path | None:
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
			path = config_file.expanduser().resolve()
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

	@staticmethod
	def _parse_yaml_file(file_path: Path) -> dict[str, Any]:
		"""
		Parse a YAML file with caching for better performance.

		Args:
			file_path: Path to the YAML file to parse

		Returns:
			Parsed YAML content as a dictionary

		Raises:
			yaml.YAMLError: If the file cannot be parsed as valid YAML
		"""
		with file_path.open(encoding="utf-8") as f:
			content = yaml.safe_load(f)
			if content is None:  # Empty file
				return {}
			if not isinstance(content, dict):
				msg = f"File {file_path} does not contain a valid YAML dictionary"
				raise yaml.YAMLError(msg)
			return content

	def _load_config(self) -> AppConfigSchema:
		"""
		Load configuration from file and parse it into AppConfigSchema.

		Returns:
			AppConfigSchema: Loaded and parsed configuration.

		Raises:
			ConfigFileNotFoundError: If specified configuration file doesn't exist
			ConfigParsingError: If configuration file exists but cannot be loaded or parsed.

		"""
		# Lazy imports
		import yaml

		from codemap.config import AppConfigSchema

		file_config_dict: dict[str, Any] = {}
		if self._resolved_config_file:
			try:
				if self._resolved_config_file.exists():
					try:
						file_config_dict = self._parse_yaml_file(self._resolved_config_file)
						logger.info("Loaded configuration from %s", self._resolved_config_file)
					except yaml.YAMLError as e:
						msg = (
							f"Configuration file {self._resolved_config_file} does not contain a valid YAML dictionary."
						)
						logger.exception(msg)
						raise ConfigParsingError(msg) from e
				else:
					msg = f"Configuration file not found: {self._resolved_config_file}."
					logger.info("%s Using default configuration.", msg)
			except OSError as e:
				error_msg = f"Error accessing configuration file {self._resolved_config_file}: {e}"
				logger.exception(error_msg)
				raise ConfigParsingError(error_msg) from e
		else:
			logger.info("No configuration file specified or found. Using default configuration.")

		try:
			# Initialize AppConfigSchema. If file_config_dict is empty, defaults will be used.
			return AppConfigSchema(**file_config_dict)
		except Exception as e:  # Catch Pydantic validation errors etc.
			error_msg = f"Error parsing configuration into schema: {e}"
			logger.exception(error_msg)
			raise ConfigParsingError(error_msg) from e

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

	@property
	def get(self) -> AppConfigSchema:
		"""
		Get the current application configuration.

		Returns:
			AppConfigSchema: The current configuration
		"""
		# Configuration is now loaded during initialization, no need for lazy loading
		return self._app_config
