"""
Configuration loader for CodeMap.

This module provides functionality for loading and managing
configuration settings.

"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from codemap.config.config_schema import AppConfigSchema

# For type checking only
if TYPE_CHECKING:
	from pydantic import BaseModel


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

	_instance: "ConfigLoader | None" = None

	@classmethod
	def get_instance(
		cls,
	) -> "ConfigLoader":
		"""
		Get the singleton instance of ConfigLoader.

		Returns:
			ConfigLoader: Singleton instance

		"""
		if cls._instance is None:
			cls._instance = cls()
		return cls._instance

	def __init__(self) -> None:
		"""
		Initialize the configuration loader.

		Args:
			config_file: Path to configuration file (optional)
			repo_root: Repository root path (optional)

		"""
		# Initialize repo_root attribute
		self.repo_root: Path | None = None
		# Load configuration eagerly during initialization instead of lazy loading
		self._app_config = self._load_config()
		logger.debug("ConfigLoader initialized with eager configuration loading")

	def _get_config_file(self) -> Path | None:
		"""
		Resolve the configuration file path.

		look in standard locations:
		1. ./.codemap.yml in the current directory
		2. ./.codemap.yml in the repository root

		Args:
			config_file: Explicitly provided config file path (optional)

		Returns:
			Optional[Path]: Resolved config file path or None if no suitable file found

		"""
		# Import GitRepoContext here to avoid circular imports
		from codemap.git.utils import GitRepoContext

		# Try current directory
		local_config = Path(".codemap.yml")
		if local_config.exists():
			self.repo_root = local_config.parent
			return local_config

		# Try repository root
		repo_root = GitRepoContext().get_repo_root()
		self.repo_root = repo_root
		repo_config = repo_root / ".codemap.yml"
		if repo_config.exists():
			return repo_config

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
		config_file = self._get_config_file()

		if config_file:
			try:
				if config_file.exists():
					try:
						file_config_dict = self._parse_yaml_file(config_file)
						logger.info("Loaded configuration from %s", config_file)
					except yaml.YAMLError as e:
						msg = f"Configuration file {config_file} does not contain a valid YAML dictionary."
						logger.exception(msg)
						raise ConfigParsingError(msg) from e
				else:
					msg = f"Configuration file not found: {config_file}."
					logger.info("%s Using default configuration.", msg)
			except OSError as e:
				error_msg = f"Error accessing configuration file {config_file}: {e}"
				logger.exception(error_msg)
				raise ConfigParsingError(error_msg) from e
		else:
			logger.info("No configuration file specified or found. Using default configuration.")

		file_config_dict["repo_root"] = self.repo_root

		try:
			# Initialize AppConfigSchema. If file_config_dict is empty, defaults will be used.
			config = AppConfigSchema(**file_config_dict)
			# Override github.token with env var if set
			env_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("CODEMAP_GITHUB_TOKEN")
			if env_token:
				config.github.token = env_token
			return config
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
