"""Configuration loader for the CodeMap tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

# Import the actual DEFAULT_CONFIG from the config module
from codemap.config import DEFAULT_CONFIG

# Import here to avoid "possibly unbound" errors
try:
	import yaml
except ImportError:
	yaml = None

logger = logging.getLogger(__name__)


class ConfigError(TypeError):
	"""Custom error for configuration validation."""

	TOKEN_LIMIT_MSG = "token_limit must be an integer"  # noqa: S105
	GITIGNORE_MSG = "use_gitignore must be a boolean"
	OUTPUT_DIR_MSG = "output_dir must be a string"


class ConfigLoader:
	"""Configuration loader for the CodeMap tool."""

	def __init__(self, config_file: str | None = None, repo_root: Path | None = None) -> None:
		"""
		Initialize the configuration loader.

		Args:
		    config_file: Path to configuration file, None for default location
		    repo_root: Repository root path, used for resolving relative paths

		Raises:
		    FileNotFoundError: If config_file is specified but not found

		"""
		self.config_file = config_file
		self.repo_root = repo_root
		self._config: dict[str, Any] = {}
		self.load_config()
		self._validate_config(self._config)

	def load_config(self) -> None:
		"""
		Load configuration from file.

		Raises:
		    FileNotFoundError: If config_file is specified but not found
		    yaml.YAMLError: If the YAML file cannot be parsed

		"""
		# Always start with default config
		self._config = DEFAULT_CONFIG.copy()

		# Try loading from specified config file, or default location
		if self.config_file:
			config_path = Path(self.config_file)
			if not config_path.exists():
				error_msg = f"Config file not found: {self.config_file}"
				logger.warning(error_msg)
				raise FileNotFoundError(error_msg)
		elif self.repo_root:
			# Look for .codemap.yml in the repository root
			config_path = self.repo_root / ".codemap.yml"
			if not config_path.exists():
				logger.debug("Config file .codemap.yml not found in repository, using default config")
				return
		else:
			# Look for .codemap.yml in the current directory
			config_path = Path(".codemap.yml")
			if not config_path.exists():
				logger.debug("Default config file .codemap.yml not found, using default config")
				return

		# Check if yaml module is available
		if yaml is None:
			logger.warning("PyYAML not installed, using default config")
			return

		# Read and parse config file
		try:
			with config_path.open("r", encoding="utf-8") as f:
				loaded_config = yaml.safe_load(f)
				if not loaded_config:
					logger.warning("Empty or invalid YAML in %s, using default config", config_path)
					return

				# Log the config file contents for debugging
				logger.debug("Loaded config from %s: %s", config_path, loaded_config)

				# Update config with loaded values
				self._config.update(loaded_config)
		except yaml.YAMLError:
			logger.exception("Failed to parse YAML from %s", config_path)
			raise  # Re-raise for tests
		except (PermissionError, OSError):
			logger.exception("Unable to read config file %s", config_path)
			raise

	def _validate_config(self, config: dict[str, Any]) -> None:
		"""
		Validate configuration values.

		Args:
		    config: Configuration dictionary to validate.

		Raises:
		    ConfigError: If any configuration values are invalid.

		"""
		if "token_limit" in config and not isinstance(config["token_limit"], int):
			raise ConfigError(ConfigError.TOKEN_LIMIT_MSG)

		if "use_gitignore" in config and not isinstance(config["use_gitignore"], bool):
			raise ConfigError(ConfigError.GITIGNORE_MSG)

		if "output_dir" in config and not isinstance(config["output_dir"], str):
			raise ConfigError(ConfigError.OUTPUT_DIR_MSG)

	@property
	def config(self) -> dict[str, Any]:
		"""
		Get the configuration dictionary.

		Returns:
		    Configuration dictionary

		"""
		return self._config

	def get_llm_config(self) -> dict[str, Any]:
		"""
		Get LLM-specific configuration.

		Returns:
		    Dictionary with LLM configuration values

		"""
		llm_config = {
			"model": "openai/gpt-4o-mini",  # Default model
			"api_base": None,
			"api_key": None,
			"provider": None,
		}

		# Check if commit section exists with LLM configuration
		if "commit" in self._config and isinstance(self._config["commit"], dict):
			commit_config = self._config["commit"]

			# Extract LLM configuration
			if "llm" in commit_config and isinstance(commit_config["llm"], dict):
				config_llm = commit_config["llm"]

				# Update model if specified
				if "model" in config_llm:
					llm_config["model"] = config_llm["model"]

					# Extract provider from model name if it contains a prefix
					if "/" in llm_config["model"]:
						provider_prefix = llm_config["model"].split("/")[0].lower()
						llm_config["provider"] = provider_prefix

				# Allow explicit provider override
				if "provider" in config_llm:
					llm_config["provider"] = config_llm["provider"]

				# Update API base if specified
				if "api_base" in config_llm:
					llm_config["api_base"] = config_llm["api_base"]

				# Get provider-specific API key if available and provider is known
				if llm_config["provider"]:
					provider_key = f"{llm_config['provider']}_api_key"
					if provider_key in config_llm:
						llm_config["api_key"] = config_llm[provider_key]

				# Fall back to generic api_key if provider-specific not found
				if not llm_config["api_key"] and "api_key" in config_llm:
					llm_config["api_key"] = config_llm["api_key"]

		return llm_config

	def get_commit_convention(self) -> dict[str, Any]:
		"""
		Get commit convention configuration.

		Returns:
		    Dictionary with commit convention settings

		"""
		convention = {
			"types": ["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore"],
			"scopes": [],
			"max_length": 72,
		}

		# Check if commit section exists with convention configuration
		if "commit" in self._config and isinstance(self._config["commit"], dict):
			commit_config = self._config["commit"]

			if "convention" in commit_config and isinstance(commit_config["convention"], dict):
				conv_config = commit_config["convention"]

				if "types" in conv_config:
					convention["types"] = conv_config["types"]
				if "scopes" in conv_config:
					convention["scopes"] = conv_config["scopes"]
				if "max_length" in conv_config:
					convention["max_length"] = conv_config["max_length"]

		return convention

	def get_commit_hooks(self) -> bool:
		"""
		Get commit hooks configuration.

		Returns:
		    Boolean indicating whether to bypass git hooks (--no-verify)

		"""
		# Default is False (do not bypass hooks)
		bypass_hooks = False

		# Check if commit section exists with bypass_hooks configuration
		if "commit" in self._config and isinstance(self._config["commit"], dict):
			commit_config = self._config["commit"]

			if "bypass_hooks" in commit_config and isinstance(commit_config["bypass_hooks"], bool):
				bypass_hooks = commit_config["bypass_hooks"]

		return bypass_hooks

	def get_pr_config(self) -> dict[str, Any]:
		"""
		Get pull request configuration.

		Returns:
		    Dictionary with PR configuration values

		"""
		# Start with default PR config
		default_pr_config = DEFAULT_CONFIG["pr"].copy()

		# Check if PR section exists in config
		if "pr" in self._config and isinstance(self._config["pr"], dict):
			pr_config = self._config["pr"]

			# Deep merge the configuration
			if "defaults" in pr_config and isinstance(pr_config["defaults"], dict):
				default_pr_config["defaults"].update(pr_config["defaults"])

			if "strategy" in pr_config:
				default_pr_config["strategy"] = pr_config["strategy"]

			if "branch_mapping" in pr_config and isinstance(pr_config["branch_mapping"], dict):
				# For each branch type in the mapping
				for branch_type, mapping in pr_config["branch_mapping"].items():
					if branch_type in default_pr_config["branch_mapping"]:
						default_pr_config["branch_mapping"][branch_type].update(mapping)
					else:
						default_pr_config["branch_mapping"][branch_type] = mapping

			if "generate" in pr_config and isinstance(pr_config["generate"], dict):
				default_pr_config["generate"].update(pr_config["generate"])

		return default_pr_config

	def get_workflow_strategy(self) -> Literal["github-flow", "gitflow", "trunk-based"]:
		"""
		Get the configured git workflow strategy.

		Returns:
		    String identifying the workflow strategy

		"""
		pr_config = self.get_pr_config()
		return pr_config["strategy"]

	def get_branch_mapping(self, branch_type: str) -> dict[str, str]:
		"""
		Get base branch and prefix for a specific branch type.

		Args:
		    branch_type: Type of branch (feature, release, hotfix, etc.)

		Returns:
		    Dictionary with base branch and prefix

		"""
		pr_config = self.get_pr_config()

		# Check if the branch type exists in the mapping
		if branch_type in pr_config["branch_mapping"]:
			return pr_config["branch_mapping"][branch_type]

		# Return default mapping
		return {
			"base": pr_config["defaults"].get("base_branch") or "main",
			"prefix": pr_config["defaults"].get("feature_prefix") or "",
		}

	def get_default_branch(self) -> str:
		"""
		Get the default base branch from the PR configuration.

		Returns:
		    String containing the default branch name

		"""
		pr_config = self.get_pr_config()
		return pr_config["defaults"].get("base_branch") or "main"

	def get_content_generation_config(self) -> dict[str, Any]:
		"""
		Get configuration for PR content generation.

		Returns:
		    Dictionary with content generation settings

		"""
		pr_config = self.get_pr_config()
		return pr_config["generate"]


def get_config(config_file: str | None = None, repo_root: Path | None = None) -> dict[str, Any]:
	"""
	Helper function to get configuration without creating a class instance.

	Args:
	    config_file: Path to configuration file, None for default location
	    repo_root: Repository root path, used for resolving relative paths

	Returns:
	    Configuration dictionary

	"""
	loader = ConfigLoader(config_file, repo_root)
	return loader.config
