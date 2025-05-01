"""Configuration for LLM module."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Default model to use if none is specified
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Known LLM providers
KNOWN_PROVIDERS = ["openai", "anthropic", "azure", "groq", "mistral", "together", "cohere", "openrouter"]

# Map of provider names to environment variable names for API keys
ENV_VAR_MAP = {
	"openai": "OPENAI_API_KEY",
	"anthropic": "ANTHROPIC_API_KEY",
	"azure": "AZURE_API_KEY",
	"groq": "GROQ_API_KEY",
	"mistral": "MISTRAL_API_KEY",
	"together": "TOGETHER_API_KEY",
	"cohere": "COHERE_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
}

# Default parameters for LLM API requests
DEFAULT_LLM_REQUEST_PARAMS = {
	"temperature": 0.3,
	"max_tokens": 1000,
}


@dataclass
class LLMConfig:
	"""Configuration for LLM operations."""

	model: str = DEFAULT_MODEL
	provider: str | None = None
	api_base: str | None = None
	api_key: str | None = None
	api_keys: dict[str, str] = field(default_factory=dict)

	def __post_init__(self) -> None:
		"""Process the configuration after initialization."""
		# Extract provider from model if not explicitly provided
		if not self.provider and "/" in self.model:
			self.provider = self.model.split("/")[0].lower()
			logger.debug("Extracted provider '%s' from model '%s'", self.provider, self.model)

		# If provider is still not set, default to OpenAI
		if not self.provider:
			self.provider = "openai"
			logger.debug("No provider found, defaulting to 'openai'")

		# Load API keys from environment if not provided
		if not self.api_keys:
			self.api_keys = self._load_api_keys_from_env()
			for provider in self.api_keys:
				logger.debug("Loaded API key for provider: %s", provider)

		# If specific api_key is provided, add it to api_keys
		if self.api_key and self.provider:
			self.api_keys[self.provider] = self.api_key
			logger.debug("Added explicit API key for provider: %s", self.provider)

		if self.provider:
			env_var = ENV_VAR_MAP.get(self.provider)
			logger.debug("Looking for API key in environment variable: %s", env_var)
			if env_var:
				key = os.environ.get(env_var)
				if key:
					logger.debug("Found API key in environment for provider: %s", self.provider)
				else:
					logger.warning(
						"No API key found in environment for provider: %s (env var: %s)", self.provider, env_var
					)

	def _load_api_keys_from_env(self) -> dict[str, str]:
		"""Load API keys from environment variables."""
		api_keys = {}

		# Check all known provider environment variables
		for provider in KNOWN_PROVIDERS:
			env_var = ENV_VAR_MAP.get(provider)
			if env_var and (key := os.environ.get(env_var)):
				api_keys[provider] = key
				logger.debug("Loaded API key for %s from environment variable %s", provider, env_var)

		return api_keys

	def get_api_key(self) -> str | None:
		"""Get the API key for the configured provider."""
		if not self.provider:
			logger.warning("No provider configured, cannot get API key")
			return None

		# Try from loaded keys
		if self.provider in self.api_keys:
			logger.debug("Using API key from loaded keys for provider: %s", self.provider)
			return self.api_keys[self.provider]

		# Try from environment as a fallback
		env_var = ENV_VAR_MAP.get(self.provider)
		if env_var:
			key = os.environ.get(env_var)
			if key:
				logger.debug("Using API key from environment for provider: %s", self.provider)
				return key
			# If key is not found or is empty, log the warning
			logger.warning("API key not found in environment for provider: %s (env var: %s)", self.provider, env_var)

		logger.warning("No API key found for provider: %s", self.provider)
		return None


def get_llm_config(
	config_loader: ConfigLoader | None = None,
	**overrides: dict[str, str | int | float | bool | None] | str | float | bool | None,
) -> LLMConfig:
	"""
	Get LLM configuration from config loader and optional overrides.

	Args:
	    config_loader: Optional ConfigLoader instance to use
	    **overrides: Optional configuration overrides

	Returns:
	    LLMConfig instance with merged configuration

	"""
	# Create a config loader if none provided
	if not config_loader:
		config_loader = ConfigLoader()

	# Get LLM config from loader
	llm_config_dict = config_loader.get_llm_config()

	# Log the model being used
	model = llm_config_dict.get("model", DEFAULT_MODEL)
	logger.debug("Using model from config: %s", model)

	# Apply overrides
	for key, value in overrides.items():
		if value is not None:  # Only override if value is not None
			llm_config_dict[key] = value
			if key == "model":
				logger.debug("Overriding model with: %s", value)

	# Create and return config object
	return LLMConfig(
		model=llm_config_dict.get("model", DEFAULT_MODEL),
		provider=llm_config_dict.get("provider"),
		api_base=llm_config_dict.get("api_base"),
		api_key=llm_config_dict.get("api_key"),
	)
