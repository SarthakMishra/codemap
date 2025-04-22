"""Commit message generation using LLMs for CodeMap."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Iterable, TypedDict, cast

import yaml

if TYPE_CHECKING:
    import pathlib


# Define DiffChunk class outside of TYPE_CHECKING
class DiffChunk:
    """Represents a logical chunk of changes to files in a diff."""

    def __init__(self, files: list[str], content: str, description: str | None = None) -> None:
        """Initialize a diff chunk.

        Args:
            files: List of files affected in this chunk
            content: The diff content
            description: Optional description of the changes
        """
        self.files: list[str] = files
        self.content: str = content
        self.description: str | None = description
        self.is_llm_generated: bool = False


logger = logging.getLogger(__name__)

# Constants to avoid magic numbers
MIN_DESCRIPTION_LENGTH = 10
MIN_SCOPE_LENGTH = 10
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
KNOWN_PROVIDERS = {
    "openai",
    "anthropic",
    "azure",
    "cohere",
    "groq",
    "mistral",
    "together",
    "openrouter",
}


# Default prompt template for commit message generation
DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant that generates conventional commit messages based on code changes.
Given a Git diff, please generate a concise and descriptive commit message following these conventions:

1. Use the format: <type>(<scope>): <description>
2. Types include: {convention[types]}
3. Scope must be short (1-2 words), concise, and represent the specific component affected
4. Total message MUST be under 72 characters including type, scope and description
5. The description should be a concise, imperative present tense summary of the *specific code changes*
   in the diff chunk (e.g., "add feature", "fix bug", "update documentation").
   Focus on *what* was changed and *why*.
6. Your response must ONLY contain the commit message string, formatted as <type>(<scope>): <description>,
   with absolutely no other text, explanation, or surrounding characters (like quotes or markdown).

Example output:
- feat(ui): add new button to toggle dark mode
- fix(api): handle null values in JSON response
- chore(deps): update dependencies
- docs(readme): add usage examples
- test(unit): add tests for new function

---
Here are some notes about the files changed:
{files}
---
Analyze the following diff and respond with ONLY the commit message string:

{diff}

---
IMPORTANT:
- Strictly follow the format <type>(<scope>): <description>
- Do not include any other text, explanation, or surrounding characters (like quotes or markdown).
"""


# Define a TypedDict to represent the structure of a DiffChunk
class DiffChunkData(TypedDict, total=False):
    """TypedDict representing the structure of a DiffChunk."""

    files: list[str]
    content: str
    description: str


class LLMError(Exception):
    """Custom exception for LLM-related errors."""


class MessageGenerator:
    """Generates commit messages using LLMs."""

    def __init__(
        self,
        repo_root: pathlib.Path,
        prompt_template: str | None = None,
        model: str = "gpt-4o-mini",  # Default model
        provider: str | None = None,  # Optional explicit provider
        api_base: str | None = None,
    ) -> None:
        """Initialize the message generator.

        Args:
            repo_root: Root directory of the Git repository
            prompt_template: Custom prompt template to use
            model: Model identifier to use (can include provider prefix like 'groq/llama3')
            provider: Explicit provider name (e.g., "openai", "anthropic").
                      Overrides provider inferred from model prefix if both are present.
            api_base: Optional API base URL for the provider
        """
        self.repo_root = repo_root
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        # Store initial config values
        self._initial_model = model
        self._initial_provider = provider
        self._initial_api_base = api_base

        # Try to load environment variables from .env.local file
        self._load_env_variables()

        # Load API keys from environment/config
        self._api_keys = self._get_api_keys()  # Load keys early

        # Load configuration values from .codemap.yml, potentially overriding initial values
        self._load_config_values()  # This will update self._initial_model etc. if found

        # --- Centralized Configuration Resolution ---
        (
            self.resolved_model,
            self.resolved_provider,
            self.resolved_api_base,
        ) = self._resolve_llm_configuration(self._initial_model, self._initial_provider, self._initial_api_base)
        logger.debug(
            "Resolved LLM Configuration: Provider=%s, Model=%s, API_Base=%s",
            self.resolved_provider,
            self.resolved_model,
            self.resolved_api_base or "Default",
        )
        # --- End Centralized Configuration Resolution ---

        # For tests - flag to bypass API key check
        self._mock_api_key_available = False

    @property
    def model(self) -> str:
        """Get the model name.

        Returns:
            The resolved model name
        """
        return self.resolved_model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model name.

        Args:
            value: The model name to set
        """
        self.resolved_model = value

    @model.deleter
    def model(self) -> None:
        """Delete the model property.

        This is needed for unit tests using patch.object.
        """

    @property
    def provider(self) -> str:
        """Get the provider name.

        Returns:
            The resolved provider name
        """
        return self.resolved_provider

    @provider.setter
    def provider(self, value: str) -> None:
        """Set the provider name.

        Args:
            value: The provider name to set
        """
        self.resolved_provider = value

    @provider.deleter
    def provider(self) -> None:
        """Delete the provider property.

        This is needed for unit tests using patch.object.
        """

    def _load_env_variables(self) -> None:
        """Load environment variables from .env.local or .env files."""
        try:
            from dotenv import load_dotenv

            # First try .env.local (higher priority)
            env_local = self.repo_root / ".env.local"
            if env_local.exists():
                load_dotenv(dotenv_path=env_local, override=True)  # Override existing env vars
                logger.debug("Loaded environment variables from %s", env_local)
                return

            # Then try .env (lower priority)
            env_file = self.repo_root / ".env"
            if env_file.exists():
                load_dotenv(dotenv_path=env_file, override=True)  # Override existing env vars
                logger.debug("Loaded environment variables from %s", env_file)
        except ImportError:
            logger.debug("python-dotenv not installed, skipping .env file loading")
        except OSError as e:
            logger.warning("Error loading .env file: %s", e)

    def _load_config_values(self) -> None:
        """Load configuration values from .codemap.yml, updating initial settings."""
        config_file = self.repo_root / ".codemap.yml"
        if not config_file.exists():
            logger.debug("No .codemap.yml found, using initial/default settings.")
            return

        try:
            if yaml is None:
                logger.warning("PyYAML not installed, cannot load .codemap.yml")
                return

            with config_file.open("r") as f:
                config = yaml.safe_load(f)

            if not config or not isinstance(config, dict) or "commit" not in config:
                logger.debug(".codemap.yml found but no 'commit' section.")
                return

            commit_config = config["commit"]
            if not isinstance(commit_config, dict):
                return

            # Load LLM settings if available
            if "llm" in commit_config and isinstance(commit_config["llm"], dict):
                llm_config = commit_config["llm"]
                logger.debug("Loading LLM config from .codemap.yml: %s", llm_config)

                # Config overrides initial values
                if "model" in llm_config:
                    self._initial_model = llm_config["model"]
                    logger.debug("Overriding model from config: %s", self._initial_model)
                if "provider" in llm_config:
                    # Allow explicit provider override in config
                    self._initial_provider = llm_config["provider"]
                    logger.debug("Overriding provider from config: %s", self._initial_provider)
                if "api_base" in llm_config:
                    self._initial_api_base = llm_config["api_base"]
                    logger.debug("Overriding api_base from config: %s", self._initial_api_base)
                # Note: API keys are handled separately in _get_api_keys

        except (ImportError, OSError) as e:
            logger.warning("Error loading config file %s: %s", config_file, e)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Error parsing config file %s: %s", config_file, e)

    def _get_api_keys(self) -> dict[str, str]:
        """Get API keys from environment or config file. Prioritizes environment variables.

        Returns:
            Dictionary of API keys found for known providers.
        """
        api_keys: dict[str, str] = {}
        provider_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        # 1. Load from Environment Variables (Highest Priority)
        for provider, env_var in provider_env_map.items():
            key = os.environ.get(env_var)
            if key:
                api_keys[provider] = key
                logger.debug("Loaded API key for %s from environment variable %s", provider, env_var)

        # 2. Load from .codemap.yml (Lower Priority)
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists() and yaml is not None:
            try:
                with config_file.open("r") as f:
                    config = yaml.safe_load(f)

                if config and isinstance(config, dict) and "commit" in config:
                    commit_config = config["commit"]
                    if isinstance(commit_config, dict) and "llm" in commit_config:
                        llm_config = commit_config["llm"]
                        if isinstance(llm_config, dict):
                            # Load provider-specific keys from config if not already loaded from env
                            for provider in provider_env_map:
                                if provider not in api_keys:  # Only load if not set by env var
                                    config_key_name = f"{provider}_api_key"
                                    if config_key_name in llm_config:
                                        api_keys[provider] = llm_config[config_key_name]
                                        logger.debug("Loaded API key for %s from config file", provider)

                            # Load generic 'api_key' from config (lowest priority)
                            # Useful if provider is specified but key isn't provider-specific
                            if "api_key" in llm_config and not api_keys:  # Check if any key was loaded
                                generic_key = llm_config["api_key"]
                                # We don't know the provider yet, store it under a temp key?
                                # Or better, let the resolve logic handle assigning it later.
                                # For now, we just log it might exist.
                                logger.debug("Found generic 'api_key' in config, will use if needed.")
                                # Store it temporarily if needed, maybe associated with the configured provider?
                                if self._initial_provider and self._initial_provider not in api_keys:
                                    api_keys[self._initial_provider] = generic_key

            except (ImportError, OSError, Exception) as e:
                logger.warning("Error reading API keys from config: %s", e)

        logger.debug("Final loaded API keys (providers): %s", list(api_keys.keys()))
        return api_keys

    def _resolve_llm_configuration(
        self, model: str, provider: str | None, api_base: str | None
    ) -> tuple[str, str, str | None]:
        """
        Resolves the final model string, provider name, and API base URL.

        Args:
            model: The model name (potentially with provider prefix).
            provider: An explicitly configured provider name.
            api_base: An explicitly configured API base URL.

        Returns:
            A tuple containing (resolved_model, resolved_provider, resolved_api_base).
            resolved_model will be in 'provider/model_name' format.
            resolved_provider will be the determined provider name.
        """
        resolved_model = model
        resolved_provider = provider
        resolved_api_base = api_base

        # 1. Check if model name already has a known provider prefix
        if "/" in resolved_model:
            prefix = resolved_model.split("/")[0].lower()
            if prefix in KNOWN_PROVIDERS:
                # Model has prefix, this is the most specific information
                inferred_provider = prefix
                resolved_provider = inferred_provider
                logger.debug("Provider '%s' inferred from model name '%s'", resolved_provider, resolved_model)
            # Has a slash, but not a known provider prefix. Treat as opaque model name.
            # If no explicit provider set, we have ambiguity. Default to openai? Or error?
            # Let's default to openai if no explicit provider is given.
            elif not resolved_provider:
                logger.warning(
                    "Model '%s' has '/' but prefix '%s' is not a known provider. Assuming 'openai'.",
                    resolved_model,
                    prefix,
                )
                resolved_provider = "openai"
                # Keep resolved_model as is, maybe it's a custom deployment name for openai?
                # else: Use the explicitly set resolved_provider

        # 2. If model has no prefix, use explicit provider or default to openai
        elif resolved_provider and resolved_provider.lower() in KNOWN_PROVIDERS:
            # Explicit provider is given and known, format the model string
            resolved_provider = resolved_provider.lower()
            resolved_model = f"{resolved_provider}/{resolved_model}"
            logger.debug("Using explicit provider '%s', formatted model as '%s'", resolved_provider, resolved_model)
        else:
            # No prefix, no (valid) explicit provider. Default to OpenAI.
            if resolved_provider:
                logger.warning("Explicit provider '%s' is not known. Defaulting to 'openai'.", resolved_provider)

            resolved_provider = "openai"
            resolved_model = f"openai/{resolved_model}"
            logger.debug("Defaulting to provider 'openai', formatted model as '%s'", resolved_model)

        # 3. Handle provider-specific API base logic
        if resolved_provider == "azure" and not resolved_api_base:
            # Note: LiteLLM might get this from AZURE_API_BASE env var automatically
            logger.warning("Azure provider typically requires an API base URL, but none was configured.")
            # We don't raise an error here, maybe env var is set. LiteLLM will handle it.

        if resolved_provider == "openrouter":
            # Set default OpenRouter API base if none is provided
            if not resolved_api_base:
                resolved_api_base = DEFAULT_OPENROUTER_API_BASE
                logger.debug("Using default API base for OpenRouter: %s", resolved_api_base)
            # Ensure OPENROUTER_API_BASE env var is set for LiteLLM if using api_base
            # LiteLLM uses this env var preferentially for OpenRouter routing logic
            if resolved_api_base and os.environ.get("OPENROUTER_API_BASE") != resolved_api_base:
                os.environ["OPENROUTER_API_BASE"] = resolved_api_base
                logger.debug("Set OPENROUTER_API_BASE environment variable to: %s", resolved_api_base)

        # Ensure resolved_provider is lowercase
        resolved_provider = resolved_provider.lower()

        return resolved_model, resolved_provider, resolved_api_base

    def _extract_file_info(self, chunk: DiffChunk | DiffChunkData) -> dict[str, Any]:
        """Extract file information from the diff chunk.

        Args:
            chunk: Diff chunk to extract information from

        Returns:
            Dictionary with information about files
        """
        file_info = {}
        files = chunk.files if isinstance(chunk, DiffChunk) else chunk.get("files", [])
        if not isinstance(files, list):
            try:
                # Convert to list only if it's actually iterable
                if hasattr(files, "__iter__") and not isinstance(files, str):
                    files = list(cast("Iterable", files))
                else:
                    files = []
            except (TypeError, ValueError):
                files = []

        for file in files:
            if not isinstance(file, str):
                continue  # Skip non-string file entries
            file_path = self.repo_root / file
            if not file_path.exists():
                continue
            try:
                extension = file_path.suffix.lstrip(".")
                file_info[file] = {
                    "extension": extension,
                    "directory": str(file_path.parent.relative_to(self.repo_root)),
                }
                path_parts = file_path.parts
                if len(path_parts) > 1:
                    if "src" in path_parts:
                        idx = path_parts.index("src")
                        if idx + 1 < len(path_parts):
                            file_info[file]["module"] = path_parts[idx + 1]
                    elif "tests" in path_parts:
                        file_info[file]["module"] = "tests"
            except (ValueError, IndexError, TypeError):
                continue
        return file_info

    def _get_commit_convention(self) -> dict[str, Any]:
        """Get commit convention settings from config. (Unchanged)."""
        convention = {
            "types": ["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore"],
            "scopes": [],
            "max_length": 72,
        }
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists() and yaml is not None:
            try:
                with config_file.open("r") as f:
                    config = yaml.safe_load(f)
                if config and isinstance(config, dict) and "commit" in config:
                    commit_config = config["commit"]
                    if isinstance(commit_config, dict) and "convention" in commit_config:
                        conv_config = commit_config["convention"]
                        if isinstance(conv_config, dict):
                            if "types" in conv_config:
                                convention["types"] = conv_config["types"]
                            if "scopes" in conv_config:
                                convention["scopes"] = conv_config["scopes"]
                            if "max_length" in conv_config:
                                convention["max_length"] = conv_config["max_length"]
            except (ImportError, OSError, Exception) as e:
                logger.warning("Error loading/parsing commit convention from config: %s", e)
        return convention

    def _prepare_prompt(self, chunk: DiffChunk | DiffChunkData) -> str:
        """Prepare the prompt for the LLM.

        Args:
            chunk: Diff chunk to prepare prompt for

        Returns:
            Prepared prompt with diff and file information
        """
        file_info = self._extract_file_info(chunk)
        convention = self._get_commit_convention()

        # Get the diff content from the chunk
        diff_content = chunk.content if isinstance(chunk, DiffChunk) else chunk.get("content", "")

        context = {
            "diff": diff_content,
            "files": file_info,
            "convention": convention,
        }
        try:
            return self.prompt_template.format(**context)
        except KeyError:
            logger.exception("Prompt template formatting error. Missing key: %s. Using default template.")
            # Fallback to default template if custom one fails
            return DEFAULT_PROMPT_TEMPLATE.format(**context)

    def _call_llm_api(self, prompt: str) -> str:
        """Call the LLM API using the resolved configuration.

        Args:
            prompt: Formatted prompt for the API

        Returns:
            Generated commit message

        Raises:
            LLMError: If API call fails or litellm is not installed.
        """
        try:
            import litellm
        except ImportError:
            msg = "LiteLLM library not installed. Install it with 'pip install litellm'."
            logger.exception(msg)
            raise LLMError(msg) from None

        # Use the resolved configuration from __init__
        model_to_use = self.resolved_model
        provider_to_use = self.resolved_provider
        api_base_to_use = self.resolved_api_base

        # Get the API key for the resolved provider
        api_key = self._api_keys.get(provider_to_use)

        # Check generic key from config if provider-specific key is missing
        # This relies on _get_api_keys potentially storing a generic key
        # associated with the initial provider if no specific key was found.
        # A bit fragile, might be better to explicitly check llm_config['api_key'] here if needed.
        if not api_key and "api_key" in self._get_llm_config_from_yaml():  # Helper needed
            api_key = self._get_llm_config_from_yaml().get("api_key")
            logger.debug("Using generic 'api_key' from config for provider %s", provider_to_use)

        # Define the env_var_map at the beginning
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        if not api_key:
            # Re-check environment directly as a last resort for specific providers
            env_var = env_var_map.get(provider_to_use)
            if env_var:
                api_key = os.environ.get(env_var)
                if api_key:
                    logger.debug(
                        "API key for resolved provider '%s' found in environment variable %s.", provider_to_use, env_var
                    )
                    return api_key

        if not api_key:
            error_msg = (
                f"No API key found for provider '{provider_to_use}'. "
                f"Checked config, environment variables ({env_var_map.get(provider_to_use, 'N/A')}), "
                f"and generic 'api_key' in config."
            )
            raise LLMError(error_msg)

        logger.debug(
            "Calling LiteLLM: Provider=%s, Model=%s, API_Base=%s, Key_Found=True",
            provider_to_use,
            model_to_use,
            api_base_to_use or "Default",
        )

        try:
            response = litellm.completion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                api_base=api_base_to_use,
                max_retries=2,
                timeout=30,
            )

            # Safely extract content from LiteLLM response
            content = ""

            # Handle response as a dictionary first (for flexibility)
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    # Try to get message content directly from dictionary structure
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        message = first_choice.get("message", {})
                        if isinstance(message, dict):
                            content = message.get("content", "")

            # Handle response as an object with attributes
            if not content and hasattr(response, "choices"):
                choices = getattr(response, "choices", None)
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    if hasattr(first_choice, "message"):
                        message = getattr(first_choice, "message", None)
                        if message and hasattr(message, "content"):
                            content = message.content or ""

            # Fallbacks for older LiteLLM versions or unexpected structures
            if not content:
                # Try direct content attribute (older versions)
                if hasattr(response, "content"):
                    content = getattr(response, "content", "") or ""
                # Try text attribute in choices (some models/providers)
                elif (
                    hasattr(response, "choices")
                    and getattr(response, "choices", None)
                    and len(getattr(response, "choices", [])) > 0
                    and hasattr(getattr(response, "choices", [])[0], "text")
                ):
                    choices = getattr(response, "choices", [])
                    content = getattr(choices[0], "text", "") or ""
                # Last resort: stringify the entire response
                elif hasattr(response, "__str__"):
                    content = str(response)

            content = content.strip()
            logger.debug("LLM API call successful, message length: %d chars", len(content))
            return content

        except litellm.exceptions.AuthenticationError as e:
            error_msg = f"LiteLLM Authentication Error for provider {provider_to_use}. Check API key."
            logger.exception("%s", error_msg)
            raise LLMError(error_msg) from e
        except litellm.exceptions.RateLimitError as e:
            error_msg = f"LiteLLM Rate Limit Error for provider {provider_to_use}."
            logger.exception("%s", error_msg)
            raise LLMError(error_msg) from e
        except litellm.exceptions.APIError as e:
            error_msg = f"LiteLLM API Error for provider {provider_to_use}."
            logger.exception(error_msg)  # Log full trace for generic API errors
            full_error_msg = f"{error_msg} Details: {e}"
            raise LLMError(full_error_msg) from e
        except litellm.exceptions.Timeout as e:
            error_msg = f"LiteLLM request timed out for provider {provider_to_use}."
            raise LLMError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during LiteLLM API call for provider {provider_to_use}."
            logger.exception(error_msg)
            raise LLMError(error_msg) from e

    # Helper to read llm config section again (used in _call_llm_api for generic key)
    def _get_llm_config_from_yaml(self) -> dict:
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists() and yaml is not None:
            try:
                with config_file.open("r") as f:
                    config = yaml.safe_load(f)
                if config and isinstance(config, dict) and "commit" in config:
                    commit_config = config["commit"]
                    if isinstance(commit_config, dict) and "llm" in commit_config:
                        llm_config = commit_config["llm"]
                        if isinstance(llm_config, dict):
                            return llm_config
            except (OSError, yaml.YAMLError, ValueError):
                logger.debug("Could not load LLM config from %s", config_file)
        return {}

    def _format_message(self, message: str) -> str:
        """Format and clean the generated commit message. (Unchanged, but added sanitization call)."""
        message = message.strip().replace("```", "").replace("`", "")
        prefixes_to_remove = ["commit message:", "message:", "response:"]
        for prefix in prefixes_to_remove:
            if message.lower().startswith(prefix):
                message = message[len(prefix) :].strip()

        scope_pattern = r"^([a-z]+)\(([^)]+)\):"
        match = re.match(scope_pattern, message)
        if match:
            commit_type, scope = match.group(1), match.group(2)
            description = message[match.end() :].strip()
            message = f"{commit_type}({scope}): {description}"

        convention = self._get_commit_convention()
        max_length = convention.get("max_length", 72)

        if len(message) > max_length:
            scope_match = re.match(scope_pattern, message)
            if scope_match:
                commit_type, scope = scope_match.group(1), scope_match.group(2)
                prefix = f"{commit_type}({scope}): "
                avail_len = max_length - len(prefix)
                if avail_len > MIN_DESCRIPTION_LENGTH:
                    description = message[scope_match.end() :].strip()
                    truncated_desc = description[: avail_len - 3].rstrip(" .") + "..."
                    message = f"{prefix}{truncated_desc}"
                elif len(scope) > MIN_SCOPE_LENGTH:
                    scope = scope[: MIN_SCOPE_LENGTH - 2] + ".."  # Shorten scope more aggressively
                    prefix = f"{commit_type}({scope}): "
                    avail_len = max_length - len(prefix)
                    if avail_len > MIN_DESCRIPTION_LENGTH:
                        description = message[scope_match.end() :].strip()
                        truncated_desc = description[: avail_len - 3].rstrip(" .") + "..."
                        message = f"{prefix}{truncated_desc}"
                    else:  # Still too long, just truncate whole message
                        message = message[: max_length - 3].rstrip(" .") + "..."
                else:
                    message = message[: max_length - 3].rstrip(" .") + "..."
            else:
                message = message[: max_length - 3].rstrip(" .") + "..."

        message = " ".join(message.splitlines())
        return self._sanitize_commit_message(message)  # Ensure sanitization is called

    def fallback_generation(self, chunk: DiffChunk | DiffChunkData) -> str:
        """Generate a fallback commit message without LLM.

        This is used when LLM-based generation fails or is disabled.

        Args:
            chunk: Diff chunk to generate message for

        Returns:
            Generated commit message
        """
        commit_type = "chore"

        # Get files from the chunk
        files = chunk.files if isinstance(chunk, DiffChunk) else chunk.get("files", [])

        string_files = [f for f in files if isinstance(f, str)]  # Filter only strings for path operations

        for file in string_files:
            if file.startswith("tests/"):
                commit_type = "test"
                break
            if file.startswith("docs/") or file.endswith(".md"):
                commit_type = "docs"
                break

        # Get content from the chunk
        content = chunk.content if isinstance(chunk, DiffChunk) else chunk.get("content", "")

        if isinstance(content, str) and ("fix" in content.lower() or "bug" in content.lower()):
            commit_type = "fix"  # Be slightly smarter about 'fix' type

        description = "update files"  # Default description
        if string_files:
            if len(string_files) == 1:
                description = f"update {string_files[0]}"
            else:
                try:
                    common_dir = os.path.commonpath(string_files)
                    # Make common_dir relative to repo root if possible
                    try:
                        common_dir_rel = os.path.relpath(common_dir, self.repo_root)
                        if common_dir_rel and common_dir_rel != ".":
                            description = f"update files in {common_dir_rel}"
                        else:
                            description = f"update {len(string_files)} files"
                    except ValueError:  # Happens if paths are on different drives (unlikely in repo)
                        description = f"update {len(string_files)} files"

                except (ValueError, TypeError):  # commonpath fails on empty list or mixed types
                    description = f"update {len(string_files)} files"

        message = f"{commit_type}: {description}"
        # Ensure fallback also respects max length and sanitization
        convention = self._get_commit_convention()
        max_length = convention.get("max_length", 72)
        if len(message) > max_length:
            message = message[: max_length - 3] + "..."

        return self._sanitize_commit_message(message)  # Ensure sanitization

    def _verify_api_key_availability(self) -> bool:
        """Verify that the API key for the *resolved* provider is available."""
        # For tests - if mock flag is set, return True
        if hasattr(self, "_mock_api_key_available") and self._mock_api_key_available:
            logger.debug("Mock API key flag is set, returning True for tests")
            return True

        # Use the resolved provider determined during initialization
        provider = self.resolved_provider
        if not provider:
            logger.error("Provider could not be resolved. Cannot verify API key.")
            return False

        if provider in self._api_keys:
            logger.debug("API key for resolved provider '%s' is available.", provider)
            return True
        # Last check in environment just in case
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var = env_var_map.get(provider)
        if env_var and os.environ.get(env_var):
            logger.debug("API key for resolved provider '%s' found in environment variable %s.", provider, env_var)
            return True

        logger.warning("API key for resolved provider '%s' is MISSING.", provider)
        return False

    def _adapt_chunk_access(self, chunk: DiffChunk | DiffChunkData) -> DiffChunkData:
        """Adapt chunk access to work with both DiffChunk objects and dictionaries.

        Args:
            chunk: Chunk to adapt

        Returns:
            Dictionary with chunk data
        """
        if isinstance(chunk, DiffChunk):
            return DiffChunkData(
                files=chunk.files,
                content=chunk.content,
                description=chunk.description if chunk.description else "",
            )
        return cast("DiffChunkData", chunk)

    def generate_message(self, chunk: DiffChunk | DiffChunkData) -> tuple[str, bool]:
        """Generate a commit message for the given diff chunk.

        Args:
            chunk: Diff chunk to generate message for

        Returns:
            Tuple of (message, was_generated_by_llm)
        """
        logger.debug(
            "Generating message for chunk ID: %s. Using resolved config: Provider=%s, Model=%s",
            id(chunk),
            self.resolved_provider,
            self.resolved_model,
        )

        chunk_dict = self._adapt_chunk_access(chunk)
        existing_desc = chunk_dict.get("description")

        # Check for existing description (same logic as before)
        if existing_desc and isinstance(existing_desc, str):
            is_generic = existing_desc.startswith(("chore: update", "fix: update", "docs: update", "test: update"))
            is_llm_gen = getattr(chunk, "is_llm_generated", False)  # Check original object if possible

            if not is_generic and is_llm_gen:
                logger.debug("Chunk already has LLM-generated description: '%s'", existing_desc)
                return existing_desc, True  # Assume it was LLM generated previously
            if not is_generic and not is_llm_gen:
                logger.debug(
                    "Chunk has existing non-generic, non-LLM description: '%s'. Attempting to improve.", existing_desc
                )
                # Proceed to generate below
            elif is_generic:
                logger.debug("Existing description is generic ('%s'). Attempting to generate.", existing_desc)
                # Proceed to generate below

        # Verify API key availability using the resolved provider
        if not self._verify_api_key_availability():
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "azure": "AZURE_API_KEY",
                "cohere": "COHERE_API_KEY",
                "groq": "GROQ_API_KEY",
                "mistral": "MISTRAL_API_KEY",
                "together": "TOGETHER_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            provider_env = env_var_map.get(self.resolved_provider, f"{self.resolved_provider.upper()}_API_KEY")
            error_msg = (
                f"No API key found for resolved provider '{self.resolved_provider}'. "
                f"Please set {provider_env} in your environment or configure "
                f"'{self.resolved_provider}_api_key' or 'api_key' in .codemap.yml -> commit -> llm section."
            )
            logger.error(error_msg)
            # Don't raise here, fall back instead
            logger.warning("API key missing for %s. Falling back to simple generation.", self.resolved_provider)
            message = self.fallback_generation(chunk_dict)
            return message, False

        chunk_content = chunk_dict.get("content", "")
        if isinstance(chunk_content, str):
            chunk_content = chunk_content.strip()

        if not chunk_content:
            logger.warning("Chunk content is empty - using fallback generation.")
            message = self.fallback_generation(chunk_dict)
            return message, False

        # Try to generate a message using LLM
        try:
            # Create a proper context manager with type annotations
            class DummyContextManager:
                def __init__(self, message: str) -> None:
                    self.message = message

                def __enter__(self) -> None:
                    return None

                def __exit__(
                    self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
                ) -> None:
                    return None

            def dummy_context_manager(message: str) -> DummyContextManager:
                """Simple do-nothing context manager."""
                return DummyContextManager(message)

            loading_spinner = dummy_context_manager

            prompt = self._prepare_prompt(chunk_dict)
            logger.debug("Prepared prompt for LLM, length: %d chars", len(prompt))

            with loading_spinner("Generating commit message..."):
                message = self._call_llm_api(prompt)

            formatted_message = self._format_message(message)
            logger.debug("LLM generated message: '%s'", formatted_message)
            # Mark the chunk if possible (requires chunk to be mutable or return new object)
            if isinstance(chunk, DiffChunk):
                chunk.is_llm_generated = True  # Mark original object if it's the class type
            return formatted_message, True

        except LLMError:
            # Handle specific LLM errors (API key, rate limit, etc.) gracefully
            logger.exception("LLM Error during generation")
            logger.info("Falling back to simple message generation.")
            message = self.fallback_generation(chunk_dict)
            return message, False
        except Exception as e:
            # Catch other unexpected errors during the process
            logger.exception("Unexpected error during message generation")
            error_msg = f"Failed to generate commit message due to unexpected error: {e}"
            # Decide whether to raise or fallback
            logger.info("Falling back to simple message generation due to unexpected error.")
            message = self.fallback_generation(chunk_dict)
            return message, False

    def _sanitize_commit_message(self, message: str) -> str:
        """Sanitize a commit message to comply with commitlint standards. (Unchanged)."""
        # Remove trailing period
        if message.endswith("."):
            message = message[:-1]
        # Potentially add more sanitization rules here if needed
        return message.strip()  # Ensure no leading/trailing whitespace
