"""Commit message generation using LLMs for CodeMap."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pathlib

    from .diff_splitter import DiffChunk

logger = logging.getLogger(__name__)

# Constants to avoid magic numbers
MIN_DESCRIPTION_LENGTH = 10
MIN_SCOPE_LENGTH = 10

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

Here are some notes about the files changed:
{files}

Analyze the following diff and respond with ONLY the commit message string:

{diff}
"""


# Type hint for DiffChunk attributes - replacing TypedDict with a regular dict type for cleaner code
class DiffChunkDict(dict):
    """Type hint for DiffChunk attributes."""

    def __init__(self, files: list[str] | None = None, content: str = "", description: str | None = None) -> None:
        """Initialize with DiffChunk attributes."""
        super().__init__()
        self["files"] = files or []
        self["content"] = content
        if description is not None:
            self["description"] = description

    def get(self, key: str, default: object = None) -> object:
        """Get a value with default."""
        return super().get(key, default)


class LLMError(Exception):
    """Custom exception for LLM-related errors."""


class MessageGenerator:
    """Generates commit messages using LLMs."""

    def __init__(
        self,
        repo_root: pathlib.Path,
        prompt_template: str | None = None,
        model: str = "gpt-4o-mini",
        provider: str | None = None,
        api_base: str | None = None,
    ) -> None:
        """Initialize the message generator.

        Args:
            repo_root: Root directory of the Git repository
            prompt_template: Custom prompt template to use
            model: Model identifier to use for generation
            provider: Provider to use (e.g., "openai", "anthropic", "azure", etc.)
                     If None, will be inferred from model if possible
            api_base: Optional API base URL for the provider
        """
        self.repo_root = repo_root
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.model = model
        self.provider = provider
        self.api_base = api_base
        self._api_keys = self._get_api_keys()

        # Load configuration values from .codemap.yml if available
        self._load_config_values()

    def _load_config_values(self) -> None:
        """Load configuration values from .codemap.yml."""
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists():
            try:
                import yaml

                with config_file.open("r") as f:
                    config = yaml.safe_load(f)

                if config is not None and isinstance(config, dict) and "commit" in config:
                    commit_config = config["commit"]

                    # Load LLM settings if available
                    if "llm" in commit_config and isinstance(commit_config["llm"], dict):
                        llm_config = commit_config["llm"]

                        # Only override if not explicitly set in constructor
                        if "model" in llm_config and self.model == "gpt-4o-mini":
                            self.model = llm_config["model"]
                            # The provider is now extracted from the model name during usage
                            # This ensures compatibility with LiteLLM's native format

                        if "api_base" in llm_config and self.api_base is None:
                            self.api_base = llm_config["api_base"]
            except (ImportError, yaml.YAMLError, OSError) as e:
                logger.warning("Error loading config: %s", e)
                # Continue with default values

    def _get_api_keys(self) -> dict[str, str | None]:
        """Get API keys from environment or config file.

        Returns:
            Dictionary of API keys for different providers
        """
        # Start with empty keys dict
        api_keys = {
            "openai": os.environ.get("OPENAI_API_KEY"),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "azure": os.environ.get("AZURE_API_KEY"),
            "cohere": os.environ.get("COHERE_API_KEY"),
            "groq": os.environ.get("GROQ_API_KEY"),
            "mistral": os.environ.get("MISTRAL_API_KEY"),
            "together": os.environ.get("TOGETHER_API_KEY"),
            "openrouter": os.environ.get("OPENROUTER_API_KEY"),
        }

        # Try to load from config file
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists():
            try:
                import yaml

                with config_file.open("r") as f:
                    config = yaml.safe_load(f)

                if config is not None and "commit" in config and "llm" in config["commit"]:
                    llm_config = config["commit"]["llm"]

                    # Load API keys from config
                    for provider in api_keys:
                        config_key = f"{provider}_api_key"
                        if config_key in llm_config:
                            api_keys[provider] = llm_config[config_key]

                    # Also check for a generic API key
                    if "api_key" in llm_config:
                        api_keys["default"] = llm_config["api_key"]
            except (ImportError, yaml.YAMLError):
                pass

        return api_keys

    def _extract_file_info(self, chunk: DiffChunkDict) -> dict[str, Any]:
        """Extract information about the files in the diff.

        Args:
            chunk: DiffChunk to analyze

        Returns:
            Dictionary with file information
        """
        file_info = {}

        for file in chunk.get("files", []):
            file_path = self.repo_root / file
            if not file_path.exists():
                continue

            try:
                # Get file type based on extension
                extension = file_path.suffix.lstrip(".")
                file_info[file] = {
                    "extension": extension,
                    "directory": str(file_path.parent.relative_to(self.repo_root)),
                }

                # Try to identify the module/component from the path
                path_parts = file_path.parts
                if len(path_parts) > 1:
                    if "src" in path_parts:
                        idx = path_parts.index("src")
                        if idx + 1 < len(path_parts):
                            file_info[file]["module"] = path_parts[idx + 1]
                    elif "tests" in path_parts:
                        file_info[file]["module"] = "tests"

            except (ValueError, IndexError):
                continue

        return file_info

    def _get_commit_convention(self) -> dict[str, Any]:
        """Get commit convention settings from config.

        Returns:
            Dictionary with commit convention settings
        """
        convention = {
            "types": ["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore"],
            "scopes": [],
            "max_length": 72,  # Conventional commit standard
        }

        # Try to load from config file
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists():
            try:
                import yaml

                with config_file.open("r") as f:
                    config = yaml.safe_load(f)

                if config is not None and "commit" in config and "convention" in config["commit"]:
                    conv_config = config["commit"]["convention"]
                    if "types" in conv_config:
                        convention["types"] = conv_config["types"]
                    if "scopes" in conv_config:
                        convention["scopes"] = conv_config["scopes"]
                    if "max_length" in conv_config:
                        convention["max_length"] = conv_config["max_length"]
            except (ImportError, yaml.YAMLError):
                pass

        return convention

    def _prepare_prompt(self, chunk: DiffChunkDict) -> str:
        """Prepare the LLM prompt based on the diff chunk.

        Args:
            chunk: DiffChunk to generate message for

        Returns:
            Formatted prompt
        """
        file_info = self._extract_file_info(chunk)
        convention = self._get_commit_convention()

        # Enhance the prompt with file and convention info
        context = {
            "diff": chunk.get("content", ""),
            "files": file_info,
            "convention": convention,
        }

        return self.prompt_template.format(**context)

    def _get_model_with_provider(self) -> tuple[str, str | None]:
        """Get the model identifier with provider prefix if needed.

        Returns:
            Tuple of (model name with provider prefix if needed, api_base if needed)
        """
        # If the model already has a provider prefix (e.g., "anthropic/claude-3-opus-20240229"), use it as is
        for provider in ["anthropic/", "azure/", "cohere/", "mistral/", "groq/", "together/", "openrouter/", "openai/"]:
            if self.model.startswith(provider):
                return self.model, self.api_base

        # If provider is explicitly specified as a legacy case, use it to prefix the model
        if self.provider:
            # Build provider-specific model identifier
            if self.provider == "azure":
                # Azure needs special handling with deployment names
                return f"azure/{self.model}", self.api_base
            if self.provider in ["anthropic", "cohere", "mistral", "groq", "together", "openrouter", "openai"]:
                # These providers need a prefix
                return f"{self.provider}/{self.model}", self.api_base
            # Default to the model as-is
            return self.model, self.api_base

        # No provider specified, assume OpenAI for backward compatibility
        return f"openai/{self.model}", self.api_base

    def _call_llm_api(self, prompt: str) -> str:
        """Call the LLM API to generate a commit message.

        Args:
            prompt: Formatted prompt for the API

        Returns:
            Generated commit message

        Raises:
            LLMError: If API call fails
        """

        def validate_config(provider: str | None, api_base: str | None) -> None:
            """Validate the LLM provider configuration.

            Args:
                provider: The provider name
                api_base: The API base URL

            Raises:
                LLMError: If configuration is invalid
            """
            if provider == "azure" and not api_base:
                msg = "Azure requires an API base URL"
                raise LLMError(msg)

            # Set OpenRouter API base if needed
            if provider == "openrouter":
                if not api_base and not os.environ.get("OPENROUTER_API_BASE"):
                    # Set default API base for OpenRouter
                    os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
                    logger.debug("Set OPENROUTER_API_BASE to %s", os.environ["OPENROUTER_API_BASE"])
                elif api_base and not os.environ.get("OPENROUTER_API_BASE"):
                    # Use provided API base for OpenRouter
                    os.environ["OPENROUTER_API_BASE"] = api_base
                    logger.debug("Set OPENROUTER_API_BASE to %s", os.environ["OPENROUTER_API_BASE"])

        def raise_api_key_error(provider_name: str) -> None:
            """Raise an error for missing API key.

            Args:
                provider_name: The name of the provider missing an API key

            Raises:
                LLMError: Always raised with appropriate message
            """
            error_msg = f"No API key found for provider {provider_name}"
            raise LLMError(error_msg)

        try:
            import litellm

            # First get the model with provider prefix
            model, api_base = self._get_model_with_provider()

            # Get provider from model or self.provider
            provider = self.provider
            if not provider and "/" in model:
                provider = model.split("/")[0]  # Use first part regardless of number of slashes

            logger.debug("Using provider: %s, model: %s", provider, model)

            # Validate configuration with the correct provider
            validate_config(provider, api_base)

            # Now get the API keys
            api_keys = self._get_api_keys()

            # Log API key presence (not the actual key)
            has_api_key = api_keys.get(provider) is not None if provider else False
            logger.debug("API key available for %s: %s", provider or "default", has_api_key)

            # Configure the API
            api_key = None
            if provider:
                api_key = api_keys.get(provider)

                # Special handling for OpenRouter
                if provider == "openrouter" and not api_key:
                    # Check environment directly as a fallback
                    env_key = os.environ.get("OPENROUTER_API_KEY")
                    if env_key:
                        api_key = env_key
                        # Update api_keys for future use
                        api_keys["openrouter"] = env_key
                        has_api_key = True

                if not api_key:
                    logger.warning("No API key found for provider %s", provider)
                    raise_api_key_error(provider)

            # Log attempt with full details
            logger.debug(
                "Calling LLM API - Provider: %s, Model: %s, API Base: %s", provider, model, api_base or "default"
            )

            # Call the API
            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=api_key,
                    api_base=api_base,
                    max_retries=2,  # Add retries for transient failures
                    timeout=30,  # Set a reasonable timeout
                )

                # Log success
                logger.debug(
                    "LLM API call successful, message length: %d chars",
                    len(response.choices[0].message.content.strip()),
                )

                # Extract and return the message content
                return response.choices[0].message.content.strip()
            except litellm.exceptions.APIError as e:
                error_detail = str(e)
                logger.exception("LiteLLM API Error")

                if "authentication" in error_detail.lower() or "api key" in error_detail.lower():
                    error_msg = "Authentication failed"
                    raise LLMError(error_msg) from e
                if "rate limit" in error_detail.lower() or "ratelimit" in error_detail.lower():
                    error_msg = "Rate limit exceeded"
                    raise LLMError(error_msg) from e

                error_msg = "API error"
                raise LLMError(error_msg) from e
            except litellm.exceptions.Timeout as e:
                error_msg = "LLM API request timed out"
                raise LLMError(error_msg) from e
            except Exception as e:
                error_msg = "Unexpected error during LLM API call"
                logger.exception(error_msg)
                raise LLMError(error_msg) from e

        except ImportError:
            msg = "LiteLLM library not installed. Install it with 'pip install litellm'."
            logger.exception(msg)
            raise LLMError(msg) from None

    def _format_message(self, message: str) -> str:
        """Format and clean the generated commit message.

        Args:
            message: Raw message from LLM

        Returns:
            Cleaned and formatted commit message
        """
        # Strip any backticks, quotes, or other markdown formatting
        message = message.strip()
        message = message.replace("```", "").replace("`", "")

        # Remove any "Response:" or similar prefixes
        prefixes_to_remove = ["commit message:", "message:", "response:"]
        for prefix in prefixes_to_remove:
            if message.lower().startswith(prefix):
                message = message[len(prefix) :].strip()

        # Simplify scopes that are too verbose - match <type>(scope): and simplify long scopes
        scope_pattern = r"^([a-z]+)\(([^)]+)\):"
        match = re.match(scope_pattern, message)
        if match:
            commit_type = match.group(1)
            scope = match.group(2)

            # Reconstruct with scope
            description = message[match.end() :].strip()
            message = f"{commit_type}({scope}): {description}"

        # Enforce character limit (72 chars is conventional commit standard)
        convention = self._get_commit_convention()
        max_length = convention.get("max_length", 72)

        if len(message) > max_length:
            # Try to truncate intelligently - keep type and scope intact
            scope_match = re.match(scope_pattern, message)
            if scope_match:
                commit_type = scope_match.group(1)
                scope = scope_match.group(2)
                prefix = f"{commit_type}({scope}): "

                # Calculate how much room we have for description
                avail_len = max_length - len(prefix)
                if avail_len > MIN_DESCRIPTION_LENGTH:  # At least have a minimal description
                    description = message[scope_match.end() :].strip()
                    truncated_desc = description[: avail_len - 3] + "..."
                    message = f"{prefix}{truncated_desc}"
                # If scope is too long, try with a shorter scope
                elif len(scope) > MIN_SCOPE_LENGTH:
                    scope = scope[:8]
                    prefix = f"{commit_type}({scope}): "
                    avail_len = max_length - len(prefix)
                    description = message[scope_match.end() :].strip()
                    truncated_desc = description[: avail_len - 3] + "..."
                    message = f"{prefix}{truncated_desc}"
                else:
                    # Last resort: just truncate
                    message = message[: max_length - 3] + "..."
            else:
                # No scope found, just truncate
                message = message[: max_length - 3] + "..."

        # Handle potential line breaks and replace with spaces
        message = " ".join(message.splitlines())

        # Apply commit message sanitization to ensure commitlint compliance
        return self._sanitize_commit_message(message)

    def fallback_generation(self, chunk: DiffChunkDict) -> str:
        """Generate a simple commit message without using an LLM.

        Args:
            chunk: DiffChunk to generate message for

        Returns:
            Simple generated commit message
        """
        # Determine the most appropriate commit type based on file paths
        commit_type = "chore"

        for file in chunk.get("files", []):
            if file.startswith("tests/"):
                commit_type = "test"
                break
            if file.startswith("docs/") or file.endswith(".md"):
                commit_type = "docs"
                break
            if "fix" in chunk.get("content", "").lower():
                commit_type = "fix"
                break

        # Get a description based on the files changed
        if len(chunk.get("files", [])) == 1:
            description = f"update {chunk.get('files', [''])[0]}"
        else:
            # Try to find a common directory
            common_dir = os.path.commonpath(chunk.get("files", []))
            if common_dir and common_dir != ".":
                description = f"update files in {common_dir}"
            else:
                description = f"update {len(chunk.get('files', []))} files"

        message = f"{commit_type}: {description}"

        # Apply sanitization to ensure commitlint compliance
        return self._sanitize_commit_message(message)

    def _verify_api_key_availability(self) -> bool:
        """Verify that the API key for the configured provider is available.

        Returns:
            True if API key is available, False otherwise
        """
        # Get provider from model name if not explicitly set
        provider = self.provider
        if not provider and self.model and "/" in self.model:
            provider = self.model.split("/")[0]  # Take first segment regardless of number of slashes

        api_keys = self._get_api_keys()

        # For test scenarios, especially where _call_llm_api is mocked
        if getattr(self, "_mock_api_key_available", False):
            logger.warning("MOCK API KEY AVAILABLE FOR TESTING")
            return True

        # For OpenRouter specifically
        if provider == "openrouter":
            key = api_keys.get("openrouter")
            if key:
                logger.warning("OPENROUTER API KEY IS AVAILABLE")
                return True
            logger.warning("OPENROUTER API KEY IS MISSING")
            # Check env specifically
            env_key = os.environ.get("OPENROUTER_API_KEY")
            if env_key:
                logger.warning("OPENROUTER API KEY IS IN ENVIRONMENT")
                api_keys["openrouter"] = env_key  # Store it in api_keys for later use
                return True
            logger.warning("OPENROUTER API KEY NOT FOUND IN ENVIRONMENT")
            return False

        # For other providers
        if provider and provider in api_keys:
            key = api_keys.get(provider)
            if key:
                logger.warning("API KEY FOR %s IS AVAILABLE", provider)
                return True
            logger.warning("API KEY FOR %s IS MISSING", provider)

            # Try fallback to OpenAI if a specific provider key is missing
            openai_key = api_keys.get("openai")
            if openai_key:
                logger.warning("FALLING BACK TO OPENAI API KEY")
                self.provider = "openai"  # Override provider to OpenAI
                self.model = "gpt-3.5-turbo"  # Use a stable model
                return True

            return False

        # Default OpenAI key check
        key = api_keys.get("openai")
        if key:
            logger.warning("OPENAI API KEY IS AVAILABLE")
            return True

        logger.warning("NO API KEYS AVAILABLE - WILL USE FALLBACK MESSAGE GENERATION")
        return False

    def generate_message(self, chunk: DiffChunkDict | DiffChunk) -> tuple[str, bool]:
        """Generate a commit message for the given diff chunk.

        Args:
            chunk: DiffChunk to generate message for

        Returns:
            Tuple of (generated message, whether LLM was used)

        Raises:
            LLMError: If something goes wrong during message generation
        """
        # Force a warning log to ensure logging is visible
        logger.warning(
            "ENTERING MessageGenerator.generate_message - Chunk ID: %s, Initial Desc: %s",
            id(chunk),
            # Use getattr for dataclass attribute access with default
            getattr(chunk, "description", "<None>"),
        )
        logger.warning("STARTING MESSAGE GENERATION WITH MODEL: %s, PROVIDER: %s", self.model, self.provider)

        # Convert DiffChunk to dict if needed
        if not isinstance(chunk, dict):
            # Import here to avoid circular imports
            from .diff_splitter import DiffChunk

            if isinstance(chunk, DiffChunk):
                chunk_dict: DiffChunkDict = {
                    "files": chunk.files,
                    "content": chunk.content,
                }
                if hasattr(chunk, "description") and chunk.description:
                    chunk_dict["description"] = chunk.description
                chunk = chunk_dict

        # --> Check if chunk already has a description <--
        existing_desc = chunk.get("description")
        if existing_desc:
            logger.warning("CHUNK ALREADY HAS DESCRIPTION: %s - RETURNING IT", existing_desc)
            # Check if this is a default description (likely from _create_semantic_chunk)
            # If it starts with "chore: update" or similar, try to generate a better one
            if existing_desc.startswith(("chore: update", "fix: update", "docs: update", "test: update")):
                logger.warning("EXISTING DESCRIPTION IS GENERIC - WILL TRY TO IMPROVE IT")
            elif not getattr(chunk, "is_llm_generated", False):
                # If it's not LLM-generated, we should try to generate a better one
                logger.warning("EXISTING DESCRIPTION IS NOT LLM-GENERATED - WILL TRY TO IMPROVE IT")
            else:
                return existing_desc, False

        # Verify API key availability
        has_api_key = self._verify_api_key_availability()
        if not has_api_key:
            logger.warning("USING FALLBACK - NO API KEY AVAILABLE")
            message = self.fallback_generation(chunk)
            return message, False

        # Check if this is an empty diff
        chunk_content = chunk.get("content", "").strip()
        if not chunk_content:
            logger.warning("CHUNK CONTENT IS EMPTY - USING FALLBACK")
            message = self.fallback_generation(chunk)
            return message, False
        # Add log to see what the content is if not empty
        logger.warning("Chunk content is NOT empty (first 100 chars): %s", chunk_content[:100])

        # Try to generate a message using LLM
        try:
            # Import here to avoid circular imports
            from .interactive import loading_spinner

            # Constants to avoid magic numbers
            max_log_message_length = 50

            # Prepare the prompt
            prompt = self._prepare_prompt(chunk)
            logger.warning("Prepared prompt for LLM, length: %d chars", len(prompt))

            # Log attempt with model info
            model_str = f"{self.provider}/{self.model}" if self.provider else self.model
            logger.warning("Attempting LLM message gen with: %s", model_str)

            # Generate and validate the message
            with loading_spinner("Generating commit message..."):
                # --> Add log before calling _call_llm_api <--
                logger.warning("About to call _call_llm_api...")
                message = self._call_llm_api(prompt)
                logger.warning(
                    "API call succeeded, got response: %s",
                    message[:max_log_message_length] + "..." if len(message) > max_log_message_length else message,
                )

            # Return the formatted message
            logger.warning("Formatting final message")
            return self._format_message(message), True
        except ImportError as e:
            # Missing dependency
            logger.exception("Failed to import required module")
            error_msg = f"Missing dependency: {e}"
            raise LLMError(error_msg) from e
        except LLMError:
            # Specific LLM errors (e.g., API issues)
            logger.exception("LLM API error")
            logger.info("Falling back to simple message generation")
            message = self.fallback_generation(chunk)
            return message, False
        except (OSError, ValueError, RuntimeError) as e:
            # Other errors
            logger.exception("Error during message generation")
            error_msg = f"Failed to generate commit message: {e}"
            raise LLMError(error_msg) from e

    def _sanitize_commit_message(self, message: str) -> str:
        """Sanitize a commit message to comply with commitlint standards.

        Args:
            message: The commit message to sanitize

        Returns:
            Sanitized commit message
        """
        # Remove trailing period
        if message.endswith("."):
            message = message[:-1]

        return message
