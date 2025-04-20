"""Commit message generation using LLMs for CodeMap."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from .diff_splitter import DiffChunk

# Import at runtime too to avoid circular imports

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
3. Scope must be short, concise, and represent the specific component affected
   (avoid long paths like src/codemap/git/commit)
4. Total message MUST be under 72 characters including type, scope and description
5. Description should use imperative present tense (e.g., "add", "fix", "update", not "added", "fixed", "updated")
6. Your response must ONLY contain the commit message with no other text, explanation, or formatting

Here are some notes about the files changed:
{files}

Analyze the following diff and respond with ONLY the commit message:

{diff}
"""


# Replace Any with a proper type hint for the chunk
class DiffChunkDict(TypedDict, total=False):
    """Type hint for DiffChunk attributes."""

    files: list[str]
    content: str
    description: str | None


class LLMError(Exception):
    """Custom exception for LLM-related errors."""


class MessageGenerator:
    """Generates commit messages using LLMs."""

    def __init__(
        self,
        repo_root: Path,
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

                if config and isinstance(config, dict) and "commit" in config:
                    commit_config = config["commit"]

                    # Load LLM settings if available
                    if "llm" in commit_config and isinstance(commit_config["llm"], dict):
                        llm_config = commit_config["llm"]

                        # Only override if not explicitly set in constructor
                        if "model" in llm_config and self.model == "gpt-4o-mini":
                            self.model = llm_config["model"]

                        if "provider" in llm_config and self.provider is None:
                            self.provider = llm_config["provider"]

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
        }

        # Try to load from config file
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists():
            try:
                import yaml

                with config_file.open("r") as f:
                    config = yaml.safe_load(f)

                if "commit" in config and "llm" in config["commit"]:
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

                if "commit" in config and "convention" in config["commit"]:
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
        # If provider is explicitly specified, use it
        if self.provider:
            # Build provider-specific model identifier
            if self.provider == "azure":
                # Azure needs special handling with deployment names
                return f"azure/{self.model}", self.api_base
            if self.provider in ["anthropic", "cohere", "mistral", "groq", "together"]:
                # These providers need a prefix
                return f"{self.provider}/{self.model}", self.api_base
            # Default to the model as-is
            return self.model, self.api_base

        # If the model already has a provider prefix (e.g., "anthropic/claude-3-opus-20240229"), use it as is
        for provider in ["anthropic/", "azure/", "cohere/", "mistral/", "groq/", "together/"]:
            if self.model.startswith(provider):
                return self.model, self.api_base

        # No provider specified, use model as-is (assumed to be OpenAI or compatible)
        return self.model, self.api_base

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

        try:
            import litellm

            # Get the API key based on provider
            model, provider = self._get_model_with_provider()
            api_keys = self._get_api_keys()

            # Configure the API
            api_key = None
            if provider:
                api_key = api_keys.get(provider)

            # Set up the API base URL if provided
            api_base = self.api_base

            # Validate configuration
            validate_config(provider, api_base)

            # Call the API
            response = litellm.completion(
                model=f"{provider}/{model}" if provider else model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                api_base=api_base,
            )

            # Extract and return the message content
            return response.choices[0].message.content.strip()

        except ImportError:
            msg = "LiteLLM library not installed. Install it with 'pip install litellm'."
            raise LLMError(msg) from None
        except Exception as e:
            msg = f"LLM API call failed: {e!s}"
            raise LLMError(msg) from e

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
        return " ".join(message.splitlines())

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

        return f"{commit_type}: {description}"

    def generate_message(self, chunk: DiffChunkDict | DiffChunk) -> tuple[str, bool]:
        """Generate a commit message for the given diff chunk.

        Args:
            chunk: DiffChunk to generate message for

        Returns:
            Tuple of (generated message, whether LLM was used)

        Raises:
            LLMError: If something goes wrong during message generation
        """
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

        # Try to get description from chunk
        if chunk.get("description"):
            return chunk["description"], True

        # Check if this is an empty diff
        if not chunk.get("content", "").strip():
            message = self.fallback_generation(chunk)
            return message, False

        # Try to generate a message using LLM
        try:
            # Import here to avoid circular imports
            from .interactive import loading_spinner

            # Prepare the prompt
            prompt = self._prepare_prompt(chunk)

            # Generate and validate the message
            with loading_spinner("Generating commit message..."):
                message = self._call_llm_api(prompt)

            # Return the formatted message
            return self._format_message(message), True
        except LLMError:
            # If LLM generation fails, use fallback
            logger.warning("LLM generation failed, using fallback")
            message = self.fallback_generation(chunk)
            return message, False
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("LLM generation failed: %s", e)
            error_msg = "Failed to generate commit message"
            raise LLMError(error_msg) from e
