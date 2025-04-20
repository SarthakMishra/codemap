"""Utilities for working with LLMs across CodeMap features."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from rich.console import Console

# Import the MessageGenerator class - avoid circular imports
from codemap.git.commit.message_generator import LLMError, MessageGenerator

console = Console()
logger = logging.getLogger(__name__)


@runtime_checkable
class DiffChunkLike(Protocol):
    """Protocol for objects that behave like DiffChunks."""

    files: list[str]
    content: str


# Type variable constrained to DiffChunkLike
ChunkT = TypeVar("ChunkT", bound=DiffChunkLike)


def load_prompt_template(template_path: str | None) -> str | None:
    """Load custom prompt template from file.

    Args:
        template_path: Path to prompt template file

    Returns:
        Loaded template or None if loading failed
    """
    if not template_path:
        return None

    try:
        template_file = Path(template_path)
        with template_file.open("r") as f:
            return f.read()
    except OSError:
        console.print(f"[yellow]Warning:[/yellow] Could not load prompt template: {template_path}")
        return None


def setup_message_generator(
    repo_path: Path,
    model: str = "openai/gpt-4o-mini",
    prompt_template: str | None = None,
    prompt_template_path: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> MessageGenerator:
    """Set up a message generator with the provided options, leveraging LiteLLM API key handling.

    Args:
        repo_path: Path to the repository
        model: LLM model name
        prompt_template: Custom prompt template content
        prompt_template_path: Path to custom prompt template file
        api_base: API base URL
        api_key: API key (will override environment variables)

    Returns:
        Configured message generator
    """
    # Load custom prompt template from file if path is provided
    custom_prompt = None
    if prompt_template:
        custom_prompt = prompt_template
    elif prompt_template_path:
        custom_prompt = load_prompt_template(prompt_template_path)

    # Extract provider from model if possible (for error reporting only)
    provider = None
    if "/" in model:
        provider = model.split("/")[0].lower()

    # Create the message generator
    # MessageGenerator will use its own _get_api_keys method to find keys
    # from environment variables or config files
    message_generator = MessageGenerator(
        repo_path,
        prompt_template=custom_prompt,
        model=model,
        provider=provider,  # Use extracted provider
        api_base=api_base,
    )

    # If api_key was provided explicitly, ensure it's set in the environment
    # This ensures litellm will find it during API calls
    if api_key and provider:
        # Set in environment for litellm to find
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "azure":
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        elif provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key
        elif provider == "mistral":
            os.environ["MISTRAL_API_KEY"] = api_key
        elif provider == "together":
            os.environ["TOGETHER_API_KEY"] = api_key
        elif provider == "cohere":
            os.environ["COHERE_API_KEY"] = api_key
        elif provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = api_key
            if not os.environ.get("OPENROUTER_API_BASE"):
                os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
    elif api_key:
        # No provider specified in model name, default to OpenAI
        os.environ["OPENAI_API_KEY"] = api_key

    return message_generator


def generate_message(
    chunk: ChunkT,
    message_generator: MessageGenerator,
    use_simple_mode: bool = False,
) -> tuple[str, bool]:
    """Generate a message for the given chunk.

    This universal function can be used by both commit.py and pr.py.

    Args:
        chunk: DiffChunk or other object to generate message for
        message_generator: Configured MessageGenerator instance
        use_simple_mode: If True, use simple generation mode without LLM

    Returns:
        Tuple of (message, whether LLM was used)
    """
    try:
        if use_simple_mode:
            # Use fallback generation without LLM
            message = message_generator.fallback_generation(chunk)
            return message, False
        # Try LLM-based generation first
        message, used_llm = message_generator.generate_message(chunk)
        return message, used_llm
    except LLMError as e:
        # If LLM generation fails, log and use fallback
        logger.warning("LLM message generation failed: %s", str(e))
        message = message_generator.fallback_generation(chunk)
        return message, False
    except (ValueError, RuntimeError):
        # For other errors, log and re-raise
        logger.exception("Message generation error")
        raise


def create_universal_generator(
    repo_path: Path,
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    prompt_template: str | None = None,
    prompt_template_path: str | None = None,
) -> MessageGenerator:
    """Create a universal message generator that can be used by any module.

    This is a simplified function that combines the provider extraction and setup
    for easy use in any module.

    Args:
        repo_path: Path to the repository
        model: Model name (with or without provider prefix)
        api_key: API key (optional, will be pulled from environment if not provided)
        api_base: API base URL (optional)
        prompt_template: Custom prompt template content (optional)
        prompt_template_path: Path to custom prompt template file (optional)

    Returns:
        Configured MessageGenerator
    """
    # Try to load .env file if it exists
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Create the generator using the centralized function
    return setup_message_generator(
        repo_path=repo_path,
        model=model,
        prompt_template=prompt_template,
        prompt_template_path=prompt_template_path,
        api_base=api_base,
        api_key=api_key,
    )


def generate_text_with_llm(
    prompt: str, model: str = "gpt-4o-mini", api_key: str | None = None, api_base: str | None = None
) -> str:
    """Generate text using an LLM.

    Args:
        prompt: The prompt to send to the LLM
        model: The model to use
        api_key: The API key to use
        api_base: The API base URL to use

    Returns:
        The generated text

    Raises:
        RuntimeError: If the LLM call fails
    """
    import logging
    import os

    from litellm import completion

    logger = logging.getLogger(__name__)

    # Extract provider from model name if it includes a provider prefix
    provider = None
    if "/" in model:
        parts = model.split("/")
        # Minimum parts needed for provider/model format
        min_parts = 2
        if len(parts) >= min_parts:
            provider = parts[0].lower()

    # Use provided API key or get it from environment
    if not api_key:
        if provider:
            # Try provider-specific environment variable
            env_var_name = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(env_var_name)

        # Fallback to OpenAI
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

    # Configure messages for chat completion
    messages = [{"role": "user", "content": prompt}]

    try:
        # Call LiteLLM for cross-platform compatibility
        response = completion(
            model=model,
            messages=messages,
            api_key=api_key,
            api_base=api_base,
            temperature=0.3,  # Conservative temperature for predictable outputs
            max_tokens=1000,
        )

        # Extract text from response
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("LLM error")
        error_message = f"Failed to generate text with LLM: {e}"
        raise RuntimeError(error_message) from e
