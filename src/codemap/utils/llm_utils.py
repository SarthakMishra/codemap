"""Utilities for working with LLMs across CodeMap features."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from rich.console import Console

# Import the MessageGenerator class - avoid circular imports
from codemap.git.commit.message_generator import DiffChunkDict, LLMError, MessageGenerator

console = Console()
logger = logging.getLogger(__name__)

# Configure LiteLLM and HTTP-related loggers more aggressively
# Since verbose mode is controlled via the setup_logging function,
# we'll set these to ERROR to ensure they don't appear in normal INFO level
logging.getLogger("litellm").setLevel(logging.WARNING)

# Also silence HTTP-related logs more aggressively
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


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
        parts = model.split("/")
        provider = parts[0].lower()
        # Special case for models like "groq/meta-llama/llama-4..."
        if provider == "groq":
            # We already have the correct provider
            pass
        elif provider in ["anthropic", "azure", "cohere", "mistral", "together", "openrouter", "openai"]:
            # These are standard providers
            pass
        else:
            # Default to openai if provider not recognized
            provider = "openai"

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
    chunk: DiffChunkLike,
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
    # Create a safe dictionary representation of chunk for fallback
    try:
        # Use getattr with default values to safely extract attributes from DiffChunkLike
        chunk_dict = DiffChunkDict(
            files=getattr(chunk, "files", []),
            content=getattr(chunk, "content", ""),
            description=getattr(chunk, "description", None),
        )

        if use_simple_mode:
            # Use fallback generation without LLM
            message = message_generator.fallback_generation(chunk_dict)
            return message, False

        # Try LLM-based generation first - using converted dict instead of original chunk
        message, used_llm = message_generator.generate_message(chunk_dict)
        return message, used_llm
    except LLMError as e:
        # If LLM generation fails, log and use fallback
        logger.warning("LLM message generation failed: %s", str(e))
        # Create a safe dictionary representation of chunk for fallback
        chunk_dict = DiffChunkDict(
            files=getattr(chunk, "files", []),
            content=getattr(chunk, "content", ""),
            description=getattr(chunk, "description", None),
        )
        message = message_generator.fallback_generation(chunk_dict)
        return message, False
    except (ValueError, RuntimeError):
        # For other errors, log and re-raise
        logger.exception("Message generation error")
        raise


def create_universal_generator(
    repo_path: Path | None,
    model: str | None = "openai/gpt-4o-mini",
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
    # Ensure repo_path and model are not None
    actual_repo_path = repo_path if repo_path is not None else Path()
    actual_model = model or "openai/gpt-4o-mini"

    # Try to load .env file if it exists
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Create the generator using the centralized function
    return setup_message_generator(
        repo_path=actual_repo_path,
        model=actual_model,
        prompt_template=prompt_template,
        prompt_template_path=prompt_template_path,
        api_base=api_base,
        api_key=api_key,
    )


def generate_text_with_llm(
    prompt: str, model: str | None = "gpt-4o-mini", api_key: str | None = None, api_base: str | None = None
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

    # Ensure model is never None
    actual_model = model or "gpt-4o-mini"

    # Extract provider from model name if it includes a provider prefix
    provider = None
    if "/" in actual_model:
        parts = actual_model.split("/")
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
            model=actual_model,
            messages=messages,
            api_key=api_key,
            api_base=api_base,
            temperature=0.3,  # Conservative temperature for predictable outputs
            max_tokens=1000,
        )

        # Extract text using a robust method that works with different response structures
        return extract_content_from_response(response)
    except Exception as e:
        logger.exception("LLM error")
        error_message = f"Failed to generate text with LLM: {e}"
        raise RuntimeError(error_message) from e


def extract_content_from_response(response: object) -> str:
    """Extract content from a LiteLLM response object.

    This function handles different response formats that might be returned by LiteLLM,
    which wraps responses from various LLM providers.

    Args:
        response: The response object from LiteLLM

    Returns:
        The extracted text content
    """
    # Try different methods to extract content
    try:
        # Method 1: Standard OpenAI format - attribute access
        if hasattr(response, "choices") and response.choices:  # type: ignore[attr-defined]
            choice = response.choices[0]  # type: ignore[attr-defined]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):  # type: ignore[attr-defined]
                content = choice.message.content  # type: ignore[attr-defined]
                if content is not None:
                    return content.strip()

        # Method 2: Dictionary-like access (for non-object responses)
        if isinstance(response, dict) and response.get("choices"):
            choice = response["choices"][0]
            # Check for message-style response (chat completion)
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                if content is not None:
                    return content.strip()
            # Check for text-style response (text completion)
            elif "text" in choice:
                content = choice["text"]
                if content is not None:
                    return content.strip()

        # Method 3: Handle direct string responses (some models might return raw text)
        if isinstance(response, str):
            return response.strip()

        # Method 4: If all else fails, convert the entire response to string
        return str(response)
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        # If we encounter any errors during extraction, log and use string representation
        import logging

        logging.getLogger(__name__).warning("Error extracting content: %s. Using string representation.", e)
        return str(response)
