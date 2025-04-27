"""Utility functions for working with LLMs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .api import extract_content_from_response as _extract_content
from .client import LLMClient
from .config import DEFAULT_MODEL, get_llm_config
from .errors import LLMError

logger = logging.getLogger(__name__)


def load_prompt_template(template_path: str | None) -> str | None:
	"""
	Load custom prompt template from file.

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
		logger.warning("Could not load prompt template: %s", template_path)
		return None


def get_llm_client() -> LLMClient:
	"""
	Create and return a LLM client.

	Returns:
	    LLMClient instance

	Raises:
	    RuntimeError: If client creation fails

	"""
	try:
		config = get_llm_config()
		return LLMClient(config=config)

	except LLMError as e:
		logger.exception("LLM error")
		msg = f"Failed to create LLM client: {e}"
		raise RuntimeError(msg) from e


# Define a type for the response that covers all expected formats
LLMResponseType = dict[str, Any] | Mapping[str, Any] | object


def extract_content_from_response(response: LLMResponseType) -> str:
	"""
	Extract content from a LLM response.

	Args:
	    response: LLM response object or dictionary

	Returns:
	    Extracted content as string

	"""
	return _extract_content(response)


def generate_text(
	prompt: str,
	model: str | None = DEFAULT_MODEL,
	api_key: str | None = None,
	api_base: str | None = None,
	**kwargs: str | float | bool | None,
) -> str:
	"""
	Generate text using an LLM with minimal configuration.

	Args:
	    prompt: The prompt to send to the LLM
	    model: The model to use
	    api_key: The API key (if None, tries to find in environment)
	    api_base: Optional API base URL
	    **kwargs: Additional parameters to pass to the LLM API

	Returns:
	    The generated text

	Raises:
	    RuntimeError: If the LLM call fails

	"""
	try:
		# Create client and generate text directly
		client = create_client(model=model, api_key=api_key, api_base=api_base)
		return client.generate_text(prompt=prompt, **kwargs)  # type: ignore[arg-type]

	except LLMError as e:
		logger.exception("LLM error")
		msg = f"Failed to generate text with LLM: {e}"
		raise RuntimeError(msg) from e


def create_client(
	repo_path: Path | None = None,
	model: str | None = None,
	api_key: str | None = None,
	api_base: str | None = None,
) -> LLMClient:
	"""
	Create an LLMClient with the specified configuration.

	Args:
	    repo_path: Repository path for configuration loading
	    model: Model identifier to use
	    api_key: API key to use
	    api_base: API base URL to use

	Returns:
	    Configured LLMClient instance

	Raises:
	    RuntimeError: If client creation fails

	"""
	try:
		# Get configuration
		config = get_llm_config(
			overrides={
				"model": model,
				"api_key": api_key,
				"api_base": api_base,
			}
		)

		# Create client
		return LLMClient(config=config, repo_path=repo_path)

	except Exception as e:
		logger.exception("Error creating LLM client")
		msg = f"Failed to create LLM client: {e}"
		raise RuntimeError(msg) from e
