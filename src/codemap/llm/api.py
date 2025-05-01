"""API interaction for LLM services."""

from __future__ import annotations

import logging
from typing import Any

from .config import DEFAULT_LLM_REQUEST_PARAMS
from .errors import LLMError

logger = logging.getLogger(__name__)

# Define a type alias for the response types
ResponseType = dict[str, Any] | Any


def call_llm_api(
	prompt: str,
	model: str,
	api_key: str,
	api_base: str | None = None,
	json_schema: dict | None = None,
	**kwargs: dict[str, str | int | float | bool | None],
) -> str:
	"""
	Call an LLM API using litellm.

	Args:
	    prompt: The prompt to send to the LLM
	    model: The model identifier (including provider prefix)
	    api_key: The API key to use
	    api_base: Optional custom API base URL
	    json_schema: Optional JSON schema for response validation
	    **kwargs: Additional parameters to pass to the LLM API

	Returns:
	    The generated text response

	Raises:
	    LLMError: If the API call fails

	"""
	try:
		import litellm
	except ImportError:
		msg = "LiteLLM library not installed. Install it with 'pip install litellm'."
		logger.exception(msg)
		raise LLMError(msg) from None

	# Set up request parameters
	request_params = {
		"model": model,
		"messages": [{"role": "user", "content": prompt}],
		"api_key": api_key,
		**DEFAULT_LLM_REQUEST_PARAMS,
		**kwargs,
	}

	# Add API base if provided
	if api_base:
		request_params["api_base"] = api_base

	# Add JSON response format if schema provided
	if json_schema:
		request_params["response_format"] = {"type": "json_object", "schema": json_schema}
		# Enable schema validation
		litellm.enable_json_schema_validation = True

	def _raise_extraction_error() -> None:
		"""Raise an error for failed content extraction."""
		msg = "Failed to extract content from LLM response"
		raise LLMError(msg)

	try:
		logger.debug("Calling LiteLLM with model: %s", model)
		response = litellm.completion(**request_params)

		# Extract content from the response
		content = extract_content_from_response(response)

		if not content:
			logger.error("Could not extract content from LLM response")
			_raise_extraction_error()

		return content

	except Exception as e:
		logger.exception("LLM API call failed")
		msg = f"LLM API call failed: {e}"
		raise LLMError(msg) from e


def extract_content_from_response(response: ResponseType) -> str:
	"""
	Extract content from a LiteLLM response.

	Args:
	    response: LiteLLM response object or dictionary

	Returns:
	    Extracted content as string

	Raises:
	    AttributeError: If content cannot be extracted

	"""
	content = ""

	# Try different response formats
	if response:
		# First try the standard OpenAI-like structure
		if hasattr(response, "choices") and isinstance(getattr(response, "choices", []), list):
			choices = getattr(response, "choices", [])
			if choices:
				first_choice = choices[0]
				if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
					content = getattr(first_choice.message, "content", "")

		# Then try as dictionary if the above failed
		if not content and isinstance(response, dict):
			choices = response.get("choices", [])
			if choices and isinstance(choices, list) and choices:
				first_choice = choices[0]
				if isinstance(first_choice, dict):
					message = first_choice.get("message", {})
					if isinstance(message, dict):
						content = message.get("content", "")

		# Try as direct string for simple APIs
		if not content and hasattr(response, "text"):
			content = getattr(response, "text", "")

	return content or ""
